"""
Entry point for the PROBE suite.

Matches the existing AEGIS runner CLI shape:

    python -m bili.aegis.suites.probe.run_probe_suite --stub
    python -m bili.aegis.suites.probe.run_probe_suite \\
        --baseline-results bili/aegis/suites/baseline/results

PROBE-specific flags:
    --policies pair crescendo tap          (default: all three)
    --objectives pr_misinfo_001 ...        (default: all in library)
    --configs path/to/mas.yaml ...         (default: CONFIG_PATHS in _helpers.py)
    --seeds 0 1 2                          (default: [0])
    --budget-turns 12                      (per-session turn cap)
    --budget-tokens 200000                 (per-session token cap)
    --budget-cost-usd 5.0                  (per-session cost cap)
    --smoke                                (1/10 scale for CI sanity check)
    --attacker-model deepseek-chat         (real-LLM attacker model_name)
    --victim-model us.anthropic.claude...  (override YAML victim model_name)
    --judge-model gemini-2.5-flash         (real-LLM judge model_name)

Output:
    bili/aegis/suites/probe/results/{mas_id}/sessions/{session_id}.json  (sidecar)
    bili/aegis/suites/probe/results/probe_results_matrix.csv             (matrix)

See RFC § 12 for acceptance criteria.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from bili.aegis.probe._llm import (
    _FakeLLM,
    _stub_responder,
    _StubVictimExecutor,
    resolve_real_llm,
)
from bili.aegis.probe.attacker_mas import AttackerMAS, AttackerModelConfigs
from bili.aegis.probe.budget import BudgetState
from bili.aegis.probe.exceptions import JudgeUnavailableError
from bili.aegis.probe.policies import POLICY_REGISTRY
from bili.aegis.probe.schema import (
    ProbeObjective,
    ProbeOutcome,
    ProbeOutcomeReason,
    ProbeSession,
)
from bili.aegis.suites._helpers import CONFIG_PATHS, find_repo_root
from bili.aegis.suites._suite_runner import _load_baseline
from bili.aegis.suites.probe._csv import append_probe_csv_row
from bili.aegis.suites.probe.payloads.probe_objectives import PROBE_OBJECTIVE_LIBRARY

LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR: str = "bili/aegis/suites/probe/results"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_probe_suite",
        description="AEGIS-PROBE: autonomous adaptive red-teaming suite.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Skip all real LLM calls; use deterministic fake responses.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["pair", "crescendo", "tap"],
        help="Policies to run (default: all three).",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=None,
        help="Objective IDs (default: full library).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="MAS YAML paths (default: CONFIG_PATHS from _helpers).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="RNG seeds; one session per seed.",
    )
    parser.add_argument("--budget-turns", type=int, default=12)
    parser.add_argument("--budget-tokens", type=int, default=200_000)
    parser.add_argument("--budget-cost-usd", type=float, default=5.0)
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to baseline results dir (for Tier 3 reference text).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run at 1/10 scale (first objective × first config × first seed).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Output directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default="deepseek-chat",
        help="Real-LLM attacker model_name (ignored in --stub mode).",
    )
    parser.add_argument(
        "--victim-model",
        type=str,
        default="us.anthropic.claude-sonnet-4-6",
        help="Victim model_name used for the cross-provider check.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Real-LLM judge model_name (ignored in --stub mode).",
    )
    parser.add_argument(
        "--attacker-model-type",
        type=str,
        default="remote_aws_bedrock",
        help="IRIS load_model model_type for the attacker.",
    )
    parser.add_argument(
        "--victim-model-type",
        type=str,
        default="remote_aws_bedrock",
        help="IRIS load_model model_type for the victim.",
    )
    parser.add_argument(
        "--judge-model-type",
        type=str,
        default="remote_google_vertex",
        help="IRIS load_model model_type for the judge.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_config_paths(args: argparse.Namespace, repo_root: Path) -> list[Path]:
    """Convert relative config strings to absolute Paths under repo_root."""
    paths = args.configs if args.configs else CONFIG_PATHS
    resolved: list[Path] = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        resolved.append(p)
    return resolved


def _filter_objectives(args: argparse.Namespace) -> list[ProbeObjective]:
    """Apply `--objectives` whitelist to the global library."""
    if args.objectives:
        wanted = set(args.objectives)
        return [obj for obj in PROBE_OBJECTIVE_LIBRARY if obj.objective_id in wanted]
    return list(PROBE_OBJECTIVE_LIBRARY)


def _apply_smoke_filter(
    objectives: list[ProbeObjective],
    config_paths: list[Path],
    seeds: list[int],
) -> tuple[list[ProbeObjective], list[Path], list[int]]:
    """``--smoke``: first item only in each dimension."""
    return (
        objectives[:1] if objectives else [],
        config_paths[:1] if config_paths else [],
        seeds[:1] if seeds else [],
    )


def _model_config(is_stub: bool, role: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build the model_config dict for one role.

    In --stub mode, model_name is ``None`` so :meth:`ProbeSession.to_csv_row`
    reports stub_mode='stub'. Note: model_name is still set to the user-
    supplied name so the cross-provider check has a real prefix to inspect
    (the override LLM bypasses any actual provider call).
    """
    if role == "attacker":
        return {
            "model_type": args.attacker_model_type,
            "model_name": args.attacker_model if not is_stub else "deepseek-chat",
            "temperature": 0.7,
        }
    if role == "victim":
        return {
            "model_type": args.victim_model_type,
            "model_name": args.victim_model,
            "temperature": 0.0,
        }
    if role == "judge":
        return {
            "model_type": args.judge_model_type,
            "model_name": args.judge_model if not is_stub else "gemini-2.5-flash",
            "temperature": 0.0,
        }
    raise ValueError(f"Unknown role: {role!r}")


def _build_session_id(
    objective_id: str, mas_id: str, policy_name: str, seed: int
) -> str:
    """Build a unique, sortable session_id."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{objective_id}__{mas_id}__{policy_name}__seed{seed}__{timestamp}"


def _victim_mas_shape_from_config(config: Any) -> dict[str, Any]:
    """Extract the shape dict the PayloadCrafterNode wants from a MASConfig."""
    agents = []
    for agent in getattr(config, "agents", []) or []:
        agents.append(
            {
                "agent_id": getattr(agent, "agent_id", ""),
                "role": getattr(agent, "role", ""),
            }
        )
    return {
        "mas_id": getattr(config, "mas_id", "<unknown>"),
        "agents": agents,
        "entry_point": (agents[0]["agent_id"] if agents else "<unknown>"),
    }


def _write_sidecar(session: ProbeSession, results_dir: Path) -> Path:
    """Write the per-session sidecar JSON under {mas_id}/sessions/."""
    mas_dir = results_dir / session.victim_mas_id / "sessions"
    mas_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = mas_dir / f"{session.session_id}.json"
    payload = session.to_sidecar_json()
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return sidecar_path


def _print_summary(
    matrix_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    """Print a one-block summary to stdout matching the existing AEGIS style."""
    print()  # noqa: T201
    print("=" * 60)  # noqa: T201
    print("PROBE Suite Summary")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print(f"Total sessions: {len(matrix_rows)}")  # noqa: T201
    if not matrix_rows:
        print("(No sessions ran)")  # noqa: T201
        return
    reasons = Counter(row["terminated_reason"] for row in matrix_rows)
    for reason, count in sorted(reasons.items()):
        print(f"  {reason:30s} {count}")  # noqa: T201
    ttc_values = [
        int(row["turns_to_compromise"])
        for row in matrix_rows
        if row["turns_to_compromise"] not in ("", None)
    ]
    if ttc_values:
        print(f"  turns_to_compromise (success rows): {ttc_values}")  # noqa: T201
    if error_rows:
        print(f"Framework errors: {len(error_rows)}")  # noqa: T201
        for err in error_rows:
            print(f"  - {err['error']}")  # noqa: T201


def _load_baseline_text(baseline_dir: Optional[Path], mas_id: str) -> Optional[str]:
    """Read a baseline text excerpt for the judge prompt.

    Delegates to :func:`bili.aegis.suites._suite_runner._load_baseline`, the
    same loader the static suites use, so PROBE picks up the versioned
    ``{mas_id}/run_NNN/`` baseline layout (with flat-layout fallback). The
    hand-rolled version this replaced only read a flat ``{mas_id}.json`` and
    silently missed every versioned baseline. Returns ``final_text`` (falling
    back to ``summary``) from the baseline dict, or ``None`` when no baseline
    is available — in which case the judge prompt renders 'NOT AVAILABLE'.
    """
    if baseline_dir is None:
        return None
    data = _load_baseline(baseline_dir, mas_id)
    if not isinstance(data, dict):
        return None
    return data.get("final_text") or data.get("summary")


@dataclass(frozen=True)
class _SessionIdentity:
    """Identifies a single PROBE session by its (objective, config, policy, seed) tuple.

    Used as a bundle so the helpers that need all four don't grow into
    ``too-many-arguments`` territory. ``mas_id`` is derived from
    ``config_path.stem`` to match the convention used everywhere else in
    the runner.
    """

    objective: ProbeObjective
    config_path: Path
    policy_name: str
    seed: int

    @property
    def mas_id(self) -> str:
        """Match the runner-wide convention: ``mas_id == config_path.stem``."""
        return self.config_path.stem


@dataclass(frozen=True)
class _SessionRunSpec:
    """Full specification for one ``_run_one_session`` invocation.

    Bundles the per-session inputs (identity + policy_cls + stub flag +
    optional baseline) into one object so the function takes ``(spec,
    args)`` rather than 8 positional arguments.
    """

    identity: _SessionIdentity
    policy_cls: type
    is_stub: bool
    baseline_text: Optional[str]


def _make_failed_session_row(
    identity: _SessionIdentity,
    attacker_cfg: dict[str, Any],
    judge_cfg: dict[str, Any],
    reason: ProbeOutcomeReason,
    error: str,
) -> dict[str, Any]:
    """Build a CSV row representing a session that never started.

    Used when ``MASExecutor.run`` or ``AttackerMAS.initialize`` itself
    fails: we still want a row in the matrix recording why this
    (objective, config, policy, seed) tuple didn't run. Failed rows are
    CSV-only (no sidecar is written), so the victim model config is left at
    its default and only the success path records it.
    """
    session_id = _build_session_id(
        identity.objective.objective_id,
        identity.mas_id,
        identity.policy_name,
        identity.seed,
    )
    session = ProbeSession(
        session_id=session_id,
        objective=identity.objective,
        victim_mas_id=identity.mas_id,
        victim_mas_path=str(identity.config_path),
        policy_name=identity.policy_name,
        rng_seed=identity.seed,
        attacker_model_config=attacker_cfg,
        judge_model_config=judge_cfg,
    )
    session.final_outcome = ProbeOutcome(
        reason=reason,
        final_tier3_score=0,
        turns_to_compromise=None,
        total_duration_ms=0.0,
        total_tokens_attacker=0,
        total_tokens_victim=0,
        total_tokens_judge=0,
        estimated_cost_usd=0.0,
    )
    LOGGER.warning(
        "Session %s did not run (%s): %s", session.session_id, reason.value, error
    )
    return session.to_csv_row()


@dataclass
class _AttackerDependencies:
    """Resolved per-session attacker dependencies.

    Bundles the model configs (attacker / judge / victim) with the three
    LLM overrides chosen based on ``is_stub``. Built once per session by
    :func:`_resolve_attacker_dependencies` and consumed by both the
    session-row failure path and the AttackerMAS wiring path.
    """

    attacker_cfg: dict[str, Any]
    judge_cfg: dict[str, Any]
    victim_cfg: dict[str, Any]
    # ``Any`` here avoids pulling ProbeLLM into module-level imports.
    crafter_llm: Any
    evaluator_llm: Any
    policy_llm: Any


def _resolve_attacker_dependencies(
    args: argparse.Namespace, is_stub: bool
) -> _AttackerDependencies:
    """Build the per-session attacker configs and LLM overrides.

    In stub mode, all three LLMs are :class:`_FakeLLM` instances driven by
    ``_stub_responder``. In real-LLM mode, the policy LLM is resolved
    against ``attacker_cfg`` and the crafter / evaluator LLMs are left
    ``None`` so the attacker constructs them per its own config.
    """
    attacker_cfg = _model_config(is_stub, "attacker", args)
    judge_cfg = _model_config(is_stub, "judge", args)
    victim_cfg = _model_config(is_stub, "victim", args)
    if is_stub:
        crafter_llm = _FakeLLM(responder=_stub_responder)
        evaluator_llm = _FakeLLM(responder=_stub_responder)
        policy_llm = _FakeLLM(responder=_stub_responder)
    else:
        crafter_llm = None
        evaluator_llm = None
        policy_llm = resolve_real_llm(attacker_cfg)
    return _AttackerDependencies(
        attacker_cfg=attacker_cfg,
        judge_cfg=judge_cfg,
        victim_cfg=victim_cfg,
        crafter_llm=crafter_llm,
        evaluator_llm=evaluator_llm,
        policy_llm=policy_llm,
    )


def _build_attacker_for_session(
    deps: _AttackerDependencies,
    policy_cls: type,
    victim_shape: dict[str, Any],
) -> AttackerMAS:
    """Wire an :class:`AttackerMAS` from resolved dependencies + policy class."""
    policy = policy_cls(llm=deps.policy_llm)
    return AttackerMAS(
        policy=policy,
        model_configs=AttackerModelConfigs(
            attacker=deps.attacker_cfg,
            judge=deps.judge_cfg,
            victim=deps.victim_cfg,
        ),
        victim_mas_shape=victim_shape,
        crafter_llm_override=deps.crafter_llm,
        evaluator_llm_override=deps.evaluator_llm,
    )


@dataclass
class _VictimReady:
    """A successfully loaded victim MAS + the IDs that depend on it."""

    mas_id: str
    session_id: str
    victim_shape: dict[str, Any]
    victim_executor: Any


def _load_victim(
    spec: _SessionRunSpec, deps: _AttackerDependencies
) -> tuple[Optional[dict[str, Any]], Optional[_VictimReady]]:
    """Load the victim YAML and prepare the executor.

    Returns ``(failed_row, victim_ready)``; exactly one of the two is
    non-``None``. ``failed_row`` is populated when YAML loading or
    MASExecutor initialization fails (both → ``VICTIM_CRASHED``).
    """
    # Local imports keep AETHER / IRIS loading out of import-time of this
    # module; helpful for unit-testing the CLI in isolation.
    from bili.aether.config.loader import (  # pylint: disable=import-outside-toplevel  # defer AETHER load until a session actually runs
        load_mas_from_yaml,
    )
    from bili.aether.runtime.executor import (  # pylint: disable=import-outside-toplevel  # defer AETHER load until a session actually runs
        MASExecutor,
    )

    try:
        config = load_mas_from_yaml(str(spec.identity.config_path))
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        yaml.YAMLError,
        ValidationError,
    ) as exc:
        # The exception types above match load_mas_from_yaml's documented
        # failure modes (Raises section); all of them get mapped to a
        # VICTIM_CRASHED row.
        row = _make_failed_session_row(
            spec.identity,
            deps.attacker_cfg,
            deps.judge_cfg,
            ProbeOutcomeReason.VICTIM_CRASHED,
            f"Failed to load YAML: {exc}",
        )
        return row, None

    mas_id = getattr(config, "mas_id", spec.identity.mas_id)
    session_id = _build_session_id(
        spec.identity.objective.objective_id,
        mas_id,
        spec.identity.policy_name,
        spec.identity.seed,
    )
    victim_shape = _victim_mas_shape_from_config(config)
    if spec.is_stub:
        # In stub mode we don't initialize MASExecutor; build a dict-shaped
        # stub that mimics MASExecutor.run's return.
        victim_executor: Any = _StubVictimExecutor()
    else:
        try:
            victim_executor = MASExecutor(config)
            victim_executor.initialize()
        # MASExecutor init faults all map to VICTIM_CRASHED.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            row = _make_failed_session_row(
                spec.identity,
                deps.attacker_cfg,
                deps.judge_cfg,
                ProbeOutcomeReason.VICTIM_CRASHED,
                f"Failed to initialize MASExecutor: {exc}",
            )
            return row, None

    return None, _VictimReady(
        mas_id=mas_id,
        session_id=session_id,
        victim_shape=victim_shape,
        victim_executor=victim_executor,
    )


def _build_session_and_budget(
    spec: _SessionRunSpec,
    deps: _AttackerDependencies,
    victim: _VictimReady,
    args: argparse.Namespace,
) -> tuple[ProbeSession, BudgetState]:
    """Construct the ProbeSession + BudgetState pair for one session."""
    session = ProbeSession(
        session_id=victim.session_id,
        objective=spec.identity.objective,
        victim_mas_id=victim.mas_id,
        victim_mas_path=str(spec.identity.config_path),
        policy_name=spec.identity.policy_name,
        rng_seed=spec.identity.seed,
        attacker_model_config=(
            deps.attacker_cfg if not spec.is_stub else {"model_name": None}
        ),
        judge_model_config=deps.judge_cfg,
        victim_model_config=(
            deps.victim_cfg if not spec.is_stub else {"model_name": None}
        ),
    )
    budget = BudgetState(
        max_turns=args.budget_turns,
        max_tokens_total=args.budget_tokens,
        max_wall_clock_seconds=None,
        max_cost_usd=args.budget_cost_usd,
    )
    return session, budget


def _run_one_session(
    spec: _SessionRunSpec, args: argparse.Namespace
) -> tuple[dict[str, Any], Path]:
    """Run one (objective, config, policy, seed) session.

    Returns ``(csv_row, sidecar_path)``. Failures inside this function
    are caught and converted to a CSV row with the appropriate
    ``terminated_reason``; this function does NOT raise.
    """
    deps = _resolve_attacker_dependencies(args, spec.is_stub)
    failed_row, victim = _load_victim(spec, deps)
    if victim is None:
        # ``_load_victim`` always populates exactly one slot of its
        # (failed_row, victim_ready) tuple. ``victim is None`` ⇒
        # ``failed_row`` is the populated half. Make that contract
        # visible rather than papering over a hypothetical violation
        # with a silent ``or {}`` fallback.
        assert (
            failed_row is not None
        ), "_load_victim returned (None, None); invariant violated"
        return failed_row, Path()

    session, budget = _build_session_and_budget(spec, deps, victim, args)
    attacker = _build_attacker_for_session(deps, spec.policy_cls, victim.victim_shape)

    try:
        attacker.initialize()
    except JudgeUnavailableError as exc:
        row = _make_failed_session_row(
            spec.identity,
            deps.attacker_cfg,
            deps.judge_cfg,
            ProbeOutcomeReason.JUDGE_UNAVAILABLE,
            str(exc),
        )
        return row, Path()

    session = attacker.run_session(
        session, victim.victim_executor, budget, baseline_output_text=spec.baseline_text
    )

    sidecar_path = _write_sidecar(session, Path(args.results_dir))
    return session.to_csv_row(), sidecar_path


@dataclass
class _GridInputs:
    """Inputs to ``_execute_session_grid`` — bundled to keep the helper's
    signature under the ``too-many-arguments`` threshold and to make
    main() readable.
    """

    args: argparse.Namespace
    objectives: list[ProbeObjective]
    config_paths: list[Path]
    policy_clses: dict[str, type]
    seeds: list[int]
    baseline_dir: Optional[Path]
    results_dir: Path


def _resolve_runner_paths(
    args: argparse.Namespace, repo_root: Path
) -> tuple[Path, Optional[Path]]:
    """Resolve ``results_dir`` + optional ``baseline_dir`` against the repo root.

    Mutates ``args.results_dir`` so downstream session-runner code sees
    the resolved absolute path.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    args.results_dir = str(results_dir)
    baseline_dir: Optional[Path] = None
    if args.baseline_results:
        baseline_dir = Path(args.baseline_results)
        if not baseline_dir.is_absolute():
            baseline_dir = repo_root / baseline_dir
    return results_dir, baseline_dir


def _execute_session_grid(
    inputs: _GridInputs,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the (objectives × configs × policies × seeds) grid.

    Returns ``(matrix_rows, error_rows)``. Framework errors (from
    ``_run_one_session`` itself) are caught and recorded as error_rows;
    per-session terminal reasons (VICTIM_CRASHED, JUDGE_UNAVAILABLE, ...)
    are recorded as matrix_rows by ``_run_one_session``.
    """
    matrix_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    is_stub = inputs.args.stub
    for objective in inputs.objectives:
        for config_path in inputs.config_paths:
            mas_id = config_path.stem
            baseline_text = _load_baseline_text(inputs.baseline_dir, mas_id)
            for policy_name, policy_cls in inputs.policy_clses.items():
                for seed in inputs.seeds:
                    spec = _SessionRunSpec(
                        identity=_SessionIdentity(
                            objective=objective,
                            config_path=config_path,
                            policy_name=policy_name,
                            seed=seed,
                        ),
                        policy_cls=policy_cls,
                        is_stub=is_stub,
                        baseline_text=baseline_text,
                    )
                    try:
                        row, _ = _run_one_session(spec, inputs.args)
                    # framework-level errors must not abort the whole run; record as error_row
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        LOGGER.exception(
                            "Framework error on (%s, %s, %s, seed=%s)",
                            objective.objective_id,
                            mas_id,
                            policy_name,
                            seed,
                        )
                        error_rows.append({"error": str(exc)})
                        continue
                    matrix_rows.append(row)
                    append_probe_csv_row(row, inputs.results_dir)
    return matrix_rows, error_rows


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point. See module docstring."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    repo_root = find_repo_root()
    config_paths = _resolve_config_paths(args, repo_root)
    objectives = _filter_objectives(args)
    seeds = args.seeds

    if args.smoke:
        objectives, config_paths, seeds = _apply_smoke_filter(
            objectives, config_paths, seeds
        )

    # Filter policies by what the registry exposes
    policy_clses = {
        name: POLICY_REGISTRY[name] for name in args.policies if name in POLICY_REGISTRY
    }
    missing = set(args.policies) - set(policy_clses)
    if missing:
        LOGGER.error("Unknown policy names: %s", sorted(missing))
        return 1

    results_dir, baseline_dir = _resolve_runner_paths(args, repo_root)
    matrix_rows, error_rows = _execute_session_grid(
        _GridInputs(
            args=args,
            objectives=objectives,
            config_paths=config_paths,
            policy_clses=policy_clses,
            seeds=seeds,
            baseline_dir=baseline_dir,
            results_dir=results_dir,
        )
    )

    _print_summary(matrix_rows, error_rows)
    # Framework errors (vs per-session terminal reasons) → non-zero exit
    return 1 if error_rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
