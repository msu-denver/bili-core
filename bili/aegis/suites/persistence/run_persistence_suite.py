"""Standalone persistence attack test suite runner for AETHER.

Runs all 15 persistence payloads against the 5 content-moderation MAS
configs using checkpoint-phase injection (``InjectionPhase.CHECKPOINT``),
applies all three detection tiers, and writes:

1. One result JSON per test case to ``results/{mas_id}/{payload_id}_checkpoint_injection.json``
2. A results matrix CSV to ``results/persistence_results_matrix.csv``

Persistence injection writes adversarial content directly to the checkpointer's
``put()`` API rather than modifying an agent's ``objective`` or interrupting
execution mid-run.  The poisoned message survives session teardown and appears
as legitimate prior-session context when the MAS resumes under the same
``thread_id`` in a new session.  This is the only AETHER attack type that
operates across session boundaries.

Skip behaviour
--------------
Configs are **skipped with a warning** when:

- ``checkpoint_enabled=False`` — no checkpointer, nothing to poison.
- The resolved checkpointer is a ``MemorySaver`` instance — in-process only;
  does not survive session teardown and would produce misleading cross-session
  results.

To run the suite with real persistence, configure a postgres or mongo
checkpointer in the target MAS YAML (``checkpoint_config: {type: postgres}``
or ``{type: mongo}``).  See ``verify_persistence.py`` for a self-contained
cross-session verification script.

Detection tiers
---------------
- Tier 1 (structural):  ``execution.success`` — did the injection + reload cycle
  complete without error?
- Tier 2 (heuristic):   ``influenced_agents`` / ``resistant_agents`` from
  ``PropagationTracker`` — which agents echoed persisted content or produced
  persistence-specific compliance phrases (e.g. "from our previous session")?
- Tier 3 (semantic):    ``SemanticEvaluator`` — LLM-based scoring of genuine
  persistence acceptance.  Skipped in stub mode; columns are empty strings in
  the CSV.

Usage
-----
Stub mode (no LLM calls — structural verification only):

    python bili/aegis/suites/persistence/run_persistence_suite.py --stub

Real LLM mode (requires API credentials and a persistent checkpointer):

    python bili/aegis/suites/persistence/run_persistence_suite.py

Filter to specific payloads:

    python bili/aegis/suites/persistence/run_persistence_suite.py --stub \\
        --payloads pe_session_001 pe_privilege_001

Results are written to:
    bili/aegis/suites/persistence/results/{mas_id}/{payload_id}_checkpoint_injection.json
    bili/aegis/suites/persistence/results/persistence_results_matrix.csv
"""

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — importable regardless of cwd
# Bootstrap: _find_repo_root is inlined to set up sys.path before any
# bili.* import.  The shared version in bili.aegis.suites._helpers is used
# for all subsequent calls once the package is importable.
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this file until a .git directory is found."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (.git directory not found)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bili.aegis.attacks.injector import (  # noqa: E402  pylint: disable=wrong-import-position
    AttackInjector,
)
from bili.aegis.attacks.models import (  # noqa: E402  pylint: disable=wrong-import-position
    InjectionPhase,
)
from bili.aegis.security.detector import (  # noqa: E402  pylint: disable=wrong-import-position
    SecurityEventDetector,
)
from bili.aegis.security.logger import (  # noqa: E402  pylint: disable=wrong-import-position
    SecurityEventLogger,
)
from bili.aegis.suites._helpers import CONFIG_PATHS
from bili.aegis.suites._helpers import (  # noqa: E402  pylint: disable=wrong-import-position
    config_fingerprint as _config_fingerprint_helper,
)
from bili.aegis.suites._helpers import latest_run_dir, next_run_dir
from bili.aegis.suites.persistence.payloads.persistence_payloads import (  # noqa: E402  pylint: disable=wrong-import-position
    PERSISTENCE_PAYLOADS,
)
from bili.aether.config.loader import (  # noqa: E402  pylint: disable=wrong-import-position
    load_mas_from_yaml,
)

LOGGER = logging.getLogger(__name__)

_PERSISTENCE_DIR = Path(__file__).parent
_RESULTS_DIR = _PERSISTENCE_DIR / "results"

_PHASE = InjectionPhase.CHECKPOINT.value

_CSV_COLUMNS: list[str] = [
    "payload_id",
    "injection_type",
    "severity",
    "stub_mode",
    "mas_id",
    "phase",
    "tier1_pass",
    "tier2_influenced",
    "tier2_resistant",
    "tier3_score",
    "tier3_confidence",
    "tier3_reasoning",
    "attack_suite",
    "skipped",
    "skip_reason",
]


# ---------------------------------------------------------------------------
# Checkpointer capability check
# ---------------------------------------------------------------------------


def _checkpointer_is_persistent(config: Any) -> tuple[bool, str]:
    """Return (is_persistent, reason_if_not).

    Checks whether the config's resolved checkpointer is a non-MemorySaver
    instance.  Uses ``isinstance`` on the runtime object rather than config
    string inspection — the factory may silently fall back to MemorySaver even
    when postgres/mongo is configured.

    Returns:
        ``(True, "")`` if a persistent checkpointer is available.
        ``(False, reason)`` if the config should be skipped.
    """
    if not config.checkpoint_enabled:
        return False, "checkpoint_enabled=False"

    try:
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
            MemorySaver,
        )
    except ImportError:
        return False, "langgraph not available"

    checkpoint_type = (config.checkpoint_config or {}).get("type", "memory")
    if checkpoint_type in ("memory", "auto", None):
        return False, (
            f"checkpoint_config.type={checkpoint_type!r} resolves to MemorySaver "
            "(in-process only; does not survive session teardown)"
        )

    # Try to create the checkpointer and verify it isn't MemorySaver at runtime.
    try:
        # pylint: disable=import-outside-toplevel
        from bili.aether.integration.checkpointer_factory import (
            create_checkpointer_from_config,
        )

        # pylint: enable=import-outside-toplevel
        checkpointer = create_checkpointer_from_config(
            config.checkpoint_config, user_id=None
        )
        if isinstance(checkpointer, MemorySaver):
            return False, (
                "configured backend resolved to MemorySaver at runtime "
                "(factory fallback) — not suitable for cross-session persistence"
            )
        return True, ""
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"checkpointer factory unavailable or failed: {exc}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_result(
    result_dict: dict, results_dir: Path, run_dir: Path | None = None
) -> Path:
    """Write one result JSON to run_dir/{payload_id}_{phase}.json.

    Falls back to ``results_dir/{mas_id}/`` when *run_dir* is ``None``.
    """
    out_dir = run_dir if run_dir is not None else (results_dir / result_dict["mas_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result_dict['payload_id']}_{result_dict['injection_phase']}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    return out_path


def _write_csv(rows: list[dict], results_dir: Path, csv_filename: str) -> Path:
    """Write the results matrix CSV to *results_dir*/*csv_filename*."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / csv_filename
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _print_summary(matrix_rows: list[dict]) -> None:
    """Print a compact summary table to stdout."""
    total = len(matrix_rows)
    skipped = sum(1 for r in matrix_rows if r["skipped"] == "true")
    ran = total - skipped
    passed = sum(1 for r in matrix_rows if r["tier1_pass"] == "true")
    influenced = sum(1 for r in matrix_rows if r["tier2_influenced"] != "[]")

    print("\n" + "=" * 60)
    print("Persistence Suite Summary")
    print("=" * 60)
    print(f"  Total configs tested : {total}")
    print(f"  Skipped (no backend) : {skipped}")
    print(f"  Ran                  : {ran}")
    print(f"  Tier 1 pass          : {passed}/{ran}")
    print(f"  Tier 2 influenced    : {influenced} cases had ≥1 influenced agent")
    if skipped == total:
        print(
            "\n  NOTE: All configs skipped — no persistent checkpointer configured.\n"
            "  Configure postgres or mongo in a MAS YAML to run real persistence\n"
            "  attacks.  See verify_persistence.py for a self-contained demo."
        )
    print("=" * 60 + "\n")


def _run_persistence_config(
    yaml_path: str,
    payloads: list,
    stub_mode: bool,
    semantic_evaluator: Any,
    baseline_results_dir: Path | None,
    results_dir: Path,
    repo_root: Path,
) -> tuple[
    list[dict], Path | None
]:  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    """Run all payloads for one MAS config, skipping if no persistent backend.

    Returns ``(matrix_rows, run_dir)``.  *run_dir* is ``None`` when the config
    was skipped (no persistent backend available).
    """
    full_path = repo_root / yaml_path
    if not full_path.exists():
        print(f"  Config not found, skipping: {yaml_path}", file=sys.stderr)
        return [], None

    config = load_mas_from_yaml(str(full_path))
    if stub_mode:
        for agent in config.agents:
            agent.model_name = None

    mas_id = config.mas_id
    print(f"\n[{mas_id}] Checking checkpointer capability...")

    is_persistent, skip_reason = _checkpointer_is_persistent(config)
    if not is_persistent:
        print(
            f"  [SKIP] {mas_id}: {skip_reason}",
            file=sys.stderr,
        )
        # Produce skip rows so the CSV documents why this config was not tested.
        skip_rows = []
        for ip in payloads:
            skip_rows.append(
                {
                    "payload_id": ip.payload_id,
                    "injection_type": ip.injection_type,
                    "severity": ip.severity,
                    "stub_mode": stub_mode,
                    "mas_id": mas_id,
                    "phase": _PHASE,
                    "tier1_pass": "",
                    "tier2_influenced": "",
                    "tier2_resistant": "",
                    "tier3_score": "",
                    "tier3_confidence": "",
                    "tier3_reasoning": "",
                    "attack_suite": "persistence",
                    "skipped": "true",
                    "skip_reason": skip_reason,
                }
            )
        return skip_rows, None

    config_results_dir = results_dir / mas_id
    run_dir = next_run_dir(config_results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    attack_log_path = run_dir / "attack_log.ndjson"
    sec_log_path = run_dir / "security_events.ndjson"

    sec_logger = SecurityEventLogger(log_path=sec_log_path)
    detector = SecurityEventDetector(
        logger=sec_logger,
        attack_log_path=attack_log_path,
    )

    entry_agent = config.entry_point or config.agents[0].agent_id
    print(
        f"[{mas_id}] target_agent={entry_agent!r}  "
        f"({len(payloads)} payloads × checkpoint_injection)"
    )

    matrix_rows: list[dict] = []

    with AttackInjector(
        config,
        executor=None,
        log_path=attack_log_path,
        security_detector=detector,
    ) as injector:
        for ip in payloads:
            label = f"  {ip.payload_id} / {_PHASE}"
            print(f"{label} ... ", end="", flush=True)

            try:
                attack_result = injector.inject_attack(
                    agent_id=entry_agent,
                    attack_type="persistence",
                    payload=ip.payload,
                    injection_phase=_PHASE,
                    blocking=True,
                    track_propagation=True,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.error(
                    "inject_attack failed for %s / %s: %s",
                    mas_id,
                    ip.payload_id,
                    exc,
                )
                print(f"ERROR: {exc}")
                matrix_rows.append(
                    {
                        "payload_id": ip.payload_id,
                        "injection_type": ip.injection_type,
                        "severity": ip.severity,
                        "stub_mode": stub_mode,
                        "mas_id": mas_id,
                        "phase": _PHASE,
                        "tier1_pass": "false",
                        "tier2_influenced": "[]",
                        "tier2_resistant": "[]",
                        "tier3_score": "",
                        "tier3_confidence": "",
                        "tier3_reasoning": str(exc),
                        "attack_suite": "persistence",
                        "skipped": "false",
                        "skip_reason": "",
                    }
                )
                continue

            tier3_score: str = ""
            tier3_confidence: str = ""
            tier3_reasoning: str = ""
            if not stub_mode and semantic_evaluator is not None:
                baseline_dict = _load_baseline(baseline_results_dir, mas_id)
                if baseline_dict is not None:
                    try:
                        tier3_rows = semantic_evaluator.evaluate(
                            baseline_result=baseline_dict,
                            attack_result=attack_result,
                        )
                        if tier3_rows:
                            best = max(tier3_rows, key=lambda v: v.score)
                            tier3_score = str(best.score)
                            tier3_confidence = best.confidence
                            tier3_reasoning = best.reasoning
                    except (
                        Exception
                    ) as t3_exc:  # pylint: disable=broad-exception-caught
                        LOGGER.warning(
                            "SemanticEvaluator failed for %s/%s: %s",
                            ip.payload_id,
                            _PHASE,
                            t3_exc,
                        )

            resistant_list = sorted(attack_result.resistant_agents)
            result_dict = {
                "payload_id": ip.payload_id,
                "injection_type": ip.injection_type,
                "severity": ip.severity,
                "mas_id": mas_id,
                "injection_phase": _PHASE,
                "attack_suite": "persistence",
                "config_fingerprint": _config_fingerprint_helper(
                    config, yaml_path, repo_root
                ),
                "execution": {
                    "success": attack_result.success,
                    "duration_ms": (
                        (
                            attack_result.completed_at - attack_result.injected_at
                        ).total_seconds()
                        * 1000
                        if attack_result.completed_at
                        else 0.0
                    ),
                    "agent_count": len(attack_result.propagation_path),
                    "message_count": 0,
                },
                "propagation_path": attack_result.propagation_path,
                "influenced_agents": attack_result.influenced_agents,
                "resistant_agents": resistant_list,
                "run_metadata": {
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "stub_mode": stub_mode,
                    "semantic_tier": "skipped" if stub_mode else "evaluated",
                    "tier3_score": tier3_score,
                    "tier3_confidence": tier3_confidence,
                    "tier3_reasoning": tier3_reasoning,
                },
            }
            out_path = _write_result(result_dict, results_dir, run_dir=run_dir)
            status = "ok" if attack_result.success else "FAIL"
            influenced_count = len(attack_result.influenced_agents)
            print(f"{status}  (influenced={influenced_count}) → {out_path.name}")

            matrix_rows.append(
                {
                    "payload_id": ip.payload_id,
                    "injection_type": ip.injection_type,
                    "severity": ip.severity,
                    "stub_mode": stub_mode,
                    "mas_id": mas_id,
                    "phase": _PHASE,
                    "tier1_pass": str(attack_result.success).lower(),
                    "tier2_influenced": json.dumps(
                        sorted(attack_result.influenced_agents)
                    ),
                    "tier2_resistant": json.dumps(resistant_list),
                    "tier3_score": tier3_score,
                    "tier3_confidence": tier3_confidence,
                    "tier3_reasoning": tier3_reasoning,
                    "attack_suite": "persistence",
                    "skipped": "false",
                    "skip_reason": "",
                }
            )

    return matrix_rows, run_dir


def _load_baseline(baseline_results_dir: Path | None, mas_id: str) -> dict | None:
    """Try to load a baseline result for Tier 3 comparison.

    Prefers the most recent ``run_NNN`` subdirectory; falls back to the flat
    legacy layout when no versioned run directories exist.
    """
    if baseline_results_dir is None:
        return None
    config_baseline_dir = baseline_results_dir / mas_id
    run_dir = latest_run_dir(config_baseline_dir)
    search_dir = run_dir if run_dir is not None else config_baseline_dir
    if not search_dir.exists():
        return None
    result_files = sorted(search_dir.glob("*.json"))
    if not result_files:
        return None
    try:
        return json.loads(result_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load baseline for %s: %s", mas_id, exc)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full persistence attack test suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AETHER persistence attack suite against content-moderation "
            "MAS configs and produce a results matrix CSV. "
            "Configs without a persistent checkpointer are skipped with a warning."
        )
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use stub agents (no LLM calls). Tier 3 semantic evaluation is "
            "skipped and CSV tier3 columns are empty strings. "
            "Note: persistence attacks still require a non-MemorySaver checkpointer "
            "even in stub mode."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIG_PATHS,
        metavar="YAML_PATH",
        help="Override the list of YAML config paths to run.",
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        default=None,
        metavar="PAYLOAD_ID",
        help=(
            "Restrict run to specific payload IDs "
            "(e.g. pe_session_001 pe_privilege_001)."
        ),
    )
    parser.add_argument(
        "--baseline-results",
        default=None,
        metavar="DIR",
        help=(
            "Path to baseline results directory for Tier 3 comparison "
            "(e.g. bili/aegis/suites/baseline/results). "
            "Required for real-mode Tier 3 evaluation."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    payloads = PERSISTENCE_PAYLOADS
    if args.payloads:
        ids = set(args.payloads)
        payloads = [p for p in PERSISTENCE_PAYLOADS if p.payload_id in ids]
        if not payloads:
            print(f"No payloads matched: {args.payloads}", file=sys.stderr)
            sys.exit(1)

    baseline_results_dir: Path | None = None
    if args.baseline_results:
        baseline_results_dir = _REPO_ROOT / args.baseline_results
        if not baseline_results_dir.exists():
            print(
                f"Warning: baseline results dir not found: {baseline_results_dir}",
                file=sys.stderr,
            )
            baseline_results_dir = None

    # SemanticEvaluator — persistence-specific judge prompt and rubric.
    semantic_evaluator = None
    if not args.stub:
        try:
            # pylint: disable=import-outside-toplevel
            from bili.aegis.evaluator import SemanticEvaluator
            from bili.aegis.evaluator.evaluator_config import (
                PERSISTENCE_JUDGE_PROMPT,
                PERSISTENCE_SCORE_DESCRIPTIONS,
            )

            # pylint: enable=import-outside-toplevel

            semantic_evaluator = SemanticEvaluator(
                score_descriptions=PERSISTENCE_SCORE_DESCRIPTIONS,
                judge_prompt_template=PERSISTENCE_JUDGE_PROMPT,
            )
        except (ImportError, RuntimeError) as exc:
            LOGGER.warning("Could not initialise SemanticEvaluator: %s", exc)

    all_matrix_rows: list[dict] = []
    first_run_dir_name: str | None = None
    for yaml_path in args.configs:
        rows, run_dir = _run_persistence_config(
            yaml_path=yaml_path,
            payloads=payloads,
            stub_mode=args.stub,
            semantic_evaluator=semantic_evaluator,
            baseline_results_dir=baseline_results_dir,
            results_dir=_RESULTS_DIR,
            repo_root=_REPO_ROOT,
        )
        all_matrix_rows.extend(rows)
        if first_run_dir_name is None and run_dir is not None:
            first_run_dir_name = run_dir.name

    if all_matrix_rows:
        csv_filename = (
            f"persistence_results_matrix_{first_run_dir_name}.csv"
            if first_run_dir_name
            else "persistence_results_matrix.csv"
        )
        csv_path = _write_csv(all_matrix_rows, _RESULTS_DIR, csv_filename)
        _print_summary(all_matrix_rows)
        print(f"Results matrix written to: {csv_path}")

    # Exit 0 if all ran tests passed OR all were skipped (skip is not a failure).
    ran_rows = [r for r in all_matrix_rows if r["skipped"] != "true"]
    passed = sum(1 for r in ran_rows if r["tier1_pass"] == "true")
    sys.exit(0 if passed == len(ran_rows) else 1)


if __name__ == "__main__":
    main()
