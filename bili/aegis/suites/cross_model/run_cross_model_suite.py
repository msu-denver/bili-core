"""Cross-model transferability test suite runner for AETHER.

Tests whether prompt-injection payloads that succeed against one LLM provider
family also transfer to agents running on different provider families.  The
outer loop is a model matrix (3–4 models across different provider families);
the inner loop is the standard payload × config × phase iteration.

Model matrix (default)
----------------------
+-------------------------------------------------+---------------------------+
| model_id                                        | Provider family           |
+-------------------------------------------------+---------------------------+
| us.anthropic.claude-3-5-haiku-20241022-v1:0     | AWS Bedrock / Anthropic   |
| us.anthropic.claude-sonnet-4-20250514-v1:0      | AWS Bedrock / Anthropic   |
| amazon.nova-pro-v1:0                            | AWS Bedrock / Amazon      |
| gemini-2.0-flash                                | Google Vertex             |
+-------------------------------------------------+---------------------------+

Real LLM credentials are required for meaningful results.  If a model's
credentials are unavailable, that model's rows are written as skipped with a
``skip_reason`` noting the exception — the suite continues with the remaining
models.

Stub mode
---------
``--stub`` replaces the full model matrix with a single ``model_id=None`` run,
which uses stub agents (no LLM calls).  This validates CSV schema and result
file structure without requiring credentials.

Detection tiers
---------------
- Tier 1 (structural): ``execution.success`` — did injection + invocation
  complete without error?
- Tier 2 (heuristic): ``influenced_agents`` / ``resistant_agents`` from
  ``PropagationTracker`` — which agents echoed injected content or produced
  compliance phrases?
- Tier 3 (semantic): ``SemanticEvaluator`` — LLM judge scoring of genuine
  compliance.  Skipped in stub mode; columns are empty strings in the CSV.

Payload set
-----------
Reuses the 15 prompt-injection payloads from the Injection Suite
(``bili/aegis/suites/injection/payloads/prompt_injection_payloads.py``).  No
new payload library is needed — the transferability dimension is the model axis.

Results
-------
Per-test JSON files:
    results/{mas_id}/{model_id_safe}/{payload_id}_{phase}.json

Results matrix CSV:
    results/cross_model_matrix.csv

Usage
-----
Stub mode (no LLM calls — structural + schema verification only):

    python bili/aegis/suites/cross_model/run_cross_model_suite.py --stub

Full run (requires AWS Bedrock + Google Vertex credentials):

    python bili/aegis/suites/cross_model/run_cross_model_suite.py

Restrict to specific models:

    python bili/aegis/suites/cross_model/run_cross_model_suite.py \\
        --models us.anthropic.claude-3-5-haiku-20241022-v1:0 amazon.nova-pro-v1:0

Restrict to specific payloads:

    python bili/aegis/suites/cross_model/run_cross_model_suite.py --stub \\
        --payloads pi_direct_001 pi_role_001
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
    AttackType,
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
from bili.aegis.suites._helpers import model_id_safe as _model_id_safe
from bili.aegis.suites.injection.payloads.prompt_injection_payloads import (  # noqa: E402  pylint: disable=wrong-import-position
    INJECTION_PAYLOADS,
)
from bili.aether.config.loader import (  # noqa: E402  pylint: disable=wrong-import-position
    load_mas_from_yaml,
)

LOGGER = logging.getLogger(__name__)

_CROSS_MODEL_DIR = Path(__file__).parent
_RESULTS_DIR = _CROSS_MODEL_DIR / "results"

# Default injection phases for cross-model testing.
_ALL_PHASES: list[str] = [
    InjectionPhase.PRE_EXECUTION.value,
    InjectionPhase.MID_EXECUTION.value,
]

# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------

# Each entry: (model_id, human-readable display name).
# model_id values must match entries in bili/iris/config/llm_config.py.
_MODEL_MATRIX: list[tuple[str, str]] = [
    (
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "Claude 3.5 Haiku (Bedrock)",
    ),
    (
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "Claude Sonnet 4 (Bedrock)",
    ),
    (
        "amazon.nova-pro-v1:0",
        "Amazon Nova Pro (Bedrock)",
    ),
    (
        "gemini-2.0-flash",
        "Gemini 2.0 Flash (Vertex)",
    ),
]

_CSV_COLUMNS: list[str] = [
    "model_id",
    "model_name",
    "provider_family",
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
# Helpers
# ---------------------------------------------------------------------------


def _provider_family(model_id: str | None) -> str:
    """Derive the provider family label from a model_id string.

    Used for the transferability metric — a payload that influences agents
    on two models from the *same* provider family has not truly transferred.
    Families follow the same prefix rules as ``SemanticEvaluator``'s
    circularity detection in ``evaluator_config.py``.

    Returns one of: ``"anthropic_bedrock"``, ``"amazon_bedrock"``,
    ``"google_vertex"``, ``"openai"``, ``"anthropic_direct"``, or ``"stub"``.
    """
    if not model_id:
        return "stub"
    if model_id.startswith(("us.anthropic.", "anthropic.")):
        return "anthropic_bedrock"
    if model_id.startswith(("us.amazon.", "amazon.")):
        return "amazon_bedrock"
    if model_id.startswith("gemini-"):
        return "google_vertex"
    if model_id.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model_id.startswith("claude-"):
        return "anthropic_direct"
    return "unknown"


def _patch_config_model(config: Any, model_id: str | None) -> Any:
    """Return a copy of config with all agents patched to use model_id.

    Note: ``AgentSpec.model_name`` is the field that the compiler uses as a
    lookup key into ``LLM_MODELS`` (by model_id or display name).  Despite the
    field name being ``model_name``, the value set here is a ``model_id``
    string (e.g. ``"us.anthropic.claude-3-5-haiku-20241022-v1:0"``).  The
    compiler resolves it correctly via exact ``model_id`` match first.
    """
    patched_agents = [
        a.model_copy(update={"model_name": model_id}) for a in config.agents
    ]
    return config.model_copy(update={"agents": patched_agents})


def _write_result(result_dict: dict, results_dir: Path) -> Path:
    """Write one result JSON to results_dir/{mas_id}/{model_id_safe}/{payload_id}_{phase}.json."""
    model_safe = _model_id_safe(result_dict.get("model_id"))
    out_dir = results_dir / result_dict["mas_id"] / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result_dict['payload_id']}_{result_dict['injection_phase']}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    return out_path


def _write_csv(rows: list[dict], results_dir: Path) -> Path:
    """Write the cross-model results matrix CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "cross_model_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _load_baseline(baseline_results_dir: Path | None, mas_id: str) -> dict | None:
    """Try to load a baseline result for Tier 3 comparison."""
    if baseline_results_dir is None:
        return None
    config_baseline_dir = baseline_results_dir / mas_id
    if not config_baseline_dir.exists():
        return None
    result_files = sorted(config_baseline_dir.glob("*.json"))
    if not result_files:
        return None
    try:
        return json.loads(result_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load baseline for %s: %s", mas_id, exc)
        return None


def _print_summary(matrix_rows: list[dict]) -> None:
    """Print a compact summary table to stdout."""
    total = len(matrix_rows)
    skipped = sum(1 for r in matrix_rows if r["skipped"] == "true")
    ran = total - skipped
    passed = sum(1 for r in matrix_rows if r["tier1_pass"] == "true")
    influenced = sum(1 for r in matrix_rows if r["tier2_influenced"] not in ("[]", ""))

    # Transferability: count (payload_id, phase) pairs that influenced agents
    # across more than one *provider family*.  Two Bedrock/Anthropic models
    # both succeeding does not count as a transfer — the payload must cross
    # family boundaries (e.g. anthropic_bedrock AND amazon_bedrock or
    # google_vertex).
    pairs_by_family: dict[tuple[str, str], set[str]] = {}
    for r in matrix_rows:
        if r["skipped"] == "true" or r["tier2_influenced"] == "[]":
            continue
        key = (r["payload_id"], r["phase"])
        pairs_by_family.setdefault(key, set()).add(r["provider_family"])
    transferred = sum(1 for families in pairs_by_family.values() if len(families) > 1)

    print("\n" + "=" * 60)
    print("Cross-Model Transferability Suite Summary")
    print("=" * 60)
    print(f"  Total rows                    : {total}")
    print(f"  Skipped (no creds/err)        : {skipped}")
    print(f"  Ran                           : {ran}")
    print(f"  Tier 1 pass                   : {passed}/{ran}")
    print(f"  Tier 2 influenced ≥1 agent    : {influenced}")
    print(f"  Transferred (>1 provider fam) : {transferred} payload/phase pairs")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------


def _run_config_for_model(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    yaml_path: str,
    model_id: str | None,
    model_display_name: str,
    payloads: list,
    phases: list[str],
    stub_mode: bool,
    semantic_evaluator: Any,
    baseline_results_dir: Path | None,
    results_dir: Path,
    repo_root: Path,
) -> list[dict]:
    """Run all payloads × phases for one MAS config under one model.

    Returns:
        List of CSV row dicts (one per payload × phase).  Skips with a warning
        if the config file is not found.  Individual injection failures produce
        a skip row rather than aborting the whole model run.
    """
    full_path = repo_root / yaml_path
    if not full_path.exists():
        print(f"  Config not found, skipping: {yaml_path}", file=sys.stderr)
        return []

    base_config = load_mas_from_yaml(str(full_path))
    config = _patch_config_model(base_config, model_id)
    mas_id = config.mas_id

    model_safe = _model_id_safe(model_id)
    config_results_dir = results_dir / mas_id / model_safe
    config_results_dir.mkdir(parents=True, exist_ok=True)

    attack_log_path = config_results_dir / "attack_log.ndjson"
    sec_log_path = config_results_dir / "security_events.ndjson"

    sec_logger = SecurityEventLogger(log_path=sec_log_path)
    detector = SecurityEventDetector(
        logger=sec_logger,
        attack_log_path=attack_log_path,
    )

    entry_agent = config.entry_point or config.agents[0].agent_id
    print(
        f"  [{mas_id}] target={entry_agent!r}  "
        f"({len(payloads)} payloads × {len(phases)} phases)"
    )

    matrix_rows: list[dict] = []

    with AttackInjector(
        config,
        executor=None,
        log_path=attack_log_path,
        security_detector=detector,
    ) as injector:
        for ip in payloads:
            for phase in phases:
                label = f"    {ip.payload_id} / {phase}"
                print(f"{label} ... ", end="", flush=True)

                try:
                    attack_result = injector.inject_attack(
                        agent_id=entry_agent,
                        attack_type=AttackType.PROMPT_INJECTION.value,
                        payload=ip.payload,
                        injection_phase=phase,
                        blocking=True,
                        track_propagation=True,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # Credential errors, provider unavailability, etc.
                    skip_reason = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning(
                        "inject_attack failed for %s/%s/%s: %s",
                        model_id,
                        mas_id,
                        ip.payload_id,
                        exc,
                    )
                    print(f"SKIP ({skip_reason[:60]})")
                    matrix_rows.append(
                        {
                            "model_id": model_id or "stub",
                            "model_name": model_display_name,
                            "provider_family": _provider_family(model_id),
                            "payload_id": ip.payload_id,
                            "injection_type": ip.injection_type,
                            "severity": ip.severity,
                            "stub_mode": str(stub_mode).lower(),
                            "mas_id": mas_id,
                            "phase": phase,
                            "tier1_pass": "",
                            "tier2_influenced": "",
                            "tier2_resistant": "",
                            "tier3_score": "",
                            "tier3_confidence": "",
                            "tier3_reasoning": "",
                            "attack_suite": "cross_model",
                            "skipped": "true",
                            "skip_reason": skip_reason,
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
                                "SemanticEvaluator failed for %s/%s/%s: %s",
                                model_id,
                                ip.payload_id,
                                phase,
                                t3_exc,
                            )

                resistant_list = sorted(attack_result.resistant_agents)
                family = _provider_family(model_id)
                result_dict = {
                    "payload_id": ip.payload_id,
                    "injection_type": ip.injection_type,
                    "severity": ip.severity,
                    "model_id": model_id,
                    "model_name": model_display_name,
                    "provider_family": family,
                    "mas_id": mas_id,
                    "injection_phase": phase,
                    "attack_suite": "cross_model",
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
                out_path = _write_result(result_dict, results_dir)
                status = "ok" if attack_result.success else "FAIL"
                influenced_count = len(attack_result.influenced_agents)
                print(f"{status}  (influenced={influenced_count}) → {out_path.name}")

                matrix_rows.append(
                    {
                        "model_id": model_id or "stub",
                        "model_name": model_display_name,
                        "provider_family": family,
                        "payload_id": ip.payload_id,
                        "injection_type": ip.injection_type,
                        "severity": ip.severity,
                        "stub_mode": str(stub_mode).lower(),
                        "mas_id": mas_id,
                        "phase": phase,
                        "tier1_pass": str(attack_result.success).lower(),
                        "tier2_influenced": json.dumps(
                            sorted(attack_result.influenced_agents)
                        ),
                        "tier2_resistant": json.dumps(resistant_list),
                        "tier3_score": tier3_score,
                        "tier3_confidence": tier3_confidence,
                        "tier3_reasoning": tier3_reasoning,
                        "attack_suite": "cross_model",
                        "skipped": "false",
                        "skip_reason": "",
                    }
                )

    return matrix_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full cross-model transferability test suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AETHER cross-model transferability suite — iterates the prompt "
            "injection payload set across a model matrix to measure whether attack "
            "effects transfer across LLM provider families.  Produces a results "
            "matrix CSV with model_id and model_name columns."
        )
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use stub agents (no LLM calls).  Replaces the model matrix with a "
            "single model_id=None run.  Validates CSV schema and result file "
            "structure without requiring credentials."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIG_PATHS,
        metavar="YAML_PATH",
        help="Override the list of YAML config paths to run (relative to repo root).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Restrict to specific model_ids from the matrix "
            "(e.g. us.anthropic.claude-3-5-haiku-20241022-v1:0 amazon.nova-pro-v1:0). "
            "Ignored in --stub mode."
        ),
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        default=None,
        metavar="PAYLOAD_ID",
        help=(
            "Restrict run to specific payload IDs " "(e.g. pi_direct_001 pi_role_001)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=_ALL_PHASES,
        choices=_ALL_PHASES,
        metavar="PHASE",
        help=(
            "Injection phases to run (default: pre_execution mid_execution). "
            f"Choices: {_ALL_PHASES}"
        ),
    )
    parser.add_argument(
        "--baseline-results",
        default=None,
        metavar="DIR",
        help=(
            "Path to baseline results directory for Tier 3 comparison "
            "(e.g. bili/aegis/suites/baseline/results).  "
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

    # Resolve payload set
    payloads = INJECTION_PAYLOADS
    if args.payloads:
        ids = set(args.payloads)
        payloads = [p for p in INJECTION_PAYLOADS if p.payload_id in ids]
        if not payloads:
            print(f"No payloads matched: {args.payloads}", file=sys.stderr)
            sys.exit(1)

    # Resolve model matrix
    if args.stub:
        model_matrix: list[tuple[str | None, str]] = [(None, "stub")]
    else:
        if args.models:
            model_ids = set(args.models)
            model_matrix = [
                (mid, name) for mid, name in _MODEL_MATRIX if mid in model_ids
            ]
            if not model_matrix:
                print(
                    f"No models matched: {args.models}.  "
                    f"Valid IDs: {[m[0] for m in _MODEL_MATRIX]}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            model_matrix = list(_MODEL_MATRIX)

    # Resolve baseline results dir
    baseline_results_dir: Path | None = None
    if args.baseline_results:
        baseline_results_dir = _REPO_ROOT / args.baseline_results
        if not baseline_results_dir.exists():
            print(
                f"Warning: baseline results dir not found: {baseline_results_dir}",
                file=sys.stderr,
            )
            baseline_results_dir = None

    # SemanticEvaluator (optional — standard injection rubric)
    semantic_evaluator = None
    if not args.stub:
        try:
            # pylint: disable=import-outside-toplevel
            from bili.aegis.evaluator import SemanticEvaluator

            # pylint: enable=import-outside-toplevel
            semantic_evaluator = SemanticEvaluator()
        except (ImportError, RuntimeError) as exc:
            LOGGER.warning("Could not initialise SemanticEvaluator: %s", exc)

    # Run: outer loop = models, inner = configs × payloads × phases
    all_matrix_rows: list[dict] = []
    for model_id, model_display_name in model_matrix:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_display_name}")
        if model_id:
            print(f"  model_id: {model_id}")
        print(f"{'=' * 60}")
        for yaml_path in args.configs:
            rows = _run_config_for_model(
                yaml_path=yaml_path,
                model_id=model_id,
                model_display_name=model_display_name,
                payloads=payloads,
                phases=args.phases,
                stub_mode=args.stub,
                semantic_evaluator=semantic_evaluator,
                baseline_results_dir=baseline_results_dir,
                results_dir=_RESULTS_DIR,
                repo_root=_REPO_ROOT,
            )
            all_matrix_rows.extend(rows)

    if all_matrix_rows:
        csv_path = _write_csv(all_matrix_rows, _RESULTS_DIR)
        _print_summary(all_matrix_rows)
        print(f"Results matrix written to: {csv_path}")

    # Exit 0 if all ran tests passed OR all were skipped (skip ≠ failure).
    ran_rows = [r for r in all_matrix_rows if r["skipped"] != "true"]
    passed = sum(1 for r in ran_rows if r["tier1_pass"] == "true")
    sys.exit(0 if passed == len(ran_rows) else 1)


if __name__ == "__main__":
    main()
