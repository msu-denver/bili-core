"""Shared suite runner for AETHER injection and jailbreak test suites.

Provides :func:`run_suite` — a single parameterized entry point used by both
``run_injection_suite.py`` and ``run_jailbreak_suite.py``.  Each runner
defines its own payload list, attack type, CSV filename, and evaluator config,
then delegates all execution to this module.

This module must NOT be imported before ``sys.path`` contains the repo root.
The inline ``_find_repo_root`` / bootstrap code in each runner script handles
that before importing this module.
"""

import csv
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

from bili.aegis.attacks.injector import AttackInjector
from bili.aegis.attacks.models import InjectionPhase
from bili.aegis.security.detector import SecurityEventDetector
from bili.aegis.security.logger import SecurityEventLogger
from bili.aegis.suites._helpers import config_fingerprint as _config_fingerprint_helper
from bili.aegis.suites._helpers import latest_run_dir, next_run_dir
from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.runtime.executor import MASExecutor

LOGGER = logging.getLogger(__name__)

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
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _target_agent_id(config: Any) -> str:
    """Return the injection target agent for a MAS config.

    Uses ``config.entry_point`` when set (e.g. supervisor workflows); falls
    back to the first agent in the config's agent list.

    Raises:
        ValueError: If the config has no agents defined.
    """
    if not config.agents:
        raise ValueError(
            f"MAS config '{config.mas_id}' has no agents defined — "
            "cannot determine injection target"
        )
    if config.entry_point:
        return config.entry_point
    return config.agents[0].agent_id


def _build_result_dict(
    ip: Any,
    config: Any,
    yaml_path: str,
    phase: str,
    attack_result: Any,
    stub_mode: bool,
    tier3_rows: list | None,
    attack_suite: str,
    repo_root: Path,
) -> dict:
    """Assemble the full result JSON dict for one test case."""
    tier3_score: str = ""
    tier3_confidence: str = ""
    tier3_reasoning: str = ""

    if tier3_rows:
        best = max(tier3_rows, key=lambda v: v.score)
        tier3_score = str(best.score)
        tier3_confidence = best.confidence
        tier3_reasoning = best.reasoning

    resistant_list = sorted(attack_result.resistant_agents)

    return {
        "payload_id": ip.payload_id,
        "injection_type": ip.injection_type,
        "severity": ip.severity,
        "mas_id": config.mas_id,
        "injection_phase": phase,
        "attack_suite": attack_suite,
        "config_fingerprint": _config_fingerprint_helper(config, yaml_path, repo_root),
        "execution": {
            "success": attack_result.success,
            "duration_ms": (
                (attack_result.completed_at - attack_result.injected_at).total_seconds()
                * 1000
                if attack_result.completed_at
                else 0.0
            ),
            "agent_count": len(attack_result.propagation_path),
            "message_count": 0,
        },
        "target_agent_id": attack_result.target_agent_id,
        "propagation_path": attack_result.propagation_path,
        "influenced_agents": attack_result.influenced_agents,
        "resistant_agents": resistant_list,
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stub_mode": stub_mode,
            "semantic_tier": (
                "skipped" if (stub_mode or tier3_rows is None) else "evaluated"
            ),
            "tier3_score": tier3_score,
            "tier3_confidence": tier3_confidence,
            "tier3_reasoning": tier3_reasoning,
        },
    }


def _write_result(
    result_dict: dict, results_dir: Path, run_dir: Path | None = None
) -> Path:
    """Write one result JSON to run_dir/{payload_id}_{phase}.json.

    When *run_dir* is provided it is used directly (versioned layout).
    Falls back to the legacy ``results_dir/{mas_id}/`` flat layout when
    *run_dir* is ``None`` so existing callers continue to work unchanged.
    """
    out_dir = run_dir if run_dir is not None else (results_dir / result_dict["mas_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result_dict['payload_id']}_{result_dict['injection_phase']}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    return out_path


def _write_csv(rows: list[dict], results_dir: Path, csv_filename: str) -> Path:
    """Write the results matrix CSV to results_dir/{csv_filename}.

    The CSV is fully RFC 4180-compliant — no comment lines are prepended.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / csv_filename

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _print_summary(matrix_rows: list[dict], suite_name: str) -> None:
    """Print a compact summary table to stdout."""
    total = len(matrix_rows)
    passed = sum(1 for r in matrix_rows if r["tier1_pass"] == "true")
    influenced_any = sum(1 for r in matrix_rows if r["tier2_influenced"] != "[]")

    print("\n" + "=" * 60)
    print(f"{suite_name} Summary")
    print("=" * 60)
    print(f"  Total test cases : {total}")
    print(f"  Tier 1 pass      : {passed}/{total}")
    print(f"  Tier 2 influenced: {influenced_any} cases had ≥1 influenced agent")
    print("=" * 60 + "\n")


def _load_baseline(
    baseline_results_dir: Path | None,
    mas_id: str,
) -> dict | None:
    """Try to load a baseline result for Tier 3 comparison.

    Prefers the most recent ``run_NNN`` subdirectory under
    ``baseline_results_dir / mas_id``; falls back to the flat legacy layout
    when no versioned run directories exist.

    Returns the first available baseline result for this config
    (alphabetically within the resolved directory), or ``None`` if no
    baseline is available.
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


def _run_config(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    yaml_path: str,
    payloads: list,
    phases: list[str],
    stub_mode: bool,
    semantic_evaluator: Any,
    baseline_results_dir: Path | None,
    results_dir: Path,
    repo_root: Path,
    attack_suite: str,
    attack_type: str,
) -> tuple[list[dict], Path]:
    """Run all payloads × phases for one MAS config.

    Returns a ``(matrix_rows, run_dir)`` tuple.  *run_dir* is the versioned
    ``run_NNN`` directory created for this run's result files.
    """
    full_path = repo_root / yaml_path
    if not full_path.exists():
        print(f"  Config not found, skipping: {yaml_path}", file=sys.stderr)
        fallback_dir = results_dir / Path(yaml_path).stem / "run_000"
        return [], fallback_dir

    config = load_mas_from_yaml(str(full_path))
    if stub_mode:
        for agent in config.agents:
            agent.model_name = None

    mas_id = config.mas_id
    target_id = _target_agent_id(config)

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

    needs_executor = InjectionPhase.MID_EXECUTION.value in phases
    executor = None
    if needs_executor:
        executor = MASExecutor(config)
        executor.initialize()

    matrix_rows: list[dict] = []
    print(
        f"\n[{mas_id}] target_agent={target_id!r}  "
        f"({len(payloads)} payloads × {len(phases)} phases)"
    )

    with AttackInjector(
        config,
        executor,
        log_path=attack_log_path,
        security_detector=detector,
    ) as injector:
        for ip in payloads:
            for phase in phases:
                label = f"  {ip.payload_id} / {phase}"
                print(f"{label} ... ", end="", flush=True)

                try:
                    attack_result = injector.inject_attack(
                        agent_id=target_id,
                        attack_type=attack_type,
                        payload=ip.payload,
                        injection_phase=phase,
                        blocking=True,
                        track_propagation=True,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    LOGGER.error(
                        "inject_attack failed for %s / %s / %s: %s",
                        mas_id,
                        ip.payload_id,
                        phase,
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
                            "phase": phase,
                            "tier1_pass": "false",
                            "tier2_influenced": "[]",
                            "tier2_resistant": "[]",
                            "tier3_score": "",
                            "tier3_confidence": "",
                            "tier3_reasoning": str(exc),
                            "attack_suite": attack_suite,
                        }
                    )
                    continue

                tier3_rows = None
                if not stub_mode and semantic_evaluator is not None:
                    baseline_dict = _load_baseline(baseline_results_dir, mas_id)
                    if baseline_dict is not None:
                        try:
                            tier3_rows = semantic_evaluator.evaluate(
                                baseline_result=baseline_dict,
                                attack_result=attack_result,
                            )
                        except (
                            Exception
                        ) as t3_exc:  # pylint: disable=broad-exception-caught
                            LOGGER.error(
                                "SemanticEvaluator failed for %s/%s: %s",
                                ip.payload_id,
                                phase,
                                t3_exc,
                                exc_info=True,
                            )

                result_dict = _build_result_dict(
                    ip=ip,
                    config=config,
                    yaml_path=yaml_path,
                    phase=phase,
                    attack_result=attack_result,
                    stub_mode=stub_mode,
                    tier3_rows=tier3_rows,
                    attack_suite=attack_suite,
                    repo_root=repo_root,
                )

                out_path = _write_result(result_dict, results_dir, run_dir=run_dir)
                status = "ok" if attack_result.success else "FAIL"
                influenced_count = len(attack_result.influenced_agents)
                print(f"{status}  (influenced={influenced_count}) → {out_path.name}")

                influenced_json = json.dumps(sorted(attack_result.influenced_agents))
                resistant_json = json.dumps(sorted(attack_result.resistant_agents))

                matrix_rows.append(
                    {
                        "payload_id": ip.payload_id,
                        "injection_type": ip.injection_type,
                        "severity": ip.severity,
                        "stub_mode": stub_mode,
                        "mas_id": mas_id,
                        "phase": phase,
                        "tier1_pass": str(attack_result.success).lower(),
                        "tier2_influenced": influenced_json,
                        "tier2_resistant": resistant_json,
                        "tier3_score": result_dict["run_metadata"]["tier3_score"],
                        "tier3_confidence": result_dict["run_metadata"][
                            "tier3_confidence"
                        ],
                        "tier3_reasoning": result_dict["run_metadata"][
                            "tier3_reasoning"
                        ],
                        "attack_suite": attack_suite,
                    }
                )

    return matrix_rows, run_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_suite(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    *,
    payloads: list,
    attack_suite: str,
    attack_type: str,
    csv_filename: str,
    suite_name: str,
    results_dir: Path,
    repo_root: Path,
    config_paths: list[str],
    phases: list[str],
    stub: bool,
    semantic_evaluator: Any = None,
    baseline_results_dir: Path | None = None,
) -> None:
    """Run a full AETHER test suite and write results + CSV matrix.

    Args:
        payloads:             List of payload objects (must have ``payload_id``,
                              ``injection_type``, ``severity``, ``payload`` attrs).
        attack_suite:         Suite label written to ``attack_suite`` CSV column.
        attack_type:          ``AttackType`` value string passed to
                              ``AttackInjector.inject_attack()``.
        csv_filename:         Output CSV filename (written to *results_dir*).
        suite_name:           Human-readable name for the summary table.
        results_dir:          Directory where per-case JSON and CSV are written.
        repo_root:            Absolute path to the repository root.
        config_paths:         YAML config paths relative to *repo_root*.
        phases:               Injection phases to run.
        stub:                 If ``True``, skip LLM calls (stub agent mode).
        semantic_evaluator:   Pre-constructed ``SemanticEvaluator`` instance,
                              or ``None`` to skip Tier 3 evaluation.
        baseline_results_dir: Path to baseline results for Tier 3 comparison.
    """
    all_matrix_rows: list[dict] = []
    first_run_dir_name: str | None = None
    for yaml_path in config_paths:
        rows, run_dir = _run_config(
            yaml_path=yaml_path,
            payloads=payloads,
            phases=phases,
            stub_mode=stub,
            semantic_evaluator=semantic_evaluator,
            baseline_results_dir=baseline_results_dir,
            results_dir=results_dir,
            repo_root=repo_root,
            attack_suite=attack_suite,
            attack_type=attack_type,
        )
        all_matrix_rows.extend(rows)
        if first_run_dir_name is None:
            first_run_dir_name = run_dir.name

    if all_matrix_rows:
        csv_stem = Path(csv_filename).stem
        versioned_csv = f"{csv_stem}_{first_run_dir_name}.csv"
        csv_path = _write_csv(all_matrix_rows, results_dir, versioned_csv)
        _print_summary(all_matrix_rows, suite_name)
        print(f"Results matrix written to: {csv_path}")

    total = len(all_matrix_rows)
    passed = sum(1 for r in all_matrix_rows if r["tier1_pass"] == "true")
    sys.exit(0 if passed == total else 1)
