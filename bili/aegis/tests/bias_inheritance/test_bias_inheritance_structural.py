"""Tier 1 structural assertions for the AETHER bias inheritance test suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files
written by ``run_bias_inheritance_suite.py``.

All tests are parametrized via the ``bias_inheritance_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first:

    python bili/aegis/suites/bias_inheritance/run_bias_inheritance_suite.py --stub
    pytest bili/aegis/tests/bias_inheritance/test_bias_inheritance_structural.py -v

Detection tier: Tier 1 (structural).

Note: Bias inheritance is a subtle attack — Tier 2 heuristic detection is less
reliable than for prompt injection or memory poisoning.  Tier 3 semantic
evaluation (SemanticEvaluator) is the primary signal for real-mode runs.
"""

from pathlib import Path


def test_bias_inheritance_execution_succeeded(bias_inheritance_result: dict) -> None:
    """Bias inheritance run completed without an unhandled error."""
    assert bias_inheritance_result["execution"]["success"] is True


def test_propagation_tracking_ran(bias_inheritance_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in bias_inheritance_result
    assert isinstance(bias_inheritance_result["propagation_path"], list)


def test_security_events_logged(bias_inheritance_result: dict, log_dir: Path) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    assert bias_inheritance_result["mas_id"] == log_dir.name
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the bias inheritance suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(bias_inheritance_result: dict, log_dir: Path) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    assert bias_inheritance_result["mas_id"] == log_dir.name
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the bias inheritance suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
