"""Tier 1 structural assertions for the AETHER cross-model transferability suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files written
by ``run_cross_model_suite.py``.

All tests are parametrized via the ``cross_model_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first:

    python bili/aegis/suites/cross_model/run_cross_model_suite.py --stub
    pytest bili/aegis/tests/cross_model/test_cross_model_structural.py -v

Detection tier: Tier 1 (structural).
"""

from pathlib import Path


def test_cross_model_execution_succeeded(cross_model_result: dict) -> None:
    """Attack injection cycle completed without an unhandled error."""
    assert cross_model_result["execution"]["success"] is True


def test_model_id_present(cross_model_result: dict) -> None:
    """Result JSON contains a model_id field (None for stub mode)."""
    assert "model_id" in cross_model_result


def test_model_name_present(cross_model_result: dict) -> None:
    """Result JSON contains a non-empty model_name field."""
    assert "model_name" in cross_model_result
    assert cross_model_result["model_name"]


def test_propagation_tracking_ran(cross_model_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in cross_model_result
    assert isinstance(cross_model_result["propagation_path"], list)


def test_security_events_logged(cross_model_result: dict, log_dir: Path) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    assert cross_model_result["mas_id"] == log_dir.parent.name
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the cross-model suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(cross_model_result: dict, log_dir: Path) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    assert cross_model_result["mas_id"] == log_dir.parent.name
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the cross-model suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
