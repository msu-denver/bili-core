"""Tier 1 structural assertions for the AETHER injection test suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files
written by ``run_injection_suite.py``.

All tests are parametrized via the ``injection_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first:

    python bili/aether/tests/injection/run_injection_suite.py --stub
    pytest bili/aether/tests/injection/test_injection_structural.py -v

Detection tier: Tier 1 (structural).
"""

from pathlib import Path


def test_injection_execution_succeeded(injection_result: dict) -> None:
    """Injection run completed without an unhandled error."""
    assert injection_result["execution"]["success"] is True


def test_propagation_tracking_ran(injection_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in injection_result
    assert isinstance(injection_result["propagation_path"], list)


def test_security_events_logged(injection_result: dict, log_dir: Path) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the injection suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(injection_result: dict, log_dir: Path) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the injection suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
