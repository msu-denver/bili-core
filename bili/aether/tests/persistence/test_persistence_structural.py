"""Tier 1 structural assertions for the AETHER persistence attack test suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files
written by ``run_persistence_suite.py``.

All tests are parametrized via the ``persistence_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first (requires a persistent checkpointer):

    python bili/aether/tests/persistence/run_persistence_suite.py
    pytest bili/aether/tests/persistence/test_persistence_structural.py -v

Detection tier: Tier 1 (structural).
"""

from pathlib import Path


def test_persistence_execution_succeeded(persistence_result: dict) -> None:
    """Persistence injection cycle completed without an unhandled error."""
    assert persistence_result["execution"]["success"] is True


def test_propagation_tracking_ran(persistence_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in persistence_result
    assert isinstance(persistence_result["propagation_path"], list)


def test_security_events_logged(  # pylint: disable=redefined-outer-name
    persistence_result: dict, log_dir: Path
) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the persistence suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(  # pylint: disable=redefined-outer-name
    persistence_result: dict, log_dir: Path
) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the persistence suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
