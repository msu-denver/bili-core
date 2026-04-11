"""Tier 1 structural assertions for the AETHER memory poisoning test suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files
written by ``run_memory_poisoning_suite.py``.

All tests are parametrized via the ``memory_poisoning_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first:

    python bili/aegis/suites/memory_poisoning/run_memory_poisoning_suite.py --stub
    pytest bili/aegis/tests/memory_poisoning/test_memory_poisoning_structural.py -v

Detection tier: Tier 1 (structural).
"""

from pathlib import Path


def test_memory_poisoning_execution_succeeded(memory_poisoning_result: dict) -> None:
    """Memory poisoning run completed without an unhandled error."""
    assert memory_poisoning_result["execution"]["success"] is True


def test_propagation_tracking_ran(memory_poisoning_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in memory_poisoning_result
    assert isinstance(memory_poisoning_result["propagation_path"], list)


def test_security_events_logged(memory_poisoning_result: dict, log_dir: Path) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    assert memory_poisoning_result["mas_id"] == log_dir.name
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the memory poisoning suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(memory_poisoning_result: dict, log_dir: Path) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    assert memory_poisoning_result["mas_id"] == log_dir.name
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the memory poisoning suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
