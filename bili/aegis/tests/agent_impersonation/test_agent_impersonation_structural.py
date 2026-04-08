"""Tier 1 structural assertions for the AETHER agent impersonation test suite.

These tests are CI-safe and stub-mode compatible: they make no LLM calls,
perform no semantic reasoning, and rely only on the JSON result files
written by ``run_agent_impersonation_suite.py``.

All tests are parametrized via the ``agent_impersonation_result`` fixture in
``conftest.py``.  When ``results/`` is empty the tests are automatically
skipped — run the suite first:

    python bili/aegis/tests/agent_impersonation/run_agent_impersonation_suite.py --stub
    pytest bili/aegis/tests/agent_impersonation/test_agent_impersonation_structural.py -v

Detection tier: Tier 1 (structural).

Note: Agent impersonation Tier 2 heuristic detection relies on identity-change
markers ("i am now", "acting as", "my new role", "new identity") which are
relatively explicit signals — Tier 2 is more reliable here than for bias
inheritance, but Tier 3 semantic evaluation remains the authoritative measure
for real-mode runs.
"""

from pathlib import Path


def test_agent_impersonation_execution_succeeded(
    agent_impersonation_result: dict,
) -> None:
    """Agent impersonation run completed without an unhandled error."""
    assert agent_impersonation_result["execution"]["success"] is True


def test_propagation_tracking_ran(agent_impersonation_result: dict) -> None:
    """Propagation path was recorded (may be empty for stub agents)."""
    assert "propagation_path" in agent_impersonation_result
    assert isinstance(agent_impersonation_result["propagation_path"], list)


def test_security_events_logged(
    agent_impersonation_result: dict, log_dir: Path  # pylint: disable=unused-argument
) -> None:
    """SecurityEventLogger wrote at least one event to security_events.ndjson."""
    events_file = log_dir / "security_events.ndjson"
    assert events_file.exists(), (
        f"security_events.ndjson not found in {log_dir}. "
        "Re-run the agent impersonation suite to regenerate log files."
    )
    assert events_file.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(
    agent_impersonation_result: dict, log_dir: Path  # pylint: disable=unused-argument
) -> None:
    """AttackLogger wrote at least one record to attack_log.ndjson."""
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), (
        f"attack_log.ndjson not found in {log_dir}. "
        "Re-run the agent impersonation suite to regenerate log files."
    )
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
