"""Tier 1 structural tests for the AETHER jailbreak test suite.

These tests assert structural invariants on every result JSON produced by
``run_jailbreak_suite.py``.  They make no LLM calls and pass in stub mode.

Run after the suite:

    python bili/aether/tests/jailbreak/run_jailbreak_suite.py --stub
    pytest bili/aether/tests/jailbreak/test_jailbreak_structural.py -v
"""


def test_jailbreak_execution_succeeded(jailbreak_result):
    assert jailbreak_result["execution"]["success"] is True


def test_propagation_tracking_ran(jailbreak_result):
    assert "propagation_path" in jailbreak_result
    assert isinstance(jailbreak_result["propagation_path"], list)


def test_security_events_logged(jailbreak_result, log_dir):
    sec_log = log_dir / "security_events.ndjson"
    assert sec_log.exists(), f"security_events.ndjson not found in {log_dir}"
    assert sec_log.stat().st_size > 0, "security_events.ndjson is empty"


def test_attack_log_written(jailbreak_result, log_dir):
    attack_log = log_dir / "attack_log.ndjson"
    assert attack_log.exists(), f"attack_log.ndjson not found in {log_dir}"
    assert attack_log.stat().st_size > 0, "attack_log.ndjson is empty"
