"""Structural baseline tests — CI-safe, model-agnostic.

These tests assert schema validity and structural completeness of baseline
result files.  They make no assertions about *what* agents said — only that
the execution completed and produced well-formed output.

Detection tier: Structural (Tier 1)
Runs in: stub mode and real-LLM mode
CI-safe: yes — hard pass/fail, no LLM calls required

The ``baseline_result`` fixture (from conftest.py) is parametrized over every
JSON file in ``results/``.  When the directory is empty, all tests are skipped.

Run:
    python -m pytest bili/aegis/tests/baseline/ -v
"""

import pytest

# ---------------------------------------------------------------------------
# Tier 1 — Structural assertions
# ---------------------------------------------------------------------------


@pytest.mark.structural
def test_all_agents_executed(baseline_result: dict) -> None:
    """Execution must have succeeded and at least one agent must have run."""
    assert baseline_result["execution"]["success"] is True, (
        f"Execution failed for {baseline_result['mas_id']} / "
        f"{baseline_result['prompt_id']}: {baseline_result.get('execution')}"
    )
    assert (
        len(baseline_result["agent_outputs"]) > 0
    ), "agent_outputs is empty — no agents ran"


@pytest.mark.structural
def test_all_agents_have_output(baseline_result: dict) -> None:
    """At least one agent must have produced raw or parsed output.

    In supervisor/hierarchical workflows, not all agents are routed to
    on every run. Agents that the supervisor didn't invoke will have
    ``raw=None, parsed=None`` which is valid.
    """
    outputs = baseline_result["agent_outputs"]
    agents_with_output = [
        aid
        for aid, out in outputs.items()
        if out.get("raw") is not None or out.get("parsed") is not None
    ]
    assert agents_with_output, (
        f"{baseline_result['mas_id']} / {baseline_result['prompt_id']} "
        f"has {len(outputs)} agents but none produced output"
    )


@pytest.mark.structural
def test_execution_metadata_present(baseline_result: dict) -> None:
    """Execution metadata fields must be non-negative numeric values."""
    execution = baseline_result["execution"]
    assert execution["duration_ms"] >= 0, "duration_ms must be >= 0"
    assert execution["message_count"] >= 0, "message_count must be >= 0"
    assert execution["agent_count"] > 0, "agent_count must be > 0"


@pytest.mark.structural
def test_config_fingerprint_present(baseline_result: dict) -> None:
    """Config fingerprint must be present and non-empty for reproducibility."""
    fp = baseline_result.get("config_fingerprint", {})
    assert fp.get("yaml_hash"), "config_fingerprint.yaml_hash is missing or empty"
    assert fp.get("model_name"), "config_fingerprint.model_name is missing or empty"
    assert isinstance(
        fp.get("temperature"), dict
    ), "config_fingerprint.temperature must be a dict of agent_id -> float"


@pytest.mark.structural
def test_run_metadata_present(baseline_result: dict) -> None:
    """run_metadata must be present with required fields."""
    meta = baseline_result.get("run_metadata", {})
    assert meta.get("timestamp"), "run_metadata.timestamp is missing"
    assert isinstance(
        meta.get("stub_mode"), bool
    ), "run_metadata.stub_mode must be a bool"
    assert "semantic_tier" in meta, "run_metadata.semantic_tier is missing"


@pytest.mark.structural
def test_prompt_fields_present(baseline_result: dict) -> None:
    """Top-level prompt identity fields must be present and non-empty."""
    assert baseline_result.get("prompt_id"), "prompt_id is missing"
    assert baseline_result.get("prompt_category") in (
        "benign",
        "violating",
        "edge_case",
    ), (
        f"prompt_category must be benign|violating|edge_case, "
        f"got: {baseline_result.get('prompt_category')!r}"
    )
    assert baseline_result.get("prompt_text"), "prompt_text is missing"
    assert baseline_result.get("mas_id"), "mas_id is missing"
