"""Tests for AgentSpec schema (domain-agnostic design)."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from bili.aether.schema import AgentSpec, OutputFormat

# Small, self-contained catalog used to isolate the per-model prompt-length
# tests from the real (large, ever-growing) LLM_MODELS catalog.
_MOCK_CATALOG = {
    "remote_test_large_context": {
        "models": [
            {
                "model_name": "Large Context Test Model",
                "model_id": "large-context-test-model",
                "max_input_tokens": 200000,
            },
        ]
    },
    "remote_test_small_context": {
        "models": [
            {
                "model_name": "Small Context Test Model",
                "model_id": "small-context-test-model",
                "max_input_tokens": 1000,
            },
        ]
    },
    "cli_test": {
        "models": [
            {
                # CLI-subprocess-style entry: no declared max_input_tokens,
                # mirroring the real cli/cli_claude_code/cli_codex/
                # cli_gemini_cli catalog entries.
                "model_name": "CLI Test Model",
                "model_id": "cli:test",
            },
        ]
    },
}


def test_minimal_agent():
    """Test creating agent with minimal fields."""
    agent = AgentSpec(
        agent_id="test_agent",
        role="content_reviewer",
        objective="Test objective",
    )

    assert agent.agent_id == "test_agent"
    assert agent.role == "content_reviewer"
    assert agent.temperature is None  # Default (uses provider default)


def test_any_role_allowed():
    """Test that any role string is allowed (domain-agnostic)."""
    # Content moderation role
    agent1 = AgentSpec(
        agent_id="reviewer",
        role="content_reviewer",
        objective="Review content for violations",
    )
    assert agent1.role == "content_reviewer"

    # Research role
    agent2 = AgentSpec(
        agent_id="researcher",
        role="senior_researcher",
        objective="Research and analyze topics",
    )
    assert agent2.role == "senior_researcher"

    # Custom domain role
    agent3 = AgentSpec(
        agent_id="code_reviewer",
        role="security_engineer",
        objective="Review code for security issues",
    )
    assert agent3.role == "security_engineer"


def test_any_capabilities_allowed():
    """Test that any capability strings are allowed."""
    agent = AgentSpec(
        agent_id="custom_agent",
        role="custom_role",
        objective="Agent with custom capabilities",
        capabilities=["custom_cap_1", "custom_cap_2", "analysis"],
    )
    assert "custom_cap_1" in agent.capabilities
    assert len(agent.capabilities) == 3


def test_structured_output_requires_schema():
    """Test that structured output requires output_schema."""
    with pytest.raises(ValidationError, match="output_schema"):
        AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            output_format=OutputFormat.STRUCTURED,
            # Missing output_schema!
        )


def test_structured_output_with_schema():
    """Test structured output with schema."""
    agent = AgentSpec(
        agent_id="test",
        role="judge",
        objective="Test objective for validation",
        output_format=OutputFormat.STRUCTURED,
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.output_schema is not None


def test_hierarchical_agent():
    """Test agent with tier for hierarchical workflows."""
    agent = AgentSpec(
        agent_id="vote_agent",
        role="judge",
        objective="Vote on content moderation",
        tier=1,
        voting_weight=2.0,
    )
    assert agent.tier == 1
    assert agent.voting_weight == 2.0


def test_supervisor_agent():
    """Test supervisor agent."""
    agent = AgentSpec(
        agent_id="supervisor",
        role="supervisor",
        objective="Coordinate specialists",
        is_supervisor=True,
    )
    assert agent.is_supervisor is True


def test_display_name():
    """Test display name generation from role."""
    agent = AgentSpec(
        agent_id="reviewer",
        role="content_reviewer",
        objective="Review flagged content",
    )
    assert agent.get_display_name() == "Content Reviewer"


def test_invalid_agent_id():
    """Test that invalid agent IDs are rejected."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="invalid id with spaces",  # no spaces allowed
            role="judge",
            objective="Test objective for validation",
        )


def test_temperature_bounds():
    """Test temperature must be 0.0-2.0."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            temperature=3.0,  # too high
        )


# =========================================================================
# INHERITANCE TESTS
# =========================================================================


def test_inherit_defaults():
    """Test that inheritance sub-flags default to True."""
    agent = AgentSpec(
        agent_id="test_inherit",
        role="judge",
        objective="Test inheritance defaults",
        inherit_from_bili_core=True,
    )
    assert agent.inherit_from_bili_core is True
    assert agent.inherit_llm_config is True
    assert agent.inherit_tools is True
    assert agent.inherit_system_prompt is True
    assert agent.inherit_memory is True
    assert agent.inherit_checkpoint is True


def test_inherit_selective():
    """Test selective inheritance (opt out of specific features)."""
    agent = AgentSpec(
        agent_id="selective",
        role="judge",
        objective="Test selective inheritance",
        inherit_from_bili_core=True,
        inherit_system_prompt=False,
        inherit_tools=False,
        system_prompt="Custom prompt override",
    )
    assert agent.inherit_from_bili_core is True
    assert agent.inherit_system_prompt is False
    assert agent.inherit_tools is False
    assert agent.inherit_llm_config is True  # Still inherited


def test_inherit_disabled_resets_subflags():
    """Test that sub-flags are reset when master toggle is off."""
    agent = AgentSpec(
        agent_id="no_inherit",
        role="judge",
        objective="Test inheritance disabled",
        inherit_from_bili_core=False,
        inherit_tools=False,  # Should be reset to True
    )
    assert agent.inherit_from_bili_core is False
    # Sub-flags normalised to defaults when master toggle is off
    assert agent.inherit_tools is True


# =========================================================================
# CONSENSUS FIELD TESTS
# =========================================================================


def test_consensus_vote_field_requires_json():
    """Test consensus_vote_field requires JSON or structured output."""
    with pytest.raises(ValidationError, match="consensus_vote_field"):
        AgentSpec(
            agent_id="voter",
            role="voter",
            objective="Vote on decisions",
            output_format=OutputFormat.TEXT,  # Not JSON!
            consensus_vote_field="decision",
        )


def test_consensus_vote_field_with_json():
    """Test consensus_vote_field works with JSON output."""
    agent = AgentSpec(
        agent_id="voter",
        role="voter",
        objective="Vote on decisions",
        output_format=OutputFormat.JSON,
        consensus_vote_field="decision",
    )
    assert agent.consensus_vote_field == "decision"


# =========================================================================
# SYSTEM PROMPT LENGTH TESTS (per-model, not a fixed hardcoded cap)
# =========================================================================


def test_system_prompt_no_hardcoded_cap_without_model():
    """A long system_prompt validates fine when no model_name is bound.

    There is no longer a fixed character ceiling (the old max_length=10000)
    on this field -- a single number is wrong for the catalog as a whole
    regardless of what it is. With no model_name to check against, there is
    nothing to validate, so length is unconstrained.
    """
    agent = AgentSpec(
        agent_id="test",
        role="judge",
        objective="Test objective for validation",
        system_prompt="x" * 15000,
    )
    assert len(agent.system_prompt) == 15000


def test_system_prompt_no_hardcoded_cap_with_unknown_model():
    """A long system_prompt validates fine for a model absent from the catalog.

    An unknown limit must never manifest as an arbitrary cap.
    """
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        agent = AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            model_name="totally-unrecognized-model-xyz",
            system_prompt="x" * 15000,
        )
        assert len(agent.system_prompt) == 15000


def test_system_prompt_no_hardcoded_cap_for_cli_provider():
    """A long system_prompt validates for a CLI-provider model with no declared limit.

    CLI-subprocess and local providers do not declare max_input_tokens in the
    catalog (their real limit depends on the underlying tool/hardware), so
    they must be treated permissively, exactly like an unknown model.
    """
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        agent = AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            model_name="cli:test",
            system_prompt="x" * 15000,
        )
        assert len(agent.system_prompt) == 15000


def test_get_prompt_length_limit_returns_none_without_model_name():
    """get_prompt_length_limit() returns None when model_name is unset."""
    agent = AgentSpec(
        agent_id="test",
        role="judge",
        objective="Test objective for validation",
    )
    assert agent.get_prompt_length_limit() is None


def test_get_prompt_length_limit_returns_none_for_unknown_model():
    """get_prompt_length_limit() returns None for a model absent from the catalog."""
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        agent = AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            model_name="totally-unrecognized-model-xyz",
        )
        assert agent.get_prompt_length_limit() is None


def test_get_prompt_length_limit_is_queryable_per_model():
    """get_prompt_length_limit() surfaces the bound model's declared limit.

    This is the mechanism a consumer uses to budget a composed prompt
    against the model it will actually run on, by display name or model_id.
    """
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        large = AgentSpec(
            agent_id="large",
            role="judge",
            objective="Test objective for validation",
            model_name="large-context-test-model",
        )
        assert large.get_prompt_length_limit() == 200000

        # Lookup by display name works too, matching resolve_model semantics.
        large_by_name = AgentSpec(
            agent_id="large2",
            role="judge",
            objective="Test objective for validation",
            model_name="Large Context Test Model",
        )
        assert large_by_name.get_prompt_length_limit() == 200000

        small = AgentSpec(
            agent_id="small",
            role="judge",
            objective="Test objective for validation",
            model_name="small-context-test-model",
        )
        assert small.get_prompt_length_limit() == 1000


def test_system_prompt_over_10k_validates_for_large_context_model():
    """A >10k-character prompt validates fine for a large-context model.

    This is the concrete case the old hardcoded max_length=10000 got wrong:
    a prompt well over 10k characters is entirely reasonable for a model
    with a 200k-token input budget.
    """
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        agent = AgentSpec(
            agent_id="test",
            role="judge",
            objective="Test objective for validation",
            model_name="large-context-test-model",
            system_prompt="x" * 15000,
        )
        assert len(agent.system_prompt) == 15000


def test_system_prompt_rejected_for_small_context_model():
    """An excessive prompt is rejected against a small-context model's own limit.

    Confirms the per-model check has real teeth (not a permissive no-op that
    always passes): a prompt that vastly exceeds the *bound* model's declared
    budget is still caught, just against that model's real limit rather than
    an arbitrary fixed number.
    """
    with patch("bili.iris.config.llm_config.LLM_MODELS", _MOCK_CATALOG):
        with pytest.raises(ValidationError, match="system_prompt"):
            AgentSpec(
                agent_id="test",
                role="judge",
                objective="Test objective for validation",
                model_name="small-context-test-model",
                # small-context-test-model declares max_input_tokens=1000,
                # i.e. an approximate 4000-character budget.
                system_prompt="x" * 20000,
            )
