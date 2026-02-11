"""Tests for AgentSpec schema."""

import pytest
from pydantic import ValidationError

from bili.aether.schema import AgentRole, AgentSpec, OutputFormat


def test_minimal_agent():
    """Test creating agent with minimal fields."""
    agent = AgentSpec(
        agent_id="test_agent",
        role=AgentRole.CONTENT_REVIEWER,
        objective="Test objective",
    )

    assert agent.agent_id == "test_agent"
    assert agent.role == AgentRole.CONTENT_REVIEWER
    assert agent.temperature == 0.0  # Default


def test_custom_role_requires_name():
    """Test that custom role requires custom_role_name."""
    with pytest.raises(ValidationError, match="custom_role_name"):
        AgentSpec(
            agent_id="custom_agent",
            role=AgentRole.CUSTOM,  # Requires custom_role_name!
            objective="Test objective for validation",
        )


def test_custom_role_with_name():
    """Test custom role with name."""
    agent = AgentSpec(
        agent_id="analyzer",
        role=AgentRole.CUSTOM,
        custom_role_name="Sentiment Analyzer",
        objective="Analyze sentiment of flagged content",
    )
    assert agent.get_display_name() == "Sentiment Analyzer"


def test_structured_output_requires_schema():
    """Test that structured output requires output_schema."""
    with pytest.raises(ValidationError, match="output_schema"):
        AgentSpec(
            agent_id="test",
            role=AgentRole.JUDGE,
            objective="Test objective for validation",
            output_format=OutputFormat.STRUCTURED,
            # Missing output_schema!
        )


def test_structured_output_with_schema():
    """Test structured output with schema."""
    agent = AgentSpec(
        agent_id="test",
        role=AgentRole.JUDGE,
        objective="Test objective for validation",
        output_format=OutputFormat.STRUCTURED,
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.output_schema is not None


def test_hierarchical_agent():
    """Test agent with tier for hierarchical workflows."""
    agent = AgentSpec(
        agent_id="vote_agent",
        role=AgentRole.JUDGE,
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
        role=AgentRole.JUDGE,
        objective="Coordinate specialists",
        is_supervisor=True,
    )
    assert agent.is_supervisor is True


def test_bili_core_role_detection():
    """Test bili-core role detection."""
    bili_agent = AgentSpec(
        agent_id="reviewer",
        role=AgentRole.CONTENT_REVIEWER,
        objective="Review flagged content",
    )

    custom_agent = AgentSpec(
        agent_id="custom",
        role=AgentRole.CUSTOM,
        custom_role_name="Custom",
        objective="Custom task for testing",
    )

    assert bili_agent.is_bili_core_role() is True
    assert custom_agent.is_bili_core_role() is False


def test_invalid_agent_id():
    """Test that invalid agent IDs are rejected."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="invalid id with spaces",  # no spaces allowed
            role=AgentRole.JUDGE,
            objective="Test objective for validation",
        )


def test_temperature_bounds():
    """Test temperature must be 0.0-2.0."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="test",
            role=AgentRole.JUDGE,
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
        role=AgentRole.JUDGE,
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
        role=AgentRole.JUDGE,
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
        role=AgentRole.JUDGE,
        objective="Test inheritance disabled",
        inherit_from_bili_core=False,
        inherit_tools=False,  # Should be reset to True
    )
    assert agent.inherit_from_bili_core is False
    # Sub-flags normalised to defaults when master toggle is off
    assert agent.inherit_tools is True
