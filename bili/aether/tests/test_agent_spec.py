"""Tests for AgentSpec schema (domain-agnostic design)."""

import pytest
from pydantic import ValidationError

from bili.aether.schema import AgentSpec, OutputFormat


def test_minimal_agent():
    """Test creating agent with minimal fields."""
    agent = AgentSpec(
        agent_id="test_agent",
        role="content_reviewer",
        objective="Test objective",
    )

    assert agent.agent_id == "test_agent"
    assert agent.role == "content_reviewer"
    assert agent.temperature == 0.0  # Default


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
