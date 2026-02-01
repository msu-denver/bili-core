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
            objective="Test",
        )


def test_custom_role_with_name():
    """Test custom role with name."""
    agent = AgentSpec(
        agent_id="analyzer",
        role=AgentRole.CUSTOM,
        custom_role_name="Sentiment Analyzer",
        objective="Analyze sentiment",
    )
    assert agent.get_display_name() == "Sentiment Analyzer"


def test_structured_output_requires_schema():
    """Test that structured output requires output_schema."""
    with pytest.raises(ValidationError, match="output_schema"):
        AgentSpec(
            agent_id="test",
            role=AgentRole.JUDGE,
            objective="Test",
            output_format=OutputFormat.STRUCTURED,
            # Missing output_schema!
        )


def test_structured_output_with_schema():
    """Test structured output with schema."""
    agent = AgentSpec(
        agent_id="test",
        role=AgentRole.JUDGE,
        objective="Test",
        output_format=OutputFormat.STRUCTURED,
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.output_schema is not None


def test_hierarchical_agent():
    """Test agent with tier for hierarchical workflows."""
    agent = AgentSpec(
        agent_id="vote_agent",
        role=AgentRole.JUDGE,
        objective="Vote",
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
        agent_id="reviewer", role=AgentRole.CONTENT_REVIEWER, objective="Review"
    )

    custom_agent = AgentSpec(
        agent_id="custom",
        role=AgentRole.CUSTOM,
        custom_role_name="Custom",
        objective="Custom task",
    )

    assert bili_agent.is_bili_core_role() is True
    assert custom_agent.is_bili_core_role() is False


def test_invalid_agent_id():
    """Test that invalid agent IDs are rejected."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="invalid id with spaces",  # no spaces allowed
            role=AgentRole.JUDGE,
            objective="Test",
        )


def test_temperature_bounds():
    """Test temperature must be 0.0-2.0."""
    with pytest.raises(ValidationError):
        AgentSpec(
            agent_id="test",
            role=AgentRole.JUDGE,
            objective="Test",
            temperature=3.0,  # too high
        )
