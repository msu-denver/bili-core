"""Shared pytest fixtures and helpers for the AETHER test suite."""

import pytest

from bili.aether.schema import AgentSpec


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    """Build an ``AgentSpec`` with sensible defaults for testing."""
    defaults = {"role": "test_role", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


@pytest.fixture
def make_agent():
    """Factory fixture: returns a callable that builds ``AgentSpec`` instances.

    Usage::

        def test_something(make_agent):
            agent = make_agent("my_agent", role="reviewer")
    """
    return _agent
