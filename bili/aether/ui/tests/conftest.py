"""Shared fixtures for AETHER UI Streamlit tests."""

import pytest

from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.enums import WorkflowType
from bili.aether.schema.mas_config import Channel, MASConfig


def make_test_config(
    mas_id: str = "test_mas",
    name: str = "Test MAS",
    description: str = "A test config",
    workflow_type: WorkflowType = WorkflowType.SEQUENTIAL,
    num_agents: int = 2,
    model_name: str = None,
) -> MASConfig:
    """Build a minimal MASConfig for testing.

    Args:
        mas_id: Unique config identifier.
        name: Human-readable name.
        description: Config description.
        workflow_type: Workflow pattern.
        num_agents: Number of agents to create.
        model_name: Optional model name for agents.

    Returns:
        A valid MASConfig instance.
    """
    agents = [
        AgentSpec(
            agent_id=f"agent_{i}",
            role=f"role_{i}",
            objective=f"Perform analysis task number {i}",
            model_name=model_name,
        )
        for i in range(num_agents)
    ]
    channels = []
    if num_agents >= 2:
        channels.append(
            Channel(
                channel_id="ch_0_1",
                protocol="direct",
                source="agent_0",
                target="agent_1",
            )
        )
    return MASConfig(
        mas_id=mas_id,
        name=name,
        description=description,
        agents=agents,
        channels=channels,
        workflow_type=workflow_type,
    )


@pytest.fixture()
def stub_config() -> MASConfig:
    """Return a stub MASConfig with no model names."""
    return make_test_config()


@pytest.fixture()
def live_config() -> MASConfig:
    """Return a MASConfig with model names set."""
    return make_test_config(model_name="gpt-4o")
