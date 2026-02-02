"""
Tests for MASConfig schema.

These tests verify the multi-agent system configuration and
its domain-agnostic design.
"""

import pytest
from pydantic import ValidationError

from bili.aether.schema.agent_presets import get_preset
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.enums import CommunicationProtocol, WorkflowType
from bili.aether.schema.mas_config import (
    Channel,
    MASConfig,
    WorkflowEdge,
    create_consensus_mas,
    create_sequential_mas,
    create_supervisor_mas,
)


class TestMASConfigBasic:
    """Basic MASConfig tests."""

    def test_create_simple_mas(self):
        """Test creating a simple MAS with minimal configuration."""
        agent = AgentSpec(
            agent_id="test_agent",
            role="test_role",
            objective="Test agent for MAS creation",
        )

        mas = MASConfig(
            mas_id="test_mas",
            name="Test MAS",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )

        assert mas.mas_id == "test_mas"
        assert mas.name == "Test MAS"
        assert len(mas.agents) == 1
        assert mas.workflow_type == WorkflowType.SEQUENTIAL

    def test_create_mas_with_multiple_agents(self):
        """Test creating MAS with multiple agents."""
        agents = [
            AgentSpec(
                agent_id=f"agent_{i}",
                role=f"role_{i}",
                objective=f"Test agent {i} objective here",
            )
            for i in range(3)
        ]

        mas = MASConfig(
            mas_id="multi_agent_mas",
            name="Multi-Agent MAS",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=agents,
        )

        assert len(mas.agents) == 3


class TestMASConfigValidation:
    """Validation tests for MASConfig."""

    def test_reject_empty_agents(self):
        """Test that MAS must have at least one agent."""
        with pytest.raises(ValidationError):
            MASConfig(
                mas_id="empty_mas",
                name="Empty MAS",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[],
            )

    def test_reject_duplicate_agent_ids(self):
        """Test that duplicate agent IDs are rejected."""
        agents = [
            AgentSpec(
                agent_id="duplicate_id",
                role="role1",
                objective="First agent with duplicate ID",
            ),
            AgentSpec(
                agent_id="duplicate_id",
                role="role2",
                objective="Second agent with duplicate ID",
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            MASConfig(
                mas_id="duplicate_mas",
                name="Duplicate MAS",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=agents,
            )
        assert "Duplicate agent IDs" in str(exc_info.value)

    def test_validate_entry_point(self):
        """Test that entry_point must exist in agents."""
        agent = AgentSpec(
            agent_id="real_agent",
            role="test",
            objective="Real agent for testing",
        )

        with pytest.raises(ValidationError) as exc_info:
            MASConfig(
                mas_id="test_mas",
                name="Test MAS",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[agent],
                entry_point="nonexistent_agent",
            )
        assert "entry_point" in str(exc_info.value)


class TestChannelValidation:
    """Tests for channel configuration."""

    def test_create_valid_channel(self):
        """Test creating a valid channel."""
        agents = [
            AgentSpec(
                agent_id="sender", role="sender", objective="Send messages to receiver"
            ),
            AgentSpec(
                agent_id="receiver",
                role="receiver",
                objective="Receive messages from sender",
            ),
        ]

        channel = Channel(
            channel_id="test_channel",
            protocol=CommunicationProtocol.DIRECT,
            source="sender",
            target="receiver",
        )

        mas = MASConfig(
            mas_id="channel_mas",
            name="Channel MAS",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=agents,
            channels=[channel],
        )

        assert len(mas.channels) == 1

    def test_validate_channel_source(self):
        """Test that channel source must exist in agents."""
        agent = AgentSpec(
            agent_id="agent", role="test", objective="Agent for channel test"
        )

        channel = Channel(
            channel_id="bad_channel",
            protocol=CommunicationProtocol.DIRECT,
            source="nonexistent",
            target="agent",
        )

        with pytest.raises(ValidationError) as exc_info:
            MASConfig(
                mas_id="test_mas",
                name="Test MAS",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[agent],
                channels=[channel],
            )
        assert "source" in str(exc_info.value)

    def test_channel_special_targets(self):
        """Test that 'any' source and 'all' target are valid."""
        agents = [
            AgentSpec(agent_id="a1", role="test", objective="Agent 1 objective"),
            AgentSpec(agent_id="a2", role="test", objective="Agent 2 objective"),
        ]

        channel = Channel(
            channel_id="broadcast",
            protocol=CommunicationProtocol.BROADCAST,
            source="a1",
            target="all",
        )

        mas = MASConfig(
            mas_id="broadcast_mas",
            name="Broadcast MAS",
            workflow_type=WorkflowType.PARALLEL,
            agents=agents,
            channels=[channel],
        )

        assert mas.channels[0].target == "all"


class TestWorkflowEdges:
    """Tests for custom workflow edges."""

    def test_create_workflow_edge(self):
        """Test creating workflow edges."""
        agents = [
            AgentSpec(agent_id="a", role="test", objective="Agent A objective"),
            AgentSpec(agent_id="b", role="test", objective="Agent B objective"),
        ]

        edge = WorkflowEdge(from_agent="a", to_agent="b")

        mas = MASConfig(
            mas_id="edge_mas",
            name="Edge MAS",
            workflow_type=WorkflowType.CUSTOM,
            agents=agents,
            workflow_edges=[edge],
        )

        assert len(mas.workflow_edges) == 1

    def test_conditional_edge(self):
        """Test creating conditional workflow edge."""
        edge = WorkflowEdge(
            from_agent="a",
            to_agent="b",
            condition="state.score > 0.5",
            label="high score",
        )

        assert edge.condition == "state.score > 0.5"

    def test_edge_to_end(self):
        """Test edge to END (terminal)."""
        agents = [
            AgentSpec(agent_id="final", role="test", objective="Final agent objective")
        ]

        edge = WorkflowEdge(from_agent="final", to_agent="END")

        mas = MASConfig(
            mas_id="end_mas",
            name="End MAS",
            workflow_type=WorkflowType.CUSTOM,
            agents=agents,
            workflow_edges=[edge],
        )

        assert mas.workflow_edges[0].to_agent == "END"


class TestConsensusWorkflow:
    """Tests for consensus workflow configuration."""

    def test_consensus_requires_threshold(self):
        """Test that consensus workflow requires threshold."""
        agents = [
            AgentSpec(agent_id="v1", role="voter", objective="Voter 1 objective"),
            AgentSpec(agent_id="v2", role="voter", objective="Voter 2 objective"),
        ]

        with pytest.raises(ValidationError) as exc_info:
            MASConfig(
                mas_id="consensus_mas",
                name="Consensus MAS",
                workflow_type=WorkflowType.CONSENSUS,
                agents=agents,
                # Missing consensus_threshold
            )
        assert "consensus_threshold" in str(exc_info.value)

    def test_valid_consensus_config(self):
        """Test valid consensus configuration."""
        agents = [
            AgentSpec(
                agent_id="v1",
                role="voter",
                objective="Voter 1 objective",
                voting_weight=1.0,
            ),
            AgentSpec(
                agent_id="v2",
                role="voter",
                objective="Voter 2 objective",
                voting_weight=1.0,
            ),
        ]

        mas = MASConfig(
            mas_id="consensus_mas",
            name="Consensus MAS",
            workflow_type=WorkflowType.CONSENSUS,
            agents=agents,
            consensus_threshold=0.66,
            max_consensus_rounds=5,
        )

        assert mas.consensus_threshold == 0.66
        assert mas.max_consensus_rounds == 5


class TestHierarchicalWorkflow:
    """Tests for hierarchical workflow configuration."""

    def test_hierarchical_requires_tiers(self):
        """Test that hierarchical workflow requires tier values."""
        agents = [
            AgentSpec(
                agent_id="a1",
                role="test",
                objective="Agent without tier value",
                # Missing tier
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            MASConfig(
                mas_id="hier_mas",
                name="Hierarchical MAS",
                workflow_type=WorkflowType.HIERARCHICAL,
                agents=agents,
            )
        assert "tier" in str(exc_info.value)

    def test_valid_hierarchical_config(self):
        """Test valid hierarchical configuration."""
        agents = [
            AgentSpec(
                agent_id="tier1",
                role="senior",
                objective="Senior tier agent objective",
                tier=1,
            ),
            AgentSpec(
                agent_id="tier2",
                role="junior",
                objective="Junior tier agent objective",
                tier=2,
            ),
        ]

        mas = MASConfig(
            mas_id="hier_mas",
            name="Hierarchical MAS",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=agents,
        )

        tier1_agents = mas.get_agents_by_tier(1)
        assert len(tier1_agents) == 1
        assert tier1_agents[0].agent_id == "tier1"


class TestFactoryFunctions:
    """Tests for MAS factory functions."""

    def test_create_sequential_mas(self):
        """Test sequential MAS factory."""
        agents = [
            AgentSpec(agent_id="a", role="test", objective="Agent A objective"),
            AgentSpec(agent_id="b", role="test", objective="Agent B objective"),
        ]

        mas = create_sequential_mas(
            mas_id="seq",
            name="Sequential",
            agents=agents,
        )

        assert mas.workflow_type == WorkflowType.SEQUENTIAL

    def test_create_supervisor_mas(self):
        """Test supervisor MAS factory."""
        supervisor = AgentSpec(
            agent_id="sup",
            role="supervisor",
            objective="Supervise specialist agents",
        )
        specialists = [
            AgentSpec(agent_id="s1", role="specialist", objective="Specialist 1"),
            AgentSpec(agent_id="s2", role="specialist", objective="Specialist 2"),
        ]

        mas = create_supervisor_mas(
            mas_id="sup_mas",
            name="Supervisor MAS",
            supervisor=supervisor,
            specialists=specialists,
        )

        assert mas.workflow_type == WorkflowType.SUPERVISOR
        assert mas.entry_point == "sup"
        assert mas.agents[0].is_supervisor is True

    def test_create_consensus_mas(self):
        """Test consensus MAS factory."""
        agents = [
            AgentSpec(
                agent_id="v1", role="voter", objective="Voter 1", voting_weight=1.0
            ),
            AgentSpec(
                agent_id="v2", role="voter", objective="Voter 2", voting_weight=1.0
            ),
        ]

        mas = create_consensus_mas(
            mas_id="cons",
            name="Consensus",
            agents=agents,
            consensus_threshold=0.75,
        )

        assert mas.workflow_type == WorkflowType.CONSENSUS
        assert mas.consensus_threshold == 0.75


class TestMASHelperMethods:
    """Tests for MAS helper methods."""

    def test_get_agent(self):
        """Test getting agent by ID."""
        agents = [
            AgentSpec(agent_id="target", role="test", objective="Target agent"),
            AgentSpec(agent_id="other", role="test", objective="Other agent"),
        ]

        mas = MASConfig(
            mas_id="test",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=agents,
        )

        found = mas.get_agent("target")
        assert found is not None
        assert found.agent_id == "target"

        not_found = mas.get_agent("nonexistent")
        assert not_found is None

    def test_get_entry_agent(self):
        """Test getting entry agent."""
        agents = [
            AgentSpec(agent_id="first", role="test", objective="First agent"),
            AgentSpec(agent_id="second", role="test", objective="Second agent"),
        ]

        mas = MASConfig(
            mas_id="test",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=agents,
        )

        # Default entry is first agent
        entry = mas.get_entry_agent()
        assert entry.agent_id == "first"

    def test_get_agents_by_role(self):
        """Test getting agents by role."""
        agents = [
            AgentSpec(agent_id="r1", role="reviewer", objective="Reviewer 1"),
            AgentSpec(agent_id="r2", role="reviewer", objective="Reviewer 2"),
            AgentSpec(agent_id="j1", role="judge", objective="Judge 1"),
        ]

        mas = MASConfig(
            mas_id="test",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=agents,
        )

        reviewers = mas.get_agents_by_role("reviewer")
        assert len(reviewers) == 2

    def test_get_supervisors(self):
        """Test getting supervisor agents."""
        agents = [
            AgentSpec(
                agent_id="sup",
                role="supervisor",
                objective="Supervisor",
                is_supervisor=True,
            ),
            AgentSpec(
                agent_id="worker", role="worker", objective="Worker", is_supervisor=False
            ),
        ]

        mas = MASConfig(
            mas_id="test",
            name="Test",
            workflow_type=WorkflowType.SUPERVISOR,
            agents=agents,
        )

        supervisors = mas.get_supervisors()
        assert len(supervisors) == 1
        assert supervisors[0].agent_id == "sup"


class TestMASWithPresets:
    """Tests demonstrating MAS creation with presets."""

    def test_mas_with_preset_agents(self):
        """Test creating MAS using preset agents."""
        researcher = get_preset("researcher", agent_id="researcher_agent")
        analyst = get_preset("analyst", agent_id="analyst_agent")

        mas = create_sequential_mas(
            mas_id="preset_mas",
            name="Preset-based MAS",
            agents=[researcher, analyst],
            description="MAS built with preset agents",
        )

        assert len(mas.agents) == 2
        assert mas.agents[0].role == "researcher"
        assert mas.agents[1].role == "analyst"
