"""Tests for yaml_to_graph converter.

Tests the pure conversion logic that turns MASConfig objects
into streamlit-flow nodes and edges. All streamlit-flow types
are imported from the real library.
"""

from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode

from bili.aether.schema import AgentSpec
from bili.aether.schema.enums import CommunicationProtocol, WorkflowType
from bili.aether.schema.mas_config import Channel, MASConfig, WorkflowEdge
from bili.aether.ui.converters.yaml_to_graph import (
    _build_edges,
    _build_nodes,
    _get_handle_positions,
    _layout_consensus,
    _layout_edge_based,
    _layout_hierarchical,
    _layout_parallel,
    _layout_sequential,
    _layout_supervisor,
    convert_mas_to_graph,
)

_OBJ = "Objective for this test agent in the system."


def _simple_config(
    workflow_type=WorkflowType.SEQUENTIAL,
    num_agents=3,
    channels=None,
    workflow_edges=None,
    extra_kwargs=None,
):
    """Build a minimal MASConfig for testing."""
    agents = [
        AgentSpec(
            agent_id=f"agent_{i}",
            role=f"role_{i}",
            objective=_OBJ,
        )
        for i in range(num_agents)
    ]
    kwargs = {
        "mas_id": "test_mas",
        "name": "Test MAS",
        "workflow_type": workflow_type,
        "agents": agents,
        "channels": channels or [],
        "workflow_edges": workflow_edges or [],
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return MASConfig(**kwargs)


class TestConvertMasToGraph:
    """Tests for convert_mas_to_graph main entry point."""

    def test_returns_nodes_and_edges(self):
        """Returns a tuple of (nodes, edges)."""
        config = _simple_config()
        nodes, edges = convert_mas_to_graph(config)
        assert isinstance(nodes, list)
        assert isinstance(edges, list)

    def test_node_count_matches_agents(self):
        """One node per agent."""
        config = _simple_config(num_agents=4)
        nodes, _ = convert_mas_to_graph(config)
        assert len(nodes) == 4

    def test_nodes_are_streamlit_flow_nodes(self):
        """All nodes are StreamlitFlowNode instances."""
        config = _simple_config()
        nodes, _ = convert_mas_to_graph(config)
        for node in nodes:
            assert isinstance(node, StreamlitFlowNode)

    def test_sequential_synthesizes_edges(self):
        """Sequential workflow with no channels gets edges."""
        config = _simple_config(
            workflow_type=WorkflowType.SEQUENTIAL,
            num_agents=3,
        )
        _, edges = convert_mas_to_graph(config)
        assert len(edges) == 2
        for edge in edges:
            assert isinstance(edge, StreamlitFlowEdge)


class TestGetHandlePositions:
    """Tests for _get_handle_positions."""

    def test_sequential_is_horizontal(self):
        """Sequential uses right/left handles."""
        src, tgt = _get_handle_positions(WorkflowType.SEQUENTIAL)
        assert src == "right"
        assert tgt == "left"

    def test_hierarchical_is_vertical(self):
        """Hierarchical uses bottom/top handles."""
        src, tgt = _get_handle_positions(WorkflowType.HIERARCHICAL)
        assert src == "bottom"
        assert tgt == "top"

    def test_parallel_is_vertical(self):
        """Parallel uses bottom/top handles."""
        src, tgt = _get_handle_positions(WorkflowType.PARALLEL)
        assert src == "bottom"
        assert tgt == "top"

    def test_supervisor_is_horizontal(self):
        """Supervisor uses right/left handles."""
        src, tgt = _get_handle_positions(WorkflowType.SUPERVISOR)
        assert src == "right"
        assert tgt == "left"


class TestLayoutSequential:
    """Tests for _layout_sequential."""

    def test_positions_agents_horizontally(self):
        """Agents are placed left to right with equal spacing."""
        config = _simple_config(num_agents=3)
        positions = _layout_sequential(config)
        assert len(positions) == 3
        xs = [positions[f"agent_{i}"][0] for i in range(3)]
        assert xs[0] < xs[1] < xs[2]

    def test_same_y_for_all(self):
        """All agents share the same y coordinate."""
        config = _simple_config(num_agents=3)
        positions = _layout_sequential(config)
        ys = [positions[f"agent_{i}"][1] for i in range(3)]
        assert ys[0] == ys[1] == ys[2]


class TestLayoutWorkflowTypes:
    """Tests for _layout_hierarchical, _layout_supervisor, and _layout_consensus."""

    def test_hierarchical_tiers_have_different_y(self):
        """Agents in different tiers have different y values."""
        agents = [
            AgentSpec(
                agent_id="top",
                role="lead",
                objective=_OBJ,
                tier=1,
            ),
            AgentSpec(
                agent_id="bottom",
                role="worker",
                objective=_OBJ,
                tier=2,
            ),
        ]
        config = MASConfig(
            mas_id="hier",
            name="Hierarchical",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=agents,
        )
        positions = _layout_hierarchical(config)
        assert positions["top"][1] < positions["bottom"][1]

    def test_supervisor_at_center(self):
        """Supervisor is placed at center position."""
        agents = [
            AgentSpec(
                agent_id="sup",
                role="supervisor",
                objective=_OBJ,
                is_supervisor=True,
            ),
            AgentSpec(
                agent_id="w1",
                role="worker",
                objective=_OBJ,
            ),
            AgentSpec(
                agent_id="w2",
                role="worker2",
                objective=_OBJ,
            ),
        ]
        config = MASConfig(
            mas_id="sup",
            name="Supervisor",
            workflow_type=WorkflowType.SUPERVISOR,
            agents=agents,
        )
        positions = _layout_supervisor(config)
        assert positions["sup"] == (400, 300)

    def test_consensus_agents_in_circle(self):
        """All agents get unique positions."""
        config = _simple_config(
            workflow_type=WorkflowType.CONSENSUS,
            num_agents=4,
            extra_kwargs={"consensus_threshold": 0.5},
        )
        positions = _layout_consensus(config)
        assert len(positions) == 4
        pos_set = set(positions.values())
        assert len(pos_set) == 4

    def test_parallel_three_rows(self):
        """Coordinator top, workers middle, aggregator bottom."""
        agents = [
            AgentSpec(
                agent_id="coord",
                role="coordinator",
                objective=_OBJ,
            ),
            AgentSpec(
                agent_id="w1",
                role="analyst",
                objective=_OBJ,
            ),
            AgentSpec(
                agent_id="agg",
                role="synthesizer",
                objective=_OBJ,
            ),
        ]
        config = MASConfig(
            mas_id="par",
            name="Parallel",
            workflow_type=WorkflowType.PARALLEL,
            agents=agents,
        )
        positions = _layout_parallel(config)
        assert positions["coord"][1] < positions["w1"][1]
        assert positions["w1"][1] < positions["agg"][1]


class TestLayoutEdgeBased:
    """Tests for _layout_edge_based (DELIBERATIVE/CUSTOM)."""

    def test_falls_back_to_sequential(self):
        """Falls back to sequential when no edges defined."""
        config = _simple_config(
            workflow_type=WorkflowType.CUSTOM,
            num_agents=2,
        )
        positions = _layout_edge_based(config)
        assert len(positions) == 2

    def test_layered_layout_with_edges(self):
        """Agents at different depths get different x positions."""
        agents = [
            AgentSpec(
                agent_id="a",
                role="r1",
                objective=_OBJ,
            ),
            AgentSpec(
                agent_id="b",
                role="r2",
                objective=_OBJ,
            ),
            AgentSpec(
                agent_id="c",
                role="r3",
                objective=_OBJ,
            ),
        ]
        edges = [
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="c"),
        ]
        config = MASConfig(
            mas_id="custom",
            name="Custom",
            workflow_type=WorkflowType.CUSTOM,
            agents=agents,
            workflow_edges=edges,
        )
        positions = _layout_edge_based(config)
        assert positions["a"][0] < positions["b"][0]
        assert positions["b"][0] < positions["c"][0]


class TestBuildEdges:
    """Tests for _build_edges."""

    def test_channel_edges_created(self):
        """Channel produces an edge."""
        channels = [
            Channel(
                channel_id="ch1",
                protocol=CommunicationProtocol.DIRECT,
                source="agent_0",
                target="agent_1",
            ),
        ]
        config = _simple_config(channels=channels)
        edges = _build_edges(config)
        assert any(e.source == "agent_0" for e in edges)

    def test_broadcast_expands_to_all(self):
        """Broadcast channel creates edges to all other agents."""
        channels = [
            Channel(
                channel_id="bcast",
                protocol=CommunicationProtocol.BROADCAST,
                source="agent_0",
                target="all",
            ),
        ]
        config = _simple_config(num_agents=3, channels=channels)
        edges = _build_edges(config)
        bcast = [e for e in edges if "bcast" in e.id]
        assert len(bcast) == 2

    def test_bidirectional_creates_reverse(self):
        """Bidirectional channel creates reverse edge."""
        channels = [
            Channel(
                channel_id="bidir",
                protocol=CommunicationProtocol.DIRECT,
                source="agent_0",
                target="agent_1",
                bidirectional=True,
            ),
        ]
        config = _simple_config(channels=channels)
        edges = _build_edges(config)
        rev = [e for e in edges if "_rev" in e.id]
        assert len(rev) == 1
        assert rev[0].source == "agent_1"
        assert rev[0].target == "agent_0"

    def test_workflow_edges_dashed(self):
        """Workflow edges have dashed stroke style."""
        we = [
            WorkflowEdge(from_agent="agent_0", to_agent="agent_1"),
        ]
        config = _simple_config(workflow_edges=we)
        edges = _build_edges(config)
        we_edges = [e for e in edges if e.id.startswith("we_")]
        assert len(we_edges) == 1
        assert "5,5" in we_edges[0].style.get("strokeDasharray", "")

    def test_end_edges_excluded(self):
        """Workflow edges to END are excluded."""
        we = [
            WorkflowEdge(from_agent="agent_0", to_agent="END"),
        ]
        config = _simple_config(workflow_edges=we)
        edges = _build_edges(config)
        we_edges = [e for e in edges if e.id.startswith("we_")]
        assert len(we_edges) == 0

    def test_conditional_edge_animated(self):
        """Conditional workflow edges are animated."""
        we = [
            WorkflowEdge(
                from_agent="agent_0",
                to_agent="agent_1",
                condition="state.score > 0.5",
            ),
        ]
        config = _simple_config(workflow_edges=we)
        edges = _build_edges(config)
        we_edges = [e for e in edges if e.id.startswith("we_")]
        assert we_edges[0].animated is True


class TestBuildNodes:
    """Tests for _build_nodes."""

    def test_node_ids_match_agent_ids(self):
        """Node IDs correspond to agent IDs."""
        config = _simple_config(num_agents=2)
        positions = {
            "agent_0": (0, 0),
            "agent_1": (100, 0),
        }
        nodes = _build_nodes(config, positions, "right", "left")
        ids = {n.id for n in nodes}
        assert ids == {"agent_0", "agent_1"}

    def test_nodes_are_not_draggable(self):
        """Nodes have draggable=False."""
        config = _simple_config(num_agents=1)
        positions = {"agent_0": (0, 0)}
        nodes = _build_nodes(config, positions, "right", "left")
        assert nodes[0].draggable is False
