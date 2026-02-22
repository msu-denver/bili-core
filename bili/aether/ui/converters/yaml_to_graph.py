"""
Convert MASConfig to streamlit-flow nodes and edges with auto-layout.

Each workflow type gets a specialized layout algorithm that positions
agents in a topology matching the workflow pattern.
"""

import math
from typing import Dict, List, Tuple

from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode

from bili.aether.schema.enums import WorkflowType
from bili.aether.schema.mas_config import MASConfig
from bili.aether.ui.styles.bili_core_theme import (
    EDGE_CONDITIONAL_COLOR,
    EDGE_WORKFLOW_COLOR,
    PROTOCOL_COLORS,
)
from bili.aether.ui.styles.node_styles import build_node_css


def convert_mas_to_graph(
    config: MASConfig,
) -> Tuple[List[StreamlitFlowNode], List[StreamlitFlowEdge]]:
    """Convert a MASConfig to streamlit-flow nodes and edges.

    Dispatches to the appropriate layout function based on workflow_type,
    then builds nodes and edges for rendering.

    Args:
        config: Validated MASConfig instance.

    Returns:
        Tuple of (nodes, edges) ready for StreamlitFlowState.
    """
    positions = _calculate_layout(config)
    source_pos, target_pos = _get_handle_positions(config.workflow_type)
    nodes = _build_nodes(config, positions, source_pos, target_pos)
    edges = _build_edges(config)
    return nodes, edges


# =============================================================================
# LAYOUT DISPATCH
# =============================================================================


def _calculate_layout(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Dispatch to layout calculator based on workflow_type."""
    layout_map = {
        WorkflowType.SEQUENTIAL: _layout_sequential,
        WorkflowType.HIERARCHICAL: _layout_hierarchical,
        WorkflowType.SUPERVISOR: _layout_supervisor,
        WorkflowType.CONSENSUS: _layout_consensus,
        WorkflowType.PARALLEL: _layout_parallel,
        WorkflowType.DELIBERATIVE: _layout_edge_based,
        WorkflowType.CUSTOM: _layout_edge_based,
    }
    layout_fn = layout_map.get(config.workflow_type, _layout_sequential)
    return layout_fn(config)


def _get_handle_positions(
    workflow_type: WorkflowType,
) -> Tuple[str, str]:
    """Return (source_position, target_position) for a workflow type."""
    vertical_types = {
        WorkflowType.HIERARCHICAL,
        WorkflowType.PARALLEL,
    }
    if workflow_type in vertical_types:
        return "bottom", "top"
    return "right", "left"


# =============================================================================
# LAYOUT ALGORITHMS
# =============================================================================


def _layout_sequential(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Horizontal chain: A -> B -> C, left to right."""
    spacing = 250
    y = 200
    start_x = 50
    return {a.agent_id: (start_x + i * spacing, y) for i, a in enumerate(config.agents)}


def _layout_hierarchical(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Multi-tier vertical layout. Tier 1 at top, higher tier numbers below."""
    positions: Dict[str, Tuple[float, float]] = {}
    tiers: Dict[int, list] = {}
    for agent in config.agents:
        t = agent.tier if agent.tier is not None else 1
        tiers.setdefault(t, []).append(agent)

    sorted_tiers = sorted(tiers.keys())
    tier_y_spacing = 200
    canvas_width = 800

    for tier_idx, tier_num in enumerate(sorted_tiers):
        tier_agents = tiers[tier_num]
        n = len(tier_agents)
        row_width = (n - 1) * 200
        start_x = (canvas_width - row_width) / 2
        y = 50 + tier_idx * tier_y_spacing
        for i, agent in enumerate(tier_agents):
            positions[agent.agent_id] = (start_x + i * 200, y)

    return positions


def _layout_supervisor(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Hub-and-spoke. Supervisor at center, workers in a circle."""
    positions: Dict[str, Tuple[float, float]] = {}
    supervisor = None
    workers = []

    for agent in config.agents:
        if agent.is_supervisor or agent.agent_id == config.entry_point:
            if supervisor is None:
                supervisor = agent
            else:
                workers.append(agent)
        else:
            workers.append(agent)

    if supervisor is None:
        supervisor = config.agents[0]
        workers = list(config.agents[1:])

    center = (400, 300)
    positions[supervisor.agent_id] = center

    radius = 220
    n = len(workers)
    if n > 0:
        for i, worker in enumerate(workers):
            angle = (2 * math.pi * i / n) - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions[worker.agent_id] = (x, y)

    return positions


def _layout_consensus(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Circular arrangement. All agents evenly spaced in a ring."""
    positions: Dict[str, Tuple[float, float]] = {}
    center = (400, 300)
    radius = 200
    n = len(config.agents)
    for i, agent in enumerate(config.agents):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        positions[agent.agent_id] = (x, y)
    return positions


def _layout_parallel(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Three rows: coordinator top, workers middle, aggregator bottom."""
    positions: Dict[str, Tuple[float, float]] = {}
    agent_ids = [a.agent_id for a in config.agents]

    coordinator_id = None
    aggregator_id = None
    worker_ids = []

    for agent in config.agents:
        role_lower = agent.role.lower()
        if "coordinator" in role_lower and coordinator_id is None:
            coordinator_id = agent.agent_id
        elif "synthesizer" in role_lower or "aggregator" in role_lower:
            if aggregator_id is None:
                aggregator_id = agent.agent_id
            else:
                worker_ids.append(agent.agent_id)
        else:
            worker_ids.append(agent.agent_id)

    if coordinator_id is None:
        coordinator_id = agent_ids[0]
        worker_ids = [w for w in worker_ids if w != coordinator_id]
    if aggregator_id is None:
        aggregator_id = agent_ids[-1]
        worker_ids = [w for w in worker_ids if w != aggregator_id]

    # Remove coordinator/aggregator from workers if present
    worker_ids = [w for w in worker_ids if w != coordinator_id and w != aggregator_id]
    if not worker_ids:
        worker_ids = [
            a for a in agent_ids if a != coordinator_id and a != aggregator_id
        ]

    canvas_width = 800
    positions[coordinator_id] = (canvas_width / 2, 50)

    n = len(worker_ids)
    if n > 0:
        row_width = (n - 1) * 200
        start_x = (canvas_width - row_width) / 2
        for i, wid in enumerate(worker_ids):
            positions[wid] = (start_x + i * 200, 275)

    positions[aggregator_id] = (canvas_width / 2, 500)
    return positions


def _layout_edge_based(config: MASConfig) -> Dict[str, Tuple[float, float]]:
    """Layered layout based on workflow_edges (for DELIBERATIVE and CUSTOM)."""
    if not config.workflow_edges:
        return _layout_sequential(config)

    agent_ids = [a.agent_id for a in config.agents]

    # Build adjacency and in-degree
    adj: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
    in_degree: Dict[str, int] = {aid: 0 for aid in agent_ids}

    for edge in config.workflow_edges:
        if edge.from_agent in adj and edge.to_agent in agent_ids:
            adj[edge.from_agent].append(edge.to_agent)
            in_degree[edge.to_agent] += 1

    # BFS to assign max depth from roots
    depth: Dict[str, int] = {aid: 0 for aid in agent_ids}
    roots = [aid for aid in agent_ids if in_degree[aid] == 0]
    if not roots:
        roots = [agent_ids[0]]

    queue = list(roots)
    visited = set()
    queued = set(roots)

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adj.get(current, []):
            depth[neighbor] = max(depth[neighbor], depth[current] + 1)
            if neighbor not in visited and neighbor not in queued:
                queued.add(neighbor)
                queue.append(neighbor)

    # Place any unvisited agents at depth 0
    for aid in agent_ids:
        if aid not in visited:
            depth[aid] = 0

    # Group by depth and position
    layers: Dict[int, List[str]] = {}
    for aid, d in depth.items():
        layers.setdefault(d, []).append(aid)

    positions: Dict[str, Tuple[float, float]] = {}
    x_spacing = 250
    y_spacing = 150
    y_center = 250

    for d in sorted(layers.keys()):
        layer_agents = layers[d]
        n = len(layer_agents)
        total_height = (n - 1) * y_spacing
        start_y = y_center - total_height / 2
        x = 50 + d * x_spacing
        for i, aid in enumerate(layer_agents):
            positions[aid] = (x, start_y + i * y_spacing)

    return positions


# =============================================================================
# NODE BUILDING
# =============================================================================


def _build_nodes(
    config: MASConfig,
    positions: Dict[str, Tuple[float, float]],
    source_position: str,
    target_position: str,
) -> List[StreamlitFlowNode]:
    """Build StreamlitFlowNode objects from agents and positions."""
    nodes = []
    for agent in config.agents:
        pos = positions.get(agent.agent_id, (0, 0))
        label = agent.get_display_name()

        node = StreamlitFlowNode(
            id=agent.agent_id,
            pos=pos,
            data={"content": label},
            node_type="default",
            source_position=source_position,
            target_position=target_position,
            draggable=False,
            connectable=False,
            deletable=False,
            style=build_node_css(agent.role),
        )
        nodes.append(node)
    return nodes


# =============================================================================
# EDGE BUILDING
# =============================================================================


def _build_edges(config: MASConfig) -> List[StreamlitFlowEdge]:
    """Build edges from both channels and workflow_edges."""
    edges: List[StreamlitFlowEdge] = []
    agent_ids = {a.agent_id for a in config.agents}

    # Channel edges: solid lines, colored by protocol
    for channel in config.channels:
        source = channel.source
        target = channel.target

        if target == "all":
            # Broadcast: expand to edges to each other agent
            for aid in agent_ids:
                if aid != source:
                    edges.append(
                        _make_channel_edge(
                            f"ch_{channel.channel_id}_{aid}",
                            source,
                            aid,
                            channel,
                        )
                    )
        else:
            edges.append(
                _make_channel_edge(
                    f"ch_{channel.channel_id}",
                    source,
                    target,
                    channel,
                )
            )

            # Add reverse edge for bidirectional channels
            if channel.bidirectional:
                edges.append(
                    _make_channel_edge(
                        f"ch_{channel.channel_id}_rev",
                        target,
                        source,
                        channel,
                        is_reverse=True,
                    )
                )

    # Workflow edges: dashed lines
    for we in config.workflow_edges:
        if we.to_agent == "END":
            continue

        has_condition = bool(we.condition)
        stroke_color = EDGE_CONDITIONAL_COLOR if has_condition else EDGE_WORKFLOW_COLOR

        label = we.label or ""
        if has_condition and not we.label:
            label = (
                we.condition[:30] + "..." if len(we.condition) > 30 else we.condition
            )

        edges.append(
            StreamlitFlowEdge(
                id=f"we_{we.from_agent}_{we.to_agent}",
                source=we.from_agent,
                target=we.to_agent,
                edge_type="smoothstep",
                label=label,
                animated=has_condition,
                style={
                    "stroke": stroke_color,
                    "strokeDasharray": "5,5",
                    "strokeWidth": 2,
                },
                marker_end={"type": "arrowclosed"},
            )
        )

    return edges


def _make_channel_edge(
    edge_id: str,
    source: str,
    target: str,
    channel,
    is_reverse: bool = False,
) -> StreamlitFlowEdge:
    """Create a StreamlitFlowEdge for a communication channel."""
    protocol_value = channel.protocol.value
    color = PROTOCOL_COLORS.get(protocol_value, PROTOCOL_COLORS["direct"])

    return StreamlitFlowEdge(
        id=edge_id,
        source=source,
        target=target,
        edge_type="smoothstep",
        label="" if is_reverse else protocol_value,
        animated=False,
        style={"stroke": color, "strokeWidth": 2},
        marker_end={"type": "arrowclosed"},
    )
