"""
Attack graph component — MAS flow graph with click-to-target and post-run overlay.

Wraps ``convert_mas_to_graph()`` and ``streamlit_flow()`` to provide:
- Click-to-target: clicking a node sets it as the attack target (red border).
- Post-run overlay: node borders are color-coded by propagation outcome after
  an attack completes (influenced=red, resisted=green, received=yellow).

The graph state is rebuilt on every render so that style overrides always
reflect the current ``target_agent_id`` and ``node_states`` values.
"""

import streamlit as st

# pylint: disable=import-error
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from bili.aether.schema.mas_config import MASConfig
from bili.aether.ui.converters.yaml_to_graph import convert_mas_to_graph

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_TARGET_BORDER = "3px solid #DC2626"  # Red — targeted node
_INFLUENCED_BORDER = "3px solid #DC2626"  # Red — payload influenced output
_RESISTED_BORDER = "3px solid #16A34A"  # Green — received but resisted
_RECEIVED_BORDER = "3px solid #CA8A04"  # Yellow/amber — received payload
_BORDER_RADIUS = "4px"

_NODE_STATE_BORDERS: dict[str, str] = {
    "influenced": _INFLUENCED_BORDER,
    "resisted": _RESISTED_BORDER,
    "received": _RECEIVED_BORDER,
    # "clean" → no override (keep role-based default style)
}

_LEGEND_TEXT = (
    "🔴 Influenced &nbsp;&nbsp; 🟢 Resisted &nbsp;&nbsp; "
    "🟡 Received payload &nbsp;&nbsp; ⚪ Clean"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_attack_graph(
    config: MASConfig,
    target_agent_id: str | None,
    node_states: dict[str, str] | None = None,
) -> str | None:
    """Render the MAS flow graph with click-to-target and optional propagation overlay.

    Args:
        config: The MASConfig to visualize.
        target_agent_id: The currently selected attack target node ID (red border).
        node_states: Optional dict mapping agent_id to state string
            (``"influenced"``, ``"resisted"``, ``"received"``, or ``"clean"``).
            Pass ``None`` before any attack has been run.

    Returns:
        The ``agent_id`` of the clicked node, or ``None`` if no click occurred.
    """
    nodes, edges = convert_mas_to_graph(config)
    styled_nodes = _apply_style_overrides(nodes, target_agent_id, node_states)

    flow_key = f"attack_graph_{config.mas_id}"
    state_key = f"attack_flow_{config.mas_id}"
    version_key = f"attack_flow_version_{config.mas_id}"

    # Only rebuild StreamlitFlowState when something that affects node appearance
    # has actually changed (config, target, or post-run overlay).  Rebuilding on
    # every render sends fresh state to the component, which fires a state-update
    # event back to Streamlit, causing an infinite rerun loop.
    graph_version = (
        config.mas_id,
        target_agent_id,
        str(node_states),
    )
    if (
        state_key not in st.session_state
        or st.session_state.get(version_key) != graph_version
    ):
        st.session_state[state_key] = StreamlitFlowState(styled_nodes, edges)
        st.session_state[version_key] = graph_version

    st.session_state[state_key] = streamlit_flow(
        key=flow_key,
        state=st.session_state[state_key],
        height=420,
        fit_view=True,
        show_controls=True,
        show_minimap=False,
        allow_new_edges=False,
        pan_on_drag=True,
        allow_zoom=True,
        min_zoom=0.3,
        get_node_on_click=True,
        get_edge_on_click=False,
        enable_pane_menu=False,
        enable_node_menu=False,
        enable_edge_menu=False,
        hide_watermark=True,
    )

    if node_states is not None:
        st.caption(_LEGEND_TEXT, unsafe_allow_html=True)

    selected_id: str | None = getattr(st.session_state[state_key], "selected_id", None)
    # Only return agent node clicks (not edge clicks or None)
    if selected_id and config.get_agent(selected_id) is not None:
        return selected_id
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_style_overrides(
    nodes: list[StreamlitFlowNode],
    target_agent_id: str | None,
    node_states: dict[str, str] | None,
) -> list[StreamlitFlowNode]:
    """Return a new list of nodes with border style overrides applied.

    Priority (highest wins): post-run state override > target selection.
    """
    result: list[StreamlitFlowNode] = []
    for node in nodes:
        border: str | None = None

        # Post-run state overrides take priority
        if node_states and node.id in node_states:
            state = node_states[node.id]
            border = _NODE_STATE_BORDERS.get(state)
        elif node.id == target_agent_id:
            border = _TARGET_BORDER

        if border is not None:
            merged_style = {
                **node.style,
                "border": border,
                "borderRadius": _BORDER_RADIUS,
            }
            node = _clone_node(node, merged_style)

        result.append(node)
    return result


def _clone_node(node: StreamlitFlowNode, style: dict) -> StreamlitFlowNode:
    """Return a new StreamlitFlowNode identical to *node* but with *style*.

    Note: copies only the fields known at the time of writing.  If a future
    version of streamlit-flow-component adds new node attributes they will be
    silently dropped here — update this function accordingly.
    """
    # StreamlitFlowNode stores pos as self.position ({"x": ..., "y": ...})
    # and node_type as self.type — use those attribute names when reading back.
    pos = (node.position["x"], node.position["y"])
    return StreamlitFlowNode(
        id=node.id,
        pos=pos,
        data=node.data,
        node_type=node.type,
        source_position=node.source_position,
        target_position=node.target_position,
        draggable=node.draggable,
        connectable=node.connectable,
        deletable=node.deletable,
        style=style,
    )


def build_node_states(agent_observations: list) -> dict[str, str]:
    """Build a node_states dict from a list of AgentObservation objects or dicts.

    Args:
        agent_observations: List of ``AgentObservation`` objects (or dicts
            with the same field names) from an ``AttackResult``.

    Returns:
        Dict mapping ``agent_id`` to state string for graph overlay.
    """
    states: dict[str, str] = {}
    for obs in agent_observations:
        # Support both object attribute access and dict key access
        if hasattr(obs, "agent_id"):
            agent_id = obs.agent_id
            influenced = obs.influenced
            resisted = obs.resisted
            received = obs.received_payload
        else:
            agent_id = obs["agent_id"]
            influenced = obs["influenced"]
            resisted = obs["resisted"]
            received = obs["received_payload"]

        if influenced:
            states[agent_id] = "influenced"
        elif resisted:
            states[agent_id] = "resisted"
        elif received:
            states[agent_id] = "received"
        else:
            states[agent_id] = "clean"
    return states
