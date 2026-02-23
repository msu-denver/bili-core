"""
Graph viewer component: renders the interactive node graph and properties panel.

The graph is read-only -- nodes cannot be dragged, connected, or deleted.
Clicking a node shows its agent properties in a side panel.
"""

# pylint: disable=import-error
from typing import Optional

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig


def render_graph_viewer(
    config: MASConfig,
    nodes: list[StreamlitFlowNode],
    edges: list[StreamlitFlowEdge],
) -> None:
    """Render the interactive graph with a properties panel.

    Args:
        config: The MASConfig being visualized.
        nodes: StreamlitFlowNode objects from the converter.
        edges: StreamlitFlowEdge objects from the converter.
    """
    graph_col, props_col = st.columns([3, 1])

    with graph_col:
        # Initialize or update state in session_state
        flow_key = f"mas_graph_{config.mas_id}"
        state_key = f"flow_state_{config.mas_id}"

        if state_key not in st.session_state:
            st.session_state[state_key] = StreamlitFlowState(nodes, edges)

        st.session_state[state_key] = streamlit_flow(
            key=flow_key,
            state=st.session_state[state_key],
            height=550,
            fit_view=True,
            show_controls=True,
            show_minimap=False,
            allow_new_edges=False,
            pan_on_drag=True,
            allow_zoom=True,
            min_zoom=0.3,
            get_node_on_click=True,
            get_edge_on_click=True,
            enable_pane_menu=False,
            enable_node_menu=False,
            enable_edge_menu=False,
            hide_watermark=True,
        )

        selected_id = st.session_state[state_key].selected_id

    with props_col:
        _render_properties_panel(config, selected_id, edges)


def _render_properties_panel(
    config: MASConfig,
    selected_id: Optional[str],
    edges: list[StreamlitFlowEdge],
) -> None:
    """Render the right-side properties panel for a clicked node or edge."""
    st.markdown("#### Properties")

    if not selected_id:
        st.caption("Click a node to view its properties.")
        return

    # Check if it's a node (agent_id)
    agent = config.get_agent(selected_id)
    if agent:
        _render_agent_properties(agent)
        return

    # Check if it's an edge
    for edge in edges:
        if edge.id == selected_id:
            _render_edge_properties(edge, config)
            return

    st.caption(f"No details for: {selected_id}")


def _render_agent_properties(agent: AgentSpec) -> None:
    """Render detailed agent properties."""
    st.markdown(f"**{agent.get_display_name()}**")
    st.caption(f"`{agent.agent_id}`")

    st.markdown("---")
    st.markdown(f"**Role:** {agent.role}")
    st.markdown(f"**Objective:** {agent.objective}")

    if agent.model_name:
        st.markdown(f"**Model:** `{agent.model_name}`")
    if agent.temperature is not None:
        st.markdown(f"**Temperature:** {agent.temperature}")
    if agent.max_tokens:
        st.markdown(f"**Max Tokens:** {agent.max_tokens}")

    if agent.capabilities:
        st.markdown("**Capabilities:**")
        for cap in agent.capabilities:
            st.markdown(f"- `{cap}`")

    if agent.tools:
        st.markdown("**Tools:**")
        for tool in agent.tools:
            st.markdown(f"- `{tool}`")

    st.markdown(f"**Output:** {agent.output_format.value}")

    if agent.is_supervisor:
        st.markdown(
            '<span class="supervisor-badge">Supervisor</span>',
            unsafe_allow_html=True,
        )
    if agent.tier is not None:
        st.markdown(f"**Tier:** {agent.tier}")
    if agent.voting_weight != 1.0:
        st.markdown(f"**Voting Weight:** {agent.voting_weight}")

    if agent.middleware:
        st.markdown("**Middleware:**")
        for mw in agent.middleware:
            st.markdown(f"- `{mw}`")

    if agent.inherit_from_bili_core:
        st.markdown(
            '<span class="inheritance-badge">Inherits from bili-core</span>',
            unsafe_allow_html=True,
        )


def _render_edge_properties(edge: StreamlitFlowEdge, config: MASConfig) -> None:
    """Render edge properties."""
    st.markdown(f"**{edge.source}** \u2192 **{edge.target}**")
    if edge.label:
        st.markdown(f"**Label:** {edge.label}")

    # Try to find matching channel for more details
    for ch in config.channels:
        if ch.source == edge.source and (
            ch.target == edge.target or ch.target == "all"
        ):
            st.markdown(f"**Protocol:** {ch.protocol.value}")
            if ch.description:
                st.markdown(f"**Description:** {ch.description}")
            if ch.bidirectional:
                st.success("Bidirectional", icon="\u2194\ufe0f")
            break

    # Check workflow edges for condition info
    for we in config.workflow_edges:
        if we.from_agent == edge.source and we.to_agent == edge.target:
            if we.condition:
                st.code(we.condition, language="python")
            break


def render_metadata_bar(config: MASConfig) -> None:
    """Render MAS metadata summary below the graph."""
    cols = st.columns(5)

    with cols[0]:
        st.metric("Agents", len(config.agents))
    with cols[1]:
        st.metric("Channels", len(config.channels))
    with cols[2]:
        st.metric("Workflow", config.workflow_type.value.title())
    with cols[3]:
        st.metric("Edges", len(config.workflow_edges))
    with cols[4]:
        if config.tags:
            tags_str = ", ".join(config.tags[:3])
            if len(config.tags) > 3:
                tags_str += "..."
        else:
            tags_str = "None"
        st.metric("Tags", tags_str)
