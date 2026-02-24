"""
Graph viewer component: renders the interactive node graph and properties panel.

The graph is read-only -- nodes cannot be dragged, connected, or deleted.
Clicking a node shows its agent properties in a side panel.
"""

# pylint: disable=import-error
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig


def _overrides_key(mas_id: str) -> str:
    return f"model_overrides_{mas_id}"


@st.cache_data
def _build_model_options() -> tuple[list[str], dict[str, str]]:
    """Return (display_list, display_to_model_name) from LLM_MODELS.

    The ``display_to_model_name`` dict maps each display string back to the
    LLM_MODELS ``model_name`` field. It is consumed by the Deploy/Run task
    to resolve UI selections back to ``AgentSpec.model_name`` values before
    patching the config and calling ``MASExecutor``.
    """
    from bili.config.llm_config import (  # pylint: disable=import-outside-toplevel
        LLM_MODELS,
    )

    options: list[str] = []
    name_to_model: dict[str, str] = {}
    for provider_info in LLM_MODELS.values():
        provider_label = provider_info["name"]
        for entry in provider_info["models"]:
            display = f"[{provider_label}] {entry['model_name']}"
            options.append(display)
            name_to_model[display] = entry["model_name"]
    return options, name_to_model


@st.cache_data
def _find_model_display(model_name: str | None) -> str | None:
    """Resolve a raw model_name (display string or model_id) to its selectbox display string."""
    from bili.config.llm_config import (  # pylint: disable=import-outside-toplevel
        LLM_MODELS,
    )

    if not model_name:
        return None
    lookup: dict[str, str] = {}
    for provider_info in LLM_MODELS.values():
        provider_label = provider_info["name"]
        for entry in provider_info["models"]:
            display = f"[{provider_label}] {entry['model_name']}"
            lookup[entry["model_id"]] = display
            lookup[entry["model_name"]] = display
    return lookup.get(model_name)


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

    # Initialize model override dict for this MAS
    overrides_key = _overrides_key(config.mas_id)
    if overrides_key not in st.session_state:
        st.session_state[overrides_key] = {}

    with props_col:
        _render_properties_panel(config, selected_id, edges, config.mas_id)


def _render_properties_panel(
    config: MASConfig,
    selected_id: str | None,
    edges: list[StreamlitFlowEdge],
    mas_id: str,
) -> None:
    """Render the right-side properties panel for a clicked node or edge."""
    st.markdown("#### Properties")

    if not selected_id:
        st.caption("Click a node to view its properties.")
        return

    # Check if it's a node (agent_id)
    agent = config.get_agent(selected_id)
    if agent:
        _render_agent_properties(agent, mas_id)
        return

    # Check if it's an edge
    for edge in edges:
        if edge.id == selected_id:
            _render_edge_properties(edge, config)
            return

    st.caption(f"No details for: {selected_id}")


def _render_list_section(title: str, items: list) -> None:
    """Render a bold title followed by a backtick-formatted bullet list."""
    st.markdown(f"**{title}:**")
    for item in items:
        st.markdown(f"- `{item}`")


def _render_model_selector(agent: AgentSpec, mas_id: str) -> None:
    """Render the model override selectbox and persist the selection to session state."""
    _keep = "(keep from YAML)"
    options, _ = _build_model_options()
    overrides = st.session_state[_overrides_key(mas_id)]

    current_override = overrides.get(agent.agent_id)
    if current_override and current_override in options:
        start_index = options.index(current_override) + 1  # +1 for sentinel
    else:
        yaml_display = _find_model_display(agent.model_name)
        start_index = (
            (options.index(yaml_display) + 1) if yaml_display in options else 0
        )

    selected = st.selectbox(
        "Model",
        [_keep] + options,
        index=start_index,
        key=f"model_select_{mas_id}_{agent.agent_id}",
        help="Override the LLM for this agent. '(keep from YAML)' uses the configured model.",
    )

    if selected == _keep:
        overrides.pop(agent.agent_id, None)
    else:
        overrides[agent.agent_id] = selected


def _render_agent_properties(agent: AgentSpec, mas_id: str) -> None:
    """Render detailed agent properties."""
    st.markdown(f"**{agent.get_display_name()}**")
    st.caption(f"`{agent.agent_id}`")

    st.markdown("---")
    st.markdown(f"**Role:** {agent.role}")
    st.markdown(f"**Objective:** {agent.objective}")
    _render_model_selector(agent, mas_id)

    if agent.temperature is not None:
        st.markdown(f"**Temperature:** {agent.temperature}")
    if agent.max_tokens:
        st.markdown(f"**Max Tokens:** {agent.max_tokens}")

    if agent.capabilities:
        _render_list_section("Capabilities", agent.capabilities)

    if agent.tools:
        _render_list_section("Tools", agent.tools)

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
        _render_list_section("Middleware", agent.middleware)

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
        if ch.source == edge.source and ch.target in (edge.target, "all"):
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
