"""
Graph viewer component: renders the interactive node graph and properties panel.

The graph is read-only -- nodes cannot be dragged, connected, or deleted.
Clicking a node shows its agent properties in a side panel.
"""

import streamlit as st

# pylint: disable=import-error
import yaml
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig

MODEL_KEEP_SENTINEL = "(keep from YAML)"


def _overrides_key(mas_id: str) -> str:
    return f"agent_overrides_{mas_id}"


def apply_agent_overrides(config: MASConfig) -> MASConfig:
    """Return *config* with current agent_overrides from session state applied.

    Called inside the ``render_graph_viewer`` fragment (so the Download YAML
    button always reflects the latest widget values) and re-exported for use
    by ``page.py``'s Send-to-Chat callback.
    """
    overrides: dict = st.session_state.get(_overrides_key(config.mas_id), {})
    if not overrides:
        return config
    _, display_to_model_name, _ = build_model_options()
    patched_agents = []
    for agent in config.agents:
        agent_override = overrides.get(agent.agent_id, {})
        patch: dict = {}
        display = agent_override.get("model_name")
        if (
            display
            and display != MODEL_KEEP_SENTINEL
            and display in display_to_model_name
        ):
            patch["model_name"] = display_to_model_name[display]
        if agent_override.get("system_prompt"):
            patch["system_prompt"] = agent_override["system_prompt"]
        if agent_override.get("objective"):
            patch["objective"] = agent_override["objective"]
        if "temperature" in agent_override:
            patch["temperature"] = agent_override["temperature"]
        if "max_tokens" in agent_override:
            patch["max_tokens"] = agent_override["max_tokens"]
        if "tools" in agent_override:
            patch["tools"] = agent_override["tools"]
        patched_agents.append(agent.model_copy(update=patch) if patch else agent)
    return config.model_copy(update={"agents": patched_agents})


@st.cache_data
def _get_tool_names() -> list[str]:
    """Return sorted list of registered tool names."""
    from bili.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
        TOOL_REGISTRY,
    )

    return sorted(TOOL_REGISTRY.keys())


@st.cache_data
def build_model_options() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (display_list, display_to_model_name, model_lookup) from LLM_MODELS.

    The ``display_to_model_name`` dict maps each display string back to the
    LLM_MODELS ``model_name`` field. It is consumed by the Deploy/Run task
    to resolve UI selections back to ``AgentSpec.model_name`` values before
    patching the config and calling ``MASExecutor``.

    The ``model_lookup`` dict maps both ``model_id`` and ``model_name`` keys to
    their selectbox display string, used to pre-select the current model when
    rendering the model override selectbox.
    """
    from bili.config.llm_config import (  # pylint: disable=import-outside-toplevel
        LLM_MODELS,
    )

    options: list[str] = []
    name_to_model: dict[str, str] = {}
    model_lookup: dict[str, str] = {}
    for provider_info in LLM_MODELS.values():
        provider_label = provider_info["name"]
        for entry in provider_info["models"]:
            display = f"[{provider_label}] {entry['model_name']}"
            options.append(display)
            name_to_model[display] = entry["model_name"]
            model_lookup[entry["model_id"]] = display
            model_lookup[entry["model_name"]] = display
    return options, name_to_model, model_lookup


@st.fragment
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

        raw_selected = st.session_state[state_key].selected_id

        # Persist the last real click across fragment reruns.  Widget changes
        # (slider, text_area) cause the fragment to rerun with selected_id=None
        # even though the user hasn't deselected anything — without this, the
        # properties panel collapses every time a field is edited.
        persist_key = f"selected_node_{config.mas_id}"
        if raw_selected is not None:
            st.session_state[persist_key] = raw_selected
        selected_id = st.session_state.get(persist_key)

    # Initialize agent override dict for this MAS
    overrides_key = _overrides_key(config.mas_id)
    if overrides_key not in st.session_state:
        st.session_state[overrides_key] = {}

    with props_col:
        patched = apply_agent_overrides(config)
        st.download_button(
            "Download YAML",
            data=yaml.dump(
                patched.model_dump(mode="json"),
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            file_name=f"{patched.mas_id}.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
        st.markdown("---")
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
    options, _, model_lookup = build_model_options()
    overrides = st.session_state[_overrides_key(mas_id)]

    # agent_overrides stores a dict per agent; read the model_name sub-key.
    agent_override = overrides.get(agent.agent_id, {})
    current_model_display = agent_override.get("model_name")
    if current_model_display and current_model_display in options:
        start_index = options.index(current_model_display) + 1  # +1 for sentinel
    else:
        yaml_display = model_lookup.get(agent.model_name)
        start_index = (
            (options.index(yaml_display) + 1) if yaml_display in options else 0
        )

    st.markdown("**Model**")
    selected = st.selectbox(
        "Model",
        [MODEL_KEEP_SENTINEL] + options,
        index=start_index,
        key=f"model_select_{mas_id}_{agent.agent_id}",
        help="Override the LLM for this agent. '(keep from YAML)' uses the configured model.",
        label_visibility="collapsed",
    )

    bucket = overrides.setdefault(agent.agent_id, {})
    if selected == MODEL_KEEP_SENTINEL:
        bucket.pop("model_name", None)
    else:
        bucket["model_name"] = selected


def _render_agent_properties(agent: AgentSpec, mas_id: str) -> None:
    """Render detailed agent properties with editable override fields."""
    st.markdown(f"**{agent.get_display_name()}**")
    st.caption(f"`{agent.agent_id}`")

    st.markdown("---")
    st.markdown(f"**Role:** {agent.role}")

    overrides = st.session_state[_overrides_key(mas_id)]
    bucket = overrides.setdefault(agent.agent_id, {})

    st.markdown("**Objective**")
    obj_val = st.text_area(
        "objective",
        value=bucket.get("objective", agent.objective),
        key=f"obj_{mas_id}_{agent.agent_id}",
        label_visibility="collapsed",
        height=100,
        help="Override this agent's objective for this session.",
    )
    bucket["objective"] = obj_val

    _render_model_selector(agent, mas_id)

    st.markdown("**System Prompt**")
    sp_val = st.text_area(
        "system_prompt",
        value=bucket.get("system_prompt", agent.system_prompt or ""),
        key=f"sp_{mas_id}_{agent.agent_id}",
        label_visibility="collapsed",
        height=80,
        help="Override this agent's system prompt for this session.",
    )
    if sp_val:
        bucket["system_prompt"] = sp_val
    else:
        bucket.pop("system_prompt", None)

    st.markdown("**Temperature**")
    default_temp = float(agent.temperature if agent.temperature is not None else 0.7)
    temp_val = st.slider(
        "temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(bucket.get("temperature", default_temp)),
        step=0.1,
        key=f"temp_{mas_id}_{agent.agent_id}",
        label_visibility="collapsed",
    )
    bucket["temperature"] = temp_val

    st.markdown("**Max Tokens**")
    max_tokens_val = st.number_input(
        "max_tokens",
        min_value=1,
        max_value=32768,
        value=int(bucket.get("max_tokens", agent.max_tokens or 1024)),
        step=256,
        key=f"max_tokens_{mas_id}_{agent.agent_id}",
        label_visibility="collapsed",
        help="Maximum tokens the agent may generate per turn.",
    )
    bucket["max_tokens"] = int(max_tokens_val)

    st.markdown("**Tools**")
    yaml_tools = agent.tools or []
    if yaml_tools:
        for tool in yaml_tools:
            st.markdown(f"- `{tool}`")
    else:
        st.caption("None configured in YAML")
    available_tools = _get_tool_names()
    if available_tools:
        current_tools = bucket.get("tools", yaml_tools)
        selected_tools = st.multiselect(
            "Override tools",
            options=available_tools,
            default=[t for t in current_tools if t in available_tools],
            key=f"tools_{mas_id}_{agent.agent_id}",
            help="Override the tool set for this session.",
        )
        if selected_tools != yaml_tools:
            bucket["tools"] = selected_tools
        else:
            bucket.pop("tools", None)

    if agent.capabilities:
        _render_list_section("Capabilities", agent.capabilities)

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
