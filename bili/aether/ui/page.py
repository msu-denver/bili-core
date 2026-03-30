"""
AETHER page — MAS Visualizer and Chat.

Renders the AETHER multi-agent system graph visualizer with the multi-turn
chat interface.  Called by the main Streamlit app (``bili/streamlit_app.py``)
as a page within ``st.navigation()``.
"""

# pylint: disable=import-error
import logging
from pathlib import Path

# Suppress the FileNotFoundError traceback that Streamlit logs when the browser
# requests bootstrap.min.css.map — a source map file absent from the
# streamlit-flow-component package distribution. Source maps are optional
# browser developer tools and the missing file has no functional impact.
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(
    logging.ERROR
)

import streamlit as st

# Graceful import check for streamlit-flow
try:
    from streamlit_flow.elements import StreamlitFlowNode  # noqa: F401
except ImportError:
    st.error(
        "**streamlit-flow-component** is required but not installed.\n\n"
        "Install it with:\n```\npip install streamlit-flow-component\n```"
    )
    st.stop()

from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.ui.chat_app import render_main as render_chat_main
from bili.aether.ui.chat_app import (
    render_sidebar_content as render_chat_sidebar_content,
)
from bili.aether.ui.components.graph_viewer import (
    apply_agent_overrides,
    render_graph_viewer,
    render_metadata_bar,
)
from bili.aether.ui.converters.yaml_to_graph import convert_mas_to_graph

# Path constants
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

_DEFAULT_PAGE = "Visualizer"


def render_aether_page() -> None:
    """Render the AETHER page content (sidebar + main area).

    Called by the unified Streamlit app after ``st.set_page_config()``
    has already been invoked.
    """
    with st.sidebar:
        page = _render_sidebar()

    if page == "Chat":
        render_chat_main()
    else:
        _render_visualizer_main()


def _render_sidebar() -> str:
    """Render the sidebar: logo, nav radio, then page-specific controls.

    Returns the active page name (``"Visualizer"`` or ``"Chat"``).
    """
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)

    st.markdown("## AETHER")
    st.markdown("---")

    st.radio(
        "Navigation",
        ["Visualizer", "Chat"],
        key="aether_page",
        label_visibility="collapsed",
        horizontal=True,
    )
    st.markdown("---")

    page: str = st.session_state.get("aether_page", _DEFAULT_PAGE)
    if page == "Chat":
        render_chat_sidebar_content(examples_dir=EXAMPLES_DIR)
    else:
        _render_visualizer_sidebar()
    return page


def _render_visualizer_sidebar() -> None:
    """Render the visualizer-specific sidebar content."""
    st.caption("Multi-Agent System Graph Viewer")

    if not EXAMPLES_DIR.exists():
        st.error(f"Examples directory not found: {EXAMPLES_DIR}")
        return

    yaml_files = sorted(EXAMPLES_DIR.glob("*.yaml"))
    if not yaml_files:
        st.warning("No YAML files found in examples directory.")
        return

    yaml_display = [f.stem.replace("_", " ").title() for f in yaml_files]

    selected_idx = st.selectbox(
        "Select Configuration",
        range(len(yaml_files)),
        format_func=lambda i: yaml_display[i],
        key="yaml_selector",
    )

    if selected_idx is not None:
        yaml_path = yaml_files[selected_idx]
        _load_config(yaml_path)


def _render_intro() -> None:
    """Render introductory text about the AETHER framework."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=100)
    st.markdown(
        "<h1>AETHER Multi-Agent System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2><a href="https://github.com/msu-denver/bili-core">'
        "BiliCore on GitHub</a></h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**AETHER** (Agent Ecosystems for Testing, Hardening, Evaluation and "
        "Research) is a declarative multi-agent orchestration framework for "
        "building, visualizing, and evaluating coordinated teams of AI agents. "
        "Developed at Metropolitan State University of Denver as part of an "
        "NSF-funded, multi-institution research collaboration, AETHER extends "
        "BiliCore with a general-purpose orchestration layer where complex "
        "multi-agent systems are defined entirely through YAML configuration, "
        "with no orchestration code required."
    )
    st.markdown(
        "LLM applications are increasingly built as Multi-Agent Systems (MAS), "
        "where specialized agents divide labor, review each other's work, and "
        "reach consensus. AETHER makes it easy to rapidly build, configure, "
        "and vary these architectures. Define teams of agents with distinct "
        "roles, objectives, tools, and communication channels, then watch "
        "them collaborate in real time."
    )
    st.markdown(
        "AETHER supports **seven workflow patterns** to cover a wide range of "
        "agent collaboration strategies:"
    )
    st.markdown(
        "- **Sequential chains** for staged analysis where each agent builds "
        "on the previous agent's output\n"
        "- **Hierarchical voting** with multi-tier adversarial debate and "
        "weighted aggregation\n"
        "- **Supervisor routing** where a central agent dynamically delegates "
        "to specialist workers\n"
        "- **Consensus networks** with peer deliberation using majority, "
        "similarity, or explicit agreement detection\n"
        "- **Deliberative escalation** with human-in-the-loop review at "
        "decision boundaries\n"
        "- **Parallel fan-out** for concurrent analysis with broadcast and "
        "pub-sub coordination\n"
        "- **Custom graphs** with conditional edge routing for fully "
        "user-defined topologies"
    )
    st.markdown(
        "Each agent can be configured with its own LLM model from any "
        "provider BiliCore supports (AWS Bedrock, Google Vertex AI, Azure "
        "OpenAI, OpenAI, or local models), along with independent temperature "
        "settings, system prompts, tools, and middleware. Different agents in "
        "the same system can use different providers and models. Agents "
        "communicate through six typed channel protocols: direct, broadcast, "
        "request-response, pub-sub, competitive debate, and peer consensus. "
        "Rich agent sub-graphs (pipelines) allow individual agents to execute "
        "multi-node internal workflows with custom state fields and "
        "conditional branching."
    )
    st.markdown(
        "Use the **Visualizer** below to explore YAML configurations as "
        "interactive graphs, inspect agent properties and communication "
        "channels, and override model assignments. Then send any configuration "
        "to the **Chat** interface for live multi-turn conversations with the "
        "agent team. All interactions support real-time token-level streaming "
        "and checkpoint persistence across sessions."
    )
    st.markdown(
        "This research is supported by the National Science Foundation (NSF) "
        "and the National Artificial Intelligence Research Resource (NAIRR) "
        "Pilot, as part of the "
        "[Sustainability Hub](https://sustainabilityhub.co) research project "
        "at Metropolitan State University of Denver."
    )
    st.markdown(
        "**Note:** Refreshing the page will restart your session. "
        "For single-agent testing, switch to the "
        "**Single-Agent RAG** page."
    )
    st.markdown("---")


def _render_visualizer_main() -> None:
    """Render the visualizer main area."""
    _render_intro()

    config = st.session_state.get("mas_config")
    if config is None:
        st.info("Select a YAML configuration from the sidebar to visualize it.")
        return

    st.markdown(f"### {config.name}")
    if config.description:
        st.caption(config.description)

    nodes, edges = convert_mas_to_graph(config)
    render_graph_viewer(config, nodes, edges)

    st.markdown("---")
    render_metadata_bar(config)

    _render_legend()


def _on_send_to_chat() -> None:
    """Button callback: push the current visualizer config to the Chat page.

    Applies any agent overrides selected in the Visualizer before sending.
    Runs before the next render cycle so ``aether_page`` can be written
    safely without conflicting with the already-instantiated radio widget.
    """
    config = st.session_state.get("mas_config")
    if config is None:
        return

    config = apply_agent_overrides(config)

    name = st.session_state.get("current_yaml_path", "visualizer_config")
    if "chat_uploaded_configs" not in st.session_state:
        st.session_state.chat_uploaded_configs = {}
    st.session_state.chat_uploaded_configs[name] = config
    st.session_state.aether_page = "Chat"
    st.session_state.chat_autoload_name = name


def _load_config(yaml_path: Path) -> None:
    """Load a YAML config and store it in session state."""
    current_path = st.session_state.get("current_yaml_path")
    if current_path == str(yaml_path) and "mas_config" in st.session_state:
        config = st.session_state.mas_config
    else:
        try:
            config = load_mas_from_yaml(yaml_path)
            st.session_state.mas_config = config
            st.session_state.current_yaml_path = str(yaml_path)
            # Clear any previous flow state, agent overrides, and widget state
            keys_to_clear = [
                k
                for k in st.session_state
                if k.startswith("flow_state_")
                or k.startswith("agent_overrides_")
                or k.startswith("model_select_")
                or k.startswith("selected_node_")
            ]
            for k in keys_to_clear:
                del st.session_state[k]
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"Failed to load `{yaml_path.name}`:\n\n{e}")
            st.session_state.mas_config = None
            return

    # Show summary in sidebar
    st.markdown("---")
    st.markdown(f"**MAS ID:** `{config.mas_id}`")
    st.markdown(f"**Version:** {config.version}")
    st.markdown(f"**Workflow:** {config.workflow_type.value}")
    st.markdown(f"**Agents:** {len(config.agents)}")
    st.markdown(f"**Channels:** {len(config.channels)}")

    if config.consensus_threshold is not None:
        st.markdown(f"**Consensus:** {config.consensus_threshold}")
    if config.human_in_loop:
        st.warning("Human-in-loop enabled")

    st.markdown("---")
    st.button(
        "Send to Chat \u2192",
        disabled=st.session_state.get("mas_config") is None,
        use_container_width=True,
        on_click=_on_send_to_chat,
    )


def _render_legend() -> None:
    """Render a color legend for node and edge types."""
    with st.expander("Legend", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Edge Styles**")
            st.markdown("- **Solid line** \u2014 Communication channel")
            st.markdown("- **Dashed line** \u2014 Workflow edge")
            st.markdown("- **Animated** \u2014 Conditional edge")
        with col2:
            st.markdown("**Channel Protocols**")
            st.markdown("- :blue[Blue] \u2014 Direct")
            st.markdown("- :orange[Orange] \u2014 Broadcast")
            st.markdown("- :green[Green] \u2014 Request/Response")
            st.markdown("- :violet[Purple] \u2014 Pub/Sub")
            st.markdown("- :red[Red] \u2014 Competitive")
            st.markdown("- Teal \u2014 Consensus")
