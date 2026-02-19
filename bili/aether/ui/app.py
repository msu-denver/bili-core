"""
AETHER MAS YAML Visualization App.

A read-only Streamlit application that visualizes AETHER multi-agent
system YAML configurations as interactive node graphs.

Usage:
    streamlit run bili/aether/ui/app.py
"""

import sys
import types
from pathlib import Path

# The AETHER UI only needs bili.aether.* subpackages. Pre-register the
# bili package in sys.modules to prevent bili/__init__.py from eagerly
# importing all subpackages (flask_api, loaders, nodes, etc.) which
# require heavy dependencies (langgraph.prebuilt, etc.) not needed here.
if "bili" not in sys.modules:
    _bili_pkg = types.ModuleType("bili")
    _bili_pkg.__path__ = [str(Path(__file__).resolve().parent.parent.parent)]
    _bili_pkg.__package__ = "bili"
    sys.modules["bili"] = _bili_pkg

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
from bili.aether.ui.components.graph_viewer import (
    render_graph_viewer,
    render_metadata_bar,
)
from bili.aether.ui.converters.yaml_to_graph import convert_mas_to_graph
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS

# Path constants
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"


def main():
    """Main entry point for the AETHER visualization app."""
    _configure_page()

    with st.sidebar:
        _render_sidebar()

    config = st.session_state.get("mas_config")
    if config is None:
        st.info("Select a YAML configuration from the sidebar to visualize it.")
        return

    # Title and description
    st.markdown(f"### {config.name}")
    if config.description:
        st.caption(config.description)

    # Convert and render
    nodes, edges = convert_mas_to_graph(config)
    render_graph_viewer(config, nodes, edges)

    # Metadata bar below graph
    st.markdown("---")
    render_metadata_bar(config)

    # Legend
    _render_legend()


def _configure_page():
    """Set up Streamlit page config."""
    st.set_page_config(
        page_title="AETHER - MAS Visualizer",
        page_icon="A",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_sidebar():
    """Render the sidebar with YAML selector and MAS info."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)

    st.markdown("## AETHER Visualizer")
    st.caption("Multi-Agent System Graph Viewer")
    st.markdown("---")

    # Find all YAML files in examples directory
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


def _load_config(yaml_path: Path):
    """Load a YAML config and store it in session state."""
    # Only reload if the path changed
    current_path = st.session_state.get("current_yaml_path")
    if current_path == str(yaml_path) and "mas_config" in st.session_state:
        config = st.session_state.mas_config
    else:
        try:
            config = load_mas_from_yaml(yaml_path)
            st.session_state.mas_config = config
            st.session_state.current_yaml_path = str(yaml_path)
            # Clear any previous flow state so graph re-renders
            keys_to_clear = [k for k in st.session_state if k.startswith("flow_state_")]
            for k in keys_to_clear:
                del st.session_state[k]
        except Exception as e:
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


def _render_legend():
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


if __name__ == "__main__":
    main()
