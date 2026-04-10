"""
Main entry point for the Bili Core Streamlit application.

Provides a multi-page layout with sidebar navigation:
  - **AETHER Multi-Agent** (``/aether``) — MAS visualizer and chat (default)
  - **Single-Agent RAG** (``/bili``) — LLM comparison chatbot

Usage:
    streamlit run bili/streamlit_app.py
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from bili.aether.ui.attack_page import render_attack_page
from bili.aether.ui.attack_results_page import render_attack_results_page
from bili.aether.ui.page import render_aether_page
from bili.aether.ui.results_page import render_results_page
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS
from bili.iris.checkpointers.checkpointer_functions import get_checkpointer
from bili.streamlit_ui.ui.auth_ui import check_auth, initialize_auth_manager
from bili.streamlit_ui.ui.chat_interface import run_app_page

# Disable tokenizers parallelism to avoid issues with Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGO_PATH = Path(__file__).parent.joinpath("images", "logo.png")


def main():
    """Multi-page Streamlit app with shared authentication."""
    logo = Image.open(LOGO_PATH.as_posix())
    st.set_page_config(
        page_title="BiliCore | RAG & Multi-Agent Platform",
        page_icon=logo,
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # Shared auth — must log in before accessing any page
    st.session_state.auth_manager = initialize_auth_manager(
        auth_provider_name="sqlite",
        profile_provider_name="sqlite",
        role_provider_name="sqlite",
    )
    check_auth()

    # Page navigation grouped by component
    pg = st.navigation(
        {
            "IRIS": [
                st.Page(
                    _run_bilicore_page,
                    title="Single-Agent System",
                    url_path="iris",
                    icon=":material/chat:",
                ),
            ],
            "AETHER": [
                st.Page(
                    _run_aether_page,
                    title="Multi-Agent System",
                    url_path="aether",
                    icon=":material/hub:",
                    default=True,
                ),
            ],
            "AEGIS": [
                st.Page(
                    _run_attack_page,
                    title="Attack Suite",
                    url_path="attack",
                    icon=":material/gps_fixed:",
                ),
                st.Page(
                    _run_attack_results_page,
                    title="Attack Results",
                    url_path="attack-results",
                    icon=":material/security:",
                ),
                st.Page(
                    _run_results_page,
                    title="Baseline Results",
                    url_path="results",
                    icon=":material/bar_chart:",
                ),
            ],
        }
    )
    pg.run()


def _run_bilicore_page():
    """Render the IRIS single-agent configuration and chat page."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
        st.markdown("## IRIS")
        st.caption("Interactive Reasoning and Integration Services")
        st.markdown("---")
        st.markdown("#### Quick Navigation")
        st.markdown(
            "- [:material/tune: Configuration](#configuration)\n"
            "  - [:material/swap_horiz: Import/Export](#import-export)\n"
            "  - [:material/model_training: LLM Config](#llm-configuration)\n"
            "  - [:material/edit_note: Prompts](#prompts)\n"
            "  - [:material/build: Tools](#tools)\n"
            "  - [:material/history: Chat History](#chat-history)\n"
            "- [:material/play_arrow: Load & Run](#load-configuration)\n"
            "- [:material/settings: Active Config](#active-configuration)\n"
            "- [:material/forum: Conversation](#conversation)",
        )

    logo = Image.open(LOGO_PATH.as_posix())
    st.image(logo, width=100)

    st.markdown("# IRIS")
    st.caption("Interactive Reasoning and Integration Services")
    st.markdown(
        "IRIS is BiliCore's single-agent orchestration system. It bridges "
        "users to LLMs, tools, and data sources through a composable "
        "node-based pipeline. Swap models mid-conversation, configure tools "
        "on the fly, and persist state across sessions."
    )
    st.markdown(
        "Select a model from any supported provider (AWS Bedrock, Google "
        "Vertex AI, Azure OpenAI, OpenAI, Ollama, or a local model), then "
        "attach RAG tools like FAISS vector search or OpenSearch alongside "
        "web search, weather APIs, and other capabilities. Every "
        "conversation is checkpointed, so you can compare how different "
        "models respond under identical conditions."
    )
    st.markdown(
        "Use the configuration panels below to customize the agent, then "
        'click **"Load Configuration"** to apply your changes and start '
        "chatting."
    )
    st.markdown("---")

    checkpointer = get_checkpointer()
    run_app_page(checkpointer)


def _run_aether_page():
    """Render the AETHER multi-agent system page."""
    render_aether_page()


def _run_results_page():
    """Render the AETHER baseline results viewer page."""
    render_results_page()


def _run_attack_results_page():
    """Render the AETHER attack suite results viewer page."""
    render_attack_results_page()


def _run_attack_page():
    """Render the AETHER interactive Attack GUI page."""
    render_attack_page()


# Run the main function when the script is executed.
if __name__ == "__main__":
    main()
