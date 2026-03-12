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

from bili.aether.ui.page import render_aether_page
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS
from bili.checkpointers.checkpointer_functions import get_checkpointer
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

    # Page navigation with direct URL routes (AETHER is the default)
    pg = st.navigation(
        [
            st.Page(
                _run_aether_page,
                title="AETHER Multi-Agent",
                url_path="aether",
                icon=":material/hub:",
                default=True,
            ),
            st.Page(
                _run_bilicore_page,
                title="Single-Agent RAG",
                url_path="bili",
                icon=":material/chat:",
            ),
        ]
    )
    pg.run()


def _run_bilicore_page():
    """Render the Single-Agent RAG configuration and chat page."""
    with st.sidebar:
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

    st.markdown(
        """<h1>Single-Agent RAG</h1>
        <h2><a href="https://github.com/msu-denver/bili-core">
        BiliCore on GitHub</a></h2>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3>Build and test single-agent Retrieval-Augmented Generation (RAG) "
        "configurations using BiliCore.</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "This interface lets you experiment with different combinations of "
        "Language Models (LLMs), tools, prompts, and middleware to understand how "
        "each component affects agent behavior. Select a model from any supported "
        "provider (AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, or a "
        "local model), then attach tools like vector search, web search, or "
        "weather APIs to see how the agent responds."
    )
    st.markdown(
        "Use the configuration panels below to customize the agent, then click "
        '**"Load Configuration"** to apply your changes and start '
        "chatting. Each conversation is checkpointed so you can compare results "
        "across different configurations."
    )
    st.markdown(
        "**Note:** Refreshing the page will restart your session. "
        "For multi-agent workflows, switch to the "
        "**AETHER Multi-Agent** page."
    )
    st.markdown("---")

    checkpointer = get_checkpointer()
    run_app_page(checkpointer)


def _run_aether_page():
    """Render the AETHER multi-agent system page."""
    render_aether_page()


# Run the main function when the script is executed.
if __name__ == "__main__":
    main()
