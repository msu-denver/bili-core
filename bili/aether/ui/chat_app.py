"""
AETHER Chat Interface.

A Streamlit chat application for multi-turn interaction with a compiled
multi-agent system (MAS). Supports stub mode for configs without LLM
model names and renders per-agent outputs in expandable panels.

Usage:
    streamlit run bili/aether/ui/chat_app.py
"""

# pylint: disable=import-error
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress the FileNotFoundError traceback that Streamlit logs when the browser
# requests bootstrap.min.css.map — a source map file absent from the
# streamlit-flow-component package distribution. Source maps are optional
# browser developer tools and the missing file has no functional impact.
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(
    logging.ERROR
)

import streamlit as st
from langchain_core.messages import BaseMessage

from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.runtime import MASExecutor
from bili.aether.schema import MASConfig
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_stub_config(config: MASConfig) -> bool:
    """Return True if any agent in *config* has no ``model_name`` (stub mode)."""
    return any(agent.model_name is None for agent in config.agents)


def _serialize_state_update(state_update: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw graph state update to a JSON-serializable dict.

    LangChain ``BaseMessage`` objects are replaced with their ``content``
    string so the result can be stored in session state and exported.
    """

    def _convert(value: Any) -> Any:
        if isinstance(value, BaseMessage):
            return {"type": type(value).__name__, "content": value.content}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return {k: _convert(v) for k, v in state_update.items()}


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------


def _configure_page() -> None:
    """Set up Streamlit page config and inject BiliCore CSS."""
    st.set_page_config(
        page_title="AETHER - Chat",
        page_icon="A",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(yaml_path: Path) -> None:
    """Load a YAML config and initialize the executor when the path changes."""
    current_path = st.session_state.get("chat_yaml_path")
    if current_path == str(yaml_path) and "chat_config" in st.session_state:
        return

    try:
        config = load_mas_from_yaml(yaml_path)
        executor = MASExecutor(config)
        executor.initialize()
        st.session_state.chat_config = config
        st.session_state.chat_yaml_path = str(yaml_path)
        st.session_state.chat_executor = executor
        st.session_state.chat_history = []
        st.session_state.pop("chat_thread_id", None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error("Failed to load config %s: %s", yaml_path.name, exc, exc_info=True)
        st.error(f"Failed to load `{yaml_path.name}`:\n\n{exc}")
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render sidebar with YAML selector, stub indicator, and controls."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)

    st.markdown("## AETHER Chat")
    st.caption("Multi-Agent System Conversation")
    st.markdown("---")

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
        index=None,
        placeholder="Choose a MAS config...",
        format_func=lambda i: yaml_display[i],
        key="chat_yaml_selector",
    )

    if selected_idx is not None:
        _load_config(yaml_files[selected_idx])
    else:
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)
        return

    config: Optional[MASConfig] = st.session_state.get("chat_config")
    if config is None:
        return

    st.markdown("---")

    if _is_stub_config(config):
        st.info("Stub mode — no LLM calls will be made.", icon="⚙️")

    st.markdown(f"**MAS ID:** `{config.mas_id}`")
    st.markdown(f"**Workflow:** {config.workflow_type.value}")
    st.markdown(f"**Agents:** {len(config.agents)}")

    st.markdown("---")

    if st.button("New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_thread_id = str(uuid.uuid4())
        st.rerun()


# ---------------------------------------------------------------------------
# Agent output rendering
# ---------------------------------------------------------------------------


def _render_agent_output(node_name: str, state_update: Dict[str, Any]) -> None:
    """Render one agent node's state update inside an expander."""
    with st.expander(f"Agent: {node_name}", expanded=True):
        # Prefer the structured agent_outputs entry for this node
        agent_outputs: Dict[str, Any] = state_update.get("agent_outputs", {})
        if node_name in agent_outputs:
            output = agent_outputs[node_name]
            status = output.get("status", "")
            if status == "stub":
                st.caption(output.get("message", str(output)))
            else:
                for key, value in output.items():
                    if key not in ("agent_id",):
                        st.markdown(f"**{key}:** {value}")
            return

        # Fall back to the most recent message in the state update
        messages: List[Any] = state_update.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", str(last))
            st.markdown(content)
            return

        # Last resort: display raw state
        st.json(
            {k: str(v) for k, v in state_update.items()},
            expanded=False,
        )


def _render_stored_turn(turn: Dict[str, Any]) -> None:
    """Re-render a previously completed turn from chat_history."""
    with st.chat_message("user"):
        st.markdown(turn["content"])
    with st.chat_message("assistant"):
        for agent_out in turn.get("agent_outputs", []):
            agent_id = agent_out["agent_id"]
            output = agent_out["output"]
            with st.expander(f"Agent: {agent_id}", expanded=False):
                agent_outputs = output.get("agent_outputs", {})
                if agent_id in agent_outputs:
                    inner = agent_outputs[agent_id]
                    status = inner.get("status", "")
                    if status == "stub":
                        st.caption(inner.get("message", str(inner)))
                    else:
                        for key, value in inner.items():
                            if key not in ("agent_id",):
                                st.markdown(f"**{key}:** {value}")
                elif "messages" in output and output["messages"]:
                    last = output["messages"][-1]
                    content = (
                        last.get("content", str(last))
                        if isinstance(last, dict)
                        else str(last)
                    )
                    st.markdown(content)
                else:
                    st.json({k: str(v) for k, v in output.items()}, expanded=False)


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------


def _run_turn(user_input: str) -> None:
    """Execute one conversation turn and append it to chat_history."""
    executor: Optional[MASExecutor] = st.session_state.get("chat_executor")
    if executor is None:
        st.error("No executor available. Select a configuration first.")
        return

    if "chat_thread_id" not in st.session_state:
        st.session_state.chat_thread_id = str(uuid.uuid4())

    from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
        HumanMessage,
    )

    turn: Dict[str, Any] = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_index": len(st.session_state.get("chat_history", [])),
        "agent_outputs": [],
    }

    with st.chat_message("user"):
        st.markdown(user_input)

    agent_outputs: List[Dict[str, Any]] = []
    with st.chat_message("assistant"):
        for node_name, state_update in executor.run_streaming(
            input_data={"messages": [HumanMessage(content=user_input)]},
            thread_id=st.session_state.chat_thread_id,
        ):
            _render_agent_output(node_name, state_update)
            agent_outputs.append(
                {
                    "agent_id": node_name,
                    "output": _serialize_state_update(state_update),
                }
            )

    turn["agent_outputs"] = agent_outputs

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(turn)


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------


def _render_chat_area() -> None:
    """Render the main chat area: empty state, history, and input."""
    config: Optional[MASConfig] = st.session_state.get("chat_config")
    if config is None:
        st.info("Select a YAML configuration from the sidebar to begin.")
        return

    st.markdown(f"### {config.name}")
    if config.description:
        st.caption(config.description)

    for turn in st.session_state.get("chat_history", []):
        _render_stored_turn(turn)

    user_input = st.chat_input("Send a message to the MAS...")
    if user_input:
        _run_turn(user_input)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_page() -> None:
    """Main entry point — callable from app.py navigation or standalone."""
    _configure_page()
    with st.sidebar:
        _render_sidebar()
    _render_chat_area()


if __name__ == "__main__":
    render_page()
