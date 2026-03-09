"""
AETHER Chat Interface.

A Streamlit chat application for multi-turn interaction with a compiled
multi-agent system (MAS). Supports stub mode for configs without LLM
model names and renders per-agent outputs in expandable panels.

Usage:
    streamlit run bili/aether/ui/chat_app.py
"""

# pylint: disable=import-error
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Suppress the FileNotFoundError traceback that Streamlit logs when the browser
# requests bootstrap.min.css.map — a source map file absent from the
# streamlit-flow-component package distribution. Source maps are optional
# browser developer tools and the missing file has no functional impact.
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(
    logging.ERROR
)

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage

from bili.aether.config.loader import load_mas_from_dict, load_mas_from_yaml
from bili.aether.runtime import MASExecutor
from bili.aether.schema import MASConfig
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS
from bili.aether.validation import validate_mas

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_stub_config(config: MASConfig) -> bool:
    """Return True if any agent in *config* has no ``model_name`` (stub mode)."""
    return any(not agent.model_name for agent in config.agents)


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
# Config validation
# ---------------------------------------------------------------------------


def _validate_config(config: MASConfig) -> bool:
    """Run structural validation and display results in the active Streamlit context.

    Returns ``True`` if the config is valid (errors list is empty).
    Warnings are displayed but do not block execution.
    """
    result = validate_mas(config)
    for err in result.errors:
        st.error(f"Config error: {err}")
    for warn in result.warnings:
        st.warning(f"Config warning: {warn}")
    if result.valid:
        if result.warnings:
            st.success("Config valid with warnings ✓")
        else:
            st.success("Config valid ✓")
    return result.valid


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


def _initialize_executor(config: MASConfig, cache_key: str) -> None:
    """Compile the executor and update session state, or store the error."""
    if (
        st.session_state.get("chat_yaml_path") == cache_key
        and "chat_config" in st.session_state
    ):
        return

    try:
        executor = MASExecutor(config)
        with st.spinner("Initializing executor..."):
            executor.initialize()
        st.session_state.chat_config = config
        st.session_state.chat_yaml_path = cache_key
        st.session_state.chat_executor = executor
        st.session_state.chat_history = []
        st.session_state.pop("chat_thread_id", None)
        st.session_state.pop("chat_load_error", None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error("Failed to load config %s: %s", cache_key, exc, exc_info=True)
        st.session_state.chat_load_error = str(exc)
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)


def _load_config(yaml_path: Path) -> None:
    """Load a YAML config and initialize the executor when the path changes."""
    try:
        config = load_mas_from_yaml(yaml_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error("Failed to load config %s: %s", yaml_path.name, exc, exc_info=True)
        st.session_state.chat_load_error = str(exc)
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)
        return
    if not _validate_config(config):
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)
        return
    _initialize_executor(config, str(yaml_path))


def _load_uploaded_config(name: str, config: MASConfig) -> None:
    """Initialize the executor from an already-parsed uploaded MASConfig."""
    if not _validate_config(config):
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)
        return
    _initialize_executor(config, f"uploaded:{name}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render standalone sidebar: logo, title, and chat controls."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)

    st.markdown("## AETHER Chat")
    st.caption("Multi-Agent System Conversation")
    st.markdown("---")
    render_sidebar_content()


def render_sidebar_content() -> None:
    """Render the chat sidebar controls — config selector, upload, and buttons.

    Public: called by ``app.py`` when the Chat page is active.
    """
    uploaded = st.file_uploader("Upload YAML config", type=["yaml", "yml"])
    if uploaded and uploaded.name not in st.session_state.get(
        "chat_uploaded_configs", {}
    ):
        try:
            raw = yaml.safe_load(uploaded.read())
            if not isinstance(raw, dict):
                st.error("Invalid config: YAML must be a mapping at the top level.")
            else:
                upload_config = load_mas_from_dict(raw)
                if "chat_uploaded_configs" not in st.session_state:
                    st.session_state.chat_uploaded_configs = {}
                st.session_state.chat_uploaded_configs[uploaded.name] = upload_config
                st.success(f"Uploaded: {uploaded.name}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Invalid config: {exc}")

    if not EXAMPLES_DIR.exists():
        st.error(f"Examples directory not found: {EXAMPLES_DIR}")
        return

    yaml_files = sorted(EXAMPLES_DIR.glob("*.yaml"))
    uploaded_configs: Dict[str, MASConfig] = st.session_state.get(
        "chat_uploaded_configs", {}
    )
    uploaded_names = sorted(uploaded_configs.keys())

    if not yaml_files and not uploaded_names:
        st.warning("No YAML files found in examples directory.")
        return

    all_display = [f.stem.replace("_", " ").title() for f in yaml_files] + [
        f"[Uploaded] {n}" for n in uploaded_names
    ]
    n_files = len(yaml_files)

    selected_idx = st.selectbox(
        "Select Configuration",
        range(len(all_display)),
        index=None,
        placeholder="Choose a MAS config...",
        format_func=lambda i: all_display[i],
        key="chat_yaml_selector",
    )

    if selected_idx is not None:
        if selected_idx < n_files:
            _load_config(yaml_files[selected_idx])
        else:
            name = uploaded_names[selected_idx - n_files]
            _load_uploaded_config(name, uploaded_configs[name])
    else:
        for key in (
            "chat_config",
            "chat_yaml_path",
            "chat_executor",
            "chat_load_error",
        ):
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

    chat_history = st.session_state.get("chat_history", [])
    thread_id = st.session_state.get("chat_thread_id", "")
    if chat_history:
        export = {
            "mas_id": config.mas_id,
            "thread_id": thread_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "turns": chat_history,
        }
        st.download_button(
            "Export Conversation",
            data=json.dumps(export, indent=2),
            file_name=f"aether_chat_{config.mas_id}_{thread_id[:8] or 'unknown'}.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Agent output rendering
# ---------------------------------------------------------------------------


def _render_agent_panel(
    agent_id: str,
    output: Dict[str, Any],
    *,
    expanded: bool,
    use_expander: bool = True,
) -> None:
    """Render one agent's output inside an expander or plain container.

    Accepts both raw graph state updates (containing ``BaseMessage`` objects)
    and already-serialized output dicts stored in ``chat_history``.

    When *use_expander* is ``False`` the panel is rendered as a ``st.container``
    with a bold header — suitable for use inside an outer expander to avoid
    nesting expanders inside expanders.
    """
    if use_expander:
        ctx = st.expander(f"Agent: {agent_id}", expanded=expanded)
    else:
        ctx = st.container()

    with ctx:
        if not use_expander:
            st.markdown(f"**Agent: {agent_id}**")

        # Prefer the structured agent_outputs entry for this node
        agent_outputs: Dict[str, Any] = output.get("agent_outputs", {})
        if agent_id in agent_outputs:
            inner = agent_outputs[agent_id]
            if inner.get("status") == "stub":
                st.caption(inner.get("message", str(inner)))
            else:
                for key, value in inner.items():
                    if key != "agent_id":
                        st.markdown(f"**{key}:** {value}")
            return

        # Fall back to the most recent message
        messages: List[Any] = output.get("messages", [])
        if messages:
            last = messages[-1]
            content = (
                last.get("content", str(last))  # serialized dict from chat_history
                if isinstance(last, dict)
                else getattr(last, "content", str(last))  # live BaseMessage
            )
            st.markdown(content)
            return

        # Last resort: display raw output
        st.json({k: str(v) for k, v in output.items()}, expanded=False)


def _render_agent_output(node_name: str, state_update: Dict[str, Any]) -> None:
    """Render a live agent node's state update (expanded=True)."""
    _render_agent_panel(node_name, state_update, expanded=True)


def _render_stored_turn(turn: Dict[str, Any]) -> None:
    """Re-render a previously completed turn from chat_history."""
    with st.chat_message("user"):
        st.markdown(turn["content"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(f"Execution failed: {turn['error']}")
        agent_trace = turn.get("agent_trace", [])
        if agent_trace:
            if "error" in turn:
                st.divider()
            with st.expander("Agent trace", expanded=False):
                for agent_out in agent_trace:
                    _render_agent_panel(
                        agent_out["agent_id"],
                        agent_out["output"],
                        expanded=False,
                        use_expander=False,
                    )


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

    turn: Dict[str, Any] = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_index": len(st.session_state.get("chat_history", [])),
        "agent_trace": [],
    }

    with st.chat_message("user"):
        st.markdown(user_input)

    agent_trace: List[Dict[str, Any]] = []
    with st.chat_message("assistant"):
        with st.status("Running MAS...", expanded=True) as status:
            try:
                for node_name, state_update in executor.run_streaming(
                    input_data={"messages": [HumanMessage(content=user_input)]},
                    thread_id=st.session_state.chat_thread_id,
                ):
                    if state_update is None:
                        continue
                    try:
                        _render_agent_output(node_name, state_update)
                    except (
                        Exception
                    ) as render_exc:  # pylint: disable=broad-exception-caught
                        st.error(f"Agent {node_name} failed to render: {render_exc}")
                    agent_trace.append(
                        {
                            "agent_id": node_name,
                            "output": _serialize_state_update(state_update),
                        }
                    )
                status.update(label="Complete", state="complete", expanded=False)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                status.update(label="Execution failed", state="error")
                st.error(f"Execution failed: {exc}")
                turn["error"] = str(exc)

    turn["agent_trace"] = agent_trace

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
        load_error = st.session_state.get("chat_load_error")
        if load_error:
            st.error(f"Failed to load configuration:\n\n{load_error}")
        else:
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


def render_main() -> None:
    """Render the chat main area.

    Public: called by ``app.py`` when the Chat page is active.
    """
    _render_chat_area()


def render_page() -> None:
    """Main entry point — callable from app.py navigation or standalone."""
    _configure_page()
    with st.sidebar:
        _render_sidebar()
    render_main()


if __name__ == "__main__":
    render_page()
