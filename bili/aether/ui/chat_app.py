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
import os
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

from bili.aether.compiler import compile_mas
from bili.aether.config.loader import load_mas_from_dict, load_mas_from_yaml
from bili.aether.runtime import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.ui.styles.bili_core_theme import CUSTOM_CSS
from bili.aether.validation import validate_mas

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

LOGGER = logging.getLogger(__name__)

# streamlit-flow-component is optional in the chat tab.  When present the MAS
# structure panel renders an interactive flow graph; when absent it falls back
# to the text-based topology diagram.  No user-visible error is raised.
try:
    from streamlit_flow import streamlit_flow  # type: ignore
    from streamlit_flow.state import StreamlitFlowState  # type: ignore

    from bili.aether.ui.converters.yaml_to_graph import convert_mas_to_graph
    from bili.aether.ui.styles.node_styles import build_node_css

    _FLOW_AVAILABLE = True
except ImportError:
    _FLOW_AVAILABLE = False
    LOGGER.warning(
        "streamlit-flow-component not installed; MAS structure panel will use "
        "the text-based topology diagram fallback."
    )

# ---------------------------------------------------------------------------
# Flow graph node style constants
# ---------------------------------------------------------------------------
# DEFAULT_NODE_STYLE mirrors the base output of build_node_css() with the
# theme's default blue, so chat-graph nodes look identical to Visualizer nodes
# for roles that don't match any keyword rule.
#
# ACTIVE_NODE_STYLE overlays an orange border + box-shadow so the currently
# executing agent is visually distinct at the 300px graph height.  The orange
# (#F97316) matches the execution-timeline chip color already in use in the
# chat UI.  Background is kept dark so the border glow reads clearly against
# all role-specific fill colors.
_ACTIVE_NODE_STYLE: dict = {
    "background": "#1a1a1a",
    "color": "#F97316",
    "border": "3px solid #F97316",
    "borderRadius": "8px",
    "padding": "10px",
    "fontSize": "1rem",
    "width": "160px",
    "textAlign": "center",
    "boxShadow": "0 0 12px rgba(249,115,22,0.6)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_stub_config(config: MASConfig) -> bool:
    """Return True if any agent in *config* has no ``model_name`` (stub mode)."""
    return any(not agent.model_name for agent in config.agents)


def _base_cache_key() -> str:
    """Return the canonical cache key for the current config, stripping any model/stub suffix."""
    raw = st.session_state.get("chat_yaml_path", "config")
    return raw.split(":model=")[0].split(":stub")[0]


def _apply_model_patch(base_config: MASConfig, model_id: Optional[str]) -> None:
    """Patch every agent in *base_config* with *model_id* (``None`` = stub), then reinit.

    Warns when pipeline agents are present and *model_id* is not ``None``, because
    pipeline agents use internal node models and the top-level override may have no effect.
    Validates the patched config before reinitialising the executor.
    """
    if model_id is not None:
        pipeline_agents = [
            a.agent_id for a in base_config.agents if a.pipeline is not None
        ]
        if pipeline_agents:
            st.warning(
                f"Pipeline agents ({', '.join(pipeline_agents)}) use internal node "
                "models — the top-level model override may have no effect on them."
            )

    patched_agents = [
        a.model_copy(update={"model_name": model_id}) for a in base_config.agents
    ]
    patched = base_config.model_copy(update={"agents": patched_agents})
    if not _validate_config(patched):
        return

    base_key = _base_cache_key()
    cache_key = (
        f"{base_key}:model={model_id}" if model_id is not None else f"{base_key}:stub"
    )
    _initialize_executor(patched, cache_key)


def _active_messages() -> List[Dict]:
    """Return the messages list for the active thread.

    Returns a direct reference to the list stored in ``chat_threads`` so that
    in-place ``append`` calls update session state correctly.

    Raises:
        RuntimeError: If no active thread exists. Callers must ensure a thread
            is active (e.g. via ``_ensure_active_thread``) before appending.
    """
    thread_id = st.session_state.get("chat_thread_id")
    threads = st.session_state.get("chat_threads", {})
    if thread_id and thread_id in threads:
        return threads[thread_id]["messages"]
    raise RuntimeError(
        "_active_messages() called with no active thread. "
        "Call _ensure_active_thread() before appending."
    )


def _active_messages_or_empty() -> List[Dict]:
    """Return the active thread's messages or an empty list when no thread exists.

    Safe for read-only call sites (iteration, length checks) where no active
    thread is a valid state (e.g. before the first message is sent).
    """
    thread_id = st.session_state.get("chat_thread_id")
    threads = st.session_state.get("chat_threads", {})
    return threads.get(thread_id, {}).get("messages", []) if thread_id else []


def _new_thread(mas_id: str) -> str:
    """Create a new thread, set it active, and return the new thread ID."""
    now = datetime.now(timezone.utc)
    display_time = now.astimezone().strftime(
        "%H:%M:%S"
    )  # convert to local time for readability
    thread_id = str(uuid.uuid4())
    threads = st.session_state.setdefault("chat_threads", {})
    threads[thread_id] = {
        "name": f"{mas_id} \u2013 {display_time}",
        "messages": [],
        "mas_id": mas_id,
        "created_at": now.timestamp(),
    }
    st.session_state.chat_thread_id = thread_id
    return thread_id


def _ensure_active_thread(mas_id: str) -> None:
    """Create a thread if no active thread exists — called before first message send."""
    thread_id = st.session_state.get("chat_thread_id")
    if not thread_id or thread_id not in st.session_state.get("chat_threads", {}):
        _new_thread(mas_id)


def _delete_thread(thread_id: str) -> None:
    """Remove a thread; auto-switch to most recent remaining thread if it was active."""
    threads = st.session_state.get("chat_threads", {})
    threads.pop(thread_id, None)
    if st.session_state.get("chat_editing_thread") == thread_id:
        st.session_state.pop("chat_editing_thread", None)
    if st.session_state.get("chat_thread_id") == thread_id:
        if threads:
            newest_id = max(threads, key=lambda t: threads[t]["created_at"])
            st.session_state.chat_thread_id = newest_id
        else:
            st.session_state.pop("chat_thread_id", None)


def _render_thread_list() -> None:
    """Render the in-sidebar thread list with filter, rename, and delete controls."""
    threads: Dict[str, Any] = st.session_state.get("chat_threads", {})
    if not threads:
        return

    st.markdown("---")
    st.markdown("**Conversations**")
    st.caption("In-session only — threads are lost on page reload.")

    filter_text: str = st.text_input(
        "Filter conversations",
        key="chat_thread_filter",
        placeholder="Search…",
        label_visibility="collapsed",
    )

    active_id: Optional[str] = st.session_state.get("chat_thread_id")
    editing_id: Optional[str] = st.session_state.get("chat_editing_thread")

    sorted_threads = sorted(
        threads.items(), key=lambda x: x[1]["created_at"], reverse=True
    )
    if filter_text:
        sorted_threads = [
            (tid, t)
            for tid, t in sorted_threads
            if filter_text.lower() in t["name"].lower()
        ]

    for tid, thread in sorted_threads:
        is_active = tid == active_id

        if editing_id == tid:
            c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="center")
            with c1:
                new_name = st.text_input(
                    "Rename thread",
                    value=thread["name"],
                    key=f"chat_rename_input_{tid}",
                    label_visibility="collapsed",
                )
            with c2:
                if st.button("✓", key=f"chat_rename_confirm_{tid}", help="Save name"):
                    threads[tid]["name"] = new_name
                    st.session_state.pop("chat_editing_thread", None)
                    st.rerun()
            with c3:
                if st.button("✕", key=f"chat_rename_cancel_{tid}", help="Cancel"):
                    st.session_state.pop("chat_editing_thread", None)
                    st.rerun()
        else:
            c1, c2, c3 = st.columns([5, 1, 1], vertical_alignment="center")
            with c1:
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    thread["name"],
                    key=f"chat_select_thread_{tid}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.pop("chat_editing_thread", None)
                    if not is_active:
                        st.session_state.chat_thread_id = tid
                        st.rerun()
            with c2:
                if st.button("✏️", key=f"chat_edit_thread_{tid}", help="Rename"):
                    st.session_state.chat_editing_thread = tid
                    st.rerun()
            with c3:
                if st.button("🗑️", key=f"chat_delete_thread_{tid}", help="Delete"):
                    _delete_thread(tid)
                    st.rerun()


def _is_provider_available(provider_key: str) -> bool:
    """Return True if the environment has credentials for *provider_key*.

    Checks well-known environment variables and credential files for each
    remote provider.  Local providers are always available.  Unknown provider
    keys default to visible (fail-open) so new providers are not silently
    hidden.
    """
    checks: Dict[str, Any] = {
        "remote_openai": lambda: bool(os.environ.get("OPENAI_API_KEY")),
        "remote_azure_openai": lambda: bool(os.environ.get("AZURE_OPENAI_API_KEY")),
        "remote_aws_bedrock": lambda: (
            bool(os.environ.get("AWS_ACCESS_KEY_ID"))
            or bool(os.environ.get("AWS_SESSION_TOKEN"))
            or bool(os.environ.get("AWS_PROFILE"))
            or Path(os.path.expanduser("~/.aws/credentials")).exists()
        ),
        "remote_google_vertex": lambda: (
            bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
            or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        ),
        "local_llamacpp": lambda: True,
        "local_huggingface": lambda: True,
    }
    check = checks.get(provider_key)
    return check() if check is not None else True


@st.cache_data
def _load_all_model_options() -> tuple[list[str], list[str], list[str]]:
    """Return ``(display_list, model_id_list, provider_key_list)`` from LLM_MODELS.

    Lazy-imports ``LLM_MODELS`` so this module loads without bili-core's heavy
    LLM dependencies.  The full list is cached for the lifetime of the Streamlit
    process — provider availability filtering is intentionally NOT done here so
    that credentials added after startup are picked up on the next render.
    """
    from bili.config.llm_config import (  # pylint: disable=import-outside-toplevel
        LLM_MODELS,
    )

    display: list[str] = []
    ids: list[str] = []
    providers: list[str] = []
    for provider_key, provider_info in LLM_MODELS.items():
        label = provider_info["name"]
        for entry in provider_info["models"]:
            display.append(f"[{label}] {entry['model_name']}")
            ids.append(entry["model_id"])
            providers.append(provider_key)
    return display, ids, providers


def _build_chat_model_options() -> tuple[list[str], list[str]]:
    """Return ``(display_list, model_id_list)`` filtered to available providers.

    Calls the cached loader then filters by ``_is_provider_available`` on every
    render so that credentials added after startup are reflected immediately
    without requiring a process restart.
    """
    all_display, all_ids, all_providers = _load_all_model_options()
    display: list[str] = []
    ids: list[str] = []
    for disp, mid, pkey in zip(all_display, all_ids, all_providers):
        if _is_provider_available(pkey):
            display.append(disp)
            ids.append(mid)
    return display, ids


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

# Keys that get special treatment in _render_agent_panel (not dumped as k:v).
_AGENT_PANEL_SKIP_KEYS: frozenset = frozenset(
    {"agent_id", "raw", "role", "status", "message", "parsed"}
)

# Maps warning message substrings to plain-English explanations shown in a
# popover when the user clicks the warning chip.  Keys are matched via
# substring search so they remain stable as message wording evolves slightly.
_WARN_EXPLANATIONS: Dict[str, str] = {
    "has no channel connections": (
        "Channels are the explicit message routes between agents. "
        "This agent has none defined — it will still execute, but it won't "
        "send or receive inter-agent messages via the channel system. "
        "Sequential and hierarchical workflows route via graph edges rather "
        "than explicit channels, so this warning is often informational only."
    ),
    "should have 'inter_agent_communication' capability": (
        "Supervisor agents route tasks to worker agents and need the "
        "'inter_agent_communication' capability to do so effectively. "
        "Without it the supervisor can still run, but routing behaviour "
        "may be degraded."
    ),
    "has a separate reverse channel": (
        "A bidirectional channel already handles communication in both "
        "directions. The additional reverse channel defined here is redundant "
        "and could cause duplicate message delivery."
    ),
    "outgoing edges (expected 1 for a linear chain)": (
        "A sequential workflow expects each agent to have exactly one "
        "outgoing edge, forming a linear chain. Multiple outgoing edges "
        "create branching, which may not behave as expected in sequential mode."
    ),
}


def _warn_explanation(warn: str) -> Optional[str]:
    """Return a plain-English explanation for *warn*, or ``None`` if unknown."""
    for pattern, explanation in _WARN_EXPLANATIONS.items():
        if pattern in warn:
            return explanation
    return None


def _validate_config(config: MASConfig) -> bool:
    """Run structural validation and display results in the active Streamlit context.

    Returns ``True`` if the config is valid (errors list is empty).
    Warnings are displayed but do not block execution.  Each warning is rendered
    as a clickable popover chip so the user can read a plain-English explanation
    of what the warning means and whether action is required.
    """
    result = validate_mas(config)
    for err in result.errors:
        st.error(f"Config error: {err}")
    for warn in result.warnings:
        explanation = _warn_explanation(warn)
        if explanation:
            label = warn if len(warn) <= 60 else warn[:57] + "…"
            with st.popover(f"⚠️ {label}"):
                st.markdown(f"**{warn}**")
                st.markdown(explanation)
                st.caption("Warnings do not block execution.")
        else:
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
        # Clear active pointer only; chat_threads persists across config switches so
        # the user can re-activate a previous thread by clicking it in the list.
        # Render paths use _active_messages_or_empty() which safely returns [] when
        # no thread is active; _active_messages() (write path) requires a prior
        # _ensure_active_thread() call and will raise if this pointer is absent.
        st.session_state.pop("chat_thread_id", None)
        st.session_state.pop("chat_load_error", None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error("Failed to load config %s: %s", cache_key, exc, exc_info=True)
        st.session_state.chat_load_error = str(exc)
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)


def _load_config(yaml_path: Path) -> None:
    """Load a YAML config and initialize the executor when the path changes."""
    # If the current executor was already initialized from this YAML (possibly
    # with a model-picker suffix applied by _apply_model_patch), skip the reload
    # to avoid overwriting the patched executor on each Streamlit rerun.
    current_key = st.session_state.get("chat_yaml_path", "")
    base = str(yaml_path)
    if (
        current_key == base or current_key.startswith(f"{base}:")
    ) and "chat_config" in st.session_state:
        return

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
    st.session_state.chat_config_base = config
    _initialize_executor(config, str(yaml_path))


def _load_uploaded_config(name: str, config: MASConfig) -> None:
    """Initialize the executor from an already-parsed uploaded MASConfig."""
    # Same guard as _load_config: skip if the current executor is already based
    # on this uploaded config (possibly with a model-picker suffix).
    current_key = st.session_state.get("chat_yaml_path", "")
    base_key = f"uploaded:{name}"
    if (
        current_key == base_key or current_key.startswith(f"{base_key}:")
    ) and "chat_config" in st.session_state:
        return

    if not _validate_config(config):
        for key in ("chat_config", "chat_yaml_path", "chat_executor"):
            st.session_state.pop(key, None)
        return
    st.session_state.chat_config_base = config
    _initialize_executor(config, base_key)


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


def _extract_content(output: Dict[str, Any]) -> str:
    """Extract a readable content string from a serialized agent output dict.

    Mirrors the display priority of ``_render_agent_panel`` but returns a plain
    string suitable for text-based exports. Unlike the render panel (which
    receives a single agent_id), this function collects *all* agent_outputs
    entries and joins them so no entry is silently dropped.
    """
    agent_outputs: Dict[str, Any] = output.get("agent_outputs", {})
    if agent_outputs:
        collected: List[str] = []
        for inner in agent_outputs.values():
            if inner.get("status") == "stub":
                collected.append(inner.get("message", str(inner)))
            elif inner.get("message"):
                collected.append(inner["message"])
            else:
                kv_parts = [f"{k}: {v}" for k, v in inner.items() if k != "agent_id"]
                if kv_parts:
                    collected.append(" | ".join(kv_parts))
        if collected:
            return "\n".join(collected)
    messages: List[Any] = output.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return last.get("content", str(last))
        return getattr(last, "content", str(last))
    return str(output)


def _build_markdown_export(
    config: MASConfig, thread_id: str, chat_history: List[Dict]
) -> str:
    """Build a Markdown export string for the active conversation."""
    lines = [
        f"# Conversation — {config.mas_id}",
        f"Thread: {thread_id}",
        f"Exported: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for i, turn in enumerate(chat_history, 1):
        lines += [
            f"## Turn {i}",
            "",
            # turn['content'] is inserted verbatim; any markdown the user typed
            # (headings, links, etc.) will render as-is in the exported file.
            # This is acceptable since users are exporting their own conversations.
            f"**User:** {turn['content']}",
            "",
            "**MAS Response:**",
            "",
        ]
        for agent_out in turn.get("agent_trace", []):
            content = _extract_content(agent_out.get("output", {}))
            # Prefix every line so multi-line content is fully blockquoted.
            content_lines = content.replace("\n", "\n> ")
            lines.append(f"> **{agent_out['agent_id']}:** {content_lines}")
        lines.append("")
    return "\n".join(lines)


def render_sidebar_content(examples_dir: Optional[Path] = None) -> None:
    """Render the chat sidebar controls — config selector, upload, and buttons.

    Public: called by ``app.py`` when the Chat page is active.

    Args:
        examples_dir: Override the directory scanned for built-in YAML configs.
            When ``app.py`` imports this module as a package the module-level
            ``EXAMPLES_DIR`` may resolve to the installed site-packages copy
            (which lacks the example files).  Pass ``app.py``'s own
            ``EXAMPLES_DIR`` to guarantee the correct source-tree path.
    """
    effective_examples_dir = examples_dir if examples_dir is not None else EXAMPLES_DIR

    if not effective_examples_dir.exists():
        st.error(f"Examples directory not found: {effective_examples_dir}")
        return

    # Build option lists once — reused by both the autoload block and the selectbox.
    yaml_files = sorted(effective_examples_dir.glob("*.yaml"))
    uploaded_configs: Dict[str, MASConfig] = st.session_state.get(
        "chat_uploaded_configs", {}
    )
    uploaded_names = sorted(uploaded_configs.keys())

    autoload_name = st.session_state.pop("chat_autoload_name", None)
    if autoload_name is not None:
        pending = uploaded_configs.get(autoload_name)
        if pending is not None:
            _load_uploaded_config(autoload_name, pending)
            # Pre-select the config in the selectbox so the selectbox's else-branch
            # does not immediately clear the session state we just populated.
            if autoload_name in uploaded_names:
                st.session_state.chat_yaml_selector = len(
                    yaml_files
                ) + uploaded_names.index(autoload_name)

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
                # Graph-level validation: catches circular edges, missing
                # entry_point, and other structural issues beyond Pydantic parsing.
                compile_mas(upload_config)
                if "chat_uploaded_configs" not in st.session_state:
                    st.session_state.chat_uploaded_configs = {}
                st.session_state.chat_uploaded_configs[uploaded.name] = upload_config
                st.success(f"Uploaded: {uploaded.name}")
                # Auto-activate the uploaded config so the stub indicator (and
                # model picker) appear immediately without a manual dropdown pick.
                st.session_state.chat_autoload_name = uploaded.name
                st.rerun()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Invalid config: {exc}")

    # Re-read uploaded_configs after the uploader so newly uploaded files
    # appear in the selectbox on the same rerun.
    uploaded_configs = st.session_state.get("chat_uploaded_configs", {})
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
        # User cleared the selectbox — leave all session state intact so that
        # the loaded config, executor, and conversation threads remain visible.
        return

    config: Optional[MASConfig] = st.session_state.get("chat_config")
    if config is None:
        return

    st.markdown("---")

    # Placeholder at the correct visual position — filled AFTER the button
    # handlers below so that clicking "Apply to all" or "Stub mode" clears
    # or shows the warning in the same script run.  Reading chat_config here
    # (before _apply_model_patch runs) would always reflect the pre-click state.
    stub_warning_ph = st.empty()

    st.markdown(f"**MAS ID:** `{config.mas_id}`")
    st.markdown(f"**Workflow:** {config.workflow_type.value}")
    st.markdown(f"**Agents:** {len(config.agents)}")

    st.markdown("---")
    st.markdown("**Model Settings**")

    display_opts, id_opts = _build_chat_model_options()
    st.selectbox(
        "Model",
        range(len(display_opts)),
        format_func=lambda i: display_opts[i],
        key="chat_model_selector",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        apply_clicked = st.button(
            "Apply to all",
            use_container_width=True,
            help="Set this model on every agent",
        )
    with col2:
        stub_clicked = st.button(
            "Stub mode",
            use_container_width=True,
            help="Clear model from all agents",
        )

    base_config: Optional[MASConfig] = st.session_state.get("chat_config_base")
    if base_config is not None:
        if apply_clicked:
            model_idx = st.session_state.get("chat_model_selector", 0)
            if model_idx < 0 or model_idx >= len(id_opts):
                st.error("Invalid model selection. Please re-select a model.")
            else:
                _apply_model_patch(base_config, id_opts[model_idx])

        if stub_clicked:
            _apply_model_patch(base_config, None)

    # Fill the placeholder now that any model patch has been applied — uses the
    # current (possibly just updated) chat_config so the flag is accurate.
    if _is_stub_config(st.session_state.get("chat_config", config)):
        stub_warning_ph.info("Stub mode — no LLM calls will be made.", icon="⚙️")

    st.markdown("---")

    if st.button("New Conversation", use_container_width=True):
        _new_thread(config.mas_id)
        for _key in (
            "aether_executing_node",
            "aether_execution_trace",
            "aether_selected_trace_node",
        ):
            st.session_state.pop(_key, None)
        st.rerun()

    chat_history = _active_messages_or_empty()
    thread_id = st.session_state.get("chat_thread_id", "")
    if chat_history:
        base_name = f"aether_chat_{config.mas_id}_{thread_id[:8] or 'unknown'}"
        export_json = {
            "mas_id": config.mas_id,
            "thread_id": thread_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "turns": [
                # Explicitly include agent_trace so the key is always present,
                # even for turns stored before this field was introduced.
                {**turn, "agent_trace": turn.get("agent_trace", [])}
                for turn in chat_history
            ],
        }
        col_json, col_md = st.columns(2)
        with col_json:
            st.download_button(
                "Export JSON",
                data=json.dumps(export_json, indent=2),
                file_name=f"{base_name}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_md:
            st.download_button(
                "Export Markdown",
                data=_build_markdown_export(config, thread_id, chat_history),
                file_name=f"{base_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    _render_thread_list()


# ---------------------------------------------------------------------------
# MAS structure panel
# ---------------------------------------------------------------------------


def _agent_card(agent: AgentSpec) -> None:
    """Render a single agent as a small card (agent_id + optional role)."""
    if agent.role != agent.agent_id:
        st.markdown(
            f"`{agent.agent_id}`  \n<small>{agent.role}</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"`{agent.agent_id}`")


def _render_sequential_diagram(config: MASConfig) -> None:
    """Flat left-to-right chain: agent_a → agent_b → agent_c."""
    n = len(config.agents)
    if n == 0:
        return
    # Alternating layout: [agent, →, agent, →, …] = 2N-1 columns.
    widths = [1 if i % 2 == 1 else 3 for i in range(2 * n - 1)]
    cols = st.columns(widths)
    agent_col_indices = list(range(0, 2 * n - 1, 2))
    for idx, agent in zip(agent_col_indices, config.agents):
        with cols[idx]:
            _agent_card(agent)
        if idx + 1 < len(cols):
            with cols[idx + 1]:
                st.markdown("→")


def _render_supervisor_diagram(config: MASConfig) -> None:
    """Hub-and-spoke: coordinator on the left, specialists stacked on the right."""
    try:
        coordinator = config.get_entry_agent()
    except (ValueError, IndexError):
        coordinator = config.agents[0] if config.agents else None

    if coordinator is None:
        _render_sequential_diagram(config)
        return

    specialists = [a for a in config.agents if a.agent_id != coordinator.agent_id]
    if not specialists:
        _render_sequential_diagram(config)
        return

    col_coord, col_arrows, col_specs = st.columns([1, 0.3, 2])
    with col_coord:
        _agent_card(coordinator)
    with col_arrows:
        for _ in specialists:
            st.markdown("→")
    with col_specs:
        for spec in specialists:
            _agent_card(spec)


def _render_consensus_diagram(config: MASConfig) -> None:
    """Parallel agents feeding into a central [consensus] label."""
    if not config.agents:
        return
    col_agents, col_arrow, col_result = st.columns([2, 0.3, 1])
    with col_agents:
        for agent in config.agents:
            _agent_card(agent)
    with col_arrow:
        for _ in config.agents:
            st.markdown("→")
    with col_result:
        st.markdown("**[consensus]**")


def _render_hierarchical_diagram(config: MASConfig) -> None:
    """Top-down tree grouped by tier (tier 1 = root)."""
    if not config.agents:
        return

    # Group agents by tier; agents without a tier go into tier 1
    tier_map: dict = {}
    for agent in config.agents:
        t = getattr(agent, "tier", None) or 1
        tier_map.setdefault(t, []).append(agent)

    sorted_tiers = sorted(tier_map.keys())
    for i, tier in enumerate(sorted_tiers):
        tier_agents = tier_map[tier]
        cols = st.columns(len(tier_agents))
        for col, agent in zip(cols, tier_agents):
            with col:
                _agent_card(agent)
        # Connector row between tiers
        if i < len(sorted_tiers) - 1:
            connector_cols = st.columns(len(tier_agents))
            for col in connector_cols:
                with col:
                    st.markdown("↓")


def _render_fallback_diagram(config: MASConfig) -> None:
    """Labeled agent + channel list for custom/unknown workflow types."""
    if config.agents:
        agent_list = ", ".join(
            f"`{a.agent_id}`" + (f" ({a.role})" if a.role != a.agent_id else "")
            for a in config.agents
        )
        st.markdown(f"**Agents:** {agent_list}")
    if config.channels:
        channel_lines = "  \n".join(
            f"`{c.source}` → `{c.target}` ({c.protocol.value})"
            for c in config.channels
            if c.source and c.target
        )
        if channel_lines:
            st.markdown(f"**Channels:**  \n{channel_lines}")


def _render_mas_diagram(config: MASConfig) -> None:
    """Dispatch to the topology-specific diagram renderer."""
    workflow = config.workflow_type
    if workflow == WorkflowType.SEQUENTIAL:
        _render_sequential_diagram(config)
    elif workflow == WorkflowType.SUPERVISOR:
        _render_supervisor_diagram(config)
    elif workflow == WorkflowType.CONSENSUS:
        _render_consensus_diagram(config)
    elif workflow == WorkflowType.HIERARCHICAL:
        _render_hierarchical_diagram(config)
    else:
        _render_fallback_diagram(config)


def _render_chat_agent_details(agent: AgentSpec) -> None:
    """Render read-only agent details below the flow graph.

    Called when a node is selected in the chat-tab flow graph.  Intentionally
    omits model-override controls — those live exclusively in the sidebar.
    """
    with st.container(border=True):
        name = agent.get_display_name()
        st.markdown(f"**{name}** &nbsp; `{agent.agent_id}`")
        if agent.objective:
            st.markdown(f"**Objective:** {agent.objective}")
        if agent.model_name:
            st.markdown(f"**Model:** `{agent.model_name}`")
        if agent.temperature is not None:
            st.markdown(f"**Temperature:** {agent.temperature}")
        if agent.max_tokens:
            st.markdown(f"**Max tokens:** {agent.max_tokens}")
        if agent.capabilities:
            caps = " ".join(f"`{c}`" for c in agent.capabilities)
            st.markdown(f"**Capabilities:** {caps}")
        if agent.tools:
            tools = " ".join(f"`{t}`" for t in agent.tools)
            st.markdown(f"**Tools:** {tools}")


@st.fragment
def _render_flow_graph(config: MASConfig) -> None:
    """Render the 300px interactive flow graph for the chat-tab MAS panel.

    Uses distinct session-state keys (``chat_flow_state_*`` /
    ``chat_mas_graph_*``) so the chat and Visualizer tabs never share graph
    state.

    Active-node highlighting: checks ``aether_executing_node`` in session state
    each render and applies ``_ACTIVE_NODE_STYLE`` to the matching node so the
    currently running agent is visually distinct during execution.  The style
    clears automatically when the streaming loop pops the key.

    Node-click details: after ``streamlit_flow()`` returns, ``selected_id`` is
    checked and ``_render_chat_agent_details`` renders agent info inline below
    the graph.  No side column is used — details appear in a bordered container
    within the same expander.
    """
    flow_key = f"chat_mas_graph_{config.mas_id}"
    state_key = f"chat_flow_state_{config.mas_id}"

    # Build nodes with active-node highlight applied per render so the style
    # tracks execution state without requiring a state rebuild.
    executing = st.session_state.get("aether_executing_node")

    if state_key not in st.session_state:
        nodes, edges = convert_mas_to_graph(config)
        for node in nodes:
            node.style = build_node_css(
                next(
                    (a.role for a in config.agents if a.agent_id == node.id),
                    node.id,
                )
            )
        st.session_state[state_key] = StreamlitFlowState(nodes, edges)

    # Only rebuild the flow state when active-node highlighting changes
    # (during execution). Rebuilding every render creates new object
    # identities that the streamlit-flow component interprets as state
    # changes, causing an infinite rerun loop.
    prev_executing = st.session_state.get(f"{state_key}_prev_executing")
    if executing != prev_executing:
        st.session_state[f"{state_key}_prev_executing"] = executing
        nodes, edges = convert_mas_to_graph(config)
        for node in nodes:
            if node.id == executing:
                node.style = _ACTIVE_NODE_STYLE
            else:
                node.style = build_node_css(
                    next(
                        (a.role for a in config.agents if a.agent_id == node.id),
                        node.id,
                    )
                )
        existing = st.session_state[state_key]
        st.session_state[state_key] = StreamlitFlowState(
            nodes, edges, selected_id=existing.selected_id
        )

    st.session_state[state_key] = streamlit_flow(
        key=flow_key,
        state=st.session_state[state_key],
        height=300,
        fit_view=True,
        show_controls=False,
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

    selected_id = st.session_state[state_key].selected_id
    if selected_id:
        agent = config.get_agent(selected_id)
        if agent:
            _render_chat_agent_details(agent)
    else:
        st.caption("Click a node to view agent details.")


def _render_mas_structure(config: MASConfig) -> None:
    """Render a collapsible MAS topology panel above the chat history.

    When streamlit-flow-component is installed, renders an interactive 300px
    flow graph with active-node highlighting and node-click details.
    Falls back to the text-based topology diagram when the dependency is absent.

    Defaults to expanded when no messages exist, collapses once the
    conversation starts so vertical space is given to the chat history.
    """
    messages = _active_messages_or_empty()
    with st.expander("MAS Structure", expanded=(len(messages) == 0)):
        if _FLOW_AVAILABLE:
            _render_flow_graph(config)
        else:
            n_channels = len(config.channels)
            st.markdown(
                f"**Workflow:** `{config.workflow_type.value}` &nbsp;·&nbsp; "
                f"**Agents:** {len(config.agents)} &nbsp;·&nbsp; "
                f"**Channels:** {n_channels}"
            )
            st.divider()
            _render_mas_diagram(config)


# ---------------------------------------------------------------------------
# Agent output rendering
# ---------------------------------------------------------------------------


def _render_agent_panel(
    agent_id: str,
    output: Dict[str, Any],
    *,
    expanded: bool,
    use_expander: bool = True,
    role: Optional[str] = None,
) -> None:
    """Render one agent's output inside an expander or plain container.

    Accepts both raw graph state updates (containing ``BaseMessage`` objects)
    and already-serialized output dicts stored in ``chat_history``.

    When *use_expander* is ``False`` the panel is rendered as a ``st.container``
    with a bold header — suitable for use inside an outer expander to avoid
    nesting expanders inside expanders.

    Args:
        role: Human-readable role string for the agent. When set and different
            from ``agent_id``, the label format is ``"{role} — {agent_id}"``.
    """
    panel_label = f"{role} — {agent_id}" if role and role != agent_id else agent_id
    if use_expander:
        ctx = st.expander(panel_label, expanded=expanded)
    else:
        ctx = st.container()

    with ctx:
        if not use_expander:
            st.markdown(f"**{panel_label}**")

        # Prefer the structured agent_outputs entry for this node
        agent_outputs: Dict[str, Any] = output.get("agent_outputs", {})
        if agent_id in agent_outputs:
            inner = agent_outputs[agent_id]
            if inner.get("status") == "stub":
                st.caption(inner.get("message", str(inner)))
            else:
                # Metadata caption (role + status on one line)
                meta = []
                if inner.get("role"):
                    meta.append(inner["role"])
                if inner.get("status"):
                    meta.append(inner["status"])
                if meta:
                    st.caption(" · ".join(meta))

                # Main content — the agent's response
                message = inner.get("message", "")
                if message:
                    st.markdown(message)

                # Structured JSON output (if the agent produced parsed data)
                parsed = inner.get("parsed")
                if parsed:
                    st.json(parsed, expanded=False)

                # Any remaining fields not handled above
                extra = {
                    k: v for k, v in inner.items() if k not in _AGENT_PANEL_SKIP_KEYS
                }
                for key, value in extra.items():
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


def _render_stored_turn(turn: Dict[str, Any]) -> None:
    """Re-render a previously completed turn from chat_history."""
    cfg: Optional[MASConfig] = st.session_state.get("chat_config")
    role_map = _build_role_map(cfg) if cfg else {}
    with st.chat_message("user", avatar="👤"):
        st.markdown(turn["content"])
    with st.chat_message("assistant", avatar="🤖"):
        if "error" in turn:
            st.error(f"Execution failed: {turn['error']}")
        agent_trace = turn.get("agent_trace", [])
        LOGGER.debug(
            "Rendering stored turn: %d agents: %s",
            len(agent_trace),
            [a["agent_id"] for a in agent_trace],
        )
        if agent_trace:
            # Deduplicate while preserving order — supervisor workflows can
            # repeat agent IDs, which would cause Streamlit widget key collisions.
            node_ids = list(dict.fromkeys(a["agent_id"] for a in agent_trace))
            turn_idx = turn.get("turn_index", 0)
            timeline_ph = st.empty()
            _render_timeline(
                timeline_ph,
                completed=node_ids,
                active=None,
                all_nodes=node_ids,
                key_prefix=f"timeline_stored_{turn_idx}",
                turn_index=turn_idx,
                role_map=role_map,
            )
            # Only consume the selection when it belongs to this turn — the
            # tuple ``(turn_index, agent_id)`` lets each stored turn check
            # ownership before popping, so earlier turns in the render loop
            # don't accidentally steal a click intended for a later turn.
            raw = st.session_state.get("aether_selected_trace_node")
            if isinstance(raw, tuple) and raw[0] == turn_idx:
                st.session_state.pop("aether_selected_trace_node")
                selected = raw[1]
            else:
                selected = None
            if "error" in turn:
                st.divider()

            # Show the last agent's response as the visible summary so
            # the user sees the final output without expanding the trace.
            last_agent_id = agent_trace[-1]["agent_id"]
            last_agent_data = (
                agent_trace[-1]["output"]
                .get("agent_outputs", {})
                .get(last_agent_id, {})
            )
            last_message = last_agent_data.get("message", "")
            if last_message:
                st.markdown(last_message)

            # Per-agent expanders — each agent's output is clearly
            # separated.  Clicking a timeline chip auto-expands the
            # matching agent.
            for agent_out in agent_trace:
                _render_agent_panel(
                    agent_out["agent_id"],
                    agent_out["output"],
                    expanded=(agent_out["agent_id"] == selected),
                    role=role_map.get(agent_out["agent_id"]),
                )


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------


def _build_role_map(config: MASConfig) -> Dict[str, str]:
    """Map agent_id → role for display labels; omits entries where role == agent_id."""
    return {a.agent_id: a.role for a in config.agents if a.role != a.agent_id}


def _render_timeline(
    placeholder: Any,
    completed: List[str],
    active: Optional[str],
    all_nodes: List[str],
    *,
    key_prefix: str,
    turn_index: int = 0,
    role_map: Optional[Dict[str, str]] = None,
    status_text: Optional[str] = None,
) -> None:
    """Render a horizontal node-chip row into a ``st.empty()`` placeholder.

    Subsequent calls to this function replace the placeholder content in-place,
    allowing the timeline to update live during streaming without re-rendering
    the whole page.

    Completed chips are enabled buttons — clicking one stores
    ``(turn_index, agent_id)`` in ``aether_selected_trace_node`` so
    ``_render_stored_turn`` can identify the correct turn and auto-expand that
    agent's panel on the next rerun.  Active and pending chips are disabled.

    Args:
        placeholder: A ``st.empty()`` container whose content is replaced on
            each call.
        completed: Agent IDs that have already finished (in order).
        active: Agent ID currently executing, or ``None``.
        all_nodes: Ordered list of all agent IDs for this config — determines
            chip order and which nodes show as pending (○).
        key_prefix: Unique prefix for Streamlit widget keys. Use a counter
            suffix for the live turn (``f"timeline_live_{n}"``) and
            ``f"timeline_stored_{turn_index}"`` for stored turns to avoid
            key collisions across multiple rendered turns.
        turn_index: The ``turn_index`` of the owning turn. Stored alongside
            ``agent_id`` in ``aether_selected_trace_node`` so the correct
            stored turn can consume the selection.
        role_map: Optional mapping of agent_id → role for chip display labels.
            When provided, chips show the role string instead of the raw agent_id.
        status_text: Optional short status caption rendered below the chip row
            (e.g. ``"⟳ Running writer..."``). Pass ``None`` to hide.
    """
    if not all_nodes:
        return
    _role_map = role_map or {}
    completed_set = set(completed)
    with placeholder.container():
        cols = st.columns(len(all_nodes))
        for col, node_id in zip(cols, all_nodes):
            label = _role_map.get(node_id, node_id)
            with col:
                if node_id in completed_set:
                    if st.button(
                        f"✓ {label}",
                        key=f"{key_prefix}_{node_id}",
                        use_container_width=True,
                        help="Click to expand this agent's output",
                    ):
                        st.session_state["aether_selected_trace_node"] = (
                            turn_index,
                            node_id,
                        )
                        st.rerun()
                elif node_id == active:
                    st.button(
                        f"⟳ {label}",
                        key=f"{key_prefix}_{node_id}",
                        disabled=True,
                        use_container_width=True,
                    )
                else:
                    st.button(
                        f"○ {label}",
                        key=f"{key_prefix}_{node_id}",
                        disabled=True,
                        use_container_width=True,
                    )
        if status_text:
            st.caption(status_text)


def _run_turn(user_input: str) -> None:
    """Execute one conversation turn and append it to chat_history."""
    executor: Optional[MASExecutor] = st.session_state.get("chat_executor")
    if executor is None:
        st.error("No executor available. Select a configuration first.")
        return

    config: Optional[MASConfig] = st.session_state.get("chat_config")
    if config is None:
        st.error("No configuration loaded.")
        return
    _ensure_active_thread(config.mas_id)

    turn: Dict[str, Any] = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_index": len(_active_messages()),
        "agent_trace": [],
    }

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    all_nodes = [a.agent_id for a in config.agents]
    role_map = _build_role_map(config)
    agent_trace: List[Dict[str, Any]] = []
    with st.chat_message("assistant", avatar="🤖"):
        st.session_state["aether_execution_trace"] = []
        st.session_state.pop("aether_selected_trace_node", None)
        timeline_placeholder = st.empty()
        # Each _render_timeline call within this run must use a unique key_prefix
        # because Streamlit registers widget keys globally for the entire script
        # execution — placeholder.container() replaces visual content but does
        # not de-register previously created widget keys.
        _tl_call = 0
        _render_timeline(
            timeline_placeholder,
            completed=[],
            active=None,
            all_nodes=all_nodes,
            key_prefix=f"timeline_live_{_tl_call}",
            role_map=role_map,
            status_text="⟳ Starting MAS...",
        )
        _tl_call += 1
        # Pre-allocate one st.empty() slot per agent in config order.
        # Tokens stream into the active slot as they arrive; a __node_complete__
        # event overwrites the slot with the full structured panel.  This
        # eliminates the need to re-render all previous agents on every update
        # and avoids the DeltaGenerator context-corruption race condition.
        agent_nodes_set = set(all_nodes)
        agent_slots: Dict[str, Any] = {node_id: st.empty() for node_id in all_nodes}
        token_buffer: Dict[str, str] = {}
        hitl_interrupt: Optional[Dict[str, Any]] = None
        try:
            for event_type, event_data in executor.run_streaming_tokens(
                input_data={"messages": [HumanMessage(content=user_input)]},
                thread_id=st.session_state.chat_thread_id,
            ):
                if event_type == "__human_interrupt__":
                    hitl_interrupt = event_data
                    break

                if event_type == "__token__":
                    node_name = event_data["node"]
                    token = event_data["token"]
                    if node_name not in agent_nodes_set:
                        continue  # skip internal routing / pipeline sub-nodes
                    # Update timeline only when the active node changes to avoid
                    # an unnecessary widget-key churn on every token.
                    if st.session_state.get("aether_executing_node") != node_name:
                        st.session_state["aether_executing_node"] = node_name
                        display_name = role_map.get(node_name, node_name)
                        _render_timeline(
                            timeline_placeholder,
                            completed=st.session_state["aether_execution_trace"],
                            active=node_name,
                            all_nodes=all_nodes,
                            key_prefix=f"timeline_live_{_tl_call}",
                            role_map=role_map,
                            status_text=f"⟳ Running {display_name}...",
                        )
                        _tl_call += 1
                    token_buffer[node_name] = token_buffer.get(node_name, "") + token
                    slot = agent_slots.get(node_name)
                    if slot is not None:
                        slot.markdown(token_buffer[node_name])

                elif event_type == "__node_complete__":
                    node_name = event_data["node"]
                    state_update = event_data["state_update"]
                    LOGGER.debug(
                        "Node complete: %s (agent=%s)",
                        node_name,
                        node_name in agent_nodes_set,
                    )
                    if state_update is None or node_name not in agent_nodes_set:
                        continue
                    serialized = _serialize_state_update(state_update)
                    agent_trace.append({"agent_id": node_name, "output": serialized})
                    st.session_state["aether_execution_trace"].append(node_name)
                    token_buffer.pop(node_name, None)
                    # Overwrite the streaming slot with the full structured panel.
                    slot = agent_slots.get(node_name)
                    if slot is not None:
                        try:
                            with slot.container():
                                _render_agent_panel(
                                    node_name,
                                    serialized,
                                    expanded=True,
                                    role=role_map.get(node_name),
                                )
                        except (
                            Exception  # pylint: disable=broad-exception-caught
                        ) as render_exc:
                            slot.error(
                                f"Agent {node_name} failed to render: {render_exc}"
                            )
                    # Clear the active node and advance the timeline to show
                    # this node as completed before the next node begins.
                    st.session_state.pop("aether_executing_node", None)
                    _render_timeline(
                        timeline_placeholder,
                        completed=st.session_state["aether_execution_trace"],
                        active=None,
                        all_nodes=all_nodes,
                        key_prefix=f"timeline_live_{_tl_call}",
                        role_map=role_map,
                    )
                    _tl_call += 1

        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Execution failed: {exc}")
            turn["error"] = str(exc)

        st.session_state.pop("aether_executing_node", None)
        _render_timeline(
            timeline_placeholder,
            completed=st.session_state["aether_execution_trace"],
            active=None,
            all_nodes=all_nodes,
            key_prefix=f"timeline_live_{_tl_call}",
            role_map=role_map,
        )

    if hitl_interrupt:
        # Graph paused for human review.  Save partial turn state so the
        # resume handler in _render_chat_area() can complete the turn.
        turn["agent_trace"] = agent_trace
        st.session_state["hitl_pending"] = {
            **hitl_interrupt,
            "partial_turn": turn,
        }
        # Rerun to show the HITL form (chat_input will be hidden).
        st.rerun()

    turn["agent_trace"] = agent_trace
    LOGGER.info(
        "Turn complete: %d agents in trace: %s",
        len(agent_trace),
        [a["agent_id"] for a in agent_trace],
    )
    # Use list reassignment rather than .append() so Streamlit's change
    # detector sees the mutation and the stored turn is immediately visible
    # on the rerun that follows.
    thread_id = st.session_state["chat_thread_id"]
    st.session_state["chat_threads"][thread_id]["messages"] = st.session_state[
        "chat_threads"
    ][thread_id]["messages"] + [turn]
    # One rerun after the stream is fully exhausted — safe here because both
    # st.empty() placeholders (timeline + outputs) have already committed their
    # final state to the frontend before this point.
    st.rerun()


# ---------------------------------------------------------------------------
# Human-in-the-loop resume form
# ---------------------------------------------------------------------------


def _render_hitl_form(executor: "MASExecutor") -> None:
    """Render the human-review form and resume graph execution on submit.

    Called by ``_render_chat_area()`` when ``hitl_pending`` is set in session
    state (i.e. after a ``__human_interrupt__`` sentinel was received from
    ``run_streaming()``).
    """
    pending: Dict[str, Any] = st.session_state["hitl_pending"]
    next_nodes: List[str] = pending.get("next", [])
    thread_id: str = pending["thread_id"]
    partial_turn: Dict[str, Any] = pending["partial_turn"]

    config: Optional[MASConfig] = st.session_state.get("chat_config")
    all_nodes = [a.agent_id for a in config.agents] if config else []
    role_map = _build_role_map(config) if config else {}

    node_label = ", ".join(role_map.get(n, n) for n in next_nodes)
    st.info(f"⏸ Human review required before: **{node_label}**")

    with st.form("hitl_resume_form"):
        human_response = st.text_area(
            "Your review / decision:",
            key="hitl_human_input",
            help="This response will be injected as a message before the paused node resumes.",
        )
        submitted = st.form_submit_button("Resume execution", type="primary")

    if not submitted or not human_response:
        return

    # --- Resume execution ---
    agent_trace: List[Dict[str, Any]] = list(partial_turn.get("agent_trace", []))
    completed_ids = [a["agent_id"] for a in agent_trace]

    with st.chat_message("assistant", avatar="🤖"):
        st.session_state["aether_execution_trace"] = completed_ids[:]
        timeline_placeholder = st.empty()
        _tl_call = 0
        _render_timeline(
            timeline_placeholder,
            completed=st.session_state["aether_execution_trace"],
            active=None,
            all_nodes=all_nodes,
            key_prefix=f"timeline_hitl_resume_{_tl_call}",
            role_map=role_map,
        )
        _tl_call += 1
        outputs_placeholder = st.empty()
        try:
            for node_name, state_update in executor.resume_streaming(
                human_response, thread_id
            ):
                if state_update is None:
                    continue
                st.session_state["aether_executing_node"] = node_name
                display_name = role_map.get(node_name, node_name)
                _render_timeline(
                    timeline_placeholder,
                    completed=st.session_state["aether_execution_trace"],
                    active=node_name,
                    all_nodes=all_nodes,
                    key_prefix=f"timeline_hitl_resume_{_tl_call}",
                    role_map=role_map,
                    status_text=f"⟳ Running {display_name}...",
                )
                _tl_call += 1
                agent_trace.append(
                    {
                        "agent_id": node_name,
                        "output": _serialize_state_update(state_update),
                    }
                )
                st.session_state["aether_execution_trace"].append(node_name)
                with outputs_placeholder.container():
                    for trace_entry in agent_trace:
                        try:
                            _render_agent_panel(
                                trace_entry["agent_id"],
                                trace_entry["output"],
                                expanded=True,
                                role=role_map.get(trace_entry["agent_id"]),
                            )
                        except (
                            Exception  # pylint: disable=broad-exception-caught
                        ) as render_exc:
                            st.error(
                                f"Agent {trace_entry['agent_id']} failed to render:"
                                f" {render_exc}"
                            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Execution failed: {exc}")
            partial_turn["error"] = str(exc)

        st.session_state.pop("aether_executing_node", None)
        _render_timeline(
            timeline_placeholder,
            completed=st.session_state["aether_execution_trace"],
            active=None,
            all_nodes=all_nodes,
            key_prefix=f"timeline_hitl_resume_{_tl_call}",
            role_map=role_map,
        )

    partial_turn["agent_trace"] = agent_trace
    thread_id = st.session_state["chat_thread_id"]
    st.session_state["chat_threads"][thread_id]["messages"] = st.session_state[
        "chat_threads"
    ][thread_id]["messages"] + [partial_turn]
    st.session_state.pop("hitl_pending", None)
    st.rerun()


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

    # MAS structure panel must remain first — do not move below stored turns.
    # The flow graph is a fixed orientational anchor; chat history grows below it.
    _render_mas_structure(config)

    for turn in _active_messages_or_empty():
        _render_stored_turn(turn)

    # If a human-in-the-loop interrupt is pending, show the review form
    # instead of the normal chat input.
    executor: Optional[MASExecutor] = st.session_state.get("chat_executor")
    if st.session_state.get("hitl_pending") and executor is not None:
        _render_hitl_form(executor)
        return

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
