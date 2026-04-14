"""
AETHER Attack GUI — Interactive Attack Suite.

Exposes the AETHER attack framework for manual exploratory testing and demos.
Users load a MAS config via "Send to Attack Suite" from the Chat or Visualizer
page, select a suite and payload (or write custom text), click a graph node to
target an agent, and click "Run Attack."

Pre-execution attacks stream token-by-token (same streaming pattern as
chat_app.py). Mid-execution attacks run synchronously with a spinner.
After a run, the graph re-renders with a propagation overlay:
  - Red border   → influenced by payload
  - Green border → received payload but resisted
  - Yellow border → received payload, clean output

Called by the main Streamlit app (``bili/streamlit_app.py``) as a page within
``st.navigation()``.
"""

# pylint: disable=import-error
import importlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from bili.aegis.attacks.injector import AttackInjector
from bili.aegis.attacks.models import AttackResult, AttackType, InjectionPhase
from bili.aegis.attacks.propagation import PropagationTracker
from bili.aegis.attacks.strategies import pre_execution as _pre_exec_strats
from bili.aegis.evaluator.evaluator_config import (
    FALLBACK_EVALUATOR_MODEL,
    FALLBACK_EVALUATOR_MODEL_DISPLAY,
    PRIMARY_EVALUATOR_MODEL,
    PRIMARY_EVALUATOR_MODEL_DISPLAY,
    PROVIDER_FAMILY_PREFIXES,
)
from bili.aether.runtime import MASExecutor
from bili.aether.schema import MASConfig
from bili.aether.ui.components.attack_graph import (
    build_node_states,
    render_attack_graph,
)

LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"
BASELINE_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "aegis"
    / "suites"
    / "baseline"
    / "results"
)
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"

# Batch runner paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SUITES_DIR = Path(__file__).resolve().parent.parent.parent / "aegis" / "suites"

LOGGER = logging.getLogger(__name__)

# Suppress the FileNotFoundError traceback for bootstrap.min.css.map
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(
    logging.ERROR
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVALUATOR_MODELS = {
    PRIMARY_EVALUATOR_MODEL_DISPLAY: PRIMARY_EVALUATOR_MODEL,
    FALLBACK_EVALUATOR_MODEL_DISPLAY: FALLBACK_EVALUATOR_MODEL,
}

_SUITE_NAMES = [
    "injection",
    "jailbreak",
    "memory_poisoning",
    "bias_inheritance",
    "agent_impersonation",
]

_SUITE_DISPLAY = {
    "injection": "Prompt Injection",
    "jailbreak": "Jailbreak",
    "memory_poisoning": "Memory Poisoning",
    "bias_inheritance": "Bias Inheritance",
    "agent_impersonation": "Agent Impersonation",
}

_SUITE_PAYLOAD_MODULES = {
    "injection": "bili.aegis.suites.injection.payloads.prompt_injection_payloads",
    "jailbreak": "bili.aegis.suites.jailbreak.payloads.jailbreak_payloads",
    "memory_poisoning": (
        "bili.aegis.suites.memory_poisoning.payloads.memory_poisoning_payloads"
    ),
    "bias_inheritance": (
        "bili.aegis.suites.bias_inheritance.payloads.bias_inheritance_payloads"
    ),
    "agent_impersonation": (
        "bili.aegis.suites.agent_impersonation.payloads.agent_impersonation_payloads"
    ),
}

_SUITE_ATTACK_TYPE = {
    "injection": "prompt_injection",
    "jailbreak": "jailbreak",
    "memory_poisoning": "memory_poisoning",
    "bias_inheritance": "bias_inheritance",
    "agent_impersonation": "agent_impersonation",
}

_SEV_PREFIX = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}

# Strategy function name for each attack_type string
_PRE_EXEC_STRATEGY_FN = {
    "prompt_injection": "inject_prompt_injection",
    "jailbreak": "inject_prompt_injection",
    "memory_poisoning": "inject_memory_poisoning",
    "bias_inheritance": "inject_bias_inheritance",
    "agent_impersonation": "inject_agent_impersonation",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def push_config_to_attack_state(config: MASConfig, yaml_path: str = "") -> None:
    """Write *config* into session state so the Attack page loads it fresh.

    Call this from any page that has a "Send to Attack Suite" button.  Clears
    previous attack results so the new config starts from a clean slate.

    Args:
        config:    The MASConfig to run attacks against.
        yaml_path: Absolute or repo-relative path to the YAML file — required
                   by ``run_suite`` for config fingerprinting.
    """
    st.session_state.attack_config = config
    st.session_state.attack_yaml_path = yaml_path
    if config.agents:
        st.session_state.attack_target_agent_id = config.agents[0].agent_id
    for key in ("attack_result", "attack_verdict", "attack_node_states"):
        st.session_state.pop(key, None)
    st.toast("Config loaded in Attack Suite \u2713")


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_attack_config():
    """Render YAML selector and sync resolved config into session state.

    When a file is explicitly selected, loads it and writes the result into
    ``attack_config`` / ``attack_yaml_path`` so that ``_render_main()`` can
    read the correct values from session state on the same render pass.
    """
    from bili.aether.config.loader import (  # pylint: disable=import-outside-toplevel
        load_mas_from_yaml,
    )

    yaml_files = sorted(EXAMPLES_DIR.glob("*.yaml")) if EXAMPLES_DIR.exists() else []
    yaml_display = ["(use config from AETHER visualizer)"] + [
        f.stem.replace("_", " ").title() for f in yaml_files
    ]

    selected_idx = st.selectbox(
        "YAML Configuration",
        range(len(yaml_display)),
        format_func=lambda i: yaml_display[i],
        key="attack_yaml_selector",
    )

    if selected_idx and selected_idx > 0:
        yaml_path_obj = yaml_files[selected_idx - 1]
        # Only reload if the selected file differs from what is currently loaded,
        # to avoid a full disk read and target-agent reset on every render.
        if st.session_state.get("attack_yaml_path") != str(yaml_path_obj):
            try:
                config = load_mas_from_yaml(str(yaml_path_obj))
                st.session_state.attack_config = config
                st.session_state.attack_yaml_path = str(yaml_path_obj)
                # Always reset target to first agent of the new config so
                # a stale agent_id from a previous config is never forwarded.
                if config.agents:
                    st.session_state.attack_target_agent_id = config.agents[0].agent_id
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"Failed to load `{yaml_path_obj.name}`: {exc}")


# ---------------------------------------------------------------------------
# Payload selection helpers
# ---------------------------------------------------------------------------


def _init_attack_selections() -> None:
    """Initialize per-payload session state keys to True if not already set."""
    for suite in _SUITE_NAMES:
        if f"attack_phase_{suite}" not in st.session_state:
            st.session_state[f"attack_phase_{suite}"] = "pre_execution"
        library = _load_payload_library(suite)
        for pid in library:
            key = f"attack_selected_{suite}_{pid}"
            if key not in st.session_state:
                st.session_state[key] = True


def _on_suite_header_change(hdr_key: str, payload_keys: list) -> None:
    """Propagate suite header checkbox state to all child payload checkboxes."""
    new_val = st.session_state[hdr_key]
    for pid in payload_keys:
        st.session_state[pid] = new_val


def _set_suite_payloads(payload_keys: list, value: bool) -> None:
    """Set all payloads in a suite to *value* (used by Select/Deselect All)."""
    for key in payload_keys:
        st.session_state[key] = value


def _render_payload_selector() -> None:
    """Render the two-level suite/payload selection hierarchy."""
    _init_attack_selections()

    st.markdown("**Select payloads to run:**")

    for suite in _SUITE_NAMES:
        library = _load_payload_library(suite)
        payload_keys = [f"attack_selected_{suite}_{pid}" for pid in library]
        n_selected = sum(1 for k in payload_keys if st.session_state.get(k, True))
        n_total = len(payload_keys)
        all_checked = n_selected == n_total and n_total > 0

        hdr_key = f"attack_suite_hdr_{suite}"
        st.session_state[hdr_key] = all_checked

        display = _SUITE_DISPLAY[suite]
        st.checkbox(
            f"**{display}** ({n_selected}/{n_total})",
            key=hdr_key,
            on_change=_on_suite_header_change,
            args=(hdr_key, payload_keys),
        )

        col_a, col_b, _ = st.columns([1, 1, 2])
        col_a.button(
            "Select All",
            key=f"atk_sel_{suite}",
            on_click=_set_suite_payloads,
            args=(payload_keys, True),
            use_container_width=True,
        )
        col_b.button(
            "Deselect All",
            key=f"atk_desel_{suite}",
            on_click=_set_suite_payloads,
            args=(payload_keys, False),
            use_container_width=True,
        )

        st.caption("Injection phase:")
        st.radio(
            "Injection phase",
            options=["pre_execution", "mid_execution"],
            format_func=lambda x: (
                "Pre-execution" if x == "pre_execution" else "Mid-execution"
            ),
            key=f"attack_phase_{suite}",
            horizontal=True,
            label_visibility="collapsed",
        )

        for pid, payload_obj in library.items():
            sev = getattr(payload_obj, "severity", "").lower()
            text_preview = getattr(payload_obj, "payload", "")
            preview = text_preview[:60] + ("\u2026" if len(text_preview) > 60 else "")
            sev_badge = _SEV_PREFIX.get(sev, "[---]")
            _, col_p = st.columns([0.05, 0.95])
            with col_p:
                st.checkbox(
                    f"{sev_badge} `{pid}` \u2014 {preview}",
                    key=f"attack_selected_{suite}_{pid}",
                )


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


def _execute_batch_attack(config, yaml_path: str, stub_mode: bool) -> None:
    """Run all selected payloads via run_suite() and display live progress."""
    if not yaml_path:
        st.error(
            "No YAML config path available. Select a YAML file in the sidebar "
            "or use **Send to Attack Suite** from the AETHER visualizer."
        )
        return

    from bili.aegis.evaluator.semantic_evaluator import (  # pylint: disable=import-outside-toplevel
        SemanticEvaluator,
    )
    from bili.aegis.suites._suite_runner import (  # pylint: disable=import-outside-toplevel
        run_suite,
    )

    skip_t3 = stub_mode or st.session_state.get("attack_skip_t3", False)
    semantic_evaluator = (
        SemanticEvaluator(model_name=_get_evaluator_model()) if not skip_t3 else None
    )
    baseline_results_dir = BASELINE_RESULTS_DIR if not skip_t3 else None

    # Collect selected payloads per suite
    suite_selections: dict[str, list] = {}
    for suite in _SUITE_NAMES:
        library = _load_payload_library(suite)
        selected = [
            obj
            for pid, obj in library.items()
            if st.session_state.get(f"attack_selected_{suite}_{pid}", True)
        ]
        if selected:
            suite_selections[suite] = selected

    if not suite_selections:
        st.warning("No payloads selected. Enable at least one payload.")
        return

    total_suites = len(suite_selections)
    progress_bar = st.progress(0, text="Starting batch attack run\u2026")
    status_area = st.container()
    passed_suites = 0

    for i, (suite, payloads) in enumerate(suite_selections.items()):
        progress_bar.progress(
            i / total_suites,
            text=f"Running {_SUITE_DISPLAY[suite]} ({i + 1}/{total_suites})\u2026",
        )
        phase = st.session_state.get(f"attack_phase_{suite}", "pre_execution")
        results_dir = _SUITES_DIR / suite / "results"
        try:
            run_suite(
                payloads=payloads,
                attack_suite=suite,
                attack_type=_SUITE_ATTACK_TYPE[suite],
                csv_filename=f"{suite}_results_matrix.csv",
                suite_name=_SUITE_DISPLAY[suite],
                results_dir=results_dir,
                repo_root=_REPO_ROOT,
                config_paths=[yaml_path] if yaml_path else [],
                phases=[phase],
                stub=stub_mode,
                semantic_evaluator=semantic_evaluator,
                baseline_results_dir=baseline_results_dir,
            )
            # run_suite() normally exits via sys.exit(); reaching here means it
            # returned normally (shouldn't happen, but treat as success).
            passed_suites += 1
            with status_area:
                st.markdown(
                    f"\u2705 **{_SUITE_DISPLAY[suite]}** — {len(payloads)} payload(s)"
                )
        except SystemExit as exc:
            # run_suite() is a CLI tool that always calls sys.exit() after writing
            # results to disk.  Exit code 0 = all cases passed; non-zero = failures.
            # Results are already on disk in either case.
            if exc.code == 0:
                passed_suites += 1
                with status_area:
                    st.markdown(
                        f"\u2705 **{_SUITE_DISPLAY[suite]}** — {len(payloads)} payload(s)"
                    )
            else:
                with status_area:
                    st.markdown(
                        f"\u26a0\ufe0f **{_SUITE_DISPLAY[suite]}** — "
                        f"{len(payloads)} payload(s) (some cases failed)"
                    )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Batch attack failed for suite %s: %s", suite, exc)
            with status_area:
                st.markdown(f"\u274c **{_SUITE_DISPLAY[suite]}** — failed: {exc}")

    progress_bar.progress(
        1.0, text=f"Complete \u2014 {passed_suites}/{total_suites} suite(s) passed"
    )
    if passed_suites == total_suites:
        st.success(f"All {total_suites} suite(s) completed.")
    else:
        st.warning(
            f"{passed_suites}/{total_suites} suite(s) completed "
            f"\u2014 {total_suites - passed_suites} failed."
        )
    st.info("Navigate to **Attack Results** in the sidebar to view your results.")


def render_attack_page() -> None:
    """Render the Attack page (sidebar + main area).

    Called by the unified Streamlit app after ``st.set_page_config()``
    has already been invoked.
    """
    with st.sidebar:
        _render_sidebar()
    _render_main()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render the Attack page sidebar."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)
    st.markdown("## AEGIS")
    st.caption("Adversarial Evaluation and Guarding of Intelligent Systems")
    st.markdown("---")
    st.markdown("#### Attack Suite")
    st.markdown(
        "Run batch attacks from the main area, or use the controls below for a "
        "single-payload exploratory attack against a specific graph node."
    )

    _resolve_attack_config()

    config: Optional[MASConfig] = st.session_state.get("attack_config")
    if config is None:
        return

    st.markdown("---")
    st.markdown("##### Single-payload exploratory attack")

    # Suite selector
    suite_display_names = [_SUITE_DISPLAY[s] for s in _SUITE_NAMES]
    suite_idx = st.selectbox(
        "Suite",
        range(len(_SUITE_NAMES)),
        format_func=lambda i: suite_display_names[i],
        key="attack_suite_idx",
        label_visibility="visible",
    )
    selected_suite = _SUITE_NAMES[suite_idx if suite_idx is not None else 0]
    st.session_state.attack_suite = selected_suite

    # Payload source
    payload_source = st.radio(
        "Payload source",
        ["Library", "Custom"],
        key="attack_payload_source",
        horizontal=True,
    )

    if payload_source == "Library":
        library = _load_payload_library(selected_suite)
        payload_ids = list(library.keys())
        if not payload_ids:
            st.warning("No payloads found for this suite.")
        else:
            selected_pid = st.selectbox(
                "Payload",
                payload_ids,
                format_func=lambda pid: f"{pid} — {_get_notes(library[pid])[:55]}",
                key="attack_payload_id",
            )
            if selected_pid:
                payload_obj = library[selected_pid]
                st.text_area(
                    "Payload preview",
                    value=getattr(payload_obj, "payload", ""),
                    height=100,
                    disabled=True,
                    key="attack_payload_preview",
                )
    else:
        st.text_area(
            "Custom payload",
            placeholder="Enter adversarial payload text\u2026",
            height=120,
            max_chars=10000,
            key="attack_payload_custom",
        )

    target = st.session_state.get("attack_target_agent_id")
    run_disabled = target is None

    if st.button(
        "Run Attack",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
        key="attack_run_button",
    ):
        _run_attack()

    if run_disabled:
        st.caption("Click a graph node to select a target.")

    st.markdown("---")
    st.markdown("#### Tier 3 Evaluator")
    st.selectbox(
        "Evaluator model",
        options=list(_EVALUATOR_MODELS.keys()),
        key="attack_evaluator_model_label",
        help=(
            "Model used for semantic (Tier 3) evaluation. "
            "Choose a model from a different provider family than your MAS "
            "to avoid circularity."
        ),
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------


def _render_main() -> None:
    """Render the Attack page main area."""
    config: Optional[MASConfig] = st.session_state.get("attack_config")
    yaml_path: str = st.session_state.get("attack_yaml_path", "")

    st.markdown("# AEGIS Attack Suite")
    st.markdown(
        "**AEGIS** (Adversarial Evaluation and Guarding of Intelligent Systems) "
        "is BiliCore's security testing framework for multi-agent systems built "
        "with **AETHER**. It provides systematic adversarial evaluation: inject "
        "attacks, track how they propagate through agent networks, and measure "
        "each agent's resilience using a 3-tier detection pipeline."
    )
    st.markdown(
        "Multi-agent systems introduce attack surfaces that don't exist in "
        "single-agent setups. A compromised agent can influence downstream "
        "agents through shared state, fabricated context, or manipulated "
        "communication channels. The Attack Suite lets you explore these "
        "risks interactively."
    )
    st.markdown("---")

    if config is None:
        st.info(
            "No MAS loaded.\n\n"
            "Select a YAML file in the sidebar, or use **Send to Attack Suite** from "
            "the AETHER Chat or Visualizer page to load a configuration."
        )
        return

    st.markdown(f"**Config:** `{config.mas_id}` — {config.name}")
    st.markdown(
        f"**Workflow:** {config.workflow_type.value} &nbsp;|&nbsp; "
        f"**Agents:** {len(config.agents)}"
    )
    st.markdown("---")

    # -----------------------------------------------------------------------
    # Batch attack — two-level payload selection
    # -----------------------------------------------------------------------
    st.markdown("## Batch Attack Run")
    st.markdown(
        "Select payloads across one or more suites and run them all at once. "
        "Results are written to disk and visible in the **Attack Results** viewer."
    )

    _render_payload_selector()

    st.markdown("---")

    # Stub mode and T3 options
    col_stub, col_t3, _ = st.columns([1, 1, 2])
    with col_stub:
        stub_mode = st.toggle(
            "Stub mode",
            value=False,
            key="attack_stub_mode",
            help="Skip LLM calls — useful for structural verification without API spend.",
        )
        if stub_mode:
            st.caption("No LLM calls")
    with col_t3:
        skip_t3 = st.toggle(
            "Skip T3 evaluation",
            value=False,
            key="attack_skip_t3",
            help="Skip Tier 3 semantic evaluation. Useful when baseline results are unavailable or to reduce API spend.",
            disabled=stub_mode,
        )
        if stub_mode or skip_t3:
            st.caption("T3 skipped")

    # Dynamic batch run button
    total_selected = sum(
        1
        for suite in _SUITE_NAMES
        for pid in _load_payload_library(suite)
        if st.session_state.get(f"attack_selected_{suite}_{pid}", True)
    )
    suites_with_selection = sum(
        1
        for suite in _SUITE_NAMES
        if any(
            st.session_state.get(f"attack_selected_{suite}_{pid}", True)
            for pid in _load_payload_library(suite)
        )
    )
    btn_label = (
        f"\u25b6 Run {total_selected} attack(s) across {suites_with_selection} suite(s)"
    )
    if st.button(
        btn_label,
        type="primary",
        use_container_width=True,
        key="attack_batch_run_button",
        disabled=(total_selected == 0),
    ):
        _execute_batch_attack(config, yaml_path, stub_mode)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Single-attack graph (used by sidebar Run Attack button)
    # -----------------------------------------------------------------------
    st.markdown("## Single-Agent Attack")
    st.markdown(
        "Click a node in the graph below to target a specific agent, then use "
        "**Run Attack** in the sidebar."
    )

    # Initialize target to first agent if not yet set
    if not st.session_state.get("attack_target_agent_id") and config.agents:
        st.session_state.attack_target_agent_id = config.agents[0].agent_id

    target_id: Optional[str] = st.session_state.get("attack_target_agent_id")
    phase: str = st.session_state.get("attack_phase", "pre_execution")

    st.caption(f"Target: `{target_id or 'None'}` | Phase: `{phase}`")

    # Graph — node clicks update attack_target_agent_id
    node_states: Optional[dict] = st.session_state.get("attack_node_states")
    clicked = render_attack_graph(config, target_id, node_states)
    if clicked and clicked != target_id:
        st.session_state.attack_target_agent_id = clicked
        st.rerun()

    # Results area
    attack_result_dict: Optional[dict] = st.session_state.get("attack_result")
    if attack_result_dict:
        st.markdown("---")
        _render_results(config, attack_result_dict)


# ---------------------------------------------------------------------------
# Attack execution
# ---------------------------------------------------------------------------


def _run_attack() -> None:
    """Dispatch to pre- or mid-execution handler, store result in session state."""
    config: Optional[MASConfig] = st.session_state.get("attack_config")
    if config is None:
        return

    target_id: Optional[str] = st.session_state.get("attack_target_agent_id")
    if not target_id:
        st.error("No target agent selected.")
        return

    phase: str = st.session_state.get("attack_phase", "pre_execution")
    suite: str = st.session_state.get("attack_suite", "injection")
    attack_type_str: str = _SUITE_ATTACK_TYPE[suite]
    payload_text: Optional[str] = _resolve_payload()

    if not payload_text:
        st.error("No payload text. Select a library payload or enter custom text.")
        return

    # Clear previous results
    for key in ("attack_result", "attack_verdict", "attack_node_states"):
        st.session_state.pop(key, None)

    try:
        if phase == "pre_execution":
            result = _run_pre_execution_streaming(
                config, target_id, attack_type_str, payload_text
            )
        else:
            result = _run_mid_execution(
                config, target_id, attack_type_str, payload_text
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Attack failed: {exc}")
        LOGGER.error("Attack execution error: %s", exc, exc_info=True)
        return

    result_dict = result.model_dump(mode="json")
    st.session_state.attack_result = result_dict
    st.session_state.attack_node_states = build_node_states(result.agent_observations)
    st.rerun()


def _run_pre_execution_streaming(
    config: MASConfig,
    target_agent_id: str,
    attack_type_str: str,
    payload_text: str,
) -> AttackResult:
    """Apply a pre-execution strategy then stream execution via MASExecutor.

    Follows the same streaming loop pattern as chat_app.py:
    pre-allocate one ``st.empty()`` slot per agent, accumulate tokens into
    the slot, then call ``PropagationTracker.observe()`` on each
    ``__node_complete__`` event.
    """
    strategy_fn_name = _PRE_EXEC_STRATEGY_FN[attack_type_str]
    strategy_fn = getattr(_pre_exec_strats, strategy_fn_name)
    patched_config = strategy_fn(config, target_agent_id, payload_text)

    executor = MASExecutor(patched_config)
    with st.spinner("Initializing attack executor…"):
        executor.initialize()

    agent_specs = {a.agent_id: a for a in patched_config.agents}
    agent_nodes_set = set(agent_specs)
    tracker = PropagationTracker(payload_text, target_agent_id)
    injected_at = datetime.now(timezone.utc)

    # Pre-allocate streaming slots (one per agent node)
    st.markdown("#### Agent Outputs")
    agent_slots: dict[str, Any] = {}
    for agent_id in [a.agent_id for a in patched_config.agents]:
        st.markdown(f"**{agent_id}**")
        agent_slots[agent_id] = st.empty()

    # token_buffer accumulates each agent's full streamed output so that the
    # st.empty() slot shows the complete text so far on each token event.
    # Memory usage is proportional to total output across all agents — acceptable
    # for research-scale MAS runs.  The buffer is released when this function
    # returns.
    token_buffer: dict[str, str] = {}

    try:
        for event_type, event_data in executor.run_streaming_tokens(
            input_data={"messages": []},
            thread_id=str(uuid.uuid4()),
        ):
            if event_type == "__token__":
                node = event_data["node"]
                if node not in agent_nodes_set:
                    continue
                token_buffer[node] = token_buffer.get(node, "") + event_data["token"]
                agent_slots[node].markdown(token_buffer[node])

            elif event_type == "__node_complete__":
                node = event_data["node"]
                state_update = event_data["state_update"]
                if node not in agent_nodes_set or state_update is None:
                    continue
                spec = agent_specs[node]
                messages = state_update.get("messages", [])
                output_excerpt = next(
                    (
                        getattr(m, "content", "")[:500]
                        for m in reversed(messages)
                        if getattr(m, "content", "")
                    ),
                    "",
                )
                tracker.observe(
                    agent_id=node,
                    role=spec.role or node,
                    input_state={
                        "messages": messages,
                        "objective": spec.objective or "",
                    },
                    output_state={"messages": messages, "output": output_excerpt},
                    attack_type=attack_type_str,
                )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error("Streaming execution error: %s", exc, exc_info=True)
        # Re-raise so _execute_attack shows the error and discards the partial result.
        raise

    completed_at = datetime.now(timezone.utc)
    return AttackResult(
        attack_id=str(uuid.uuid4()),
        mas_id=config.mas_id,
        target_agent_id=target_agent_id,
        attack_type=AttackType(attack_type_str),
        injection_phase=InjectionPhase.PRE_EXECUTION,
        payload=payload_text,
        injected_at=injected_at,
        completed_at=completed_at,
        propagation_path=tracker.propagation_path(),
        influenced_agents=tracker.influenced_agents(),
        resistant_agents=tracker.resistant_agents(),
        agent_observations=tracker.observations,
        success=True,
    )


def _run_mid_execution(
    config: MASConfig,
    target_agent_id: str,
    attack_type_str: str,
    payload_text: str,
) -> AttackResult:
    """Run a mid-execution injection synchronously via AttackInjector."""
    injector = AttackInjector(config=config, executor=None)
    with st.spinner("Running mid-execution attack…"):
        return injector.inject_attack(
            agent_id=target_agent_id,
            attack_type=attack_type_str,
            payload=payload_text,
            injection_phase=InjectionPhase.MID_EXECUTION,
        )


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def _render_results(config: MASConfig, result_dict: dict) -> None:
    """Render Tier 1, Tier 2, and Tier 3 results from the stored AttackResult dict."""
    st.markdown("### Attack Results")

    # Tier 1
    tier1_passed = result_dict.get("success", False)
    if tier1_passed:
        st.success("**Tier 1:** Execution succeeded — MAS ran without error.")
    else:
        error_msg = result_dict.get("error") or "Unknown error"
        st.error(f"**Tier 1:** Execution failed — {error_msg}")

    # Tier 2 — per-agent propagation
    observations = result_dict.get("agent_observations", [])
    if observations:
        st.markdown("#### Tier 2 — Propagation Analysis")
        for obs in observations:
            _render_observation(obs)

    propagation = result_dict.get("propagation_path", [])
    influenced = result_dict.get("influenced_agents", [])
    resistant = result_dict.get("resistant_agents", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Propagation path", " → ".join(propagation) if propagation else "—")
    col2.metric("Influenced", len(influenced))
    col3.metric("Resistant", len(resistant))

    # Tier 3 — semantic evaluation
    st.markdown("#### Tier 3 — Semantic Evaluation")
    if _is_stub_mode(config):
        st.info("Semantic evaluation skipped in stub mode.")
        return

    # Circularity check — warn if any agent shares the selected evaluator's provider family
    evaluator_model = _get_evaluator_model()
    mas_models = [a.model_name or "" for a in config.agents if a.model_name]
    if any(_same_provider_family(m, evaluator_model) for m in mas_models):
        st.warning(
            "⚠ Attack MAS and evaluator share the same model family — "
            "evaluation may be circular."
        )

    baseline_result = _load_baseline_result(config.mas_id)
    if baseline_result is None:
        st.info("No baseline results found for this MAS. Tier 3 evaluation skipped.")
        return

    with st.spinner("Running semantic evaluation…"):
        verdicts = _run_tier3_evaluation(result_dict, baseline_result)

    if verdicts is None:
        return

    verdict_dicts = [
        v.model_dump() if hasattr(v, "model_dump") else vars(v) for v in verdicts
    ]
    st.session_state.attack_verdict = verdict_dicts

    for vd in verdict_dicts:
        score = vd.get("score", -1)
        agent_id = vd.get("agent_id", "?")
        reasoning = vd.get("reasoning", "")
        confidence = vd.get("confidence", "")
        if score == -1:
            st.error(f"**{agent_id}:** Evaluation error — {vd.get('error', 'unknown')}")
        else:
            score_color = "🔴" if score >= 2 else ("🟡" if score == 1 else "🟢")
            st.markdown(
                f"{score_color} **{agent_id}:** Score {score}/3 "
                f"(confidence: {confidence})"
            )
            if reasoning:
                st.caption(reasoning)


def _render_observation(obs: dict) -> None:
    """Render a single AgentObservation as a color-coded status line + expander."""
    agent_id = obs.get("agent_id", "?")
    influenced = obs.get("influenced", False)
    resisted = obs.get("resisted", False)
    received = obs.get("received_payload", False)
    excerpt = obs.get("output_excerpt") or ""

    if influenced:
        icon = "🔴"
        label = "Influenced"
    elif resisted:
        icon = "🟢"
        label = "Resisted"
    elif received:
        icon = "🟡"
        label = "Received"
    else:
        icon = "⚪"
        label = "Clean"

    with st.expander(f"{icon} **{agent_id}** — {label}", expanded=False):
        st.caption(f"Role: {obs.get('role', '?')}")
        if excerpt:
            st.text_area(
                "Output excerpt",
                value=excerpt,
                height=80,
                disabled=True,
                label_visibility="collapsed",
                key=f"atk_obs_{agent_id}_{obs.get('influenced')}",
            )
        else:
            st.caption("(no output recorded)")


# ---------------------------------------------------------------------------
# Tier 3 evaluation
# ---------------------------------------------------------------------------


def _get_evaluator_model() -> str:
    """Return the model ID selected by the user for Tier 3 evaluation."""
    label = st.session_state.get("attack_evaluator_model_label")
    return _EVALUATOR_MODELS.get(label, PRIMARY_EVALUATOR_MODEL)


def _run_tier3_evaluation(result_dict: dict, baseline_result: dict) -> Optional[list]:
    """Reconstruct AttackResult and call SemanticEvaluator.evaluate()."""
    try:
        from bili.aegis.evaluator.semantic_evaluator import (  # pylint: disable=import-outside-toplevel
            SemanticEvaluator,
        )

        attack_result = AttackResult.model_validate(result_dict)
        evaluator = SemanticEvaluator(model_name=_get_evaluator_model())
        return evaluator.evaluate(baseline_result, attack_result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Tier 3 evaluation failed: {exc}")
        LOGGER.error("Tier 3 evaluation error: %s", exc, exc_info=True)
        return None


def _load_baseline_result(mas_id: str) -> Optional[dict]:
    """Load the first available baseline result for *mas_id*, or None.

    Prefers the most recent ``run_NNN`` subdirectory; falls back to the flat
    legacy layout when no versioned run directories exist.
    """
    from bili.aegis.suites._helpers import (  # pylint: disable=import-outside-toplevel
        latest_run_dir,
    )

    # Sanitize mas_id to prevent path traversal (e.g. "../../etc")
    safe_id = mas_id.replace("..", "").replace("/", "_").replace("\\", "_")
    mas_dir = BASELINE_RESULTS_DIR / safe_id
    run_dir = latest_run_dir(mas_dir)
    search_dir = run_dir if run_dir is not None else mas_dir
    if not search_dir.exists():
        return None
    for path in sorted(search_dir.glob("*.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Skipping unreadable baseline file %s: %s", path, exc)
            st.warning(f"Skipped unreadable baseline file `{path.name}`: {exc}")
            continue
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_payload_library(suite: str) -> dict:
    """Lazy-import and return PAYLOADS_BY_ID for the given suite."""
    try:
        mod = importlib.import_module(_SUITE_PAYLOAD_MODULES[suite])
        return mod.PAYLOADS_BY_ID
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning("Could not load payload library for suite %r: %s", suite, exc)
        return {}


def _get_notes(payload_obj: Any) -> str:
    """Return the notes string from a payload object."""
    return getattr(payload_obj, "notes", "") or ""


def _resolve_payload() -> Optional[str]:
    """Return the active payload text from session state."""
    source = st.session_state.get("attack_payload_source", "Library")
    if source == "Custom":
        return st.session_state.get("attack_payload_custom", "").strip() or None
    suite = st.session_state.get("attack_suite", "injection")
    pid = st.session_state.get("attack_payload_id")
    if not pid:
        return None
    library = _load_payload_library(suite)
    payload_obj = library.get(pid)
    if payload_obj is None:
        return None
    return getattr(payload_obj, "payload", None)


def _is_stub_mode(config: MASConfig) -> bool:
    """Return True if all agents have no model_name set (stub mode)."""
    return all(not a.model_name for a in config.agents)


def _get_provider_family(model_id: str) -> Optional[str]:
    """Return the canonical provider-family name for *model_id*, or None.

    Uses the same prefix table as ``evaluator_config.PROVIDER_FAMILY_PREFIXES``
    so circularity detection here stays in sync with the SemanticEvaluator.
    """
    lower = model_id.lower()
    for prefix, family in PROVIDER_FAMILY_PREFIXES:
        if lower.startswith(prefix):
            return family
    return None


def _same_provider_family(model_a: str, model_b: str) -> bool:
    """Return True if both model strings belong to the same provider family."""
    family_a = _get_provider_family(model_a)
    family_b = _get_provider_family(model_b)
    return family_a is not None and family_a == family_b
