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

# Suppress the FileNotFoundError traceback for bootstrap.min.css.map
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(
    logging.ERROR
)

from bili.aether.attacks.injector import AttackInjector
from bili.aether.attacks.models import AttackResult, AttackType, InjectionPhase
from bili.aether.attacks.propagation import PropagationTracker
from bili.aether.attacks.strategies import pre_execution as _pre_exec_strats
from bili.aether.runtime import MASExecutor
from bili.aether.schema import MASConfig
from bili.aether.ui.components.attack_graph import (
    build_node_states,
    render_attack_graph,
)

LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"
BASELINE_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "baseline" / "results"
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVALUATOR_PRIMARY_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

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
    "injection": "bili.aether.tests.injection.payloads.prompt_injection_payloads",
    "jailbreak": "bili.aether.tests.jailbreak.payloads.jailbreak_payloads",
    "memory_poisoning": (
        "bili.aether.tests.memory_poisoning.payloads.memory_poisoning_payloads"
    ),
    "bias_inheritance": (
        "bili.aether.tests.bias_inheritance.payloads.bias_inheritance_payloads"
    ),
    "agent_impersonation": (
        "bili.aether.tests.agent_impersonation.payloads.agent_impersonation_payloads"
    ),
}

_SUITE_ATTACK_TYPE = {
    "injection": "prompt_injection",
    "jailbreak": "jailbreak",
    "memory_poisoning": "memory_poisoning",
    "bias_inheritance": "bias_inheritance",
    "agent_impersonation": "agent_impersonation",
}

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


def push_config_to_attack_state(config: MASConfig) -> None:
    """Write *config* into session state so the Attack page loads it fresh.

    Call this from any page that has a "Send to Attack Suite" button.  Clears
    previous attack results so the new config starts from a clean slate.
    """
    st.session_state.attack_config = config
    if config.agents:
        st.session_state.attack_target_agent_id = config.agents[0].agent_id
    for key in ("attack_result", "attack_verdict", "attack_node_states"):
        st.session_state.pop(key, None)
    st.toast("Config loaded in Attack Suite \u2713")


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
    st.markdown("## Attack Suite")
    st.markdown("---")

    config: Optional[MASConfig] = st.session_state.get("attack_config")
    if config is None:
        st.caption("No MAS loaded.")
        return

    st.caption(f"Config: `{config.mas_id}`")

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
            return

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
            placeholder="Enter adversarial payload text…",
            height=120,
            key="attack_payload_custom",
        )

    # Phase
    st.radio(
        "Injection phase",
        ["pre_execution", "mid_execution"],
        key="attack_phase",
        format_func=lambda p: (
            "Pre-execution" if p == "pre_execution" else "Mid-execution"
        ),
        horizontal=True,
    )

    st.markdown("---")

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


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------


def _render_main() -> None:
    """Render the Attack page main area."""
    config: Optional[MASConfig] = st.session_state.get("attack_config")

    if config is None:
        st.markdown("# Attack Suite")
        st.info(
            "No MAS loaded.\n\n"
            "Use **Send to Attack Suite** from the Chat or Visualizer page to "
            "load a configuration."
        )
        return

    # Initialize target to first agent if not yet set
    if not st.session_state.get("attack_target_agent_id") and config.agents:
        st.session_state.attack_target_agent_id = config.agents[0].agent_id

    target_id: Optional[str] = st.session_state.get("attack_target_agent_id")
    phase: str = st.session_state.get("attack_phase", "pre_execution")

    st.markdown("# Attack Suite")
    st.caption(
        f"Config: `{config.mas_id}` | "
        f"Target: `{target_id or 'None'}` | "
        f"Phase: `{phase}`"
    )
    st.markdown("---")

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

    # Circularity check — warn if any agent shares the evaluator's provider family
    mas_models = [a.model_name or "" for a in config.agents if a.model_name]
    if any(_same_provider_family(m, _EVALUATOR_PRIMARY_MODEL) for m in mas_models):
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


def _run_tier3_evaluation(result_dict: dict, baseline_result: dict) -> Optional[list]:
    """Reconstruct AttackResult and call SemanticEvaluator.evaluate()."""
    try:
        from bili.aether.evaluator.semantic_evaluator import (  # pylint: disable=import-outside-toplevel
            SemanticEvaluator,
        )

        attack_result = AttackResult.model_validate(result_dict)
        evaluator = SemanticEvaluator(model_name=_EVALUATOR_PRIMARY_MODEL)
        return evaluator.evaluate(baseline_result, attack_result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Tier 3 evaluation failed: {exc}")
        LOGGER.error("Tier 3 evaluation error: %s", exc, exc_info=True)
        return None


def _load_baseline_result(mas_id: str) -> Optional[dict]:
    """Load the first available baseline result for *mas_id*, or None."""
    mas_dir = BASELINE_RESULTS_DIR / mas_id
    if not mas_dir.exists():
        return None
    for path in sorted(mas_dir.glob("**/*.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # pylint: disable=broad-exception-caught
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


def _same_provider_family(model_a: str, model_b: str) -> bool:
    """Return True if both model strings belong to the same provider family."""
    families = [
        {"anthropic", "claude"},
        {"google", "gemini", "vertex"},
        {"amazon", "nova", "bedrock"},
        {"openai", "gpt"},
    ]
    for family in families:
        if any(k in model_a.lower() for k in family) and any(
            k in model_b.lower() for k in family
        ):
            return True
    return False
