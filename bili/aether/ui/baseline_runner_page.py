"""
AETHER Baseline Runner page — launch AEGIS baseline suite from the GUI.

Allows a user to push a MAS config from the AETHER visualizer and run
selected baseline prompts against it with live per-prompt progress.  Results
are written to the same directory read by the Baseline Results viewer
(``bili/aegis/suites/baseline/results/``).

Called by the main Streamlit app (``bili/streamlit_app.py``) as a page within
``st.navigation()``.
"""

import copy
import logging
from pathlib import Path

import streamlit as st

from bili.aether.schema import MASConfig

LOGGER = logging.getLogger(__name__)

LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "config" / "examples"

_CATEGORY_ICON = {
    "benign": "🟢",
    "violating": "🔴",
    "edge_case": "🟡",
}
_CATEGORY_ORDER = ["benign", "edge_case", "violating"]
_CATEGORY_LABELS = {
    "benign": "Benign",
    "edge_case": "Edge Case",
    "violating": "Violating",
}


# ---------------------------------------------------------------------------
# Session state helper — called from AETHER visualizer
# ---------------------------------------------------------------------------


def push_config_to_baseline_state(config: MASConfig, yaml_path: str) -> None:
    """Write *config* into session state so the Baseline Runner page loads it.

    Call this from any page that has a "Send to Baseline" button.  Clears
    previous run results so the new config starts from a clean slate.

    Args:
        config:    The MASConfig to run baselines against.
        yaml_path: Absolute or repo-relative path to the YAML file — required
                   by ``run_one`` → ``yaml_hash`` (reads file for SHA-256).
    """
    st.session_state.baseline_config = config
    st.session_state.baseline_yaml_path = yaml_path
    for key in ("baseline_run_results",):
        st.session_state.pop(key, None)
    st.toast("Config loaded in Baseline Runner \u2713")


# ---------------------------------------------------------------------------
# Prompt selection helpers
# ---------------------------------------------------------------------------


def _init_prompt_selections() -> None:
    """Initialize per-prompt session state keys to True if not already set."""
    from bili.aegis.suites.baseline.prompts.baseline_prompts import (  # pylint: disable=import-outside-toplevel
        BASELINE_PROMPTS,
    )

    for p in BASELINE_PROMPTS:
        key = f"baseline_prompt_{p.prompt_id}"
        if key not in st.session_state:
            st.session_state[key] = True


def _on_cat_header_change(hdr_key: str, prompt_ids: list) -> None:
    """Propagate category header checkbox state to all child prompt checkboxes."""
    new_val = st.session_state[hdr_key]
    for pid in prompt_ids:
        st.session_state[f"baseline_prompt_{pid}"] = new_val


def _set_cat_prompts(prompt_ids: list, value: bool) -> None:
    """Set all prompts in a category to *value* (used by Select/Deselect All)."""
    for pid in prompt_ids:
        st.session_state[f"baseline_prompt_{pid}"] = value


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render_baseline_runner_page() -> None:
    """Render the Baseline Runner page (sidebar + main area).

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
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)
    st.markdown("## AEGIS")
    st.caption("Adversarial Evaluation and Guarding of Intelligent Systems")
    st.markdown("---")
    st.markdown("#### Baseline Runner")
    st.markdown(
        "Run baseline prompts against any AETHER MAS config. "
        "Results are saved to disk and visible in the **Baseline Results** viewer.\n\n"
        "Load a config from the **AETHER Multi-Agent System** page and click "
        "**Send to Baseline**, or select a YAML file below."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _render_main() -> None:
    _init_prompt_selections()

    st.markdown("# AEGIS Baseline Runner")
    st.markdown(
        "Run the baseline prompt suite against a MAS configuration to establish "
        "ground-truth outputs for each agent. Baseline results are required before "
        "running attack suites — they serve as the control group for Tier-3 semantic "
        "evaluation."
    )
    st.markdown("---")

    config, yaml_path = _resolve_config()

    if config is None:
        st.info(
            "No config loaded. Select a YAML file above, or load a config in the "
            "**AETHER Multi-Agent System** visualizer and click **Send to Baseline**."
        )
        return

    st.markdown(f"**Config:** `{config.mas_id}` — {config.name}")
    st.markdown(
        f"**Workflow:** {config.workflow_type.value} &nbsp;|&nbsp; "
        f"**Agents:** {len(config.agents)}"
    )
    st.markdown("---")

    stub_mode = _render_prompt_selector()

    st.markdown("---")

    # Total count label above Run button
    from bili.aegis.suites.baseline.prompts.baseline_prompts import (  # pylint: disable=import-outside-toplevel
        BASELINE_PROMPTS,
    )

    total = len(BASELINE_PROMPTS)
    n_sel = sum(
        1
        for p in BASELINE_PROMPTS
        if st.session_state.get(f"baseline_prompt_{p.prompt_id}", True)
    )
    if n_sel == total:
        st.caption(f"{total} prompts selected")
    else:
        st.caption(f"{n_sel} of {total} prompts selected")

    if st.button(
        "▶ Run Baseline",
        type="primary",
        use_container_width=True,
        key="run_baseline_btn",
        disabled=(n_sel == 0),
    ):
        _execute_run(config, yaml_path, stub_mode)

    _render_previous_results()


def _resolve_config():
    """Return (config, yaml_path) from session state or file selector."""
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
        key="baseline_yaml_selector",
    )

    if selected_idx and selected_idx > 0:
        # User explicitly picked a file — takes precedence over session state
        yaml_path_obj = yaml_files[selected_idx - 1]
        try:
            config = load_mas_from_yaml(str(yaml_path_obj))
            return config, str(yaml_path_obj)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Failed to load `{yaml_path_obj.name}`: {exc}")
            return None, None

    # Fall back to session state (pushed from visualizer)
    config = st.session_state.get("baseline_config")
    yaml_path = st.session_state.get("baseline_yaml_path", "")
    return config, yaml_path


def _render_prompt_selector() -> bool:
    """Render the two-level prompt selection hierarchy and stub toggle.

    Returns:
        stub_mode (bool): Whether LLM calls should be skipped.
    """
    from bili.aegis.suites.baseline.prompts.baseline_prompts import (  # pylint: disable=import-outside-toplevel
        BASELINE_PROMPTS,
    )

    # Stub mode toggle (global setting, shown above the prompt list)
    col_stub, _ = st.columns([1, 2])
    stub_mode = col_stub.toggle(
        "Stub mode",
        value=False,
        key="baseline_stub_mode",
        help="Skip LLM calls — useful for structural verification without API spend.",
    )
    if stub_mode:
        col_stub.caption("No LLM calls")

    st.markdown("**Select prompts to run:**")

    for cat in _CATEGORY_ORDER:
        cat_prompts = [p for p in BASELINE_PROMPTS if p.category == cat]
        prompt_ids = [p.prompt_id for p in cat_prompts]
        icon = _CATEGORY_ICON.get(cat, "⚪")
        label = _CATEGORY_LABELS[cat]

        n_selected = sum(
            1
            for p in cat_prompts
            if st.session_state.get(f"baseline_prompt_{p.prompt_id}", True)
        )
        n_total = len(cat_prompts)
        all_checked = n_selected == n_total

        # Force-sync header to match current child state (bidirectional sync)
        hdr_key = f"baseline_cat_hdr_{cat}"
        st.session_state[hdr_key] = all_checked

        # Category header checkbox
        st.checkbox(
            f"{icon} **{label}** ({n_selected}/{n_total})",
            key=hdr_key,
            on_change=_on_cat_header_change,
            args=(hdr_key, prompt_ids),
        )

        # Select All / Deselect All convenience buttons
        col_a, col_b, _ = st.columns([1, 1, 3])
        col_a.button(
            "Select All",
            key=f"bl_sel_{cat}",
            on_click=_set_cat_prompts,
            args=(prompt_ids, True),
            use_container_width=True,
        )
        col_b.button(
            "Deselect All",
            key=f"bl_desel_{cat}",
            on_click=_set_cat_prompts,
            args=(prompt_ids, False),
            use_container_width=True,
        )

        # Individual prompt checkboxes (indented)
        for p in cat_prompts:
            preview = p.text[:60] + ("\u2026" if len(p.text) > 60 else "")
            _, col_prompt = st.columns([0.05, 0.95])
            with col_prompt:
                st.checkbox(
                    f"`{p.prompt_id}` \u2014 {preview}",
                    key=f"baseline_prompt_{p.prompt_id}",
                )

    return stub_mode


def _execute_run(config, yaml_path: str, stub_mode: bool) -> None:
    """Run the selected baseline prompts and display live progress."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.suites.baseline.prompts.baseline_prompts import BASELINE_PROMPTS
    from bili.aegis.suites.baseline.run_baseline import run_one, write_result

    prompts_to_run = [
        p
        for p in BASELINE_PROMPTS
        if st.session_state.get(f"baseline_prompt_{p.prompt_id}", True)
    ]

    if not prompts_to_run:
        st.warning("No prompts selected. Enable at least one prompt.")
        return

    # Deep-copy so stub mode does not mutate the session state config
    run_config = copy.deepcopy(config)
    if stub_mode:
        for agent in run_config.agents:
            agent.model_name = None

    results = []
    progress_bar = st.progress(0, text="Starting baseline run\u2026")
    status_area = st.container()

    passed = 0
    for i, prompt in enumerate(prompts_to_run):
        icon = _CATEGORY_ICON.get(prompt.category, "\u26aa")
        progress_bar.progress(
            i / len(prompts_to_run),
            text=f"Running {prompt.prompt_id} ({i + 1}/{len(prompts_to_run)})\u2026",
        )

        try:
            result = run_one(run_config, yaml_path, prompt, stub_mode=stub_mode)
            write_result(result)
            success = result["execution"]["success"]
            duration = result["execution"]["duration_ms"]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Baseline run failed for %s: %s", prompt.prompt_id, exc)
            result = None
            success = False
            duration = 0.0

        status_icon = "\u2705" if success else "\u274c"
        if success:
            passed += 1

        with status_area:
            st.markdown(
                f"{status_icon} {icon} `{prompt.prompt_id}` "
                f"({_CATEGORY_LABELS.get(prompt.category, prompt.category)}) "
                f"\u2014 {duration:.0f} ms"
            )

        if result:
            results.append(result)

    total = len(prompts_to_run)
    progress_bar.progress(1.0, text=f"Complete \u2014 {passed}/{total} passed")

    st.session_state.baseline_run_results = results

    if passed == total:
        st.success(f"All {total} prompts passed.")
    else:
        failed = total - passed
        st.warning(f"{passed}/{total} passed \u2014 {failed} failed.")

    st.info("Navigate to **Baseline Results** in the sidebar to view your results.")


def _render_previous_results() -> None:
    """Show a summary of the most recent run stored in session state."""
    results = st.session_state.get("baseline_run_results")
    if not results:
        return

    st.markdown("---")
    st.markdown("#### Last Run Summary")
    total = len(results)
    passed = sum(1 for r in results if r["execution"]["success"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Prompts Run", total)
    col2.metric("Passed", passed)
    col3.metric("Success Rate", f"{passed / total * 100:.0f}%" if total else "N/A")
    st.info("Navigate to **Baseline Results** in the sidebar to view your results.")
