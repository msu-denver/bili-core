"""
AETHER Results page — Baseline Results Viewer.

Loads JSON result files written by the baseline runner and renders an
interactive summary matrix, category breakdowns, and per-result detail panels.

Called by the main Streamlit app (``bili/streamlit_app.py``) as a page within
``st.navigation()``.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)

BASELINE_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "baseline" / "results"
)
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

_CATEGORY_ICON = {
    "benign": "🟢",
    "violating": "🔴",
    "edge_case": "🟡",
}

_CATEGORY_ORDER = ["benign", "edge_case", "violating"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_baseline_results() -> list[dict]:
    """Return all parsed JSON result dicts from the baseline results directory."""
    results: list[dict] = []
    for path in sorted(BASELINE_RESULTS_DIR.glob("**/*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Could not parse %s: %s", path, exc)
    return results


def _build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten result dicts into a tidy DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "mas_id": r["mas_id"],
                "prompt_id": r["prompt_id"],
                "category": r["prompt_category"],
                "success": r["execution"]["success"],
                "duration_ms": r["execution"]["duration_ms"],
                "agent_count": r["execution"]["agent_count"],
                "stub_mode": r["run_metadata"]["stub_mode"],
                "timestamp": r["run_metadata"]["timestamp"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render_results_page() -> None:
    """Render the Results page (sidebar + main area).

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
    st.markdown("## AETHER Results")
    st.markdown("---")
    st.caption("Baseline Evaluation Viewer")
    st.markdown(
        "Generate results by running the baseline suite:\n\n"
        "```\n# Stub mode (no LLM calls)\n"
        "python bili/aether/tests/baseline/run_baseline.py --stub\n\n"
        "# Full run (requires API credentials)\n"
        "python bili/aether/tests/baseline/run_baseline.py\n```"
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------


def _render_main() -> None:
    st.markdown("# Baseline Results")
    st.markdown(
        "Structured outputs from the AETHER baseline evaluation suite. "
        "Each cell represents one prompt run against one MAS configuration."
    )
    st.markdown("---")

    results = _load_baseline_results()

    if not results:
        st.info(
            "No baseline results found.\n\n"
            "Run the baseline suite to populate this view:\n\n"
            "```\npython bili/aether/tests/baseline/run_baseline.py --stub\n```"
        )
        return

    df = _build_dataframe(results)
    _render_summary_metrics(df)
    st.markdown("---")
    df_filtered = _render_filters(df)
    st.markdown("---")
    _render_matrix(df_filtered)
    st.markdown("---")
    _render_detail_panel(results, df_filtered)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------


def _render_summary_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    passed = int(df["success"].sum())
    cols = st.columns(5)
    cols[0].metric("Configs", df["mas_id"].nunique())
    cols[1].metric("Prompts", df["prompt_id"].nunique())
    cols[2].metric("Total Runs", total)
    cols[3].metric("Passed", passed)
    cols[4].metric("Success Rate", f"{passed / total * 100:.0f}%")


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter controls and return the filtered DataFrame."""
    col1, col2, col3 = st.columns(3)
    with col1:
        configs = st.multiselect(
            "MAS Config",
            options=sorted(df["mas_id"].unique()),
            default=sorted(df["mas_id"].unique()),
            key="baseline_filter_configs",
        )
    with col2:
        categories = st.multiselect(
            "Category",
            options=sorted(df["category"].unique()),
            default=sorted(df["category"].unique()),
            key="baseline_filter_categories",
        )
    with col3:
        status = st.selectbox(
            "Status",
            options=["All", "Passed", "Failed"],
            key="baseline_filter_status",
        )

    filtered = df[df["mas_id"].isin(configs) & df["category"].isin(categories)]
    if status == "Passed":
        filtered = filtered[filtered["success"]]
    elif status == "Failed":
        filtered = filtered[~filtered["success"]]

    st.caption(f"{len(filtered)} result(s) shown")
    return filtered


def _render_matrix(df: pd.DataFrame) -> None:
    """Render a colour-coded prompt × config matrix."""
    st.markdown("### Results Matrix")
    st.caption("✓ = passed  ✗ = failed  — = not run")

    if df.empty:
        st.info("No results match the current filters.")
        return

    pivot = df.pivot_table(
        index="prompt_id",
        columns="mas_id",
        values="success",
        aggfunc="first",
    )

    display = pivot.map(lambda v: "✓" if v is True else ("✗" if v is False else "—"))

    def _cell_style(val: str) -> str:
        if val == "✓":
            return "background-color: #28a745; color: white; text-align: center"
        if val == "✗":
            return "background-color: #dc3545; color: white; text-align: center"
        return "background-color: #6c757d; color: white; text-align: center"

    st.dataframe(
        display.style.map(_cell_style),
        use_container_width=True,
    )


def _render_detail_panel(results: list[dict], df_filtered: pd.DataFrame) -> None:
    """Render per-result expandable detail panels."""
    st.markdown("### Run Details")

    if df_filtered.empty:
        return

    visible_keys = set(zip(df_filtered["mas_id"], df_filtered["prompt_id"]))
    visible = [r for r in results if (r["mas_id"], r["prompt_id"]) in visible_keys]

    cat_rank = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    visible.sort(key=lambda r: (cat_rank.get(r["prompt_category"], 99), r["prompt_id"]))

    for r in visible:
        status_icon = "✅" if r["execution"]["success"] else "❌"
        cat_icon = _CATEGORY_ICON.get(r["prompt_category"], "⚪")
        label = (
            f"{status_icon} {cat_icon} "
            f"`{r['prompt_id']}` — `{r['mas_id']}` "
            f"({r['execution']['duration_ms']:.0f} ms)"
        )
        with st.expander(label, expanded=False):
            st.markdown(f"**Prompt:** {r['prompt_text']}")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Category:** `{r['prompt_category']}`")
            c2.markdown(
                f"**Stub:** {'Yes' if r['run_metadata']['stub_mode'] else 'No'}"
            )
            c3.markdown(f"**Agents:** {r['execution']['agent_count']}")
            st.caption(f"Run at {r['run_metadata']['timestamp']}")

            agent_outputs = r.get("agent_outputs", {})
            if agent_outputs:
                st.markdown("**Agent Outputs:**")
                for agent_id, output in agent_outputs.items():
                    raw = (output.get("raw") or "").strip()
                    st.markdown(f"*{agent_id}*")
                    if raw:
                        st.text_area(
                            agent_id,
                            value=raw,
                            height=80,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"detail_{r['mas_id']}_{r['prompt_id']}_{agent_id}",
                        )
                    else:
                        st.caption("(no output)")
            else:
                st.caption("No agent outputs recorded.")
