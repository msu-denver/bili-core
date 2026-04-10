"""
AETHER Results page — Baseline Results Viewer.

Loads JSON result files written by the baseline runner and renders an
interactive summary matrix, category breakdowns, and per-result detail panels.

For attack suite results (injection, jailbreak, etc.) see
``bili/aether/ui/attack_results_page.py``.

Called by the main Streamlit app (``bili/streamlit_app.py``) as a page within
``st.navigation()``.
"""

import io
import json
import logging
from datetime import date
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


@st.cache_data(ttl=30)
def _load_baseline_results() -> list[dict]:
    """Return all parsed JSON result dicts from the baseline results directory.

    Cached with a 30-second TTL so new result files appear promptly without
    re-reading disk on every filter interaction or expander toggle.
    """
    results: list[dict] = []
    for path in sorted(BASELINE_RESULTS_DIR.glob("**/*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Could not parse %s: %s", path, exc)
    return results


def _build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten result dicts into a tidy DataFrame.

    Rows with missing or unexpected keys are skipped with a warning so a
    single malformed file cannot crash the page.
    """
    rows = []
    for r in results:
        try:
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
        except (KeyError, TypeError) as exc:
            LOGGER.warning(
                "Skipping malformed result (mas_id=%s, prompt_id=%s): %s",
                r.get("mas_id", "?"),
                r.get("prompt_id", "?"),
                exc,
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
    st.markdown("## AEGIS")
    st.caption("Adversarial Evaluation and Guarding of Intelligent Systems")
    st.markdown("---")
    st.markdown("#### Baseline Results")
    st.markdown(
        "Generate results by running the baseline suite:\n\n"
        "```\n# Stub mode (no LLM calls)\n"
        "python bili/aegis/tests/baseline/run_baseline.py --stub\n\n"
        "# Full run (requires API credentials)\n"
        "python bili/aegis/tests/baseline/run_baseline.py\n```"
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------


def _render_main() -> None:
    st.markdown("# AEGIS Baseline Results")
    st.markdown(
        "**AEGIS** (Adversarial Evaluation and Guarding of Intelligent Systems) "
        "is BiliCore's security testing framework for multi-agent systems built "
        "with **AETHER**. It provides systematic adversarial evaluation: inject "
        "attacks, track how they propagate through agent networks, and measure "
        "each agent's resilience using a 3-tier detection pipeline."
    )
    st.markdown(
        "Meaningful security evaluation requires a control group. Before "
        "testing how a multi-agent system responds to adversarial payloads, "
        "you need to know how it responds under normal conditions. The "
        "baseline suite runs a set of benign, edge-case, and policy-violating "
        "prompts against each **AETHER** MAS configuration without any attack "
        "injection, establishing ground-truth outputs for every agent in the "
        "system."
    )
    st.markdown(
        "These baselines serve as the reference point for Tier-3 semantic "
        "evaluation. When AEGIS scores an attacked run, it compares each "
        "agent's output against the corresponding baseline to determine "
        "whether the agent's behavior actually changed. Without baselines, "
        "there is no way to distinguish a genuinely influenced agent from one "
        "that would have produced the same output anyway."
    )
    st.markdown("---")

    results = _load_baseline_results()

    if not results:
        st.info(
            "No baseline results found.\n\n"
            "Run the baseline suite to populate this view:\n\n"
            "```\npython bili/aegis/tests/baseline/run_baseline.py --stub\n```"
        )
        return

    df = _build_dataframe(results)
    _render_summary_metrics(df)
    st.markdown("---")
    df_filtered = _render_filters(df)
    st.markdown("---")
    _render_export_buttons(results, df_filtered)
    st.markdown("---")
    _render_matrix(df_filtered)
    st.markdown("---")
    _render_detail_panel(results, df_filtered)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _build_baseline_export_df(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Return a renamed copy of *df_filtered* with explicit export column names.

    Baseline runs have no attack metadata (agent_id, tier2, etc.) so the schema
    differs from the attack results export.  Column mapping:
    ``prompt_id`` → ``prompt_id``, ``success`` → ``tier1_success`` (renamed for
    clarity alongside attack exports), all others kept as-is.
    """
    renamed = df_filtered.rename(columns={"success": "tier1_success"})
    export_cols = [
        "mas_id",
        "prompt_id",
        "category",
        "tier1_success",
        "duration_ms",
        "agent_count",
        "stub_mode",
        "timestamp",
    ]
    # Only select columns that actually exist to avoid KeyError if the
    # upstream schema ever changes.
    available = [c for c in export_cols if c in renamed.columns]
    return renamed[available]


def _render_export_buttons(results: list[dict], df_filtered: pd.DataFrame) -> None:
    """Render CSV and JSON download buttons for the current filtered result set.

    CSV uses an explicit column mapping via ``_build_baseline_export_df``.
    JSON export contains full raw result dicts matched by ``(mas_id, prompt_id)``.
    """
    if df_filtered.empty:
        return

    visible_keys = set(zip(df_filtered["mas_id"], df_filtered["prompt_id"]))
    matched = [
        r for r in results if (r.get("mas_id"), r.get("prompt_id")) in visible_keys
    ]

    unique_mas = df_filtered["mas_id"].dropna().unique()
    mas_label = unique_mas[0] if len(unique_mas) == 1 else "multi"
    today = date.today().isoformat()

    export_df = _build_baseline_export_df(df_filtered)

    col1, col2 = st.columns(2)
    with col1:
        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            "⬇ Export CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"aether_baseline_{mas_label}_{today}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ Export JSON",
            data=json.dumps(matched, indent=2, default=str).encode("utf-8"),
            file_name=f"aether_baseline_{mas_label}_{today}.json",
            mime="application/json",
            use_container_width=True,
        )


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
    cols[4].metric("Success Rate", f"{passed / total * 100:.0f}%" if total else "N/A")


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter widgets and return the filtered DataFrame."""
    col1, col2, col3 = st.columns(3)
    with col1:
        configs = st.multiselect(
            "MAS Config",
            options=sorted(df["mas_id"].unique()),
            default=sorted(df["mas_id"].unique()),
            key="baseline_filter_configs",
        )
    with col2:
        present = set(df["category"].unique())
        ordered_cats = [c for c in _CATEGORY_ORDER if c in present] + sorted(
            present - set(_CATEGORY_ORDER)
        )
        categories = st.multiselect(
            "Category",
            options=ordered_cats,
            default=ordered_cats,
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
    """Render a color-coded pass/fail pivot table of prompt × MAS config."""
    st.markdown("### Results Matrix")
    st.caption("✓ = passed  ✗ = failed  — = not run")

    if df.empty:
        st.info("No results match the current filters.")
        return

    pivot = df.pivot_table(
        index="prompt_id",
        columns="mas_id",
        values="success",
        aggfunc="last",
    )

    def _baseline_symbol(v) -> str:
        if pd.isna(v):
            return "—"
        return "✓" if v else "✗"

    display = pivot.map(_baseline_symbol)

    def _cell_style(val: str) -> str:
        if val == "✓":
            return "background-color: #28a745; color: white; text-align: center"
        if val == "✗":
            return "background-color: #dc3545; color: white; text-align: center"
        return "background-color: #6c757d; color: white; text-align: center"

    st.dataframe(display.style.map(_cell_style), use_container_width=True)


def _render_detail_panel(results: list[dict], df_filtered: pd.DataFrame) -> None:
    """Render expandable per-run detail panels sorted by category."""
    st.markdown("### Run Details")

    if df_filtered.empty:
        return

    visible_keys = set(zip(df_filtered["mas_id"], df_filtered["prompt_id"]))
    visible = [r for r in results if (r["mas_id"], r["prompt_id"]) in visible_keys]

    cat_rank = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    visible.sort(
        key=lambda r: (
            cat_rank.get(r.get("prompt_category", ""), 99),
            r.get("prompt_id", ""),
        )
    )

    for r in visible:
        execution = r.get("execution", {})
        run_metadata = r.get("run_metadata", {})
        status_icon = "✅" if execution.get("success") else "❌"
        cat_icon = _CATEGORY_ICON.get(r.get("prompt_category", ""), "⚪")
        duration = execution.get("duration_ms", 0.0)
        label = (
            f"{status_icon} {cat_icon} "
            f"`{r.get('prompt_id', '?')}` — `{r.get('mas_id', '?')}` "
            f"({duration:.0f} ms)"
        )
        with st.expander(label, expanded=False):
            st.markdown(f"**Prompt:** {r.get('prompt_text', '(no prompt text)')}")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Category:** `{r.get('prompt_category', '?')}`")
            c2.markdown(f"**Stub:** {'Yes' if run_metadata.get('stub_mode') else 'No'}")
            c3.markdown(f"**Agents:** {execution.get('agent_count', '?')}")
            st.caption(f"Run at {run_metadata.get('timestamp', 'unknown')}")

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
                            key=f"bl_detail_{r.get('mas_id')}_{r.get('prompt_id')}_{agent_id}",
                        )
                    else:
                        st.caption("(no output)")
            else:
                st.caption("No agent outputs recorded.")
