"""
AETHER Results page — Baseline & Injection Suite Results Viewer.

Loads JSON result files written by the baseline and injection suite runners
and renders interactive summary matrices, filters, and per-result detail panels
across two tabs.

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
INJECTION_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "injection" / "results"
)
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

# Baseline category display
_CATEGORY_ICON = {
    "benign": "🟢",
    "violating": "🔴",
    "edge_case": "🟡",
}
_CATEGORY_ORDER = ["benign", "edge_case", "violating"]

# Injection tier-3 score display (0=fully resisted → green, 3=fully complied → red)
_TIER3_COLOR = {
    0: "#28a745",  # green  — no compliance (fully resisted)
    1: "#ffc107",  # yellow — partial compliance
    2: "#fd7e14",  # orange — clear compliance
    3: "#dc3545",  # red    — full compliance (fully susceptible)
}
_SEVERITY_ORDER = ["high", "medium", "low"]


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


@st.cache_data(ttl=30)
def _load_injection_results() -> list[dict]:
    """Return all parsed JSON result dicts from the injection results directory.

    Cached with a 30-second TTL. NDJSON log files and the CSV matrix are
    excluded automatically because glob only matches ``*.json``.
    """
    results: list[dict] = []
    for path in sorted(INJECTION_RESULTS_DIR.glob("**/*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Could not parse %s: %s", path, exc)
    return results


def _build_baseline_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten baseline result dicts into a tidy DataFrame.

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
                "Skipping malformed baseline result (mas_id=%s, prompt_id=%s): %s",
                r.get("mas_id", "?"),
                r.get("prompt_id", "?"),
                exc,
            )
    return pd.DataFrame(rows)


def _build_injection_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten injection result dicts into a tidy DataFrame.

    ``tier3_score`` is stored as a string in run_metadata ("0"–"3" or "").
    It is parsed to ``Optional[int]`` — ``None`` means skipped/not evaluated.
    """
    rows = []
    for r in results:
        try:
            execution = r.get("execution", {})
            run_meta = r.get("run_metadata", {})
            tier3_raw = run_meta.get("tier3_score", "")
            tier3_score: int | None = int(tier3_raw) if tier3_raw != "" else None
            rows.append(
                {
                    "mas_id": r["mas_id"],
                    "payload_id": r["payload_id"],
                    "injection_type": r.get("injection_type", "?"),
                    "severity": r.get("severity", "?"),
                    "phase": r.get("injection_phase", "?"),
                    "tier1_pass": execution.get("success", False),
                    "influenced_agents": r.get("influenced_agents", []),
                    "resistant_agents": r.get("resistant_agents", []),
                    "tier3_score": tier3_score,
                    "tier3_confidence": run_meta.get("tier3_confidence", ""),
                    "stub_mode": run_meta.get("stub_mode", False),
                    "timestamp": run_meta.get("timestamp", ""),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Skipping malformed injection result (mas_id=%s, payload_id=%s): %s",
                r.get("mas_id", "?"),
                r.get("payload_id", "?"),
                exc,
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render_results_page() -> None:
    """Render the Results page (sidebar + tabbed main area).

    Called by the unified Streamlit app after ``st.set_page_config()``
    has already been invoked.
    """
    with st.sidebar:
        _render_sidebar()

    tab_baseline, tab_injection = st.tabs(["Baseline", "Injection Suite"])
    with tab_baseline:
        _render_baseline_tab()
    with tab_injection:
        _render_injection_tab()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)
    st.markdown("## AETHER Results")
    st.markdown("---")
    st.caption("Baseline & Injection Suite Viewer")
    st.markdown(
        "**Baseline runner:**\n\n"
        "```\n# Stub mode (no LLM calls)\n"
        "python bili/aether/tests/baseline/run_baseline.py --stub\n\n"
        "# Full run (requires API credentials)\n"
        "python bili/aether/tests/baseline/run_baseline.py\n```\n\n"
        "**Injection suite runner:**\n\n"
        "```\n# Stub mode\n"
        "python bili/aether/tests/injection/run_injection_suite.py --stub\n\n"
        "# Full run\n"
        "python bili/aether/tests/injection/run_injection_suite.py\n```"
    )


# ---------------------------------------------------------------------------
# Baseline tab
# ---------------------------------------------------------------------------


def _render_baseline_tab() -> None:
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

    df = _build_baseline_dataframe(results)
    _render_baseline_metrics(df)
    st.markdown("---")
    df_filtered = _render_baseline_filters(df)
    st.markdown("---")
    _render_baseline_matrix(df_filtered)
    st.markdown("---")
    _render_baseline_detail_panel(results, df_filtered)


def _render_baseline_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    passed = int(df["success"].sum())
    cols = st.columns(5)
    cols[0].metric("Configs", df["mas_id"].nunique())
    cols[1].metric("Prompts", df["prompt_id"].nunique())
    cols[2].metric("Total Runs", total)
    cols[3].metric("Passed", passed)
    cols[4].metric("Success Rate", f"{passed / total * 100:.0f}%" if total else "N/A")


def _render_baseline_filters(df: pd.DataFrame) -> pd.DataFrame:
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


def _render_baseline_matrix(df: pd.DataFrame) -> None:
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
    display = pivot.map(lambda v: "✓" if v is True else ("✗" if v is False else "—"))

    def _cell_style(val: str) -> str:
        if val == "✓":
            return "background-color: #28a745; color: white; text-align: center"
        if val == "✗":
            return "background-color: #dc3545; color: white; text-align: center"
        return "background-color: #6c757d; color: white; text-align: center"

    st.dataframe(display.style.map(_cell_style), use_container_width=True)


def _render_baseline_detail_panel(
    results: list[dict], df_filtered: pd.DataFrame
) -> None:
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


# ---------------------------------------------------------------------------
# Injection Suite tab
# ---------------------------------------------------------------------------


def _render_injection_tab() -> None:
    st.markdown("# Injection Suite Results")
    st.markdown(
        "Tier 1–3 results from the AETHER prompt-injection evaluation suite. "
        "Each cell shows the tier-3 compliance score for one payload against "
        "one MAS configuration (0 = fully resisted, 3 = fully complied)."
    )
    st.markdown("---")

    results = _load_injection_results()

    if not results:
        st.info(
            "No injection suite results found.\n\n"
            "Run the injection suite to populate this view:\n\n"
            "```\npython bili/aether/tests/injection/run_injection_suite.py --stub\n```"
        )
        return

    df = _build_injection_dataframe(results)
    _render_injection_metrics(df)
    st.markdown("---")
    df_filtered = _render_injection_filters(df)
    st.markdown("---")
    _render_injection_matrix(df_filtered)
    st.markdown("---")
    _render_injection_detail_panel(results, df_filtered)


def _render_injection_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    tier1_passed = int(df["tier1_pass"].sum())
    evaluated = df["tier3_score"].notna()
    avg_score = df.loc[evaluated, "tier3_score"].mean() if evaluated.any() else None

    cols = st.columns(5)
    cols[0].metric("Configs", df["mas_id"].nunique())
    cols[1].metric("Payloads", df["payload_id"].nunique())
    cols[2].metric("Total Runs", total)
    cols[3].metric(
        "Tier 1 Pass",
        f"{tier1_passed / total * 100:.0f}%" if total else "N/A",
    )
    cols[4].metric(
        "Avg Tier-3 Score",
        f"{avg_score:.2f}" if avg_score is not None else "N/A",
        help="Average compliance score across evaluated runs (0=resistant, 3=susceptible).",
    )


def _render_injection_filters(df: pd.DataFrame) -> pd.DataFrame:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        configs = st.multiselect(
            "MAS Config",
            options=sorted(df["mas_id"].unique()),
            default=sorted(df["mas_id"].unique()),
            key="inj_filter_configs",
        )
    with col2:
        inj_types = st.multiselect(
            "Injection Type",
            options=sorted(df["injection_type"].unique()),
            default=sorted(df["injection_type"].unique()),
            key="inj_filter_types",
        )
    with col3:
        present_sev = set(df["severity"].unique())
        ordered_sev = [s for s in _SEVERITY_ORDER if s in present_sev] + sorted(
            present_sev - set(_SEVERITY_ORDER)
        )
        severities = st.multiselect(
            "Severity",
            options=ordered_sev,
            default=ordered_sev,
            key="inj_filter_severity",
        )
    with col4:
        phase_options = ["All"] + sorted(df["phase"].unique())
        phase = st.selectbox(
            "Phase",
            options=phase_options,
            key="inj_filter_phase",
        )

    filtered = df[
        df["mas_id"].isin(configs)
        & df["injection_type"].isin(inj_types)
        & df["severity"].isin(severities)
    ]
    if phase != "All":
        filtered = filtered[filtered["phase"] == phase]

    st.caption(f"{len(filtered)} result(s) shown")
    return filtered


def _render_injection_matrix(df: pd.DataFrame) -> None:
    """Render a tier-3 compliance score matrix (payload × config)."""
    st.markdown("### Tier-3 Compliance Matrix")
    st.caption(
        "Score: 0 = no compliance (green)  1 = partial (yellow)  "
        "2 = clear (orange)  3 = full (red)  — = not evaluated  ! = tier-1 failure"
    )

    if df.empty:
        st.info("No results match the current filters.")
        return

    # Row label includes phase to distinguish pre- vs mid-execution payloads
    df = df.copy()
    df["row_label"] = df["payload_id"] + " (" + df["phase"].str[:3] + ")"

    pivot_score = df.pivot_table(
        index="row_label",
        columns="mas_id",
        values="tier3_score",
        aggfunc="last",
    )
    pivot_tier1 = df.pivot_table(
        index="row_label",
        columns="mas_id",
        values="tier1_pass",
        aggfunc="last",
    )

    def _display_val(score, tier1) -> str:
        if not tier1:
            return "!"
        if pd.isna(score):
            return "—"
        return str(int(score))

    display = pd.DataFrame(
        {
            col: [
                _display_val(
                    (
                        pivot_score.loc[row, col]
                        if col in pivot_score.columns and row in pivot_score.index
                        else float("nan")
                    ),
                    (
                        pivot_tier1.loc[row, col]
                        if col in pivot_tier1.columns and row in pivot_tier1.index
                        else True
                    ),
                )
                for row in pivot_score.index
            ]
            for col in pivot_score.columns
        },
        index=pivot_score.index,
    )

    def _cell_style(val: str) -> str:
        if val == "!":
            return "background-color: #343a40; color: white; text-align: center"
        if val == "—":
            return "background-color: #6c757d; color: white; text-align: center"
        score_int = int(val) if val.isdigit() else -1
        color = _TIER3_COLOR.get(score_int, "#6c757d")
        return f"background-color: {color}; color: white; text-align: center"

    st.dataframe(display.style.map(_cell_style), use_container_width=True)


_TIER3_ICON = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}


def _render_injection_detail_panel(
    results: list[dict], df_filtered: pd.DataFrame
) -> None:
    st.markdown("### Run Details")

    if df_filtered.empty:
        return

    # Build a lookup from the already-parsed DataFrame so we don't re-parse
    # tier3_score from the raw result dicts.
    tier3_lookup: dict[tuple, int | None] = {
        (row["mas_id"], row["payload_id"], row["phase"]): row["tier3_score"]
        for _, row in df_filtered.iterrows()
    }

    visible_keys = set(tier3_lookup.keys())
    visible = [
        r
        for r in results
        if (r.get("mas_id"), r.get("payload_id"), r.get("injection_phase"))
        in visible_keys
    ]

    sev_rank = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
    visible.sort(
        key=lambda r: (
            sev_rank.get(r.get("severity", ""), 99),
            r.get("payload_id", ""),
            r.get("injection_phase", ""),
        )
    )

    for r in visible:
        execution = r.get("execution", {})
        run_meta = r.get("run_metadata", {})
        tier1_ok = execution.get("success", False)
        key = (r.get("mas_id"), r.get("payload_id"), r.get("injection_phase"))
        tier3_score: int | None = tier3_lookup.get(key)

        t1_icon = "✅" if tier1_ok else "❌"
        t3_icon = (
            _TIER3_ICON.get(tier3_score, "⚫") if tier3_score is not None else "⬜"
        )
        t3_badge = f"T3:{tier3_score}" if tier3_score is not None else "T3:—"

        label = (
            f"{t1_icon} [{t3_badge}] "
            f"`{r.get('payload_id', '?')}` ({r.get('injection_phase', '?')}) "
            f"— `{r.get('mas_id', '?')}` "
            f"[{r.get('severity', '?')} / {r.get('injection_type', '?')}]"
        )

        with st.expander(label, expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Severity:** `{r.get('severity', '?')}`")
            c2.markdown(f"**Phase:** `{r.get('injection_phase', '?')}`")
            c3.markdown(f"**Type:** `{r.get('injection_type', '?')}`")
            c4.markdown(f"**Stub:** {'Yes' if run_meta.get('stub_mode') else 'No'}")
            st.caption(f"Run at {run_meta.get('timestamp', 'unknown')}")

            # Tier 2: propagation
            prop_path = r.get("propagation_path", [])
            influenced = r.get("influenced_agents", [])
            resistant = r.get("resistant_agents", [])

            if prop_path:
                st.markdown(
                    "**Propagation Path:** " + " → ".join(f"`{a}`" for a in prop_path)
                )
            col_inf, col_res = st.columns(2)
            col_inf.markdown(
                "**Influenced:** "
                + (", ".join(f"`{a}`" for a in influenced) if influenced else "none")
            )
            col_res.markdown(
                "**Resistant:** "
                + (", ".join(f"`{a}`" for a in resistant) if resistant else "none")
            )

            # Tier 3: semantic score
            st.markdown("---")
            if tier3_score is not None:
                confidence = run_meta.get("tier3_confidence", "")
                reasoning = run_meta.get("tier3_reasoning", "")
                s1, s2 = st.columns(2)
                s1.markdown(f"**Tier-3 Score:** {t3_icon} `{tier3_score}`")
                s2.markdown(f"**Confidence:** `{confidence}`")
                if reasoning:
                    st.markdown(f"**Reasoning:** {reasoning}")
            else:
                st.caption("Tier-3 evaluation skipped (stub mode or not configured).")
