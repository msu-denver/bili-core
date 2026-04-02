"""
AETHER Attack Results page — unified viewer for all attack suite results.

Loads JSON result files from all seven attack suite result directories and
renders an interactive compliance matrix, filters, and per-result detail panels.

Suites supported:
  injection, jailbreak, memory_poisoning, bias_inheritance,
  agent_impersonation, persistence, cross_model

Schema notes:
  - All suites share the base schema written by ``_suite_runner._build_result_dict()``.
  - ``cross_model`` adds ``model_id``, ``model_name``, and ``provider_family`` fields
    at the top level of each result JSON.
  - ``persistence`` phase is always ``checkpoint_injection``; ``skipped``/``skip_reason``
    are CSV-only and do not appear in the result JSON.
  - Missing suite-specific fields are normalised to ``None`` so a single DataFrame
    covers all suites without per-suite branch logic in rendering.

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

_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "images" / "logo.png"

# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

# Maps display name → (suite_dir_name, has_model_dimension)
# has_model_dimension=True means results are nested one level deeper:
#   {mas_id}/{model_id_safe}/{payload_id}_{phase}.json
_SUITE_REGISTRY: dict[str, tuple[str, bool]] = {
    "Injection": ("injection", False),
    "Jailbreak": ("jailbreak", False),
    "Memory Poisoning": ("memory_poisoning", False),
    "Bias Inheritance": ("bias_inheritance", False),
    "Agent Impersonation": ("agent_impersonation", False),
    "Persistence": ("persistence", False),
    "Cross-Model": ("cross_model", True),
}

_ALL_SUITES = "All Suites"

# ---------------------------------------------------------------------------
# Tier-3 display
# ---------------------------------------------------------------------------

_TIER3_COLOR = {
    0: "#28a745",  # green  — no compliance (fully resisted)
    1: "#ffc107",  # yellow — partial compliance
    2: "#fd7e14",  # orange — clear compliance
    3: "#dc3545",  # red    — fully complied (fully susceptible)
}
_TIER3_ICON = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
_SEVERITY_ORDER = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def _load_suite_results(suite_dir: str) -> list[dict]:
    """Load and normalise all JSON result files for one suite.

    Cached per ``suite_dir`` with a 30-second TTL.  NDJSON logs and CSV files
    are excluded automatically (glob only matches ``*.json``).

    Cross-model results live one level deeper
    (``{mas_id}/{model_id_safe}/*.json``); the flat ``**/*.json`` glob covers
    both layouts transparently.

    Fields not present in the raw JSON are normalised to ``None`` so
    downstream code can use a single unified DataFrame regardless of suite.
    """
    results_dir = _TESTS_DIR / suite_dir / "results"
    results: list[dict] = []

    if not results_dir.exists():
        return results

    for path in sorted(results_dir.glob("**/*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            results.append(_normalise(raw))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Could not parse %s: %s", path, exc)

    return results


def _normalise(r: dict) -> dict:
    """Return a copy of *r* with all expected fields guaranteed present.

    Suite-specific fields (``model_id``, ``model_name``, ``provider_family``)
    default to ``None`` when absent so the DataFrame schema is uniform.
    """
    execution = r.get("execution", {})
    run_meta = r.get("run_metadata", {})
    tier3_raw = run_meta.get("tier3_score", "")
    tier3_score: int | None = int(tier3_raw) if tier3_raw not in ("", None) else None

    return {
        # Core identity
        "payload_id": r.get("payload_id", "?"),
        "injection_type": r.get("injection_type", "?"),
        "severity": r.get("severity", "?"),
        "mas_id": r.get("mas_id", "?"),
        "phase": r.get("injection_phase", "?"),
        "attack_suite": r.get("attack_suite", "?"),
        # Execution (Tier 1)
        "tier1_pass": execution.get("success", False),
        "duration_ms": execution.get("duration_ms", 0.0),
        "agent_count": execution.get("agent_count", 0),
        # Propagation (Tier 2)
        "target_agent_id": r.get("target_agent_id", ""),
        "propagation_path": r.get("propagation_path", []),
        "influenced_agents": r.get("influenced_agents", []),
        "resistant_agents": r.get("resistant_agents", []),
        # Semantic evaluation (Tier 3)
        "tier3_score": tier3_score,
        "tier3_confidence": run_meta.get("tier3_confidence", ""),
        "tier3_reasoning": run_meta.get("tier3_reasoning", ""),
        # Metadata
        "stub_mode": run_meta.get("stub_mode", False),
        "timestamp": run_meta.get("timestamp", ""),
        # Cross-model fields (None for other suites)
        "model_id": r.get("model_id"),
        "model_name": r.get("model_name"),
        "provider_family": r.get("provider_family"),
        # Config fingerprint
        "config_path": r.get("config_fingerprint", {}).get("config_path", ""),
    }


def _build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten normalised result dicts into a tidy DataFrame.

    Malformed rows are skipped with a warning so a single bad file cannot
    crash the page.
    """
    rows = []
    for r in results:
        try:
            rows.append(
                {
                    "payload_id": r["payload_id"],
                    "injection_type": r["injection_type"],
                    "severity": r["severity"],
                    "mas_id": r["mas_id"],
                    "phase": r["phase"],
                    "attack_suite": r["attack_suite"],
                    "tier1_pass": r["tier1_pass"],
                    "tier3_score": r["tier3_score"],
                    "stub_mode": r["stub_mode"],
                    "timestamp": r["timestamp"],
                    "model_id": r["model_id"],
                    "model_name": r["model_name"],
                    "provider_family": r["provider_family"],
                    "tier2_influenced": bool(r.get("influenced_agents", [])),
                }
            )
        except (KeyError, TypeError) as exc:
            LOGGER.warning(
                "Skipping malformed row (mas_id=%s, payload_id=%s): %s",
                r.get("mas_id", "?"),
                r.get("payload_id", "?"),
                exc,
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render_attack_results_page() -> None:
    """Render the Attack Results page.

    Called by the unified Streamlit app after ``st.set_page_config()`` has
    already been invoked.
    """
    with st.sidebar:
        selected_suite, extra_paths = _render_sidebar()

    _render_main(selected_suite, extra_paths)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[str, list[Path]]:
    """Render sidebar controls.

    Returns:
        selected_suite: display name from _SUITE_REGISTRY or _ALL_SUITES
        extra_paths: additional result directories from the override expander
    """
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)
    st.markdown("## Attack Results")
    st.markdown("---")

    suite_options = [_ALL_SUITES] + list(_SUITE_REGISTRY)
    selected = st.selectbox(
        "Suite",
        options=suite_options,
        key="attack_suite_selector",
    )

    extra_paths: list[Path] = []
    with st.expander("Custom result paths", expanded=False):
        st.caption(
            "Add additional result directories to include alongside the selected suite. "
            "One absolute path per line."
        )
        raw = st.text_area(
            "Paths",
            key="attack_custom_paths",
            label_visibility="collapsed",
            height=100,
            placeholder="/path/to/results\n/another/path",
        )
        # NOTE: This accepts arbitrary absolute paths. Acceptable because this app
        # runs locally with the user's own filesystem permissions and is not
        # exposed as a public web service. If ever deployed to untrusted users,
        # path access should be restricted to a safe root directory.
        for line in raw.splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                if p.is_dir():
                    extra_paths.append(p)
                else:
                    st.warning(f"Not a directory: `{line}`")

    st.markdown("---")
    st.caption("**Runner commands:**")
    st.markdown(
        "```\n# Stub mode (no LLM calls)\n"
        "python bili/aether/tests/{suite}/run_{suite}_suite.py --stub\n\n"
        "# Full run\n"
        "python bili/aether/tests/{suite}/run_{suite}_suite.py\n```"
    )

    return selected, extra_paths


# ---------------------------------------------------------------------------
# Main area dispatch
# ---------------------------------------------------------------------------


def _render_main(selected_suite: str, extra_paths: list[Path]) -> None:
    st.markdown("# Attack Results")
    st.markdown(
        "Tier 1–3 results from AETHER attack evaluation suites. "
        "Cells show the Tier-3 compliance score (0 = fully resisted, 3 = fully complied) "
        "where available, falling back to a Tier-2 heuristic label where semantic "
        "evaluation was skipped."
    )
    st.markdown("---")

    # Load results for the selected suite(s)
    if selected_suite == _ALL_SUITES:
        all_results: list[dict] = []
        for _, (suite_dir, _) in _SUITE_REGISTRY.items():
            all_results.extend(_load_suite_results(suite_dir))
    else:
        suite_dir, _ = _SUITE_REGISTRY[selected_suite]
        all_results = list(_load_suite_results(suite_dir))

    # Load any custom paths
    for custom_dir in extra_paths:
        for path in sorted(custom_dir.glob("**/*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                all_results.append(_normalise(raw))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("Could not parse custom path %s: %s", path, exc)

    if not all_results:
        suite_label = selected_suite if selected_suite != _ALL_SUITES else "any suite"
        st.info(
            f"No results found for **{suite_label}**.\n\n"
            "Run the suite to populate this view. Example:\n\n"
            "```\npython bili/aether/tests/injection/run_injection_suite.py --stub\n```"
        )
        return

    df_all = _build_dataframe(all_results)
    # In "All Suites" view, if any cross-model results exist the matrix gains a
    # model_id dimension for all rows. Non-cross-model rows show "?" in that
    # column — this is intentional so cross-model and standard results can
    # coexist in the same matrix without splitting into two separate tables.
    is_cross_model = selected_suite == "Cross-Model" or (
        selected_suite == _ALL_SUITES and df_all["model_id"].notna().any()
    )

    _render_metrics(df_all)
    st.markdown("---")
    df_filtered = _render_filters(df_all, selected_suite, is_cross_model)
    st.markdown("---")
    _render_export_buttons(all_results, df_filtered, is_cross_model)
    st.markdown("---")
    _render_matrix(df_filtered, is_cross_model)
    st.markdown("---")
    _render_detail_panel(all_results, df_filtered, is_cross_model)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _result_export_key(r: dict, is_cross_model: bool) -> tuple:
    """Return the lookup key used to match normalised results to df_filtered rows."""
    return (
        r.get("mas_id"),
        r.get("payload_id"),
        r.get("phase"),
        r.get("model_id") if is_cross_model else None,
    )


def _build_export_df(
    all_results: list[dict], df_filtered: pd.DataFrame, is_cross_model: bool
) -> pd.DataFrame:
    """Build a flat export DataFrame matching the current filter state.

    Columns follow the DoD CSV schema with additional context fields.
    List fields (influenced_agents, resistant_agents, propagation_path) are
    serialised as semicolon-joined strings for easy loading in pandas/Excel.
    """
    key_set: set = set()
    for _, row in df_filtered.iterrows():
        key_set.add(
            (
                row["mas_id"],
                row["payload_id"],
                row["phase"],
                row["model_id"] if is_cross_model else None,
            )
        )

    rows = []
    for r in all_results:
        if _result_export_key(r, is_cross_model) not in key_set:
            continue
        rows.append(
            {
                "mas_id": r.get("mas_id", ""),
                "agent_id": r.get("target_agent_id", ""),
                "attack_type": r.get("injection_type", ""),
                "payload_id": r.get("payload_id", ""),
                "tier1_success": r.get("tier1_pass", False),
                "tier2_influenced": ";".join(r.get("influenced_agents") or []),
                "tier2_resisted": ";".join(r.get("resistant_agents") or []),
                "tier3_score": r.get("tier3_score", ""),
                "model_name": r.get("model_name") or "",
                "timestamp": r.get("timestamp", ""),
                # Extra context columns
                "phase": r.get("phase", ""),
                "severity": r.get("severity", ""),
                "attack_suite": r.get("attack_suite", ""),
                "propagation_path": ";".join(r.get("propagation_path") or []),
            }
        )
    return pd.DataFrame(rows)


def _export_filename(
    df_filtered: pd.DataFrame, ext: str, prefix: str = "aether_results"
) -> str:
    """Return a default filename based on unique mas_ids and today's date."""
    unique_mas = df_filtered["mas_id"].dropna().unique()
    mas_label = unique_mas[0] if len(unique_mas) == 1 else "multi"
    return f"{prefix}_{mas_label}_{date.today().isoformat()}.{ext}"


def _render_export_buttons(
    all_results: list[dict], df_filtered: pd.DataFrame, is_cross_model: bool
) -> None:
    """Render CSV and JSON download buttons for the current filtered result set."""
    if df_filtered.empty:
        return

    key_set: set = {
        (
            row["mas_id"],
            row["payload_id"],
            row["phase"],
            row["model_id"] if is_cross_model else None,
        )
        for _, row in df_filtered.iterrows()
    }
    export_df = _build_export_df(all_results, df_filtered, is_cross_model)
    matched = [
        r for r in all_results if _result_export_key(r, is_cross_model) in key_set
    ]

    col1, col2 = st.columns(2)
    with col1:
        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            "⬇ Export CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name=_export_filename(df_filtered, "csv"),
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ Export JSON",
            data=json.dumps(matched, indent=2, default=str).encode("utf-8"),
            file_name=_export_filename(df_filtered, "json"),
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Metrics bar
# ---------------------------------------------------------------------------


def _render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    tier1_passed = int(df["tier1_pass"].sum())
    evaluated = df["tier3_score"].notna()
    avg_score = df.loc[evaluated, "tier3_score"].mean() if evaluated.any() else None

    cols = st.columns(6)
    cols[0].metric("Suites", df["attack_suite"].nunique())
    cols[1].metric("Configs", df["mas_id"].nunique())
    cols[2].metric("Payloads", df["payload_id"].nunique())
    cols[3].metric("Total Runs", total)
    cols[4].metric(
        "Tier-1 Pass",
        f"{tier1_passed / total * 100:.0f}%" if total else "N/A",
    )
    cols[5].metric(
        "Avg T3 Score",
        f"{avg_score:.2f}" if avg_score is not None else "N/A",
        help="Average Tier-3 compliance score for evaluated runs (0=resistant, 3=susceptible).",
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _render_filters(
    df: pd.DataFrame, selected_suite: str, is_cross_model: bool
) -> pd.DataFrame:
    """Render filter widgets and return filtered DataFrame."""
    row1 = st.columns(4 if selected_suite != _ALL_SUITES else 5)

    col_idx = 0

    # Suite filter — only shown in All Suites view
    if selected_suite == _ALL_SUITES:
        with row1[col_idx]:
            suite_opts = sorted(df["attack_suite"].unique())
            suites = st.multiselect(
                "Suite",
                options=suite_opts,
                default=suite_opts,
                key="atk_filter_suite",
            )
        col_idx += 1
    else:
        suites = list(df["attack_suite"].unique())

    with row1[col_idx]:
        configs = st.multiselect(
            "MAS Config",
            options=sorted(df["mas_id"].unique()),
            default=sorted(df["mas_id"].unique()),
            key="atk_filter_configs",
        )
    col_idx += 1

    with row1[col_idx]:
        inj_types = st.multiselect(
            "Attack Type",
            options=sorted(df["injection_type"].unique()),
            default=sorted(df["injection_type"].unique()),
            key="atk_filter_types",
        )
    col_idx += 1

    with row1[col_idx]:
        present_sev = set(df["severity"].dropna().unique())
        ordered_sev = [s for s in _SEVERITY_ORDER if s in present_sev] + sorted(
            present_sev - set(_SEVERITY_ORDER)
        )
        severities = st.multiselect(
            "Severity",
            options=ordered_sev,
            default=ordered_sev,
            key="atk_filter_severity",
        )
    col_idx += 1

    with row1[col_idx]:
        phase_opts = ["All"] + sorted(df["phase"].dropna().unique())
        phase = st.selectbox("Phase", options=phase_opts, key="atk_filter_phase")

    # Second row: model_id filter for cross-model, Tier-3 score range
    row2 = st.columns(3)
    with row2[0]:
        tier3_status = st.selectbox(
            "Detection Tier",
            options=[
                "All",
                "Tier-3 evaluated",
                "Tier-2 only (skipped)",
                "Tier-1 failed",
            ],
            key="atk_filter_tier",
        )
    with row2[1]:
        if is_cross_model and df["model_id"].notna().any():
            model_opts = sorted(df["model_id"].dropna().unique())
            model_ids = st.multiselect(
                "Model ID",
                options=model_opts,
                default=model_opts,
                key="atk_filter_model_id",
            )
        else:
            model_ids = None
    with row2[2]:
        if is_cross_model and df["provider_family"].notna().any():
            pf_opts = sorted(df["provider_family"].dropna().unique())
            provider_families = st.multiselect(
                "Provider Family",
                options=pf_opts,
                default=pf_opts,
                key="atk_filter_provider",
            )
        else:
            provider_families = None

    # Apply filters
    mask = (
        df["attack_suite"].isin(suites)
        & df["mas_id"].isin(configs)
        & df["injection_type"].isin(inj_types)
        & df["severity"].isin(severities)
    )
    if phase != "All":
        mask &= df["phase"] == phase
    if tier3_status == "Tier-3 evaluated":
        mask &= df["tier3_score"].notna()
    elif tier3_status == "Tier-2 only (skipped)":
        mask &= df["tier3_score"].isna() & df["tier1_pass"]
    elif tier3_status == "Tier-1 failed":
        mask &= ~df["tier1_pass"]
    if model_ids is not None:
        mask &= df["model_id"].isin(model_ids)
    if provider_families is not None:
        mask &= df["provider_family"].isin(provider_families)

    filtered = df[mask]
    st.caption(f"{len(filtered)} result(s) shown")
    return filtered


# ---------------------------------------------------------------------------
# Results matrix
# ---------------------------------------------------------------------------


def _render_matrix(df: pd.DataFrame, is_cross_model: bool) -> None:
    """Render the compliance score matrix.

    Rows = payload_id (+ phase suffix).
    Columns = mas_id (+ model_id for cross-model view).
    Cell value = Tier-3 score if evaluated, T2:✓/✗ if skipped, ! for Tier-1 failure.
    """
    st.markdown("### Results Matrix")
    st.caption(
        "0🟢 = fully resisted  1🟡 = partial  2🟠 = clear  3🔴 = fully complied  "
        "T2:✓ = Tier-2 heuristic pass  T2:✗ = Tier-2 heuristic fail  "
        "— = not evaluated  ! = Tier-1 failure"
    )

    if df.empty:
        st.info("No results match the current filters.")
        return

    df = df.copy()
    df["row_label"] = df["payload_id"] + " (" + df["phase"].str[:3] + ")"

    if is_cross_model and df["model_id"].notna().any():
        df["col_label"] = df["mas_id"] + " / " + df["model_id"].fillna("?")
    else:
        df["col_label"] = df["mas_id"]

    pivot_score = df.pivot_table(
        index="row_label",
        columns="col_label",
        values="tier3_score",
        aggfunc="last",
    )
    pivot_tier1 = df.pivot_table(
        index="row_label",
        columns="col_label",
        values="tier1_pass",
        aggfunc="last",
    )
    # For Tier-2 fallback: use pre-computed boolean column from _build_dataframe
    pivot_tier2 = df.pivot_table(
        index="row_label",
        columns="col_label",
        values="tier2_influenced",
        aggfunc="last",
    )

    def _cell_val(row_label: str, col_label: str) -> str:
        tier1 = (
            pivot_tier1.loc[row_label, col_label]
            if col_label in pivot_tier1.columns and row_label in pivot_tier1.index
            else True
        )
        if not tier1:
            return "!"
        score = (
            pivot_score.loc[row_label, col_label]
            if col_label in pivot_score.columns and row_label in pivot_score.index
            else float("nan")
        )
        if not pd.isna(score):
            return str(int(score))
        # Fall back to Tier-2 heuristic
        t2 = (
            pivot_tier2.loc[row_label, col_label]
            if col_label in pivot_tier2.columns and row_label in pivot_tier2.index
            else None
        )
        if t2 is not None and not pd.isna(t2):
            return "T2:✓" if not t2 else "T2:✗"
        return "—"

    display = pd.DataFrame(
        {
            col: [_cell_val(row, col) for row in pivot_score.index]
            for col in pivot_score.columns
        },
        index=pivot_score.index,
    )

    def _cell_style(val: str) -> str:
        if val == "!":
            return "background-color: #343a40; color: white; text-align: center"
        if val == "—":
            return "background-color: #6c757d; color: white; text-align: center"
        if val.startswith("T2:"):
            # Tier-2 heuristic fallback — muted styling
            return "background-color: #495057; color: white; text-align: center"
        score_int = int(val) if val.isdigit() else -1
        color = _TIER3_COLOR.get(score_int, "#6c757d")
        return f"background-color: {color}; color: white; text-align: center"

    st.dataframe(display.style.map(_cell_style), use_container_width=True)


# ---------------------------------------------------------------------------
# Detail panels
# ---------------------------------------------------------------------------


def _render_detail_panel(
    results: list[dict], df_filtered: pd.DataFrame, is_cross_model: bool
) -> None:
    st.markdown("### Run Details")

    if df_filtered.empty:
        return

    # Build lookup key → (tier3_score, tier1_pass) from the parsed DataFrame
    if is_cross_model and df_filtered["model_id"].notna().any():
        score_lookup: dict = {
            (row["mas_id"], row["payload_id"], row["phase"], row["model_id"]): row
            for _, row in df_filtered.iterrows()
        }
    else:
        score_lookup = {
            (row["mas_id"], row["payload_id"], row["phase"], None): row
            for _, row in df_filtered.iterrows()
        }

    def _result_key(r: dict) -> tuple:
        if is_cross_model:
            return (
                r.get("mas_id"),
                r.get("payload_id"),
                r.get("phase"),
                r.get("model_id"),
            )
        return (r.get("mas_id"), r.get("payload_id"), r.get("phase"), None)

    visible = [r for r in results if _result_key(r) in score_lookup]

    sev_rank = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
    visible.sort(
        key=lambda r: (
            sev_rank.get(r.get("severity", ""), 99),
            r.get("attack_suite", ""),
            r.get("payload_id", ""),
            r.get("phase", ""),
        )
    )

    for r in visible:
        row = score_lookup.get(_result_key(r), {})
        tier3_score: int | None = row.get("tier3_score")
        tier1_ok: bool = bool(row.get("tier1_pass", False))

        t1_icon = "✅" if tier1_ok else "❌"
        t3_icon = (
            _TIER3_ICON.get(tier3_score, "⚫") if tier3_score is not None else "⬜"
        )
        t3_badge = f"T3:{tier3_score}" if tier3_score is not None else "T3:—"

        # Tier 2/3 disagreement flag
        influenced = r.get("influenced_agents", [])
        disagree = _tier2_tier3_disagree(influenced, tier3_score)
        disagree_badge = " ⚠️" if disagree else ""

        label = (
            f"{t1_icon} {t3_icon} [{t3_badge}]{disagree_badge} "
            f"`{r.get('payload_id', '?')}` ({r.get('phase', '?')}) "
            f"— `{r.get('mas_id', '?')}` "
            f"[{r.get('severity', '?')} / {r.get('attack_suite', '?')}]"
        )
        if is_cross_model and r.get("model_id"):
            label += f" · `{r['model_id']}`"

        with st.expander(label, expanded=False):
            _render_expander_content(r, tier3_score, tier1_ok, disagree, is_cross_model)


def _tier2_tier3_disagree(influenced_agents: list, tier3_score: int | None) -> bool:
    """Return True if Tier-2 heuristic and Tier-3 semantic verdict disagree.

    Disagreement cases:
    - Tier-2 says influenced (influenced_agents non-empty) but Tier-3 score == 0
    - Tier-2 says not influenced (influenced_agents empty) but Tier-3 score >= 2
    """
    if tier3_score is None:
        return False
    tier2_influenced = bool(influenced_agents)
    if tier2_influenced and tier3_score == 0:
        return True
    if not tier2_influenced and tier3_score >= 2:
        return True
    return False


def _render_expander_content(
    r: dict,
    tier3_score: int | None,
    tier1_ok: bool,
    disagree: bool,
    is_cross_model: bool,
) -> None:
    """Render the body of a run detail expander."""
    # Header metadata row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Severity:** `{r.get('severity', '?')}`")
    c2.markdown(f"**Phase:** `{r.get('injection_phase', r.get('phase', '?'))}`")
    c3.markdown(f"**Type:** `{r.get('injection_type', '?')}`")
    c4.markdown(f"**Stub:** {'Yes' if r.get('stub_mode') else 'No'}")
    st.caption(
        f"Suite: {r.get('attack_suite', '?')}  ·  "
        f"Run at {r.get('timestamp', 'unknown')}  ·  "
        f"Duration: {r.get('duration_ms', 0.0):.0f} ms"
    )

    if is_cross_model and r.get("model_id"):
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"**Model ID:** `{r['model_id']}`")
        m2.markdown(f"**Model Name:** {r.get('model_name', '?')}")
        m3.markdown(f"**Provider:** `{r.get('provider_family', '?')}`")

    # Tier 2 propagation
    prop_path = r.get("propagation_path", [])
    influenced = r.get("influenced_agents", [])
    resistant = r.get("resistant_agents", [])

    st.markdown("---")
    if prop_path:
        st.markdown("**Propagation Path:** " + " → ".join(f"`{a}`" for a in prop_path))
    col_inf, col_res = st.columns(2)
    col_inf.markdown(
        "**Influenced (T2):** "
        + (", ".join(f"`{a}`" for a in influenced) if influenced else "none")
    )
    col_res.markdown(
        "**Resistant (T2):** "
        + (", ".join(f"`{a}`" for a in resistant) if resistant else "none")
    )

    # Tier 2 / Tier 3 disagreement callout
    if disagree:
        t3_direction = (
            "score 0 (no compliance)" if tier3_score == 0 else f"score {tier3_score}"
        )
        t2_direction = "influenced" if influenced else "not influenced"
        st.warning(
            f"⚠️ **Tier-2/Tier-3 disagreement** — "
            f"Tier-2 heuristic says **{t2_direction}** but Tier-3 semantic verdict is **{t3_direction}**. "
            f"Manual review recommended."
        )

    # Tier 3 semantic score
    st.markdown("---")
    if tier3_score is not None:
        confidence = r.get("tier3_confidence", "")
        reasoning = r.get("tier3_reasoning", "")
        s1, s2 = st.columns(2)
        t3_icon = _TIER3_ICON.get(tier3_score, "⚫")
        s1.markdown(f"**Tier-3 Score:** {t3_icon} `{tier3_score}`")
        s2.markdown(f"**Confidence:** `{confidence}`")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")
    else:
        st.caption("Tier-3 evaluation skipped (stub mode or evaluator not configured).")

    # View MAS graph button
    config_path = r.get("config_path", "")
    if config_path:
        _render_view_graph_button(
            config_path,
            r.get("mas_id", ""),
            r.get("payload_id", ""),
            r.get("phase", ""),
        )


def _render_view_graph_button(
    config_path: str, mas_id: str, payload_id: str, phase: str
) -> None:
    """Render a 'View MAS graph →' button that loads the config into the visualizer."""
    st.markdown("---")
    full_path = _TESTS_DIR.parent.parent.parent / config_path  # repo root / config_path

    if not full_path.exists():
        st.caption(f"Config not found at `{config_path}` — graph view unavailable.")
        return

    button_key = (
        f"view_graph_{mas_id}_{payload_id}_{phase}_{config_path.replace('/', '_')}"
    )
    if st.button("View MAS graph →", key=button_key, use_container_width=False):
        try:
            from bili.aether.config.loader import (  # pylint: disable=import-outside-toplevel
                load_mas_from_yaml,
            )

            config = load_mas_from_yaml(str(full_path))
            st.session_state.mas_config = config
            st.session_state.current_yaml_path = str(full_path)
            st.session_state.aether_page = "Visualizer"
            # Navigate to the AETHER page
            st.switch_page("aether")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Could not load config: {exc}")
