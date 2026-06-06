"""Tests for bili.aether.ui.results_page -- Baseline Results page.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported

from streamlit.testing.v1 import AppTest

from bili.aether.ui import results_page as rp_mod


def test_empty_state_shows_info_message():
    """When no baseline results exist the page shows an info message."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
with patch.object(rp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(rp, "_load_baseline_results", return_value=[]):
        rp._render_main()
"""
    )
    at.run()
    assert not at.exception
    assert "No baseline results" in " ".join(m.value for m in at.info)


def test_main_renders_aegis_heading():
    """The main area renders the AEGIS Baseline Results heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
with patch.object(rp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(rp, "_load_baseline_results", return_value=[]):
        rp._render_main()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS Baseline Results" in " ".join(m.value for m in at.markdown)


def test_sidebar_renders_aegis_heading():
    """The sidebar contains the AEGIS heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import results_page as rp
with st.sidebar:
    with patch.object(rp, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        rp._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)


def test_sidebar_shows_runner_commands():
    """The sidebar shows baseline runner command examples."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import results_page as rp
with st.sidebar:
    with patch.object(rp, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        rp._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "run_baseline.py" in " ".join(m.value for m in at.sidebar.markdown)


def test_render_results_page_no_exception():
    """The full render_results_page runs without exception."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
with patch.object(rp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(rp, "_load_baseline_results", return_value=[]):
        rp.render_results_page()
"""
    )
    at.run()
    assert not at.exception


def test_build_dataframe_creates_correct_columns():
    """_build_dataframe creates a DataFrame with expected columns."""
    results = [
        {
            "mas_id": "t",
            "prompt_id": "p1",
            "prompt_category": "benign",
            "execution": {"success": True, "duration_ms": 100, "agent_count": 2},
            "run_metadata": {"stub_mode": True, "timestamp": "2025-01-01"},
        }
    ]
    df = rp_mod._build_dataframe(results)
    assert {"mas_id", "prompt_id", "category", "success"}.issubset(set(df.columns))
    assert len(df) == 1


def test_build_dataframe_skips_malformed():
    """_build_dataframe skips results with missing keys."""
    df = rp_mod._build_dataframe([{"mas_id": "x"}])
    assert len(df) == 0


def test_build_baseline_export_df_renames_success():
    """_build_baseline_export_df renames success to tier1_success."""
    results = [
        {
            "mas_id": "t",
            "prompt_id": "p1",
            "prompt_category": "benign",
            "execution": {"success": True, "duration_ms": 100, "agent_count": 2},
            "run_metadata": {"stub_mode": True, "timestamp": "2025-01-01"},
        }
    ]
    df = rp_mod._build_dataframe(results)
    export_df = rp_mod._build_baseline_export_df(df)
    assert "tier1_success" in export_df.columns
    assert "success" not in export_df.columns


# ---------------------------------------------------------------------------
# _render_matrix with data
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = [
    {
        "mas_id": "cfg_a",
        "prompt_id": "p1",
        "prompt_text": "Hello",
        "prompt_category": "benign",
        "execution": {
            "success": True,
            "duration_ms": 50,
            "agent_count": 2,
        },
        "run_metadata": {
            "stub_mode": True,
            "timestamp": "2026-01-01T00:00:00",
        },
        "agent_outputs": {
            "agent_0": {"raw": "Hello back"},
        },
    },
    {
        "mas_id": "cfg_a",
        "prompt_id": "p2",
        "prompt_text": "Bad request",
        "prompt_category": "violating",
        "execution": {
            "success": False,
            "duration_ms": 120,
            "agent_count": 2,
        },
        "run_metadata": {
            "stub_mode": True,
            "timestamp": "2026-01-01T00:01:00",
        },
        "agent_outputs": {},
    },
]


def test_render_matrix_with_data():
    """_render_matrix renders a pivot table with pass/fail data."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
rp._render_matrix(df)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Results Matrix" in all_md


def test_render_matrix_empty():
    """_render_matrix shows info when dataframe is empty."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import results_page as rp
rp._render_matrix(pd.DataFrame())
"""
    )
    at.run()
    assert not at.exception
    assert "No results" in " ".join(m.value for m in at.info)


# ---------------------------------------------------------------------------
# _render_detail_panel with results
# ---------------------------------------------------------------------------


def test_render_detail_panel_with_results():
    """_render_detail_panel renders expandable per-run details."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
rp._render_detail_panel(_SAMPLE_RESULTS, df)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Run Details" in all_md


def test_render_detail_panel_empty():
    """_render_detail_panel handles empty filtered dataframe."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import results_page as rp
rp._render_detail_panel([], pd.DataFrame())
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_filters filter application
# ---------------------------------------------------------------------------


def test_render_filters_returns_filtered_df():
    """_render_filters renders filter widgets and returns data."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
filtered = rp._render_filters(df)
st.markdown(f"count:{len(filtered)}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:" in all_md


# ---------------------------------------------------------------------------
# _render_export_buttons
# ---------------------------------------------------------------------------


def test_render_export_buttons_with_data():
    """_render_export_buttons renders download buttons."""
    at = AppTest.from_string(
        """
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
rp._render_export_buttons(_SAMPLE_RESULTS, df)
"""
    )
    at.run()
    assert not at.exception


def test_render_export_buttons_empty():
    """_render_export_buttons is a no-op for empty data."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import results_page as rp
rp._render_export_buttons([], pd.DataFrame())
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_summary_metrics
# ---------------------------------------------------------------------------


def test_render_summary_metrics():
    """_render_summary_metrics renders metric cards."""
    at = AppTest.from_string(
        """
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
rp._render_summary_metrics(df)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_main with results
# ---------------------------------------------------------------------------


def test_render_main_with_results():
    """_render_main renders the full page when results exist."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
with patch.object(rp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(
        rp, "_load_baseline_results", return_value=_SAMPLE_RESULTS
    ):
        rp._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "AEGIS Baseline Results" in all_md


def test_build_dataframe_multiple_results():
    """_build_dataframe handles multiple results correctly."""
    df = rp_mod._build_dataframe(_SAMPLE_RESULTS)
    assert len(df) == 2
    assert set(df["mas_id"].unique()) == {"cfg_a"}
    assert df["success"].sum() == 1


# ---------------------------------------------------------------------------
# _load_baseline_results disk-reading (lines 58-76)
# ---------------------------------------------------------------------------


def _write_baseline_file(path, mas_id, prompt_id, success):
    """Write a minimal baseline result JSON file at *path*."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mas_id": mas_id,
                "prompt_id": prompt_id,
                "prompt_category": "benign",
                "execution": {
                    "success": success,
                    "duration_ms": 10,
                    "agent_count": 1,
                },
                "run_metadata": {"stub_mode": True, "timestamp": "t"},
            }
        ),
        encoding="utf-8",
    )


def test_load_baseline_results_versioned_and_legacy(tmp_path):
    """_load_baseline_results derives run_id from versioned and legacy layouts."""
    from unittest.mock import patch

    # Versioned layout: {mas_id}/run_001/{prompt}.json
    _write_baseline_file(
        tmp_path / "cfg_a" / "run_001" / "p1.json", "cfg_a", "p1", True
    )
    # Legacy flat layout: {mas_id}/{prompt}.json
    _write_baseline_file(tmp_path / "cfg_b" / "p2.json", "cfg_b", "p2", False)
    with patch.object(rp_mod, "BASELINE_RESULTS_DIR", tmp_path):
        results = rp_mod._load_baseline_results.__wrapped__()
    by_prompt = {r["prompt_id"]: r for r in results}
    assert by_prompt["p1"]["run_id"] == "run_001"
    assert by_prompt["p2"]["run_id"] == "run_000 (legacy)"


def test_load_baseline_results_skips_malformed(tmp_path):
    """_load_baseline_results logs and skips a file that is not valid JSON."""
    from unittest.mock import patch

    (tmp_path / "cfg_a").mkdir(parents=True)
    (tmp_path / "cfg_a" / "broken.json").write_text("{not json", encoding="utf-8")
    _write_baseline_file(tmp_path / "cfg_a" / "ok.json", "cfg_a", "good", True)
    with patch.object(rp_mod, "BASELINE_RESULTS_DIR", tmp_path):
        results = rp_mod._load_baseline_results.__wrapped__()
    # Only the valid file is returned; the broken file is skipped.
    assert len(results) == 1
    assert results[0]["prompt_id"] == "good"


# ---------------------------------------------------------------------------
# Sidebar logo branch (line 134)
# ---------------------------------------------------------------------------


def test_sidebar_renders_logo_when_present():
    """_render_sidebar calls st.image when the logo file exists."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import results_page as rp
with st.sidebar:
    with patch.object(rp, "LOGO_PATH") as lp:
        lp.exists.return_value = True
        lp.__str__ = lambda self: "/fake/logo.png"
        with patch("streamlit.image") as img:
            rp._render_sidebar()
            st.markdown(f"image_called:{img.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "image_called:True" in " ".join(m.value for m in at.sidebar.markdown)


# ---------------------------------------------------------------------------
# Status filter branches (lines 340, 342)
# ---------------------------------------------------------------------------


def test_render_filters_status_passed():
    """_render_filters with status 'Passed' keeps only successful rows."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
st.session_state["baseline_filter_status"] = "Passed"
filtered = rp._render_filters(df)
st.markdown(f"count:{len(filtered)}")
st.markdown(f"allpass:{bool(filtered['success'].all())}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:1" in all_md
    assert "allpass:True" in all_md


def test_render_filters_status_failed():
    """_render_filters with status 'Failed' keeps only failed rows."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SAMPLE_RESULTS
df = rp._build_dataframe(_SAMPLE_RESULTS)
st.session_state["baseline_filter_status"] = "Failed"
filtered = rp._render_filters(df)
st.markdown(f"count:{len(filtered)}")
st.markdown(f"anypass:{bool(filtered['success'].any())}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:1" in all_md
    assert "anypass:False" in all_md


# ---------------------------------------------------------------------------
# Matrix NaN cell rendering (lines 366, 376)
# ---------------------------------------------------------------------------

_SPARSE_RESULTS = [
    {
        "mas_id": "cfg_a",
        "prompt_id": "p1",
        "prompt_text": "Hello",
        "prompt_category": "benign",
        "execution": {"success": True, "duration_ms": 50, "agent_count": 2},
        "run_metadata": {"stub_mode": True, "timestamp": "2026-01-01T00:00:00"},
        "agent_outputs": {},
    },
    {
        "mas_id": "cfg_b",
        "prompt_id": "p2",
        "prompt_text": "Other",
        "prompt_category": "benign",
        "execution": {"success": False, "duration_ms": 60, "agent_count": 2},
        "run_metadata": {"stub_mode": True, "timestamp": "2026-01-01T00:00:00"},
        "agent_outputs": {},
    },
]


def test_render_matrix_renders_not_run_cells():
    """_render_matrix shows a dash for prompt/config combos that were not run."""
    at = AppTest.from_string(
        """
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _SPARSE_RESULTS
df = rp._build_dataframe(_SPARSE_RESULTS)
rp._render_matrix(df)
"""
    )
    at.run()
    assert not at.exception
    # p1 was only run on cfg_a, p2 only on cfg_b: the pivot has NaN cells
    # rendered as the not-run dash symbol with gray cell styling.
    assert len(at.dataframe) == 1


# ---------------------------------------------------------------------------
# Detail panel agent output empty (line 443)
# ---------------------------------------------------------------------------

_RESULTS_EMPTY_AGENT_OUTPUT = [
    {
        "mas_id": "cfg_a",
        "prompt_id": "p1",
        "prompt_text": "Hello",
        "prompt_category": "benign",
        "execution": {"success": True, "duration_ms": 50, "agent_count": 1},
        "run_metadata": {"stub_mode": True, "timestamp": "2026-01-01T00:00:00"},
        "agent_outputs": {"agent_0": {"raw": "   "}},
    },
]


def test_render_detail_panel_empty_agent_output():
    """_render_detail_panel shows '(no output)' when an agent output is blank."""
    at = AppTest.from_string(
        """
from bili.aether.ui import results_page as rp
from bili.aether.ui.tests.test_results_page import _RESULTS_EMPTY_AGENT_OUTPUT
df = rp._build_dataframe(_RESULTS_EMPTY_AGENT_OUTPUT)
rp._render_detail_panel(_RESULTS_EMPTY_AGENT_OUTPUT, df)
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "(no output)" in all_captions
