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
