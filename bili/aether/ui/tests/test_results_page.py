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
