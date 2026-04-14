"""Tests for bili.aether.ui.baseline_runner_page -- Baseline Runner GUI.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported


from streamlit.testing.v1 import AppTest

from bili.aether.ui import baseline_runner_page as brp_mod

# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def test_render_baseline_runner_page_no_exception():
    """The full render_baseline_runner_page runs without exception."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
with patch.object(brp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(brp, "EXAMPLES_DIR") as ed:
        ed.exists.return_value = False
        brp.render_baseline_runner_page()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Sidebar rendering
# ---------------------------------------------------------------------------


def test_sidebar_renders_aegis_heading():
    """The sidebar renders the AEGIS heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
with st.sidebar:
    with patch.object(brp, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        brp._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)


def test_sidebar_shows_baseline_runner_heading():
    """The sidebar shows Baseline Runner section heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
with st.sidebar:
    with patch.object(brp, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        brp._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.sidebar.markdown)
    assert "Baseline Runner" in all_md


# ---------------------------------------------------------------------------
# Main area rendering -- no config
# ---------------------------------------------------------------------------


def test_main_no_config_shows_info():
    """_render_main shows info when no config is loaded."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    brp._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_info = " ".join(m.value for m in at.info)
    assert "No config" in all_info or "config" in all_info.lower()


def test_main_renders_heading():
    """_render_main renders the Baseline Runner heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    brp._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "AEGIS Baseline Runner" in all_md


# ---------------------------------------------------------------------------
# Main area rendering -- with config
# ---------------------------------------------------------------------------


def test_main_with_config_shows_config_info():
    """_render_main with a loaded config shows the config mas_id."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="baseline_test")
st.session_state.baseline_config = cfg
st.session_state.baseline_yaml_path = "/fake/test.yaml"
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    brp._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "baseline_test" in all_md


def test_main_with_config_shows_run_button():
    """_render_main with a config shows the Run Baseline button."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="button_test")
st.session_state.baseline_config = cfg
st.session_state.baseline_yaml_path = "/fake/test.yaml"
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    brp._render_main()
"""
    )
    at.run()
    assert not at.exception
    button_labels = [b.label for b in at.button]
    assert any("Run Baseline" in lbl for lbl in button_labels)


# ---------------------------------------------------------------------------
# push_config_to_baseline_state
# ---------------------------------------------------------------------------


def test_push_config_sets_session_state():
    """push_config_to_baseline_state stores config in session state."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.baseline_runner_page import push_config_to_baseline_state
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="push_test")
push_config_to_baseline_state(cfg, "/fake/path.yaml")
st.markdown(f"config_set:{st.session_state.get('baseline_config') is not None}")
st.markdown(f"path:{st.session_state.get('baseline_yaml_path')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "config_set:True" in all_md
    assert "path:/fake/path.yaml" in all_md


def test_push_config_clears_previous_results():
    """push_config_to_baseline_state clears prior run results."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.baseline_runner_page import push_config_to_baseline_state
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.baseline_run_results = [{"some": "data"}]
cfg = mk()
push_config_to_baseline_state(cfg, "/fake/path.yaml")
st.markdown(f"cleared:{'baseline_run_results' not in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_previous_results
# ---------------------------------------------------------------------------


def test_render_previous_results_no_results():
    """_render_previous_results does nothing when no results in session state."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.baseline_runner_page import _render_previous_results
st.session_state.pop("baseline_run_results", None)
_render_previous_results()
st.markdown("no_output:True")
"""
    )
    at.run()
    assert not at.exception
    # Only our marker markdown should be present
    all_md = " ".join(m.value for m in at.markdown)
    assert "no_output:True" in all_md
    assert "Last Run Summary" not in all_md


def test_render_previous_results_with_data():
    """_render_previous_results shows summary when results exist."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.baseline_runner_page import _render_previous_results
st.session_state.baseline_run_results = [
    {"execution": {"success": True}},
    {"execution": {"success": False}},
]
_render_previous_results()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Last Run Summary" in all_md


# ---------------------------------------------------------------------------
# _init_prompt_selections
# ---------------------------------------------------------------------------


def test_init_prompt_selections_sets_defaults():
    """_init_prompt_selections sets default True for all prompts."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.baseline_runner_page import _init_prompt_selections
_init_prompt_selections()
# Check that at least one baseline_prompt_ key exists
has_keys = any(k.startswith("baseline_prompt_") for k in st.session_state)
st.markdown(f"has_keys:{has_keys}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_keys:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------


def test_category_constants_exist():
    """Module-level category constants are properly defined."""
    assert "benign" in brp_mod._CATEGORY_ORDER
    assert "violating" in brp_mod._CATEGORY_ORDER
    assert "edge_case" in brp_mod._CATEGORY_ORDER
    assert len(brp_mod._CATEGORY_LABELS) >= 3
