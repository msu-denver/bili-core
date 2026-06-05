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


# ---------------------------------------------------------------------------
# _on_cat_header_change and _set_cat_prompts (lines 81-83, 88-89)
# ---------------------------------------------------------------------------


def test_on_cat_header_change_propagates():
    """_on_cat_header_change pushes the header value to all child prompt keys."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
st.session_state["hdr"] = False
st.session_state["baseline_prompt_p1"] = True
st.session_state["baseline_prompt_p2"] = True
brp._on_cat_header_change("hdr", ["p1", "p2"])
st.markdown(f"p1:{st.session_state['baseline_prompt_p1']}")
st.markdown(f"p2:{st.session_state['baseline_prompt_p2']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "p1:False" in all_md
    assert "p2:False" in all_md


def test_set_cat_prompts_sets_all():
    """_set_cat_prompts writes the value to every prompt key in the category."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
brp._set_cat_prompts(["x1", "x2"], True)
st.markdown(f"x1:{st.session_state['baseline_prompt_x1']}")
st.markdown(f"x2:{st.session_state['baseline_prompt_x2']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "x1:True" in all_md
    assert "x2:True" in all_md


# ---------------------------------------------------------------------------
# Sidebar logo branch (line 115)
# ---------------------------------------------------------------------------


def test_sidebar_renders_logo_when_present():
    """_render_sidebar calls st.image when the logo file exists."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
with st.sidebar:
    with patch.object(brp, "LOGO_PATH") as lp:
        lp.exists.return_value = True
        lp.__str__ = lambda self: "/fake/logo.png"
        with patch("streamlit.image") as img:
            brp._render_sidebar()
            st.markdown(f"image_called:{img.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "image_called:True" in " ".join(m.value for m in at.sidebar.markdown)


# ---------------------------------------------------------------------------
# _render_main partial-selection caption (line 179) + stub caption (254)
# ---------------------------------------------------------------------------


def test_main_partial_selection_caption():
    """_render_main shows 'n of total' caption when not all prompts selected."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
from bili.aegis.suites.baseline.prompts.baseline_prompts import BASELINE_PROMPTS
cfg = mk(mas_id="partial_test")
st.session_state.baseline_config = cfg
st.session_state.baseline_yaml_path = "/fake/test.yaml"
# Deselect the first prompt so the selection is partial.
st.session_state[f"baseline_prompt_{BASELINE_PROMPTS[0].prompt_id}"] = False
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    brp._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_caps = " ".join(c.value for c in at.caption)
    assert "of" in all_caps and "prompts selected" in all_caps


def test_prompt_selector_stub_caption():
    """_render_prompt_selector shows 'No LLM calls' caption when stub toggled on."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import baseline_runner_page as brp
st.session_state["baseline_stub_mode"] = True
stub = brp._render_prompt_selector()
st.markdown(f"stub:{stub}")
"""
    )
    at.run()
    assert not at.exception
    assert "stub:True" in " ".join(m.value for m in at.markdown)
    assert "No LLM calls" in " ".join(c.value for c in at.caption)


# ---------------------------------------------------------------------------
# _resolve_config file-selected branch (lines 215-227)
# ---------------------------------------------------------------------------


def test_resolve_config_loads_selected_yaml(tmp_path):
    """_resolve_config loads the chosen YAML and stores it in session state."""
    (tmp_path / "demo_config.yaml").write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch, MagicMock
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
import bili.aether.config.loader as loader_mod
fake_cfg = MagicMock()
st.session_state["baseline_yaml_selector"] = 1
with patch.object(brp, "EXAMPLES_DIR", Path({str(tmp_path)!r})):
    with patch.object(loader_mod, "load_mas_from_yaml", return_value=fake_cfg):
        config, path = brp._resolve_config()
st.markdown(f"loaded:{{config is not None}}")
st.markdown(f"has_path:{{path.endswith('demo_config.yaml')}}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "loaded:True" in all_md
    assert "has_path:True" in all_md


def test_resolve_config_load_error(tmp_path):
    """_resolve_config surfaces an error and returns (None, None) on load failure."""
    (tmp_path / "broken_config.yaml").write_text("bad", encoding="utf-8")
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
import bili.aether.config.loader as loader_mod
st.session_state["baseline_yaml_selector"] = 1
with patch.object(brp, "EXAMPLES_DIR", Path({str(tmp_path)!r})):
    with patch.object(loader_mod, "load_mas_from_yaml",
                      side_effect=ValueError("bad yaml")):
        config, path = brp._resolve_config()
st.markdown(f"none:{{config is None and path is None}}")
"""
    )
    at.run()
    assert not at.exception
    assert "none:True" in " ".join(m.value for m in at.markdown)
    assert "Failed to load" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _execute_run (lines 326-398) + run-button click (line 188)
# ---------------------------------------------------------------------------


def test_execute_run_no_prompts_selected(tmp_path):
    """_execute_run warns when every prompt is deselected."""
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
from bili.aegis.suites.baseline.prompts.baseline_prompts import BASELINE_PROMPTS
cfg = mk(mas_id="no_prompts")
for p in BASELINE_PROMPTS:
    st.session_state[f"baseline_prompt_{{p.prompt_id}}"] = False
with patch.object(brp, "_BASELINE_RESULTS_DIR", Path({str(tmp_path)!r})):
    brp._execute_run(cfg, "/fake/path.yaml", stub_mode=True)
"""
    )
    at.run()
    assert not at.exception
    assert "No prompts selected" in " ".join(w.value for w in at.warning)


def test_execute_run_writes_results(tmp_path):
    """_execute_run runs selected prompts, writes results, and shows progress."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    # Stub the heavy run_baseline + helpers imports done inside _execute_run.
    runner_mod = types.ModuleType("bili.aegis.suites.baseline.run_baseline")
    runner_mod.run_one = MagicMock(
        return_value={
            "execution": {"success": True, "duration_ms": 12.0},
        }
    )
    runner_mod.write_result = MagicMock()
    helpers_mod = types.ModuleType("bili.aegis.suites._helpers")
    helpers_mod.next_run_dir = MagicMock(return_value=tmp_path / "run_001")
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites.baseline.run_baseline": runner_mod,
            "bili.aegis.suites._helpers": helpers_mod,
        },
    ):
        at = AppTest.from_string(
            f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
from bili.aegis.suites.baseline.prompts.baseline_prompts import BASELINE_PROMPTS
cfg = mk(mas_id="exec_test", model_name="gpt-4o")
# Select only the first prompt so the run is small and deterministic.
for p in BASELINE_PROMPTS:
    st.session_state[f"baseline_prompt_{{p.prompt_id}}"] = False
st.session_state[f"baseline_prompt_{{BASELINE_PROMPTS[0].prompt_id}}"] = True
with patch.object(brp, "_BASELINE_RESULTS_DIR", Path({str(tmp_path)!r})):
    brp._execute_run(cfg, "/fake/path.yaml", stub_mode=False)
st.markdown(f"stored:{{'baseline_run_results' in st.session_state}}")
"""
        )
        at.run()
    assert not at.exception
    assert runner_mod.run_one.called
    assert runner_mod.write_result.called
    all_md = " ".join(m.value for m in at.markdown)
    assert "stored:True" in all_md
    assert "All 1 prompts passed" in " ".join(s.value for s in at.success)


def test_execute_run_handles_prompt_failure(tmp_path):
    """_execute_run records a failed prompt and shows the failure warning."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    runner_mod = types.ModuleType("bili.aegis.suites.baseline.run_baseline")
    runner_mod.run_one = MagicMock(side_effect=RuntimeError("run blew up"))
    runner_mod.write_result = MagicMock()
    helpers_mod = types.ModuleType("bili.aegis.suites._helpers")
    helpers_mod.next_run_dir = MagicMock(return_value=tmp_path / "run_002")
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites.baseline.run_baseline": runner_mod,
            "bili.aegis.suites._helpers": helpers_mod,
        },
    ):
        at = AppTest.from_string(
            f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
from bili.aegis.suites.baseline.prompts.baseline_prompts import BASELINE_PROMPTS
cfg = mk(mas_id="fail_test")
for p in BASELINE_PROMPTS:
    st.session_state[f"baseline_prompt_{{p.prompt_id}}"] = False
st.session_state[f"baseline_prompt_{{BASELINE_PROMPTS[0].prompt_id}}"] = True
with patch.object(brp, "_BASELINE_RESULTS_DIR", Path({str(tmp_path)!r})):
    brp._execute_run(cfg, "/fake/path.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    assert runner_mod.run_one.called
    assert "0/1 passed" in " ".join(w.value for w in at.warning)


def test_main_run_button_click_invokes_execute(tmp_path):
    """Clicking the Run Baseline button invokes _execute_run."""
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import baseline_runner_page as brp
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="click_run")
st.session_state.baseline_config = cfg
st.session_state.baseline_yaml_path = "/fake/test.yaml"
with patch.object(brp, "EXAMPLES_DIR") as ed:
    ed.exists.return_value = False
    with patch.object(brp, "_execute_run") as run:
        brp._render_main()
        st.session_state["__exec_mock"] = run
"""
    )
    at.run()
    assert not at.exception
    run_buttons = [b for b in at.button if "Run Baseline" in b.label]
    assert run_buttons
    run_buttons[0].click().run()
    assert not at.exception
