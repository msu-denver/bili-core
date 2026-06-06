"""Tests for bili.aether.ui.attack_page -- Interactive Attack Suite.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported

from streamlit.testing.v1 import AppTest

from bili.aether.ui import attack_page as ap_mod
from bili.aether.ui.tests.conftest import make_test_config


def test_no_config_shows_info_message():
    """Without a config the page shows an info message."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    assert "No MAS loaded" in " ".join(m.value for m in at.info)


def test_main_renders_aegis_heading():
    """The main area renders the AEGIS Attack Suite heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS Attack Suite" in " ".join(m.value for m in at.markdown)


def test_sidebar_no_config_shows_yaml_selector():
    """The sidebar shows YAML selector when no config exists."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import attack_page as ap
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(ap, "EXAMPLES_DIR") as ed:
            ed.exists.return_value = False
            ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)


def test_sidebar_renders_aegis_heading():
    """The sidebar renders the AEGIS heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import attack_page as ap
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)


def test_push_config_sets_session_state():
    """push_config_to_attack_state stores config in session state."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.attack_page import push_config_to_attack_state
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="push_test")
push_config_to_attack_state(cfg)
st.markdown(f"config_set:{st.session_state.get('attack_config') is not None}")
st.markdown(f"target:{st.session_state.get('attack_target_agent_id')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "config_set:True" in all_md
    assert "target:agent_0" in all_md


def test_push_config_clears_previous_results():
    """push_config_to_attack_state clears prior attack results."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.attack_page import push_config_to_attack_state
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_result = {"some": "data"}
st.session_state.attack_verdict = [{"score": 1}]
cfg = mk()
push_config_to_attack_state(cfg)
st.markdown(f"cleared:{'attack_result' not in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_render_attack_page_no_config_no_exception():
    """The full page renders without exception when no config loaded."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    ap.render_attack_page()
"""
    )
    at.run()
    assert not at.exception


def test_is_stub_mode_true():
    """_is_stub_mode returns True when no agents have model_name."""
    config = make_test_config()
    assert ap_mod._is_stub_mode(config) is True


def test_is_stub_mode_false():
    """_is_stub_mode returns False when agents have model_name."""
    config = make_test_config(model_name="gpt-4o")
    assert ap_mod._is_stub_mode(config) is False


def test_get_notes_returns_notes():
    """_get_notes returns the notes attribute from a payload object."""
    from unittest.mock import MagicMock

    obj = MagicMock()
    obj.notes = "test notes"
    assert ap_mod._get_notes(obj) == "test notes"


def test_get_notes_returns_empty_when_missing():
    """_get_notes returns empty string when notes is absent."""
    from unittest.mock import MagicMock

    obj = MagicMock(spec=[])
    assert ap_mod._get_notes(obj) == ""


def test_render_observation_influenced():
    """_render_observation renders an influenced agent."""
    at = AppTest.from_string(
        """
from bili.aether.ui.attack_page import _render_observation
obs = {
    "agent_id": "a0",
    "influenced": True,
    "resisted": False,
    "received_payload": True,
    "output_excerpt": "I comply",
    "role": "x",
}
_render_observation(obs)
"""
    )
    at.run()
    assert not at.exception


def test_render_observation_clean():
    """_render_observation renders a clean agent."""
    at = AppTest.from_string(
        """
from bili.aether.ui.attack_page import _render_observation
obs = {
    "agent_id": "a1",
    "influenced": False,
    "resisted": False,
    "received_payload": False,
    "output_excerpt": "",
    "role": "y",
}
_render_observation(obs)
"""
    )
    at.run()
    assert not at.exception


def test_render_observation_resisted():
    """_render_observation renders a resisted agent."""
    at = AppTest.from_string(
        """
from bili.aether.ui.attack_page import _render_observation
obs = {
    "agent_id": "a2",
    "influenced": False,
    "resisted": True,
    "received_payload": True,
    "output_excerpt": "I refuse.",
    "role": "z",
}
_render_observation(obs)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _resolve_payload
# ---------------------------------------------------------------------------


def test_resolve_payload_custom_source():
    """_resolve_payload returns custom text when source is Custom."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
st.session_state["attack_payload_source"] = "Custom"
st.session_state["attack_payload_custom"] = "Custom adversarial text"
result = ap._resolve_payload()
st.markdown(f"payload:{result}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "payload:Custom adversarial text" in all_md


def test_resolve_payload_custom_empty():
    """_resolve_payload returns None when custom text is empty."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
st.session_state["attack_payload_source"] = "Custom"
st.session_state["attack_payload_custom"] = "   "
result = ap._resolve_payload()
st.markdown(f"none:{result is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "none:True" in " ".join(m.value for m in at.markdown)


def test_resolve_payload_library_no_pid():
    """_resolve_payload returns None when no payload ID is selected."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
st.session_state["attack_payload_source"] = "Library"
st.session_state["attack_suite"] = "injection"
st.session_state.pop("attack_payload_id", None)
with patch.object(ap, "_load_payload_library", return_value={}):
    result = ap._resolve_payload()
st.markdown(f"none:{result is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "none:True" in " ".join(m.value for m in at.markdown)


def test_resolve_payload_library_with_pid():
    """_resolve_payload returns payload text from library."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
st.session_state["attack_payload_source"] = "Library"
st.session_state["attack_suite"] = "injection"
st.session_state["attack_payload_id"] = "p1"
mock_payload = MagicMock()
mock_payload.payload = "Injected text here"
with patch.object(ap, "_load_payload_library", return_value={"p1": mock_payload}):
    result = ap._resolve_payload()
st.markdown(f"payload:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "payload:Injected text here" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_results with sample attack result
# ---------------------------------------------------------------------------


def test_render_results_tier1_success():
    """_render_results shows success for Tier 1 pass."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk()
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_success = " ".join(m.value for m in at.success)
    assert "Tier 1" in all_success


def test_render_results_tier1_failure():
    """_render_results shows error for Tier 1 failure."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk()
result_dict = {
    "success": False,
    "error": "Timeout occurred",
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_err = " ".join(m.value for m in at.error)
    assert "Timeout" in all_err


def test_render_results_with_observations():
    """_render_results renders agent observations for Tier 2."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk()
result_dict = {
    "success": True,
    "agent_observations": [
        {"agent_id": "a0", "influenced": True, "resisted": False,
         "received_payload": True, "output_excerpt": "Bad output", "role": "x"},
    ],
    "propagation_path": ["a0"],
    "influenced_agents": ["a0"],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Tier 2" in all_md


# ---------------------------------------------------------------------------
# Provider family helpers
# ---------------------------------------------------------------------------


def test_same_provider_family_true():
    """_same_provider_family returns True for same family models."""
    assert ap_mod._same_provider_family("gpt-4o", "gpt-3.5-turbo") is True


def test_same_provider_family_false():
    """_same_provider_family returns False for different family models."""
    assert ap_mod._same_provider_family("gpt-4o", "claude-3-opus") is False


def test_same_provider_family_unknown():
    """_same_provider_family returns False when family is unknown."""
    assert ap_mod._same_provider_family("unknown-model-xyz", "gpt-4o") is False


def test_get_provider_family_openai():
    """_get_provider_family returns openai for gpt models."""
    result = ap_mod._get_provider_family("gpt-4o")
    assert result == "openai"


def test_get_provider_family_unknown():
    """_get_provider_family returns None for unknown models."""
    result = ap_mod._get_provider_family("totally-unknown-model")
    assert result is None


# ---------------------------------------------------------------------------
# _load_payload_library caching behavior
# ---------------------------------------------------------------------------


def test_load_payload_library_missing_module():
    """_load_payload_library returns empty dict when module not found."""
    # Call the underlying function directly (bypassing st.cache_resource)
    from unittest.mock import patch as _patch

    with _patch("importlib.import_module", side_effect=ImportError("not found")):
        result = ap_mod._load_payload_library.__wrapped__("injection")
    assert result == {}


# ---------------------------------------------------------------------------
# _render_main with config loaded
# ---------------------------------------------------------------------------


def test_render_main_with_config_shows_heading():
    """_render_main with a loaded config shows the attack suite heading and config id."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="main_test")
st.session_state.attack_config = cfg
st.session_state.attack_target_agent_id = "agent_0"
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(ap, "render_attack_graph", return_value=None):
        ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "AEGIS Attack Suite" in all_md
    assert "main_test" in all_md


# ---------------------------------------------------------------------------
# _render_sidebar with config loaded
# ---------------------------------------------------------------------------


def test_sidebar_with_config_shows_suite_selector():
    """The sidebar shows a suite selectbox when config is loaded."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="sidebar_test")
st.session_state.attack_config = cfg
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(ap, "_load_payload_library", return_value={}):
            ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert len(at.sidebar.selectbox) >= 1


def test_sidebar_with_config_shows_single_payload_section():
    """The sidebar shows 'Single-payload exploratory attack' when config loaded."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="sidebar_id_test")
st.session_state.attack_config = cfg
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(ap, "EXAMPLES_DIR") as ed:
            ed.exists.return_value = False
            with patch.object(ap, "_load_payload_library", return_value={}):
                ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.sidebar.markdown)
    assert "Single-payload" in all_md


# ---------------------------------------------------------------------------
# _render_results with tier 3 stub mode
# ---------------------------------------------------------------------------


def test_render_results_tier3_stub_skipped():
    """Tier 3 evaluation is skipped in stub mode."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk()
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_info = " ".join(m.value for m in at.info)
    assert "stub mode" in all_info


# ---------------------------------------------------------------------------
# _get_evaluator_model
# ---------------------------------------------------------------------------


def test_get_evaluator_model_default():
    """_get_evaluator_model returns primary model when no selection."""
    result = ap_mod._get_evaluator_model()
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# _load_baseline_result
# ---------------------------------------------------------------------------


def test_load_baseline_result_missing_dir():
    """_load_baseline_result returns None for non-existent mas_id."""
    result = ap_mod._load_baseline_result("completely_nonexistent_mas_99999")
    assert result is None


def test_load_baseline_result_sanitizes_traversal():
    """_load_baseline_result sanitizes path traversal attempts."""
    result = ap_mod._load_baseline_result("../../etc/passwd")
    assert result is None


# ---------------------------------------------------------------------------
# _render_main with result in session state
# ---------------------------------------------------------------------------


def test_render_main_with_result_shows_results():
    """_render_main renders results when attack_result is in session state."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="result_test")
st.session_state.attack_config = cfg
st.session_state.attack_target_agent_id = "agent_0"
st.session_state.attack_result = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(ap, "render_attack_graph", return_value=None):
        with patch.object(ap, "_is_stub_mode", return_value=True):
            ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Attack Results" in all_md


# ---------------------------------------------------------------------------
# _run_attack error paths
# ---------------------------------------------------------------------------


def test_run_attack_no_config():
    """_run_attack returns early when no config loaded."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
st.session_state.pop("attack_config", None)
ap._run_attack()
st.markdown("no_crash:True")
"""
    )
    at.run()
    assert not at.exception
    assert "no_crash:True" in " ".join(m.value for m in at.markdown)


def test_run_attack_no_target():
    """_run_attack shows error when no target agent selected."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_config = mk()
st.session_state.pop("attack_target_agent_id", None)
ap._run_attack()
"""
    )
    at.run()
    assert not at.exception
    assert "No target" in " ".join(e.value for e in at.error)


def test_run_attack_no_payload():
    """_run_attack shows error when no payload text available."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_config = mk()
st.session_state.attack_target_agent_id = "agent_0"
st.session_state.attack_suite = "injection"
with patch.object(ap, "_resolve_payload", return_value=None):
    ap._run_attack()
"""
    )
    at.run()
    assert not at.exception
    assert "No payload" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _render_observation edge case: received but not influenced
# ---------------------------------------------------------------------------


def test_render_observation_received_not_influenced():
    """_render_observation renders a received but not influenced agent."""
    at = AppTest.from_string(
        """
from bili.aether.ui.attack_page import _render_observation
obs = {
    "agent_id": "a3",
    "influenced": False,
    "resisted": False,
    "received_payload": True,
    "output_excerpt": "Some output",
    "role": "w",
}
_render_observation(obs)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_sidebar with config and payload library
# ---------------------------------------------------------------------------


def test_sidebar_with_config_payload_preview():
    """Sidebar shows payload preview when library payload selected."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="preview_test")
st.session_state.attack_config = cfg
mock_payload = MagicMock()
mock_payload.payload = "Adversarial text here"
mock_payload.notes = "Test notes for payload"
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(
            ap, "_load_payload_library",
            return_value={"p1": mock_payload}
        ):
            ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception


def test_sidebar_custom_payload_source():
    """Sidebar shows custom payload text area when Custom selected."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="custom_test")
st.session_state.attack_config = cfg
st.session_state.attack_payload_source = "Custom"
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        ap._render_sidebar()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_results with full attack result dict
# ---------------------------------------------------------------------------


def test_render_results_with_full_observations():
    """_render_results renders multiple observations with propagation metrics."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(num_agents=3)
result_dict = {
    "success": True,
    "agent_observations": [
        {"agent_id": "agent_0", "influenced": True, "resisted": False,
         "received_payload": True, "output_excerpt": "I comply", "role": "role_0"},
        {"agent_id": "agent_1", "influenced": False, "resisted": True,
         "received_payload": True, "output_excerpt": "I refuse", "role": "role_1"},
        {"agent_id": "agent_2", "influenced": False, "resisted": False,
         "received_payload": False, "output_excerpt": "", "role": "role_2"},
    ],
    "propagation_path": ["agent_0", "agent_1"],
    "influenced_agents": ["agent_0"],
    "resistant_agents": ["agent_1"],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Tier 2" in all_md


# ---------------------------------------------------------------------------
# _render_results with tier 3 non-stub, no baseline
# ---------------------------------------------------------------------------


def test_render_results_tier3_no_baseline():
    """_render_results shows info when no baseline found for Tier 3."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(model_name="gpt-4o")
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=False):
    with patch.object(ap, "_load_baseline_result", return_value=None):
        ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_info = " ".join(m.value for m in at.info)
    assert "baseline" in all_info.lower()


# ---------------------------------------------------------------------------
# _render_results with tier 3 circularity warning
# ---------------------------------------------------------------------------


def test_render_results_tier3_circularity_warning():
    """_render_results shows circularity warning when providers match."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(model_name="gpt-4o")
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=False):
    with patch.object(ap, "_load_baseline_result", return_value=None):
        with patch.object(ap, "_get_evaluator_model", return_value="gpt-4o-mini"):
            ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_warn = " ".join(w.value for w in at.warning)
    assert "circular" in all_warn.lower()


# ---------------------------------------------------------------------------
# _render_observation without output excerpt
# ---------------------------------------------------------------------------


def test_render_observation_no_excerpt():
    """_render_observation shows 'no output recorded' when excerpt is empty."""
    at = AppTest.from_string(
        """
from bili.aether.ui.attack_page import _render_observation
obs = {
    "agent_id": "a4",
    "influenced": False,
    "resisted": False,
    "received_payload": False,
    "output_excerpt": "",
    "role": "test_role",
}
_render_observation(obs)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _same_provider_family additional cases
# ---------------------------------------------------------------------------


def test_same_provider_family_anthropic():
    """_same_provider_family identifies Anthropic models."""
    assert ap_mod._same_provider_family("claude-3-opus", "claude-3-haiku") is True


def test_get_provider_family_anthropic():
    """_get_provider_family returns anthropic family for Claude models."""
    result = ap_mod._get_provider_family("claude-3-opus")
    assert result is not None
    assert "anthropic" in result


# ---------------------------------------------------------------------------
# _resolve_payload library with missing payload
# ---------------------------------------------------------------------------


def test_resolve_payload_library_missing_pid_in_library():
    """_resolve_payload returns None when pid not in library."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
st.session_state["attack_payload_source"] = "Library"
st.session_state["attack_suite"] = "injection"
st.session_state["attack_payload_id"] = "nonexistent_pid"
with patch.object(ap, "_load_payload_library", return_value={"p1": None}):
    result = ap._resolve_payload()
st.markdown(f"none:{result is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "none:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_main with node click
# ---------------------------------------------------------------------------


def test_render_main_initializes_target():
    """_render_main initializes target to first agent when not set."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="init_target_test")
st.session_state.attack_config = cfg
st.session_state.pop("attack_target_agent_id", None)
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(ap, "render_attack_graph", return_value=None):
        ap._render_main()
st.markdown(f"target:{st.session_state.get('attack_target_agent_id')}")
"""
    )
    at.run()
    assert not at.exception
    assert "target:agent_0" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# push_config_to_attack_state with no agents
# ---------------------------------------------------------------------------


def test_push_config_multiple_agents():
    """push_config_to_attack_state sets target to first agent."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui.attack_page import push_config_to_attack_state
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(num_agents=3, mas_id="multi_agent_push")
push_config_to_attack_state(cfg)
st.markdown(f"config_set:{st.session_state.get('attack_config') is not None}")
st.markdown(f"target:{st.session_state.get('attack_target_agent_id')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "config_set:True" in all_md
    assert "target:agent_0" in all_md


# ---------------------------------------------------------------------------
# _render_results with error in result dict
# ---------------------------------------------------------------------------


def test_render_results_tier1_error_message():
    """_render_results displays the error message from result dict."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk()
result_dict = {
    "success": False,
    "error": "Connection refused",
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=True):
    ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_err = " ".join(m.value for m in at.error)
    assert "Connection refused" in all_err


# ---------------------------------------------------------------------------
# _on_suite_header_change and _set_suite_payloads (lines 244-246, 251-252)
# ---------------------------------------------------------------------------


def test_on_suite_header_change_propagates():
    """_on_suite_header_change pushes the header value to all child keys."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
st.session_state["hdr"] = False
st.session_state["c1"] = True
st.session_state["c2"] = True
ap._on_suite_header_change("hdr", ["c1", "c2"])
st.markdown(f"c1:{st.session_state['c1']}")
st.markdown(f"c2:{st.session_state['c2']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "c1:False" in all_md
    assert "c2:False" in all_md


def test_set_suite_payloads_sets_all():
    """_set_suite_payloads writes the value to every payload key."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
ap._set_suite_payloads(["k1", "k2"], True)
st.markdown(f"k1:{st.session_state['k1']}")
st.markdown(f"k2:{st.session_state['k2']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "k1:True" in all_md
    assert "k2:True" in all_md


# ---------------------------------------------------------------------------
# _report_suite_result (lines 353-373)
# ---------------------------------------------------------------------------


def test_report_suite_result_all_skipped():
    """_report_suite_result counts an all-skipped suite as passed."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
rows = [{"skipped": "true"}, {"skipped": "true"}]
area = st.container()
passed = ap._report_suite_result(rows, "injection", ["p1"], area, 0)
st.markdown(f"passed:{passed}")
"""
    )
    at.run()
    assert not at.exception
    assert "passed:1" in " ".join(m.value for m in at.markdown)


def test_report_suite_result_all_passed():
    """_report_suite_result counts a suite where every ran row passed."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
rows = [{"skipped": "false", "tier1_pass": "true"}]
area = st.container()
passed = ap._report_suite_result(rows, "injection", ["p1"], area, 0)
st.markdown(f"passed:{passed}")
"""
    )
    at.run()
    assert not at.exception
    assert "passed:1" in " ".join(m.value for m in at.markdown)


def test_report_suite_result_some_failed():
    """_report_suite_result does not increment when a ran row failed."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.aether.ui import attack_page as ap
rows = [{"skipped": "false", "tier1_pass": "false"}]
area = st.container()
passed = ap._report_suite_result(rows, "injection", ["p1"], area, 0)
st.markdown(f"passed:{passed}")
"""
    )
    at.run()
    assert not at.exception
    assert "passed:0" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _resolve_attack_config yaml selection (lines 205-218)
# ---------------------------------------------------------------------------


def test_resolve_attack_config_loads_selected_yaml(tmp_path):
    """_resolve_attack_config loads the chosen YAML into session state."""
    (tmp_path / "demo_config.yaml").write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch, MagicMock
from pathlib import Path
from bili.aether.ui import attack_page as ap
import bili.aether.config.loader as loader_mod
fake_cfg = MagicMock()
fake_cfg.agents = [MagicMock(agent_id="agent_0")]
st.session_state["attack_yaml_selector"] = 1
with patch.object(ap, "EXAMPLES_DIR", Path({str(tmp_path)!r})):
    with patch.object(loader_mod, "load_mas_from_yaml", return_value=fake_cfg):
        ap._resolve_attack_config()
st.markdown(f"loaded:{{st.session_state.get('attack_config') is not None}}")
st.markdown(f"target:{{st.session_state.get('attack_target_agent_id')}}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "loaded:True" in all_md
    assert "target:agent_0" in all_md


def test_resolve_attack_config_load_error(tmp_path):
    """_resolve_attack_config surfaces an error when YAML loading fails."""
    (tmp_path / "broken_config.yaml").write_text("bad", encoding="utf-8")
    at = AppTest.from_string(
        f"""
import streamlit as st
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import attack_page as ap
import bili.aether.config.loader as loader_mod
st.session_state["attack_yaml_selector"] = 1
with patch.object(ap, "EXAMPLES_DIR", Path({str(tmp_path)!r})):
    with patch.object(loader_mod, "load_mas_from_yaml",
                      side_effect=ValueError("bad yaml")):
        ap._resolve_attack_config()
"""
    )
    at.run()
    assert not at.exception
    assert "Failed to load" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# Sidebar logo branch (line 561)
# ---------------------------------------------------------------------------


def test_sidebar_renders_logo_when_present():
    """_render_sidebar calls st.image when the logo file exists."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import attack_page as ap
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = True
        lp.__str__ = lambda self: "/fake/logo.png"
        with patch.object(ap, "EXAMPLES_DIR") as ed:
            ed.exists.return_value = False
            with patch("streamlit.image") as img:
                ap._render_sidebar()
                st.markdown(f"image_called:{img.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "image_called:True" in " ".join(m.value for m in at.sidebar.markdown)


# ---------------------------------------------------------------------------
# _run_mid_execution (lines 953-955)
# ---------------------------------------------------------------------------


def test_run_mid_execution_calls_injector():
    """_run_mid_execution delegates to AttackInjector.inject_attack."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk()
fake_result = MagicMock(name="attack_result")
fake_injector = MagicMock()
fake_injector.inject_attack.return_value = fake_result
with patch.object(ap, "AttackInjector", return_value=fake_injector):
    result = ap._run_mid_execution(cfg, "agent_0", "prompt_injection", "payload")
st.markdown(f"called:{fake_injector.inject_attack.called}")
st.markdown(f"same:{result is fake_result}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "called:True" in all_md
    assert "same:True" in all_md


# ---------------------------------------------------------------------------
# _load_baseline_result file reading (lines 1124-1131)
# ---------------------------------------------------------------------------


def test_load_baseline_result_reads_legacy_file(tmp_path):
    """_load_baseline_result returns the first JSON file in the legacy layout."""
    from unittest.mock import patch

    mas_dir = tmp_path / "mas_a"
    mas_dir.mkdir(parents=True)
    (mas_dir / "p1.json").write_text(
        '{"mas_id": "mas_a", "prompt_id": "p1"}', encoding="utf-8"
    )
    with patch.object(ap_mod, "BASELINE_RESULTS_DIR", tmp_path):
        result = ap_mod._load_baseline_result("mas_a")
    assert result is not None
    assert result["prompt_id"] == "p1"


def test_load_baseline_result_skips_unreadable(tmp_path):
    """_load_baseline_result skips an unreadable file and warns."""
    at = AppTest.from_string(
        f"""
from unittest.mock import patch
from pathlib import Path
from bili.aether.ui import attack_page as ap
mas_dir = Path({str(tmp_path)!r}) / "mas_a"
mas_dir.mkdir(parents=True, exist_ok=True)
(mas_dir / "bad.json").write_text("{{not json", encoding="utf-8")
with patch.object(ap, "BASELINE_RESULTS_DIR", Path({str(tmp_path)!r})):
    result = ap._load_baseline_result("mas_a")
import streamlit as st
st.markdown(f"result_none:{{result is None}}")
"""
    )
    at.run()
    assert not at.exception
    assert "result_none:True" in " ".join(m.value for m in at.markdown)
    assert "unreadable" in " ".join(w.value for w in at.warning)


# ---------------------------------------------------------------------------
# _run_tier3_evaluation (lines 1093-1104)
# ---------------------------------------------------------------------------


def test_run_tier3_evaluation_success():
    """_run_tier3_evaluation reconstructs the result and calls the evaluator."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.aether.ui import attack_page as ap
import bili.aegis.evaluator.semantic_evaluator as se_mod
fake_eval = MagicMock()
fake_eval.evaluate.return_value = [MagicMock(name="verdict")]
result_dict = {"some": "value"}
baseline = {"baseline": "value"}
with patch.object(ap.AttackResult, "model_validate", return_value=MagicMock()):
    with patch.object(se_mod, "SemanticEvaluator", return_value=fake_eval):
        verdicts = ap._run_tier3_evaluation(result_dict, baseline)
st.markdown(f"count:{len(verdicts)}")
"""
    )
    at.run()
    assert not at.exception
    assert "count:1" in " ".join(m.value for m in at.markdown)


def test_run_tier3_evaluation_error_returns_none():
    """_run_tier3_evaluation returns None and shows an error on failure."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import attack_page as ap
with patch.object(ap.AttackResult, "model_validate",
                  side_effect=ValueError("bad result")):
    verdicts = ap._run_tier3_evaluation({}, {})
st.markdown(f"none:{verdicts is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "none:True" in " ".join(m.value for m in at.markdown)
    assert "Tier 3 evaluation failed" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _render_results Tier-3 verdict rendering (lines 1016-1041)
# ---------------------------------------------------------------------------


def test_render_results_tier3_with_verdicts():
    """_render_results renders Tier-3 verdict scores including the error case."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(model_name="gpt-4o")
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
class _Verdict:
    def __init__(self, data):
        self._data = data
    def model_dump(self):
        return self._data

verdicts = [
    _Verdict({"agent_id": "agent_0", "score": 3, "reasoning": "High compliance",
              "confidence": "high"}),
    _Verdict({"agent_id": "agent_1", "score": 1, "reasoning": "Partial",
              "confidence": "low"}),
    _Verdict({"agent_id": "agent_2", "score": 0, "reasoning": "",
              "confidence": "high"}),
    _Verdict({"agent_id": "agent_3", "score": -1, "error": "evaluator crashed"}),
]
with patch.object(ap, "_is_stub_mode", return_value=False):
    with patch.object(ap, "_load_baseline_result", return_value={"baseline": 1}):
        with patch.object(ap, "_get_evaluator_model", return_value="claude-3"):
            with patch.object(ap, "_run_tier3_evaluation", return_value=verdicts):
                ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "agent_0" in all_md
    assert "Score 3/3" in all_md
    all_err = " ".join(e.value for e in at.error)
    assert "evaluator crashed" in all_err


def test_render_results_tier3_evaluation_returns_none():
    """_render_results stops cleanly when Tier-3 evaluation returns None."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(model_name="gpt-4o")
result_dict = {
    "success": True,
    "agent_observations": [],
    "propagation_path": [],
    "influenced_agents": [],
    "resistant_agents": [],
}
with patch.object(ap, "_is_stub_mode", return_value=False):
    with patch.object(ap, "_load_baseline_result", return_value={"baseline": 1}):
        with patch.object(ap, "_get_evaluator_model", return_value="claude-3"):
            with patch.object(ap, "_run_tier3_evaluation", return_value=None):
                ap._render_results(config, result_dict)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _run_pre_execution_streaming (lines 859-929)
# ---------------------------------------------------------------------------


def test_run_pre_execution_streaming_builds_result():
    """_run_pre_execution_streaming streams tokens and returns an AttackResult."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(num_agents=2)

# Patch the strategy function to return the config unchanged.
with patch.object(
    ap._pre_exec_strats, "inject_prompt_injection", return_value=config
):
    fake_executor = MagicMock()
    msg = MagicMock()
    msg.content = "agent output text"
    events = [
        ("__token__", {"node": "agent_0", "token": "hello"}),
        ("__token__", {"node": "not_an_agent", "token": "ignored"}),
        ("__node_complete__", {"node": "agent_0",
                               "state_update": {"messages": [msg]}}),
        ("__node_complete__", {"node": "agent_1", "state_update": None}),
    ]
    fake_executor.run_streaming_tokens.return_value = iter(events)
    with patch.object(ap, "MASExecutor", return_value=fake_executor):
        result = ap._run_pre_execution_streaming(
            config, "agent_0", "prompt_injection", "payload text"
        )
st.markdown(f"mas_id:{result.mas_id}")
st.markdown(f"target:{result.target_agent_id}")
st.markdown(f"success:{result.success}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert f"mas_id:{'test_mas'}" in all_md
    assert "target:agent_0" in all_md
    assert "success:True" in all_md


def test_run_pre_execution_streaming_reraises_on_error():
    """_run_pre_execution_streaming re-raises when the stream loop fails."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
config = mk(num_agents=1)

def _boom(*args, **kwargs):
    raise RuntimeError("stream broke")

with patch.object(
    ap._pre_exec_strats, "inject_prompt_injection", return_value=config
):
    fake_executor = MagicMock()
    fake_executor.run_streaming_tokens.side_effect = _boom
    with patch.object(ap, "MASExecutor", return_value=fake_executor):
        raised = False
        try:
            ap._run_pre_execution_streaming(
                config, "agent_0", "prompt_injection", "payload"
            )
        except RuntimeError:
            raised = True
st.markdown(f"raised:{raised}")
"""
    )
    at.run()
    assert not at.exception
    assert "raised:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _run_attack success flow (lines 823-843)
# ---------------------------------------------------------------------------


def test_run_attack_pre_execution_success():
    """_run_attack stores the result and node states on a successful pre-exec run."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_config = mk()
st.session_state.attack_target_agent_id = "agent_0"
st.session_state.attack_suite = "injection"
st.session_state.attack_phase = "pre_execution"

fake_result = MagicMock()
fake_result.model_dump.return_value = {"success": True}
fake_result.agent_observations = []
with patch.object(ap.st, "rerun"):
    with patch.object(ap, "_resolve_payload", return_value="payload"):
        with patch.object(ap, "_run_pre_execution_streaming", return_value=fake_result):
            with patch.object(ap, "build_node_states", return_value={"agent_0": "ok"}):
                ap._run_attack()
st.markdown(f"stored:{'attack_result' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "stored:True" in " ".join(m.value for m in at.markdown)


def test_run_attack_mid_execution_path():
    """_run_attack uses the mid-execution handler when phase is mid_execution."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_config = mk()
st.session_state.attack_target_agent_id = "agent_0"
st.session_state.attack_suite = "injection"
st.session_state.attack_phase = "mid_execution"

fake_result = MagicMock()
fake_result.model_dump.return_value = {"success": True}
fake_result.agent_observations = []
with patch.object(ap.st, "rerun"):
    with patch.object(ap, "_resolve_payload", return_value="payload"):
        with patch.object(ap, "_run_mid_execution", return_value=fake_result) as mid:
            with patch.object(ap, "build_node_states", return_value={}):
                ap._run_attack()
st.markdown(f"mid_called:{mid.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "mid_called:True" in " ".join(m.value for m in at.markdown)


def test_run_attack_execution_error():
    """_run_attack shows an error when the execution handler raises."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
st.session_state.attack_config = mk()
st.session_state.attack_target_agent_id = "agent_0"
st.session_state.attack_suite = "injection"
st.session_state.attack_phase = "pre_execution"
with patch.object(ap, "_resolve_payload", return_value="payload"):
    with patch.object(ap, "_run_pre_execution_streaming",
                      side_effect=RuntimeError("boom")):
        ap._run_attack()
"""
    )
    at.run()
    assert not at.exception
    assert "Attack failed" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _execute_batch_attack (lines 378-539)
# ---------------------------------------------------------------------------


def test_execute_batch_attack_no_yaml_path():
    """_execute_batch_attack errors out when no YAML path is available."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
ap._execute_batch_attack(mk(), "", stub_mode=True)
"""
    )
    at.run()
    assert not at.exception
    assert "No YAML config path" in " ".join(e.value for e in at.error)


def test_execute_batch_attack_no_payloads_selected():
    """_execute_batch_attack warns when no payloads are selected."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
# Empty libraries mean no suite ends up with a selection.
with patch.object(ap, "_load_payload_library", return_value={}):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
    )
    at.run()
    assert not at.exception
    assert "No payloads selected" in " ".join(w.value for w in at.warning)


def test_execute_batch_attack_standard_suite_success():
    """_execute_batch_attack reports a standard suite that returns normally."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    # Provide a stub _suite_runner whose run_suite returns normally.
    runner_mod = types.ModuleType("bili.aegis.suites._suite_runner")
    runner_mod.run_suite = MagicMock(return_value=None)
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites._suite_runner": runner_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk

payload = MagicMock()

def _lib(suite):
    return {"pld_1": payload} if suite == "injection" else {}

with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    assert runner_mod.run_suite.called
    all_success = " ".join(s.value for s in at.success)
    assert "completed" in all_success


def test_execute_batch_attack_standard_suite_sysexit_pass():
    """_execute_batch_attack treats sys.exit(0) from run_suite as a pass."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    def _exit_zero(**kwargs):
        raise SystemExit(0)

    runner_mod = types.ModuleType("bili.aegis.suites._suite_runner")
    runner_mod.run_suite = _exit_zero
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites._suite_runner": runner_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
payload = MagicMock()
def _lib(suite):
    return {"pld_1": payload} if suite == "injection" else {}
with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    assert "completed" in " ".join(s.value for s in at.success)


def test_execute_batch_attack_standard_suite_sysexit_fail():
    """_execute_batch_attack treats sys.exit(non-zero) from run_suite as a failure."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    def _exit_one(**kwargs):
        raise SystemExit(1)

    runner_mod = types.ModuleType("bili.aegis.suites._suite_runner")
    runner_mod.run_suite = _exit_one
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites._suite_runner": runner_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
payload = MagicMock()
def _lib(suite):
    return {"pld_1": payload} if suite == "injection" else {}
with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    # One suite failed -> overall warning is shown.
    assert "failed" in " ".join(w.value for w in at.warning)


def test_execute_batch_attack_suite_exception():
    """_execute_batch_attack reports a suite that raises an unexpected error."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    def _raise(**kwargs):
        raise RuntimeError("runner exploded")

    runner_mod = types.ModuleType("bili.aegis.suites._suite_runner")
    runner_mod.run_suite = _raise
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites._suite_runner": runner_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
payload = MagicMock()
def _lib(suite):
    return {"pld_1": payload} if suite == "injection" else {}
with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "failed" in all_md


def test_execute_batch_attack_persistence_suite():
    """_execute_batch_attack runs the persistence suite via its own runner."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    persistence_mod = types.ModuleType(
        "bili.aegis.suites.persistence.run_persistence_suite"
    )
    persistence_mod.run_persistence_suite = MagicMock(
        return_value=([{"skipped": "false", "tier1_pass": "true"}], None)
    )
    eval_cfg_mod = types.ModuleType("bili.aegis.evaluator.evaluator_config")
    eval_cfg_mod.PERSISTENCE_JUDGE_PROMPT = "judge"
    eval_cfg_mod.PERSISTENCE_SCORE_DESCRIPTIONS = {}
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites.persistence.run_persistence_suite": persistence_mod,
            "bili.aegis.evaluator.evaluator_config": eval_cfg_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
payload = MagicMock()
def _lib(suite):
    return {"pld_1": payload} if suite == "persistence" else {}
with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=False)
"""
        )
        at.run()
    assert not at.exception
    assert persistence_mod.run_persistence_suite.called


def test_execute_batch_attack_cross_model_suite():
    """_execute_batch_attack runs the cross-model suite via its own runner."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    cm_mod = types.ModuleType("bili.aegis.suites.cross_model.run_cross_model_suite")
    cm_mod.MODEL_MATRIX = [("m1", "Model One")]
    cm_mod.run_cross_model_suite = MagicMock(
        return_value=([{"skipped": "false", "tier1_pass": "true"}], None)
    )
    se_mod = types.ModuleType("bili.aegis.evaluator.semantic_evaluator")
    se_mod.SemanticEvaluator = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "bili.aegis.suites.cross_model.run_cross_model_suite": cm_mod,
            "bili.aegis.evaluator.semantic_evaluator": se_mod,
        },
    ):
        at = AppTest.from_string(
            """
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
payload = MagicMock()
def _lib(suite):
    return {"pld_1": payload} if suite == "cross_model" else {}
with patch.object(ap, "_load_payload_library", side_effect=_lib):
    ap._execute_batch_attack(mk(), "/path/to/config.yaml", stub_mode=True)
"""
        )
        at.run()
    assert not at.exception
    assert cm_mod.run_cross_model_suite.called


# ---------------------------------------------------------------------------
# _render_main toggle captions (lines 724, 734)
# ---------------------------------------------------------------------------


def test_render_main_stub_and_t3_captions():
    """_render_main shows 'No LLM calls' and 'T3 skipped' captions when toggled."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="toggle_test")
st.session_state.attack_config = cfg
st.session_state.attack_target_agent_id = "agent_0"
st.session_state["attack_stub_mode"] = True
st.session_state["attack_skip_t3"] = True
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(ap, "render_attack_graph", return_value=None):
        with patch.object(ap, "_load_payload_library", return_value={}):
            ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "No LLM calls" in all_captions
    assert "T3 skipped" in all_captions


# ---------------------------------------------------------------------------
# _render_main node click triggers rerun (lines 787-788)
# ---------------------------------------------------------------------------


def test_render_main_node_click_updates_target():
    """_render_main updates the target agent when the graph returns a new node."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(num_agents=2, mas_id="click_test")
st.session_state.attack_config = cfg
st.session_state.attack_target_agent_id = "agent_0"
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    # The graph returns a different node than the current target, so the
    # click branch updates session state and reruns.
    with patch.object(ap, "render_attack_graph", return_value="agent_1"):
        with patch.object(ap, "_load_payload_library", return_value={}):
            with patch.object(ap.st, "rerun"):
                ap._render_main()
st.markdown(f"target:{st.session_state.get('attack_target_agent_id')}")
"""
    )
    at.run()
    assert not at.exception
    assert "target:agent_1" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Sidebar Run Attack button click (line 640)
# ---------------------------------------------------------------------------


def test_sidebar_run_attack_button_click():
    """Clicking the sidebar Run Attack button invokes _run_attack."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="run_btn_test")
st.session_state.attack_config = cfg
st.session_state.attack_target_agent_id = "agent_0"
mock_payload = MagicMock()
mock_payload.payload = "text"
mock_payload.notes = "notes"
with st.sidebar:
    with patch.object(ap, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(ap, "EXAMPLES_DIR") as ed:
            ed.exists.return_value = False
            with patch.object(ap, "_load_payload_library",
                              return_value={"p1": mock_payload}):
                with patch.object(ap, "_run_attack") as run_attack:
                    ap._render_sidebar()
                    st.session_state["__run_attack_mock_called"] = run_attack
"""
    )
    at.run()
    assert not at.exception
    # Locate and click the Run Attack button, then re-run to fire its callback.
    run_buttons = [b for b in at.sidebar.button if b.label == "Run Attack"]
    assert run_buttons
    run_buttons[0].click().run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Main batch run button click (line 761)
# ---------------------------------------------------------------------------


def test_main_batch_run_button_click():
    """Clicking the batch run button invokes _execute_batch_attack."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch, MagicMock
from bili.aether.ui import attack_page as ap
from bili.aether.ui.tests.conftest import make_test_config as mk
cfg = mk(mas_id="batch_btn_test")
st.session_state.attack_config = cfg
st.session_state.attack_yaml_path = "/path/cfg.yaml"
st.session_state.attack_target_agent_id = "agent_0"
payload = MagicMock()
payload.payload = "adversarial"
payload.severity = "high"
def _lib(suite):
    return {"p1": payload} if suite == "injection" else {}
with patch.object(ap, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(ap, "render_attack_graph", return_value=None):
        with patch.object(ap, "_load_payload_library", side_effect=_lib):
            with patch.object(ap, "_execute_batch_attack") as batch:
                ap._render_main()
"""
    )
    at.run()
    assert not at.exception
    run_buttons = [b for b in at.button if "Run" in b.label and "attack" in b.label]
    assert run_buttons
    run_buttons[0].click().run()
    assert not at.exception
