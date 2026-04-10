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


def test_sidebar_no_config_shows_caption():
    """The sidebar shows No MAS loaded when no config exists."""
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
    assert "No MAS loaded" in " ".join(c.value for c in at.sidebar.caption)


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
    """_render_main with a loaded config shows the attack suite heading."""
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
    assert "main_test" in " ".join(c.value for c in at.caption)
