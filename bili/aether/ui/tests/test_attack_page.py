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

    def _app():
        from unittest.mock import patch

        from bili.aether.ui import attack_page as ap

        with patch.object(ap, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            ap._render_main()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "No MAS loaded" in " ".join(m.value for m in at.info)


def test_main_renders_aegis_heading():
    """The main area renders the AEGIS Attack Suite heading."""

    def _app():
        from unittest.mock import patch

        from bili.aether.ui import attack_page as ap

        with patch.object(ap, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            ap._render_main()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "AEGIS Attack Suite" in " ".join(m.value for m in at.markdown)


def test_sidebar_no_config_shows_caption():
    """The sidebar shows No MAS loaded when no config exists."""

    def _app():
        from unittest.mock import patch

        import streamlit as st

        from bili.aether.ui import attack_page as ap

        with st.sidebar:
            with patch.object(ap, "LOGO_PATH") as lp:
                lp.exists.return_value = False
                ap._render_sidebar()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "No MAS loaded" in " ".join(c.value for c in at.sidebar.caption)


def test_sidebar_renders_aegis_heading():
    """The sidebar renders the AEGIS heading."""

    def _app():
        from unittest.mock import patch

        import streamlit as st

        from bili.aether.ui import attack_page as ap

        with st.sidebar:
            with patch.object(ap, "LOGO_PATH") as lp:
                lp.exists.return_value = False
                ap._render_sidebar()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)


def test_push_config_sets_session_state():
    """push_config_to_attack_state stores config in session state."""

    def _app():
        import streamlit as st

        from bili.aether.ui.attack_page import push_config_to_attack_state
        from bili.aether.ui.tests.conftest import make_test_config as mk

        cfg = mk(mas_id="push_test")
        push_config_to_attack_state(cfg)
        st.markdown(f"config_set:{st.session_state.get('attack_config') is not None}")
        st.markdown(f"target:{st.session_state.get('attack_target_agent_id')}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "config_set:True" in all_md
    assert "target:agent_0" in all_md


def test_push_config_clears_previous_results():
    """push_config_to_attack_state clears prior attack results."""

    def _app():
        import streamlit as st

        from bili.aether.ui.attack_page import push_config_to_attack_state
        from bili.aether.ui.tests.conftest import make_test_config as mk

        st.session_state.attack_result = {"some": "data"}
        st.session_state.attack_verdict = [{"score": 1}]
        cfg = mk()
        push_config_to_attack_state(cfg)
        st.markdown(f"cleared:{'attack_result' not in st.session_state}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_render_attack_page_no_config_no_exception():
    """The full page renders without exception when no config loaded."""

    def _app():
        from unittest.mock import patch

        from bili.aether.ui import attack_page as ap

        with patch.object(ap, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            ap.render_attack_page()

    at = AppTest.from_function(_app)
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

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception


def test_render_observation_clean():
    """_render_observation renders a clean agent."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
