"""Tests for bili.streamlit_ui.ui.chat_interface -- IRIS Chat Interface.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access

from streamlit.testing.v1 import AppTest


def test_unauthenticated_shows_login():
    """When not authenticated the page shows the login/signup form."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=False):
    mock_auth = MagicMock()
    mock_auth.attempt_reauthentication.return_value = None
    with patch.object(ci, "display_login_signup"):
        st.session_state.auth_manager = mock_auth
        ci.run_app_page()
""",
        default_timeout=10,
    )
    at.run()
    assert not at.exception


def test_authenticated_shows_configuration_header():
    """When authenticated the page renders a Configuration header."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception
    assert any("Configuration" in h.value for h in at.header)


def test_no_chain_shows_warning():
    """Without a conversation chain the page shows a warning."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception
    assert "load the configuration" in " ".join(w.value for w in at.warning)


def test_conversation_header_present():
    """The Conversation header appears when authenticated."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception
    assert any("Conversation" in h.value for h in at.header)


def test_load_config_button_present():
    """The Load Configuration button renders on the page."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception
    assert any("Load Configuration" in b.label for b in at.button)


def test_model_config_not_loaded_shows_warning():
    """When model_config is absent a warning is shown."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception
    assert "not loaded" in " ".join(w.value for w in at.warning)


def test_model_config_loaded_no_exception():
    """When model_config exists the function renders without error."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model-v1"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_state_management_no_chain_shows_warning():
    """Without a conversation chain the state management shows a warning."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_state_management_management()
"""
    )
    at.run()
    assert not at.exception
    assert "No conversation chain" in " ".join(w.value for w in at.warning)


def test_active_configuration_header():
    """The Active Configuration header appears on the page."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception
    assert any("Active Configuration" in h.value for h in at.header)


def test_state_management_defaults_memory_limit_type():
    """State management defaults memory_limit_type to message_count."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_state_management_management()
st.markdown(f"type:{st.session_state.get('memory_limit_type')}")
"""
    )
    at.run()
    assert not at.exception
    assert "type:message_count" in " ".join(m.value for m in at.markdown)


def test_state_management_defaults_memory_strategy():
    """State management defaults memory_strategy to summarize."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_state_management_management()
st.markdown(f"strategy:{st.session_state.get('memory_strategy')}")
"""
    )
    at.run()
    assert not at.exception
    assert "strategy:summarize" in " ".join(m.value for m in at.markdown)
