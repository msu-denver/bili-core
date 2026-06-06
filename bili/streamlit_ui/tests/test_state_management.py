"""Tests for streamlit_ui.utils.state_management form-state helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bili.streamlit_ui.utils import state_management


class TestFormProcessingState:
    """disable_form and enable_form toggle the processing flag."""

    def test_disable_form_sets_processing_true(self):
        """disable_form marks the form as processing a query."""
        fake_st = MagicMock()
        fake_st.session_state = SimpleNamespace()
        with patch.object(state_management, "st", fake_st):
            state_management.disable_form()
        assert fake_st.session_state.is_processing_query is True

    def test_enable_form_sets_processing_false(self):
        """enable_form clears the processing flag."""
        fake_st = MagicMock()
        fake_st.session_state = SimpleNamespace()
        with patch.object(state_management, "st", fake_st):
            state_management.enable_form()
        assert fake_st.session_state.is_processing_query is False


class TestGetStateConfig:
    """get_state_config derives the thread_id from the user's email."""

    def test_config_uses_email_as_thread_id(self):
        """The configurable thread_id is the session user's email."""
        fake_st = MagicMock()
        fake_st.session_state = {"user_info": {"email": "u@example.com"}}
        with patch.object(state_management, "st", fake_st):
            config = state_management.get_state_config()
        assert config == {"configurable": {"thread_id": "u@example.com"}}

    def test_config_handles_missing_user_info(self):
        """A missing user_info yields a None-derived thread_id, not a crash."""
        fake_st = MagicMock()
        fake_st.session_state = {}
        with patch.object(state_management, "st", fake_st):
            config = state_management.get_state_config()
        assert config == {"configurable": {"thread_id": "None"}}
