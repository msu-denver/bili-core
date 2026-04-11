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


# ---------------------------------------------------------------------------
# Configuration loading and application
# ---------------------------------------------------------------------------


def test_state_management_defaults_memory_limit_value():
    """State management defaults memory_limit_value to 15."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_state_management_management()
st.markdown(f"value:{st.session_state.get('memory_limit_value')}")
"""
    )
    at.run()
    assert not at.exception
    assert "value:15" in " ".join(m.value for m in at.markdown)


def test_state_management_defaults_trim_value():
    """State management defaults memory_limit_trim_value to 15."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
ci.display_state_management_management()
st.markdown(f"trim:{st.session_state.get('memory_limit_trim_value')}")
"""
    )
    at.run()
    assert not at.exception
    assert "trim:15" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Conversation display with messages
# ---------------------------------------------------------------------------


def test_display_state_management_with_chain_and_state():
    """display_state_management renders state when conversation chain exists."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

# Create mock chain with state
mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {
    "messages": [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-thread"

form = st.form(key="test_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception


def test_display_state_management_no_state():
    """display_state_management shows warning when state is None."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_chain.get_state.return_value = None
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-thread"

form = st.form(key="test_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception
    assert "No saved state" in " ".join(w.value for w in at.warning)


# ---------------------------------------------------------------------------
# Model switching / configuration display
# ---------------------------------------------------------------------------


def test_model_config_shows_checkpointer():
    """display_model_configuration shows checkpointer type."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model-config"
mock_chain = MagicMock()
mock_chain.checkpointer = "PostgresSaver"
st.session_state["conversation_chain"] = mock_chain
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_model_config_shows_memory_settings():
    """display_model_configuration shows memory settings when present."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["memory_limit_type"] = "token_length"
st.session_state["memory_strategy"] = "trim"
st.session_state["memory_limit_value"] = 5000
st.session_state["memory_limit_trim_value"] = 3000
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_model_config_shows_tool_configuration():
    """display_model_configuration shows tool config when tools exist."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = ["weather_api_tool"]
st.session_state["weather_api_tool_prompt"] = "Get weather data"
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_model_config_no_tools_when_unsupported():
    """display_model_configuration skips tools when supports_tools is False."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = False
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# run_app_page with conversation chain loaded
# ---------------------------------------------------------------------------


def test_run_app_page_with_chain_shows_form():
    """When a conversation chain exists the page shows a conversation form."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
mock_chain = MagicMock()
st.session_state["conversation_chain"] = mock_chain
st.session_state["is_processing_query"] = False
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            with patch.object(ci, "display_state_management"):
                ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Memory limit type switching
# ---------------------------------------------------------------------------


def test_state_management_token_length_labels():
    """State management uses token-based labels when memory_limit_type is token_length."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["memory_limit_type"] = "token_length"
ci.display_state_management_management()
st.markdown(f"type:{st.session_state.get('memory_limit_type')}")
"""
    )
    at.run()
    assert not at.exception
    assert "type:token_length" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Configuration panel interactions
# ---------------------------------------------------------------------------


def test_display_model_config_with_tools_list():
    """display_model_configuration shows multiple tools."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = [
    "weather_api_tool", "serp_api_tool"
]
st.session_state["weather_api_tool_prompt"] = "Get weather"
st.session_state["serp_api_tool_prompt"] = "Search web"
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_display_model_config_with_middleware():
    """display_model_configuration shows middleware settings."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = False
st.session_state["memory_limit_type"] = "message_count"
st.session_state["memory_strategy"] = "trim"
st.session_state["memory_limit_value"] = 10
st.session_state["memory_limit_trim_value"] = 8
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


def test_display_state_management_with_messages():
    """display_state_management renders messages inside a form."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {
    "messages": [
        HumanMessage(content="What is AI?"),
        AIMessage(content="AI stands for Artificial Intelligence"),
        HumanMessage(content="Tell me more"),
        AIMessage(content="It encompasses many fields"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test"

form = st.form(key="test_form2")
with patch.object(
    ci, "get_state_config",
    return_value={"configurable": {"thread_id": "t"}}
):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception


def test_run_app_page_processing_query_state():
    """run_app_page handles is_processing_query = True."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
mock_chain = MagicMock()
st.session_state["conversation_chain"] = mock_chain
st.session_state["is_processing_query"] = True
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            with patch.object(ci, "display_state_management"):
                ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception


def test_display_state_mgmt_no_thread_id():
    """display_state_management_management with no thread_id set."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state.pop("thread_id", None)
ci.display_state_management_management()
st.markdown(f"ran:True")
"""
    )
    at.run()
    assert not at.exception
    assert "ran:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# display_state_management with processing messages
# ---------------------------------------------------------------------------


def test_display_state_management_with_processing_messages():
    """display_state_management renders intermediate steps between messages."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
# Two HumanMessages with AI responses to trigger processing messages
mock_state.values = {
    "messages": [
        HumanMessage(content="What is the weather?"),
        AIMessage(content="The weather is 72F"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-processing"

form = st.form(key="test_processing_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
""",
        default_timeout=10,
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# display_model_configuration with memory settings
# ---------------------------------------------------------------------------


def test_display_model_config_with_all_memory_settings():
    """display_model_configuration shows all memory settings."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model-v2"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["memory_limit_type"] = "message_count"
st.session_state["memory_strategy"] = "summarize"
st.session_state["memory_limit_value"] = 20
st.session_state["memory_limit_trim_value"] = 15
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = ["weather_api_tool", "serp_api_tool"]
st.session_state["weather_api_tool_prompt"] = "Get weather"
st.session_state["serp_api_tool_prompt"] = "Search web"
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# display_state_management state cleared flag
# ---------------------------------------------------------------------------


def test_display_state_management_state_cleared():
    """display_state_management shows success when state_cleared is True."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {"messages": [HumanMessage(content="Hi")]}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test"
st.session_state["state_cleared"] = True

form = st.form(key="test_cleared_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared" in " ".join(s.value for s in at.success)


# ---------------------------------------------------------------------------
# display_state_management_management trim labels
# ---------------------------------------------------------------------------


def test_state_management_trim_labels():
    """State management uses trim-specific labels when strategy is trim."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["memory_limit_type"] = "message_count"
st.session_state["memory_strategy"] = "trim"
ci.display_state_management_management()
st.markdown(f"strategy:{st.session_state.get('memory_strategy')}")
"""
    )
    at.run()
    assert not at.exception
    assert "strategy:trim" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# display_state_management with empty messages
# ---------------------------------------------------------------------------


def test_display_state_management_empty_messages():
    """display_state_management handles state with no messages."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {"messages": []}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-empty"

form = st.form(key="test_empty_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception
