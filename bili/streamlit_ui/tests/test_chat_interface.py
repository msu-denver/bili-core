"""Tests for bili.streamlit_ui.ui.chat_interface -- IRIS Chat Interface.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access

import pytest
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


# ---------------------------------------------------------------------------
# display_state_management_management with token_length + trim
# ---------------------------------------------------------------------------


def test_state_management_token_length_trim():
    """State management uses token-based trim labels correctly."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["memory_limit_type"] = "token_length"
st.session_state["memory_strategy"] = "trim"
ci.display_state_management_management()
st.markdown(f"type:{st.session_state.get('memory_limit_type')}")
st.markdown(f"strategy:{st.session_state.get('memory_strategy')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "type:token_length" in all_md
    assert "strategy:trim" in all_md


def test_state_management_token_length_summarize():
    """State management uses token-based summarize labels correctly."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["memory_limit_type"] = "token_length"
st.session_state["memory_strategy"] = "summarize"
ci.display_state_management_management()
st.markdown(f"type:{st.session_state.get('memory_limit_type')}")
"""
    )
    at.run()
    assert not at.exception
    assert "type:token_length" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# display_state_management_management defaults for token_length
# ---------------------------------------------------------------------------


def test_state_management_token_length_default_values():
    """State management defaults to 10000 for token_length limit values."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["memory_limit_type"] = "token_length"
st.session_state.pop("memory_limit_value", None)
st.session_state.pop("memory_limit_trim_value", None)
ci.display_state_management_management()
st.markdown(f"value:{st.session_state.get('memory_limit_value')}")
st.markdown(f"trim:{st.session_state.get('memory_limit_trim_value')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "value:10000" in all_md
    assert "trim:10000" in all_md


# ---------------------------------------------------------------------------
# display_state_management_management with conversation chain present
# ---------------------------------------------------------------------------


def test_state_management_with_chain_no_warning():
    """State management does not warn when conversation chain is loaded."""
    at = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import MagicMock
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["conversation_chain"] = MagicMock()
ci.display_state_management_management()
"""
    )
    at.run()
    assert not at.exception
    # Should not have warning about missing conversation chain
    warnings = [w.value for w in at.warning]
    assert not any("No conversation chain" in w for w in warnings)


# ---------------------------------------------------------------------------
# display_state_management with intermediate steps
# ---------------------------------------------------------------------------


def test_display_state_management_with_intermediate_steps():
    """display_state_management renders intermediate processing messages."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {
    "messages": [
        HumanMessage(content="What tools do you have?"),
        ToolMessage(content="tool result", tool_call_id="tc1"),
        AIMessage(content="I have weather tools"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-steps"

form = st.form(key="test_steps_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# display_model_configuration with empty tools list
# ---------------------------------------------------------------------------


def test_model_config_empty_tools_list():
    """display_model_configuration handles empty tools list."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model-v3"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = []
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# display_state_management renders state JSON
# ---------------------------------------------------------------------------


def test_display_state_management_renders_state_json():
    """display_state_management renders a JSON expander for the current state."""
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
        HumanMessage(content="Show state"),
        AIMessage(content="Here it is"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state["thread_id"] = "test-json"

form = st.form(key="test_json_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
"""
    )
    at.run()
    assert not at.exception
    # Should have buttons for clear and export
    labels = [b.label for b in at.button]
    assert any("Clear" in l for l in labels)
    assert any("Export" in l for l in labels)


# ---------------------------------------------------------------------------
# run_app_page with reauthentication attempt
# ---------------------------------------------------------------------------


def test_run_app_page_reauthentication():
    """run_app_page attempts reauthentication before showing login."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch, call
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

mock_auth = MagicMock()
mock_auth.attempt_reauthentication.return_value = None
st.session_state.auth_manager = mock_auth

with patch.object(ci, "is_authenticated", return_value=False):
    with patch.object(ci, "display_login_signup") as mock_login:
        ci.run_app_page()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# display_model_configuration with no memory settings
# ---------------------------------------------------------------------------


def test_model_config_no_memory_settings():
    """display_model_configuration renders without memory settings."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state["model_config"] = "test-model-bare"
mock_chain = MagicMock()
mock_chain.checkpointer = "MemorySaver"
st.session_state["conversation_chain"] = mock_chain
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = []
# Ensure no memory keys are set
st.session_state.pop("memory_limit_type", None)
st.session_state.pop("memory_strategy", None)
st.session_state.pop("memory_limit_value", None)
st.session_state.pop("memory_limit_trim_value", None)
ci.display_model_configuration()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# load_system_components
# ---------------------------------------------------------------------------


def test_load_system_components_basic_flow():
    """load_system_components builds a chain with tools and stores it in state."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_id"] = "model.id"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = False
st.session_state["supports_tools"] = True
st.session_state["selected_tools"] = ["aws_opensearch_retriever"]
st.session_state["aws_opensearch_retriever_prompt"] = "Search the index"
st.session_state["memory_strategy"] = "summarize"
st.session_state["memory_limit_type"] = "message_count"
st.session_state["memory_limit_value"] = 15
st.session_state["memory_limit_trim_value"] = 15
st.session_state["persona"] = "You are helpful"
st.session_state["user_profile"] = {"name": "U"}
st.session_state["thinking_budget"] = 0

with patch.object(ci, "load_model", return_value="MODEL") as m_load:
    with patch.object(ci, "initialize_tools", return_value=["TOOL"]) as m_tools:
        with patch.object(ci, "build_agent_graph", return_value="AGENT") as m_graph:
            ci.load_system_components(None)
            st.session_state["_load_called"] = m_load.called
            st.session_state["_tools_called"] = m_tools.called
            st.session_state["_graph_called"] = m_graph.called
st.markdown(f"chain:{st.session_state.get('conversation_chain')}")
st.markdown(f"cfg:{st.session_state.get('model_config')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "chain:AGENT" in all_md
    assert "cfg:MODEL" in all_md
    assert at.session_state["_load_called"] is True
    assert at.session_state["_tools_called"] is True
    assert at.session_state["_graph_called"] is True


def test_load_system_components_structured_output_valid_schema():
    """A valid JSON response schema is parsed into model_kwargs."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_id"] = "gemini"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = True
st.session_state["response_mime_type"] = "application/json"
st.session_state["custom_response_schema"] = '{"type": "object"}'
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 0

captured = {}
def fake_load_model(**kwargs):
    captured.update(kwargs)
    return "MODEL"

with patch.object(ci, "load_model", side_effect=fake_load_model):
    with patch.object(ci, "initialize_tools", return_value=[]):
        with patch.object(ci, "build_agent_graph", return_value="AGENT"):
            ci.load_system_components(None)
st.markdown(f"schema_obj:{captured.get('response_schema') == {'type': 'object'}}")
st.markdown(f"mime:{captured.get('response_mime_type')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "schema_obj:True" in all_md
    assert "mime:application/json" in all_md


def test_load_system_components_structured_output_invalid_schema():
    """An invalid JSON response schema falls back to the default string schema."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_id"] = "gemini"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = True
st.session_state["response_mime_type"] = "application/json"
st.session_state["custom_response_schema"] = "{not valid"
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 0

captured = {}
def fake_load_model(**kwargs):
    captured.update(kwargs)
    return "MODEL"

with patch.object(ci, "load_model", side_effect=fake_load_model):
    with patch.object(ci, "initialize_tools", return_value=[]):
        with patch.object(ci, "build_agent_graph", return_value="AGENT"):
            ci.load_system_components(None)
st.markdown(f"fallback:{captured.get('response_schema') == {'type': 'string'}}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "fallback:True" in " ".join(m.value for m in at.markdown)


def test_load_system_components_structured_output_no_custom_schema():
    """Structured output with no custom schema defaults to a string schema."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_id"] = "gemini"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = True
st.session_state["response_mime_type"] = "application/json"
st.session_state.pop("custom_response_schema", None)
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 0

captured = {}
def fake_load_model(**kwargs):
    captured.update(kwargs)
    return "MODEL"

with patch.object(ci, "load_model", side_effect=fake_load_model):
    with patch.object(ci, "initialize_tools", return_value=[]):
        with patch.object(ci, "build_agent_graph", return_value="AGENT"):
            ci.load_system_components(None)
st.markdown(f"default:{captured.get('response_schema') == {'type': 'string'}}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "default:True" in " ".join(m.value for m in at.markdown)


def test_load_system_components_text_plain_mime():
    """A text/plain MIME type sets the plain-text response MIME on model_kwargs."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_id"] = "gemini"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = True
st.session_state["response_mime_type"] = "text/plain"
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 0

captured = {}
def fake_load_model(**kwargs):
    captured.update(kwargs)
    return "MODEL"

with patch.object(ci, "load_model", side_effect=fake_load_model):
    with patch.object(ci, "initialize_tools", return_value=[]):
        with patch.object(ci, "build_agent_graph", return_value="AGENT"):
            ci.load_system_components(None)
st.markdown(f"mime:{captured.get('response_mime_type')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "mime:text/plain" in " ".join(m.value for m in at.markdown)


def test_load_system_components_thinking_budget_branch():
    """A positive thinking budget inserts the prepare_llm_config node and config."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_id"] = "gemini"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = False
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 8192
st.session_state["memory_strategy"] = "summarize"
st.session_state["memory_limit_type"] = "message_count"
st.session_state["memory_limit_value"] = 15
st.session_state["memory_limit_trim_value"] = 15

captured = {}
def fake_build(**kwargs):
    captured.update(kwargs)
    return "AGENT"

with patch.object(ci, "load_model", return_value="MODEL"):
    with patch.object(ci, "initialize_tools", return_value=[]):
        with patch.object(ci, "build_agent_graph", side_effect=fake_build):
            ci.load_system_components(None)
node_kwargs = captured.get("node_kwargs", {})
st.markdown(f"thinking:{node_kwargs.get('thinking_config')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "thinking:{'budget': 8192}" in " ".join(m.value for m in at.markdown)


def test_load_system_components_no_tools_branch():
    """When tools are unsupported, the loader passes no active tools."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci

st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_id"] = "model.id"
st.session_state["model_kwargs"] = {}
st.session_state["supports_structured_output"] = False
st.session_state["supports_tools"] = False
st.session_state["thinking_budget"] = 0
st.session_state["memory_strategy"] = "trim"
st.session_state["memory_limit_type"] = "token_length"
st.session_state["memory_limit_value"] = 10000
st.session_state["memory_limit_trim_value"] = 8000

captured = {}
def fake_tools(**kwargs):
    captured.update(kwargs)
    return []

with patch.object(ci, "load_model", return_value="MODEL"):
    with patch.object(ci, "initialize_tools", side_effect=fake_tools):
        with patch.object(ci, "build_agent_graph", return_value="AGENT"):
            ci.load_system_components(None)
st.markdown(f"active:{captured.get('active_tools')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "active:None" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# run_app_page -- Load Configuration button and form submission
# ---------------------------------------------------------------------------


def test_load_configuration_button_click_loads_components():
    """Clicking Load Configuration invokes load_system_components."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.streamlit_ui.ui import chat_interface as ci
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            with patch.object(ci, "display_model_configuration"):
                with patch.object(ci, "load_system_components"):
                    # Patch rerun so the success message survives for the assertion.
                    with patch.object(ci.st, "rerun"):
                        ci.run_app_page("CHECKPOINTER")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    load_buttons = [b for b in at.button if b.label == "Load Configuration"]
    assert load_buttons
    load_buttons[0].click()
    at.run()
    assert not at.exception
    assert any("loaded successfully" in s.value for s in at.success)


def test_form_submit_non_streaming_processes_query():
    """Submitting the conversation form without streaming calls process_query."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state.setdefault("conversation_chain", MagicMock())
st.session_state["is_processing_query"] = False
st.session_state["streaming_enabled"] = False
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            with patch.object(ci, "display_model_configuration"):
                with patch.object(ci, "display_state_management"):
                    with patch.object(ci, "process_query") as mock_pq:
                        with patch.object(ci.st, "rerun"):
                            ci.run_app_page()
                            st.session_state["_pq"] = mock_pq
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    submit = [b for b in at.button if b.label == "Submit"]
    assert submit
    submit[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["_pq"].called


def test_form_submit_streaming_processes_query():
    """Submitting the form with streaming enabled calls process_query_streaming."""
    at = AppTest.from_string(
        """
from unittest.mock import patch, MagicMock
import streamlit as st
from bili.streamlit_ui.ui import chat_interface as ci
st.session_state.setdefault("conversation_chain", MagicMock())
st.session_state["is_processing_query"] = False
st.session_state["streaming_enabled"] = True
with patch.object(ci, "is_authenticated", return_value=True):
    with patch.object(ci, "display_configuration_panels"):
        with patch.object(ci, "display_state_management_management"):
            with patch.object(ci, "display_model_configuration"):
                with patch.object(ci, "display_state_management"):
                    with patch.object(
                        ci, "process_query_streaming",
                        return_value=iter(["a", "b"]),
                    ) as mock_stream:
                        with patch.object(ci.st, "write_stream"):
                            ci.run_app_page()
                            st.session_state["_stream"] = mock_stream
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    submit = [b for b in at.button if b.label == "Submit"]
    assert submit
    submit[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["_stream"].called


# ---------------------------------------------------------------------------
# display_state_management -- intermediate steps, clear/export/import buttons
# ---------------------------------------------------------------------------


def test_display_state_management_renders_intermediate_steps_area():
    """Intermediate steps render when the last human message is not first."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
# Leading AI message keeps the last human index > 0 so the processing
# message slice between the human and the final AI message is non-empty.
mock_state.values = {
    "messages": [
        AIMessage(content="Welcome"),
        HumanMessage(content="Use a tool"),
        ToolMessage(content="tool output", tool_call_id="tc1"),
        AIMessage(content="Done"),
    ]
}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain

form = st.form(key="inter_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    # A processing-message text area is rendered for the intermediate tool call.
    labels = [t.label for t in at.text_area]
    assert any("Processing Message" in (l or "") for l in labels)


def test_clear_conversation_state_button_updates_state():
    """Clicking Clear Conversation State updates the chain state and flags it."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {"messages": [HumanMessage(content="Hi"), AIMessage(content="Hello")]}
mock_chain.get_state.return_value = mock_state
st.session_state.setdefault("conversation_chain", mock_chain)

form = st.form(key="clear_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    with patch.object(ci, "clear_state", return_value={"messages": []}):
        with patch.object(ci.st, "rerun"):
            ci.display_state_management(form)
form.form_submit_button("submit")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    clear_buttons = [b for b in at.button if b.label == "Clear Conversation State"]
    assert clear_buttons
    clear_buttons[0].click()
    at.run()
    assert not at.exception
    chain = at.session_state["conversation_chain"]
    chain.update_state.assert_called()
    # The cleared confirmation renders on the click rerun (the flag is then
    # reset to False within the same run after the success is shown).
    assert any("cleared" in s.value for s in at.success)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: Export handler calls JsonPlusSerializer().dumps() "
        "(chat_interface.py line 391), but the installed langgraph "
        "JsonPlusSerializer exposes dumps_typed/loads_typed, not "
        "dumps/loads. Clicking Export raises AttributeError and crashes "
        "the panel."
    ),
)
def test_export_conversation_state_button_renders_download():
    """Clicking Export Conversation State should render a download button."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {"messages": [HumanMessage(content="Hi"), AIMessage(content="Yo")]}
mock_chain.get_state.return_value = mock_state
st.session_state.setdefault("conversation_chain", mock_chain)

form = st.form(key="export_form")
with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    ci.display_state_management(form)
form.form_submit_button("submit")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    export_buttons = [b for b in at.button if "Export Conversation State" in b.label]
    assert export_buttons
    export_buttons[0].click()
    at.run()
    assert not at.exception


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: Import handler calls JsonPlusSerializer().loads() "
        "(chat_interface.py line 405), but the installed langgraph "
        "JsonPlusSerializer exposes dumps_typed/loads_typed, not "
        "dumps/loads. Uploading a state file raises AttributeError and "
        "crashes the panel."
    ),
)
def test_import_conversation_state_applies_uploaded_state():
    """Uploading a conversation state should import messages and summary."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bili.streamlit_ui.ui import chat_interface as ci

mock_chain = MagicMock()
mock_state = MagicMock()
mock_state.values = {"messages": [HumanMessage(content="Hi"), AIMessage(content="Yo")]}
mock_chain.get_state.return_value = mock_state
st.session_state["conversation_chain"] = mock_chain
st.session_state.pop("state_imported", None)

fake_upload = MagicMock()
fake_upload.read.return_value = b"serialized"

with patch.object(ci, "get_state_config", return_value={"configurable": {"thread_id": "t"}}):
    with patch.object(ci, "clear_state", return_value={"messages": []}):
        with patch.object(ci.st, "file_uploader", return_value=fake_upload):
            with patch.object(ci.st, "rerun"):
                form = st.form(key="import_form")
                ci.display_state_management(form)
                form.form_submit_button("submit")
st.markdown(f"imported:{st.session_state.get('state_imported')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "imported:True" in " ".join(m.value for m in at.markdown)
