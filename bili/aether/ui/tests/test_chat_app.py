"""Tests for bili.aether.ui.chat_app -- AETHER Chat Interface.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported
# pylint: disable=duplicate-code

from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from bili.aether.ui.tests.conftest import make_test_config

_FM = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(),
    "streamlit_flow.state": MagicMock(),
}


def test_chat_area_no_config_shows_info():
    """Without a config the chat area shows an info prompt."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    assert "Select a YAML" in " ".join(m.value for m in at.info)


def test_chat_area_load_error_shows_error():
    """When a load error exists the chat area shows an error."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
st.session_state.chat_load_error = "Bad config"
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    assert "Bad config" in " ".join(m.value for m in at.error)


def test_render_main_no_config_no_exception():
    """render_main() runs without exception when no config loaded."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca.render_main()
"""
    )
    at.run()
    assert not at.exception


def test_sidebar_missing_examples_dir_shows_error():
    """When examples_dir does not exist the sidebar shows an error."""
    at = AppTest.from_string(
        """
from pathlib import Path
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca.render_sidebar_content(examples_dir=Path("/nonexistent/dir"))
"""
    )
    at.run()
    assert not at.exception
    assert "not found" in " ".join(m.value for m in at.error)


def test_sidebar_empty_dir_shows_warning():
    """When examples dir is empty and no uploads, a warning is shown."""
    at = AppTest.from_string(
        """
from pathlib import Path
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
mock_dir = _Mock(spec=Path)
mock_dir.exists.return_value = True
mock_dir.glob.return_value = []
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca.render_sidebar_content(examples_dir=mock_dir)
"""
    )
    at.run()
    assert not at.exception
    assert "No YAML" in " ".join(m.value for m in at.warning)


def test_new_thread_creates_entry():
    """_new_thread creates a thread in session state."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    tid = ca._new_thread("test_mas")
    threads = st.session_state.get("chat_threads", {})
    st.markdown(f"created:{tid in threads}")
    st.markdown(f"active:{st.session_state.get('chat_thread_id') == tid}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "created:True" in all_md
    assert "active:True" in all_md


def test_delete_thread_removes_entry():
    """_delete_thread removes a thread from session state."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    tid = ca._new_thread("test_mas")
    ca._delete_thread(tid)
    st.markdown(f"gone:{tid not in st.session_state.get('chat_threads', {})}")
"""
    )
    at.run()
    assert not at.exception
    assert "gone:True" in " ".join(m.value for m in at.markdown)


def test_active_messages_or_empty_returns_list():
    """_active_messages_or_empty returns [] when no thread is active."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    st.markdown(f"empty:{len(ca._active_messages_or_empty()) == 0}")
"""
    )
    at.run()
    assert not at.exception
    assert "empty:True" in " ".join(m.value for m in at.markdown)


def test_ensure_active_thread_creates_when_missing():
    """_ensure_active_thread creates a thread if none exists."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca._ensure_active_thread("test_mas")
    st.markdown(f"has:{st.session_state.get('chat_thread_id') is not None}")
"""
    )
    at.run()
    assert not at.exception
    assert "has:True" in " ".join(m.value for m in at.markdown)


# --- Pure unit tests (no AppTest needed) ---


def test_serialize_state_update_converts_messages():
    """_serialize_state_update converts BaseMessage to dicts."""
    from langchain_core.messages import HumanMessage

    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._serialize_state_update(
            {"messages": [HumanMessage(content="hello")]}
        )
    assert result["messages"][0]["content"] == "hello"


def test_serialize_state_update_preserves_strings():
    """_serialize_state_update passes non-message values through."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._serialize_state_update({"key": "value", "count": 42})
    assert result["key"] == "value"
    assert result["count"] == 42


def test_is_stub_config_true_when_no_model():
    """_is_stub_config returns True when agents have no model_name."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._is_stub_config(make_test_config()) is True


def test_is_stub_config_false_when_model_set():
    """_is_stub_config returns False when all agents have model_name."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._is_stub_config(make_test_config(model_name="gpt-4o")) is False


def test_extract_content_from_agent_outputs():
    """_extract_content pulls text from agent_outputs dict."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"agent_outputs": {"a0": {"message": "Hello world"}}}
        assert "Hello world" in ca._extract_content(output)


def test_extract_content_from_messages():
    """_extract_content falls back to messages list."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert "From msgs" in ca._extract_content(
            {"messages": [{"content": "From msgs"}]}
        )


def test_extract_content_stub_status():
    """_extract_content handles stub status outputs."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"agent_outputs": {"a0": {"status": "stub", "message": "Stub resp"}}}
        assert "Stub resp" in ca._extract_content(output)


def test_extract_content_empty_output():
    """_extract_content returns str for empty output."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert isinstance(ca._extract_content({}), str)


def test_build_markdown_export_structure():
    """_build_markdown_export produces valid markdown with turns."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        config = make_test_config(mas_id="export_test")
        history = [
            {
                "content": "Hello MAS",
                "agent_trace": [
                    {
                        "agent_id": "a0",
                        "output": {"agent_outputs": {"a0": {"message": "Hi"}}},
                    }
                ],
            }
        ]
        md = ca._build_markdown_export(config, "t-123", history)
    assert "export_test" in md
    assert "Hello MAS" in md
    assert "Turn 1" in md


def test_build_markdown_export_empty_history():
    """_build_markdown_export handles empty chat history."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        md = ca._build_markdown_export(make_test_config(), "t", [])
    assert "Turn" not in md


def test_warn_explanation_known_pattern():
    """_warn_explanation returns text for a known warning pattern."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._warn_explanation("agent X has no channel connections")
        assert result is not None
        assert "message routes" in result


def test_warn_explanation_unknown_pattern():
    """_warn_explanation returns None for an unknown pattern."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._warn_explanation("totally unknown warning") is None


def test_warn_explanation_supervisor_pattern():
    """_warn_explanation matches the supervisor capability warning."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._warn_explanation(
            "should have 'inter_agent_communication' capability"
        )
        assert result is not None


def test_local_provider_always_available():
    """Local providers are always available."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._is_provider_available("local_llamacpp") is True
        assert ca._is_provider_available("local_huggingface") is True


def test_unknown_provider_defaults_available():
    """Unknown provider keys default to available (fail-open)."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._is_provider_available("future_provider") is True


def test_openai_provider_needs_api_key():
    """remote_openai requires OPENAI_API_KEY env var."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {}, clear=True):
            assert ca._is_provider_available("remote_openai") is False


def test_openai_provider_available_with_key():
    """remote_openai is available when OPENAI_API_KEY is set."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            assert ca._is_provider_available("remote_openai") is True


# ---------------------------------------------------------------------------
# render_sidebar_content with configs loaded
# ---------------------------------------------------------------------------


def test_sidebar_with_yaml_files_shows_selectbox():
    """When YAML files exist the sidebar renders a config selectbox."""
    at = AppTest.from_string(
        """
from pathlib import Path
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
mock_dir = _Mock(spec=Path)
mock_dir.exists.return_value = True
mock_yaml = _Mock(spec=Path)
mock_yaml.stem = "my_config"
mock_dir.glob.return_value = [mock_yaml]
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca.render_sidebar_content(examples_dir=mock_dir)
"""
    )
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 1


def test_sidebar_with_uploaded_config_shows_upload_label():
    """When an uploaded config exists in session state the selectbox contains it."""
    at = AppTest.from_string(
        """
from pathlib import Path
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
mock_dir = _Mock(spec=Path)
mock_dir.exists.return_value = True
mock_dir.glob.return_value = []
st.session_state["chat_uploaded_configs"] = {"upload.yaml": mk()}
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ca.render_sidebar_content(examples_dir=mock_dir)
"""
    )
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 1


# ---------------------------------------------------------------------------
# _render_chat_area with active conversation
# ---------------------------------------------------------------------------


def test_chat_area_with_config_renders_name():
    """When a config is loaded the chat area renders its name."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(mas_id="chat_test", name="Chat Test MAS")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    # Patch _render_mas_structure to avoid flow graph dependency
    with _patch.object(ca, "_render_mas_structure"):
        ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Chat Test MAS" in all_md


def test_chat_area_with_stored_turns_renders_history():
    """When chat history has turns the chat area renders them."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(mas_id="history_test")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    tid = ca._new_thread("history_test")
    st.session_state["chat_threads"][tid]["messages"] = [
        {
            "role": "user",
            "content": "Hello MAS",
            "turn_index": 0,
            "agent_trace": [
                {
                    "agent_id": "agent_0",
                    "output": {"agent_outputs": {"agent_0": {"message": "Hi back"}}},
                }
            ],
        }
    ]
    with _patch.object(ca, "_render_mas_structure"):
        ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Hello MAS" in all_md


# ---------------------------------------------------------------------------
# _render_stored_turn and _render_agent_panel
# ---------------------------------------------------------------------------


def test_render_stored_turn_with_agent_trace():
    """_render_stored_turn renders user content and agent panels."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(mas_id="stored_turn_test")
    st.session_state.chat_config = cfg
    turn = {
        "content": "User question here",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "Agent response text", "role": "role_0", "status": "done"}
                    }
                },
            }
        ],
    }
    ca._render_stored_turn(turn)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "User question here" in all_md


def test_render_stored_turn_with_error():
    """_render_stored_turn renders an error when the turn has an error."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk()
    st.session_state.chat_config = cfg
    turn = {
        "content": "Fail question",
        "turn_index": 0,
        "error": "Something went wrong",
        "agent_trace": [],
    }
    ca._render_stored_turn(turn)
"""
    )
    at.run()
    assert not at.exception
    all_err = " ".join(m.value for m in at.error)
    assert "Something went wrong" in all_err


def test_render_agent_panel_stub_output():
    """_render_agent_panel renders stub status as a caption."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {"agent_outputs": {"a0": {"status": "stub", "message": "Stub response text"}}}
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "Stub response text" in all_captions


def test_render_agent_panel_message_output():
    """_render_agent_panel renders a message from agent_outputs."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {
        "agent_outputs": {
            "a0": {"message": "Model response here", "role": "analyst", "status": "complete"}
        }
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Model response here" in all_md


def test_render_agent_panel_fallback_to_messages():
    """_render_agent_panel falls back to messages list when agent_outputs is absent."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {"messages": [{"content": "Fallback message content"}]}
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Fallback message content" in all_md


def test_render_agent_panel_with_role_label():
    """_render_agent_panel displays role in label when role differs from agent_id."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {"agent_outputs": {"a0": {"message": "Test", "role": "writer"}}}
    ca._render_agent_panel("a0", output, expanded=True, role="writer")
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------


def test_delete_thread_switches_to_newest():
    """Deleting the active thread switches to the most recent remaining thread."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
import time
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    tid1 = ca._new_thread("mas1")
    time.sleep(0.01)
    tid2 = ca._new_thread("mas2")
    ca._delete_thread(tid2)
    st.markdown(f"active:{st.session_state.get('chat_thread_id') == tid1}")
"""
    )
    at.run()
    assert not at.exception
    assert "active:True" in " ".join(m.value for m in at.markdown)


def test_delete_all_threads_clears_active():
    """Deleting all threads clears the active thread pointer."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    tid = ca._new_thread("mas1")
    ca._delete_thread(tid)
    st.markdown(f"cleared:{st.session_state.get('chat_thread_id') is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_active_messages_returns_thread_messages():
    """_active_messages returns messages for the active thread."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    tid = ca._new_thread("test_mas")
    st.session_state["chat_threads"][tid]["messages"].append({"content": "msg1"})
    msgs = ca._active_messages()
    st.markdown(f"count:{len(msgs)}")
    st.markdown(f"content:{msgs[0]['content']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:1" in all_md
    assert "content:msg1" in all_md


def test_active_messages_raises_when_no_thread():
    """_active_messages raises RuntimeError when no thread exists."""
    with patch.dict("sys.modules", _FM):
        import pytest

        from bili.aether.ui import chat_app as ca

        with pytest.raises(RuntimeError):
            ca._active_messages()


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def test_build_markdown_export_with_multiple_turns():
    """_build_markdown_export includes all turns in order."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        config = make_test_config(mas_id="multi_turn_test")
        history = [
            {
                "content": "Turn one",
                "agent_trace": [
                    {
                        "agent_id": "a0",
                        "output": {"agent_outputs": {"a0": {"message": "Reply 1"}}},
                    }
                ],
            },
            {
                "content": "Turn two",
                "agent_trace": [
                    {
                        "agent_id": "a0",
                        "output": {"agent_outputs": {"a0": {"message": "Reply 2"}}},
                    }
                ],
            },
        ]
        md = ca._build_markdown_export(config, "t-456", history)
    assert "Turn 1" in md
    assert "Turn 2" in md
    assert "Turn one" in md
    assert "Turn two" in md
    assert "Reply 1" in md
    assert "Reply 2" in md


def test_build_markdown_export_contains_thread_id():
    """_build_markdown_export includes the thread ID."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        md = ca._build_markdown_export(make_test_config(), "thread-xyz", [])
    assert "thread-xyz" in md


# ---------------------------------------------------------------------------
# _build_role_map
# ---------------------------------------------------------------------------


def test_build_role_map_omits_same():
    """_build_role_map omits entries where role == agent_id."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    from bili.aether.schema.mas_config import MASConfig
    from bili.aether.schema.enums import WorkflowType

    agents = [
        AgentSpec(agent_id="a0", role="a0", objective="Perform analysis tasks"),
        AgentSpec(agent_id="a1", role="writer", objective="Perform writing tasks"),
    ]
    config = MASConfig(
        mas_id="test", name="t", description="t", agents=agents,
        channels=[], workflow_type=WorkflowType.SEQUENTIAL,
    )
    role_map = ca._build_role_map(config)
    st.markdown(f"a0_absent:{'a0' not in role_map}")
    st.markdown(f"a1_role:{role_map.get('a1')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "a0_absent:True" in all_md
    assert "a1_role:writer" in all_md


# ---------------------------------------------------------------------------
# _validate_config display
# ---------------------------------------------------------------------------


def test_validate_config_valid_shows_success():
    """_validate_config shows success for a valid config."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    import streamlit as st
    cfg = mk()
    result = ca._validate_config(cfg)
    st.markdown(f"valid:{result}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "valid:True" in all_md


# ---------------------------------------------------------------------------
# _agent_card rendering
# ---------------------------------------------------------------------------


def test_agent_card_with_different_role():
    """_agent_card renders role when it differs from agent_id."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    agent = AgentSpec(agent_id="a0", role="analyst", objective="Perform analysis tasks")
    ca._agent_card(agent)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "a0" in all_md
    assert "analyst" in all_md


def test_agent_card_same_role():
    """_agent_card renders only agent_id when role matches."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    agent = AgentSpec(agent_id="a0", role="a0", objective="Perform analysis tasks")
    ca._agent_card(agent)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "a0" in all_md


# ---------------------------------------------------------------------------
# MAS diagram rendering
# ---------------------------------------------------------------------------


def test_render_sequential_diagram():
    """_render_sequential_diagram renders arrows between agents."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(num_agents=3)
    ca._render_sequential_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_consensus_diagram():
    """_render_consensus_diagram renders consensus label."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    from bili.aether.schema.mas_config import MASConfig
    from bili.aether.schema.enums import WorkflowType
    agents = [
        AgentSpec(agent_id="a0", role="voter_0", objective="Vote on proposals submitted"),
        AgentSpec(agent_id="a1", role="voter_1", objective="Vote on proposals submitted"),
    ]
    cfg = MASConfig(
        mas_id="cons_test", name="t", description="t",
        agents=agents, channels=[], workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
    )
    ca._render_consensus_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "consensus" in all_md


def test_render_fallback_diagram():
    """_render_fallback_diagram renders agent list."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk()
    ca._render_fallback_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "agent_0" in all_md


# ---------------------------------------------------------------------------
# _serialize_state_update nested structures
# ---------------------------------------------------------------------------


def test_serialize_state_update_nested_dict():
    """_serialize_state_update handles nested dicts with BaseMessage."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from langchain_core.messages import AIMessage
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    result = ca._serialize_state_update(
        {"nested": {"inner": AIMessage(content="nested content")}}
    )
    st.markdown(f"content:{result['nested']['inner']['content']}")
"""
    )
    at.run()
    assert not at.exception
    assert "content:nested content" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _extract_content edge cases
# ---------------------------------------------------------------------------


def test_extract_content_from_multiple_agent_outputs():
    """_extract_content joins text from multiple agent_outputs."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {
            "agent_outputs": {
                "a0": {"message": "Output A"},
                "a1": {"message": "Output B"},
            }
        }
        content = ca._extract_content(output)
    assert "Output A" in content
    assert "Output B" in content


def test_extract_content_kv_fallback():
    """_extract_content renders key-value pairs when no message field."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"agent_outputs": {"a0": {"score": 42, "agent_id": "a0"}}}
        content = ca._extract_content(output)
    assert "score" in content


# ---------------------------------------------------------------------------
# Provider availability checks
# ---------------------------------------------------------------------------


def test_azure_provider_needs_api_key():
    """remote_azure_openai requires AZURE_OPENAI_API_KEY."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {}, clear=True):
            assert ca._is_provider_available("remote_azure_openai") is False


def test_azure_provider_available_with_key():
    """remote_azure_openai is available when AZURE_OPENAI_API_KEY is set."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "key"}):
            assert ca._is_provider_available("remote_azure_openai") is True


def test_google_provider_needs_credentials():
    """remote_google_vertex requires GOOGLE env vars."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {}, clear=True):
            assert ca._is_provider_available("remote_google_vertex") is False


def test_google_provider_available_with_project():
    """remote_google_vertex is available when GOOGLE_CLOUD_PROJECT is set."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "proj"}):
            assert ca._is_provider_available("remote_google_vertex") is True


# ---------------------------------------------------------------------------
# _render_stored_turn — additional structures
# ---------------------------------------------------------------------------


def test_render_stored_turn_no_agent_trace():
    """_render_stored_turn handles turns with empty agent_trace."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk()
    st.session_state.chat_config = cfg
    turn = {
        "content": "No agents ran",
        "turn_index": 0,
        "agent_trace": [],
    }
    ca._render_stored_turn(turn)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "No agents ran" in all_md


def test_render_stored_turn_multiple_agents():
    """_render_stored_turn renders panels for multiple agents."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(num_agents=3)
    st.session_state.chat_config = cfg
    turn = {
        "content": "Multi-agent question",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "First agent"}
                    }
                },
            },
            {
                "agent_id": "agent_1",
                "output": {
                    "agent_outputs": {
                        "agent_1": {"message": "Second agent"}
                    }
                },
            },
        ],
    }
    ca._render_stored_turn(turn)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Multi-agent question" in all_md


# ---------------------------------------------------------------------------
# _render_agent_panel — additional output formats
# ---------------------------------------------------------------------------


def test_render_agent_panel_with_parsed_json():
    """_render_agent_panel renders parsed JSON output."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {
        "agent_outputs": {
            "a0": {
                "message": "Parsed output",
                "parsed": {"key": "value"},
            }
        }
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Parsed output" in all_md


def test_render_agent_panel_extra_fields():
    """_render_agent_panel renders extra fields not in skip set."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {
        "agent_outputs": {
            "a0": {
                "message": "Main text",
                "confidence": 0.95,
            }
        }
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "confidence" in all_md


def test_render_agent_panel_raw_output_fallback():
    """_render_agent_panel falls back to raw JSON when no messages."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {"custom_field": "custom_value"}
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception


def test_render_agent_panel_no_expander():
    """_render_agent_panel uses container when use_expander=False."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {
        "agent_outputs": {
            "a0": {"message": "Container mode"}
        }
    }
    ca._render_agent_panel(
        "a0", output, expanded=True, use_expander=False
    )
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "a0" in all_md


def test_render_agent_panel_metadata_caption():
    """_render_agent_panel shows role and status as caption."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {
        "agent_outputs": {
            "a0": {
                "message": "Response",
                "role": "analyst",
                "status": "complete",
            }
        }
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "analyst" in all_captions


# ---------------------------------------------------------------------------
# _warn_explanation — additional edge cases
# ---------------------------------------------------------------------------


def test_warn_explanation_reverse_channel_pattern():
    """_warn_explanation matches the reverse channel warning."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._warn_explanation("channel X has a separate reverse channel")
        assert result is not None
        assert "bidirectional" in result


def test_warn_explanation_outgoing_edges_pattern():
    """_warn_explanation matches the outgoing edges warning."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        result = ca._warn_explanation(
            "agent X has 3 outgoing edges (expected 1 for a linear chain)"
        )
        assert result is not None
        assert "sequential" in result.lower()


# ---------------------------------------------------------------------------
# _is_stub_config — additional variations
# ---------------------------------------------------------------------------


def test_is_stub_config_mixed_agents():
    """_is_stub_config returns True when any agent lacks model_name."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.schema.agent_spec import AgentSpec
        from bili.aether.schema.enums import WorkflowType
        from bili.aether.schema.mas_config import MASConfig
        from bili.aether.ui import chat_app as ca

        agents = [
            AgentSpec(
                agent_id="a0",
                role="a0",
                objective="Perform task zero analysis",
                model_name="gpt-4o",
            ),
            AgentSpec(
                agent_id="a1",
                role="a1",
                objective="Perform task one analysis",
                model_name=None,
            ),
        ]
        config = MASConfig(
            mas_id="mixed",
            name="Mixed",
            description="Mixed agents",
            agents=agents,
            channels=[],
            workflow_type=WorkflowType.SEQUENTIAL,
        )
        assert ca._is_stub_config(config) is True


def test_is_stub_config_single_agent_with_model():
    """_is_stub_config returns False with one model-equipped agent."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        config = make_test_config(num_agents=1, model_name="gpt-4o")
        assert ca._is_stub_config(config) is False


# ---------------------------------------------------------------------------
# _extract_content — additional cases
# ---------------------------------------------------------------------------


def test_extract_content_messages_with_basemessage():
    """_extract_content handles BaseMessage objects in messages."""
    with patch.dict("sys.modules", _FM):
        from langchain_core.messages import AIMessage

        from bili.aether.ui import chat_app as ca

        output = {"messages": [AIMessage(content="AI response")]}
        assert "AI response" in ca._extract_content(output)


def test_extract_content_agent_outputs_no_message_no_id():
    """_extract_content renders k:v when no message or agent_id."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"agent_outputs": {"a0": {"result": "success", "count": 5}}}
        content = ca._extract_content(output)
        assert "result" in content


def test_extract_content_empty_agent_outputs():
    """_extract_content handles empty agent_outputs dict."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"agent_outputs": {}}
        result = ca._extract_content(output)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _build_role_map — additional variations
# ---------------------------------------------------------------------------


def test_build_role_map_all_different_roles():
    """_build_role_map includes all agents with different roles."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.schema.agent_spec import AgentSpec
        from bili.aether.schema.enums import WorkflowType
        from bili.aether.schema.mas_config import MASConfig
        from bili.aether.ui import chat_app as ca

        agents = [
            AgentSpec(
                agent_id="a0",
                role="writer",
                objective="Write tasks",
            ),
            AgentSpec(
                agent_id="a1",
                role="editor",
                objective="Edit tasks",
            ),
        ]
        config = MASConfig(
            mas_id="t",
            name="t",
            description="t",
            agents=agents,
            channels=[],
            workflow_type=WorkflowType.SEQUENTIAL,
        )
        role_map = ca._build_role_map(config)
        assert role_map == {"a0": "writer", "a1": "editor"}


def test_build_role_map_all_same_roles():
    """_build_role_map returns empty dict when all roles match ids."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        config = make_test_config(num_agents=1)
        # Override role to match agent_id
        config.agents[0].role = config.agents[0].agent_id
        assert ca._build_role_map(config) == {}


# ---------------------------------------------------------------------------
# _serialize_state_update — additional edge cases
# ---------------------------------------------------------------------------


def test_serialize_state_update_list_of_messages():
    """_serialize_state_update handles a list of mixed types."""
    with patch.dict("sys.modules", _FM):
        from langchain_core.messages import AIMessage, HumanMessage

        from bili.aether.ui import chat_app as ca

        result = ca._serialize_state_update(
            {
                "messages": [
                    HumanMessage(content="q"),
                    AIMessage(content="a"),
                    "plain string",
                ]
            }
        )
        assert result["messages"][0]["content"] == "q"
        assert result["messages"][1]["content"] == "a"
        assert result["messages"][2] == "plain string"


def test_serialize_state_update_empty_dict():
    """_serialize_state_update handles empty dict."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        assert ca._serialize_state_update({}) == {}


# ---------------------------------------------------------------------------
# _render_timeline
# ---------------------------------------------------------------------------


def test_render_timeline_empty_nodes():
    """_render_timeline returns early for empty node list."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ph = st.empty()
    ca._render_timeline(
        ph, completed=[], active=None, all_nodes=[],
        key_prefix="test_empty",
    )
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


def test_render_timeline_with_nodes():
    """_render_timeline renders chips for all nodes."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ph = st.empty()
    ca._render_timeline(
        ph,
        completed=["a0"],
        active="a1",
        all_nodes=["a0", "a1", "a2"],
        key_prefix="test_nodes",
    )
"""
    )
    at.run()
    assert not at.exception


def test_render_timeline_with_status_text():
    """_render_timeline renders a status caption when given."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    ph = st.empty()
    ca._render_timeline(
        ph,
        completed=[],
        active="a0",
        all_nodes=["a0"],
        key_prefix="test_status",
        status_text="Running agent...",
    )
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _is_provider_available — AWS Bedrock
# ---------------------------------------------------------------------------


def test_aws_bedrock_available_with_profile():
    """remote_aws_bedrock available when AWS_PROFILE is set."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"AWS_PROFILE": "default"}, clear=True):
            assert ca._is_provider_available("remote_aws_bedrock") is True


def test_aws_bedrock_unavailable_no_creds():
    """remote_aws_bedrock unavailable without credentials."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                assert ca._is_provider_available("remote_aws_bedrock") is False


# ---------------------------------------------------------------------------
# _validate_config display — warnings
# ---------------------------------------------------------------------------


def test_validate_config_with_warnings():
    """_validate_config shows success with warnings message."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    from bili.aether.validation import validate_mas
    from unittest.mock import MagicMock

    # Mock validate_mas to return warnings
    mock_result = MagicMock()
    mock_result.errors = []
    mock_result.warnings = ["some unknown warning text"]
    mock_result.valid = True
    with _patch.object(ca, "validate_mas", return_value=mock_result):
        result = ca._validate_config(mk())
    st.markdown(f"valid:{result}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "valid:True" in all_md


# ---------------------------------------------------------------------------
# Diagram renderers
# ---------------------------------------------------------------------------


def test_render_hierarchical_diagram():
    """_render_hierarchical_diagram renders tiered agents."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    from bili.aether.schema.mas_config import MASConfig
    from bili.aether.schema.enums import WorkflowType
    agents = [
        AgentSpec(
            agent_id="root", role="root",
            objective="Root node for orchestration", tier=1,
        ),
        AgentSpec(
            agent_id="child1", role="child1",
            objective="First child worker node", tier=2,
        ),
        AgentSpec(
            agent_id="child2", role="child2",
            objective="Second child worker node", tier=2,
        ),
    ]
    cfg = MASConfig(
        mas_id="hier", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.HIERARCHICAL,
    )
    ca._render_hierarchical_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_supervisor_diagram():
    """_render_supervisor_diagram renders hub-and-spoke."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    import importlib
    import bili.aether.ui.chat_app
    importlib.reload(bili.aether.ui.chat_app)
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.agent_spec import AgentSpec
    from bili.aether.schema.mas_config import MASConfig
    from bili.aether.schema.enums import WorkflowType
    agents = [
        AgentSpec(
            agent_id="supervisor", role="supervisor",
            objective="Supervise workers",
        ),
        AgentSpec(
            agent_id="worker1", role="worker",
            objective="Perform work task one",
        ),
        AgentSpec(
            agent_id="worker2", role="worker",
            objective="Perform work task two",
        ),
    ]
    cfg = MASConfig(
        mas_id="sup", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="supervisor",
    )
    ca._render_supervisor_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception
