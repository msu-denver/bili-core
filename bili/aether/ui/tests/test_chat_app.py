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
        import importlib

        import bili.aether.ui.chat_app

        importlib.reload(bili.aether.ui.chat_app)
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


# ---------------------------------------------------------------------------
# _on_send_to_attack_from_chat
# ---------------------------------------------------------------------------


def test_on_send_to_attack_no_config():
    """_on_send_to_attack_from_chat does nothing when no config loaded."""
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
    ca._on_send_to_attack_from_chat()
    st.markdown(f"no_crash:True")
"""
    )
    at.run()
    assert not at.exception
    assert "no_crash:True" in " ".join(m.value for m in at.markdown)


def test_on_send_to_attack_with_config():
    """_on_send_to_attack_from_chat pushes config to attack state."""
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
    cfg = mk(mas_id="attack_push_test")
    st.session_state.chat_config = cfg
    ca._on_send_to_attack_from_chat()
    st.markdown(f"pushed:{st.session_state.get('attack_config') is not None}")
"""
    )
    at.run()
    assert not at.exception
    assert "pushed:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _base_cache_key
# ---------------------------------------------------------------------------


def test_base_cache_key_strips_model_suffix():
    """_base_cache_key strips :model= suffix from cache key."""
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
    st.session_state.chat_yaml_path = "/path/config.yaml:model=gpt-4o"
    result = ca._base_cache_key()
    st.markdown(f"key:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "key:/path/config.yaml" in " ".join(m.value for m in at.markdown)


def test_base_cache_key_strips_stub_suffix():
    """_base_cache_key strips :stub suffix from cache key."""
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
    st.session_state.chat_yaml_path = "/path/config.yaml:stub"
    result = ca._base_cache_key()
    st.markdown(f"key:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "key:/path/config.yaml" in " ".join(m.value for m in at.markdown)


def test_base_cache_key_no_suffix():
    """_base_cache_key returns path unchanged when no suffix present."""
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
    st.session_state.chat_yaml_path = "/path/plain_config.yaml"
    result = ca._base_cache_key()
    st.markdown(f"key:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "key:/path/plain_config.yaml" in " ".join(m.value for m in at.markdown)


def test_base_cache_key_default_when_no_path():
    """_base_cache_key returns 'config' when no chat_yaml_path set."""
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
    st.session_state.pop("chat_yaml_path", None)
    result = ca._base_cache_key()
    st.markdown(f"key:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "key:config" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_chat_agent_details
# ---------------------------------------------------------------------------


def test_render_chat_agent_details():
    """_render_chat_agent_details renders agent info."""
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
    agent = AgentSpec(
        agent_id="a0",
        role="analyst",
        objective="Perform deep analysis",
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=1000,
        capabilities=["inter_agent_communication"],
        tools=["weather_api_tool"],
    )
    ca._render_chat_agent_details(agent)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "a0" in all_md
    assert "gpt-4o" in all_md


# ---------------------------------------------------------------------------
# _render_supervisor_diagram
# ---------------------------------------------------------------------------


def test_render_supervisor_diagram():
    """_render_supervisor_diagram renders hub-and-spoke layout."""
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
        AgentSpec(agent_id="supervisor", role="supervisor", objective="Coordinate agents"),
        AgentSpec(agent_id="worker_0", role="worker_0", objective="Perform work type 0"),
        AgentSpec(agent_id="worker_1", role="worker_1", objective="Perform work type 1"),
    ]
    cfg = MASConfig(
        mas_id="sup_test", name="t", description="t",
        agents=agents, channels=[], workflow_type=WorkflowType.SUPERVISOR,
        entry_point="supervisor",
    )
    ca._render_supervisor_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_hierarchical_diagram (first instance)
# ---------------------------------------------------------------------------


def test_render_hierarchical_diagram():
    """_render_hierarchical_diagram renders tiered layout."""
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
        AgentSpec(agent_id="root", role="root", objective="Root coordinator"),
        AgentSpec(agent_id="child_0", role="child_0", objective="Perform child tasks"),
    ]
    cfg = MASConfig(
        mas_id="hier_test", name="t", description="t",
        agents=agents, channels=[], workflow_type=WorkflowType.HIERARCHICAL,
    )
    ca._render_hierarchical_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_mas_diagram dispatch
# ---------------------------------------------------------------------------


def test_render_mas_diagram_sequential():
    """_render_mas_diagram dispatches to sequential renderer."""
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
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_mas_structure fallback (no flow)
# ---------------------------------------------------------------------------


def test_render_mas_structure_fallback():
    """_render_mas_structure renders text-based diagram when flow unavailable."""
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
    cfg = mk(mas_id="struct_test")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    # Force flow unavailable
    with _patch.object(ca, "_FLOW_AVAILABLE", False):
        ca._render_mas_structure(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _validate_config with warnings
# ---------------------------------------------------------------------------


def test_validate_config_with_warnings():
    """_validate_config shows success with warnings when warnings present."""
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
    mock_result = _Mock()
    mock_result.errors = []
    mock_result.warnings = ["unknown warning text here"]
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
    assert "warnings" in " ".join(s.value for s in at.success)


def test_validate_config_with_errors():
    """_validate_config returns False when errors present."""
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
    mock_result = _Mock()
    mock_result.errors = ["Missing entry point"]
    mock_result.warnings = []
    mock_result.valid = False
    with _patch.object(ca, "validate_mas", return_value=mock_result):
        result = ca._validate_config(mk())
    st.markdown(f"valid:{result}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "valid:False" in all_md
    assert "Missing entry point" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _render_timeline with role_map and status_text
# ---------------------------------------------------------------------------


def test_render_timeline_with_role_map():
    """_render_timeline uses role_map for chip labels."""
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
        active=None,
        all_nodes=["a0", "a1"],
        key_prefix="test_roles",
        role_map={"a0": "writer", "a1": "editor"},
        status_text="Done processing",
    )
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_fallback_diagram with channels
# ---------------------------------------------------------------------------


def test_render_fallback_diagram_with_channels():
    """_render_fallback_diagram renders channels when present."""
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
    from bili.aether.ui.tests.conftest import make_test_config as mk
    cfg = mk(num_agents=2)
    ca._render_fallback_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Channels" in all_md


# ---------------------------------------------------------------------------
# _initialize_executor caching
# ---------------------------------------------------------------------------


def test_initialize_executor_caches_on_same_key():
    """_initialize_executor skips reinit for same cache key."""
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
    st.session_state.chat_yaml_path = "cached_key"
    st.session_state.chat_config = cfg
    # Should return early without reinitializing
    ca._initialize_executor(cfg, "cached_key")
    st.markdown(f"still_cached:{st.session_state.get('chat_config') is not None}")
"""
    )
    at.run()
    assert not at.exception
    assert "still_cached:True" in " ".join(m.value for m in at.markdown)


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


def test_render_supervisor_diagram_second():
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


# ===========================================================================
# NEW TESTS: Coverage expansion for chat_app.py
# ===========================================================================


# ---------------------------------------------------------------------------
# _apply_model_patch
# ---------------------------------------------------------------------------


def test_apply_model_patch_sets_model():
    """_apply_model_patch patches all agents with the given model_id."""
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
    cfg = mk(mas_id="patch_test")
    st.session_state.chat_yaml_path = "patch_test"
    st.session_state.chat_config = cfg
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "MASExecutor", return_value=mock_exec):
        ca._apply_model_patch(cfg, "gpt-4o")
    patched_cfg = st.session_state.get("chat_config")
    if patched_cfg:
        st.markdown(f"model:{patched_cfg.agents[0].model_name}")
    else:
        st.markdown("model:None")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "model:gpt-4o" in all_md


def test_apply_model_patch_stub_mode():
    """_apply_model_patch with None sets stub mode."""
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
    cfg = mk(mas_id="stub_patch", model_name="gpt-4o")
    st.session_state.chat_yaml_path = "stub_patch"
    st.session_state.chat_config = cfg
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "MASExecutor", return_value=mock_exec):
        ca._apply_model_patch(cfg, None)
    patched_cfg = st.session_state.get("chat_config")
    if patched_cfg:
        st.markdown(f"stub:{patched_cfg.agents[0].model_name is None}")
    else:
        st.markdown("stub:no_config")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "stub:True" in all_md


def test_apply_model_patch_warns_pipeline_agents():
    """_apply_model_patch warns when pipeline agents are present."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.enums import WorkflowType
from bili.aether.schema.mas_config import MASConfig
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    from bili.aether.schema.pipeline_spec import PipelineSpec, PipelineNodeSpec, PipelineEdgeSpec
    agents = [
        AgentSpec(
            agent_id="pipe_agent", role="pipe",
            objective="Pipelined agent for processing",
            pipeline=PipelineSpec(
                nodes=[PipelineNodeSpec(node_id="n1", node_type="react_agent")],
                edges=[PipelineEdgeSpec(from_node="n1", to_node="END")],
            ),
        ),
    ]
    cfg = MASConfig(
        mas_id="pipe_test", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.SEQUENTIAL,
    )
    st.session_state.chat_yaml_path = "pipe_test"
    st.session_state.chat_config = cfg
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "MASExecutor", return_value=mock_exec):
        ca._apply_model_patch(cfg, "gpt-4o")
"""
    )
    at.run()
    assert not at.exception
    all_warnings = " ".join(w.value for w in at.warning)
    assert "pipeline" in all_warnings.lower() or "Pipeline" in all_warnings


# ---------------------------------------------------------------------------
# _render_thread_list
# ---------------------------------------------------------------------------


def test_render_thread_list_empty():
    """_render_thread_list returns early when no threads exist."""
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
    ca._render_thread_list()
    st.markdown("no_threads:True")
"""
    )
    at.run()
    assert not at.exception
    assert "no_threads:True" in " ".join(m.value for m in at.markdown)


def test_render_thread_list_with_threads():
    """_render_thread_list renders buttons for existing threads."""
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
    ca._render_thread_list()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Conversations" in all_md


def test_render_thread_list_with_multiple_threads():
    """_render_thread_list renders buttons for multiple threads."""
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
    ca._new_thread("alpha_mas")
    ca._new_thread("beta_mas")
    ca._render_thread_list()
"""
    )
    at.run()
    assert not at.exception
    assert len(at.button) >= 2


def test_render_thread_list_editing_mode():
    """_render_thread_list renders rename input when editing."""
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
    tid = ca._new_thread("edit_test")
    st.session_state.chat_editing_thread = tid
    ca._render_thread_list()
"""
    )
    at.run()
    assert not at.exception
    assert len(at.text_input) >= 2


# ---------------------------------------------------------------------------
# _build_chat_model_options
# ---------------------------------------------------------------------------


def test_build_chat_model_options_filters_by_provider():
    """_build_chat_model_options filters out unavailable providers."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        mock_data = (
            ["[OpenAI] gpt-4o", "[Local] llama"],
            ["gpt-4o", "llama-local"],
            ["remote_openai", "local_llamacpp"],
        )
        with patch.object(ca, "_load_all_model_options", return_value=mock_data):
            with patch.object(
                ca,
                "_is_provider_available",
                side_effect=lambda k: k == "local_llamacpp",
            ):
                display, ids = ca._build_chat_model_options()
        assert len(display) == 1
        assert ids[0] == "llama-local"


# ---------------------------------------------------------------------------
# _configure_page
# ---------------------------------------------------------------------------


def test_configure_page_runs():
    """_configure_page sets page config without error."""
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
    ca._configure_page()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _initialize_executor — error and success paths
# ---------------------------------------------------------------------------


def test_initialize_executor_stores_error_on_failure():
    """_initialize_executor stores error in session state on exception."""
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
    with _patch.object(ca, "MASExecutor", side_effect=RuntimeError("init failed")):
        ca._initialize_executor(cfg, "fail_key")
    st.markdown(f"error:{st.session_state.get('chat_load_error')}")
    st.markdown(f"no_config:{st.session_state.get('chat_config') is None}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "error:init failed" in all_md
    assert "no_config:True" in all_md


def test_initialize_executor_success():
    """_initialize_executor stores config and executor on success."""
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
    cfg = mk(mas_id="success_init")
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "MASExecutor", return_value=mock_exec):
        ca._initialize_executor(cfg, "success_key")
    st.markdown(f"has_config:{st.session_state.get('chat_config') is not None}")
    st.markdown(f"has_executor:{st.session_state.get('chat_executor') is not None}")
    st.markdown(f"key:{st.session_state.get('chat_yaml_path')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_config:True" in all_md
    assert "has_executor:True" in all_md
    assert "key:success_key" in all_md


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_skips_when_already_loaded():
    """_load_config returns early when config already loaded for same path."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
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
    st.session_state.chat_yaml_path = "/some/path.yaml"
    st.session_state.chat_config = cfg
    ca._load_config(Path("/some/path.yaml"))
    st.markdown(f"skipped:{st.session_state.get('chat_config') is cfg}")
"""
    )
    at.run()
    assert not at.exception
    assert "skipped:True" in " ".join(m.value for m in at.markdown)


def test_load_config_handles_parse_error():
    """_load_config stores error when YAML parsing fails."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    with _patch.object(ca, "load_mas_from_yaml", side_effect=ValueError("bad yaml")):
        ca._load_config(Path("/bad/config.yaml"))
    st.markdown(f"error:{st.session_state.get('chat_load_error')}")
"""
    )
    at.run()
    assert not at.exception
    assert "error:bad yaml" in " ".join(m.value for m in at.markdown)


def test_load_config_skips_with_model_suffix():
    """_load_config skips reload when current key has model suffix from same base."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
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
    st.session_state.chat_yaml_path = "/some/path.yaml:model=gpt-4o"
    st.session_state.chat_config = cfg
    ca._load_config(Path("/some/path.yaml"))
    st.markdown(f"preserved:{st.session_state.get('chat_config') is cfg}")
"""
    )
    at.run()
    assert not at.exception
    assert "preserved:True" in " ".join(m.value for m in at.markdown)


def test_load_config_invalid_clears_state():
    """_load_config clears state when validation fails."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
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
    with _patch.object(ca, "load_mas_from_yaml", return_value=cfg):
        with _patch.object(ca, "_validate_config", return_value=False):
            ca._load_config(Path("/test/invalid.yaml"))
    st.markdown(f"cleared:{st.session_state.get('chat_config') is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_load_config_success():
    """_load_config initializes executor on successful load."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    cfg = mk(mas_id="load_success")
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "load_mas_from_yaml", return_value=cfg):
        with _patch.object(ca, "MASExecutor", return_value=mock_exec):
            ca._load_config(Path("/test/good.yaml"))
    st.markdown(f"loaded:{st.session_state.get('chat_config') is not None}")
"""
    )
    at.run()
    assert not at.exception
    assert "loaded:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _load_uploaded_config
# ---------------------------------------------------------------------------


def test_load_uploaded_config_skips_when_already_loaded():
    """_load_uploaded_config skips reinit when same config already loaded."""
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
    st.session_state.chat_yaml_path = "uploaded:test.yaml"
    st.session_state.chat_config = cfg
    ca._load_uploaded_config("test.yaml", cfg)
    st.markdown(f"skipped:{st.session_state.get('chat_config') is cfg}")
"""
    )
    at.run()
    assert not at.exception
    assert "skipped:True" in " ".join(m.value for m in at.markdown)


def test_load_uploaded_config_invalid_clears_state():
    """_load_uploaded_config clears state when validation fails."""
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
    with _patch.object(ca, "_validate_config", return_value=False):
        ca._load_uploaded_config("bad.yaml", cfg)
    st.markdown(f"cleared:{st.session_state.get('chat_config') is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_load_uploaded_config_success():
    """_load_uploaded_config initializes executor on valid config."""
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
    cfg = mk(mas_id="upload_success")
    mock_exec = _Mock()
    mock_exec.initialize = _Mock()
    with _patch.object(ca, "MASExecutor", return_value=mock_exec):
        ca._load_uploaded_config("upload.yaml", cfg)
    st.markdown(f"loaded:{st.session_state.get('chat_config') is not None}")
    st.markdown(f"base:{st.session_state.get('chat_config_base') is not None}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "loaded:True" in all_md
    assert "base:True" in all_md


def test_load_uploaded_config_skips_with_model_suffix():
    """_load_uploaded_config skips when current key has model suffix."""
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
    st.session_state.chat_yaml_path = "uploaded:test.yaml:model=gpt-4o"
    st.session_state.chat_config = cfg
    ca._load_uploaded_config("test.yaml", cfg)
    st.markdown(f"skipped:{st.session_state.get('chat_config') is cfg}")
"""
    )
    at.run()
    assert not at.exception
    assert "skipped:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_sidebar
# ---------------------------------------------------------------------------


def test_render_sidebar_renders_title():
    """_render_sidebar renders the AETHER Chat title."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    with _patch.object(ca, "LOGO_PATH", Path("/nonexistent/logo.png")):
        with _patch.object(ca, "render_sidebar_content"):
            ca._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "AETHER Chat" in all_md


# ---------------------------------------------------------------------------
# render_page
# ---------------------------------------------------------------------------


def test_render_page_runs():
    """render_page sets page config and renders without error."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from pathlib import Path
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    with _patch.object(ca, "LOGO_PATH", Path("/nonexistent/logo.png")):
        with _patch.object(ca, "render_sidebar_content"):
            ca.render_page()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _run_turn
# ---------------------------------------------------------------------------


def test_run_turn_no_executor():
    """_run_turn shows error when no executor is available."""
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
    ca._run_turn("Hello")
"""
    )
    at.run()
    assert not at.exception
    all_err = " ".join(e.value for e in at.error)
    assert "No executor" in all_err


def test_run_turn_no_config():
    """_run_turn shows error when no config is loaded."""
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
    st.session_state.chat_executor = _Mock()
    ca._run_turn("Hello")
"""
    )
    at.run()
    assert not at.exception
    all_err = " ".join(e.value for e in at.error)
    assert "No configuration" in all_err


def test_run_turn_with_streaming():
    """_run_turn processes streaming events and appends turn to history."""
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
    cfg = mk(mas_id="run_turn_test")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("run_turn_test")
    mock_exec = _Mock()
    events = [
        ("__token__", {"node": "agent_0", "token": "Hello"}),
        ("__token__", {"node": "agent_0", "token": " world"}),
        ("__node_complete__", {
            "node": "agent_0",
            "state_update": {
                "agent_outputs": {"agent_0": {"message": "Hello world"}}
            },
        }),
    ]
    mock_exec.run_streaming_tokens.return_value = iter(events)
    st.session_state.chat_executor = mock_exec
    with _patch.object(st, "rerun", side_effect=Exception("rerun_called")):
        try:
            ca._run_turn("Test input")
        except Exception as e:
            if "rerun_called" not in str(e):
                raise
    msgs = st.session_state["chat_threads"][tid]["messages"]
    st.markdown(f"count:{len(msgs)}")
"""
    )
    at.run()
    assert not at.exception or any("rerun_called" in str(e.value) for e in at.exception)


def test_run_turn_handles_execution_error():
    """_run_turn stores error in turn when executor raises."""
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
    cfg = mk(mas_id="error_turn")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("error_turn")
    mock_exec = _Mock()
    mock_exec.run_streaming_tokens.side_effect = RuntimeError("LLM error")
    st.session_state.chat_executor = mock_exec
    with _patch.object(st, "rerun", side_effect=Exception("rerun_called")):
        try:
            ca._run_turn("Failing input")
        except Exception as e:
            if "rerun_called" not in str(e):
                raise
"""
    )
    at.run()
    assert not at.exception or any("rerun_called" in str(e.value) for e in at.exception)


def test_run_turn_skips_non_agent_tokens():
    """_run_turn ignores tokens from internal nodes not in agents list."""
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
    cfg = mk(mas_id="skip_internal")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("skip_internal")
    mock_exec = _Mock()
    events = [
        ("__token__", {"node": "__internal_router__", "token": "ignored"}),
        ("__node_complete__", {
            "node": "__internal_router__",
            "state_update": {"routing": "done"},
        }),
        ("__node_complete__", {
            "node": "agent_0",
            "state_update": {
                "agent_outputs": {"agent_0": {"message": "Real output"}}
            },
        }),
    ]
    mock_exec.run_streaming_tokens.return_value = iter(events)
    st.session_state.chat_executor = mock_exec
    with _patch.object(st, "rerun", side_effect=Exception("rerun_called")):
        try:
            ca._run_turn("Test")
        except Exception as e:
            if "rerun_called" not in str(e):
                raise
"""
    )
    at.run()
    assert not at.exception or any("rerun_called" in str(e.value) for e in at.exception)


def test_run_turn_skips_none_state_update():
    """_run_turn skips node_complete events with None state_update."""
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
    cfg = mk(mas_id="none_state")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("none_state")
    mock_exec = _Mock()
    events = [
        ("__node_complete__", {
            "node": "agent_0",
            "state_update": None,
        }),
        ("__node_complete__", {
            "node": "agent_0",
            "state_update": {
                "agent_outputs": {"agent_0": {"message": "Real result"}}
            },
        }),
    ]
    mock_exec.run_streaming_tokens.return_value = iter(events)
    st.session_state.chat_executor = mock_exec
    with _patch.object(st, "rerun", side_effect=Exception("rerun_called")):
        try:
            ca._run_turn("Test none state")
        except Exception as e:
            if "rerun_called" not in str(e):
                raise
"""
    )
    at.run()
    assert not at.exception or any("rerun_called" in str(e.value) for e in at.exception)


def test_run_turn_hitl_interrupt():
    """_run_turn stores HITL pending state on interrupt event."""
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
    cfg = mk(mas_id="hitl_interrupt")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("hitl_interrupt")
    mock_exec = _Mock()
    events = [
        ("__human_interrupt__", {"next": ["agent_1"], "thread_id": tid}),
    ]
    mock_exec.run_streaming_tokens.return_value = iter(events)
    st.session_state.chat_executor = mock_exec
    with _patch.object(st, "rerun", side_effect=Exception("rerun_called")):
        try:
            ca._run_turn("HITL test input")
        except Exception as e:
            if "rerun_called" not in str(e):
                raise
"""
    )
    at.run()
    assert not at.exception or any("rerun_called" in str(e.value) for e in at.exception)


# ---------------------------------------------------------------------------
# _render_chat_area — additional paths
# ---------------------------------------------------------------------------


def test_chat_area_with_hitl_pending():
    """_render_chat_area shows HITL form when hitl_pending is set."""
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
    cfg = mk(mas_id="hitl_test")
    st.session_state.chat_config = cfg
    mock_exec = _Mock()
    st.session_state.chat_executor = mock_exec
    tid = ca._new_thread("hitl_test")
    st.session_state.hitl_pending = {
        "next": ["agent_1"],
        "thread_id": tid,
        "partial_turn": {
            "role": "user",
            "content": "Paused question",
            "turn_index": 0,
            "agent_trace": [],
        },
    }
    with _patch.object(ca, "_render_mas_structure"):
        ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    all_info = " ".join(i.value for i in at.info)
    assert "review" in all_info.lower() or "agent_1" in all_info or "role_1" in all_info


def test_chat_area_renders_description():
    """_render_chat_area renders config description as caption."""
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
    cfg = mk(description="My special MAS description")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    with _patch.object(ca, "_render_mas_structure"):
        ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "My special MAS description" in all_captions


# ---------------------------------------------------------------------------
# _render_mas_diagram — dispatch for each workflow type
# ---------------------------------------------------------------------------


def test_render_mas_diagram_supervisor_dispatch():
    """_render_mas_diagram dispatches to supervisor renderer."""
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
        AgentSpec(agent_id="sup", role="sup", objective="Supervise tasks"),
        AgentSpec(agent_id="w1", role="w1", objective="Work tasks"),
    ]
    cfg = MASConfig(
        mas_id="t", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="sup",
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_mas_diagram_consensus_dispatch():
    """_render_mas_diagram dispatches to consensus renderer."""
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
        AgentSpec(agent_id="v0", role="v0", objective="Vote on proposals"),
        AgentSpec(agent_id="v1", role="v1", objective="Vote on proposals"),
    ]
    cfg = MASConfig(
        mas_id="t", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_mas_diagram_hierarchical_dispatch():
    """_render_mas_diagram dispatches to hierarchical renderer."""
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
        AgentSpec(agent_id="r", role="r", objective="Root tasks", tier=1),
        AgentSpec(agent_id="c", role="c", objective="Child tasks", tier=2),
    ]
    cfg = MASConfig(
        mas_id="t", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.HIERARCHICAL,
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_mas_diagram_custom_fallback():
    """_render_mas_diagram uses fallback for custom workflow type."""
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
        AgentSpec(agent_id="a0", role="a0", objective="Custom agent tasks"),
    ]
    cfg = MASConfig(
        mas_id="t", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.CUSTOM,
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_stored_turn — trace node selection
# ---------------------------------------------------------------------------


def test_render_stored_turn_with_selected_trace_node():
    """_render_stored_turn expands the selected agent panel."""
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
    st.session_state.aether_selected_trace_node = (0, "agent_0")
    turn = {
        "content": "Trace selection test",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "Selected agent output"}
                    }
                },
            },
            {
                "agent_id": "agent_1",
                "output": {
                    "agent_outputs": {
                        "agent_1": {"message": "Other agent output"}
                    }
                },
            },
        ],
    }
    ca._render_stored_turn(turn)
    st.markdown(f"consumed:{st.session_state.get('aether_selected_trace_node') is None}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "consumed:True" in all_md


def test_render_stored_turn_ignores_wrong_turn_selection():
    """_render_stored_turn ignores selected trace node for a different turn."""
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
    st.session_state.aether_selected_trace_node = (5, "agent_0")
    turn = {
        "content": "Wrong turn selection",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "Agent output"}
                    }
                },
            },
        ],
    }
    ca._render_stored_turn(turn)
    st.markdown(f"kept:{st.session_state.get('aether_selected_trace_node') is not None}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "kept:True" in all_md


def test_render_stored_turn_shows_last_agent_summary():
    """_render_stored_turn shows the last agent's message as visible summary."""
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
    cfg = mk(num_agents=2)
    st.session_state.chat_config = cfg
    turn = {
        "content": "Summary test",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "First agent reply"}
                    }
                },
            },
            {
                "agent_id": "agent_1",
                "output": {
                    "agent_outputs": {
                        "agent_1": {"message": "Final summary reply"}
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
    assert "Final summary reply" in all_md


def test_render_stored_turn_deduplicates_agent_ids():
    """_render_stored_turn deduplicates repeated agent IDs in timeline."""
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
    cfg = mk(num_agents=1)
    st.session_state.chat_config = cfg
    turn = {
        "content": "Dedup test",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "First pass"}
                    }
                },
            },
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "Second pass"}
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


# ---------------------------------------------------------------------------
# _validate_config — additional branches
# ---------------------------------------------------------------------------


def test_validate_config_known_warning_popover():
    """_validate_config renders popover for known warning patterns."""
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
    mock_result = _Mock()
    mock_result.errors = []
    mock_result.warnings = ["agent X has no channel connections"]
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
    assert "warnings" in " ".join(s.value for s in at.success)


def test_validate_config_no_warnings_clean_success():
    """_validate_config shows clean success with no warnings."""
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
    mock_result = _Mock()
    mock_result.errors = []
    mock_result.warnings = []
    mock_result.valid = True
    with _patch.object(ca, "validate_mas", return_value=mock_result):
        result = ca._validate_config(mk())
    st.markdown(f"valid:{result}")
"""
    )
    at.run()
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "valid" in success_text.lower()


# ---------------------------------------------------------------------------
# _render_agent_panel — more edge cases
# ---------------------------------------------------------------------------


def test_render_agent_panel_live_basemessage():
    """_render_agent_panel handles live BaseMessage in messages list."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from langchain_core.messages import AIMessage
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui import chat_app as ca
    output = {"messages": [AIMessage(content="Live AI response")]}
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Live AI response" in all_md


def test_render_agent_panel_mismatched_agent_id():
    """_render_agent_panel handles agent_id not found in agent_outputs."""
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
        "agent_outputs": {"other_agent": {"message": "Wrong agent"}},
        "messages": [{"content": "Fallback for mismatch"}],
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Fallback for mismatch" in all_md


def test_render_agent_panel_empty_message():
    """_render_agent_panel handles agent output with empty message string."""
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
            "a0": {"message": "", "status": "complete", "role": "tester"}
        }
    }
    ca._render_agent_panel("a0", output, expanded=True)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_chat_agent_details — minimal
# ---------------------------------------------------------------------------


def test_render_chat_agent_details_minimal():
    """_render_chat_agent_details handles agent with minimal fields."""
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
    agent = AgentSpec(
        agent_id="minimal",
        role="minimal",
        objective="Minimal agent without extras",
    )
    ca._render_chat_agent_details(agent)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "minimal" in all_md


# ---------------------------------------------------------------------------
# _extract_content — plain string in messages
# ---------------------------------------------------------------------------


def test_extract_content_messages_with_str():
    """_extract_content handles raw string last in messages."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        output = {"messages": ["plain string"]}
        assert "plain string" in ca._extract_content(output)


# ---------------------------------------------------------------------------
# Diagram edge cases — empty configs
# ---------------------------------------------------------------------------


def test_render_sequential_diagram_single_agent():
    """_render_sequential_diagram handles a single agent."""
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
    cfg = mk(num_agents=1)
    ca._render_sequential_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_sequential_diagram_empty():
    """_render_sequential_diagram returns early with no agents."""
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
    cfg = _Mock()
    cfg.agents = []
    ca._render_sequential_diagram(cfg)
    import streamlit as st
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


def test_render_supervisor_diagram_no_specialists():
    """_render_supervisor_diagram falls back when only coordinator."""
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
        AgentSpec(agent_id="sup", role="sup", objective="Solo supervisor"),
    ]
    cfg = MASConfig(
        mas_id="t", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="sup",
    )
    ca._render_supervisor_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_consensus_diagram_empty():
    """_render_consensus_diagram returns early with no agents."""
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
    cfg = _Mock()
    cfg.agents = []
    ca._render_consensus_diagram(cfg)
    import streamlit as st
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


def test_render_hierarchical_diagram_empty():
    """_render_hierarchical_diagram returns early with no agents."""
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
    cfg = _Mock()
    cfg.agents = []
    ca._render_hierarchical_diagram(cfg)
    import streamlit as st
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


def test_render_fallback_diagram_empty():
    """_render_fallback_diagram handles empty config."""
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
    cfg = _Mock()
    cfg.agents = []
    cfg.channels = []
    ca._render_fallback_diagram(cfg)
    import streamlit as st
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_mas_structure with flow
# ---------------------------------------------------------------------------


def test_render_mas_structure_with_flow():
    """_render_mas_structure calls _render_flow_graph when flow available."""
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
    cfg = mk(mas_id="flow_struct")
    st.session_state.chat_config = cfg
    with _patch.object(ca, "_FLOW_AVAILABLE", True):
        with _patch.object(ca, "_render_flow_graph"):
            ca._render_mas_structure(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Thread management — additional
# ---------------------------------------------------------------------------


def test_delete_thread_clears_editing_state():
    """_delete_thread clears editing state when deleted thread is being edited."""
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
    tid = ca._new_thread("edit_del_test")
    st.session_state.chat_editing_thread = tid
    ca._delete_thread(tid)
    st.markdown(f"editing_cleared:{st.session_state.get('chat_editing_thread') is None}")
"""
    )
    at.run()
    assert not at.exception
    assert "editing_cleared:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _base_cache_key — default
# ---------------------------------------------------------------------------


def test_base_cache_key_default():
    """_base_cache_key returns 'config' when no yaml path set."""
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
    result = ca._base_cache_key()
    st.markdown(f"key:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "key:config" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_hitl_form
# ---------------------------------------------------------------------------


def test_render_hitl_form_shows_form():
    """_render_hitl_form renders the review form with correct info."""
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
    cfg = mk(mas_id="hitl_form")
    st.session_state.chat_config = cfg
    tid = ca._new_thread("hitl_form")
    mock_exec = _Mock()
    st.session_state.chat_executor = mock_exec
    st.session_state.hitl_pending = {
        "next": ["agent_1"],
        "thread_id": tid,
        "partial_turn": {
            "role": "user",
            "content": "Waiting for review",
            "turn_index": 0,
            "agent_trace": [],
        },
    }
    ca._render_hitl_form(mock_exec)
"""
    )
    at.run()
    assert not at.exception
    all_info = " ".join(i.value for i in at.info)
    assert "review" in all_info.lower() or "agent_1" in all_info or "role_1" in all_info


# ---------------------------------------------------------------------------
# _is_provider_available — additional credentials
# ---------------------------------------------------------------------------


def test_aws_bedrock_available_with_session_token():
    """remote_aws_bedrock available with AWS_SESSION_TOKEN."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"AWS_SESSION_TOKEN": "tok"}, clear=True):
            assert ca._is_provider_available("remote_aws_bedrock") is True


def test_aws_bedrock_available_with_access_key():
    """remote_aws_bedrock available with AWS_ACCESS_KEY_ID."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "AKIA..."}, clear=True):
            assert ca._is_provider_available("remote_aws_bedrock") is True


def test_google_provider_available_with_app_creds():
    """remote_google_vertex available with GOOGLE_APPLICATION_CREDENTIALS."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        with patch.dict(
            "os.environ",
            {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"},
            clear=True,
        ):
            assert ca._is_provider_available("remote_google_vertex") is True


# ---------------------------------------------------------------------------
# _render_chat_area — streaming response display
# ---------------------------------------------------------------------------


def test_render_chat_area_with_config_and_empty_thread():
    """_render_chat_area renders chat input when config loaded and no messages."""
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
    cfg = mk(mas_id="empty_thread_test")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    with _patch.object(ca, "_render_mas_structure"):
        ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception


def test_render_chat_area_with_hitl_pending():
    """_render_chat_area shows HITL form when hitl_pending is in state."""
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
    cfg = mk(mas_id="hitl_test")
    st.session_state.chat_config = cfg
    st.session_state.chat_executor = _Mock()
    st.session_state.hitl_pending = {
        "agent_id": "agent_0",
        "question": "Continue?",
        "partial_turn": {
            "role": "user", "content": "test",
            "turn_index": 0, "agent_trace": [],
        },
    }
    with _patch.object(ca, "_render_mas_structure"):
        with _patch.object(ca, "_render_hitl_form"):
            ca._render_chat_area()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_main and render_main layout tests
# ---------------------------------------------------------------------------


def test_render_main_renders_sidebar_and_chat():
    """render_main renders both sidebar and chat area."""
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
    with _patch.object(ca, "_render_sidebar"):
        with _patch.object(ca, "_render_chat_area"):
            ca.render_main()
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _delete_thread while editing that thread
# ---------------------------------------------------------------------------


def test_delete_thread_clears_editing_state():
    """Deleting a thread that is being edited clears editing state."""
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
    st.session_state.chat_editing_thread = tid
    ca._delete_thread(tid)
    st.markdown(f"editing_cleared:{'chat_editing_thread' not in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "editing_cleared:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_thread_list with threads
# ---------------------------------------------------------------------------


def test_render_thread_list_with_threads():
    """_render_thread_list renders thread buttons when threads exist."""
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
    ca._new_thread("test_mas_1")
    ca._new_thread("test_mas_2")
    ca._render_thread_list()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Conversations" in all_md


def test_render_thread_list_empty():
    """_render_thread_list renders nothing when no threads exist."""
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
    st.session_state.pop("chat_threads", None)
    ca._render_thread_list()
    st.markdown("done:True")
"""
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _validate_config with known warning pattern (popover)
# ---------------------------------------------------------------------------


def test_validate_config_known_warning_shows_popover():
    """_validate_config shows popover for known warning patterns."""
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
    mock_result = _Mock()
    mock_result.errors = []
    mock_result.warnings = ["agent X has no channel connections"]
    mock_result.valid = True
    with _patch.object(ca, "validate_mas", return_value=mock_result):
        result = ca._validate_config(mk())
    st.markdown(f"valid:{result}")
"""
    )
    at.run()
    assert not at.exception
    assert "valid:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _initialize_executor error handling
# ---------------------------------------------------------------------------


def test_initialize_executor_stores_error_on_failure():
    """_initialize_executor stores error in session state on failure."""
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
    with _patch.object(ca, "MASExecutor", side_effect=ValueError("init failed")):
        ca._initialize_executor(cfg, "fail_key")
    st.markdown(f"error:{st.session_state.get('chat_load_error')}")
"""
    )
    at.run()
    assert not at.exception
    assert "error:init failed" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_mas_structure with flow available
# ---------------------------------------------------------------------------


def test_render_mas_structure_with_flow_available():
    """_render_mas_structure calls _render_flow_graph when flow is available."""
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
    cfg = mk(mas_id="flow_test")
    with _patch.object(ca, "_FLOW_AVAILABLE", True):
        with _patch.object(ca, "_render_flow_graph"):
            ca._render_mas_structure(cfg)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_stored_turn with selected trace node
# ---------------------------------------------------------------------------


def test_render_stored_turn_with_selected_trace_node():
    """_render_stored_turn auto-expands agent when trace node selected."""
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
    cfg = mk(mas_id="trace_select_test")
    st.session_state.chat_config = cfg
    st.session_state.aether_selected_trace_node = (0, "agent_0")
    turn = {
        "content": "Trace selection test",
        "turn_index": 0,
        "agent_trace": [
            {
                "agent_id": "agent_0",
                "output": {
                    "agent_outputs": {
                        "agent_0": {"message": "Selected agent output"}
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
    assert "Trace selection test" in all_md


# ---------------------------------------------------------------------------
# _run_turn error paths
# ---------------------------------------------------------------------------


def test_run_turn_no_executor():
    """_run_turn shows error when no executor available."""
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
    st.session_state.pop("chat_executor", None)
    ca._run_turn("Hello")
"""
    )
    at.run()
    assert not at.exception
    assert "No executor" in " ".join(e.value for e in at.error)


def test_run_turn_no_config():
    """_run_turn shows error when no config loaded."""
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
    st.session_state.chat_executor = _Mock()
    st.session_state.pop("chat_config", None)
    ca._run_turn("Hello")
"""
    )
    at.run()
    assert not at.exception
    assert "No configuration" in " ".join(e.value for e in at.error)


# ---------------------------------------------------------------------------
# _extract_content with BaseMessage in messages (non-dict)
# ---------------------------------------------------------------------------


def test_extract_content_basemessage_content_attr():
    """_extract_content gets content attr from non-dict message object."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui import chat_app as ca

        class FakeMsg:
            """Fake message object with content attribute."""

            content = "Fake content"

        output = {"messages": [FakeMsg()]}
        assert "Fake content" in ca._extract_content(output)


# ---------------------------------------------------------------------------
# _load_config error path
# ---------------------------------------------------------------------------


def test_load_config_invalid_yaml_stores_error():
    """_load_config stores error when YAML parsing fails."""
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
    from pathlib import Path
    from bili.aether.ui import chat_app as ca
    st.session_state.pop("chat_yaml_path", None)
    st.session_state.pop("chat_config", None)
    with _patch.object(ca, "load_mas_from_yaml", side_effect=ValueError("bad yaml")):
        ca._load_config(Path("/fake/config.yaml"))
    st.markdown(f"error:{st.session_state.get('chat_load_error')}")
"""
    )
    at.run()
    assert not at.exception
    assert "error:bad yaml" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _load_uploaded_config paths
# ---------------------------------------------------------------------------


def test_load_uploaded_config_caches():
    """_load_uploaded_config skips reinit for same upload name."""
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
    st.session_state.chat_yaml_path = "uploaded:test.yaml"
    st.session_state.chat_config = cfg
    ca._load_uploaded_config("test.yaml", cfg)
    st.markdown(f"cached:{st.session_state.get('chat_config') is not None}")
"""
    )
    at.run()
    assert not at.exception
    assert "cached:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_mas_diagram dispatch to other types
# ---------------------------------------------------------------------------


def test_render_mas_diagram_consensus():
    """_render_mas_diagram dispatches to consensus renderer."""
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
        AgentSpec(agent_id="v0", role="voter0", objective="Vote on item zero"),
        AgentSpec(agent_id="v1", role="voter1", objective="Vote on item one"),
    ]
    cfg = MASConfig(
        mas_id="cons", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception


def test_render_mas_diagram_custom_fallback():
    """_render_mas_diagram dispatches to fallback for custom workflow."""
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
        AgentSpec(agent_id="a0", role="a0", objective="Custom agent zero"),
    ]
    cfg = MASConfig(
        mas_id="custom", name="t", description="t",
        agents=agents, channels=[],
        workflow_type=WorkflowType.CUSTOM,
    )
    ca._render_mas_diagram(cfg)
"""
    )
    at.run()
    assert not at.exception
