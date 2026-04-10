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

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "Select a YAML" in " ".join(m.value for m in at.info)


def test_chat_area_load_error_shows_error():
    """When a load error exists the chat area shows an error."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "Bad config" in " ".join(m.value for m in at.error)


def test_render_main_no_config_no_exception():
    """render_main() runs without exception when no config loaded."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception


def test_sidebar_missing_examples_dir_shows_error():
    """When examples_dir does not exist the sidebar shows an error."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "not found" in " ".join(m.value for m in at.error)


def test_sidebar_empty_dir_shows_warning():
    """When examples dir is empty and no uploads, a warning is shown."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "No YAML" in " ".join(m.value for m in at.warning)


def test_new_thread_creates_entry():
    """_new_thread creates a thread in session state."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "created:True" in all_md
    assert "active:True" in all_md


def test_delete_thread_removes_entry():
    """_delete_thread removes a thread from session state."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "gone:True" in " ".join(m.value for m in at.markdown)


def test_active_messages_or_empty_returns_list():
    """_active_messages_or_empty returns [] when no thread is active."""

    def _app():
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

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    assert "empty:True" in " ".join(m.value for m in at.markdown)


def test_ensure_active_thread_creates_when_missing():
    """_ensure_active_thread creates a thread if none exists."""

    def _app():
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

    at = AppTest.from_function(_app)
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
