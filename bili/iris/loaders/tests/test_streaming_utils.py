"""Tests for bili.iris.loaders.streaming_utils public helpers."""

from langchain_core.messages import AIMessageChunk, HumanMessage

from bili.iris.loaders.streaming_utils import (
    _build_config,
    _build_input,
    _extract_token,
)

# ---------------------------------------------------------------------------
# _build_input (format_input_for_agent equivalent)
# ---------------------------------------------------------------------------


class TestBuildInput:
    """Verify that user prompts are wrapped into the expected state dict."""

    def test_wraps_prompt_as_human_message(self):
        """Verify prompt is wrapped as a HumanMessage in the state dict."""
        result = _build_input("Hello", None)
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hello"

    def test_default_verbose_is_false(self):
        """Verify verbose defaults to False."""
        result = _build_input("Hello", None)
        assert result["verbose"] is False

    def test_overrides_are_merged(self):
        """Verify override dict values are merged into the result."""
        result = _build_input("Hi", {"verbose": True, "extra_key": 42})
        assert result["verbose"] is True
        assert result["extra_key"] == 42

    def test_overrides_none_leaves_defaults(self):
        """Verify None overrides leave only default keys."""
        result = _build_input("test", None)
        assert set(result.keys()) == {"messages", "verbose"}


# ---------------------------------------------------------------------------
# _build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """Verify config dict construction."""

    def test_with_thread_id(self):
        """Verify config includes thread_id when provided."""
        config = _build_config("abc-123")
        assert config == {"configurable": {"thread_id": "abc-123"}}

    def test_without_thread_id(self):
        """Verify empty config is returned for None thread_id."""
        assert not _build_config(None)

    def test_empty_string_thread_id(self):
        """Verify empty string thread_id is treated as falsy."""
        assert not _build_config("")


# ---------------------------------------------------------------------------
# _extract_token
# ---------------------------------------------------------------------------


class TestExtractToken:
    """Verify token extraction from stream chunks."""

    def test_extracts_from_ai_message_chunk_tuple(self):
        """Verify token extraction from an AIMessageChunk tuple."""
        chunk = AIMessageChunk(content="hello")
        token = _extract_token((chunk,))
        assert token == "hello"

    def test_returns_none_for_non_ai_chunk_tuple(self):
        """Verify None is returned for non-AI message tuples."""
        non_ai = HumanMessage(content="hello")
        assert _extract_token((non_ai,)) is None

    def test_returns_none_for_empty_content_tuple(self):
        """Verify None is returned when chunk content is empty."""
        chunk = AIMessageChunk(content="")
        assert _extract_token((chunk,)) is None

    def test_extracts_from_object_with_content_attr(self):
        """Verify extraction from an object with a content attribute."""
        obj = type("FakeChunk", (), {"content": "world"})()
        assert _extract_token(obj) == "world"

    def test_returns_none_for_none_input(self):
        """Verify None input returns None."""
        assert _extract_token(None) is None

    def test_returns_none_for_empty_string(self):
        """Verify empty string input returns None."""
        assert _extract_token("") is None
