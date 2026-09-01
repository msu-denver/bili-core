"""Tests for bili.iris.loaders.streaming_utils public helpers."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from bili.iris.loaders.streaming_utils import (
    _build_config,
    _build_input,
    _extract_token,
    astream_agent,
    invoke_agent,
    stream_agent,
)
from bili.iris.multimodal import MultimodalContentError

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


# ---------------------------------------------------------------------------
# stream_agent
# ---------------------------------------------------------------------------


class TestStreamAgent:
    """Verify stream_agent with a mocked compiled graph."""

    def test_yields_tokens_from_stream(self):
        """Verify tokens are yielded from agent.stream()."""
        chunk1 = (AIMessageChunk(content="Hello"),)
        chunk2 = (AIMessageChunk(content=" world"),)
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [chunk1, chunk2]

        tokens = list(stream_agent(mock_agent, "Hi", thread_id="t1"))
        assert tokens == ["Hello", " world"]

    def test_passes_config_with_thread_id(self):
        """Verify thread_id is passed in config."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        list(stream_agent(mock_agent, "Hi", thread_id="abc"))
        call_kwargs = mock_agent.stream.call_args[1]
        assert call_kwargs["config"] == {"configurable": {"thread_id": "abc"}}

    def test_passes_empty_config_without_thread_id(self):
        """Verify empty config when thread_id is None."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        list(stream_agent(mock_agent, "Hi"))
        call_kwargs = mock_agent.stream.call_args[1]
        assert call_kwargs["config"] == {}

    def test_uses_messages_stream_mode(self):
        """Verify stream_mode='messages' is used."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        list(stream_agent(mock_agent, "Hi"))
        call_kwargs = mock_agent.stream.call_args[1]
        assert call_kwargs["stream_mode"] == "messages"

    def test_skips_non_ai_chunks(self):
        """Verify non-AI message chunks are skipped."""
        human_chunk = (HumanMessage(content="user msg"),)
        ai_chunk = (AIMessageChunk(content="response"),)
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            human_chunk,
            ai_chunk,
        ]

        tokens = list(stream_agent(mock_agent, "Hi"))
        assert tokens == ["response"]

    def test_handles_stream_exception(self):
        """Verify error token is yielded on exception."""
        mock_agent = MagicMock()
        mock_agent.stream.side_effect = RuntimeError("boom")

        tokens = list(stream_agent(mock_agent, "Hi"))
        assert len(tokens) == 1
        assert "[Error:" in tokens[0]
        assert "RuntimeError" in tokens[0]

    def test_input_overrides_merged(self):
        """Verify input_overrides are merged into state."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        list(
            stream_agent(
                mock_agent,
                "Hi",
                input_overrides={"verbose": True},
            )
        )
        input_state = mock_agent.stream.call_args[0][0]
        assert input_state["verbose"] is True


# ---------------------------------------------------------------------------
# invoke_agent
# ---------------------------------------------------------------------------


class TestInvokeAgent:
    """Verify invoke_agent with a mocked compiled graph."""

    def test_returns_ai_message_content(self):
        """Verify content is extracted from final AI message."""
        final_msg = MagicMock()
        final_msg.content = "The answer is 42"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="What?"),
                final_msg,
            ]
        }

        result = invoke_agent(mock_agent, "What?", thread_id="t1")
        assert result == "The answer is 42"

    def test_passes_config_with_thread_id(self):
        """Verify thread_id is forwarded to invoke config."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}
        invoke_agent(mock_agent, "Hi", thread_id="xyz")
        call_kwargs = mock_agent.invoke.call_args[1]
        assert call_kwargs["config"] == {"configurable": {"thread_id": "xyz"}}

    def test_returns_fallback_for_invalid_format(self):
        """Verify fallback string for unexpected result format."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = "unexpected string"

        result = invoke_agent(mock_agent, "Hi")
        assert "No response" in result

    def test_returns_error_on_exception(self):
        """Verify error string is returned on exception."""
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = ValueError("bad input")

        result = invoke_agent(mock_agent, "Hi")
        assert "[Error:" in result
        assert "ValueError" in result

    def test_input_overrides_merged(self):
        """Verify input_overrides are merged into invoke state."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}
        invoke_agent(
            mock_agent,
            "Hi",
            input_overrides={"extra": "val"},
        )
        input_state = mock_agent.invoke.call_args[0][0]
        assert input_state["extra"] == "val"

    def test_returns_content_from_dict_result(self):
        """Verify content extraction from dict with messages."""
        final_msg = MagicMock()
        final_msg.content = "result text"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [final_msg]}

        result = invoke_agent(mock_agent, "query")
        assert result == "result text"


class TestAstreamAgent:
    """Async streaming: astream_agent yields tokens and degrades on error."""

    @pytest.mark.anyio
    async def test_yields_content_from_chat_model_stream_events(self):
        """Tokens are yielded from on_chat_model_stream events with content."""
        chunk = MagicMock()
        chunk.content = "hello"
        empty_chunk = MagicMock()
        empty_chunk.content = ""

        async def fake_events(*_args, **_kwargs):
            # A content chunk is yielded; an empty one and a non-stream event
            # are both skipped.
            yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": empty_chunk}}
            yield {"event": "on_chain_start", "data": {}}

        agent = MagicMock()
        agent.astream_events = fake_events

        tokens = [tok async for tok in astream_agent(agent, "hi")]
        assert tokens == ["hello"]

    @pytest.mark.anyio
    async def test_yields_error_message_when_stream_raises(self):
        """A failure during streaming yields a single error token, not a crash."""

        async def boom_events(*_args, **_kwargs):
            raise RuntimeError("stream boom")
            yield  # pragma: no cover  (makes this an async generator)

        agent = MagicMock()
        agent.astream_events = boom_events

        tokens = [tok async for tok in astream_agent(agent, "hi")]
        assert len(tokens) == 1
        assert "Async streaming response failed" in tokens[0]
        assert "RuntimeError" in tokens[0]


# ---------------------------------------------------------------------------
# Multimodal input at the IRIS entry points
# ---------------------------------------------------------------------------


class TestMultimodalPrompt:
    """A prompt may be a list of content parts, so an image reaches the model.

    The string path is pinned alongside every list case: widening the accepted
    type is only backwards compatible if a ``str`` still produces exactly the
    message it always did.
    """

    IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://x.invalid/i.png"}}
    PARTS = [{"type": "text", "text": "What is this?"}, IMAGE_PART]

    def test_build_input_carries_content_parts(self):
        """The parts reach the HumanMessage intact: no join, no str()."""
        result = _build_input(self.PARTS, None)
        assert result["messages"][0].content == self.PARTS

    def test_build_input_copies_rather_than_aliasing_the_caller_list(self):
        """A caller reusing its parts list must not see it mutated later."""
        parts = list(self.PARTS)
        result = _build_input(parts, None)
        assert result["messages"][0].content is not parts

    def test_build_input_accepts_a_tuple_of_parts(self):
        """Build input accepts a tuple of parts."""
        result = _build_input(tuple(self.PARTS), None)
        assert result["messages"][0].content == self.PARTS

    def test_build_input_string_path_is_unchanged(self):
        """The backwards-compatibility claim, stated as an equality against a
        message built the historical way."""
        result = _build_input("Hello", None)
        assert result["messages"][0].content == HumanMessage(content="Hello").content

    def test_build_input_rejects_an_unsupported_prompt_type(self):
        """Build input rejects an unsupported prompt type."""
        with pytest.raises(MultimodalContentError):
            _build_input(42, None)

    def test_invoke_agent_passes_parts_through_to_the_agent(self):
        """Invoke agent passes parts through to the agent."""
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [MagicMock(content="an answer")]}

        assert invoke_agent(agent, self.PARTS) == "an answer"

        state = agent.invoke.call_args.args[0]
        assert state["messages"][0].content == self.PARTS

    def test_stream_agent_passes_parts_through_to_the_agent(self):
        """Stream agent passes parts through to the agent."""
        agent = MagicMock()
        agent.stream.return_value = iter(
            [(AIMessageChunk(content="tok"), {"langgraph_node": "n"})]
        )

        assert list(stream_agent(agent, self.PARTS)) == ["tok"]

        state = agent.stream.call_args.args[0]
        assert state["messages"][0].content == self.PARTS

    @pytest.mark.anyio
    async def test_astream_agent_passes_parts_through_to_the_agent(self):
        """Astream agent passes parts through to the agent."""
        captured = {}

        async def fake_events(state, *_args, **_kwargs):
            captured["state"] = state
            chunk = MagicMock()
            chunk.content = "tok"
            yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}

        agent = MagicMock()
        agent.astream_events = fake_events

        tokens = [tok async for tok in astream_agent(agent, self.PARTS)]
        assert tokens == ["tok"]
        assert captured["state"]["messages"][0].content == self.PARTS
