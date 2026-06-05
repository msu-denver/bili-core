"""Tests for streamlit_ui.query.streamlit_query_handler."""

from unittest.mock import MagicMock, patch

from bili.streamlit_ui.query.streamlit_query_handler import (
    process_query,
    process_query_streaming,
)

_MODULE = "bili.streamlit_ui.query.streamlit_query_handler"


class TestProcessQuery:
    """process_query invokes the chain and returns the final message text."""

    def test_returns_final_message_pretty_repr(self):
        """The last message's pretty_repr is returned for a dict result."""
        final_msg = MagicMock()
        final_msg.pretty_repr.return_value = "the answer"
        chain = MagicMock()
        chain.invoke.return_value = {"messages": [MagicMock(), final_msg]}

        with patch(
            f"{_MODULE}.get_state_config",
            return_value={"configurable": {"thread_id": "u@example.com"}},
        ):
            result = process_query(chain, "What is the weather?")

        assert result == "the answer"
        # The query is wrapped as a HumanMessage in a messages list.
        sent_state = chain.invoke.call_args[0][0]
        assert sent_state["messages"][0].content == "What is the weather?"
        assert sent_state["verbose"] is False

    def test_returns_fallback_for_unexpected_result(self):
        """A non-dict result yields the no-response fallback string."""
        chain = MagicMock()
        chain.invoke.return_value = "unexpected"

        with patch(f"{_MODULE}.get_state_config", return_value={}):
            result = process_query(chain, "Hello")

        assert result == "No response or invalid format."


class TestProcessQueryStreaming:
    """process_query_streaming delegates to stream_agent with the thread id."""

    def test_yields_tokens_from_stream_agent(self):
        """Tokens from stream_agent are yielded through with the thread id."""
        chain = MagicMock()
        config = {"configurable": {"thread_id": "thread-7"}}
        with patch(f"{_MODULE}.get_state_config", return_value=config), patch(
            "bili.iris.loaders.streaming_utils.stream_agent",
            return_value=iter(["a", "b", "c"]),
        ) as mock_stream:
            tokens = list(process_query_streaming(chain, "Hi"))

        assert tokens == ["a", "b", "c"]
        mock_stream.assert_called_once_with(chain, "Hi", thread_id="thread-7")

    def test_handles_missing_thread_id(self):
        """A config without a thread id passes thread_id=None to stream_agent."""
        chain = MagicMock()
        with patch(f"{_MODULE}.get_state_config", return_value={}), patch(
            "bili.iris.loaders.streaming_utils.stream_agent",
            return_value=iter(["x"]),
        ) as mock_stream:
            tokens = list(process_query_streaming(chain, "Hi"))

        assert tokens == ["x"]
        assert mock_stream.call_args.kwargs["thread_id"] is None
