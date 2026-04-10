"""Tests for the persistence attack strategy.

Validates inject_persistence function signature, behavior with
a mocked compiled graph, and error handling.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from bili.aegis.attacks.strategies.persistence import inject_persistence


class TestInjectPersistence:
    """Tests for inject_persistence function."""

    def test_writes_poisoned_message_to_graph(self):
        """Calls update_state with a HumanMessage payload."""
        graph = MagicMock()
        graph.checkpointer = MagicMock()

        inject_persistence(graph, "thread_42", "evil payload")

        graph.update_state.assert_called_once()
        call_args = graph.update_state.call_args
        config = call_args[0][0]
        state = call_args[0][1]
        assert config["configurable"]["thread_id"] == "thread_42"
        msgs = state["messages"]
        assert len(msgs) == 1
        assert "evil payload" in msgs[0].content

    def test_message_is_human_message(self):
        """Injected message is a HumanMessage instance."""
        graph = MagicMock()
        graph.checkpointer = MagicMock()

        inject_persistence(graph, "t1", "payload")

        msg = graph.update_state.call_args[0][1]["messages"][0]
        assert isinstance(msg, HumanMessage)

    def test_message_wraps_payload_in_context(self):
        """Message content wraps payload in persisted context."""
        graph = MagicMock()
        graph.checkpointer = MagicMock()

        inject_persistence(graph, "t1", "test_payload")

        msg = graph.update_state.call_args[0][1]["messages"][0]
        assert "[Persisted context:" in msg.content
        assert "test_payload" in msg.content

    def test_raises_when_no_checkpointer(self):
        """Raises RuntimeError when graph has no checkpointer."""
        graph = MagicMock()
        graph.checkpointer = None

        with pytest.raises(RuntimeError, match="non-None checkpointer"):
            inject_persistence(graph, "t1", "payload")

    def test_accepts_empty_payload(self):
        """Does not raise with an empty payload string."""
        graph = MagicMock()
        graph.checkpointer = MagicMock()

        inject_persistence(graph, "t1", "")
        graph.update_state.assert_called_once()
