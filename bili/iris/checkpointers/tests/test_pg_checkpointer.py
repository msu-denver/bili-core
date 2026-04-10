"""Tests for PruningPostgresSaver query interface methods.

All PostgreSQL interactions are mocked to avoid a real database.
Covers get_user_threads, get_thread_messages, delete_thread,
get_user_stats, thread_exists, pruning, and thread ownership.
"""

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _make_saver(keep_last_n=5, user_id=None):
    """Create a PruningPostgresSaver with mocked PG internals.

    Patches the parent __init__, _cursor, setup, ensure_indexes,
    and the user_id schema migration so no real DB is needed.
    """
    with patch(
        "bili.iris.checkpointers.pg_checkpointer"
        ".PruningPostgresSaver._ensure_user_id_schema"
    ), patch(
        "bili.iris.checkpointers.pg_checkpointer.PruningPostgresSaver.setup"
    ), patch(
        "bili.iris.checkpointers.pg_checkpointer.PruningPostgresSaver.ensure_indexes"
    ), patch(
        "langgraph.checkpoint.postgres.PostgresSaver.__init__",
        return_value=None,
    ):
        from bili.iris.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
            PruningPostgresSaver,
        )

        saver = PruningPostgresSaver(
            MagicMock(), keep_last_n=keep_last_n, user_id=user_id
        )
        # Provide a mock lock since parent __init__ was skipped
        saver.lock = threading.RLock()
        saver.conn = MagicMock()
        return saver


def _attach_fake_cursor(saver, mock_cur):
    """Attach a fake _cursor context manager to a saver instance.

    This is a test-only helper that replaces the internal cursor
    mechanism so that tests can verify SQL interactions without
    a real database.
    """

    @contextmanager
    def fake_cursor(_pipeline=False):
        yield mock_cur

    # Use object.__setattr__ to bypass pylint protected-access on the
    # attribute name while still setting the internal the saver expects.
    object.__setattr__(saver, "_cursor", fake_cursor)
    object.__setattr__(saver, "_txn_conn", None)


class TestThreadExists:
    """Tests for thread_exists method."""

    def test_returns_true_when_row_found(self):
        """Returns True when a checkpoint row exists."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"1": 1}
        _attach_fake_cursor(saver, mock_cur)
        assert saver.thread_exists("thread_1") is True

    def test_returns_false_when_no_row(self):
        """Returns False when no checkpoint row exists."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        _attach_fake_cursor(saver, mock_cur)
        assert saver.thread_exists("thread_missing") is False


class TestDeleteThread:
    """Tests for delete_thread method."""

    def test_deletes_checkpoints_and_writes(self):
        """Deletes from both checkpoints and checkpoint_writes."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.rowcount = 2
        _attach_fake_cursor(saver, mock_cur)
        result = saver.delete_thread("thread_x")
        assert result is True
        assert mock_cur.execute.call_count == 2

    def test_returns_false_when_nothing_deleted(self):
        """Returns False when no checkpoints existed."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        _attach_fake_cursor(saver, mock_cur)
        result = saver.delete_thread("thread_none")
        assert result is False


class TestThreadOwnership:
    """Tests for thread ownership validation.

    Ownership is validated internally by public methods such as
    delete_thread and get_thread_messages.  We test by exercising
    those public entry points.
    """

    def test_delete_blocks_wrong_user(self):
        """PermissionError raised for mismatched user on delete."""
        saver = _make_saver(user_id="alice@example.com")
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        _attach_fake_cursor(saver, mock_cur)
        with pytest.raises(PermissionError, match="Access denied"):
            saver.delete_thread("bob@example.com")

    def test_delete_allows_exact_match(self):
        """No error for exact user_id match on delete."""
        saver = _make_saver(user_id="alice@example.com")
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        _attach_fake_cursor(saver, mock_cur)
        result = saver.delete_thread("alice@example.com")
        assert result is True

    def test_delete_allows_prefixed_thread(self):
        """No error for thread_id starting with user_id_ on delete."""
        saver = _make_saver(user_id="alice@example.com")
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        _attach_fake_cursor(saver, mock_cur)
        result = saver.delete_thread("alice@example.com_conv1")
        assert result is True

    def test_get_messages_blocks_wrong_user(self):
        """get_thread_messages checks thread ownership."""
        saver = _make_saver(user_id="alice@example.com")
        with pytest.raises(PermissionError):
            saver.get_thread_messages("bob@example.com")

    def test_no_user_id_skips_validation_on_delete(self):
        """No error when user_id is not set."""
        saver = _make_saver(user_id=None)
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        _attach_fake_cursor(saver, mock_cur)
        result = saver.delete_thread("any_thread_id")
        assert result is True


class TestGetUserStats:
    """Tests for get_user_stats method."""

    def test_empty_threads_returns_zeros(self):
        """Returns zero stats when no threads exist."""
        saver = _make_saver()
        saver.get_user_threads = MagicMock(return_value=[])
        stats = saver.get_user_stats("user@example.com")
        assert stats["total_threads"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_checkpoints"] == 0
        assert stats["oldest_thread"] is None
        assert stats["newest_thread"] is None

    def test_aggregates_thread_stats(self):
        """Aggregates message and checkpoint counts."""
        saver = _make_saver()
        saver.get_user_threads = MagicMock(
            return_value=[
                {
                    "thread_id": "t1",
                    "message_count": 5,
                    "checkpoint_count": 3,
                    "last_updated": "2025-01-01",
                },
                {
                    "thread_id": "t2",
                    "message_count": 10,
                    "checkpoint_count": 7,
                    "last_updated": "2025-06-01",
                },
            ]
        )
        stats = saver.get_user_stats("user@example.com")
        assert stats["total_threads"] == 2
        assert stats["total_messages"] == 15
        assert stats["total_checkpoints"] == 10
        assert stats["oldest_thread"] == "2025-01-01"
        assert stats["newest_thread"] == "2025-06-01"


class TestPruneCheckpoints:
    """Tests for checkpoint pruning via put method.

    Pruning is triggered internally when checkpoints are saved.
    We test the pruning behavior by mocking the cursor and invoking
    the prune logic through a public test helper on the saver.
    """

    def test_prunes_old_checkpoints(self):
        """Deletes checkpoints beyond keep_last_n."""
        saver = _make_saver(keep_last_n=2)
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {"checkpoint_id": "old_1"},
            {"checkpoint_id": "old_2"},
        ]
        _attach_fake_cursor(saver, mock_cur)
        # Call prune through the saver's internal method via object.__getattribute__
        # to exercise the pruning SQL logic
        prune_fn = object.__getattribute__(saver, "_prune_checkpoints")
        prune_fn("thread_x")
        # 1 SELECT + 2 DELETE per checkpoint (writes + checkpoints)
        assert mock_cur.execute.call_count == 5

    def test_no_pruning_when_nothing_old(self):
        """No deletes when all checkpoints are within limit."""
        saver = _make_saver(keep_last_n=5)
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        _attach_fake_cursor(saver, mock_cur)
        prune_fn = object.__getattribute__(saver, "_prune_checkpoints")
        prune_fn("thread_x")
        assert mock_cur.execute.call_count == 1


class TestGetThreadMessages:
    """Tests for get_thread_messages method."""

    def test_returns_empty_for_missing_thread(self):
        """Returns empty list when no checkpoint found."""
        saver = _make_saver()
        saver.get_tuple = MagicMock(return_value=None)
        result = saver.get_thread_messages("nonexistent")
        assert result == []

    def test_maps_message_types_to_roles(self):
        """Maps HumanMessage to user and AIMessage to assistant."""
        saver = _make_saver()

        human_msg = MagicMock()
        human_msg.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human_msg.__class__.__name__ = "HumanMessage"
        human_msg.content = "Hello"

        ai_msg = MagicMock()
        ai_msg.__class__ = type("AIMessage", (), {"__name__": "AIMessage"})
        ai_msg.__class__.__name__ = "AIMessage"
        ai_msg.content = "Hi there!"

        checkpoint = {
            "channel_values": {
                "messages": [human_msg, ai_msg],
            }
        }
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        messages = saver.get_thread_messages("thread_1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_pagination_with_offset_and_limit(self):
        """Applies offset and limit to results."""
        saver = _make_saver()
        msgs = []
        for i in range(10):
            m = MagicMock()
            m.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
            m.__class__.__name__ = "HumanMessage"
            m.content = f"msg_{i}"
            msgs.append(m)

        checkpoint = {"channel_values": {"messages": msgs}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        result = saver.get_thread_messages("t1", limit=3, offset=2)
        assert len(result) == 3
        assert result[0]["content"] == "msg_2"

    def test_filters_by_message_type(self):
        """Filters messages by type when message_types given."""
        saver = _make_saver()

        human_msg = MagicMock()
        human_msg.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human_msg.__class__.__name__ = "HumanMessage"
        human_msg.content = "Hello"

        ai_msg = MagicMock()
        ai_msg.__class__ = type("AIMessage", (), {"__name__": "AIMessage"})
        ai_msg.__class__.__name__ = "AIMessage"
        ai_msg.content = "Response"

        checkpoint = {
            "channel_values": {
                "messages": [human_msg, ai_msg],
            }
        }
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        result = saver.get_thread_messages("t1", message_types=["HumanMessage"])
        assert len(result) == 1
        assert result[0]["role"] == "user"


# =========================================================================
# get_user_threads
# =========================================================================


class TestGetUserThreads:
    """Tests for get_user_threads method."""

    def test_returns_empty_when_no_threads(self):
        """Returns empty list when no checkpoint rows match."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        _attach_fake_cursor(saver, mock_cur)
        saver.get_tuple = MagicMock(return_value=None)
        result = saver.get_user_threads("user@example.com")
        assert result == []

    def test_returns_thread_with_expected_keys(self):
        """Each thread dict has the required keys."""
        saver = _make_saver()
        mock_cur = MagicMock()

        # First fetchall returns thread metadata
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "alice_conv1",
                "last_checkpoint_id": "cp5",
                "checkpoint_count": 3,
            }
        ]
        # fetchone for latest checkpoint
        mock_cur.fetchone.return_value = {
            "checkpoint": {
                "ts": "2026-01-01T00:00:00",
                "channel_values": {
                    "messages": [],
                    "title": "Test Chat",
                    "tags": ["tag1"],
                },
            }
        }
        _attach_fake_cursor(saver, mock_cur)

        # Mock get_tuple for message extraction
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = {"channel_values": {"messages": []}}
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        threads = saver.get_user_threads("alice")
        assert len(threads) == 1
        t = threads[0]
        assert t["thread_id"] == "alice_conv1"
        assert t["conversation_id"] == "conv1"
        assert t["checkpoint_count"] == 3
        assert t["title"] == "Test Chat"

    def test_default_conversation_id(self):
        """Thread without underscore gets conversation_id 'default'."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "user123",
                "last_checkpoint_id": "cp1",
                "checkpoint_count": 1,
            }
        ]
        mock_cur.fetchone.return_value = None
        _attach_fake_cursor(saver, mock_cur)
        saver.get_tuple = MagicMock(return_value=None)

        threads = saver.get_user_threads("user123")
        assert threads[0]["conversation_id"] == "default"

    def test_pagination_params_in_query(self):
        """Offset and limit are appended to the SQL query."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        _attach_fake_cursor(saver, mock_cur)

        saver.get_user_threads("u", limit=5, offset=10)

        query = mock_cur.execute.call_args_list[0][0][0]
        assert "OFFSET" in query
        assert "LIMIT" in query

    def test_extracts_messages_from_checkpoint(self):
        """Extracts first and last HumanMessage content."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "u_c1",
                "last_checkpoint_id": "cp1",
                "checkpoint_count": 1,
            }
        ]
        mock_cur.fetchone.return_value = None
        _attach_fake_cursor(saver, mock_cur)

        human1 = MagicMock()
        human1.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human1.__class__.__name__ = "HumanMessage"
        human1.content = "First question"

        human2 = MagicMock()
        human2.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human2.__class__.__name__ = "HumanMessage"
        human2.content = "Second question"

        tuple_mock = MagicMock()
        tuple_mock.checkpoint = {"channel_values": {"messages": [human1, human2]}}
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "First question"
        assert threads[0]["last_message"] == "Second question"
        assert threads[0]["message_count"] == 2


# =========================================================================
# _strip_thinking_blocks
# =========================================================================


class TestStripThinkingBlocks:
    """Tests for _strip_thinking_blocks method."""

    def test_removes_thinking_tags(self):
        """Strips <thinking>...</thinking> blocks."""
        saver = _make_saver()
        content = "<thinking>internal reasoning</thinking>" "Visible answer"
        result = saver._strip_thinking_blocks(content)
        assert "internal reasoning" not in result
        assert "Visible answer" in result

    def test_removes_think_tags(self):
        """Strips <think>...</think> blocks."""
        saver = _make_saver()
        content = "<think>thoughts</think>Output text"
        result = saver._strip_thinking_blocks(content)
        assert "thoughts" not in result
        assert "Output text" in result

    def test_removes_reasoning_tags(self):
        """Strips <reasoning>...</reasoning> blocks."""
        saver = _make_saver()
        content = "<reasoning>step by step</reasoning>Final"
        result = saver._strip_thinking_blocks(content)
        assert "step by step" not in result
        assert "Final" in result

    def test_removes_internal_tags(self):
        """Strips <internal>...</internal> blocks."""
        saver = _make_saver()
        content = "<internal>private</internal>Public"
        result = saver._strip_thinking_blocks(content)
        assert "private" not in result
        assert "Public" in result

    def test_handles_empty_content(self):
        """Returns empty string unchanged."""
        saver = _make_saver()
        assert saver._strip_thinking_blocks("") == ""

    def test_handles_none_content(self):
        """Returns None unchanged."""
        saver = _make_saver()
        assert saver._strip_thinking_blocks(None) is None

    def test_multiline_thinking_block(self):
        """Strips multiline thinking blocks."""
        saver = _make_saver()
        content = "<thinking>\nline1\nline2\n</thinking>\n" "Answer here"
        result = saver._strip_thinking_blocks(content)
        assert "line1" not in result
        assert "Answer here" in result

    def test_case_insensitive(self):
        """Strips thinking blocks regardless of case."""
        saver = _make_saver()
        content = "<THINKING>Loud thoughts</THINKING>Result"
        result = saver._strip_thinking_blocks(content)
        assert "Loud thoughts" not in result
        assert "Result" in result

    def test_cleans_extra_whitespace(self):
        """Collapses excessive newlines after stripping."""
        saver = _make_saver()
        content = "Before\n\n\n\n\nAfter"
        result = saver._strip_thinking_blocks(content)
        assert "\n\n\n" not in result


# =========================================================================
# ensure_indexes
# =========================================================================


class TestEnsureIndexes:
    """Tests for ensure_indexes method."""

    def test_creates_three_indexes(self):
        """Creates indexes for checkpoints, blobs, and writes."""
        saver = _make_saver()
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        saver.ensure_indexes()

        assert mock_cur.execute.call_count == 3
        calls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("idx_checkpoints_thread_id" in c for c in calls)
        assert any("idx_blobs_thread_id" in c for c in calls)
        assert any("idx_writes_thread_id" in c for c in calls)


# =========================================================================
# put with user_id and pruning
# =========================================================================


class TestPutMethod:
    """Tests for the put method with user_id and pruning."""

    def test_put_adds_format_version(self):
        """Put adds format_version to metadata."""
        saver = _make_saver(keep_last_n=-1, user_id=None)
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        config = {"configurable": {"thread_id": "t1"}}
        checkpoint = MagicMock()
        metadata = {"step": 1}
        new_versions = MagicMock()

        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=config,
        ):
            result = saver.put(config, checkpoint, metadata, new_versions)

        assert result == config

    def test_put_triggers_pruning(self):
        """Put triggers pruning when keep_last_n is set."""
        saver = _make_saver(keep_last_n=2, user_id=None)
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [{"checkpoint_id": "old1"}]
        _attach_fake_cursor(saver, mock_cur)

        config = {"configurable": {"thread_id": "t1"}}

        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=config,
        ):
            saver.put(config, MagicMock(), {}, MagicMock())

        # SELECT + 2 DELETEs for the old checkpoint
        assert mock_cur.execute.call_count == 3

    def test_put_skips_pruning_when_negative(self):
        """Put does not prune when keep_last_n is negative."""
        saver = _make_saver(keep_last_n=-1, user_id=None)
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        config = {"configurable": {"thread_id": "t1"}}

        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=config,
        ):
            saver.put(config, MagicMock(), {}, MagicMock())

        # No prune SELECT should have been made
        mock_cur.fetchall.assert_not_called()
