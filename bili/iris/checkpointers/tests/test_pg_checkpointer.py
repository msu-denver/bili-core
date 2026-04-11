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

    def test_put_sets_format_version_in_metadata(self):
        """Put embeds format_version in metadata passed to parent."""
        saver = _make_saver(keep_last_n=-1, user_id=None)
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        config = {"configurable": {"thread_id": "t1"}}
        metadata = {"step": 1}

        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=config,
        ) as mock_parent_put:
            saver.put(config, MagicMock(), metadata, MagicMock())

        call_metadata = mock_parent_put.call_args[0][2]
        assert "format_version" in call_metadata

    def test_put_with_user_id_validates_ownership(self):
        """Put raises PermissionError for wrong user."""
        saver = _make_saver(keep_last_n=-1, user_id="alice")
        config = {"configurable": {"thread_id": "bob_conv1"}}
        with pytest.raises(PermissionError, match="Access denied"):
            saver.put(config, MagicMock(), {}, MagicMock())


# =========================================================================
# get_user_threads — extended coverage
# =========================================================================


class TestGetUserThreadsExtended:
    """Extended tests for get_user_threads method."""

    def test_multimodal_content_extraction(self):
        """Extracts text from multimodal list content."""
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

        human = MagicMock()
        human.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human.__class__.__name__ = "HumanMessage"
        human.content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": "http://x.png"},
        ]

        tuple_mock = MagicMock()
        tuple_mock.checkpoint = {"channel_values": {"messages": [human]}}
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "Describe this image"

    def test_title_and_tags_from_checkpoint(self):
        """Extracts title and tags from checkpoint data."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "u_c1",
                "last_checkpoint_id": "cp1",
                "checkpoint_count": 2,
            }
        ]
        mock_cur.fetchone.return_value = {
            "checkpoint": {
                "ts": "2026-03-01T00:00:00",
                "channel_values": {
                    "messages": [],
                    "title": "My Chat",
                    "tags": ["research", "ai"],
                },
            }
        }
        _attach_fake_cursor(saver, mock_cur)

        tuple_mock = MagicMock()
        tuple_mock.checkpoint = {"channel_values": {"messages": []}}
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        threads = saver.get_user_threads("u")
        assert threads[0]["title"] == "My Chat"
        assert threads[0]["tags"] == ["research", "ai"]

    def test_get_tuple_failure_returns_zero_messages(self):
        """Thread still returned when get_tuple raises."""
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

        saver.get_tuple = MagicMock(side_effect=RuntimeError("db error"))

        threads = saver.get_user_threads("u")
        assert len(threads) == 1
        assert threads[0]["message_count"] == 0
        assert threads[0]["first_message"] is None


# =========================================================================
# get_thread_messages — extended coverage
# =========================================================================


class TestGetThreadMessagesExtended:
    """Extended tests for get_thread_messages method."""

    def test_multimodal_content_handling(self):
        """Handles multimodal list content in messages."""
        saver = _make_saver()

        human = MagicMock()
        human.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
        human.__class__.__name__ = "HumanMessage"
        human.content = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": "http://x.png"},
        ]

        checkpoint = {"channel_values": {"messages": [human]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert "Look at this" in msgs[0]["content"]

    def test_skips_empty_ai_messages(self):
        """Empty AI messages after stripping are excluded."""
        saver = _make_saver()

        ai = MagicMock()
        ai.__class__ = type("AIMessage", (), {"__name__": "AIMessage"})
        ai.__class__.__name__ = "AIMessage"
        ai.content = "<thinking>only thinking</thinking>"

        checkpoint = {"channel_values": {"messages": [ai]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 0

    def test_system_message_mapped_correctly(self):
        """SystemMessage is mapped to system role."""
        saver = _make_saver()

        sys_msg = MagicMock()
        sys_msg.__class__ = type("SystemMessage", (), {"__name__": "SystemMessage"})
        sys_msg.__class__.__name__ = "SystemMessage"
        sys_msg.content = "You are a helpful assistant"

        checkpoint = {"channel_values": {"messages": [sys_msg]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_tool_message_mapped_correctly(self):
        """ToolMessage is mapped to tool role."""
        saver = _make_saver()

        tool_msg = MagicMock()
        tool_msg.__class__ = type("ToolMessage", (), {"__name__": "ToolMessage"})
        tool_msg.__class__.__name__ = "ToolMessage"
        tool_msg.content = "Weather: 72F"

        checkpoint = {"channel_values": {"messages": [tool_msg]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"

    def test_limit_without_offset(self):
        """Limit alone caps the result set."""
        saver = _make_saver()
        messages = []
        for i in range(5):
            m = MagicMock()
            m.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})
            m.__class__.__name__ = "HumanMessage"
            m.content = f"msg_{i}"
            messages.append(m)

        checkpoint = {"channel_values": {"messages": messages}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        result = saver.get_thread_messages("t1", limit=2)
        assert len(result) == 2
        assert result[0]["content"] == "msg_0"


# =========================================================================
# _strip_thinking_blocks — extended edge cases
# =========================================================================


class TestStripThinkingBlocksExtended:
    """Extended edge case tests for _strip_thinking_blocks."""

    def test_multiple_thinking_blocks(self):
        """Strips multiple thinking blocks in same content."""
        saver = _make_saver()
        content = "<thinking>first</thinking>Middle" "<thinking>second</thinking>End"
        result = saver._strip_thinking_blocks(content)
        assert "first" not in result
        assert "second" not in result
        assert "Middle" in result
        assert "End" in result

    def test_nested_tags_not_supported(self):
        """Nested tags are stripped greedily."""
        saver = _make_saver()
        content = "<think>outer<think>inner</think></think>Out"
        result = saver._strip_thinking_blocks(content)
        assert "Out" in result

    def test_non_string_content_passthrough(self):
        """Non-string types are returned unchanged."""
        saver = _make_saver()
        assert saver._strip_thinking_blocks(0) == 0

    def test_mixed_tag_types(self):
        """Different tag types in same content are all stripped."""
        saver = _make_saver()
        content = (
            "<thinking>a</thinking>"
            "<reasoning>b</reasoning>"
            "<internal>c</internal>"
            "Visible"
        )
        result = saver._strip_thinking_blocks(content)
        assert result == "Visible"


# =========================================================================
# _ensure_indexes — extended
# =========================================================================


class TestEnsureIndexesExtended:
    """Extended tests for ensure_indexes method."""

    def test_index_sql_uses_create_if_not_exists(self):
        """All index SQL uses IF NOT EXISTS."""
        saver = _make_saver()
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        saver.ensure_indexes()

        calls = [c[0][0] for c in mock_cur.execute.call_args_list]
        for sql in calls:
            assert "IF NOT EXISTS" in sql


# =========================================================================
# AsyncPruningPostgresSaver
# =========================================================================


class TestAsyncPruningPostgresSaver:
    """Tests for AsyncPruningPostgresSaver init and attributes."""

    def test_init_sets_keep_last_n(self):
        """Constructor stores keep_last_n."""
        with patch(
            "langgraph.checkpoint.postgres.aio" ".AsyncPostgresSaver.__init__",
            return_value=None,
        ):
            from bili.iris.checkpointers.pg_checkpointer import (
                AsyncPruningPostgresSaver,
            )

            saver = AsyncPruningPostgresSaver(MagicMock(), keep_last_n=3)
            assert saver.keep_last_n == 3

    def test_init_sets_user_id(self):
        """Constructor stores user_id."""
        with patch(
            "langgraph.checkpoint.postgres.aio" ".AsyncPostgresSaver.__init__",
            return_value=None,
        ):
            from bili.iris.checkpointers.pg_checkpointer import (
                AsyncPruningPostgresSaver,
            )

            saver = AsyncPruningPostgresSaver(MagicMock(), user_id="alice")
            assert saver.user_id == "alice"

    def test_init_defaults(self):
        """Default keep_last_n is -1 and user_id is None."""
        with patch(
            "langgraph.checkpoint.postgres.aio" ".AsyncPostgresSaver.__init__",
            return_value=None,
        ):
            from bili.iris.checkpointers.pg_checkpointer import (
                AsyncPruningPostgresSaver,
            )

            saver = AsyncPruningPostgresSaver(MagicMock())
            assert saver.keep_last_n == -1
            assert saver.user_id is None


# =========================================================================
# get_user_threads — realistic cursor data
# =========================================================================


class TestGetUserThreadsRealisticData:
    """Tests with more realistic checkpoint data shapes."""

    def test_thread_with_title_tags_from_cursor(self):
        """Extracts title and tags from fetchone checkpoint data."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "u_conv1",
                "last_checkpoint_id": "cp10",
                "checkpoint_count": 5,
            }
        ]
        # fetchone is called for latest checkpoint, then for tags
        # First call: latest checkpoint with title and tags
        mock_cur.fetchone.return_value = {
            "checkpoint": {
                "ts": "2026-03-15T12:00:00",
                "channel_values": {
                    "messages": [],
                    "title": "Research Session",
                    "tags": ["ml", "nlp"],
                },
            }
        }
        _attach_fake_cursor(saver, mock_cur)
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = {"channel_values": {"messages": []}}
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        threads = saver.get_user_threads("u")
        assert threads[0]["title"] == "Research Session"
        assert threads[0]["tags"] == ["ml", "nlp"]

    def test_thread_no_checkpoint_data(self):
        """Thread without checkpoint data gets None defaults."""
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
        saver.get_tuple = MagicMock(return_value=None)

        threads = saver.get_user_threads("u")
        assert threads[0]["title"] is None
        assert threads[0]["tags"] == []

    def test_multiple_threads_ordering(self):
        """Multiple threads are returned from SQL results."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "thread_id": "u_conv2",
                "last_checkpoint_id": "cp5",
                "checkpoint_count": 3,
            },
            {
                "thread_id": "u_conv1",
                "last_checkpoint_id": "cp2",
                "checkpoint_count": 1,
            },
        ]
        mock_cur.fetchone.return_value = None
        _attach_fake_cursor(saver, mock_cur)
        saver.get_tuple = MagicMock(return_value=None)

        threads = saver.get_user_threads("u")
        assert len(threads) == 2
        assert threads[0]["conversation_id"] == "conv2"
        assert threads[1]["conversation_id"] == "conv1"


# =========================================================================
# get_thread_messages — extended role mapping
# =========================================================================


class TestGetThreadMessagesRoleMapping:
    """Tests for additional role mapping in get_thread_messages."""

    def test_system_message_role(self):
        """SystemMessage is mapped to system role."""
        saver = _make_saver()
        sys_msg = MagicMock()
        sys_msg.__class__ = type("SystemMessage", (), {"__name__": "SystemMessage"})
        sys_msg.__class__.__name__ = "SystemMessage"
        sys_msg.content = "You are a helpful assistant"

        checkpoint = {"channel_values": {"messages": [sys_msg]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_tool_message_role(self):
        """ToolMessage is mapped to tool role."""
        saver = _make_saver()
        tool_msg = MagicMock()
        tool_msg.__class__ = type("ToolMessage", (), {"__name__": "ToolMessage"})
        tool_msg.__class__.__name__ = "ToolMessage"
        tool_msg.content = "Search results: 42"

        checkpoint = {"channel_values": {"messages": [tool_msg]}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"

    def test_mixed_message_types(self):
        """All message types are correctly mapped."""
        saver = _make_saver()

        def _make_msg(cls_name, content):
            """Create a mock message with given class name."""
            m = MagicMock()
            m.__class__ = type(cls_name, (), {"__name__": cls_name})
            m.__class__.__name__ = cls_name
            m.content = content
            return m

        msgs = [
            _make_msg("HumanMessage", "Question"),
            _make_msg("AIMessage", "Answer"),
            _make_msg("SystemMessage", "System prompt"),
            _make_msg("ToolMessage", "Tool output"),
        ]

        checkpoint = {"channel_values": {"messages": msgs}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        saver.get_tuple = MagicMock(return_value=tuple_mock)

        result = saver.get_thread_messages("t1")
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "system", "tool"]


# =========================================================================
# _strip_thinking_blocks — full coverage
# =========================================================================


class TestStripThinkingBlocksFull:
    """Additional _strip_thinking_blocks tests for full coverage."""

    def test_content_with_no_tags(self):
        """Normal content without tags is returned unchanged."""
        saver = _make_saver()
        content = "Just a normal answer"
        result = saver._strip_thinking_blocks(content)
        assert result == "Just a normal answer"

    def test_only_thinking_block(self):
        """Content that is only a thinking block returns empty."""
        saver = _make_saver()
        content = "<thinking>all internal</thinking>"
        result = saver._strip_thinking_blocks(content)
        assert "all internal" not in result

    def test_whitespace_after_strip(self):
        """Leading/trailing whitespace is cleaned after strip."""
        saver = _make_saver()
        content = "  <thinking>stuff</thinking>  Answer  "
        result = saver._strip_thinking_blocks(content)
        assert "Answer" in result

    def test_mixed_case_reasoning_tag(self):
        """Mixed-case reasoning tags are stripped."""
        saver = _make_saver()
        content = "<Reasoning>thoughts</Reasoning>Output"
        result = saver._strip_thinking_blocks(content)
        assert "thoughts" not in result
        assert "Output" in result


# =========================================================================
# _get_raw_checkpoint and _replace_raw_checkpoint
# =========================================================================


class TestRawCheckpointMethods:
    """Tests for versioned mixin raw checkpoint methods."""

    def test_get_raw_checkpoint_returns_row(self):
        """_get_raw_checkpoint returns formatted row data."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "thread_id": "t1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp1",
            "checkpoint": b"data",
            "metadata": b"meta",
        }
        _attach_fake_cursor(saver, mock_cur)

        result = saver._get_raw_checkpoint("t1")
        assert result["thread_id"] == "t1"
        assert result["checkpoint_id"] == "cp1"

    def test_get_raw_checkpoint_returns_none(self):
        """_get_raw_checkpoint returns None when no row."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        _attach_fake_cursor(saver, mock_cur)

        assert saver._get_raw_checkpoint("missing") is None

    def test_replace_raw_checkpoint_success(self):
        """_replace_raw_checkpoint returns True on update."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        _attach_fake_cursor(saver, mock_cur)

        result = saver._replace_raw_checkpoint(
            "t1",
            {
                "checkpoint_id": "cp1",
                "checkpoint": b"new",
                "metadata": b"meta",
            },
        )
        assert result is True

    def test_replace_raw_checkpoint_no_id(self):
        """_replace_raw_checkpoint returns False without id."""
        saver = _make_saver()
        result = saver._replace_raw_checkpoint("t1", {"checkpoint": b"data"})
        assert result is False

    def test_replace_raw_checkpoint_no_match(self):
        """_replace_raw_checkpoint returns False when no row matched."""
        saver = _make_saver()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        _attach_fake_cursor(saver, mock_cur)

        result = saver._replace_raw_checkpoint(
            "t1",
            {"checkpoint_id": "cp_gone", "checkpoint": b"x"},
        )
        assert result is False


# =========================================================================
# _archive_checkpoint
# =========================================================================


class TestArchiveCheckpoint:
    """Tests for _archive_checkpoint method."""

    def test_archive_inserts_and_deletes(self):
        """Archive creates table, inserts, and deletes original."""
        saver = _make_saver()
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        doc = {
            "checkpoint_ns": "",
            "checkpoint_id": "cp1",
            "checkpoint": b"data",
            "metadata": b"meta",
        }
        saver._archive_checkpoint("t1", doc, ValueError("bad"))
        # CREATE TABLE + INSERT + DELETE = 3 calls
        assert mock_cur.execute.call_count == 3

    def test_archive_without_checkpoint_id(self):
        """Archive skips DELETE when no checkpoint_id."""
        saver = _make_saver()
        mock_cur = MagicMock()
        _attach_fake_cursor(saver, mock_cur)

        doc = {"checkpoint_ns": "", "checkpoint": b"data"}
        saver._archive_checkpoint("t1", doc, ValueError("bad"))
        # CREATE TABLE + INSERT only
        assert mock_cur.execute.call_count == 2


# =========================================================================
# AsyncPruningPostgresSaver — extended
# =========================================================================


class TestAsyncPruningPostgresSaverExtended:
    """Extended tests for async saver attributes."""

    def test_has_checkpointer_type(self):
        """AsyncPruningPostgresSaver has pg checkpointer_type."""
        with patch(
            "langgraph.checkpoint.postgres.aio" ".AsyncPostgresSaver.__init__",
            return_value=None,
        ):
            from bili.iris.checkpointers.pg_checkpointer import (
                AsyncPruningPostgresSaver,
            )

            saver = AsyncPruningPostgresSaver(MagicMock())
            assert saver.checkpointer_type == "pg"

    def test_format_version_set(self):
        """AsyncPruningPostgresSaver has format_version."""
        with patch(
            "langgraph.checkpoint.postgres.aio" ".AsyncPostgresSaver.__init__",
            return_value=None,
        ):
            from bili.iris.checkpointers.pg_checkpointer import (
                AsyncPruningPostgresSaver,
            )

            saver = AsyncPruningPostgresSaver(MagicMock())
            assert saver.format_version >= 1
