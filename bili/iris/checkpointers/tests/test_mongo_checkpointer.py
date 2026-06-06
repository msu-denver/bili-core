"""Tests for PruningMongoDBSaver query interface and pruning logic.

All MongoDB interactions are mocked — no real database is needed.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from bili.iris.checkpointers.mongo_checkpointer import PruningMongoDBSaver

# =========================================================================
# Helpers
# =========================================================================


def _make_saver(user_id=None, keep_last_n=-1):
    """Build a PruningMongoDBSaver with fully mocked MongoDB.

    Patches MongoClient and MongoDBSaver.__init__ so no real
    connection is attempted.
    """
    with patch(
        "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.__init__",
        return_value=None,
    ):

        with patch.object(PruningMongoDBSaver, "_ensure_indexes"):
            saver = PruningMongoDBSaver.__new__(PruningMongoDBSaver)
            saver.keep_last_n = keep_last_n
            saver.user_id = user_id
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
    return saver


def _human_msg(content):
    """Return a mock HumanMessage."""
    msg = MagicMock()
    msg.__class__ = type("HumanMessage", (), {})
    msg.__class__.__name__ = "HumanMessage"
    msg.content = content
    return msg


def _ai_msg(content):
    """Return a mock AIMessage."""
    msg = MagicMock()
    msg.__class__ = type("AIMessage", (), {})
    msg.__class__.__name__ = "AIMessage"
    msg.content = content
    return msg


# =========================================================================
# get_user_threads
# =========================================================================


class TestGetUserThreads:
    """Tests for PruningMongoDBSaver.get_user_threads."""

    def test_returns_empty_list_when_no_results(self):
        """No matching threads yields an empty list."""
        saver = _make_saver()
        saver.checkpoint_collection.aggregate.return_value = []
        threads = saver.get_user_threads("user@example.com")
        assert not threads

    def test_returns_thread_dicts_with_expected_keys(self):
        """Each thread dict has required keys."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "user@example.com_conv1",
                "last_updated": ts,
                "checkpoint_count": 3,
            }
        ]
        threads = saver.get_user_threads("user@example.com")
        assert len(threads) == 1
        t = threads[0]
        assert t["thread_id"] == "user@example.com_conv1"
        assert t["conversation_id"] == "conv1"
        assert t["last_updated"] == ts
        assert t["checkpoint_count"] == 3

    def test_default_conversation_id(self):
        """Thread without underscore gets conversation_id 'default'."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "user123",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]
        threads = saver.get_user_threads("user123")
        assert threads[0]["conversation_id"] == "default"

    def test_pagination_params_appended(self):
        """Limit and offset cause $skip/$limit in the pipeline."""
        saver = _make_saver()
        saver.checkpoint_collection.aggregate.return_value = []
        saver.get_user_threads("u", limit=5, offset=10)
        pipeline = saver.checkpoint_collection.aggregate.call_args[0][0]
        stage_types = [list(s.keys())[0] for s in pipeline]
        assert "$skip" in stage_types
        assert "$limit" in stage_types


# =========================================================================
# get_thread_messages
# =========================================================================


class TestGetThreadMessages:
    """Tests for PruningMongoDBSaver.get_thread_messages."""

    def test_returns_empty_when_no_checkpoint(self):
        """Returns [] when no checkpoint document is found."""
        saver = _make_saver()
        saver.checkpoint_collection.find_one.return_value = None
        msgs = saver.get_thread_messages("thread1")
        assert msgs == []

    def test_returns_empty_when_checkpoint_missing_key(self):
        """Returns [] when document lacks 'checkpoint' key."""
        saver = _make_saver()
        saver.checkpoint_collection.find_one.return_value = {"thread_id": "t1"}
        msgs = saver.get_thread_messages("t1")
        assert msgs == []

    def test_returns_messages_from_checkpoint(self):
        """Extracts messages from a legacy dict checkpoint."""
        saver = _make_saver()
        human = _human_msg("Hello")
        ai = _ai_msg("Hi there")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [human, ai]}},
        }
        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    def test_message_type_filter(self):
        """Only messages matching message_types are returned."""
        saver = _make_saver()
        human = _human_msg("Hello")
        ai = _ai_msg("Hi")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [human, ai]}},
        }
        msgs = saver.get_thread_messages("t1", message_types=["HumanMessage"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_pagination(self):
        """Offset and limit slice the message list."""
        saver = _make_saver()
        messages = [_human_msg(f"msg{i}") for i in range(5)]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": messages}},
        }
        msgs = saver.get_thread_messages("t1", limit=2, offset=1)
        assert len(msgs) == 2

    def test_thread_ownership_validated(self):
        """User ID mismatch raises PermissionError."""
        saver = _make_saver(user_id="user_a")
        with pytest.raises(PermissionError, match="Access denied"):
            saver.get_thread_messages("user_b_conv1")


# =========================================================================
# delete_thread
# =========================================================================


class TestDeleteThread:
    """Tests for PruningMongoDBSaver.delete_thread."""

    def test_deletes_from_both_collections(self):
        """Deletes from checkpoint and writes collections."""
        saver = _make_saver()
        mock_result = MagicMock()
        mock_result.deleted_count = 2
        saver.checkpoint_collection.delete_many.return_value = mock_result
        result = saver.delete_thread("t1")
        assert result is True
        saver.checkpoint_collection.delete_many.assert_called_once_with(
            {"thread_id": "t1"}
        )
        saver.writes_collection.delete_many.assert_called_once_with({"thread_id": "t1"})

    def test_returns_false_when_nothing_deleted(self):
        """Returns False when no documents matched."""
        saver = _make_saver()
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        saver.checkpoint_collection.delete_many.return_value = mock_result
        result = saver.delete_thread("nonexistent")
        assert result is False

    def test_validates_thread_ownership(self):
        """User ID mismatch raises PermissionError."""
        saver = _make_saver(user_id="alice")
        with pytest.raises(PermissionError):
            saver.delete_thread("bob_conv1")


# =========================================================================
# get_user_stats
# =========================================================================


class TestGetUserStats:
    """Tests for PruningMongoDBSaver.get_user_stats."""

    def test_empty_stats_when_no_threads(self):
        """Returns zeroed stats when user has no threads."""
        saver = _make_saver()
        saver.checkpoint_collection.aggregate.return_value = []
        stats = saver.get_user_stats("nobody")
        assert stats["total_threads"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_checkpoints"] == 0
        assert stats["oldest_thread"] is None
        assert stats["newest_thread"] is None

    def test_aggregates_stats_from_threads(self):
        """Aggregates message and checkpoint counts across threads."""
        saver = _make_saver()
        ts1 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts1,
                "checkpoint_count": 3,
            },
            {
                "_id": "u_c2",
                "last_updated": ts2,
                "checkpoint_count": 5,
            },
        ]
        stats = saver.get_user_stats("u")
        assert stats["total_threads"] == 2
        assert stats["total_checkpoints"] == 8
        assert stats["oldest_thread"] == ts1
        assert stats["newest_thread"] == ts2


# =========================================================================
# thread_exists
# =========================================================================


class TestThreadExists:
    """Tests for PruningMongoDBSaver.thread_exists."""

    def test_returns_true_when_found(self):
        """Returns True when at least one document matches."""
        saver = _make_saver()
        saver.checkpoint_collection.count_documents.return_value = 1
        assert saver.thread_exists("t1") is True

    def test_returns_false_when_not_found(self):
        """Returns False when count is zero."""
        saver = _make_saver()
        saver.checkpoint_collection.count_documents.return_value = 0
        assert saver.thread_exists("t1") is False


# =========================================================================
# _deserialize_checkpoint_data
# =========================================================================


class TestDeserializeCheckpointData:  # pylint: disable=protected-access
    """Tests for _deserialize_checkpoint_data internal helper."""

    def test_returns_empty_dict_when_no_checkpoint(self):
        """Returns {} when checkpoint key is missing."""
        saver = _make_saver()
        result = saver._deserialize_checkpoint_data({})
        assert result == {}

    def test_returns_dict_as_is_for_legacy_format(self):
        """Legacy dict checkpoints are returned without transformation."""
        saver = _make_saver()
        data = {"channel_values": {"messages": []}}
        result = saver._deserialize_checkpoint_data({"checkpoint": data})
        assert result == data

    def test_deserializes_bytes_format(self):
        """Bytes checkpoints are decoded via serde.loads_typed."""
        saver = _make_saver()
        expected = {"channel_values": {"messages": []}}
        saver.serde.loads_typed.return_value = expected

        raw_bytes = b'{"channel_values": {"messages": []}}'
        result = saver._deserialize_checkpoint_data(
            {"checkpoint": raw_bytes, "type": "json"}
        )
        saver.serde.loads_typed.assert_called_once_with(("json", raw_bytes))
        assert result == expected


# =========================================================================
# Thread ownership validation
# =========================================================================


class TestThreadOwnershipValidation:  # pylint: disable=protected-access
    """Tests for thread ownership checking."""

    def test_no_user_id_allows_any_thread(self):
        """Without user_id, all thread access is allowed."""
        saver = _make_saver(user_id=None)
        saver._validate_thread_ownership("any_thread")

    def test_matching_user_id_allows_access(self):
        """Thread starting with user_id_ passes validation."""
        saver = _make_saver(user_id="alice")
        saver._validate_thread_ownership("alice_conv1")

    def test_exact_user_id_allows_access(self):
        """Thread exactly matching user_id passes validation."""
        saver = _make_saver(user_id="alice")
        saver._validate_thread_ownership("alice")

    def test_mismatched_user_id_raises(self):
        """Thread belonging to another user raises PermissionError."""
        saver = _make_saver(user_id="alice")
        with pytest.raises(PermissionError, match="Access denied"):
            saver._validate_thread_ownership("bob_conv1")


# =========================================================================
# put with pruning
# =========================================================================


class TestPutWithPruning:
    """Tests for PruningMongoDBSaver.put with pruning logic."""

    def test_pruning_disabled_when_keep_last_n_negative(self):
        """No pruning occurs when keep_last_n is -1."""
        saver = _make_saver(keep_last_n=-1)

        config = {"configurable": {"thread_id": "t1"}}
        checkpoint = MagicMock()
        metadata = {}
        new_versions = MagicMock()

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.put",
            return_value=config,
        ):
            result = saver.put(config, checkpoint, metadata, new_versions)

        assert result == config
        saver.checkpoint_collection.find.assert_not_called()

    def test_pruning_deletes_excess_checkpoints(self):
        """Excess checkpoints are deleted when keep_last_n is set."""
        saver = _make_saver(keep_last_n=2)

        config = {"configurable": {"thread_id": "t1"}}
        checkpoint = MagicMock()
        metadata = {}
        new_versions = MagicMock()

        docs = [
            {"checkpoint_id": "cp3"},
            {"checkpoint_id": "cp2"},
            {"checkpoint_id": "cp1"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = docs
        saver.checkpoint_collection.find.return_value = mock_cursor

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.put",
            return_value=config,
        ):
            saver.put(config, checkpoint, metadata, new_versions)

        saver.checkpoint_collection.delete_one.assert_called_once()
        saver.writes_collection.delete_many.assert_called_once()

    def test_pruning_deletes_multiple_excess(self):
        """Deletes all checkpoints beyond keep_last_n threshold."""
        saver = _make_saver(keep_last_n=1)

        config = {"configurable": {"thread_id": "t1"}}
        docs = [
            {"checkpoint_id": "cp3"},
            {"checkpoint_id": "cp2"},
            {"checkpoint_id": "cp1"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = docs
        saver.checkpoint_collection.find.return_value = mock_cursor

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.put",
            return_value=config,
        ):
            saver.put(config, MagicMock(), {}, MagicMock())

        # 2 excess checkpoints should be deleted
        assert saver.checkpoint_collection.delete_one.call_count == 2
        assert saver.writes_collection.delete_many.call_count == 2

    def test_put_no_pruning_when_under_limit(self):
        """No deletes when checkpoint count is within limit."""
        saver = _make_saver(keep_last_n=5)

        config = {"configurable": {"thread_id": "t1"}}
        docs = [
            {"checkpoint_id": "cp2"},
            {"checkpoint_id": "cp1"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = docs
        saver.checkpoint_collection.find.return_value = mock_cursor

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.put",
            return_value=config,
        ):
            saver.put(config, MagicMock(), {}, MagicMock())

        saver.checkpoint_collection.delete_one.assert_not_called()

    def test_put_sets_user_id_when_configured(self):
        """Put updates user_id field when user_id is set."""
        saver = _make_saver(user_id="alice", keep_last_n=-1)

        config = {"configurable": {"thread_id": "alice_conv1"}}

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.put",
            return_value=config,
        ):
            saver.put(config, MagicMock(), {}, MagicMock())

        saver.checkpoint_collection.update_many.assert_called_once_with(
            {"thread_id": "alice_conv1"},
            {"$set": {"user_id": "alice"}},
        )

    def test_put_adds_format_version_to_metadata(self):
        """Put embeds format_version in checkpoint metadata."""
        saver = _make_saver(keep_last_n=-1)

        config = {"configurable": {"thread_id": "t1"}}
        metadata = {"step": 1}

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.put",
            return_value=config,
        ) as mock_put:
            saver.put(config, MagicMock(), metadata, MagicMock())

        # The metadata passed to super().put should have format_version
        call_metadata = mock_put.call_args[0][2]
        assert "format_version" in call_metadata


# =========================================================================
# _ensure_indexes
# =========================================================================


class TestEnsureIndexes:
    """Tests for _ensure_indexes method."""

    def test_creates_required_indexes(self):
        """Creates checkpoint, exact-match, and writes indexes."""
        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = PruningMongoDBSaver.__new__(PruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = None
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()

            # Reset mock call counts then call _ensure_indexes
            saver.checkpoint_collection.reset_mock()
            saver.writes_collection.reset_mock()
            saver._ensure_indexes()

        # At least 2 indexes on checkpoint_collection
        assert saver.checkpoint_collection.create_index.call_count >= 2
        # At least 1 index on writes_collection
        assert saver.writes_collection.create_index.call_count >= 1

    def test_creates_user_id_index_when_configured(self):
        """Creates user_id index when user_id is set."""
        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = PruningMongoDBSaver.__new__(PruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = "alice"
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
            saver._ensure_indexes()

        # Find the user_id index creation call
        calls = saver.checkpoint_collection.create_index.call_args_list
        user_idx_calls = [c for c in calls if c[1].get("name") == "idx_user_thread"]
        assert len(user_idx_calls) == 1


# =========================================================================
# _drop_conflicting_indexes
# =========================================================================


class TestDropConflictingIndexes:
    """Tests for _drop_conflicting_indexes static method."""

    def test_drops_conflicting_index(self):
        """Drops index with same keys but different name."""
        collection = MagicMock()
        collection.index_information.return_value = {
            "_id_": {"key": [("_id", 1)]},
            "old_name": {"key": [("thread_id", 1), ("checkpoint_id", -1)]},
        }

        PruningMongoDBSaver._drop_conflicting_indexes(
            collection,
            [("thread_id", 1), ("checkpoint_id", -1)],
            "new_name",
        )

        collection.drop_index.assert_called_once_with("old_name")

    def test_skips_same_name_index(self):
        """Does not drop index with the desired name."""
        collection = MagicMock()
        collection.index_information.return_value = {
            "_id_": {"key": [("_id", 1)]},
            "desired_name": {"key": [("thread_id", 1)]},
        }

        PruningMongoDBSaver._drop_conflicting_indexes(
            collection,
            [("thread_id", 1)],
            "desired_name",
        )

        collection.drop_index.assert_not_called()

    def test_skips_different_key_pattern(self):
        """Does not drop index with different key pattern."""
        collection = MagicMock()
        collection.index_information.return_value = {
            "_id_": {"key": [("_id", 1)]},
            "other_idx": {"key": [("other_field", 1)]},
        }

        PruningMongoDBSaver._drop_conflicting_indexes(
            collection,
            [("thread_id", 1)],
            "my_idx",
        )

        collection.drop_index.assert_not_called()


# =========================================================================
# _strip_thinking_blocks (via get_thread_messages)
# =========================================================================


class TestStripThinkingBlocks:
    """Tests for _strip_thinking_blocks in mongo context."""

    def test_strips_thinking_from_ai_messages(self):
        """AI message content has thinking tags removed."""
        saver = _make_saver()
        ai = _ai_msg("<thinking>reasoning</thinking>Answer")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [ai]}},
        }

        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert "reasoning" not in msgs[0]["content"]
        assert "Answer" in msgs[0]["content"]

    def test_does_not_strip_from_human_messages(self):
        """Human message content is not processed for thinking."""
        saver = _make_saver()
        human = _human_msg("<thinking>my thoughts</thinking>Q")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [human]}},
        }

        msgs = saver.get_thread_messages("t1")
        # Human messages are not stripped
        assert "<thinking>" in msgs[0]["content"]


# =========================================================================
# Async sync query methods
# =========================================================================


class TestAsyncPruningSyncMethods:
    """Tests for sync query methods on PruningMongoDBSaver."""

    def test_thread_exists_calls_count_documents(self):
        """thread_exists uses count_documents with limit=1."""
        saver = _make_saver()
        saver.checkpoint_collection.count_documents.return_value = 1
        result = saver.thread_exists("t1")
        assert result is True
        saver.checkpoint_collection.count_documents.assert_called_once()

    def test_get_user_stats_with_threads(self):
        """get_user_stats computes stats from aggregation results."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 3, 15, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 4,
            }
        ]
        stats = saver.get_user_stats("u")
        assert stats["total_threads"] == 1
        assert stats["total_checkpoints"] == 4


# =========================================================================
# get_user_threads — extended coverage
# =========================================================================


class TestGetUserThreadsExtended:
    """Extended tests for PruningMongoDBSaver.get_user_threads."""

    def test_multiple_threads_ordered(self):
        """Multiple threads are returned from aggregation."""
        saver = _make_saver()
        ts1 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_conv2",
                "last_updated": ts2,
                "checkpoint_count": 5,
            },
            {
                "_id": "u_conv1",
                "last_updated": ts1,
                "checkpoint_count": 2,
            },
        ]
        threads = saver.get_user_threads("u")
        assert len(threads) == 2
        assert threads[0]["conversation_id"] == "conv2"
        assert threads[1]["conversation_id"] == "conv1"

    def test_thread_has_all_required_keys(self):
        """Each thread dict includes all required keys."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]
        threads = saver.get_user_threads("u")
        required_keys = [
            "thread_id",
            "conversation_id",
            "last_updated",
            "checkpoint_count",
            "message_count",
            "first_message",
            "last_message",
            "title",
            "tags",
        ]
        for key in required_keys:
            assert key in threads[0]

    def test_message_count_defaults_to_zero(self):
        """Message count defaults to 0 in aggregation."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]
        threads = saver.get_user_threads("u")
        assert threads[0]["message_count"] == 0


# =========================================================================
# get_thread_messages — extended coverage
# =========================================================================


class TestGetThreadMessagesExtended:
    """Extended tests for get_thread_messages."""

    def test_multiple_message_types(self):
        """Returns both human and AI messages with roles."""
        saver = _make_saver()
        h1 = _human_msg("Hello")
        a1 = _ai_msg("Hi")
        h2 = _human_msg("How are you?")
        a2 = _ai_msg("Good!")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [h1, a1, h2, a2]}},
        }
        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 4
        assert [m["role"] for m in msgs] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_offset_beyond_messages_returns_empty(self):
        """Offset beyond message count returns empty list."""
        saver = _make_saver()
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [_human_msg("one")]}},
        }
        msgs = saver.get_thread_messages("t1", offset=10)
        assert msgs == []

    def test_limit_larger_than_messages(self):
        """Limit larger than available returns all messages."""
        saver = _make_saver()
        messages = [_human_msg(f"m{i}") for i in range(3)]
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": messages}},
        }
        msgs = saver.get_thread_messages("t1", limit=100)
        assert len(msgs) == 3

    def test_filter_to_ai_messages_only(self):
        """Filtering to AIMessage returns only assistant msgs."""
        saver = _make_saver()
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {
                "channel_values": {
                    "messages": [
                        _human_msg("Q"),
                        _ai_msg("A"),
                    ]
                }
            },
        }
        msgs = saver.get_thread_messages("t1", message_types=["AIMessage"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_no_user_id_allows_access(self):
        """Without user_id, any thread can be accessed."""
        saver = _make_saver(user_id=None)
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "anyone_conv",
            "checkpoint": {"channel_values": {"messages": [_human_msg("Hi")]}},
        }
        msgs = saver.get_thread_messages("anyone_conv")
        assert len(msgs) == 1


# =========================================================================
# _deserialize_checkpoint_data — extended coverage
# =========================================================================


class TestDeserializeCheckpointDataExtended:
    """Extended tests for _deserialize_checkpoint_data."""

    def test_returns_empty_for_none_checkpoint_value(self):
        """Returns empty dict when checkpoint value is None."""
        saver = _make_saver()
        result = saver._deserialize_checkpoint_data({"checkpoint": None})
        assert result == {}

    def test_handles_custom_type_field(self):
        """Uses doc type field for bytes deserialization."""
        saver = _make_saver()
        expected = {"channel_values": {"messages": []}}
        saver.serde.loads_typed.return_value = expected

        raw = b"binary_data"
        result = saver._deserialize_checkpoint_data(
            {"checkpoint": raw, "type": "msgpack"}
        )
        saver.serde.loads_typed.assert_called_once_with(("msgpack", raw))
        assert result == expected


# =========================================================================
# delete_thread — extended coverage
# =========================================================================


class TestDeleteThreadExtended:
    """Extended tests for delete_thread."""

    def test_delete_allows_exact_user_match(self):
        """Exact user_id match allows deletion."""
        saver = _make_saver(user_id="alice")
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        saver.checkpoint_collection.delete_many.return_value = mock_result
        result = saver.delete_thread("alice")
        assert result is True

    def test_delete_allows_prefixed_thread(self):
        """Thread prefixed with user_id_ is allowed."""
        saver = _make_saver(user_id="alice")
        mock_result = MagicMock()
        mock_result.deleted_count = 2
        saver.checkpoint_collection.delete_many.return_value = mock_result
        result = saver.delete_thread("alice_conv1")
        assert result is True

    def test_delete_cleans_writes_collection(self):
        """Writes collection is always cleaned on delete."""
        saver = _make_saver()
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        saver.checkpoint_collection.delete_many.return_value = mock_result
        saver.delete_thread("t1")
        saver.writes_collection.delete_many.assert_called_once_with({"thread_id": "t1"})


# =========================================================================
# AsyncPruningMongoDBSaver sync query methods
# =========================================================================


class TestAsyncPruningMongoDBSaverSyncMethods:
    """Tests for AsyncPruningMongoDBSaver sync query delegation."""

    def _make_async_saver(self, user_id=None):
        """Build an AsyncPruningMongoDBSaver with mocked MongoDB."""
        from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = AsyncPruningMongoDBSaver.__new__(AsyncPruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = user_id
            saver._indexes_ensured = True
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
        return saver

    def test_get_user_threads_returns_threads(self):
        """get_user_threads aggregates and returns threads."""
        saver = self._make_async_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 2,
            }
        ]
        threads = saver.get_user_threads("u")
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "u_c1"

    def test_get_thread_messages_returns_messages(self):
        """get_thread_messages extracts messages from doc."""
        saver = self._make_async_saver()
        human = _human_msg("Hello")
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [human]}},
        }
        msgs = saver.get_thread_messages("t1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_delete_thread_returns_true(self):
        """delete_thread returns True on successful delete."""
        saver = self._make_async_saver()
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        saver.checkpoint_collection.delete_many.return_value = mock_result
        assert saver.delete_thread("t1") is True

    def test_thread_exists_returns_true(self):
        """thread_exists returns True when document found."""
        saver = self._make_async_saver()
        saver.checkpoint_collection.count_documents.return_value = 1
        assert saver.thread_exists("t1") is True

    def test_thread_exists_returns_false(self):
        """thread_exists returns False when no document found."""
        saver = self._make_async_saver()
        saver.checkpoint_collection.count_documents.return_value = 0
        assert saver.thread_exists("t1") is False

    def test_get_user_stats_empty(self):
        """get_user_stats returns zeros when no threads."""
        saver = self._make_async_saver()
        saver.checkpoint_collection.aggregate.return_value = []
        stats = saver.get_user_stats("nobody")
        assert stats["total_threads"] == 0
        assert stats["total_messages"] == 0

    def test_ownership_validated_on_messages(self):
        """get_thread_messages validates thread ownership."""
        saver = self._make_async_saver(user_id="alice")
        with pytest.raises(PermissionError):
            saver.get_thread_messages("bob_conv")

    def test_deserialize_bytes_format(self):
        """Bytes checkpoint data is deserialized via serde."""
        saver = self._make_async_saver()
        expected = {"channel_values": {"messages": []}}
        saver.serde.loads_typed.return_value = expected
        raw = b"binary"
        result = saver._deserialize_checkpoint_data({"checkpoint": raw, "type": "json"})
        assert result == expected


# =========================================================================
# PruningMongoDBSaver.get_user_threads — message extraction
# =========================================================================


class TestGetUserThreadsMessageExtraction:
    """Tests for message extraction from checkpoints."""

    def test_extracts_first_last_human_messages(self):
        """Extracts first and last HumanMessage content."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 3,
            }
        ]

        h1 = _human_msg("First question")
        h2 = _human_msg("Second question")
        a1 = _ai_msg("Answer")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": {
                "channel_values": {
                    "messages": [h1, a1, h2],
                }
            },
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "First question"
        assert threads[0]["last_message"] == "Second question"
        assert threads[0]["message_count"] == 3

    def test_extracts_title_and_tags(self):
        """Title and tags are extracted from checkpoint state."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": {
                "channel_values": {
                    "messages": [],
                    "title": "Research Chat",
                    "tags": ["nlp", "ml"],
                }
            },
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["title"] == "Research Chat"
        assert threads[0]["tags"] == ["nlp", "ml"]

    def test_multimodal_content_extraction(self):
        """Extracts text from multimodal list content."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]

        h = _human_msg("placeholder")
        h.content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": "http://x.png"},
        ]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": {"channel_values": {"messages": [h]}},
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "Describe this image"

    def test_no_checkpoint_key_defaults(self):
        """Thread defaults to empty messages when no checkpoint."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]
        saver.checkpoint_collection.find_one.return_value = None

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] is None
        assert threads[0]["message_count"] == 0

    def test_bytes_checkpoint_deserialized(self):
        """Bytes checkpoint data is deserialized correctly."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]

        saver.serde.loads_typed.return_value = {
            "channel_values": {"messages": [_human_msg("Deserialized msg")]}
        }
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": b"binary_data",
            "type": "msgpack",
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "Deserialized msg"


# =========================================================================
# get_thread_messages — multimodal and stripping
# =========================================================================


class TestGetThreadMessagesMultimodal:
    """Tests for multimodal message handling."""

    def test_multimodal_content_in_messages(self):
        """Multimodal list content is joined as text."""
        saver = _make_saver()
        h = _human_msg("placeholder")
        h.content = [
            {"type": "text", "text": "Look at"},
            {"type": "text", "text": "this"},
            {"type": "image_url", "image_url": "http://x.png"},
        ]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [h]}},
        }
        msgs = saver.get_thread_messages("t1")
        assert "Look at" in msgs[0]["content"]
        assert "this" in msgs[0]["content"]

    def test_ai_message_thinking_stripped(self):
        """AI messages have thinking blocks stripped."""
        saver = _make_saver()
        ai = _ai_msg("<thinking>internal</thinking>Visible answer")

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [ai]}},
        }
        msgs = saver.get_thread_messages("t1")
        assert "internal" not in msgs[0]["content"]
        assert "Visible answer" in msgs[0]["content"]


# =========================================================================
# AsyncPruningMongoDBSaver — remaining sync wrappers
# =========================================================================


class TestAsyncPruningSyncMethodsExtended:
    """Extended tests for async saver's sync query methods."""

    def _make_async_saver(self, user_id=None):
        """Build an AsyncPruningMongoDBSaver with mocked MongoDB."""
        from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = AsyncPruningMongoDBSaver.__new__(AsyncPruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = user_id
            saver._indexes_ensured = True
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
        return saver

    def test_delete_thread_returns_false(self):
        """delete_thread returns False when nothing deleted."""
        saver = self._make_async_saver()
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        saver.checkpoint_collection.delete_many.return_value = mock_result
        assert saver.delete_thread("nonexistent") is False

    def test_get_thread_messages_empty(self):
        """get_thread_messages returns [] with no checkpoint."""
        saver = self._make_async_saver()
        saver.checkpoint_collection.find_one.return_value = None
        msgs = saver.get_thread_messages("missing_thread")
        assert msgs == []

    def test_get_user_threads_pagination(self):
        """get_user_threads uses $skip/$limit in pipeline."""
        saver = self._make_async_saver()
        saver.checkpoint_collection.aggregate.return_value = []
        saver.get_user_threads("u", limit=10, offset=5)
        pipeline = saver.checkpoint_collection.aggregate.call_args[0][0]
        stage_types = [list(s.keys())[0] for s in pipeline]
        assert "$skip" in stage_types
        assert "$limit" in stage_types

    def test_get_user_stats_aggregates(self):
        """get_user_stats aggregates checkpoint counts."""
        saver = self._make_async_saver()
        ts = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 5,
            },
            {
                "_id": "u_c2",
                "last_updated": ts,
                "checkpoint_count": 3,
            },
        ]
        stats = saver.get_user_stats("u")
        assert stats["total_threads"] == 2
        assert stats["total_checkpoints"] == 8


# =========================================================================
# _ensure_indexes async variant
# =========================================================================


class TestAsyncEnsureIndexes:
    """Tests for AsyncPruningMongoDBSaver._ensure_indexes."""

    def test_ensure_indexes_creates_required_indexes(self):
        """_ensure_indexes creates checkpoint and writes indexes."""
        from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = AsyncPruningMongoDBSaver.__new__(AsyncPruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = None
            saver._indexes_ensured = False
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()

            saver._ensure_indexes()

        assert saver.checkpoint_collection.create_index.call_count >= 2
        assert saver.writes_collection.create_index.call_count >= 1

    def test_ensure_indexes_with_user_id(self):
        """_ensure_indexes creates user_id index when configured."""
        from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = AsyncPruningMongoDBSaver.__new__(AsyncPruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = "alice"
            saver._indexes_ensured = False
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()

            saver._ensure_indexes()

        calls = saver.checkpoint_collection.create_index.call_args_list
        user_calls = [c for c in calls if c[1].get("name") == "idx_user_thread"]
        assert len(user_calls) == 1


# =========================================================================
# _create_index_safe retry behavior
# =========================================================================


class TestCreateIndexSafe:
    """Tests for _create_index_safe retry/conflict handling."""

    def test_successful_creation(self):
        """Index is created on first attempt."""

        collection = MagicMock()
        PruningMongoDBSaver._create_index_safe(
            collection, [("thread_id", 1)], "idx_test"
        )
        collection.create_index.assert_called_once()

    def test_retries_on_build_in_progress(self):
        """Retries when another index build is in progress."""
        from pymongo.errors import OperationFailure

        collection = MagicMock()
        exc = OperationFailure("build in progress")
        exc._OperationFailure__code = 40333
        type(exc).code = property(lambda s: 40333)
        collection.create_index.side_effect = [exc, None]

        with patch("time.sleep"):
            PruningMongoDBSaver._create_index_safe(
                collection, [("thread_id", 1)], "idx_test"
            )
        assert collection.create_index.call_count == 2


# =========================================================================
# VersionedCheckpointerMixin methods on Mongo
# =========================================================================


class TestMongoVersionedMixin:
    """Tests for versioned mixin methods on PruningMongoDBSaver."""

    def test_get_raw_checkpoint_returns_doc(self):
        """_get_raw_checkpoint queries by thread_id."""
        saver = _make_saver()
        expected = {"thread_id": "t1", "checkpoint": {}}
        saver.checkpoint_collection.find_one.return_value = expected
        result = saver._get_raw_checkpoint("t1")
        assert result == expected

    def test_get_raw_checkpoint_returns_none(self):
        """_get_raw_checkpoint returns None when not found."""
        saver = _make_saver()
        saver.checkpoint_collection.find_one.return_value = None
        assert saver._get_raw_checkpoint("missing") is None

    def test_replace_raw_checkpoint_success(self):
        """_replace_raw_checkpoint returns True on match."""
        saver = _make_saver()
        mock_result = MagicMock()
        mock_result.matched_count = 1
        saver.checkpoint_collection.replace_one.return_value = mock_result
        result = saver._replace_raw_checkpoint("t1", {"_id": "abc", "checkpoint": {}})
        assert result is True

    def test_replace_raw_checkpoint_no_id(self):
        """_replace_raw_checkpoint returns False without _id."""
        saver = _make_saver()
        result = saver._replace_raw_checkpoint("t1", {"checkpoint": {}})
        assert result is False

    def test_replace_raw_checkpoint_no_match(self):
        """_replace_raw_checkpoint returns False on no match."""
        saver = _make_saver()
        mock_result = MagicMock()
        mock_result.matched_count = 0
        saver.checkpoint_collection.replace_one.return_value = mock_result
        result = saver._replace_raw_checkpoint("t1", {"_id": "abc", "checkpoint": {}})
        assert result is False


# =========================================================================
# async query method coverage
# =========================================================================


class TestAsyncPutMethod:
    """Tests for PruningMongoDBSaver.aput method."""

    def test_aput_adds_format_version(self):
        """aput embeds format_version in metadata."""
        import asyncio  # pylint: disable=import-outside-toplevel

        saver = _make_saver(keep_last_n=-1)
        config = {"configurable": {"thread_id": "t1"}}

        async def _run():
            with patch(
                "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.aput",
                return_value=config,
            ) as mock_aput:
                result = await saver.aput(config, MagicMock(), {"step": 1}, MagicMock())
            return result, mock_aput

        result, mock_aput = asyncio.run(_run())
        call_metadata = mock_aput.call_args[0][2]
        assert "format_version" in call_metadata
        assert result == config

    def test_aput_sets_user_id(self):
        """aput updates user_id when configured."""
        import asyncio  # pylint: disable=import-outside-toplevel

        saver = _make_saver(user_id="alice", keep_last_n=-1)
        config = {"configurable": {"thread_id": "alice_c1"}}

        async def _run():
            with patch(
                "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.aput",
                return_value=config,
            ):
                await saver.aput(config, MagicMock(), {}, MagicMock())

        asyncio.run(_run())
        saver.checkpoint_collection.update_many.assert_called_once()

    def test_aput_validates_ownership(self):
        """aput raises PermissionError for wrong user."""
        import asyncio  # pylint: disable=import-outside-toplevel

        saver = _make_saver(user_id="alice", keep_last_n=-1)
        config = {"configurable": {"thread_id": "bob_c1"}}

        async def _run():
            await saver.aput(config, MagicMock(), {}, MagicMock())

        with pytest.raises(PermissionError):
            asyncio.run(_run())


# =========================================================================
# _ensure_indexes — full path with user_id
# =========================================================================


class TestEnsureIndexesFullPath:
    """Tests for _ensure_indexes with all code paths."""

    def test_indexes_without_user_id(self):
        """Creates base indexes without user_id index."""
        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = PruningMongoDBSaver.__new__(PruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = None
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
            saver.checkpoint_collection.reset_mock()
            saver.writes_collection.reset_mock()
            saver._ensure_indexes()

        cp_calls = saver.checkpoint_collection.create_index.call_args_list
        user_calls = [c for c in cp_calls if c[1].get("name") == "idx_user_thread"]
        assert len(user_calls) == 0

    def test_indexes_with_user_id(self):
        """Creates user_id index when user_id is set."""
        with patch(
            "bili.iris.checkpointers.mongo_checkpointer" ".MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = PruningMongoDBSaver.__new__(PruningMongoDBSaver)
            saver.keep_last_n = -1
            saver.user_id = "alice"
            saver.checkpoint_collection = MagicMock()
            saver.writes_collection = MagicMock()
            saver.db = MagicMock()
            saver.serde = MagicMock()
            saver._ensure_indexes()

        cp_calls = saver.checkpoint_collection.create_index.call_args_list
        user_calls = [c for c in cp_calls if c[1].get("name") == "idx_user_thread"]
        assert len(user_calls) == 1


# =========================================================================
# _create_index_safe — retry and conflict handling
# =========================================================================


class TestCreateIndexSafe:
    """Tests for _create_index_safe with error handling."""

    def test_creates_index_on_first_attempt(self):
        """Index creation succeeds on first try."""
        collection = MagicMock()
        PruningMongoDBSaver._create_index_safe(
            collection, [("thread_id", 1)], "test_idx"
        )
        collection.create_index.assert_called_once()

    def test_retries_on_concurrent_build(self):
        """Retries on code 40333 (concurrent build)."""
        from pymongo.errors import (  # pylint: disable=import-outside-toplevel
            OperationFailure,
        )

        collection = MagicMock()
        exc = OperationFailure("concurrent", code=40333)
        collection.create_index.side_effect = [exc, None]

        with patch("time.sleep"):
            PruningMongoDBSaver._create_index_safe(collection, [("a", 1)], "idx")

        assert collection.create_index.call_count == 2

    def test_handles_code_85_conflict(self):
        """Handles code 85 (different options) by dropping."""
        from pymongo.errors import (  # pylint: disable=import-outside-toplevel
            OperationFailure,
        )

        collection = MagicMock()
        exc = OperationFailure("options", code=85)
        collection.create_index.side_effect = [exc, None]
        collection.index_information.return_value = {}

        with patch("time.sleep"):
            PruningMongoDBSaver._create_index_safe(collection, [("a", 1)], "idx")

        assert collection.create_index.call_count == 2

    def test_handles_code_86_name_conflict(self):
        """Handles code 86 (name conflict) by dropping name."""
        from pymongo.errors import (  # pylint: disable=import-outside-toplevel
            OperationFailure,
        )

        collection = MagicMock()
        exc = OperationFailure("name conflict", code=86)
        collection.create_index.side_effect = [exc, None]
        collection.index_information.return_value = {}

        with patch("time.sleep"):
            PruningMongoDBSaver._create_index_safe(collection, [("a", 1)], "idx")

        collection.drop_index.assert_called_with("idx")


# =========================================================================
# _archive_checkpoint
# =========================================================================


class TestArchiveCheckpoint:
    """Tests for _archive_checkpoint method."""

    def test_archives_and_removes_document(self):
        """Archives to separate collection and removes original."""
        saver = _make_saver()
        archive_coll = MagicMock()
        saver.db.__getitem__.return_value = archive_coll

        doc = {
            "_id": "abc123",
            "thread_id": "t1",
            "checkpoint": {"data": True},
        }
        saver._archive_checkpoint("t1", doc, RuntimeError("migration failed"))

        archive_coll.insert_one.assert_called_once()
        saver.checkpoint_collection.delete_one.assert_called_once()

    def test_archive_handles_insert_failure(self):
        """Logs error when archive insert fails."""
        saver = _make_saver()
        archive_coll = MagicMock()
        archive_coll.insert_one.side_effect = RuntimeError("db err")
        saver.db.__getitem__.return_value = archive_coll

        doc = {"_id": "abc", "thread_id": "t1"}
        # Should not raise
        saver._archive_checkpoint("t1", doc, RuntimeError("err"))


# =========================================================================
# get_user_threads — message extraction with multimodal
# =========================================================================


class TestGetUserThreadsMessageExtraction:
    """Tests for message extraction in get_user_threads."""

    def test_extracts_multimodal_messages(self):
        """Handles multimodal content in user threads."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]

        human = _human_msg("text content")
        human.content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "url": "http://img.png"},
        ]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": {"channel_values": {"messages": [human]}},
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["first_message"] == "Describe this image"

    def test_extracts_title_and_tags(self):
        """Extracts title and tags from channel_values."""
        saver = _make_saver()
        ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        saver.checkpoint_collection.aggregate.return_value = [
            {
                "_id": "u_c1",
                "last_updated": ts,
                "checkpoint_count": 1,
            }
        ]
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "u_c1",
            "checkpoint": {
                "channel_values": {
                    "messages": [],
                    "title": "Chat Title",
                    "tags": ["tag1"],
                }
            },
        }

        threads = saver.get_user_threads("u")
        assert threads[0]["title"] == "Chat Title"
        assert threads[0]["tags"] == ["tag1"]


# =========================================================================
# get_thread_messages — multimodal content
# =========================================================================


class TestGetThreadMessagesMultimodal:
    """Tests for multimodal content handling in messages."""

    def test_multimodal_list_content(self):
        """Extracts text from list-format multimodal content."""
        saver = _make_saver()
        msg = _human_msg("placeholder")
        msg.content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]

        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [msg]}},
        }

        msgs = saver.get_thread_messages("t1")
        assert "Part 1" in msgs[0]["content"]
        assert "Part 2" in msgs[0]["content"]


# =========================================================================
# Module-level factory functions
# =========================================================================


class TestModuleLevelFunctions:
    """Tests for get_mongo_client, close_mongo_client, get_mongo_checkpointer."""

    @patch("bili.iris.checkpointers.mongo_checkpointer.atexit.register")
    @patch("bili.iris.checkpointers.mongo_checkpointer.MongoClient")
    @patch.dict(
        "os.environ", {"MONGO_CONNECTION_STRING": "mongodb://localhost"}, clear=True
    )
    def test_get_mongo_client_returns_langgraph_db(self, mock_client_cls, mock_atexit):
        """A set connection string yields the 'langgraph' database and registers cleanup."""
        from bili.iris.checkpointers.mongo_checkpointer import (
            close_mongo_client,
            get_mongo_client,
        )

        fake_client = MagicMock()
        fake_db = MagicMock()
        fake_client.__getitem__.return_value = fake_db
        mock_client_cls.return_value = fake_client

        db = get_mongo_client()
        assert db is fake_db
        mock_client_cls.assert_called_once_with("mongodb://localhost")
        fake_client.__getitem__.assert_called_once_with("langgraph")
        mock_atexit.assert_called_once_with(close_mongo_client, fake_client)

    @patch.dict("os.environ", {}, clear=True)
    def test_get_mongo_client_returns_none_without_env(self):
        """No connection string returns None."""
        from bili.iris.checkpointers.mongo_checkpointer import get_mongo_client

        assert get_mongo_client() is None

    def test_close_mongo_client_closes_active(self):
        """An active client is closed."""
        from bili.iris.checkpointers.mongo_checkpointer import close_mongo_client

        client = MagicMock()
        close_mongo_client(client)
        client.close.assert_called_once_with()

    def test_close_mongo_client_noop_for_none(self):
        """Passing None does not raise."""
        from bili.iris.checkpointers.mongo_checkpointer import close_mongo_client

        # Should simply return without error.
        assert close_mongo_client(None) is None

    @patch("bili.iris.checkpointers.mongo_checkpointer.get_mongo_client")
    def test_get_mongo_checkpointer_returns_saver(self, mock_get_client):
        """A live db produces a PruningMongoDBSaver."""
        from bili.iris.checkpointers.mongo_checkpointer import get_mongo_checkpointer

        mock_get_client.return_value = MagicMock()
        with patch.object(PruningMongoDBSaver, "_ensure_indexes"), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = get_mongo_checkpointer(keep_last_n=3, user_id="u@example.com")
        assert isinstance(saver, PruningMongoDBSaver)
        assert saver.keep_last_n == 3
        assert saver.user_id == "u@example.com"

    @patch("bili.iris.checkpointers.mongo_checkpointer.get_mongo_client")
    def test_get_mongo_checkpointer_returns_none_without_db(self, mock_get_client):
        """No db yields None."""
        from bili.iris.checkpointers.mongo_checkpointer import get_mongo_checkpointer

        mock_get_client.return_value = None
        assert get_mongo_checkpointer() is None


# =========================================================================
# get_async_mongo_checkpointer
# =========================================================================


pytest_anyio_mark = pytest.mark.anyio


class TestGetAsyncMongoCheckpointer:
    """Tests for the async checkpointer factory."""

    pytestmark = pytest.mark.anyio

    @patch("bili.iris.checkpointers.mongo_checkpointer.get_mongo_client")
    async def test_returns_saver_when_db_available(self, mock_get_client):
        """A live db produces a PruningMongoDBSaver from the async factory."""
        from bili.iris.checkpointers.mongo_checkpointer import (
            get_async_mongo_checkpointer,
        )

        mock_get_client.return_value = MagicMock()
        with patch.object(PruningMongoDBSaver, "_ensure_indexes"), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.__init__",
            return_value=None,
        ):
            saver = await get_async_mongo_checkpointer(keep_last_n=2)
        assert isinstance(saver, PruningMongoDBSaver)
        assert saver.keep_last_n == 2

    @patch("bili.iris.checkpointers.mongo_checkpointer.get_mongo_client")
    async def test_returns_none_without_db(self, mock_get_client):
        """No db yields None from the async factory."""
        from bili.iris.checkpointers.mongo_checkpointer import (
            get_async_mongo_checkpointer,
        )

        mock_get_client.return_value = None
        assert await get_async_mongo_checkpointer() is None


# =========================================================================
# Index helper error paths
# =========================================================================


class TestIndexHelperErrorPaths:
    """Tests for _drop_conflicting_indexes and _create_index_safe edge cases."""

    def test_drop_conflicting_handles_operation_failure(self):
        """An OperationFailure while listing indexes is logged, not raised."""
        from pymongo.errors import OperationFailure

        collection = MagicMock()
        collection.name = "checkpoints"
        collection.index_information.side_effect = OperationFailure("no perms")
        # Should not raise.
        PruningMongoDBSaver._drop_conflicting_indexes(
            collection, [("thread_id", 1)], "idx_x"
        )
        collection.drop_index.assert_not_called()

    def test_create_index_safe_code86_drops_by_name_then_retries(self):
        """Code 86 conflict drops the same-named index and retries successfully."""
        from pymongo.errors import OperationFailure

        collection = MagicMock()
        collection.name = "checkpoints"
        collection.index_information.return_value = {}
        # First create_index raises code 86, second succeeds.
        collection.create_index.side_effect = [
            OperationFailure("conflict", code=86),
            None,
        ]
        PruningMongoDBSaver._create_index_safe(
            collection, [("thread_id", 1)], "idx_dup"
        )
        collection.drop_index.assert_called_once_with("idx_dup")
        assert collection.create_index.call_count == 2

    def test_create_index_safe_code86_drop_failure_swallowed(self):
        """A failure dropping the code-86 index by name is swallowed before retry."""
        from pymongo.errors import OperationFailure

        collection = MagicMock()
        collection.name = "checkpoints"
        collection.index_information.return_value = {}
        collection.drop_index.side_effect = OperationFailure("drop failed")
        collection.create_index.side_effect = [
            OperationFailure("conflict", code=86),
            None,
        ]
        PruningMongoDBSaver._create_index_safe(
            collection, [("thread_id", 1)], "idx_dup"
        )
        assert collection.create_index.call_count == 2

    def test_create_index_safe_unhandled_code_raises(self):
        """An unhandled OperationFailure code propagates."""
        from pymongo.errors import OperationFailure

        collection = MagicMock()
        collection.name = "checkpoints"
        collection.create_index.side_effect = OperationFailure("fatal", code=999)
        with pytest.raises(OperationFailure):
            PruningMongoDBSaver._create_index_safe(
                collection, [("thread_id", 1)], "idx_x"
            )


# =========================================================================
# get_tuple / aget_tuple migration orchestration (sync saver)
# =========================================================================


class TestGetTupleMigration:
    """Tests for PruningMongoDBSaver.get_tuple migration handling."""

    def test_get_tuple_returns_none_on_migration_error(self):
        """An exception during migration causes get_tuple to return None."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", side_effect=RuntimeError("boom")
        ):
            assert saver.get_tuple(config) is None

    def test_get_tuple_returns_none_when_still_needs_migration(self):
        """A checkpoint still needing migration after the attempt returns None."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=True
        ), patch.object(
            saver, "_get_raw_checkpoint", return_value={"type": "msgpack"}
        ), patch.object(
            saver, "_needs_migration", return_value=True
        ):
            assert saver.get_tuple(config) is None

    def test_get_tuple_fixes_string_step(self):
        """A string step value in returned metadata is coerced to int."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        fake_result = MagicMock()
        fake_result.metadata = {"step": "4"}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch.object(saver, "_get_raw_checkpoint", return_value={}), patch.object(
            saver, "_needs_migration", return_value=False
        ), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.get_tuple",
            return_value=fake_result,
        ):
            result = saver.get_tuple(config)
        assert result.metadata["step"] == 4

    def test_get_tuple_handles_unconvertible_step(self):
        """A non-numeric string step is left in place when conversion fails."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        fake_result = MagicMock()
        fake_result.metadata = {"step": "abc"}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch.object(saver, "_get_raw_checkpoint", return_value={}), patch.object(
            saver, "_needs_migration", return_value=False
        ), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.get_tuple",
            return_value=fake_result,
        ):
            result = saver.get_tuple(config)
        assert result.metadata["step"] == "abc"

    def test_get_tuple_validates_ownership(self):
        """Ownership is validated before migration in get_tuple."""
        saver = _make_saver(user_id="alice@example.com")
        config = {
            "configurable": {"thread_id": "bob@example.com_c1", "checkpoint_ns": ""}
        }
        with pytest.raises(PermissionError, match="Access denied"):
            saver.get_tuple(config)


class TestAsyncGetTupleMigration:
    """Tests for PruningMongoDBSaver.aget_tuple migration handling."""

    pytestmark = pytest.mark.anyio

    async def test_aget_tuple_returns_none_on_migration_error(self):
        """An exception during migration causes aget_tuple to return None."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", side_effect=RuntimeError("boom")
        ):
            assert await saver.aget_tuple(config) is None

    async def test_aget_tuple_returns_none_when_still_needs_migration(self):
        """A checkpoint still needing migration after the attempt returns None."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=True
        ), patch.object(
            saver, "_get_raw_checkpoint", return_value={"type": "msgpack"}
        ), patch.object(
            saver, "_needs_migration", return_value=True
        ):
            assert await saver.aget_tuple(config) is None

    async def test_aget_tuple_fixes_string_step(self):
        """A string step in returned metadata is coerced to int (async path)."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        fake_result = MagicMock()
        fake_result.metadata = {"step": "9"}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch.object(saver, "_get_raw_checkpoint", return_value={}), patch.object(
            saver, "_needs_migration", return_value=False
        ), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.aget_tuple",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            result = await saver.aget_tuple(config)
        assert result.metadata["step"] == 9

    async def test_aget_tuple_handles_unconvertible_step(self):
        """A non-numeric step string is left in place (async path)."""
        saver = _make_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        fake_result = MagicMock()
        fake_result.metadata = {"step": "zzz"}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch.object(saver, "_get_raw_checkpoint", return_value={}), patch.object(
            saver, "_needs_migration", return_value=False
        ), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.aget_tuple",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            result = await saver.aget_tuple(config)
        assert result.metadata["step"] == "zzz"


# =========================================================================
# AsyncPruningMongoDBSaver — raw checkpoint, archive, aput, aget_tuple
# =========================================================================


def _make_async_saver_mocked(user_id=None, keep_last_n=-1):
    """Build an AsyncPruningMongoDBSaver with mocked MongoDB collections."""
    from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

    with patch(
        "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.__init__",
        return_value=None,
    ):
        saver = AsyncPruningMongoDBSaver.__new__(AsyncPruningMongoDBSaver)
        saver.keep_last_n = keep_last_n
        saver.user_id = user_id
        saver._indexes_ensured = True
        saver.checkpoint_collection = MagicMock()
        saver.writes_collection = MagicMock()
        saver.db = MagicMock()
        saver.serde = MagicMock()
    return saver


class TestAsyncSaverRawCheckpoint:
    """Tests for AsyncPruningMongoDBSaver raw checkpoint helpers."""

    def test_get_raw_checkpoint_queries_collection(self):
        """_get_raw_checkpoint issues a find_one sorted by checkpoint_id."""
        saver = _make_async_saver_mocked()
        expected = {"_id": "x", "thread_id": "t1"}
        saver.checkpoint_collection.find_one.return_value = expected
        result = saver._get_raw_checkpoint("t1", "")
        assert result is expected
        args = saver.checkpoint_collection.find_one.call_args[0]
        assert args[0] == {"thread_id": "t1", "checkpoint_ns": ""}

    def test_replace_raw_checkpoint_without_id_returns_false(self):
        """A document lacking _id cannot be replaced."""
        saver = _make_async_saver_mocked()
        assert saver._replace_raw_checkpoint("t1", {"thread_id": "t1"}) is False
        saver.checkpoint_collection.replace_one.assert_not_called()

    def test_replace_raw_checkpoint_success(self):
        """A document with _id is replaced and reports matched."""
        saver = _make_async_saver_mocked()
        result_obj = MagicMock()
        result_obj.matched_count = 1
        saver.checkpoint_collection.replace_one.return_value = result_obj
        doc = {"_id": "abc", "thread_id": "t1"}
        assert saver._replace_raw_checkpoint("t1", doc) is True
        saver.checkpoint_collection.replace_one.assert_called_once_with(
            {"_id": "abc"}, doc
        )

    def test_archive_checkpoint_inserts_and_deletes(self):
        """_archive_checkpoint inserts into archive and removes from main."""
        saver = _make_async_saver_mocked()
        archive_col = MagicMock()
        saver.db.__getitem__.return_value = archive_col
        doc = {"_id": "abc", "thread_id": "t1"}
        saver._archive_checkpoint("t1", doc, RuntimeError("bad"))
        archive_col.insert_one.assert_called_once()
        saver.checkpoint_collection.delete_one.assert_called_once_with({"_id": "abc"})

    def test_archive_checkpoint_handles_insert_error(self):
        """An insert failure during archiving is swallowed."""
        saver = _make_async_saver_mocked()
        archive_col = MagicMock()
        archive_col.insert_one.side_effect = RuntimeError("insert down")
        saver.db.__getitem__.return_value = archive_col
        # Should not raise.
        saver._archive_checkpoint("t1", {"_id": "x"}, RuntimeError("orig"))
        saver.checkpoint_collection.delete_one.assert_not_called()


class TestAsyncSaverAput:
    """Tests for AsyncPruningMongoDBSaver.aput pruning and versioning."""

    pytestmark = pytest.mark.anyio

    async def test_aput_adds_format_version_and_skips_pruning(self):
        """aput stamps the format version and skips pruning when disabled."""
        saver = _make_async_saver_mocked(keep_last_n=-1)
        next_cfg = {"configurable": {"thread_id": "t1", "checkpoint_id": "c1"}}
        config = {"configurable": {"thread_id": "t1"}}
        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.aput",
            new_callable=AsyncMock,
            return_value=next_cfg,
        ) as mock_aput:
            result = await saver.aput(config, {"v": 1}, {"source": "loop"}, {})
        assert result is next_cfg
        # versioned metadata includes format_version
        passed_metadata = mock_aput.call_args[0][2]
        assert passed_metadata["format_version"] == saver.format_version
        saver.checkpoint_collection.find.assert_not_called()

    async def test_aput_prunes_excess_checkpoints(self):
        """aput deletes checkpoints beyond keep_last_n."""
        saver = _make_async_saver_mocked(keep_last_n=1)
        next_cfg = {"configurable": {"thread_id": "t1", "checkpoint_id": "c2"}}
        config = {"configurable": {"thread_id": "t1"}}

        # find().sort() returns two docs; keep_last_n=1 means one is deleted.
        sort_cursor = [
            {"checkpoint_id": "c2"},
            {"checkpoint_id": "c1"},
        ]
        find_cursor = MagicMock()
        find_cursor.sort.return_value = sort_cursor
        saver.checkpoint_collection.find.return_value = find_cursor

        with patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.aput",
            new_callable=AsyncMock,
            return_value=next_cfg,
        ):
            await saver.aput(config, {"v": 1}, {}, {})

        saver.checkpoint_collection.delete_one.assert_called_once_with(
            {"thread_id": "t1", "checkpoint_id": "c1"}
        )
        saver.writes_collection.delete_many.assert_called_once_with(
            {"thread_id": "t1", "checkpoint_id": "c1"}
        )


class TestAsyncSaverAgetTuple:
    """Tests for AsyncPruningMongoDBSaver.aget_tuple migration handling."""

    pytestmark = pytest.mark.anyio

    async def test_aget_tuple_returns_none_on_migration_error(self):
        """A migration error returns None from the async saver's aget_tuple."""
        saver = _make_async_saver_mocked()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with patch.object(
            saver, "migrate_checkpoint_if_needed", side_effect=RuntimeError("boom")
        ):
            assert await saver.aget_tuple(config) is None

    async def test_aget_tuple_delegates_to_super(self):
        """A clean migration delegates to the parent aget_tuple."""
        saver = _make_async_saver_mocked()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        sentinel = object()
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch(
            "bili.iris.checkpointers.mongo_checkpointer.MongoDBSaver.aget_tuple",
            new_callable=AsyncMock,
            return_value=sentinel,
        ):
            assert await saver.aget_tuple(config) is sentinel


class TestAsyncSaverQueryFilters:
    """Tests for AsyncPruningMongoDBSaver query filtering and pagination."""

    def test_get_thread_messages_filters_by_type(self):
        """message_types filters out non-matching message classes."""
        saver = _make_async_saver_mocked()
        human = _human_msg("hi")
        ai = _ai_msg("hello")
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": [human, ai]}},
        }
        msgs = saver.get_thread_messages("t1", message_types=["HumanMessage"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_get_thread_messages_pagination(self):
        """offset and limit slice the returned messages."""
        saver = _make_async_saver_mocked()
        msgs_in = [_human_msg(f"m{i}") for i in range(4)]
        saver.checkpoint_collection.find_one.return_value = {
            "thread_id": "t1",
            "checkpoint": {"channel_values": {"messages": msgs_in}},
        }
        msgs = saver.get_thread_messages("t1", limit=2, offset=1)
        assert len(msgs) == 2

    def test_deserialize_returns_empty_for_missing_checkpoint(self):
        """A document without a checkpoint yields an empty dict."""
        saver = _make_async_saver_mocked()
        assert saver._deserialize_checkpoint_data({}) == {}

    def test_athread_exists_delegates(self):
        """athread_exists delegates to the sync thread_exists."""
        saver = _make_async_saver_mocked()
        saver.checkpoint_collection.count_documents.return_value = 1
        assert anyio.run(saver.athread_exists, "t1") is True
