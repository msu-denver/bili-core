"""Tests for PruningMongoDBSaver query interface and pruning logic.

All MongoDB interactions are mocked — no real database is needed.
"""

import datetime
from unittest.mock import MagicMock, patch

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
