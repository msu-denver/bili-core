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
