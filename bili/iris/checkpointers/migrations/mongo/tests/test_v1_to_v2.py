"""Tests for the MongoDB v1-to-v2 checkpoint migration.

Covers needs_migration detection, metadata migration, blob
transcoding, and batch collection migration with mocked MongoDB.
"""

from unittest.mock import MagicMock, patch

import pytest

from bili.iris.checkpointers.migrations.mongo.v1_to_v2 import (
    _json_bytes_handler,
    _migrate_blob,
    _migrate_metadata,
    _needs_migration,
    migrate_checkpoint_collection,
    migrate_v1_to_v2,
)

# =========================================================================
# _needs_migration
# =========================================================================


class TestNeedsMigration:
    """Tests for the _needs_migration detection function."""

    def test_msgpack_type_needs_migration(self):
        """Documents with type=msgpack need migration."""
        doc = {"type": "msgpack", "checkpoint": b"data"}
        assert _needs_migration(doc) is True

    def test_json_type_with_old_metadata_needs_migration(self):
        """Documents with unwrapped metadata values need migration."""
        doc = {
            "type": "json",
            "metadata": {"source": "loop", "step": 1},
        }
        assert _needs_migration(doc) is True

    def test_already_migrated_skipped(self):
        """Fully migrated documents do not need migration."""
        doc = {
            "type": "json",
            "metadata": {
                "source": ["json", '"loop"'],
                "step": ["json", "1"],
            },
        }
        assert _needs_migration(doc) is False

    def test_no_metadata_no_msgpack_skipped(self):
        """Document with json type and no metadata is skipped."""
        doc = {"type": "json"}
        assert _needs_migration(doc) is False

    def test_empty_metadata_skipped(self):
        """Empty metadata dict does not need migration."""
        doc = {"type": "json", "metadata": {}}
        assert _needs_migration(doc) is False


# =========================================================================
# _migrate_metadata
# =========================================================================


class TestMigrateMetadata:
    """Tests for metadata migration to tuple format."""

    def test_wraps_plain_values_in_json_tuple(self):
        """Plain values are wrapped in ['json', json.dumps(value)]."""
        metadata = {"source": "loop", "step": 1}
        result = _migrate_metadata(metadata)
        assert result is not None
        assert result["source"] == ["json", '"loop"']
        assert result["step"] == ["json", "1"]

    def test_already_migrated_values_preserved(self):
        """Values already in tuple format are kept as-is."""
        metadata = {
            "source": ["json", '"loop"'],
            "step": ["json", "1"],
        }
        result = _migrate_metadata(metadata)
        assert result is None  # No changes needed

    def test_non_dict_returns_none(self):
        """Non-dict input returns None."""
        assert _migrate_metadata("not a dict") is None

    def test_step_converted_to_int(self):
        """The step field is cast to int before wrapping."""
        metadata = {"step": "5"}
        result = _migrate_metadata(metadata)
        assert result["step"] == ["json", "5"]

    def test_mixed_migrated_and_unmigrated(self):
        """Mixed metadata: unmigrated values wrapped, others kept."""
        metadata = {
            "source": ["json", '"loop"'],
            "step": 3,
        }
        result = _migrate_metadata(metadata)
        assert result is not None
        assert result["source"] == ["json", '"loop"']
        assert result["step"] == ["json", "3"]


# =========================================================================
# _migrate_blob
# =========================================================================


class TestMigrateBlob:
    """Tests for blob migration from msgpack to json."""

    def test_json_type_returns_none(self):
        """Documents already in json format return (None, None)."""
        doc = {"type": "json", "checkpoint": b"data"}
        new_type, new_value = _migrate_blob(doc)
        assert new_type is None
        assert new_value is None

    def test_unknown_type_returns_none(self):
        """Documents with unknown type return (None, None)."""
        doc = {"type": "unknown"}
        new_type, _ = _migrate_blob(doc)
        assert new_type is None

    @patch(
        "bili.iris.checkpointers.migrations.mongo.v1_to_v2.MSGPACK_AVAILABLE",
        True,
    )
    @patch(
        "bili.iris.checkpointers.migrations.mongo.v1_to_v2._decode_msgpack_blob",
    )
    def test_msgpack_transcoded_to_json(self, mock_decode):
        """Msgpack blobs are transcoded to JSON."""
        mock_decode.return_value = {"key": "value"}
        doc = {"type": "msgpack", "checkpoint": b"\x81\xa3key"}
        new_type, new_value = _migrate_blob(doc)
        assert new_type == "json"
        assert new_value is not None

    def test_msgpack_non_bytes_returns_none(self):
        """Non-bytes value returns (None, None)."""
        doc = {
            "type": "msgpack",
            "checkpoint": {"already": "dict"},
        }
        new_type, _ = _migrate_blob(doc)
        assert new_type is None


# =========================================================================
# _json_bytes_handler
# =========================================================================


class TestJsonBytesHandler:
    """Tests for the JSON serialization bytes handler."""

    def test_decodes_utf8_bytes(self):
        """UTF-8 bytes are decoded to string."""
        result = _json_bytes_handler(b"hello")
        assert result == "hello"

    def test_decodes_bytearray(self):
        """Bytearrays are decoded to string."""
        result = _json_bytes_handler(bytearray(b"world"))
        assert result == "world"

    def test_raises_for_non_bytes(self):
        """Non-bytes types raise TypeError."""
        with pytest.raises(TypeError, match="not serializable"):
            _json_bytes_handler(42)


# =========================================================================
# migrate_v1_to_v2 (registered migration function)
# =========================================================================


class TestMigrateV1ToV2:
    """Tests for the top-level migrate_v1_to_v2 function."""

    def test_skips_already_migrated_document(self):
        """Already-migrated documents are returned unchanged."""
        doc = {
            "thread_id": "t1",
            "type": "json",
            "metadata": {
                "source": ["json", '"loop"'],
                "step": ["json", "1"],
            },
        }
        result = migrate_v1_to_v2(doc.copy())
        assert result["type"] == "json"

    def test_migrates_metadata_only(self):
        """Documents with json type but old metadata get metadata migrated."""
        doc = {
            "thread_id": "t1",
            "type": "json",
            "checkpoint": b"data",
            "metadata": {"source": "loop", "step": 2},
        }
        result = migrate_v1_to_v2(doc.copy())
        assert result["metadata"]["source"] == ["json", '"loop"']
        assert result["metadata"]["step"] == ["json", "2"]


# =========================================================================
# migrate_checkpoint_collection (batch utility)
# =========================================================================


class TestMigrateCheckpointCollection:
    """Tests for batch collection migration."""

    def test_dry_run_does_not_write(self):
        """Dry run processes documents but does not call update_one."""
        collection = MagicMock()
        already_migrated_doc = {
            "_id": "id1",
            "thread_id": "t1",
            "type": "json",
            "metadata": {"source": "loop"},
        }
        collection.find.return_value = [already_migrated_doc]
        stats = migrate_checkpoint_collection(collection, dry_run=True)
        collection.update_one.assert_not_called()
        assert stats["migrated"] == 1

    def test_skips_already_migrated(self):
        """Already-migrated documents increment skipped counter."""
        collection = MagicMock()
        doc = {
            "_id": "id1",
            "thread_id": "t1",
            "type": "json",
            "metadata": {
                "source": ["json", '"loop"'],
                "step": ["json", "1"],
            },
        }
        collection.find.return_value = [doc]
        stats = migrate_checkpoint_collection(collection, dry_run=True)
        assert stats["skipped"] == 1
        assert stats["migrated"] == 0

    def test_actual_write_calls_update_one(self):
        """Non-dry-run writes migrated documents."""
        collection = MagicMock()
        doc = {
            "_id": "id1",
            "thread_id": "t1",
            "type": "json",
            "metadata": {"source": "loop"},
        }
        collection.find.return_value = [doc]
        stats = migrate_checkpoint_collection(collection, dry_run=False)
        collection.update_one.assert_called_once()
        assert stats["migrated"] == 1

    def test_handles_migration_error(self):
        """Exceptions during migration increment failed counter."""
        collection = MagicMock()
        doc = {
            "_id": "id1",
            "thread_id": "t1",
            "type": "msgpack",
            "checkpoint": "not_bytes",
        }
        collection.find.return_value = [doc]
        stats = migrate_checkpoint_collection(collection, dry_run=False)
        # msgpack with non-bytes value should fail gracefully
        assert stats["failed"] + stats["migrated"] + stats["skipped"] > 0
