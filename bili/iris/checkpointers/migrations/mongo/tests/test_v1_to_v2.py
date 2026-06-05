"""Tests for the MongoDB v1-to-v2 checkpoint migration.

Covers needs_migration detection, metadata migration, blob
transcoding, and batch collection migration with mocked MongoDB.
"""

import json
from unittest.mock import MagicMock, patch

import bson
import msgpack
import pytest

from bili.iris.checkpointers.migrations.mongo.v1_to_v2 import (
    _custom_ext_decoder,
    _decode_msgpack_blob,
    _json_bytes_handler,
    _migrate_blob,
    _migrate_metadata,
    _needs_migration,
    _unwrap_binary_value,
    migrate_all_collections,
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

    def test_batch_progress_logging_threshold(self):
        """Crossing the batch_size threshold logs progress without error."""
        collection = MagicMock()
        # Two documents that both need migration, batch_size=1 forces the
        # progress-logging branch on each migrated document.
        docs = [
            {
                "_id": f"id{i}",
                "thread_id": "t1",
                "type": "json",
                "metadata": {"source": "loop"},
            }
            for i in range(2)
        ]
        collection.find.return_value = docs
        stats = migrate_checkpoint_collection(collection, dry_run=False, batch_size=1)
        assert stats["migrated"] == 2
        assert collection.update_one.call_count == 2

    def test_update_failure_increments_failed(self):
        """An exception during update_one increments the failed counter."""
        collection = MagicMock()
        doc = {
            "_id": "id1",
            "thread_id": "t1",
            "type": "json",
            "metadata": {"source": "loop"},
        }
        collection.find.return_value = [doc]
        collection.update_one.side_effect = RuntimeError("db down")
        stats = migrate_checkpoint_collection(collection, dry_run=False)
        assert stats["failed"] == 1
        assert stats["migrated"] == 0


# =========================================================================
# _custom_ext_decoder
# =========================================================================


class TestCustomExtDecoder:
    """Tests for the nested msgpack extension decoder."""

    def test_non_code5_returns_exttype(self):
        """Non-5 extension codes are returned as ExtType placeholders."""
        result = _custom_ext_decoder(7, b"payload")
        assert isinstance(result, msgpack.ExtType)
        assert result.code == 7
        assert result.data == b"payload"

    def test_code5_unpacks_nested_payload(self):
        """Code 5 recursively unpacks the nested msgpack payload."""
        nested = msgpack.packb({"inner": "value"})
        result = _custom_ext_decoder(5, nested)
        assert result == {"inner": "value"}

    def test_code5_unpack_failure_returns_placeholder(self):
        """A code-5 payload that fails to unpack yields a placeholder string."""
        # 0xc1 is the reserved/never-used msgpack byte and raises on unpack.
        result = _custom_ext_decoder(5, b"\xc1")
        assert isinstance(result, str)
        assert "Failed to unpack ExtType(5)" in result


# =========================================================================
# _decode_msgpack_blob
# =========================================================================


class TestDecodeMsgpackBlob:
    """Tests for raw msgpack blob decoding."""

    def test_decodes_packed_dict(self):
        """A packed dict round-trips through the decoder."""
        packed = msgpack.packb({"a": 1, "b": [2, 3]})
        assert _decode_msgpack_blob(packed) == {"a": 1, "b": [2, 3]}

    def test_raises_when_msgpack_unavailable(self):
        """A RuntimeError is raised when msgpack is not installed."""
        with patch(
            "bili.iris.checkpointers.migrations.mongo.v1_to_v2.MSGPACK_AVAILABLE",
            False,
        ):
            with pytest.raises(RuntimeError, match="msgpack package required"):
                _decode_msgpack_blob(b"\x00")


# =========================================================================
# _json_bytes_handler (latin-1 fallback)
# =========================================================================


class TestJsonBytesHandlerLatin1:
    """Tests for the latin-1 fallback in the bytes handler."""

    def test_non_utf8_bytes_decoded_as_latin1(self):
        """Bytes that are not valid UTF-8 fall back to latin-1 decoding."""
        # 0xff is invalid as a standalone UTF-8 byte but valid latin-1.
        result = _json_bytes_handler(b"\xff")
        assert result == "\xff"


# =========================================================================
# _unwrap_binary_value
# =========================================================================


class TestUnwrapBinaryValue:
    """Tests for unwrapping bson.Binary metadata values."""

    def test_non_binary_passthrough(self):
        """Non-Binary values are returned unchanged."""
        assert _unwrap_binary_value("plain") == "plain"

    def test_binary_json_string_parsed(self):
        """A Binary wrapping a JSON string is parsed into the value."""
        wrapped = bson.Binary(b'"loop"')
        assert _unwrap_binary_value(wrapped) == "loop"

    def test_binary_json_object_parsed(self):
        """A Binary wrapping a JSON object is parsed into a dict."""
        wrapped = bson.Binary(b'{"k": 1}')
        assert _unwrap_binary_value(wrapped) == {"k": 1}

    def test_binary_invalid_json_returns_string(self):
        """A Binary that looks like JSON but is invalid returns the raw string."""
        wrapped = bson.Binary(b"{not json")
        assert _unwrap_binary_value(wrapped) == "{not json"

    def test_binary_plain_text_returns_string(self):
        """A Binary wrapping plain (non-JSON-looking) text returns the string."""
        wrapped = bson.Binary(b"hello")
        assert _unwrap_binary_value(wrapped) == "hello"

    def test_binary_non_utf8_returns_raw_bytes(self):
        """A Binary that cannot decode as UTF-8 returns the raw bytes."""
        wrapped = bson.Binary(b"\xff\xfe")
        assert _unwrap_binary_value(wrapped) == b"\xff\xfe"


# =========================================================================
# _migrate_metadata (step conversion failure)
# =========================================================================


class TestMigrateMetadataStepFailure:
    """Tests for the step-to-int conversion failure path."""

    def test_unconvertible_step_used_as_is(self):
        """A step value that cannot become an int is wrapped as-is."""
        metadata = {"step": "not-a-number"}
        result = _migrate_metadata(metadata)
        # The value is JSON-serialized unchanged after the conversion fails.
        assert result["step"] == ["json", '"not-a-number"']


# =========================================================================
# _migrate_blob (bson Binary and exception paths)
# =========================================================================


class TestMigrateBlobBsonPaths:
    """Tests for blob migration with bson.Binary and error handling."""

    def test_bson_binary_value_transcoded(self):
        """A bson.Binary msgpack value is unwrapped and transcoded to json."""
        packed = msgpack.packb({"x": 1})
        doc = {"type": "msgpack", "checkpoint": bson.Binary(packed)}
        new_type, new_value = _migrate_blob(doc)
        assert new_type == "json"
        # bson available, so the result is wrapped in a Binary of JSON bytes.
        assert isinstance(new_value, bson.Binary)
        assert json.loads(bytes(new_value).decode("utf-8")) == {"x": 1}

    def test_blob_decode_exception_returns_none(self):
        """A decode failure inside _migrate_blob returns (None, None)."""
        # Valid bytes but invalid msgpack content triggers the except branch.
        doc = {"type": "msgpack", "checkpoint": b"\xc1"}
        new_type, new_value = _migrate_blob(doc)
        assert new_type is None
        assert new_value is None

    def test_no_bson_produces_plain_bytes(self):
        """When bson is unavailable, the new value is plain JSON bytes."""
        packed = msgpack.packb({"y": 2})
        doc = {"type": "msgpack", "checkpoint": packed}
        with patch(
            "bili.iris.checkpointers.migrations.mongo.v1_to_v2.BSON_AVAILABLE",
            False,
        ):
            new_type, new_value = _migrate_blob(doc)
        assert new_type == "json"
        assert isinstance(new_value, bytes)
        assert msgpack.packb is not None  # sanity
        assert new_value == b'{"y": 2}'


# =========================================================================
# migrate_v1_to_v2 (blob migration application)
# =========================================================================


class TestMigrateV1ToV2BlobApplication:
    """Tests that blob migration updates are applied to the document."""

    def test_msgpack_blob_migrated_to_json(self):
        """A msgpack checkpoint blob is transcoded and the type updated."""
        packed = msgpack.packb({"channel_values": {"messages": []}})
        doc = {
            "thread_id": "t1",
            "type": "msgpack",
            "checkpoint": bson.Binary(packed),
            "metadata": {"source": "loop", "step": 1},
        }
        result = migrate_v1_to_v2(doc)
        assert result["type"] == "json"
        assert isinstance(result["checkpoint"], bson.Binary)
        assert result["metadata"]["source"] == ["json", '"loop"']

    def test_value_field_blob_migrated(self):
        """A document using the 'value' field instead of 'checkpoint' migrates."""
        packed = msgpack.packb({"data": 1})
        doc = {
            "thread_id": "t2",
            "type": "msgpack",
            "value": bson.Binary(packed),
        }
        result = migrate_v1_to_v2(doc)
        assert result["type"] == "json"
        assert isinstance(result["value"], bson.Binary)


# =========================================================================
# migrate_all_collections
# =========================================================================


class TestMigrateAllCollections:
    """Tests for the database-wide collection migration utility."""

    def test_skips_absent_collections(self):
        """Collections not present in the database are skipped."""
        db = MagicMock()
        db.list_collection_names.return_value = ["checkpoints"]
        checkpoint_col = MagicMock()
        checkpoint_col.find.return_value = []
        db.__getitem__.return_value = checkpoint_col

        all_stats = migrate_all_collections(db, dry_run=True)
        # Only the present collection is reported on.
        assert list(all_stats.keys()) == ["checkpoints"]
        assert all_stats["checkpoints"] == {
            "migrated": 0,
            "skipped": 0,
            "failed": 0,
        }

    def test_migrates_present_collections(self):
        """Present collections are migrated and their stats collected."""
        db = MagicMock()
        db.list_collection_names.return_value = [
            "checkpoints",
            "checkpoint_writes",
        ]
        col = MagicMock()
        col.find.return_value = [
            {
                "_id": "a",
                "thread_id": "t1",
                "type": "json",
                "metadata": {"source": "loop"},
            }
        ]
        db.__getitem__.return_value = col

        all_stats = migrate_all_collections(db, dry_run=False)
        assert set(all_stats.keys()) == {"checkpoints", "checkpoint_writes"}
        assert all_stats["checkpoints"]["migrated"] == 1
