"""
Migration: v1 to v2 (MongoDB)

Migrates checkpoint data from pre-LangGraph 1.0 msgpack format to the new
JSON format expected by LangGraph 1.0+ checkpointers.

Background:
    LangGraph 1.0 changed the checkpoint serialization format:
    - Old: type='msgpack', value=Binary(msgpack-encoded data)
    - New: type='json', value=Binary(JSON-encoded data)

    Additionally, metadata format changed:
    - Old: {'source': 'loop', 'step': 1}
    - New: {'source': ['json', '"loop"'], 'step': ['json', '1']}

    This migration transcodes msgpack blobs to JSON and wraps metadata values
    in the new tuple format with serializer type hints.

Credit: Based on Kade Shockey's working migration script.

Usage:
    This migration is automatically registered when the mongo migrations
    package is imported. It will be applied by VersionedCheckpointerMixin
    when reading checkpoints with format_version < 2.
"""

import json
from typing import Any, Dict, Optional, Tuple

from bili.checkpointers.versioning import register_migration
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Try to import msgpack for decoding old format
try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    LOGGER.warning(
        "msgpack not available - migration may fail for msgpack-encoded data"
    )

# Try to import bson for Binary handling
try:
    import bson

    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    LOGGER.warning("bson not available - migration may have limited Binary support")


def _custom_ext_decoder(code: int, data: bytes) -> Any:
    """
    Handles nested MsgPack extensions (Code 5 = LangChain objects).

    Args:
        code: Extension type code
        data: Extension data bytes

    Returns:
        Decoded object or ExtType placeholder
    """
    if code == 5:
        try:
            return msgpack.unpackb(data, raw=False, ext_hook=_custom_ext_decoder)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Failed to unpack ExtType(5): %s", e)
            return f"<Failed to unpack ExtType(5): {e}>"
    return msgpack.ExtType(code, data)


def _decode_msgpack_blob(binary_data: bytes) -> Any:
    """
    Decodes msgpack binary blob using custom extension hook.

    Args:
        binary_data: Raw msgpack bytes

    Returns:
        Decoded Python object (dict/list) ready for JSON serialization
    """
    if not MSGPACK_AVAILABLE:
        raise RuntimeError("msgpack package required for migration")

    return msgpack.unpackb(
        binary_data, raw=False, strict_map_key=False, ext_hook=_custom_ext_decoder
    )


def _json_bytes_handler(obj: Any) -> str:
    """
    Helper to handle bytes objects during JSON serialization.

    Args:
        obj: Object to serialize

    Returns:
        String representation

    Raises:
        TypeError: If object type is not serializable
    """
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 to preserve byte values
            return obj.decode("latin-1")
    raise TypeError(f"Type {type(obj)} not serializable")


def _migrate_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Migrates metadata from old format to new tuple format.

    Old format: {'source': 'loop', 'step': 1}
    New format: {'source': ['json', '"loop"'], 'step': ['json', '1']}

    Args:
        metadata: Original metadata dict

    Returns:
        Migrated metadata dict, or None if no changes needed
    """
    if not isinstance(metadata, dict):
        return None

    new_metadata = {}
    modified = False

    for key, value in metadata.items():
        # Check if already migrated (list with 2 items, first is 'json' or 'msgpack')
        if (
            isinstance(value, list)
            and len(value) == 2
            and value[0] in ["json", "msgpack"]
        ):
            new_metadata[key] = value
        else:
            # Special handling for 'step' field - must always be an integer
            # This fixes cases where step was incorrectly stored as a string
            if key == "step":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    LOGGER.warning(
                        "Failed to convert step value %r to int, using as-is", value
                    )

            # Needs migration: Wrap the value in a json tuple
            new_metadata[key] = ["json", json.dumps(value, default=_json_bytes_handler)]
            modified = True

    if modified:
        return new_metadata
    return None


def _migrate_blob(doc: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    """
    Migrates blob from msgpack to json encoding.

    Old: type='msgpack', value/checkpoint=Binary(msgpack bytes)
    New: type='json', value/checkpoint=Binary(JSON bytes)

    Args:
        doc: Document containing 'type' and 'value' or 'checkpoint' fields

    Returns:
        Tuple of (new_type, new_value) or (None, None) if no migration needed
    """
    current_type = doc.get("type")

    if current_type == "json":
        return None, None  # Already json

    if current_type == "msgpack":
        try:
            # 1. Decode MsgPack -> Python Object
            # MongoDB uses 'checkpoint' field, some versions may use 'value'
            value = doc.get("checkpoint", doc.get("value")) 

            # Handle bson.Binary if available
            if BSON_AVAILABLE and isinstance(value, bson.Binary):
                value = bytes(value)

            if not isinstance(value, (bytes, bytearray)):
                LOGGER.warning("Unexpected value type: %s", type(value))
                return None, None

            decoded_obj = _decode_msgpack_blob(value)

            # 2. Encode Python Object -> JSON Bytes
            json_str = json.dumps(decoded_obj, default=_json_bytes_handler)

            # Create Binary if bson available, otherwise just bytes
            if BSON_AVAILABLE:
                new_value = bson.Binary(json_str.encode("utf-8"))
            else:
                new_value = json_str.encode("utf-8")

            return "json", new_value

        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to transcode doc %s: %s", doc.get("_id"), e)
            return None, None

    return None, None


def _needs_migration(doc: Dict[str, Any]) -> bool:
    """
    Check if a document needs migration.

    Args:
        doc: Raw MongoDB document

    Returns:
        True if migration is needed
    """
    # Check if blob needs migration (msgpack -> json)
    if doc.get("type") == "msgpack":
        return True

    # Check if metadata needs migration (old format -> tuple format)
    metadata = doc.get("metadata", {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if not (
                isinstance(value, list)
                and len(value) == 2
                and value[0] in ["json", "msgpack"]
            ):
                return True

    return False


@register_migration("mongo", 1, 2)
def migrate_v1_to_v2(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a checkpoint document from v1 (msgpack) to v2 (json) format.

    This migration handles:
    1. Transcoding msgpack blobs to JSON format
    2. Converting metadata to new tuple format with serializer hints

    Args:
        document: Raw checkpoint document from MongoDB

    Returns:
        Migrated document with updated format

    Example v1 format (old):
        {
            "type": "msgpack",
            "value": Binary(msgpack_bytes),
            "metadata": {"source": "loop", "step": 1}
        }

    Example v2 format (new):
        {
            "type": "json",
            "value": Binary(json_bytes),
            "metadata": {
                "source": ["json", '"loop"'],
                "step": ["json", "1"]
            }
        }
    """
    LOGGER.info("Starting v1 to v2 migration (msgpack -> json)")

    if not _needs_migration(document):
        LOGGER.debug("Document already in v2 format, skipping migration")
        return document

    updates = {}

    # 1. Migrate metadata
    if "metadata" in document:
        new_metadata = _migrate_metadata(document["metadata"])
        if new_metadata:
            updates["metadata"] = new_metadata
            LOGGER.debug("Migrated metadata to tuple format")

    # 2. Migrate blob (type and value/checkpoint fields)
    # MongoDB uses 'checkpoint' field, some versions may use 'value'
    blob_field = "checkpoint" if "checkpoint" in document else "value"
    if blob_field in document and "type" in document:
        new_type, new_value = _migrate_blob(document)
        if new_type:
            updates["type"] = new_type
            updates[blob_field] = new_value
            LOGGER.debug("Migrated %s blob from msgpack to json", blob_field)

    # Apply updates to document
    if updates:
        document.update(updates)
        LOGGER.info("Migration completed, updated fields: %s", list(updates.keys()))
    else:
        LOGGER.debug("No fields needed migration")

    return document


def migrate_checkpoint_collection(
    collection, dry_run: bool = True, batch_size: int = 100
) -> Dict[str, int]:
    """
    Batch migrate an entire checkpoint collection.

    This is a utility function for running migrations on existing data
    outside of the lazy migration system.

    Args:
        collection: MongoDB collection to migrate
        dry_run: If True, don't write changes
        batch_size: Number of documents to process before logging progress

    Returns:
        Dict with migration statistics

    Example:
        from pymongo import MongoClient
        from bili.checkpointers.migrations.mongo.v1_to_v2 import migrate_checkpoint_collection

        client = MongoClient("mongodb://localhost:27017")
        db = client["langgraph"]

        # Dry run first
        stats = migrate_checkpoint_collection(db["checkpoints"], dry_run=True)
        print(stats)

        # Then apply for real
        stats = migrate_checkpoint_collection(db["checkpoints"], dry_run=False)
    """
    stats = {"migrated": 0, "skipped": 0, "failed": 0}

    cursor = collection.find({})

    for doc in cursor:
        try:
            if not _needs_migration(doc):
                stats["skipped"] += 1
                continue

            migrated = migrate_v1_to_v2(doc.copy())

            if dry_run:
                LOGGER.debug("[DRY RUN] Would update %s", doc["_id"])
            else:
                collection.update_one({"_id": doc["_id"]}, {"$set": migrated})

            stats["migrated"] += 1

            if stats["migrated"] % batch_size == 0:
                LOGGER.info("Progress: %d documents migrated", stats["migrated"])

        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to migrate %s: %s", doc.get("_id"), e)
            stats["failed"] += 1

    LOGGER.info(
        "Migration complete: %d migrated, %d skipped, %d failed",
        stats["migrated"],
        stats["skipped"],
        stats["failed"],
    )

    return stats


def migrate_all_collections(db, dry_run: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Migrate all checkpoint-related collections in a database.

    Args:
        db: MongoDB database object
        dry_run: If True, don't write changes

    Returns:
        Dict mapping collection name to migration statistics

    Example:
        from pymongo import MongoClient
        from bili.checkpointers.migrations.mongo.v1_to_v2 import migrate_all_collections

        client = MongoClient("mongodb://localhost:27017")
        db = client["langgraph"]

        # Migrate all collections
        all_stats = migrate_all_collections(db, dry_run=False)
        for col_name, stats in all_stats.items():
            print(f"{col_name}: {stats}")
    """
    collections_to_migrate = [
        "checkpoints",
        "checkpoint_writes",
        "checkpoints_aio",
        "checkpoint_writes_aio",
    ]

    all_stats = {}

    for col_name in collections_to_migrate:
        if col_name not in db.list_collection_names():
            LOGGER.info("Collection '%s' not found, skipping", col_name)
            continue

        LOGGER.info("Migrating collection: %s", col_name)
        stats = migrate_checkpoint_collection(db[col_name], dry_run=dry_run)
        all_stats[col_name] = stats

    return all_stats
