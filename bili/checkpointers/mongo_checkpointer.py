"""
Module: mongo_checkpointer

This module provides functions to manage MongoDB checkpointing for conversation
states within a Streamlit application. It includes functions to initialize a
MongoDB client, close the client, and create a MongoDB checkpointer.

Functions:
    - get_mongo_client():
      Fetches and initializes a shared MongoDB client if the `MONGO_CONNECTION_STRING`
      environment variable is set. Ensures the client is properly closed during
      application shutdown.
    - close_mongo_client(client):
      Closes the provided MongoDB client if it is active.
    - get_mongo_checkpointer():
      Creates and returns a MongoDB checkpointer instance if a MongoDB client
      can be successfully initialized.

Dependencies:
    - atexit: Provides functions to register cleanup functions at program exit.
    - os: Provides functions to interact with the operating system.
    - langgraph.checkpoint.mongodb: Imports MongoDBSaver for MongoDB-based
      checkpointing.
    - pymongo: Provides MongoClient for connecting to MongoDB.
    - bili.streamlit.utils.streamlit_utils: Imports conditional_cache_resource
      for caching resources conditionally.
    - bili.utils.logging_utils: Imports get_logger for logging purposes.

Usage:
    This module is intended to be used within a Streamlit application to manage
    MongoDB checkpointing of conversation states. It provides functions to
    initialize a MongoDB client, close the client, and create a MongoDB
    checkpointer.

Example:
    from bili.streamlit.checkpointer.mongo_checkpointer import \
        get_mongo_client, get_mongo_checkpointer

    # Get the MongoDB client
    mongo_db = get_mongo_client()

    # Get the MongoDB checkpointer
    mongo_checkpointer = get_mongo_checkpointer()
"""

import atexit
import os
import re
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import OperationFailure

# Import MongoDB-specific migrations to register them
import bili.checkpointers.migrations.mongo  # noqa: F401  # pylint: disable=unused-import
from bili.checkpointers.base_checkpointer import QueryableCheckpointerMixin
from bili.checkpointers.versioning import (
    CURRENT_FORMAT_VERSION,
    VersionedCheckpointerMixin,
)
from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@conditional_cache_resource()
def get_mongo_client():
    """
    Fetches and initializes a shared MongoDB client if the `MONGO_CONNECTION_STRING`
    environment variable is set. If the variable is not set, logs the absence and
    returns None. Ensures the MongoDB client is properly closed during application
    shutdown.

    :return: A MongoDB Database object if the `MONGO_CONNECTION_STRING` is set,
             otherwise None.
    :rtype: Database | None
    """
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING", None)

    if mongo_connection_string:
        LOGGER.info("Initializing shared MongoDB client.")
        client = MongoClient(mongo_connection_string)
        db = client["langgraph"]
        atexit.register(close_mongo_client, client)
        return db

    LOGGER.info(
        "MONGO_CONNECTION_STRING environment variable not set. "
        "No MongoDB client created."
    )
    return None


def close_mongo_client(client):
    """
    Closes the provided MongoDB client if it is active.

    :param client: The MongoDB client instance to close.
    :type client: pymongo.MongoClient or None
    """
    if client is not None:
        LOGGER.info("Closing shared MongoDB client.")
        client.close()
        LOGGER.info("Shared MongoDB client closed.")


def get_mongo_checkpointer(keep_last_n: int = 5, user_id: str | None = None):
    """
    Creates and returns a MongoDB checkpointer instance if a MongoDB database
    can be successfully initialized.

    :param keep_last_n: Number of checkpoints to retain per thread (-1 for unlimited)
    :param user_id: Optional user identifier for thread ownership validation.
        When provided, enables multi-tenant security with automatic index creation.
    :return: Returns an instance of `MongoDBSaver` if the MongoDB database
             is successfully initialized, otherwise returns `None`.
    :rtype: MongoDBSaver | None
    """
    mongo_db = get_mongo_client()
    if mongo_db is not None:
        return PruningMongoDBSaver(mongo_db, keep_last_n=keep_last_n, user_id=user_id)
    return None


class PruningMongoDBSaver(
    VersionedCheckpointerMixin, QueryableCheckpointerMixin, MongoDBSaver
):
    """
    Manages saving and pruning of checkpoints in MongoDB for efficient storage.

    This class extends MongoDBSaver to include functionality for pruning old checkpoints
    while maintaining newer ones based on the specified configuration. It ensures that
    only the most recent checkpoints, specified by the `keep_last_n` parameter, are retained
    in the database. The class also ensures necessary MongoDB indexes exist for optimal
    performance.

    Also implements:
    - QueryableCheckpointerMixin: Query methods for conversation data retrieval
    - VersionedCheckpointerMixin: Version detection and lazy migration

    :ivar keep_last_n: Determines the maximum number of recent checkpoints to retain
        in the database. If set to -1 or None, pruning is disabled.
    :type keep_last_n: int
    :ivar format_version: Current checkpoint format version for migrations.
    :type format_version: int
    """

    # Identify this checkpointer type for migrations
    checkpointer_type: str = "mongo"

    # Use the global format version
    format_version: int = CURRENT_FORMAT_VERSION

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        user_id: str | None = None,
        **kwargs: Any,
    ):
        """
        Initializes a new instance of the class, inheriting from its parent class.
        This constructor allows optional arguments and keyword arguments to be
        passed to the parent class, with an additional specific parameter `keep_last_n`
        to manage result retention.

        :param args: Positional arguments passed to the parent class.
        :type args: Any
        :param keep_last_n: Number of most recent results to keep. If set to -1,
            all results are retained without deletion. Defaults to -1.
        :type keep_last_n: int
        :param user_id: Optional user identifier for thread ownership validation.
            When provided, automatically creates user_id index (on-demand migration)
            and validates that thread_ids belong to this user.
        :type user_id: str | None
        :param kwargs: Keyword arguments passed to the parent class.
        :type kwargs: Any
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        self.user_id = user_id
        self._ensure_indexes()

    @staticmethod
    def _drop_conflicting_indexes(collection, desired_keys, desired_name):
        """
        Drop any existing index on the same key pattern that has a different
        name or options. This resolves conflicts between the parent
        MongoDBSaver's auto-named indexes and our custom-named indexes,
        which is required for DocumentDB compatibility.

        :param collection: The MongoDB collection to check.
        :param desired_keys: The key pattern as a list of (field, direction) tuples.
        :param desired_name: The name we want to give our index.
        """
        desired_key_tuple = tuple(desired_keys)
        try:
            existing = collection.index_information()
            for idx_name, idx_info in existing.items():
                if idx_name == "_id_" or idx_name == desired_name:
                    continue
                existing_key_tuple = tuple((k, d) for k, d in idx_info["key"])
                if existing_key_tuple == desired_key_tuple:
                    LOGGER.info(
                        "Dropping conflicting index '%s' on %s "
                        "(same keys as '%s' but different options)",
                        idx_name,
                        collection.name,
                        desired_name,
                    )
                    collection.drop_index(idx_name)
        except OperationFailure as exc:
            LOGGER.warning(
                "Could not check/drop conflicting indexes on %s: %s",
                collection.name,
                exc,
            )

    @staticmethod
    def _create_index_safe(collection, keys, name, **kwargs):
        """
        Create an index, handling DocumentDB limitations gracefully.

        DocumentDB only allows one index build per collection at a time.
        If another worker is already building an index, retry after a
        short delay. If the index already exists with identical options,
        this is a no-op.

        :param collection: The MongoDB collection.
        :param keys: The index key pattern.
        :param name: The index name.
        :param kwargs: Additional index options (unique, background, etc.).
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                collection.create_index(keys, name=name, **kwargs)
                return
            except OperationFailure as exc:
                # Code 40333: "Existing index build in progress"
                if exc.code == 40333 and attempt < max_retries - 1:
                    LOGGER.info(
                        "Index build in progress on %s, retrying in %ds...",
                        collection.name,
                        attempt + 1,
                    )
                    time.sleep(attempt + 1)
                    continue
                # Code 85: "Index already exists with different options"
                if exc.code == 85:
                    LOGGER.warning(
                        "Index conflict on %s for '%s': %s. "
                        "Dropping conflicting index and retrying.",
                        collection.name,
                        name,
                        exc,
                    )
                    PruningMongoDBSaver._drop_conflicting_indexes(
                        collection, keys, name
                    )
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                raise

    def _ensure_indexes(self) -> None:
        """
        Ensures that the required indexes exist in the MongoDB collections.

        Handles conflicts with the parent MongoDBSaver's auto-created indexes
        by dropping any existing index on the same key pattern that has a
        different name or options before creating our custom-named indexes.
        Also handles DocumentDB's single-concurrent-index-build limitation
        by retrying on transient failures.
        """
        LOGGER.info("Ensuring indexes exist in MongoDB checkpointer collections.")

        # Checkpoint index: thread_id + checkpoint_ns + checkpoint_id (desc)
        cp_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", DESCENDING),
        ]
        self._drop_conflicting_indexes(
            self.checkpoint_collection, cp_keys, "idx_thread_ns_id"
        )
        self._create_index_safe(
            self.checkpoint_collection,
            cp_keys,
            "idx_thread_ns_id",
            unique=True,
            background=True,
        )

        # Checkpoint exact-match index (ascending checkpoint_id)
        cp_exact_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", ASCENDING),
        ]
        self._drop_conflicting_indexes(
            self.checkpoint_collection, cp_exact_keys, "idx_thread_ns_exact"
        )
        self._create_index_safe(
            self.checkpoint_collection,
            cp_exact_keys,
            "idx_thread_ns_exact",
            background=True,
        )

        # Writes lookup index
        writes_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", ASCENDING),
        ]
        self._drop_conflicting_indexes(
            self.writes_collection, writes_keys, "idx_writes_lookup"
        )
        self._create_index_safe(
            self.writes_collection,
            writes_keys,
            "idx_writes_lookup",
            background=True,
        )

        # On-demand migration: Add user_id index if user_id is configured
        if self.user_id:
            LOGGER.info("Creating user_id index (on-demand migration)")
            self._create_index_safe(
                self.checkpoint_collection,
                [("user_id", ASCENDING), ("thread_id", ASCENDING)],
                "idx_user_thread",
                background=True,
            )

    # VersionedCheckpointerMixin implementation for MongoDB

    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> dict[str, Any] | None:
        """
        Get raw checkpoint document directly from MongoDB.

        Bypasses LangGraph's deserialization to allow inspection
        and migration of incompatible formats.

        :param thread_id: Thread ID to retrieve
        :param checkpoint_ns: Checkpoint namespace
        :return: Raw checkpoint document or None
        """
        query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        return self.checkpoint_collection.find_one(
            query, sort=[("checkpoint_id", DESCENDING)]
        )

    def _replace_raw_checkpoint(
        self, thread_id: str, document: dict[str, Any], checkpoint_ns: str = ""
    ) -> bool:
        """
        Replace raw checkpoint document in MongoDB.

        :param thread_id: Thread ID to update
        :param document: Migrated document to write
        :param checkpoint_ns: Checkpoint namespace
        :return: True if replacement was successful
        """
        if "_id" not in document:
            LOGGER.warning(
                "Cannot replace document without _id for thread %s", thread_id
            )
            return False

        result = self.checkpoint_collection.replace_one(
            {"_id": document["_id"]}, document
        )
        return result.matched_count > 0

    def _archive_checkpoint(
        self, thread_id: str, document: dict[str, Any], error: Exception
    ) -> None:
        """
        Archive a checkpoint that failed migration.

        Moves the document to a separate collection for manual review.

        :param thread_id: Thread ID of failed checkpoint
        :param document: Raw document that couldn't be migrated
        :param error: Exception that occurred during migration
        """
        archive_collection = self.db["checkpoints_archive"]

        # Add migration failure metadata
        archived_doc = {
            **document,
            "_migration_error": str(error),
            "_migration_timestamp": __import__("datetime").datetime.utcnow(),
            "_original_thread_id": thread_id,
        }

        try:
            archive_collection.insert_one(archived_doc)
            LOGGER.info("Archived failed checkpoint for thread %s", thread_id)

            # Remove from main collection
            if "_id" in document:
                self.checkpoint_collection.delete_one({"_id": document["_id"]})
                LOGGER.info("Removed failed checkpoint from main collection")
        except Exception as archive_error:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to archive checkpoint: %s", archive_error)

    def get_tuple(self, config: RunnableConfig):
        """
        Get checkpoint tuple with automatic migration support.

        Checks if the checkpoint needs migration before calling the parent's
        get_tuple method to avoid deserialization errors.

        :param config: Runnable configuration with thread_id
        :return: Checkpoint tuple or None
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Migrate checkpoint if needed before LangGraph deserializes it
        try:
            LOGGER.debug("Checking if migration needed for thread %s", thread_id)
            did_migrate = self.migrate_checkpoint_if_needed(thread_id, checkpoint_ns)
            if did_migrate:
                LOGGER.info("Migration completed for thread %s", thread_id)
            else:
                LOGGER.debug("No migration needed for thread %s", thread_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error(
                "Migration failed for thread %s: %s", thread_id, e, exc_info=True
            )
            return None

        # Verify migration succeeded - if checkpoint still needs migration, it's corrupted
        raw_doc = self._get_raw_checkpoint(thread_id, checkpoint_ns)
        if self._needs_migration(raw_doc):
            LOGGER.error(
                "Checkpoint for thread %s still needs migration after migration attempt. "
                "This checkpoint is corrupted and cannot be recovered."
                " Returning None to force fresh start.",
                thread_id,
            )
            return None

        # Call parent's get_tuple
        result = super().get_tuple(config)

        # Fix corrupted step value if present (handles legacy data where step was stored as string)
        if result and result.metadata:
            step = result.metadata.get("step")
            if isinstance(step, str):
                try:
                    result.metadata["step"] = int(step)
                    LOGGER.debug(
                        "Fixed corrupted step value for thread %s: %r -> %d",
                        thread_id,
                        step,
                        result.metadata["step"],
                    )
                except (ValueError, TypeError):
                    LOGGER.warning(
                        "Could not convert step value %r to int for thread %s",
                        step,
                        thread_id,
                    )

        return result

    async def aget_tuple(self, config: RunnableConfig):
        """
        Get checkpoint tuple with automatic migration support (async version).

        Checks if the checkpoint needs migration before calling the parent's
        aget_tuple method to avoid deserialization errors.

        :param config: Runnable configuration with thread_id
        :return: Checkpoint tuple or None
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Validate thread ownership before reading
        self._validate_thread_ownership(thread_id)

        # Migrate checkpoint if needed before LangGraph deserializes it
        # Note: Migration uses sync methods since it's infrequent
        try:
            self.migrate_checkpoint_if_needed(thread_id, checkpoint_ns)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Migration failed for thread %s, returning None: %s", thread_id, e
            )
            return None

        # Verify migration succeeded - if checkpoint still needs migration, it's corrupted
        raw_doc = self._get_raw_checkpoint(thread_id, checkpoint_ns)
        if self._needs_migration(raw_doc):
            LOGGER.error(
                "Checkpoint for thread %s still needs migration after migration attempt. "
                "This checkpoint is corrupted and cannot be recovered."
                " Returning None to force fresh start.",
                thread_id,
            )
            return None

        # Call parent's aget_tuple
        result = await super().aget_tuple(config)

        # Fix corrupted step value if present (handles legacy data where step was stored as string)
        if result and result.metadata:
            step = result.metadata.get("step")
            if isinstance(step, str):
                try:
                    result.metadata["step"] = int(step)
                    LOGGER.debug(
                        "Fixed corrupted step value for thread %s: %r -> %d",
                        thread_id,
                        step,
                        result.metadata["step"],
                    )
                except (ValueError, TypeError):
                    LOGGER.warning(
                        "Could not convert step value %r to int for thread %s",
                        step,
                        thread_id,
                    )

        return result

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Async version of put method with format versioning.

        Saves a checkpoint with format version metadata for future migrations.

        :param config: Runnable configuration
        :param checkpoint: Checkpoint to save
        :param metadata: Checkpoint metadata
        :param new_versions: Channel versions
        :return: Updated configuration
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate thread ownership if user_id is configured
        self._validate_thread_ownership(thread_id)

        # Add format version to metadata for future migrations
        versioned_metadata = dict(metadata) if metadata else {}

        # Store format_version as a plain integer
        # LangGraph's dumps_metadata will automatically wrap it in the appropriate format
        versioned_metadata["format_version"] = self.format_version

        # Save the checkpoint with versioned metadata
        result = await super().aput(
            config, checkpoint, versioned_metadata, new_versions
        )

        # update_many is intentionally sync: PruningMongoDBSaver inherits from the
        # synchronous MongoDBSaver (pymongo), so checkpoint_collection is a sync
        # pymongo Collection. For async MongoDB usage, use AsyncPruningMongoDBSaver.
        if self.user_id:
            self.checkpoint_collection.update_many(
                {"thread_id": thread_id}, {"$set": {"user_id": self.user_id}}
            )

        return result

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Saves a checkpoint and prunes older checkpoints if the pruning condition
        is enabled. The pruning removes checkpoints exceeding the specified
        maximum number of retained checkpoints for a given thread ID in the
        checkpoint collection.

        :param config: The configuration object containing details about the
            runnable and thread context.
        :type config: RunnableConfig
        :param checkpoint: The checkpoint object to be saved, containing the
            state of the process that should be stored.
        :type checkpoint: Checkpoint
        :param metadata: Metadata associated with the provided checkpoint,
            describing additional context or properties.
        :type metadata: CheckpointMetadata
        :param new_versions: A collection of new channel versions for data handling,
            used for synchronization during checkpoint operations.
        :type new_versions: ChannelVersions
        :return: A possibly updated configuration object after saving the
            checkpoint and performing potential pruning actions.
        :rtype: RunnableConfig
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate thread ownership if user_id is configured
        self._validate_thread_ownership(thread_id)

        # Add format version to metadata for future migrations
        versioned_metadata = dict(metadata) if metadata else {}

        # Store format_version as a plain integer
        # LangGraph's dumps_metadata will automatically wrap it in the appropriate format
        versioned_metadata["format_version"] = self.format_version

        # Save the checkpoint with versioned metadata
        result_config = super().put(
            config, checkpoint, versioned_metadata, new_versions
        )

        # Update user_id field if configured
        if self.user_id:
            self.checkpoint_collection.update_many(
                {"thread_id": thread_id}, {"$set": {"user_id": self.user_id}}
            )

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return result_config

        # Perform pruning
        query = {
            "thread_id": thread_id,
        }

        docs = list(
            self.checkpoint_collection.find(query).sort("checkpoint_id", DESCENDING)
        )

        if len(docs) > self.keep_last_n:

            to_delete = docs[self.keep_last_n :]

            for doc in to_delete:
                del_query = {
                    "thread_id": thread_id,
                    "checkpoint_id": doc["checkpoint_id"],
                }
                self.checkpoint_collection.delete_one(del_query)
                self.writes_collection.delete_many(del_query)

        return result_config

    # QueryableCheckpointerMixin implementation for MongoDB

    def _deserialize_checkpoint_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a raw checkpoint document.

        Handles both legacy dict and serialized bytes formats.
        After the v1→v2 migration, checkpoints are stored as serialized bytes (msgpack).
        Legacy checkpoints may still be stored as plain dicts.

        :param doc: Raw MongoDB checkpoint document
        :return: Deserialized checkpoint dict
        """
        checkpoint_data = doc.get("checkpoint")
        if checkpoint_data is None:
            return {}
        if isinstance(checkpoint_data, bytes):
            doc_type = doc.get("type", "json")
            return self.serde.loads_typed((doc_type, checkpoint_data))
        # Legacy format: already a dict
        return checkpoint_data

    def get_user_threads(
        self, user_identifier: str, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get all conversation threads for a user from MongoDB."""
        escaped_id = re.escape(user_identifier)
        pipeline = [
            {
                "$match": {
                    "thread_id": {
                        "$regex": f"^{escaped_id}(_|$)"  # user_id or user_id_*
                    }
                }
            },
            {
                "$group": {
                    "_id": "$thread_id",
                    # Use max ObjectId to derive timestamp — works for both
                    # v1 (dict) and v2 (binary) checkpoint formats, unlike
                    # $checkpoint.ts which is inaccessible in binary data.
                    "last_oid": {"$max": "$_id"},
                    "checkpoint_count": {"$sum": 1},
                }
            },
            {
                "$addFields": {
                    "last_updated": {"$toDate": "$last_oid"},
                }
            },
            {"$sort": {"last_updated": DESCENDING}},
        ]

        # Add pagination if specified
        if offset > 0:
            pipeline.append({"$skip": offset})
        if limit is not None:
            pipeline.append({"$limit": limit})

        results = list(self.checkpoint_collection.aggregate(pipeline))

        threads = []
        for result in results:
            thread_id = result["_id"]

            # Extract conversation_id from thread_id
            if "_" in thread_id:
                conversation_id = thread_id.split("_", 1)[1]
            else:
                conversation_id = "default"

            # Get the latest checkpoint to extract first message
            latest_checkpoint = self.checkpoint_collection.find_one(
                {"thread_id": thread_id}, sort=[("checkpoint.ts", DESCENDING)]
            )

            first_message = None
            last_message = None
            message_count = 0
            title = None
            tags = []

            if latest_checkpoint and "checkpoint" in latest_checkpoint:
                checkpoint_data = self._deserialize_checkpoint_data(latest_checkpoint)
                channel_values = checkpoint_data.get("channel_values", {})
                messages = channel_values.get("messages", [])

                # Extract title and tags from state
                title = channel_values.get("title")
                tags = channel_values.get("tags", [])

                if messages:
                    message_count = len(messages)
                    # Get the first and last user messages
                    for msg in messages:
                        if (
                            hasattr(msg, "content")
                            and msg.content
                            and msg.__class__.__name__ == "HumanMessage"
                        ):
                            # Extract text content (handle multimodal)
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [
                                    p.get("text", "")
                                    for p in content
                                    if isinstance(p, dict) and p.get("type") == "text"
                                ]
                                content = " ".join(text_parts)

                            if first_message is None:
                                first_message = content
                            last_message = content

            threads.append(
                {
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "last_updated": result["last_updated"],
                    "checkpoint_count": result["checkpoint_count"],
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "title": title,
                    "tags": tags,
                }
            )

        return threads

    def get_thread_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        offset: int = 0,
        message_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all messages from a conversation thread."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Get the latest checkpoint for this thread
        latest_checkpoint = self.checkpoint_collection.find_one(
            {"thread_id": thread_id}, sort=[("checkpoint.ts", DESCENDING)]
        )

        if not latest_checkpoint or "checkpoint" not in latest_checkpoint:
            return []

        # Extract messages from checkpoint (handles both legacy dict and serialized bytes)
        checkpoint_data = self._deserialize_checkpoint_data(latest_checkpoint)
        channel_values = checkpoint_data.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            msg_type = msg.__class__.__name__

            # Apply message type filter if specified
            if message_types is not None and msg_type not in message_types:
                continue

            # Extract content (handle multimodal)
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)

            # Strip thinking blocks from AI messages
            if msg_type == "AIMessage":
                content = self._strip_thinking_blocks(content)

            messages.append(
                {
                    "role": "user" if msg_type == "HumanMessage" else "assistant",
                    "content": content,
                    "timestamp": None,  # Messages don't have individual timestamps in LangGraph
                }
            )

        # Apply pagination
        if offset > 0 or limit is not None:
            end_index = offset + limit if limit is not None else None
            messages = messages[offset:end_index]

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        result = self.checkpoint_collection.delete_many({"thread_id": thread_id})
        # Also delete writes for this thread
        self.writes_collection.delete_many({"thread_id": thread_id})
        return result.deleted_count > 0

    def get_user_stats(self, user_identifier: str) -> dict[str, Any]:
        """Get usage statistics for a user."""
        threads = self.get_user_threads(user_identifier)

        if not threads:
            return {
                "total_threads": 0,
                "total_messages": 0,
                "total_checkpoints": 0,
                "oldest_thread": None,
                "newest_thread": None,
            }

        total_messages = sum(thread["message_count"] for thread in threads)
        total_checkpoints = sum(thread["checkpoint_count"] for thread in threads)
        oldest_thread = min(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )
        newest_thread = max(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )

        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "total_checkpoints": total_checkpoints,
            "oldest_thread": oldest_thread,
            "newest_thread": newest_thread,
        }

    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists in MongoDB."""
        return (
            self.checkpoint_collection.count_documents(
                {"thread_id": thread_id}, limit=1
            )
            > 0
        )


# Async MongoDB Checkpointer Support for Streaming


class AsyncClientManager:
    """Manages async MongoDB client singleton."""

    def __init__(self):
        self._client = None

    async def get_client(self):
        """Get async MongoDB client (not database)."""
        if self._client is None:
            connection_string = os.getenv("MONGO_CONNECTION_STRING")
            if not connection_string:
                LOGGER.info(
                    "MONGO_CONNECTION_STRING not set. No async MongoDB client created."
                )
                return None

            LOGGER.info("Initializing shared async MongoDB client for streaming.")
            self._client = AsyncIOMotorClient(connection_string)
            atexit.register(self._close_client)

        return self._client

    def _close_client(self):
        """Close async client."""
        if self._client:
            LOGGER.info("Closing shared async MongoDB client.")
            self._client.close()
            self._client = None


_async_client_manager = AsyncClientManager()


async def get_async_mongo_client():
    """Get async MongoDB client."""
    return await _async_client_manager.get_client()


async def get_async_mongo_checkpointer(
    keep_last_n: int = 5, user_id: str | None = None
):
    """
    Creates and returns an async-capable MongoDB checkpointer instance for streaming operations.

    Note: Uses MongoDBSaver (sync init) with async methods as AsyncMongoDBSaver is deprecated.

    :param keep_last_n: Number of checkpoints to keep per thread
    :param user_id: Optional user identifier for thread ownership validation
    :return: PruningMongoDBSaver instance or None (supports async methods)
    :rtype: PruningMongoDBSaver | None
    """
    # Use synchronous client initialization as MongoDBSaver expects a database object
    mongo_db = get_mongo_client()
    if mongo_db is not None:
        LOGGER.info("Using MongoDBSaver with async method support for checkpointing.")
        return PruningMongoDBSaver(mongo_db, keep_last_n=keep_last_n, user_id=user_id)
    return None


class AsyncPruningMongoDBSaver(
    VersionedCheckpointerMixin, QueryableCheckpointerMixin, AsyncMongoDBSaver
):
    """
    Async version of PruningMongoDBSaver for streaming operations.

    Manages saving and pruning of checkpoints in MongoDB using async operations
    for improved performance during streaming.

    Also implements:
    - VersionedCheckpointerMixin: Version detection and lazy migration
    """

    # Identify this checkpointer type for migrations
    checkpointer_type: str = "mongo"

    # Use the global format version
    format_version: int = CURRENT_FORMAT_VERSION

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        user_id: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize async pruning MongoDB saver.

        :param args: Positional arguments for parent class
        :param keep_last_n: Number of checkpoints to keep (-1 for unlimited)
        :param user_id: Optional user identifier for thread ownership validation.
            When provided, automatically creates user_id index (on-demand migration)
            and validates that thread_ids belong to this user.
        :param kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        self.user_id = user_id
        # Lazy-initialized attributes (set on first use)
        self._sync_db: Any = None
        self._indexes_ensured: bool = False

    @staticmethod
    async def _async_drop_conflicting_indexes(collection, desired_keys, desired_name):
        """
        Async version of _drop_conflicting_indexes. Drop any existing index
        on the same key pattern that has a different name or options.

        :param collection: The async MongoDB collection.
        :param desired_keys: The key pattern as a list of (field, direction) tuples.
        :param desired_name: The name we want to give our index.
        """
        desired_key_tuple = tuple(desired_keys)
        try:
            existing = await collection.index_information()
            for idx_name, idx_info in existing.items():
                if idx_name == "_id_" or idx_name == desired_name:
                    continue
                existing_key_tuple = tuple((k, d) for k, d in idx_info["key"])
                if existing_key_tuple == desired_key_tuple:
                    LOGGER.info(
                        "Dropping conflicting index '%s' on %s "
                        "(same keys as '%s' but different options)",
                        idx_name,
                        collection.name,
                        desired_name,
                    )
                    await collection.drop_index(idx_name)
        except OperationFailure as exc:
            LOGGER.warning(
                "Could not check/drop conflicting indexes on %s: %s",
                collection.name,
                exc,
            )

    @staticmethod
    async def _async_create_index_safe(collection, keys, name, **kwargs):
        """
        Async version of _create_index_safe. Create an index, handling
        DocumentDB limitations gracefully.

        :param collection: The async MongoDB collection.
        :param keys: The index key pattern.
        :param name: The index name.
        :param kwargs: Additional index options.
        """
        import asyncio

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await collection.create_index(keys, name=name, **kwargs)
                return
            except OperationFailure as exc:
                if exc.code == 40333 and attempt < max_retries - 1:
                    LOGGER.info(
                        "Index build in progress on %s, retrying in %ds...",
                        collection.name,
                        attempt + 1,
                    )
                    await asyncio.sleep(attempt + 1)
                    continue
                if exc.code == 85:
                    LOGGER.warning(
                        "Index conflict on %s for '%s': %s. "
                        "Dropping conflicting index and retrying.",
                        collection.name,
                        name,
                        exc,
                    )
                    await AsyncPruningMongoDBSaver._async_drop_conflicting_indexes(
                        collection, keys, name
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                raise

    async def _ensure_indexes(self) -> None:
        """
        Ensures that required indexes exist in MongoDB collections (async version).

        Handles conflicts with the parent AsyncMongoDBSaver's auto-created
        indexes and DocumentDB's single-concurrent-index-build limitation.
        """
        LOGGER.info("Ensuring indexes exist in async MongoDB checkpointer collections.")

        cp_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", DESCENDING),
        ]
        await self._async_drop_conflicting_indexes(
            self.checkpoint_collection, cp_keys, "idx_thread_ns_id"
        )
        await self._async_create_index_safe(
            self.checkpoint_collection,
            cp_keys,
            "idx_thread_ns_id",
            unique=True,
            background=True,
        )

        cp_exact_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", ASCENDING),
        ]
        await self._async_drop_conflicting_indexes(
            self.checkpoint_collection, cp_exact_keys, "idx_thread_ns_exact"
        )
        await self._async_create_index_safe(
            self.checkpoint_collection,
            cp_exact_keys,
            "idx_thread_ns_exact",
            background=True,
        )

        writes_keys = [
            ("thread_id", ASCENDING),
            ("checkpoint_ns", ASCENDING),
            ("checkpoint_id", ASCENDING),
        ]
        await self._async_drop_conflicting_indexes(
            self.writes_collection, writes_keys, "idx_writes_lookup"
        )
        await self._async_create_index_safe(
            self.writes_collection,
            writes_keys,
            "idx_writes_lookup",
            background=True,
        )

        # On-demand migration: Add user_id index if user_id is configured
        if self.user_id:
            LOGGER.info("Creating user_id index (on-demand migration)")
            await self._async_create_index_safe(
                self.checkpoint_collection,
                [("user_id", ASCENDING), ("thread_id", ASCENDING)],
                "idx_user_thread",
                background=True,
            )

    # VersionedCheckpointerMixin implementation for async MongoDB

    def _get_sync_db(self):
        """Get a sync MongoDB database for migration operations."""
        # For migrations we need a sync client; VersionedCheckpointerMixin expects sync methods
        # Use the connection string from the async client to create a sync client
        if self._sync_db is None:
            # Get a sync mongo client for migrations
            sync_db = get_mongo_client()
            if sync_db is None:
                LOGGER.error("Failed to get sync MongoDB client for migrations")
                raise RuntimeError("Sync MongoDB client not available for migrations")
            self._sync_db = sync_db
        return self._sync_db

    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> dict[str, Any] | None:
        """
        Get raw checkpoint document directly from MongoDB (sync method for migration).

        Bypasses LangGraph's deserialization to allow inspection and migration
        of incompatible formats.

        :param thread_id: Thread ID to retrieve
        :param checkpoint_ns: Checkpoint namespace
        :return: Raw checkpoint document or None
        """
        # Use sync client for migration (migration is infrequent)
        sync_db = self._get_sync_db()
        collection_name = self.checkpoint_collection.name
        sync_collection = sync_db[collection_name]

        query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        return sync_collection.find_one(query, sort=[("checkpoint_id", DESCENDING)])

    def _replace_raw_checkpoint(
        self, thread_id: str, document: dict[str, Any], checkpoint_ns: str = ""
    ) -> bool:
        """
        Replace raw checkpoint document in MongoDB (sync method for migration).

        :param thread_id: Thread ID to update
        :param document: Migrated document to write
        :param checkpoint_ns: Checkpoint namespace
        :return: True if replacement was successful
        """
        if "_id" not in document:
            LOGGER.warning(
                "Cannot replace document without _id for thread %s", thread_id
            )
            return False

        # Use sync client for migration
        sync_db = self._get_sync_db()
        collection_name = self.checkpoint_collection.name
        sync_collection = sync_db[collection_name]

        result = sync_collection.replace_one({"_id": document["_id"]}, document)
        return result.matched_count > 0

    def _archive_checkpoint(
        self, thread_id: str, document: dict[str, Any], error: Exception
    ) -> None:
        """
        Archive a checkpoint that failed migration.

        Moves the document to a separate collection for manual review.

        :param thread_id: Thread ID of failed checkpoint
        :param document: Raw document that couldn't be migrated
        :param error: Exception that occurred during migration
        """
        sync_db = self._get_sync_db()
        archive_collection = sync_db["checkpoints_archive"]

        # Add migration failure metadata
        archived_doc = {
            **document,
            "_migration_error": str(error),
            "_migration_timestamp": __import__("datetime").datetime.utcnow(),
            "_original_thread_id": thread_id,
        }

        try:
            archive_collection.insert_one(archived_doc)
            LOGGER.info("Archived failed checkpoint for thread %s", thread_id)

            # Remove from main collection
            if "_id" in document:
                collection_name = self.checkpoint_collection.name
                sync_collection = sync_db[collection_name]
                sync_collection.delete_one({"_id": document["_id"]})
                LOGGER.info("Removed failed checkpoint from main collection")
        except Exception as archive_error:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to archive checkpoint: %s", archive_error)

    async def aget_tuple(self, config: RunnableConfig):
        """
        Get checkpoint tuple with automatic migration support (async version).

        Checks if the checkpoint needs migration before calling the parent's
        aget_tuple method to avoid deserialization errors.

        :param config: Runnable configuration with thread_id
        :return: Checkpoint tuple or None
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Migrate checkpoint if needed before LangGraph deserializes it
        # Note: Migration methods are sync since they're infrequent
        try:
            self.migrate_checkpoint_if_needed(thread_id, checkpoint_ns)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Migration failed for thread %s, returning None: %s", thread_id, e
            )
            return None

        # Now safe to call parent's aget_tuple
        return await super().aget_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Async version of put method with pruning support and format versioning.

        :param config: Runnable configuration
        :param checkpoint: Checkpoint to save
        :param metadata: Checkpoint metadata
        :param new_versions: Channel versions
        :return: Updated configuration
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Note: user_id is not written to checkpoint documents here. This class
        # is considered deprecated in this codebase — get_async_mongo_checkpointer()
        # returns a PruningMongoDBSaver (sync init, async methods) which handles
        # user_id tracking in its aput(). AsyncPruningMongoDBSaver is retained for
        # completeness but is not used in production.

        # Ensure indexes exist on first use
        if not self._indexes_ensured:
            await self._ensure_indexes()
            self._indexes_ensured = True

        # Add format version to metadata for future migrations
        versioned_metadata = dict(metadata) if metadata else {}

        # Store format_version as a plain integer
        # LangGraph's dumps_metadata will automatically wrap it in the appropriate format
        versioned_metadata["format_version"] = self.format_version

        # Save the checkpoint using parent's async method with versioned metadata
        result_config = await super().aput(
            config, checkpoint, versioned_metadata, new_versions
        )

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return result_config

        # Perform async pruning
        thread_id = config["configurable"]["thread_id"]
        query = {"thread_id": thread_id}

        # Find documents to potentially delete
        cursor = self.checkpoint_collection.find(query).sort(
            "checkpoint_id", DESCENDING
        )
        docs = await cursor.to_list(length=None)

        if len(docs) > self.keep_last_n:
            to_delete = docs[self.keep_last_n :]

            # Delete old checkpoints and writes asynchronously
            for doc in to_delete:
                del_query = {
                    "thread_id": thread_id,
                    "checkpoint_id": doc["checkpoint_id"],
                }
                await self.checkpoint_collection.delete_one(del_query)
                await self.writes_collection.delete_many(del_query)

        return result_config

    # Async query methods for conversation data retrieval

    def _deserialize_checkpoint_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a raw checkpoint document.

        Handles both legacy dict and serialized bytes formats.

        :param doc: Raw MongoDB checkpoint document
        :return: Deserialized checkpoint dict
        """
        checkpoint_data = doc.get("checkpoint")
        if checkpoint_data is None:
            return {}
        if isinstance(checkpoint_data, bytes):
            doc_type = doc.get("type", "json")
            return self.serde.loads_typed((doc_type, checkpoint_data))
        return checkpoint_data

    async def aget_user_threads(
        self, user_identifier: str, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get all conversation threads for a user from MongoDB (async)."""
        escaped_id = re.escape(user_identifier)
        pipeline = [
            {"$match": {"thread_id": {"$regex": f"^{escaped_id}(_|$)"}}},
            {
                "$group": {
                    "_id": "$thread_id",
                    "last_oid": {"$max": "$_id"},
                    "checkpoint_count": {"$sum": 1},
                }
            },
            {
                "$addFields": {
                    "last_updated": {"$toDate": "$last_oid"},
                }
            },
            {"$sort": {"last_updated": -1}},
        ]

        if offset > 0:
            pipeline.append({"$skip": offset})
        if limit is not None:
            pipeline.append({"$limit": limit})

        results = await self.checkpoint_collection.aggregate(pipeline).to_list(
            length=None
        )

        threads = []
        for result in results:
            thread_id = result["_id"]

            if "_" in thread_id:
                conversation_id = thread_id.split("_", 1)[1]
            else:
                conversation_id = "default"

            latest_checkpoint = await self.checkpoint_collection.find_one(
                {"thread_id": thread_id}, sort=[("checkpoint.ts", DESCENDING)]
            )

            first_message = None
            last_message = None
            message_count = 0
            title = None
            tags = []

            if latest_checkpoint and "checkpoint" in latest_checkpoint:
                checkpoint_data = self._deserialize_checkpoint_data(latest_checkpoint)
                channel_values = checkpoint_data.get("channel_values", {})
                messages = channel_values.get("messages", [])

                title = channel_values.get("title")
                tags = channel_values.get("tags", [])

                if messages:
                    message_count = len(messages)
                    for msg in messages:
                        if (
                            hasattr(msg, "content")
                            and msg.content
                            and msg.__class__.__name__ == "HumanMessage"
                        ):
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [
                                    p.get("text", "")
                                    for p in content
                                    if isinstance(p, dict) and p.get("type") == "text"
                                ]
                                content = " ".join(text_parts)

                            if first_message is None:
                                first_message = content
                            last_message = content

            threads.append(
                {
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "last_updated": result["last_updated"],
                    "checkpoint_count": result["checkpoint_count"],
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "title": title,
                    "tags": tags,
                }
            )

        return threads

    async def aget_thread_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        offset: int = 0,
        message_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all messages from a conversation thread (async)."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        latest_checkpoint = await self.checkpoint_collection.find_one(
            {"thread_id": thread_id}, sort=[("checkpoint.ts", DESCENDING)]
        )

        if not latest_checkpoint or "checkpoint" not in latest_checkpoint:
            return []

        checkpoint_data = self._deserialize_checkpoint_data(latest_checkpoint)
        channel_values = checkpoint_data.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            msg_type = msg.__class__.__name__

            if message_types is not None and msg_type not in message_types:
                continue

            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if msg_type == "AIMessage":
                content = self._strip_thinking_blocks(content)

            messages.append(
                {
                    "role": "user" if msg_type == "HumanMessage" else "assistant",
                    "content": content,
                    "timestamp": None,
                }
            )

        if offset > 0:
            messages = messages[offset:]
        if limit is not None:
            messages = messages[:limit]

        return messages

    async def adelete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread (async)."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        result = await self.checkpoint_collection.delete_many({"thread_id": thread_id})
        # Also delete writes for this thread
        await self.writes_collection.delete_many({"thread_id": thread_id})
        return result.deleted_count > 0

    async def aget_user_stats(self, user_identifier: str) -> dict[str, Any]:
        """Get usage statistics for a user (async)."""
        threads = await self.aget_user_threads(user_identifier)

        if not threads:
            return {
                "total_threads": 0,
                "total_messages": 0,
                "total_checkpoints": 0,
                "oldest_thread": None,
                "newest_thread": None,
            }

        total_messages = sum(thread["message_count"] for thread in threads)
        total_checkpoints = sum(thread["checkpoint_count"] for thread in threads)
        oldest_thread = min(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )
        newest_thread = max(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )

        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "total_checkpoints": total_checkpoints,
            "oldest_thread": oldest_thread,
            "newest_thread": newest_thread,
        }

    async def athread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists in MongoDB (async)."""
        count = await self.checkpoint_collection.count_documents(
            {"thread_id": thread_id}, limit=1
        )
        return count > 0
