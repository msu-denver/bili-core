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
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from pymongo import ASCENDING, DESCENDING, MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

from bili.checkpointers.base_checkpointer import QueryableCheckpointerMixin
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
    :rtype: Optional[Database]
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


def get_mongo_checkpointer(keep_last_n: int = 5):
    """
    Creates and returns a MongoDB checkpointer instance if a MongoDB database
    can be successfully initialized.

    :return: Returns an instance of `MongoDBSaver` if the MongoDB database
             is successfully initialized, otherwise returns `None`.
    :rtype: MongoDBSaver | None
    """
    mongo_db = get_mongo_client()
    if mongo_db is not None:
        return PruningMongoDBSaver(mongo_db, keep_last_n=keep_last_n)
    return None


class PruningMongoDBSaver(QueryableCheckpointerMixin, MongoDBSaver):
    """
    Manages saving and pruning of checkpoints in MongoDB for efficient storage.

    This class extends MongoDBSaver to include functionality for pruning old checkpoints
    while maintaining newer ones based on the specified configuration. It ensures that
    only the most recent checkpoints, specified by the `keep_last_n` parameter, are retained
    in the database. The class also ensures necessary MongoDB indexes exist for optimal
    performance.

    Also implements QueryableCheckpointerMixin to provide query methods for conversation
    data retrieval without exposing MongoDB-specific implementation details.

    :ivar keep_last_n: Determines the maximum number of recent checkpoints to retain
        in the database. If set to -1 or None, pruning is disabled.
    :type keep_last_n: int
    """

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
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
        :param kwargs: Keyword arguments passed to the parent class.
        :type kwargs: Any
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """
        Ensures that the required indexes exist in the MongoDB collections. These indexes are used
        to optimize query performance and enforce index-specific constraints in the checkpoint
        and writes collections.

        Raises:
            Any exception that occurs during index creation will not be explicitly captured.

        :param self: The instance of the object containing the MongoDB collections.

        :return: None
        """
        LOGGER.info("Ensuring indexes exist in MongoDB checkpointer collections.")
        self.checkpoint_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", DESCENDING),
            ],
            name="idx_thread_ns_id",
            background=True,
        )
        self.checkpoint_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", ASCENDING),
            ],
            name="idx_thread_ns_exact",
            background=True,
        )
        self.writes_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", ASCENDING),
            ],
            name="idx_writes_lookup",
            background=True,
        )

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
        # Save the checkpoint
        result_config = super().put(config, checkpoint, metadata, new_versions)

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return result_config

        # Perform pruning
        thread_id = config["configurable"]["thread_id"]
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

    def get_user_threads(
        self,
        user_identifier: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversation threads for a user from MongoDB."""
        pipeline = [
            {
                "$match": {
                    "thread_id": {
                        "$regex": f"^{user_identifier}(_|$)"  # Matches user_identifier or user_identifier_*
                    }
                }
            },
            {
                "$group": {
                    "_id": "$thread_id",
                    "last_updated": {"$max": "$checkpoint.ts"},
                    "checkpoint_count": {"$sum": 1}
                }
            },
            {
                "$sort": {"last_updated": DESCENDING}
            }
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
                {"thread_id": thread_id},
                sort=[("checkpoint.ts", DESCENDING)]
            )

            first_message = None
            message_count = 0
            title = None
            tags = []

            if latest_checkpoint and "checkpoint" in latest_checkpoint:
                channel_values = latest_checkpoint["checkpoint"].get("channel_values", {})
                messages = channel_values.get("messages", [])

                # Extract title and tags from state
                title = channel_values.get("title")
                tags = channel_values.get("tags", [])

                if messages:
                    message_count = len(messages)
                    # Get the first user message
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "HumanMessage":
                            first_message = msg.content
                            break

            threads.append({
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "last_updated": result["last_updated"],
                "checkpoint_count": result["checkpoint_count"],
                "message_count": message_count,
                "first_message": first_message,
                "title": title,
                "tags": tags,
            })

        return threads

    def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all messages from a conversation thread."""
        # Get the latest checkpoint for this thread
        latest_checkpoint = self.checkpoint_collection.find_one(
            {"thread_id": thread_id},
            sort=[("checkpoint.ts", DESCENDING)]
        )

        if not latest_checkpoint or "checkpoint" not in latest_checkpoint:
            return []

        # Extract messages from checkpoint
        channel_values = latest_checkpoint["checkpoint"].get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            messages.append({
                "role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                "content": msg.content if hasattr(msg, 'content') else str(msg),
                "timestamp": None,  # Messages don't have individual timestamps in LangGraph
            })

        # Apply pagination
        if offset > 0 or limit is not None:
            end_index = offset + limit if limit is not None else None
            messages = messages[offset:end_index]

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread."""
        result = self.checkpoint_collection.delete_many({"thread_id": thread_id})
        # Also delete writes for this thread
        self.writes_collection.delete_many({"thread_id": thread_id})
        return result.deleted_count > 0

    def get_user_stats(self, user_identifier: str) -> Dict[str, Any]:
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
        oldest_thread = min((t["last_updated"] for t in threads if t["last_updated"]), default=None)
        newest_thread = max((t["last_updated"] for t in threads if t["last_updated"]), default=None)

        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "total_checkpoints": total_checkpoints,
            "oldest_thread": oldest_thread,
            "newest_thread": newest_thread,
        }

    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists in MongoDB."""
        return self.checkpoint_collection.count_documents({"thread_id": thread_id}, limit=1) > 0


# Async MongoDB Checkpointer Support for Streaming

class AsyncClientManager:
    """Manages async MongoDB client singleton."""
    def __init__(self):
        self._client = None

    async def get_client(self):
        """Get async MongoDB database."""
        if self._client is None:
            connection_string = os.getenv("MONGO_CONNECTION_STRING")
            if not connection_string:
                LOGGER.info("MONGO_CONNECTION_STRING not set. No async MongoDB client created.")
                return None

            LOGGER.info("Initializing shared async MongoDB client for streaming.")
            self._client = AsyncIOMotorClient(connection_string)
            atexit.register(self._close_client)

        return self._client["langgraph"]

    def _close_client(self):
        """Close async client."""
        if self._client:
            LOGGER.info("Closing shared async MongoDB client.")
            self._client.close()
            self._client = None

_async_client_manager = AsyncClientManager()

async def get_async_mongo_client():
    """Get async MongoDB database."""
    return await _async_client_manager.get_client()


async def get_async_mongo_checkpointer(keep_last_n: int = 5):
    """
    Creates and returns an async MongoDB checkpointer instance for streaming operations.

    :param keep_last_n: Number of checkpoints to keep per thread
    :return: AsyncPruningMongoDBSaver instance or None
    :rtype: AsyncPruningMongoDBSaver | None
    """
    mongo_db = await get_async_mongo_client()
    if mongo_db is not None:
        return AsyncPruningMongoDBSaver(mongo_db, keep_last_n=keep_last_n)
    return None


class AsyncPruningMongoDBSaver(AsyncMongoDBSaver):
    """
    Async version of PruningMongoDBSaver for streaming operations.

    Manages saving and pruning of checkpoints in MongoDB using async operations
    for improved performance during streaming.
    """

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        **kwargs: Any,
    ):
        """
        Initialize async pruning MongoDB saver.

        :param args: Positional arguments for parent class
        :param keep_last_n: Number of checkpoints to keep (-1 for unlimited)
        :param kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        # Note: Index creation will be handled lazily on first use

    async def _ensure_indexes(self) -> None:
        """
        Ensures that required indexes exist in MongoDB collections (async version).
        """
        LOGGER.info("Ensuring indexes exist in async MongoDB checkpointer collections.")

        # Create indexes asynchronously
        await self.checkpoint_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", DESCENDING),
            ],
            name="idx_thread_ns_id",
            background=True,
        )
        await self.checkpoint_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", ASCENDING),
            ],
            name="idx_thread_ns_exact",
            background=True,
        )
        await self.writes_collection.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", ASCENDING),
            ],
            name="idx_writes_lookup",
            background=True,
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Async version of put method with pruning support.

        :param config: Runnable configuration
        :param checkpoint: Checkpoint to save
        :param metadata: Checkpoint metadata
        :param new_versions: Channel versions
        :return: Updated configuration
        """
        # Ensure indexes exist on first use
        if not hasattr(self, '_indexes_ensured'):
            await self._ensure_indexes()
            self._indexes_ensured = True

        # Save the checkpoint using parent's async method
        result_config = await super().aput(config, checkpoint, metadata, new_versions)

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return result_config

        # Perform async pruning
        thread_id = config["configurable"]["thread_id"]
        query = {"thread_id": thread_id}

        # Find documents to potentially delete
        cursor = self.checkpoint_collection.find(query).sort("checkpoint_id", DESCENDING)
        docs = await cursor.to_list(length=None)

        if len(docs) > self.keep_last_n:
            to_delete = docs[self.keep_last_n:]

            # Delete old checkpoints and writes asynchronously
            for doc in to_delete:
                del_query = {
                    "thread_id": thread_id,
                    "checkpoint_id": doc["checkpoint_id"],
                }
                await self.checkpoint_collection.delete_one(del_query)
                await self.writes_collection.delete_many(del_query)

        return result_config
