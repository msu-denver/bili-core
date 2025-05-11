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
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import ASCENDING, DESCENDING, MongoClient

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


class PruningMongoDBSaver(MongoDBSaver):
    """
    Manages saving and pruning of checkpoints in MongoDB for efficient storage.

    This class extends MongoDBSaver to include functionality for pruning old checkpoints
    while maintaining newer ones based on the specified configuration. It ensures that
    only the most recent checkpoints, specified by the `keep_last_n` parameter, are retained
    in the database. The class also ensures necessary MongoDB indexes exist for optimal
    performance.

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
