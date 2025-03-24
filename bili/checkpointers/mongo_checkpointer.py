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

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

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


def get_mongo_checkpointer():
    """
    Creates and returns a MongoDB checkpointer instance if a MongoDB database
    can be successfully initialized.

    :return: Returns an instance of `MongoDBSaver` if the MongoDB database
             is successfully initialized, otherwise returns `None`.
    :rtype: MongoDBSaver | None
    """
    mongo_db = get_mongo_client()
    if mongo_db is not None:
        return MongoDBSaver(mongo_db)
    return None
