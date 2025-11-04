"""
Module: checkpointer_functions

This module provides functions to manage checkpointing for conversation states
within a Streamlit application. It includes functions to initialize and manage
a Postgres connection pool, determine the appropriate checkpointer, and create
a state configuration for conversation tracking.

Functions:
    - get_checkpointer():
      Determines and returns the appropriate checkpointer (PostgresSaver,
      MongoDBSaver, or MemorySaver).

Dependencies:
    - streamlit: Provides the Streamlit library for building web applications.
    - langgraph.checkpoint.memory: Imports MemorySaver for in-memory
      checkpointing.
    - langgraph.checkpoint.mongodb: Imports MongoDBSaver for MongoDB-based
      checkpointing.
    - langgraph.checkpoint.postgres: Imports PostgresSaver for Postgres-based
      checkpointing.
    - bili.streamlit.checkpointer.mongo_checkpointer: Imports functions to get
      MongoDB client and checkpointer.
    - bili.streamlit.checkpointer.pg_checkpointer: Imports functions to get
      Postgres connection pool and checkpointer.
    - bili.utils.logging_utils: Imports get_logger for logging purposes.

Usage:
    This module is intended to be used within a Streamlit application to manage
    checkpointing of conversation states. It provides functions to initialize
    and manage a Postgres connection pool, determine the appropriate
    checkpointer, and create a state configuration for conversation tracking.

Example:
    from bili.streamlit.checkpointer.checkpointer_functions import \
        get_checkpointer, get_state_config

    # Get the appropriate checkpointer
    checkpointer = get_checkpointer()

"""

import os

from bili.checkpointers.memory_checkpointer import QueryableMemorySaver
from bili.checkpointers.mongo_checkpointer import (
    get_async_mongo_checkpointer,
    get_mongo_checkpointer,
)
from bili.checkpointers.pg_checkpointer import (
    get_async_pg_checkpointer,
    get_pg_checkpointer,
)
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def get_checkpointer():
    """
    Determine and return the appropriate checkpointer instance for conversation
    state checkpointing. The function first attempts to acquire a PostgreSQL
    checkpointer, then a MongoDB checkpointer, and finally defaults to an in-memory
    checkpointer if no database persistence options are available. The process
    is logged for debugging purposes.

    :returns: An instance of the checkpointer based on availability. The following
        instances can be returned:

        - PostgreSQL checkpointer if available.
        - MongoDB checkpointer if PostgreSQL is not available.
        - Memory checkpointer as a fallback if both PostgreSQL and MongoDB are
          unavailable.

    :rtype: Checkpointer
    """
    # if POSTGRES_CONNECTION_STRING exists, use PostgresSaver
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        LOGGER.debug("Using PostgresSaver for conversation state checkpointing.")
        return get_pg_checkpointer()

    # if MONGO_CONNECTION_STRING exists, use MongoDBSaver
    if os.getenv("MONGO_CONNECTION_STRING"):
        LOGGER.debug("Using MongoDBSaver for conversation state checkpointing.")
        return get_mongo_checkpointer()

    # If no database persistence is available, use QueryableMemorySaver as a fallback
    LOGGER.debug("Using QueryableMemorySaver for conversation state checkpointing.")
    return QueryableMemorySaver()


async def get_async_checkpointer():
    """
    Determine and return the appropriate async checkpointer instance for streaming
    operations. Supports async PostgreSQL, MongoDB, and Memory checkpointers.

    The priority order matches get_checkpointer():
    1. PostgreSQL if POSTGRES_CONNECTION_STRING is set
    2. MongoDB if MONGO_CONNECTION_STRING is set
    3. MemorySaver as fallback (inherently async-compatible)

    :returns: An async checkpointer instance based on availability.
        - Async PostgreSQL checkpointer if POSTGRES_CONNECTION_STRING is available.
        - Async MongoDB checkpointer if MONGO_CONNECTION_STRING is available.
        - MemorySaver as fallback (works in async contexts).

    :rtype: AsyncPostgresSaver | AsyncMongoDBSaver | MemorySaver
    """
    # Priority 1: PostgreSQL async checkpointer
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        LOGGER.debug("Using AsyncPostgresSaver for streaming operations.")
        return await get_async_pg_checkpointer()

    # Priority 2: MongoDB async checkpointer
    if os.getenv("MONGO_CONNECTION_STRING"):
        LOGGER.debug("Using AsyncMongoDBSaver for streaming operations.")
        return await get_async_mongo_checkpointer()

    # Priority 3: Memory checkpointer (inherently async-compatible, no await needed)
    LOGGER.debug(
        "Using QueryableMemorySaver for streaming operations (async-compatible fallback)."
    )
    return QueryableMemorySaver()
