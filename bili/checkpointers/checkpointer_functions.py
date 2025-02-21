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

from langgraph.checkpoint.memory import MemorySaver

from bili.checkpointers.mongo_checkpointer import get_mongo_checkpointer
from bili.checkpointers.pg_checkpointer import get_pg_checkpointer
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
    checkpointer = get_pg_checkpointer()
    if checkpointer:
        LOGGER.debug("Using PostgresSaver for conversation state checkpointing.")
        return checkpointer

    checkpointer = get_mongo_checkpointer()
    if checkpointer:
        LOGGER.debug("Using MongoDBSaver for conversation state checkpointing.")
        return checkpointer

    # If no database persistence is available, use MemorySaver as a fallback
    LOGGER.debug("Using MemorySaver for conversation state checkpointing.")
    return MemorySaver()
