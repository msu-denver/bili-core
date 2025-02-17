"""
Module: pg_checkpointer

This module provides functions to manage PostgreSQL checkpointing for
conversation states within a Streamlit application. It includes functions to
initialize a PostgreSQL connection pool, close the pool, and create a PostgreSQL
checkpointer.

Functions:
    - get_pg_connection_pool():
      Initializes and returns a shared PostgreSQL connection pool if the
      `POSTGRES_CONNECTION_STRING` environment variable is set.
    - close_pg_connection_pool():
      Closes the shared PostgreSQL connection pool if it exists.
    - get_pg_checkpointer():
      Creates and returns a PostgreSQL checkpointer instance if a PostgreSQL
      connection pool can be successfully initialized.

Dependencies:
    - atexit: Provides functions to register cleanup functions at program exit.
    - os: Provides functions to interact with the operating system.
    - langgraph.checkpoint.postgres: Imports PostgresSaver for PostgreSQL-based
      checkpointing.
    - psycopg_pool: Provides ConnectionPool for managing PostgreSQL connections.
    - bili.streamlit.utils.streamlit_utils: Imports conditional_cache_resource
      for caching resources conditionally.
    - bili.utils.logging_utils: Imports get_logger for logging purposes.

Usage:
    This module is intended to be used within a Streamlit application to manage
    PostgreSQL checkpointing of conversation states. It provides functions to
    initialize a PostgreSQL connection pool, close the pool, and create a
    PostgreSQL checkpointer.

Example:
    from bili.streamlit.checkpointer.pg_checkpointer import \
        get_pg_connection_pool, get_pg_checkpointer

    # Get the PostgreSQL connection pool
    pg_connection_pool = get_pg_connection_pool()

    # Get the PostgreSQL checkpointer
    pg_checkpointer = get_pg_checkpointer()
"""

import atexit
import os

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from bili.streamlit.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


@conditional_cache_resource()
def get_pg_connection_pool():
    """
    This function initializes and returns a shared PostgreSQL connection pool if the
    required environment variable `POSTGRES_CONNECTION_STRING` is set. The connection
    pool is configured to use a maximum size as specified by the environment variable
    `POSTGRES_CONNECTION_POOL_MAX_SIZE` or defaults to 20 if not defined. If the
    `POSTGRES_CONNECTION_STRING` is not provided, the function logs a message and
    returns None. The function also ensures the connection pool is properly closed
    during application shutdown by registering `close_pg_connection_pool` with `atexit`.

    :return: A configured PostgreSQL connection pool instance if the `POSTGRES_CONNECTION_STRING`
             is set, otherwise None.
    :rtype: Optional[ConnectionPool]
    """
    postgres_connection_string = os.getenv("POSTGRES_CONNECTION_STRING", None)
    postgres_connection_pool_max_size = int(
        os.getenv("POSTGRES_CONNECTION_POOL_MAX_SIZE", "20")
    )

    if postgres_connection_string:
        LOGGER.info("Initializing shared Postgres connection pool.")
        pool = ConnectionPool(
            conninfo=f"{postgres_connection_string.rstrip('/')}/langgraph",
            max_size=postgres_connection_pool_max_size,
            kwargs={"autocommit": True},
        )
        atexit.register(
            close_pg_connection_pool
        )  # Ensure the pool is closed when the app shuts down
        return pool

    LOGGER.info(
        "POSTGRES_CONNECTION_STRING environment variable not set. "
        "No PostgreSQL connection pool created."
    )
    return None


def close_pg_connection_pool():
    """
    Closes the shared Postgres connection pool if it exists.

    This function retrieves the shared Postgres connection pool and checks if it is
    initialized. If the connection pool exists, it logs the closing process, properly
    closes the connection pool to release resources, and logs the successful closure.

    Raises:
        Any exceptions raised during the closing of the connection pool will need
        to be handled by the caller or higher-level processes.

    :return: None
    """
    connection_pool = get_pg_connection_pool()
    if connection_pool:
        LOGGER.info("Closing shared Postgres connection pool.")
        connection_pool.close()
        LOGGER.info("Shared Postgres connection pool closed.")


def get_pg_checkpointer():
    """
    Retrieves a PostgreSQL checkpointer.

    This function attempts to retrieve a PostgreSQL connection pool. If the pool is
    available, it initializes a `PostgresSaver` object using the connection pool,
    performs its setup if it is being used for the first time, and returns the
    checkpointer. If the connection pool is not available, the function will return
    None.

    :raises Exception: If there is a failure during the setup process.
    :return: A `PostgresSaver` instance if the PostgreSQL connection pool is
        available, otherwise None.
    :rtype: PostgresSaver | None
    """
    pg_connection_pool = get_pg_connection_pool()
    if pg_connection_pool:
        checkpointer = PostgresSaver(pg_connection_pool)
        checkpointer.setup()  # Perform setup if this is the first time
        return checkpointer
    return None
