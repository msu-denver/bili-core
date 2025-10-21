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
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
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


def get_pg_checkpointer(keep_last_n: int = 5) -> Optional[PostgresSaver]:
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
        checkpointer = PruningPostgresSaver(pg_connection_pool, keep_last_n=keep_last_n)
        checkpointer.setup()  # Perform setup if this is the first time
        checkpointer.ensure_indexes()  # Create indexes after tables exist
        return checkpointer
    return None


class PruningPostgresSaver(PostgresSaver):
    """
    Handles saving checkpoints to a PostgreSQL database with additional logic
    for pruning old checkpoints. The class ensures the database has the
    necessary indexes for pruning and optimizes performance for checkpoint
    storage and retrieval operations.

    :ivar keep_last_n: Specifies the number of most recent checkpoints to keep.
                       If set to a negative value, pruning is disabled.
    :type keep_last_n: int
    """

    def __init__(self, *args: Any, keep_last_n: int = -1, **kwargs: Any) -> None:
        """
        Initializes an instance of the class. Configures the object with the provided
        parameters, initializes state, and performs any necessary setup like ensuring
        indexes.

        :param args: Positional arguments passed to the parent class constructor.
        :param keep_last_n: Specifies the maximum number of entries to retain. If set
            to -1, no limit is applied.
        :param kwargs: Keyword arguments passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n

    def ensure_indexes(self) -> None:
        """
        Ensures that necessary database indexes are created for improving query
        performance during operations such as fetching and pruning checkpoints,
        cleanup of blobs, and cleanup of writes.

        This method establishes the following indexes if they do not already exist:
        1. `idx_checkpoints_thread_id`: On the `checkpoints` table, indexed by
           `thread_id` and `checkpoint_id` in descending order for efficient pruning
           and fetching.
        2. `idx_blobs_thread_id`: On the `checkpoint_blobs` table, indexed by
           `thread_id` and `checkpoint_id` for efficient cleanup of blobs.
        3. `idx_writes_thread_id`: On the `checkpoint_writes` table, indexed by
           `thread_id` and `checkpoint_id` for efficient cleanup of writes.

        This method operates within a managed cursor context to interact with the
        database, ensuring proper cleanup and resource management.

        :return: None
        """
        with self._cursor() as cur:
            # For fetching and pruning from checkpoints table
            cur.execute(
                """
                        CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
                            ON checkpoints (thread_id, checkpoint_id DESC)
                        """
            )
            # For cleanup of blobs
            cur.execute(
                """
                        CREATE INDEX IF NOT EXISTS idx_blobs_thread_id
                            ON checkpoint_blobs (thread_id, checkpoint_ns)
                        """
            )
            # For cleanup of writes
            cur.execute(
                """
                        CREATE INDEX IF NOT EXISTS idx_writes_thread_id
                            ON checkpoint_writes (thread_id, checkpoint_id)
                        """
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Saves a checkpoint and manages the pruning of old checkpoints
        based on the defined retention policy. This method ensures that some
        of the oldest checkpoints are removed when the specified retention limit
        `keep_last_n` is reached, in order to manage storage usage.

        It uses an SQL-based strategy to delete checkpoint records and associated
        data from the database. The pruning process only executes when the `keep_last_n`
        attribute is set to a value greater than or equal to zero.

        :param config: The current runnable configuration.
        :param checkpoint: The checkpoint to be saved.
        :param metadata: Metadata associated with the checkpoint.
        :param new_versions: New channel versions corresponding to the checkpoint.
        :return: Updated runnable configuration after saving the checkpoint and
        performing pruning, if applicable.
        """
        # Save checkpoint using base implementation
        next_config = super().put(config, checkpoint, metadata, new_versions)

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return next_config

        thread_id = config["configurable"]["thread_id"]

        with self._cursor() as cur:
            # Find old checkpoints to prune
            cur.execute(
                """
                SELECT checkpoint_id
                FROM checkpoints
                WHERE thread_id = %s
                ORDER BY checkpoint_id DESC
                    OFFSET %s
                """,
                (thread_id, self.keep_last_n),
            )
            to_delete = [row["checkpoint_id"] for row in cur.fetchall()]

            for checkpoint_id in to_delete:
                cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )
                cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )

        return next_config
