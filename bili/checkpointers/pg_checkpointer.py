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
import contextlib
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

# Import PostgreSQL-specific migrations (none registered yet, for future-proofing)
import bili.checkpointers.migrations.pg  # noqa: F401
from bili.checkpointers.base_checkpointer import QueryableCheckpointerMixin
from bili.checkpointers.versioning import (
    CURRENT_FORMAT_VERSION,
    VersionedCheckpointerMixin,
)
from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)

# SQL constants shared by sync and async user_id schema migration methods
_SQL_CHECK_USER_ID_COLUMN = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'checkpoints'
    AND column_name = 'user_id'
"""
_SQL_ADD_USER_ID_COLUMN = "ALTER TABLE checkpoints ADD COLUMN user_id TEXT"
_SQL_CHECK_USER_ID_INDEX = """
    SELECT indexname
    FROM pg_indexes
    WHERE tablename = 'checkpoints'
    AND indexname = 'idx_checkpoints_user_thread'
"""
_SQL_CREATE_USER_ID_INDEX = """
    CREATE INDEX idx_checkpoints_user_thread
    ON checkpoints(user_id, thread_id)
"""


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
    postgres_connection_pool_min_size = int(
        os.getenv("POSTGRES_CONNECTION_POOL_MIN_SIZE", "1")
    )
    postgres_connection_pool_max_size = int(
        os.getenv("POSTGRES_CONNECTION_POOL_MAX_SIZE", "20")
    )

    if postgres_connection_string:
        LOGGER.info(
            "Initializing shared Postgres connection pool (min=%d, max=%d).",
            postgres_connection_pool_min_size,
            postgres_connection_pool_max_size,
        )
        pool = ConnectionPool(
            conninfo=f"{postgres_connection_string.rstrip('/')}/langgraph",
            min_size=postgres_connection_pool_min_size,
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


def get_pg_checkpointer(
    keep_last_n: int = 5, user_id: Optional[str] = None
) -> Optional[PostgresSaver]:
    """
    Retrieves a PostgreSQL checkpointer.

    This function attempts to retrieve a PostgreSQL connection pool. If the pool is
    available, it initializes a `PostgresSaver` object using the connection pool,
    performs its setup if it is being used for the first time, and returns the
    checkpointer. If the connection pool is not available, the function will return
    None.

    :param keep_last_n: Number of checkpoints to retain per thread (-1 for unlimited)
    :param user_id: Optional user identifier for thread ownership validation.
        When provided, enables multi-tenant security with automatic schema migration.
    :raises Exception: If there is a failure during the setup process.
    :return: A `PostgresSaver` instance if the PostgreSQL connection pool is
        available, otherwise None.
    :rtype: PostgresSaver | None
    """
    pg_connection_pool = get_pg_connection_pool()
    if pg_connection_pool:
        checkpointer = PruningPostgresSaver(
            pg_connection_pool, keep_last_n=keep_last_n, user_id=user_id
        )
        checkpointer.setup()  # Perform setup if this is the first time
        checkpointer.ensure_indexes()  # Create indexes after tables exist
        return checkpointer
    return None


class PruningPostgresSaver(
    VersionedCheckpointerMixin, QueryableCheckpointerMixin, PostgresSaver
):
    """
    Handles saving checkpoints to a PostgreSQL database with additional logic
    for pruning old checkpoints. The class ensures the database has the
    necessary indexes for pruning and optimizes performance for checkpoint
    storage and retrieval operations.

    Also implements:
    - QueryableCheckpointerMixin: Query methods for conversation data retrieval
    - VersionedCheckpointerMixin: Version detection and lazy migration (future-proofing)

    :ivar keep_last_n: Specifies the number of most recent checkpoints to keep.
                       If set to a negative value, pruning is disabled.
    :type keep_last_n: int
    :ivar format_version: Current checkpoint format version for migrations.
    :type format_version: int
    """

    # Identify this checkpointer type for migrations
    checkpointer_type: str = "pg"

    # Use the global format version
    format_version: int = CURRENT_FORMAT_VERSION

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an instance of the class. Configures the object with the provided
        parameters, initializes state, and performs any necessary setup like ensuring
        indexes.

        :param args: Positional arguments passed to the parent class constructor.
        :param keep_last_n: Specifies the maximum number of entries to retain. If set
            to -1, no limit is applied.
        :param user_id: Optional user identifier for thread ownership validation.
            When provided, automatically adds user_id column to database (on-demand migration)
            and validates that thread_ids belong to this user.
        :param kwargs: Keyword arguments passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        self.user_id = user_id

        # Override parent's Lock with RLock so _cursor can be called reentrantly
        # from put() -> super().put() -> _cursor() when using shared transactions
        self.lock = threading.RLock()
        # Shared connection for atomic put() + user_id UPDATE (None when inactive)
        self._txn_conn = None

        # On-demand migration: Add user_id schema if user_id is provided
        if self.user_id:
            self._ensure_user_id_schema()

    def _ensure_user_id_schema(self) -> None:
        """
        Ensure user_id column and index exist in checkpoints table (on-demand migration).

        This method performs an idempotent schema migration when user_id validation
        is first enabled. It checks if the user_id column exists and creates it if missing,
        along with an index for efficient user-based queries.

        The migration is safe to run multiple times and will not cause errors or
        duplicate structures if already applied.

        :return: None
        """
        with self._cursor() as cur:
            # Check if user_id column exists
            cur.execute(_SQL_CHECK_USER_ID_COLUMN)
            column_exists = cur.fetchone() is not None

            if not column_exists:
                LOGGER.info(
                    "Adding user_id column to checkpoints table (on-demand migration)"
                )
                cur.execute(_SQL_ADD_USER_ID_COLUMN)

            # Check if index exists
            cur.execute(_SQL_CHECK_USER_ID_INDEX)
            index_exists = cur.fetchone() is not None

            if not index_exists:
                LOGGER.info("Creating user_id index on checkpoints table")
                cur.execute(_SQL_CREATE_USER_ID_INDEX)

    @contextmanager
    def _cursor(self, *, pipeline: bool = False):
        """Override parent's _cursor to support shared-connection transactions.

        When self._txn_conn is set (by _put_with_user_id for atomic operations),
        reuse that connection instead of checking out a new one from the pool.
        This allows super().put() and the user_id UPDATE to share the same
        transaction.
        """
        if self._txn_conn is not None:
            with self._txn_conn.cursor(binary=True, row_factory=dict_row) as cur:
                yield cur
        else:
            with super()._cursor(pipeline=pipeline) as cur:
                yield cur

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
            cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
                            ON checkpoints (thread_id, checkpoint_id DESC)
                        """)
            # For cleanup of blobs
            cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_blobs_thread_id
                            ON checkpoint_blobs (thread_id, checkpoint_ns)
                        """)
            # For cleanup of writes
            cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_writes_thread_id
                            ON checkpoint_writes (thread_id, checkpoint_id)
                        """)

    # VersionedCheckpointerMixin implementation for PostgreSQL
    # Note: Currently no migrations are registered for PostgreSQL.
    # These methods are implemented for future-proofing.

    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get raw checkpoint data directly from PostgreSQL.

        Note: PostgreSQL checkpoints use msgpack for some fields, so the raw
        data may need different handling than MongoDB's JSON format.

        :param thread_id: Thread ID to retrieve
        :param checkpoint_ns: Checkpoint namespace
        :return: Raw checkpoint data or None
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = %s AND checkpoint_ns = %s
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (thread_id, checkpoint_ns),
            )
            row = cur.fetchone()
            if row:
                return {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                    "checkpoint": row["checkpoint"],
                    "metadata": row["metadata"],
                }
        return None

    def _replace_raw_checkpoint(
        self, thread_id: str, document: Dict[str, Any], checkpoint_ns: str = ""
    ) -> bool:
        """
        Replace raw checkpoint data in PostgreSQL.

        :param thread_id: Thread ID to update
        :param document: Migrated document to write
        :param checkpoint_ns: Checkpoint namespace
        :return: True if replacement was successful
        """
        if "checkpoint_id" not in document:
            LOGGER.warning(
                "Cannot replace checkpoint without checkpoint_id for thread %s",
                thread_id,
            )
            return False

        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE checkpoints
                SET checkpoint = %s, metadata = %s
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                """,
                (
                    document.get("checkpoint"),
                    document.get("metadata"),
                    thread_id,
                    checkpoint_ns,
                    document["checkpoint_id"],
                ),
            )
            return cur.rowcount > 0

    def _archive_checkpoint(
        self, thread_id: str, document: Dict[str, Any], error: Exception
    ) -> None:
        """
        Archive a checkpoint that failed migration.

        For PostgreSQL, we create an archive table if needed and move the data there.

        :param thread_id: Thread ID of failed checkpoint
        :param document: Raw document that couldn't be migrated
        :param error: Exception that occurred during migration
        """

        with self._cursor() as cur:
            # Create archive table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints_archive (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    checkpoint JSONB,
                    metadata JSONB,
                    migration_error TEXT,
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

            # Insert into archive
            cur.execute(
                """
                INSERT INTO checkpoints_archive
                (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata, migration_error)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    thread_id,
                    document.get("checkpoint_ns", ""),
                    document.get("checkpoint_id", ""),
                    document.get("checkpoint"),
                    document.get("metadata"),
                    str(error),
                ),
            )

            LOGGER.info("Archived failed checkpoint for thread %s", thread_id)

            # Remove from main table
            if document.get("checkpoint_id"):
                cur.execute(
                    """
                    DELETE FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_id = %s
                    """,
                    (thread_id, document["checkpoint_id"]),
                )
                LOGGER.info("Removed failed checkpoint from main table")

    def get_tuple(self, config: RunnableConfig):
        """
        Get checkpoint tuple with automatic migration support.

        Checks if the checkpoint needs migration before calling the parent's
        get_tuple method to avoid deserialization errors.

        Note: Currently no PostgreSQL migrations are registered, so this
        method simply calls the parent implementation.

        :param config: Runnable configuration with thread_id
        :return: Checkpoint tuple or None
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Migrate checkpoint if needed before LangGraph deserializes it
        # Note: This is a no-op if no migrations are registered
        try:
            self.migrate_checkpoint_if_needed(thread_id, checkpoint_ns)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Migration failed for thread %s, returning None: %s", thread_id, e
            )
            return None

        # Now safe to call parent's get_tuple
        return super().get_tuple(config)

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

        When user_id is configured, the checkpoint save and user_id assignment
        are performed atomically within a single database transaction.

        :param config: The current runnable configuration.
        :param checkpoint: The checkpoint to be saved.
        :param metadata: Metadata associated with the checkpoint.
        :param new_versions: New channel versions corresponding to the checkpoint.
        :return: Updated runnable configuration after saving the checkpoint and
        performing pruning, if applicable.
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate thread ownership if user_id is configured
        self._validate_thread_ownership(thread_id)

        # Add format version to metadata for future migrations
        versioned_metadata = dict(metadata) if metadata else {}
        versioned_metadata["format_version"] = self.format_version

        if self.user_id:
            # Atomic path: checkpoint + user_id in one transaction
            next_config = self._put_with_user_id(
                config, checkpoint, versioned_metadata, new_versions, thread_id
            )
        else:
            # Standard path: no user_id needed
            next_config = super().put(
                config, checkpoint, versioned_metadata, new_versions
            )

        # Pruning (separate operation — OK to be non-atomic with the save)
        if self.keep_last_n is not None and self.keep_last_n >= 0:
            self._prune_checkpoints(thread_id)

        return next_config

    def _put_with_user_id(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        thread_id: str,
    ) -> RunnableConfig:
        """Save checkpoint and set user_id atomically in a single transaction.

        Gets a connection from the pool, disables autocommit, and stores the
        connection in self._txn_conn so that the overridden _cursor() reuses it.
        This ensures super().put() and the user_id UPDATE share the same
        transaction.

        :param config: The current runnable configuration.
        :param checkpoint: The checkpoint to be saved.
        :param metadata: Metadata associated with the checkpoint.
        :param new_versions: New channel versions corresponding to the checkpoint.
        :param thread_id: The thread identifier for this checkpoint.
        :return: Updated runnable configuration.
        """
        if isinstance(self.conn, ConnectionPool):
            conn_ctx = self.conn.connection()
        else:
            conn_ctx = contextlib.nullcontext(self.conn)

        with conn_ctx as conn:
            conn.autocommit = False
            try:
                self._txn_conn = conn

                # super().put() calls self._cursor() which sees _txn_conn
                # and reuses our connection inside our transaction
                next_config = super().put(config, checkpoint, metadata, new_versions)

                # UPDATE user_id on the same connection/transaction
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    cur.execute(
                        "UPDATE checkpoints SET user_id = %s WHERE thread_id = %s "
                        "AND checkpoint_id = %s",
                        (
                            self.user_id,
                            thread_id,
                            next_config["configurable"]["checkpoint_id"],
                        ),
                    )
                    if cur.rowcount == 0:
                        checkpoint_id = next_config["configurable"]["checkpoint_id"]
                        raise RuntimeError(
                            f"Failed to set user_id for checkpoint {checkpoint_id}"
                        )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._txn_conn = None
                conn.autocommit = True

        return next_config

    def _prune_checkpoints(self, thread_id: str) -> None:
        """Remove old checkpoints beyond the retention limit.

        :param thread_id: The thread identifier whose checkpoints to prune.
        """
        with self._cursor() as cur:
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

    # QueryableCheckpointerMixin implementation for PostgreSQL

    def get_user_threads(
        self, user_identifier: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversation threads for a user from PostgreSQL."""
        # First, collect all thread metadata using a single cursor
        # We avoid calling get_tuple() inside the cursor block to prevent connection pool deadlock
        thread_metadata = []

        with self._cursor() as cur:
            # Find all unique thread IDs for this user with metadata
            query = """
                SELECT
                    thread_id,
                    MAX(checkpoint_id) as last_checkpoint_id,
                    COUNT(*) as checkpoint_count
                FROM checkpoints
                WHERE thread_id = %s OR thread_id LIKE %s
                GROUP BY thread_id
                ORDER BY MAX(checkpoint_id) DESC
            """
            # Escape LIKE special characters so user_identifier is treated literally
            like_user_id = (
                user_identifier.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            params = [user_identifier, f"{like_user_id}_%"]

            if offset > 0:
                query += " OFFSET %s"
                params.append(offset)
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            results = cur.fetchall()

            for row in results:
                thread_id = row["thread_id"]

                # Extract conversation_id from thread_id
                if "_" in thread_id:
                    conversation_id = thread_id.split("_", 1)[1]
                else:
                    conversation_id = "default"

                # Get the latest checkpoint for metadata (title, tags, timestamp)
                cur.execute(
                    """
                    SELECT checkpoint
                    FROM checkpoints
                    WHERE thread_id = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                latest = cur.fetchone()

                title = None
                tags = []
                latest_checkpoint_ts = None

                if latest and latest["checkpoint"]:
                    checkpoint_data = latest["checkpoint"]
                    channel_values = checkpoint_data.get("channel_values", {})

                    # Extract timestamp from checkpoint (not UUID!)
                    latest_checkpoint_ts = checkpoint_data.get("ts")

                    # Extract title from channel_values (small scalar, stored directly)
                    title = channel_values.get("title")

                    # Get tags from channel_values first
                    tags = channel_values.get("tags", [])
                    if not tags:
                        # Query checkpoint_writes for tags channel
                        cur.execute(
                            """
                            SELECT blob
                            FROM checkpoint_writes
                            WHERE thread_id = %s AND channel = 'tags'
                            ORDER BY checkpoint_id DESC
                            LIMIT 1
                            """,
                            (thread_id,),
                        )
                        tags_row = cur.fetchone()
                        if tags_row and tags_row["blob"]:
                            # Deserialize the value (it's stored as bytes/msgpack)
                            import msgpack

                            tags = msgpack.unpackb(tags_row["blob"], raw=False)

                thread_metadata.append(
                    {
                        "thread_id": thread_id,
                        "conversation_id": conversation_id,
                        "checkpoint_count": row["checkpoint_count"],
                        "latest_checkpoint_ts": latest_checkpoint_ts,
                        "title": title,
                        "tags": tags,
                    }
                )

        # Now, outside the cursor block, call get_tuple() for each thread
        # This avoids connection pool deadlock from nested cursor usage
        threads = []
        for meta in thread_metadata:
            thread_id = meta["thread_id"]
            first_message = None
            last_message = None
            message_count = 0

            # Get message count using get_tuple for proper deserialization
            try:
                config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                checkpoint_tuple = self.get_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    cv = checkpoint_tuple.checkpoint.get("channel_values", {})
                    messages = cv.get("messages", [])
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
                                        if isinstance(p, dict)
                                        and p.get("type") == "text"
                                    ]
                                    content = " ".join(text_parts)

                                if not first_message:
                                    first_message = content
                                last_message = (
                                    content  # Keep updating to get the last one
                                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    "get_user_threads: Failed to get messages for %s: %s", thread_id, e
                )

            threads.append(
                {
                    "thread_id": thread_id,
                    "conversation_id": meta["conversation_id"],
                    "last_updated": meta["latest_checkpoint_ts"],
                    "checkpoint_count": meta["checkpoint_count"],
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "title": meta["title"],
                    "tags": meta["tags"],
                }
            )

        return threads

    def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        message_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all messages from a conversation thread.

        Args:
            thread_id: Thread ID to retrieve messages from
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            message_types: Optional list of message type names to include
                          (e.g., ["HumanMessage", "AIMessage"])
                          If None, returns all message types

        Returns:
            List of message dictionaries with role, content, and timestamp
        """
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Use LangGraph's get_tuple() to retrieve the fully reconstructed checkpoint
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint_tuple = self.get_tuple(config)

        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            LOGGER.warning(
                "get_thread_messages: No checkpoint found for thread %s", thread_id
            )
            return []

        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        LOGGER.info(
            "get_thread_messages: Thread %s - Found %d messages",
            thread_id,
            len(raw_messages),
        )

        # Log message types for debugging
        if raw_messages:
            msg_types = [type(msg).__name__ for msg in raw_messages]
            LOGGER.info("get_thread_messages: Message types: %s", msg_types)

        messages = []
        for msg in raw_messages:
            # Get message type - handle both objects and dicts
            if hasattr(msg, "__class__"):
                msg_class = msg.__class__.__name__
            elif isinstance(msg, dict):
                # For dict-like messages, try to determine type from 'type' field
                msg_class = msg.get("type", msg.get("__class__", "unknown"))
                # Handle serialized format where type might be in different places
                if msg_class == "unknown" and "kwargs" in msg:
                    msg_class = (
                        msg.get("id", ["unknown"])[-1]
                        if isinstance(msg.get("id"), list)
                        else "unknown"
                    )
            else:
                msg_class = "unknown"

            # Map various type representations to standard names
            type_normalization = {
                "human": "HumanMessage",
                "ai": "AIMessage",
                "system": "SystemMessage",
                "tool": "ToolMessage",
                "function": "FunctionMessage",
            }
            msg_class = (
                type_normalization.get(msg_class.lower(), msg_class)
                if isinstance(msg_class, str)
                else msg_class
            )

            # Apply message type filter if specified
            if message_types is not None and msg_class not in message_types:
                LOGGER.debug(
                    "get_thread_messages: Filtering out message of type %s", msg_class
                )
                continue

            # Map message types to roles
            role_mapping = {
                "HumanMessage": "user",
                "AIMessage": "assistant",
                "SystemMessage": "system",
                "ToolMessage": "tool",
                "FunctionMessage": "function",
            }
            role = role_mapping.get(msg_class, "unknown")

            # Get content, handling both objects, dicts, and multimodal formats
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                # Try different possible content locations in dict
                content = msg.get(
                    "content", msg.get("kwargs", {}).get("content", str(msg))
                )
            else:
                content = str(msg)

            # Handle multimodal content (list of {text, type} objects)
            if isinstance(content, list):
                # Extract text from multimodal format
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)

            # Strip thinking tags from AI messages using shared helper
            if msg_class == "AIMessage":
                content = self._strip_thinking_blocks(content)

                # Skip empty AI messages (they should have been removed by normalize_state)
                if not content:
                    LOGGER.debug("get_thread_messages: Skipping empty AI message")
                    continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": None,  # Messages don't have individual timestamps in LangGraph
                }
            )

        # Apply pagination
        if offset > 0 or limit is not None:
            end_index = offset + limit if limit is not None else None
            messages = messages[offset:end_index]

        LOGGER.info(
            "get_thread_messages: Returning %d messages after filtering", len(messages)
        )
        if messages:
            roles = [m["role"] for m in messages]
            LOGGER.info("get_thread_messages: Message roles: %s", roles)

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        with self._cursor() as cur:
            # Delete checkpoints
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            deleted_count = cur.rowcount

            # Also delete writes for this thread
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
            )

            return deleted_count > 0

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
        """Check if a thread exists in PostgreSQL."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id = %s LIMIT 1", (thread_id,)
            )
            return cur.fetchone() is not None


# Async PostgreSQL Checkpointer Support for Streaming


class AsyncConnectionManager:
    """Manages async PostgreSQL connection pool singleton."""

    def __init__(self):
        self._pool = None

    async def get_pool(self):
        """Get async PostgreSQL connection pool."""
        if self._pool is None:
            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
            if not connection_string:
                LOGGER.info(
                    "POSTGRES_CONNECTION_STRING not set. No async PostgreSQL pool created."
                )
                return None

            pool_min_size = int(os.getenv("POSTGRES_CONNECTION_POOL_MIN_SIZE", "1"))
            pool_max_size = int(os.getenv("POSTGRES_CONNECTION_POOL_MAX_SIZE", "20"))

            LOGGER.info(
                "Initializing shared async PostgreSQL connection pool (min=%d, max=%d).",
                pool_min_size,
                pool_max_size,
            )
            self._pool = AsyncConnectionPool(
                conninfo=f"{connection_string.rstrip('/')}/langgraph",
                min_size=pool_min_size,
                max_size=pool_max_size,
                kwargs={"autocommit": True},
            )
            atexit.register(self._close_pool)

        return self._pool

    def _close_pool(self):
        """Close async connection pool."""
        if self._pool:
            LOGGER.info("Closing shared async PostgreSQL connection pool.")
            # AsyncConnectionPool doesn't have a sync close, will close on app exit
            self._pool = None


_async_pool_manager = AsyncConnectionManager()


async def get_async_pg_connection_pool():
    """Get async PostgreSQL connection pool."""
    return await _async_pool_manager.get_pool()


async def get_async_pg_checkpointer(
    keep_last_n: int = 5, user_id: Optional[str] = None
):
    """
    Creates and returns an async PostgreSQL checkpointer instance for streaming operations.

    :param keep_last_n: Number of checkpoints to keep per thread
    :param user_id: Optional user identifier for thread ownership validation.
        When provided, enables multi-tenant security with automatic schema migration.
    :return: AsyncPruningPostgresSaver instance or None
    :rtype: AsyncPruningPostgresSaver | None
    """
    pg_pool = await get_async_pg_connection_pool()
    if pg_pool is not None:
        checkpointer = AsyncPruningPostgresSaver(
            pg_pool, keep_last_n=keep_last_n, user_id=user_id
        )
        await checkpointer.asetup()  # Async setup
        await checkpointer.aensure_indexes()  # Create indexes
        return checkpointer
    return None


class AsyncPruningPostgresSaver(VersionedCheckpointerMixin, AsyncPostgresSaver):
    """
    Async version of PruningPostgresSaver for streaming operations.

    Manages saving and pruning of checkpoints in PostgreSQL using async operations
    for improved performance during streaming.

    Also implements:
    - VersionedCheckpointerMixin: Version detection and lazy migration
    """

    # Identify this checkpointer type for migrations
    checkpointer_type: str = "pg"

    # Use the global format version
    format_version: int = CURRENT_FORMAT_VERSION

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize async pruning PostgreSQL saver.

        :param args: Positional arguments for parent class
        :param keep_last_n: Number of checkpoints to keep (-1 for unlimited)
        :param user_id: Optional user identifier for thread ownership validation.
            When provided, automatically adds user_id column to database (on-demand migration)
            and validates that thread_ids belong to this user.
        :param kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n
        self.user_id = user_id

        # Shared connection for atomic aput() + user_id UPDATE (None when inactive)
        self._async_txn_conn = None

    @asynccontextmanager
    async def _acursor(self, *, pipeline: bool = False):
        """Override parent's _acursor to support shared-connection transactions.

        When self._async_txn_conn is set (by _aput_with_user_id for atomic
        operations), reuse that connection instead of checking out a new one
        from the pool.
        """
        if self._async_txn_conn is not None:
            async with self._async_txn_conn.cursor(
                binary=True, row_factory=dict_row
            ) as cur:
                yield cur
        else:
            async with super()._acursor(pipeline=pipeline) as cur:
                yield cur

    async def aensure_indexes(self) -> None:
        """
        Ensures that required indexes exist in PostgreSQL tables (async version).
        """
        LOGGER.info("Ensuring indexes exist in async PostgreSQL checkpointer tables.")

        async with self._acursor() as cur:
            # For fetching and pruning from checkpoints table
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
                    ON checkpoints (thread_id, checkpoint_id DESC)
                """)
            # For cleanup of blobs
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_blobs_thread_id
                    ON checkpoint_blobs (thread_id, checkpoint_ns)
                """)
            # For cleanup of writes
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_writes_thread_id
                    ON checkpoint_writes (thread_id, checkpoint_id)
                """)

        # On-demand migration: Add user_id schema if user_id is provided
        if self.user_id:
            await self._aensure_user_id_schema()

    async def _aensure_user_id_schema(self) -> None:
        """
        Ensure user_id column and index exist in checkpoints table (async, on-demand migration).

        This method performs an idempotent schema migration when user_id validation
        is first enabled. It checks if the user_id column exists and creates it if missing,
        along with an index for efficient user-based queries.

        The migration is safe to run multiple times and will not cause errors or
        duplicate structures if already applied.

        :return: None
        """
        async with self._acursor() as cur:
            # Check if user_id column exists
            await cur.execute(_SQL_CHECK_USER_ID_COLUMN)
            row = await cur.fetchone()
            column_exists = row is not None

            if not column_exists:
                LOGGER.info(
                    "Adding user_id column to checkpoints table (async on-demand migration)"
                )
                await cur.execute(_SQL_ADD_USER_ID_COLUMN)

            # Check if index exists
            await cur.execute(_SQL_CHECK_USER_ID_INDEX)
            row = await cur.fetchone()
            index_exists = row is not None

            if not index_exists:
                LOGGER.info("Creating user_id index on checkpoints table (async)")
                await cur.execute(_SQL_CREATE_USER_ID_INDEX)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Async version of put method with pruning support.

        When user_id is configured, the checkpoint save and user_id assignment
        are performed atomically within a single database transaction.

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
        versioned_metadata["format_version"] = self.format_version

        if self.user_id:
            # Atomic path: checkpoint + user_id in one transaction
            next_config = await self._aput_with_user_id(
                config, checkpoint, versioned_metadata, new_versions, thread_id
            )
        else:
            # Standard path: no user_id needed
            next_config = await super().aput(
                config, checkpoint, versioned_metadata, new_versions
            )

        # Pruning (separate operation — OK to be non-atomic with the save)
        if self.keep_last_n is not None and self.keep_last_n >= 0:
            await self._aprune_checkpoints(thread_id)

        return next_config

    async def _aput_with_user_id(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        thread_id: str,
    ) -> RunnableConfig:
        """Async: save checkpoint and set user_id atomically in a single transaction.

        :param config: The current runnable configuration.
        :param checkpoint: The checkpoint to be saved.
        :param metadata: Metadata associated with the checkpoint.
        :param new_versions: New channel versions corresponding to the checkpoint.
        :param thread_id: The thread identifier for this checkpoint.
        :return: Updated runnable configuration.
        """
        if isinstance(self.conn, AsyncConnectionPool):
            conn_ctx = self.conn.connection()
        else:
            conn_ctx = contextlib.nullcontext(self.conn)

        async with conn_ctx as conn:
            await conn.set_autocommit(False)
            try:
                self._async_txn_conn = conn

                # super().aput() calls self._acursor() which sees _async_txn_conn
                # and reuses our connection inside our transaction
                next_config = await super().aput(
                    config, checkpoint, metadata, new_versions
                )

                # UPDATE user_id on the same connection/transaction
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    await cur.execute(
                        "UPDATE checkpoints SET user_id = %s WHERE thread_id = %s "
                        "AND checkpoint_id = %s",
                        (
                            self.user_id,
                            thread_id,
                            next_config["configurable"]["checkpoint_id"],
                        ),
                    )
                    if cur.rowcount == 0:
                        checkpoint_id = next_config["configurable"]["checkpoint_id"]
                        raise RuntimeError(
                            f"Failed to set user_id for checkpoint {checkpoint_id}"
                        )

                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            finally:
                self._async_txn_conn = None
                await conn.set_autocommit(True)

        return next_config

    async def _aprune_checkpoints(self, thread_id: str) -> None:
        """Async: remove old checkpoints beyond the retention limit.

        :param thread_id: The thread identifier whose checkpoints to prune.
        """
        async with self._acursor() as cur:
            await cur.execute(
                """
                SELECT checkpoint_id
                FROM checkpoints
                WHERE thread_id = %s
                ORDER BY checkpoint_id DESC
                OFFSET %s
                """,
                (thread_id, self.keep_last_n),
            )
            rows = await cur.fetchall()
            to_delete = [row["checkpoint_id"] for row in rows]

            for checkpoint_id in to_delete:
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )

    # VersionedCheckpointerMixin implementation for async PostgreSQL

    def _get_sync_pool(self) -> ConnectionPool:
        """Get a sync PostgreSQL connection pool for migration operations."""
        # For migrations, we need a sync pool since VersionedCheckpointerMixin expects sync methods
        if not hasattr(self, "_sync_pool"):
            sync_pool = get_pg_connection_pool()
            if sync_pool is None:
                LOGGER.error("Failed to get sync PostgreSQL pool for migrations")
                raise RuntimeError("Sync PostgreSQL pool not available for migrations")
            self._sync_pool = sync_pool
        return self._sync_pool

    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get raw checkpoint data directly from PostgreSQL (sync method for migration).

        Bypasses LangGraph's deserialization to allow inspection and migration
        of incompatible formats.

        :param thread_id: Thread ID to retrieve
        :param checkpoint_ns: Checkpoint namespace
        :return: Raw checkpoint data or None
        """
        # Use sync pool for migration (migration is infrequent)
        sync_pool = self._get_sync_pool()

        with sync_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "thread_id": row[0],
                        "checkpoint_ns": row[1],
                        "checkpoint_id": row[2],
                        "checkpoint": row[3],
                        "metadata": row[4],
                    }
        return None

    def _replace_raw_checkpoint(
        self, thread_id: str, document: Dict[str, Any], checkpoint_ns: str = ""
    ) -> bool:
        """
        Replace raw checkpoint data in PostgreSQL (sync method for migration).

        :param thread_id: Thread ID to update
        :param document: Migrated document to write
        :param checkpoint_ns: Checkpoint namespace
        :return: True if replacement was successful
        """
        if "checkpoint_id" not in document:
            LOGGER.warning(
                "Cannot replace checkpoint without checkpoint_id for thread %s",
                thread_id,
            )
            return False

        # Use sync pool for migration
        sync_pool = self._get_sync_pool()

        with sync_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE checkpoints
                    SET checkpoint = %s, metadata = %s
                    WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    """,
                    (
                        document.get("checkpoint"),
                        document.get("metadata"),
                        thread_id,
                        checkpoint_ns,
                        document["checkpoint_id"],
                    ),
                )
                conn.commit()
                return cur.rowcount > 0

    def _archive_checkpoint(
        self, thread_id: str, document: Dict[str, Any], error: Exception
    ) -> None:
        """
        Archive a checkpoint that failed migration.

        For PostgreSQL, we create an archive table if needed and move the data there.

        :param thread_id: Thread ID of failed checkpoint
        :param document: Raw document that couldn't be migrated
        :param error: Exception that occurred during migration
        """
        # Use sync pool for migration
        sync_pool = self._get_sync_pool()

        with sync_pool.connection() as conn:
            with conn.cursor() as cur:
                # Create archive table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints_archive (
                        id SERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        checkpoint JSONB,
                        metadata JSONB,
                        migration_error TEXT,
                        archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """)

                # Insert into archive
                cur.execute(
                    """
                    INSERT INTO checkpoints_archive
                    (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata, migration_error)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        thread_id,
                        document.get("checkpoint_ns", ""),
                        document.get("checkpoint_id", ""),
                        document.get("checkpoint"),
                        document.get("metadata"),
                        str(error),
                    ),
                )

                LOGGER.info("Archived failed checkpoint for thread %s", thread_id)

                # Remove from main table
                if document.get("checkpoint_id"):
                    cur.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = %s AND checkpoint_id = %s
                        """,
                        (thread_id, document["checkpoint_id"]),
                    )
                    LOGGER.info("Removed failed checkpoint from main table")

                conn.commit()

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
        # Note: Migration uses sync methods internally
        try:
            self.migrate_checkpoint_if_needed(thread_id, checkpoint_ns)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Migration failed for thread %s, continuing anyway: %s",
                thread_id,
                str(e),
            )

        # Now call parent's async get_tuple
        return await super().aget_tuple(config)
