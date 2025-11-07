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
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from bili.checkpointers.base_checkpointer import QueryableCheckpointerMixin
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


class PruningPostgresSaver(QueryableCheckpointerMixin, PostgresSaver):
    """
    Handles saving checkpoints to a PostgreSQL database with additional logic
    for pruning old checkpoints. The class ensures the database has the
    necessary indexes for pruning and optimizes performance for checkpoint
    storage and retrieval operations.

    Also implements QueryableCheckpointerMixin to provide query methods for conversation
    data retrieval without exposing PostgreSQL-specific implementation details.

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

    # QueryableCheckpointerMixin implementation for PostgreSQL

    def get_user_threads(
        self, user_identifier: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversation threads for a user from PostgreSQL."""
        with self._cursor() as cur:
            # Find all unique thread IDs for this user with metadata
            query = """
                SELECT
                    thread_id,
                    MAX(checkpoint_id) as last_checkpoint_id,
                    COUNT(*) as checkpoint_count
                FROM checkpoints
                WHERE thread_id ~ %s
                GROUP BY thread_id
                ORDER BY MAX(checkpoint_id) DESC
            """
            params = [
                f"^{user_identifier}(_|$)"
            ]  # Regex: user_identifier or user_identifier_*

            if offset > 0:
                query += " OFFSET %s"
                params.append(offset)
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            results = cur.fetchall()

            threads = []
            for row in results:
                thread_id = row["thread_id"]

                # Extract conversation_id from thread_id
                if "_" in thread_id:
                    conversation_id = thread_id.split("_", 1)[1]
                else:
                    conversation_id = "default"

                # Get the latest checkpoint for metadata
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

                first_message = None
                last_message = None
                message_count = 0
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

                    # Get tags from checkpoint_writes if not in channel_values
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

                    # Get message count from checkpoint_writes without loading all messages
                    cur.execute(
                        """
                        SELECT blob
                        FROM checkpoint_writes
                        WHERE thread_id = %s AND channel = 'messages'
                        ORDER BY checkpoint_id DESC
                        LIMIT 1
                        """,
                        (thread_id,),
                    )
                    messages_row = cur.fetchone()
                    if messages_row and messages_row["blob"]:
                        # Deserialize to get message count and first message
                        import msgpack
                        messages = msgpack.unpackb(messages_row["blob"], raw=False)
                        if messages:
                            message_count = len(messages)
                            # Get the first and last user messages
                            for msg in messages:
                                if (
                                    hasattr(msg, "content")
                                    and msg.content
                                    and msg.__class__.__name__ == "HumanMessage"
                                ):
                                    if not first_message:
                                        first_message = msg.content
                                    last_message = msg.content  # Keep updating to get the last one

                threads.append(
                    {
                        "thread_id": thread_id,
                        "conversation_id": conversation_id,
                        "last_updated": latest_checkpoint_ts,
                        "checkpoint_count": row["checkpoint_count"],
                        "message_count": message_count,
                        "first_message": first_message,
                        "last_message": last_message,
                        "title": title,
                        "tags": tags,
                    }
                )

            return threads

    def get_thread_messages(
        self, thread_id: str, limit: Optional[int] = None, offset: int = 0,
        message_types: Optional[List[str]] = None
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
        import re

        # Use LangGraph's get() method to retrieve fully reconstructed checkpoint
        # This properly merges channel_values with data from checkpoint_writes table
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.get(config)

        if not checkpoint:
            return []

        # Extract messages from the reconstructed checkpoint
        channel_values = checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            # Get message type
            msg_class = msg.__class__.__name__

            # Apply message type filter if specified
            if message_types is not None and msg_class not in message_types:
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

            # Get content, handling both string and multimodal formats
            content = msg.content if hasattr(msg, "content") else str(msg)

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

            # Strip thinking tags from AI messages (application can control this via filtering)
            if msg_class == "AIMessage":
                # Patterns for different LLM providers (case-insensitive)
                patterns = [
                    r"<thinking>(.*?)</thinking>",
                    r"<think>(.*?)</think>",
                    r"<reasoning>(.*?)</reasoning>",
                    r"<internal>(.*?)</internal>",
                ]

                for pattern in patterns:
                    content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

                # Clean up extra whitespace
                content = content.strip()

                # Skip empty AI messages (they should have been removed by normalize_state)
                if not content:
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

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread."""
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

            pool_max_size = int(os.getenv("POSTGRES_CONNECTION_POOL_MAX_SIZE", "20"))

            LOGGER.info(
                "Initializing shared async PostgreSQL connection pool for streaming."
            )
            self._pool = AsyncConnectionPool(
                conninfo=f"{connection_string.rstrip('/')}/langgraph",
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


async def get_async_pg_checkpointer(keep_last_n: int = 5):
    """
    Creates and returns an async PostgreSQL checkpointer instance for streaming operations.

    :param keep_last_n: Number of checkpoints to keep per thread
    :return: AsyncPruningPostgresSaver instance or None
    :rtype: AsyncPruningPostgresSaver | None
    """
    pg_pool = await get_async_pg_connection_pool()
    if pg_pool is not None:
        checkpointer = AsyncPruningPostgresSaver(pg_pool, keep_last_n=keep_last_n)
        await checkpointer.asetup()  # Async setup
        await checkpointer.aensure_indexes()  # Create indexes
        return checkpointer
    return None


class AsyncPruningPostgresSaver(AsyncPostgresSaver):
    """
    Async version of PruningPostgresSaver for streaming operations.

    Manages saving and pruning of checkpoints in PostgreSQL using async operations
    for improved performance during streaming.
    """

    def __init__(
        self,
        *args: Any,
        keep_last_n: int = -1,
        **kwargs: Any,
    ):
        """
        Initialize async pruning PostgreSQL saver.

        :param args: Positional arguments for parent class
        :param keep_last_n: Number of checkpoints to keep (-1 for unlimited)
        :param kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.keep_last_n = keep_last_n

    async def aensure_indexes(self) -> None:
        """
        Ensures that required indexes exist in PostgreSQL tables (async version).
        """
        LOGGER.info("Ensuring indexes exist in async PostgreSQL checkpointer tables.")

        async with self._acursor() as cur:
            # For fetching and pruning from checkpoints table
            await cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
                    ON checkpoints (thread_id, checkpoint_id DESC)
                """
            )
            # For cleanup of blobs
            await cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_blobs_thread_id
                    ON checkpoint_blobs (thread_id, checkpoint_ns)
                """
            )
            # For cleanup of writes
            await cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_writes_thread_id
                    ON checkpoint_writes (thread_id, checkpoint_id)
                """
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
        # Save the checkpoint using parent's async method
        next_config = await super().aput(config, checkpoint, metadata, new_versions)

        # Skip pruning if disabled
        if self.keep_last_n is None or self.keep_last_n < 0:
            return next_config

        # Perform async pruning
        thread_id = config["configurable"]["thread_id"]

        async with self._acursor() as cur:
            # Find old checkpoints to prune
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

            # Delete old checkpoints and writes asynchronously
            for checkpoint_id in to_delete:
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
                    (thread_id, checkpoint_id),
                )

        return next_config
