"""
Module: checkpointer_functions

This module provides functions to manage checkpointing for conversation states.
It determines the appropriate checkpointer backend based on available
environment variables and returns a ready-to-use checkpointer instance.

Functions:
    - get_checkpointer():
      Determines and returns the appropriate checkpointer using a
      priority cascade:
        1. PostgreSQL (POSTGRES_CONNECTION_STRING)
        2. MongoDB (MONGO_CONNECTION_STRING)
        3. JSONL / local-file (JSONL_CHECKPOINT_PATH) — no database server
        4. In-memory QueryableMemorySaver (fallback)

    - get_async_checkpointer():
      Async variant of get_checkpointer() for streaming operations.

Dependencies:
    - bili.iris.checkpointers.pg_checkpointer: PostgreSQL-backed saver.
    - bili.iris.checkpointers.mongo_checkpointer: MongoDB-backed saver.
    - bili.iris.checkpointers.jsonl_checkpointer: Local-file JSONL saver
      (no server required; triggered by JSONL_CHECKPOINT_PATH env var).
    - bili.iris.checkpointers.memory_checkpointer: QueryableMemorySaver fallback.
    - bili.utils.logging_utils: Logging.

Usage:
    from bili.iris.checkpointers.checkpointer_functions import get_checkpointer

    checkpointer = get_checkpointer()

"""

import os

from bili.iris.checkpointers.memory_checkpointer import QueryableMemorySaver
from bili.iris.checkpointers.mongo_checkpointer import (
    get_async_mongo_checkpointer,
    get_mongo_checkpointer,
)
from bili.iris.checkpointers.pg_checkpointer import (
    get_async_pg_checkpointer,
    get_pg_checkpointer,
)
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def get_checkpointer():
    """
    Determine and return the appropriate checkpointer instance for conversation
    state checkpointing.

    Priority cascade:
        1. PostgreSQL if ``POSTGRES_CONNECTION_STRING`` is set.
        2. MongoDB if ``MONGO_CONNECTION_STRING`` is set.
        3. Local-file JSONL if ``JSONL_CHECKPOINT_PATH`` is set — persists to
           disk without any database server.
        4. In-memory ``QueryableMemorySaver`` as the zero-config fallback.

    :returns: A checkpointer instance selected by the cascade above.
    :rtype: Checkpointer
    """
    # Priority 1: PostgreSQL
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        LOGGER.debug("Using PostgresSaver for conversation state checkpointing.")
        return get_pg_checkpointer()

    # Priority 2: MongoDB
    if os.getenv("MONGO_CONNECTION_STRING"):
        LOGGER.debug("Using MongoDBSaver for conversation state checkpointing.")
        return get_mongo_checkpointer()

    # Priority 3: Local-file JSONL (no server required)
    if os.getenv("JSONL_CHECKPOINT_PATH"):
        from bili.iris.checkpointers.jsonl_checkpointer import (  # pylint: disable=import-outside-toplevel
            get_jsonl_checkpointer,
        )

        LOGGER.debug(
            "Using JSONLCheckpointSaver for conversation state checkpointing "
            "(path=%s).",
            os.getenv("JSONL_CHECKPOINT_PATH"),
        )
        return get_jsonl_checkpointer()

    # Priority 4: In-memory fallback
    LOGGER.debug("Using QueryableMemorySaver for conversation state checkpointing.")
    return QueryableMemorySaver()


async def get_async_checkpointer():
    """
    Determine and return the appropriate async checkpointer instance for
    streaming operations.

    Priority cascade matches ``get_checkpointer()``:
        1. PostgreSQL if ``POSTGRES_CONNECTION_STRING`` is set.
        2. MongoDB if ``MONGO_CONNECTION_STRING`` is set.
        3. Local-file JSONL if ``JSONL_CHECKPOINT_PATH`` is set.
        4. In-memory ``QueryableMemorySaver`` as the zero-config fallback.

    :returns: An async-compatible checkpointer instance.
    :rtype: AsyncPostgresSaver | PruningMongoDBSaver | JSONLCheckpointSaver | QueryableMemorySaver
    """
    # Priority 1: PostgreSQL async checkpointer
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        LOGGER.debug("Using AsyncPostgresSaver for streaming operations.")
        return await get_async_pg_checkpointer()

    # Priority 2: MongoDB async checkpointer
    if os.getenv("MONGO_CONNECTION_STRING"):
        LOGGER.debug("Using MongoDBSaver for streaming operations.")
        return await get_async_mongo_checkpointer()

    # Priority 3: JSONL local-file (async methods delegate via asyncio.to_thread)
    if os.getenv("JSONL_CHECKPOINT_PATH"):
        from bili.iris.checkpointers.jsonl_checkpointer import (  # pylint: disable=import-outside-toplevel
            get_async_jsonl_checkpointer,
        )

        LOGGER.debug(
            "Using JSONLCheckpointSaver for streaming operations (path=%s).",
            os.getenv("JSONL_CHECKPOINT_PATH"),
        )
        return await get_async_jsonl_checkpointer()

    # Priority 4: In-memory fallback (inherently async-compatible)
    LOGGER.debug(
        "Using QueryableMemorySaver for streaming operations (async-compatible fallback)."
    )
    return QueryableMemorySaver()
