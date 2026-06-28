"""Checkpointer factory — maps MASConfig.checkpoint_config to bili-core checkpointers.

All bili-core imports are lazy so this module loads without heavy
dependencies (psycopg, pymongo, etc.).
"""

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

# Supported type aliases → canonical names
_TYPE_ALIASES: dict[str, str] = {
    "memory": "memory",
    "postgres": "postgres",
    "pg": "postgres",
    "mongo": "mongo",
    "mongodb": "mongo",
    "auto": "auto",
    "jsonl": "jsonl",  # Local-file JSONL saver — no database server required
    "file": "jsonl",  # Alias for "jsonl"
}


def create_checkpointer_from_config(
    config: dict[str, Any], user_id: str | None = None
) -> Any:
    """Create a checkpointer instance from a checkpoint_config dict.

    Args:
        config: Dict with at minimum a ``"type"`` key.  Additional keys
            (e.g. ``"keep_last_n"``) are forwarded to the checkpointer
            constructor where supported.
        user_id: Optional user identifier for multi-tenant security.
            When provided, enables thread ownership validation and
            triggers on-demand schema migration in database checkpointers.

    Returns:
        A checkpointer instance suitable for
        ``StateGraph.compile(checkpointer=...)``.  Falls back to
        ``MemorySaver`` if the requested type is unavailable.
    """
    raw_type = config.get("type", "memory")
    checkpoint_type = _TYPE_ALIASES.get(raw_type.lower())

    if checkpoint_type is None:
        LOGGER.warning(
            "Unknown checkpoint type '%s'; falling back to memory. "
            "Supported types: %s",
            raw_type,
            list(_TYPE_ALIASES.keys()),
        )
        return _create_memory_checkpointer(user_id=user_id)

    dispatch = {
        "memory": _create_memory_checkpointer,
        "postgres": _create_postgres_checkpointer,
        "mongo": _create_mongo_checkpointer,
        "auto": _create_auto_checkpointer,
        "jsonl": _create_jsonl_checkpointer,
    }
    return dispatch[checkpoint_type](config, user_id=user_id)


# =========================================================================
# Per-type helpers
# =========================================================================


def _create_memory_checkpointer(
    _config: dict[str, Any] | None = None,  # pylint: disable=unused-argument
    user_id: str | None = None,
) -> Any:
    """Create a QueryableMemorySaver, falling back to plain MemorySaver."""
    try:
        from bili.iris.checkpointers.memory_checkpointer import (  # pylint: disable=import-outside-toplevel
            QueryableMemorySaver,
        )

        LOGGER.info(
            "Created QueryableMemorySaver checkpointer%s",
            f" (user_id={user_id})" if user_id else "",
        )
        return QueryableMemorySaver(user_id=user_id)
    except ImportError:
        pass

    from langgraph.checkpoint.memory import (  # pylint: disable=import-error,import-outside-toplevel
        MemorySaver,
    )

    LOGGER.info("Created MemorySaver checkpointer (QueryableMemorySaver unavailable)")
    return MemorySaver()


def _create_postgres_checkpointer(
    config: dict[str, Any], user_id: str | None = None
) -> Any:
    """Create a PostgreSQL checkpointer via bili-core."""
    keep_last_n = config.get("keep_last_n", 5)
    try:
        from bili.iris.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
            get_pg_checkpointer,
        )

        checkpointer = get_pg_checkpointer(keep_last_n=keep_last_n, user_id=user_id)
        if checkpointer is not None:
            LOGGER.info(
                "Created PostgreSQL checkpointer (keep_last_n=%d%s)",
                keep_last_n,
                f", user_id={user_id}" if user_id else "",
            )
            return checkpointer
        LOGGER.warning(
            "PostgreSQL checkpointer returned None "
            "(POSTGRES_CONNECTION_STRING not set?); "
            "falling back to memory"
        )
    except ImportError:
        LOGGER.warning(
            "bili.iris.checkpointers.pg_checkpointer not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)


def _create_mongo_checkpointer(
    config: dict[str, Any], user_id: str | None = None
) -> Any:
    """Create a MongoDB checkpointer via bili-core."""
    keep_last_n = config.get("keep_last_n", 5)
    try:
        from bili.iris.checkpointers.mongo_checkpointer import (  # pylint: disable=import-outside-toplevel
            get_mongo_checkpointer,
        )

        checkpointer = get_mongo_checkpointer(keep_last_n=keep_last_n, user_id=user_id)
        if checkpointer is not None:
            LOGGER.info(
                "Created MongoDB checkpointer (keep_last_n=%d%s)",
                keep_last_n,
                f", user_id={user_id}" if user_id else "",
            )
            return checkpointer
        LOGGER.warning(
            "MongoDB checkpointer returned None "
            "(MONGO_CONNECTION_STRING not set?); "
            "falling back to memory"
        )
    except ImportError:
        LOGGER.warning(
            "bili.iris.checkpointers.mongo_checkpointer not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)


def _create_auto_checkpointer(
    config: dict[str, Any] = None, user_id: str | None = None
) -> Any:
    """Auto-detect checkpointer by checking environment variables.

    Forwards keep_last_n and user_id parameters if specified.
    Mirrors the logic from bili.iris.checkpointers.checkpointer_functions.get_checkpointer
    but passes arguments correctly.

    Note: JSONL is intentionally excluded from auto-detection here.  The 'auto'
    type targets server-backed stores (Postgres, Mongo); callers that want the
    local-file backend should declare ``type: jsonl`` explicitly.
    """
    config = config or {}
    keep_last_n = config.get("keep_last_n", 5)

    try:
        import os  # pylint: disable=import-outside-toplevel

        # If POSTGRES_CONNECTION_STRING exists, use PostgresSaver
        if os.getenv("POSTGRES_CONNECTION_STRING"):
            from bili.iris.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
                get_pg_checkpointer,
            )

            LOGGER.debug("Auto-detected Postgres checkpointer.")
            return get_pg_checkpointer(keep_last_n=keep_last_n, user_id=user_id)

        # If MONGO_CONNECTION_STRING exists, use MongoDBSaver
        if os.getenv("MONGO_CONNECTION_STRING"):
            from bili.iris.checkpointers.mongo_checkpointer import (  # pylint: disable=import-outside-toplevel
                get_mongo_checkpointer,
            )

            LOGGER.debug("Auto-detected Mongo checkpointer.")
            return get_mongo_checkpointer(keep_last_n=keep_last_n, user_id=user_id)

        # Fallback to in-memory
        from bili.iris.checkpointers.memory_checkpointer import (  # pylint: disable=import-outside-toplevel
            QueryableMemorySaver,
        )

        LOGGER.debug("Auto-detected Memory checkpointer.")
        return QueryableMemorySaver(user_id=user_id)

    except ImportError:
        LOGGER.warning(
            "bili.iris.checkpointers.checkpointer_functions not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)


def _create_jsonl_checkpointer(
    config: dict[str, Any] | None = None, user_id: str | None = None
) -> Any:
    """Create a local-file JSONL checkpointer (no database server required).

    Args:
        config: Dict with optional keys:
            - ``"path"``: Absolute path to the JSONL file.  Defaults to
              the ``JSONL_CHECKPOINT_PATH`` env var or
              ``~/.bili/checkpoints/aether.jsonl``.
            - ``"keep_last_n"``: Pruning limit (``-1`` = unlimited).
        user_id: Optional user identifier for thread ownership validation.

    Returns:
        A ``JSONLCheckpointSaver`` instance.
    """
    config = config or {}
    path = config.get("path")
    keep_last_n = config.get("keep_last_n", -1)
    try:
        from bili.iris.checkpointers.jsonl_checkpointer import (  # pylint: disable=import-outside-toplevel
            JSONLCheckpointSaver,
        )

        saver = JSONLCheckpointSaver(
            path=path, keep_last_n=keep_last_n, user_id=user_id
        )
        LOGGER.info(
            "Created JSONLCheckpointSaver (path=%s, keep_last_n=%d%s)",
            saver.path,
            keep_last_n,
            f", user_id={user_id}" if user_id else "",
        )
        return saver
    except ImportError:  # pragma: no cover  # ImportError only in broken installs
        LOGGER.warning(
            "bili.iris.checkpointers.jsonl_checkpointer not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)
