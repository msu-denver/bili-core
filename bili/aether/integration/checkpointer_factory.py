"""Checkpointer factory — maps MASConfig.checkpoint_config to bili-core checkpointers.

All bili-core imports are lazy so this module loads without heavy
dependencies (psycopg, pymongo, etc.).
"""

import logging
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Supported type aliases → canonical names
_TYPE_ALIASES: Dict[str, str] = {
    "memory": "memory",
    "postgres": "postgres",
    "pg": "postgres",
    "mongo": "mongo",
    "mongodb": "mongo",
    "auto": "auto",
}


def create_checkpointer_from_config(
    config: Dict[str, Any], user_id: Optional[str] = None
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
    }
    return dispatch[checkpoint_type](config, user_id=user_id)


# =========================================================================
# Per-type helpers
# =========================================================================


def _create_memory_checkpointer(
    _config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    user_id: Optional[str] = None,
) -> Any:
    """Create a QueryableMemorySaver, falling back to plain MemorySaver."""
    try:
        from bili.checkpointers.memory_checkpointer import (  # pylint: disable=import-outside-toplevel
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

    LOGGER.info(
        "Created MemorySaver checkpointer " "(QueryableMemorySaver unavailable)"
    )
    return MemorySaver()


def _create_postgres_checkpointer(
    config: Dict[str, Any], user_id: Optional[str] = None
) -> Any:
    """Create a PostgreSQL checkpointer via bili-core."""
    keep_last_n = config.get("keep_last_n", 5)
    try:
        from bili.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
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
            "bili.checkpointers.pg_checkpointer not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)


def _create_mongo_checkpointer(
    config: Dict[str, Any], user_id: Optional[str] = None
) -> Any:
    """Create a MongoDB checkpointer via bili-core."""
    keep_last_n = config.get("keep_last_n", 5)
    try:
        from bili.checkpointers.mongo_checkpointer import (  # pylint: disable=import-outside-toplevel
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
            "bili.checkpointers.mongo_checkpointer not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)


def _create_auto_checkpointer(
    config: Dict[str, Any] = None, user_id: Optional[str] = None
) -> Any:
    """Auto-detect checkpointer using bili-core's ``get_checkpointer()``.

    Forwards keep_last_n and user_id parameters if specified.
    """
    config = config or {}
    keep_last_n = config.get("keep_last_n", 5)

    try:
        from bili.checkpointers.checkpointer_functions import (  # pylint: disable=import-outside-toplevel
            get_checkpointer,
        )

        # Try passing both keep_last_n and user_id if supported
        try:
            checkpointer = get_checkpointer(keep_last_n=keep_last_n, user_id=user_id)
            LOGGER.info(
                "Auto-detected checkpointer: %s (keep_last_n=%d%s)",
                type(checkpointer).__name__,
                keep_last_n,
                f", user_id={user_id}" if user_id else "",
            )
        except TypeError:
            # Fall back to keep_last_n only if user_id not supported
            try:
                checkpointer = get_checkpointer(keep_last_n=keep_last_n)
                LOGGER.info(
                    "Auto-detected checkpointer: %s (keep_last_n=%d, user_id not supported)",
                    type(checkpointer).__name__,
                    keep_last_n,
                )
            except TypeError:
                # Fall back to no args if neither supported
                checkpointer = get_checkpointer()
                LOGGER.info(
                    "Auto-detected checkpointer: %s (keep_last_n/user_id not supported)",
                    type(checkpointer).__name__,
                )
        return checkpointer
    except ImportError:
        LOGGER.warning(
            "bili.checkpointers.checkpointer_functions not available; "
            "falling back to memory"
        )
    return _create_memory_checkpointer(user_id=user_id)
