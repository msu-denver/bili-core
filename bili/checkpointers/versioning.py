"""
Module: versioning

This module provides versioning infrastructure for checkpointers to handle
migrations between different data formats when LangGraph or LangChain
libraries are upgraded.

Classes:
    - VersionedCheckpointerMixin:
      Mixin class that provides version detection and lazy migration
      capabilities for checkpointers.

Background:
    The CVE-2025-64439 security fix in langgraph-checkpoint>=3.0.0 changed
    the JsonPlusSerializer to use an allow-list for constructor-style objects,
    breaking compatibility with old checkpoints saved in "json" mode.

    This module implements lazy migration - reading old format data, converting
    on access, and writing back in new format.

Usage:
    Checkpointer implementations should inherit from this mixin to gain
    versioning and migration capabilities.

Example:
    class PruningMongoDBSaver(VersionedCheckpointerMixin, QueryableCheckpointerMixin, MongoDBSaver):
        pass
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Current format version - increment when making breaking changes
# NOTE: Set to 1 to disable migrations until LangGraph fixes dumps_metadata bug
# See: langgraph/checkpoint/mongodb/utils.py line 47
# TODO: Re-enable to 2 when LangGraph is fixed
CURRENT_FORMAT_VERSION = 1

# Migration registry: maps (checkpointer_type, from_version, to_version) -> migration function
# Migrations are applied sequentially to upgrade through multiple versions
# checkpointer_type is a string like "mongo" or "pg"
MIGRATION_REGISTRY: Dict[
    Tuple[str, int, int], Callable[[Dict[str, Any]], Dict[str, Any]]
] = {}


def register_migration(checkpointer_type: str, from_version: int, to_version: int):
    """
    Decorator to register a migration function for a specific checkpointer type.

    Args:
        checkpointer_type: Type of checkpointer (e.g., "mongo", "pg")
        from_version: Source format version
        to_version: Target format version

    Example:
        @register_migration("mongo", 1, 2)
        def migrate_v1_to_v2(document: dict) -> dict:
            # Migration logic
            return document
    """

    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        MIGRATION_REGISTRY[(checkpointer_type, from_version, to_version)] = func
        LOGGER.info(
            "Registered %s migration: v%d -> v%d",
            checkpointer_type,
            from_version,
            to_version,
        )
        return func

    return decorator


def get_migration_path(
    checkpointer_type: str, from_version: int, to_version: int
) -> List[Tuple[int, int]]:
    """
    Calculate the migration path from one version to another for a checkpointer type.

    Args:
        checkpointer_type: Type of checkpointer (e.g., "mongo", "pg")
        from_version: Starting version
        to_version: Target version

    Returns:
        List of (from, to) tuples representing migration steps

    Raises:
        ValueError: If no valid migration path exists
    """
    if from_version >= to_version:
        return []  # Already at or past target version

    # Filter migrations for this checkpointer type
    type_migrations = {
        (fv, tv): func
        for (ct, fv, tv), func in MIGRATION_REGISTRY.items()
        if ct == checkpointer_type
    }

    if not type_migrations:
        # No migrations registered for this checkpointer type - that's OK
        return []

    path = []
    current = from_version

    while current < to_version:
        # Find the next step
        next_step = None
        for fv, tv in type_migrations.keys():
            if fv == current and tv <= to_version:
                if next_step is None or tv > next_step[1]:
                    next_step = (fv, tv)

        if next_step is None:
            # No migration path found - could be intentional (no migrations needed)
            LOGGER.debug(
                "No %s migration path from v%d to v%d",
                checkpointer_type,
                current,
                to_version,
            )
            break

        path.append(next_step)
        current = next_step[1]

    return path


class VersionedCheckpointerMixin(ABC):
    """
    Abstract mixin class that provides versioning and migration capabilities.

    This mixin should be included in checkpointer classes to enable:
    - Format version tracking in checkpoint metadata
    - Lazy migration of old format data on read
    - Automatic version stamping on write

    The migration happens BEFORE LangGraph's deserialization to avoid errors
    from incompatible serialization formats.

    Subclasses must define:
    - checkpointer_type: str identifying the checkpointer (e.g., "mongo", "pg")
    """

    # Subclasses must override this to identify their type
    checkpointer_type: str = ""

    # Subclasses can override this
    format_version: int = CURRENT_FORMAT_VERSION

    @abstractmethod
    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get raw checkpoint document directly from storage.

        This bypasses LangGraph's deserialization to allow inspection
        and migration of incompatible formats.

        Args:
            thread_id: Thread ID to retrieve
            checkpoint_ns: Checkpoint namespace (usually empty string)

        Returns:
            Raw checkpoint document or None if not found
        """

    @abstractmethod
    def _replace_raw_checkpoint(
        self, thread_id: str, document: Dict[str, Any], checkpoint_ns: str = ""
    ) -> bool:
        """
        Replace raw checkpoint document in storage.

        Used to write back migrated checkpoints.

        Args:
            thread_id: Thread ID to update
            document: Migrated document to write
            checkpoint_ns: Checkpoint namespace (usually empty string)

        Returns:
            True if replacement was successful
        """

    @abstractmethod
    def _archive_checkpoint(
        self, thread_id: str, document: Dict[str, Any], error: Exception
    ) -> None:
        """
        Archive a checkpoint that failed migration.

        Subclasses should implement storage-specific archival logic.

        Args:
            thread_id: Thread ID of failed checkpoint
            document: Raw document that couldn't be migrated
            error: Exception that occurred during migration
        """

    def _get_document_version(self, document: Dict[str, Any]) -> int:
        """
        Extract format version from a checkpoint document.

        Checks multiple locations where version might be stored:
        1. metadata.format_version (preferred)
        2. Top-level format_version
        3. Default to version 1 for unversioned documents

        Args:
            document: Raw checkpoint document

        Returns:
            Format version number
        """
        import json as json_module

        # Try to import msgpack for handling msgpack-encoded metadata
        try:
            import msgpack

            msgpack_available = True
        except ImportError:
            msgpack_available = False

        # Check metadata first (preferred location)
        metadata = document.get("metadata", {})
        if isinstance(metadata, dict) and "format_version" in metadata:
            version_value = metadata["format_version"]

            # Handle v2+ format: ["json"|"msgpack", serialized_value]
            if isinstance(version_value, list) and len(version_value) == 2:
                serializer_type, serialized_value = version_value

                if serializer_type == "json":
                    try:
                        return int(json_module.loads(serialized_value))
                    except (ValueError, TypeError, json_module.JSONDecodeError):
                        # Fallback if deserialization fails
                        pass
                elif serializer_type == "msgpack":
                    try:
                        # Handle bson.Binary if present
                        if hasattr(serialized_value, "__bytes__"):
                            serialized_value = bytes(serialized_value)

                        if isinstance(serialized_value, (bytes, bytearray)):
                            # LangGraph's JsonPlusSerializer actually uses JSON encoding,
                            # despite the "msgpack" wrapper from dumps_metadata
                            # Try JSON decode first
                            try:
                                decoded_str = serialized_value.decode("utf-8")
                                return int(json_module.loads(decoded_str))
                            except (
                                UnicodeDecodeError,
                                json_module.JSONDecodeError,
                                ValueError,
                            ):
                                # If JSON fails, try msgpack (for old double-wrapped data)
                                if msgpack_available:
                                    unpacked = msgpack.unpackb(
                                        serialized_value, raw=False
                                    )
                                    # The unpacked value might be a list like ["json", "2"]
                                    if (
                                        isinstance(unpacked, list)
                                        and len(unpacked) == 2
                                    ):
                                        if unpacked[0] == "json":
                                            return int(json_module.loads(unpacked[1]))
                                    # Or it might be a direct integer
                                    elif isinstance(unpacked, int):
                                        return unpacked
                    except Exception:  # pylint: disable=broad-exception-caught
                        # Fallback if deserialization fails
                        pass

            # Handle v1 format: direct integer value
            if isinstance(version_value, int):
                return version_value

        # Check top-level
        if "format_version" in document:
            version_value = document["format_version"]
            if isinstance(version_value, int):
                return version_value

        # Default to version 1 (pre-versioning)
        return 1

    def _set_document_version(
        self, document: Dict[str, Any], version: int
    ) -> Dict[str, Any]:
        """
        Set format version in a checkpoint document.

        Args:
            document: Checkpoint document to update
            version: Version number to set

        Returns:
            Updated document
        """
        import json

        if "metadata" not in document:
            document["metadata"] = {}

        # For v2+, metadata values must be in tuple format [type, value]
        # This matches the format expected by loads_metadata/loads_typed
        if version >= 2:
            document["metadata"]["format_version"] = ["json", json.dumps(version)]
        else:
            document["metadata"]["format_version"] = version

        return document

    def _has_registered_migrations(self) -> bool:
        """
        Check if there are any migrations registered for this checkpointer type.

        Returns:
            True if migrations exist for this checkpointer type
        """
        return any(
            ct == self.checkpointer_type for ct, _, _ in MIGRATION_REGISTRY.keys()
        )

    def _needs_migration(self, document: Optional[Dict[str, Any]]) -> bool:
        """
        Check if a document needs migration.

        A document needs migration if:
        1. Its version is lower than the current format version, OR
        2. Version is 2+ but blob type is still "msgpack" (inconsistent state)

        Args:
            document: Raw checkpoint document

        Returns:
            True if migration is needed
        """
        if document is None:
            return False

        # Only consider migration if there are registered migrations for this type
        if not self._has_registered_migrations():
            return False

        # Check version
        doc_version = self._get_document_version(document)

        # If document version is HIGHER than current format version,
        # we've likely downgraded as a workaround (e.g., for LangGraph bug).
        # Don't try to migrate - just use existing checkpoint as-is.
        if doc_version > self.format_version:
            LOGGER.debug(
                "Document version %d > current format version %d. "
                "Skipping migration (likely intentional downgrade).",
                doc_version,
                self.format_version,
            )
            return False

        if doc_version < self.format_version:
            return True

        # Check for inconsistent state: v2+ metadata but msgpack blob
        # This can happen if migration was interrupted or partially applied
        # Only applies when doc_version == format_version (not when downgraded)
        if doc_version >= 2 and document.get("type") == "msgpack":
            LOGGER.warning(
                "Detected inconsistent checkpoint state: version %d but type='msgpack'. "
                "Will trigger migration to repair.",
                doc_version,
            )
            return True

        return False

    def _migrate_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a document through all necessary version upgrades.

        Args:
            document: Raw checkpoint document

        Returns:
            Migrated document at current version

        Raises:
            ValueError: If no valid migration path exists
        """
        current_version = self._get_document_version(document)
        target_version = self.format_version

        if current_version >= target_version:
            return document

        # Get migration path for this checkpointer type
        path = get_migration_path(
            self.checkpointer_type, current_version, target_version
        )

        # If no migrations registered for this checkpointer type, just update version
        if not path:
            LOGGER.debug(
                "No %s migrations needed from v%d to v%d",
                self.checkpointer_type,
                current_version,
                target_version,
            )
            document = self._set_document_version(document, target_version)
            return document

        LOGGER.info(
            "Migrating %s checkpoint from v%d to v%d. Path: %s",
            self.checkpointer_type,
            current_version,
            target_version,
            path,
        )

        # Apply migrations sequentially
        for from_v, to_v in path:
            migration_func = MIGRATION_REGISTRY.get(
                (self.checkpointer_type, from_v, to_v)
            )
            if migration_func is None:
                raise ValueError(
                    f"Missing {self.checkpointer_type} migration for v{from_v} -> v{to_v}"
                )

            try:
                document = migration_func(document)
                LOGGER.debug(
                    "Applied %s migration v%d -> v%d",
                    self.checkpointer_type,
                    from_v,
                    to_v,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                LOGGER.error(
                    "%s migration v%d -> v%d failed: %s",
                    self.checkpointer_type,
                    from_v,
                    to_v,
                    e,
                )
                raise

        # Update version stamp
        document = self._set_document_version(document, target_version)
        return document

    def migrate_checkpoint_if_needed(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> bool:
        """
        Check and migrate a checkpoint if needed.

        This should be called BEFORE LangGraph's get_tuple() to ensure
        the data is in a compatible format.

        Args:
            thread_id: Thread ID to check/migrate
            checkpoint_ns: Checkpoint namespace

        Returns:
            True if migration was performed, False if not needed
        """
        raw_doc = self._get_raw_checkpoint(thread_id, checkpoint_ns)

        if not self._needs_migration(raw_doc):
            return False

        try:
            migrated = self._migrate_document(raw_doc)
            success = self._replace_raw_checkpoint(thread_id, migrated, checkpoint_ns)

            if success:
                LOGGER.info("Successfully migrated checkpoint for thread %s", thread_id)
            else:
                LOGGER.warning(
                    "Migration completed but failed to save for thread %s", thread_id
                )

            return success

        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to migrate checkpoint for thread %s: %s", thread_id, e)

            # Archive the incompatible checkpoint
            self._archive_checkpoint(thread_id, raw_doc, e)

            # Re-raise to let caller decide how to handle
            raise

    def is_format_incompatibility_error(self, error: Exception) -> bool:
        """
        Check if an exception indicates a format incompatibility.

        Subclasses can override to add storage-specific error detection.

        Args:
            error: Exception to check

        Returns:
            True if this appears to be a format incompatibility
        """
        error_str = str(error).lower()
        incompatibility_indicators = [
            "not json serializable",
            "object of type",
            "cannot deserialize",
            "unexpected type",
            "__class__",
            "constructor",
        ]
        return any(indicator in error_str for indicator in incompatibility_indicators)
