"""Tests for checkpointer versioning and migration infrastructure.

Covers version detection, migration registry, migration path
calculation, and the VersionedCheckpointerMixin logic.
"""

import json
from unittest.mock import patch

import pytest

from bili.iris.checkpointers.versioning import (
    CURRENT_FORMAT_VERSION,
    MIGRATION_REGISTRY,
    VersionedCheckpointerMixin,
    get_migration_path,
    register_migration,
)


class TestCurrentFormatVersion:
    """Tests for CURRENT_FORMAT_VERSION constant."""

    def test_is_positive_int(self):
        """Current format version is a positive integer."""
        assert isinstance(CURRENT_FORMAT_VERSION, int)
        assert CURRENT_FORMAT_VERSION >= 1

    def test_register_migration_function(self):
        """Decorator registers function in MIGRATION_REGISTRY."""
        key = ("test_type_reg", 100, 101)
        try:

            @register_migration("test_type_reg", 100, 101)
            def migrate_test(doc):
                """Test migration."""
                return doc

            assert key in MIGRATION_REGISTRY
            assert MIGRATION_REGISTRY[key] is migrate_test
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_decorator_returns_original_function(self):
        """Decorated function is still callable."""
        key = ("test_type_ret", 200, 201)
        try:

            @register_migration("test_type_ret", 200, 201)
            def migrate_ret(doc):
                """Return doc unchanged."""
                return doc

            result = migrate_ret({"key": "val"})
            assert result == {"key": "val"}
        finally:
            MIGRATION_REGISTRY.pop(key, None)


class TestGetMigrationPath:
    """Tests for get_migration_path function."""

    def test_no_migration_needed(self):
        """Returns empty path when already at target."""
        path = get_migration_path("pg", 2, 2)
        assert not path

    def test_past_target_returns_empty(self):
        """Returns empty path when past target version."""
        path = get_migration_path("pg", 3, 2)
        assert not path

    def test_finds_registered_path(self):
        """Finds path through registered migrations."""
        key1 = ("test_path", 1, 2)
        key2 = ("test_path", 2, 3)
        try:
            MIGRATION_REGISTRY[key1] = lambda d: d
            MIGRATION_REGISTRY[key2] = lambda d: d

            path = get_migration_path("test_path", 1, 3)
            assert path == [(1, 2), (2, 3)]
        finally:
            MIGRATION_REGISTRY.pop(key1, None)
            MIGRATION_REGISTRY.pop(key2, None)

    def test_no_migrations_for_type(self):
        """Returns empty path when no migrations for type."""
        path = get_migration_path("nonexistent_type", 1, 2)
        assert not path

    def test_partial_path(self):
        """Returns partial path when gap exists."""
        key = ("test_partial", 1, 2)
        try:
            MIGRATION_REGISTRY[key] = lambda d: d
            path = get_migration_path("test_partial", 1, 5)
            assert path == [(1, 2)]
        finally:
            MIGRATION_REGISTRY.pop(key, None)


class ConcreteVersionedCheckpointer(VersionedCheckpointerMixin):
    """Concrete test implementation of the mixin.

    Exposes protected mixin methods as public for testing.
    """

    checkpointer_type = "test_concrete"
    format_version = CURRENT_FORMAT_VERSION

    def __init__(self):
        """Initialize with empty storage."""
        self.storage = {}

    def _get_raw_checkpoint(self, thread_id, checkpoint_ns=""):
        """Return stored document or None."""
        return self.storage.get(thread_id)

    def _replace_raw_checkpoint(self, thread_id, document, checkpoint_ns=""):
        """Store migrated document."""
        self.storage[thread_id] = document
        return True

    def _archive_checkpoint(self, thread_id, document, error):
        """Archive failed checkpoint."""
        self.storage[f"archive_{thread_id}"] = {
            "document": document,
            "error": str(error),
        }

    def get_document_version(self, doc):
        """Public proxy for _get_document_version."""
        return self._get_document_version(doc)

    def set_document_version(self, doc, version):
        """Public proxy for _set_document_version."""
        return self._set_document_version(doc, version)

    def needs_migration(self, doc):
        """Public proxy for _needs_migration."""
        return self._needs_migration(doc)

    def has_registered_migrations(self):
        """Public proxy for _has_registered_migrations."""
        return self._has_registered_migrations()

    def do_migrate_document(self, doc):
        """Public proxy for _migrate_document."""
        return self._migrate_document(doc)


class TestVersionedCheckpointerMixin:
    """Tests for VersionedCheckpointerMixin methods."""

    def test_get_document_version_from_metadata(self):
        """Extracts version from metadata.format_version."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"metadata": {"format_version": 2}}
        assert vc.get_document_version(doc) == 2

    def test_get_document_version_from_top_level(self):
        """Extracts version from top-level format_version."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"format_version": 3}
        assert vc.get_document_version(doc) == 3

    def test_get_document_version_default_v1(self):
        """Returns 1 for unversioned documents."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"some_key": "value"}
        assert vc.get_document_version(doc) == 1

    def test_get_version_from_json_tuple(self):
        """Handles v2+ format with ['json', serialized_value]."""
        vc = ConcreteVersionedCheckpointer()
        doc = {
            "metadata": {
                "format_version": ["json", json.dumps(2)],
            }
        }
        assert vc.get_document_version(doc) == 2

    def test_set_document_version_v2_format(self):
        """Sets version in v2+ tuple format."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"metadata": {}}
        result = vc.set_document_version(doc, 2)
        fv = result["metadata"]["format_version"]
        assert fv[0] == "json"
        assert json.loads(fv[1]) == 2

    def test_set_document_version_v1_format(self):
        """Sets version as direct int for v1."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"metadata": {}}
        result = vc.set_document_version(doc, 1)
        assert result["metadata"]["format_version"] == 1

    def test_set_document_version_creates_metadata(self):
        """Creates metadata dict if missing."""
        vc = ConcreteVersionedCheckpointer()
        doc = {}
        result = vc.set_document_version(doc, 2)
        assert "metadata" in result

    def test_needs_migration_none_document(self):
        """Returns False for None document."""
        vc = ConcreteVersionedCheckpointer()
        assert vc.needs_migration(None) is False

    def test_needs_migration_no_registered_migrations(self):
        """Returns False when no migrations registered."""
        vc = ConcreteVersionedCheckpointer()
        doc = {"metadata": {"format_version": 1}}
        assert vc.needs_migration(doc) is False

    def test_needs_migration_with_registered_migration(self):
        """Returns True when migration exists and version is old."""
        key = ("test_concrete", 1, 2)
        try:
            MIGRATION_REGISTRY[key] = lambda d: d
            vc = ConcreteVersionedCheckpointer()
            doc = {"metadata": {"format_version": 1}}
            assert vc.needs_migration(doc) is True
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_needs_migration_current_version(self):
        """Returns False when document is at current version."""
        key = ("test_concrete", 1, 2)
        try:
            MIGRATION_REGISTRY[key] = lambda d: d
            vc = ConcreteVersionedCheckpointer()
            doc = {
                "metadata": {
                    "format_version": CURRENT_FORMAT_VERSION,
                }
            }
            assert vc.needs_migration(doc) is False
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_has_registered_migrations_false(self):
        """Returns False when no migrations for this type."""
        vc = ConcreteVersionedCheckpointer()
        assert vc.has_registered_migrations() is False

    def test_has_registered_migrations_true(self):
        """Returns True when migration exists for this type."""
        key = ("test_concrete", 1, 2)
        try:
            MIGRATION_REGISTRY[key] = lambda d: d
            vc = ConcreteVersionedCheckpointer()
            assert vc.has_registered_migrations() is True
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_migrate_document_no_change_needed(self):
        """Returns document unchanged if at current version."""
        vc = ConcreteVersionedCheckpointer()
        doc = {
            "metadata": {
                "format_version": CURRENT_FORMAT_VERSION,
            }
        }
        result = vc.do_migrate_document(doc)
        assert result is doc

    def test_migrate_document_applies_migration(self):
        """Applies registered migration function."""
        key = ("test_concrete", 1, 2)
        try:

            def add_marker(doc):
                """Add migrated marker."""
                doc["migrated"] = True
                return doc

            MIGRATION_REGISTRY[key] = add_marker
            vc = ConcreteVersionedCheckpointer()
            vc.format_version = 2
            doc = {"metadata": {"format_version": 1}}
            result = vc.do_migrate_document(doc)
            assert result["migrated"] is True
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_migrate_checkpoint_if_needed_no_op(self):
        """Returns False when no migration needed."""
        vc = ConcreteVersionedCheckpointer()
        vc.storage["t1"] = {
            "metadata": {
                "format_version": CURRENT_FORMAT_VERSION,
            }
        }
        assert vc.migrate_checkpoint_if_needed("t1") is False

    def test_is_format_incompatibility_error(self):
        """Detects format incompatibility errors."""
        vc = ConcreteVersionedCheckpointer()
        assert vc.is_format_incompatibility_error(ValueError("not json serializable"))
        assert vc.is_format_incompatibility_error(TypeError("object of type bytes"))
        assert not vc.is_format_incompatibility_error(ValueError("some other error"))


class TestGetDocumentVersionMsgpack:
    """Tests for the msgpack serializer branch of _get_document_version."""

    def test_msgpack_wrapper_with_json_bytes(self):
        """A ['msgpack', json-bytes] wrapper is decoded as JSON first."""
        vc = ConcreteVersionedCheckpointer()
        doc = {
            "metadata": {
                "format_version": ["msgpack", b"2"],
            }
        }
        # LangGraph's serializer actually JSON-encodes despite the msgpack tag.
        assert vc.get_document_version(doc) == 2

    def test_msgpack_wrapper_with_bson_binary(self):
        """A bson.Binary serialized value is converted via __bytes__ then decoded."""
        import bson

        vc = ConcreteVersionedCheckpointer()
        doc = {
            "metadata": {
                "format_version": ["msgpack", bson.Binary(b"3")],
            }
        }
        assert vc.get_document_version(doc) == 3

    def test_msgpack_wrapper_true_msgpack_payload(self):
        """When JSON decode fails, true-msgpack bytes are unpacked to a list."""
        import msgpack

        vc = ConcreteVersionedCheckpointer()
        # Pack the value ["json", "2"] which is not valid UTF-8 JSON on its own.
        packed = msgpack.packb(["json", "2"])
        doc = {
            "metadata": {
                "format_version": ["msgpack", packed],
            }
        }
        assert vc.get_document_version(doc) == 2

    def test_msgpack_wrapper_direct_int_payload(self):
        """A msgpack payload that unpacks to a bare int returns that int."""
        import msgpack

        vc = ConcreteVersionedCheckpointer()
        packed = msgpack.packb(7)
        doc = {
            "metadata": {
                "format_version": ["msgpack", packed],
            }
        }
        assert vc.get_document_version(doc) == 7

    def test_msgpack_unavailable_falls_back_to_default(self):
        """When msgpack import fails and JSON decode fails, defaults to v1."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "msgpack":
                raise ImportError("no msgpack")
            return real_import(name, *args, **kwargs)

        vc = ConcreteVersionedCheckpointer()
        # Non-UTF8 bytes force the JSON decode to fail, then msgpack is unavailable.
        doc = {
            "metadata": {
                "format_version": ["msgpack", b"\xff\xfe"],
            }
        }
        with patch("builtins.__import__", side_effect=fake_import):
            assert vc.get_document_version(doc) == 1

    def test_msgpack_outer_exception_falls_back_to_default(self):
        """A msgpack unpack failure is swallowed and defaults to v1."""
        vc = ConcreteVersionedCheckpointer()
        # Bytes that fail UTF-8 JSON decode and also fail msgpack.unpackb,
        # exercising the broad-except fallback around the msgpack branch.
        doc = {
            "metadata": {
                "format_version": ["msgpack", b"\xc1"],
            }
        }
        assert vc.get_document_version(doc) == 1

    def test_json_wrapper_decode_failure_falls_through(self):
        """Invalid JSON in a ['json', ...] wrapper falls through to default v1."""
        vc = ConcreteVersionedCheckpointer()
        doc = {
            "metadata": {
                "format_version": ["json", "{not valid json"],
            }
        }
        assert vc.get_document_version(doc) == 1


class TestMigrateDocumentPaths:
    """Tests for _migrate_document path resolution and error handling."""

    def test_no_path_just_stamps_version(self):
        """With no registered migration path, the version stamp is applied."""
        # checkpointer_type with no registered migrations and an old version.
        vc = ConcreteVersionedCheckpointer()
        vc.format_version = 2
        doc = {"metadata": {"format_version": 1}}
        result = vc.do_migrate_document(doc)
        fv = result["metadata"]["format_version"]
        assert fv == ["json", json.dumps(2)]

    def test_missing_migration_raises_value_error(self):
        """A path step with no registered function raises ValueError."""
        # Register only the path-discovery key to produce a path, then remove the
        # function so the lookup inside the apply loop returns None.
        key = ("test_concrete", 1, 2)
        vc = ConcreteVersionedCheckpointer()
        vc.format_version = 2
        doc = {"metadata": {"format_version": 1}}

        # Patch get_migration_path to return a step that has no registered func.
        with patch(
            "bili.iris.checkpointers.versioning.get_migration_path",
            return_value=[(1, 2)],
        ):
            MIGRATION_REGISTRY.pop(key, None)
            with pytest.raises(ValueError, match="Missing test_concrete migration"):
                vc.do_migrate_document(doc)

    def test_migration_function_exception_propagates(self):
        """An exception raised by a migration function propagates out."""
        key = ("test_concrete", 1, 2)
        try:

            def boom(_doc):
                """Raise to simulate a failing migration."""
                raise RuntimeError("migration exploded")

            MIGRATION_REGISTRY[key] = boom
            vc = ConcreteVersionedCheckpointer()
            vc.format_version = 2
            doc = {"metadata": {"format_version": 1}}
            with pytest.raises(RuntimeError, match="migration exploded"):
                vc.do_migrate_document(doc)
        finally:
            MIGRATION_REGISTRY.pop(key, None)


class TestMigrateCheckpointIfNeeded:
    """Tests for migrate_checkpoint_if_needed orchestration."""

    def test_successful_migration_returns_true_and_writes(self):
        """A needed migration runs, writes back, and reports True."""
        key = ("test_concrete", 1, 2)
        try:

            def add_marker(doc):
                """Mark the document as migrated."""
                doc["migrated"] = True
                return doc

            MIGRATION_REGISTRY[key] = add_marker
            vc = ConcreteVersionedCheckpointer()
            vc.format_version = 2
            vc.storage["t1"] = {"metadata": {"format_version": 1}}
            assert vc.migrate_checkpoint_if_needed("t1") is True
            assert vc.storage["t1"]["migrated"] is True
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_write_failure_returns_false(self):
        """When the write-back fails, the method returns False."""
        key = ("test_concrete", 1, 2)
        try:
            MIGRATION_REGISTRY[key] = lambda d: d
            vc = ConcreteVersionedCheckpointer()
            vc.format_version = 2
            vc.storage["t1"] = {"metadata": {"format_version": 1}}
            with patch.object(vc, "_replace_raw_checkpoint", return_value=False):
                assert vc.migrate_checkpoint_if_needed("t1") is False
        finally:
            MIGRATION_REGISTRY.pop(key, None)

    def test_failed_migration_archives_and_reraises(self):
        """When migration raises, the checkpoint is archived and the error re-raises."""
        key = ("test_concrete", 1, 2)
        try:

            def boom(_doc):
                """Fail the migration."""
                raise RuntimeError("cannot migrate")

            MIGRATION_REGISTRY[key] = boom
            vc = ConcreteVersionedCheckpointer()
            vc.format_version = 2
            vc.storage["t1"] = {"metadata": {"format_version": 1}}
            with pytest.raises(RuntimeError, match="cannot migrate"):
                vc.migrate_checkpoint_if_needed("t1")
            # The archive hook should have captured the failure.
            assert "archive_t1" in vc.storage
            assert "cannot migrate" in vc.storage["archive_t1"]["error"]
        finally:
            MIGRATION_REGISTRY.pop(key, None)
