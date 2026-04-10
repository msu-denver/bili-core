"""Tests for checkpointer versioning and migration infrastructure.

Covers version detection, migration registry, migration path
calculation, and the VersionedCheckpointerMixin logic.
"""

import json

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
