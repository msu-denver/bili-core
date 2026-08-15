"""Tests for per-provider structured-output schema adaptation.

The seam emits a valid JSON schema; each cloud provider accepts a stricter
subset.  These verify the two adapters turn an ordinary nested schema (optional
properties, ``$defs``/``$ref``, ``additionalProperties``) into each subset:

- OpenAI strict: every object all-required, ``additionalProperties: false``,
  optional properties made nullable.
- Gemini: no ``$ref``/``$defs``/``additionalProperties``; references inlined.
"""

import copy
import sys

import pytest
from pydantic import BaseModel

from bili.iris.providers.structured_output import (
    adapt_schema_for_gemini,
    adapt_schema_for_openai_strict,
    gemini_response_schema,
)

# A schema with the constructs that trip the cloud subsets: an optional
# property (not in required), $defs/$ref, and additionalProperties.
_SCHEMA = {
    "title": "Doc",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "optional_flag": {"type": "boolean"},
        "meta": {"$ref": "#/$defs/Meta"},
        "segments": {"type": "array", "items": {"$ref": "#/$defs/Segment"}},
    },
    "required": ["title", "meta", "segments"],
    "additionalProperties": False,
    "$defs": {
        "Meta": {
            "type": "object",
            "properties": {"author": {"type": "string"}, "note": {"type": "string"}},
            "required": ["author"],
            "additionalProperties": False,
        },
        "Segment": {
            "type": "object",
            "properties": {"intent": {"type": "string"}, "body": {"type": "string"}},
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}

# A self-referential schema (a node whose children are nodes) to exercise the
# Gemini inliner's cycle guard.
_RECURSIVE = {
    "type": "object",
    "properties": {
        "val": {"type": "string"},
        "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
    },
    "$defs": {
        "Node": {
            "type": "object",
            "properties": {
                "val": {"type": "string"},
                "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            },
        }
    },
}


def _all_objects(node):
    """Yield every object-typed schema node in *node* (recursively)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            yield node
        for value in node.values():
            yield from _all_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_objects(item)


def _has_key(node, key):
    """Return True if any dict anywhere in *node* has *key*.

    Checks for a schema KEY (e.g. ``$ref``), not a substring: a schema's
    ``description`` may legitimately contain the text ``$ref``/``$defs``.
    """
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_has_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(_has_key(item, key) for item in node)
    return False


class TestAdaptOpenAIStrict:
    """Every object becomes all-required + additionalProperties:false; optional
    properties become nullable."""

    def test_every_object_lists_all_properties_required(self):
        """Strict requires required == all property keys, on every object."""
        adapted = adapt_schema_for_openai_strict(_SCHEMA)
        for obj in _all_objects(adapted):
            assert set(obj["required"]) == set(obj["properties"])

    def test_every_object_forbids_additional_properties(self):
        """Strict requires additionalProperties:false on every object."""
        adapted = adapt_schema_for_openai_strict(_SCHEMA)
        for obj in _all_objects(adapted):
            assert obj["additionalProperties"] is False

    def test_optional_property_becomes_nullable(self):
        """A previously-optional property is made nullable, staying optional."""
        adapted = adapt_schema_for_openai_strict(_SCHEMA)
        assert adapted["properties"]["optional_flag"] == {
            "anyOf": [{"type": "boolean"}, {"type": "null"}]
        }

    def test_defs_objects_are_adapted(self):
        """Objects inside $defs are adapted too (OpenAI resolves $ref to them)."""
        adapted = adapt_schema_for_openai_strict(_SCHEMA)
        meta = adapted["$defs"]["Meta"]
        assert set(meta["required"]) == {"author", "note"}
        assert meta["properties"]["note"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }

    def test_required_property_is_not_wrapped(self):
        """A required property keeps its original schema (not wrapped nullable)."""
        adapted = adapt_schema_for_openai_strict(_SCHEMA)
        assert adapted["properties"]["title"] == {"type": "string"}

    def test_input_is_not_mutated(self):
        """The adapter does not mutate its input."""
        original = copy.deepcopy(_SCHEMA)
        adapt_schema_for_openai_strict(_SCHEMA)
        assert _SCHEMA == original


class TestAdaptGemini:
    """References are inlined; $defs / $ref / additionalProperties are gone."""

    def test_no_ref_defs_or_additional_properties(self):
        """The adapted schema contains none of the rejected keys."""
        adapted = adapt_schema_for_gemini(_SCHEMA)
        assert not _has_key(adapted, "$ref")
        assert not _has_key(adapted, "$defs")
        assert not _has_key(adapted, "additionalProperties")

    def test_ref_is_inlined(self):
        """A $ref is replaced with the referenced definition's content."""
        adapted = adapt_schema_for_gemini(_SCHEMA)
        assert adapted["properties"]["meta"]["properties"]["author"] == {
            "type": "string"
        }

    def test_recursive_schema_terminates_with_permissive_fallback(self):
        """A self-referential schema does not hang; the cycle becomes a
        permissive object."""
        adapted = adapt_schema_for_gemini(_RECURSIVE)
        assert not _has_key(adapted, "$ref")
        # The recursive descent bottoms out at a permissive object.
        assert adapted["properties"]["kids"]["items"]["properties"]["kids"][
            "items"
        ] == {"type": "object"}

    def test_ref_with_sibling_keys_merges(self):
        """A $ref carrying sibling keys inlines the target and keeps the siblings
        (JSON Schema 2020-12 allows keywords alongside $ref)."""
        schema = {
            "type": "object",
            "properties": {
                "meta": {"$ref": "#/$defs/Meta", "description": "the meta block"},
            },
            "$defs": {
                "Meta": {
                    "type": "object",
                    "properties": {"author": {"type": "string"}},
                }
            },
        }
        adapted = adapt_schema_for_gemini(schema)
        meta = adapted["properties"]["meta"]
        assert meta["description"] == "the meta block"
        assert meta["properties"]["author"] == {"type": "string"}
        assert not _has_key(adapted, "$ref")

    def test_input_is_not_mutated(self):
        """The adapter does not mutate its input."""
        original = copy.deepcopy(_SCHEMA)
        adapt_schema_for_gemini(_SCHEMA)
        assert _SCHEMA == original


class TestGeminiResponseSchema:
    """gemini_response_schema normalizes then adapts."""

    def test_normalizes_pydantic_and_adapts(self):
        """A Pydantic class is converted and adapted within the subset."""

        class Inner(BaseModel):
            """A nested model, which produces a reference in the schema."""

            name: str

        class Outer(BaseModel):
            """An outer model that references the nested one."""

            title: str
            inner: Inner

        adapted = gemini_response_schema(Outer)
        assert not _has_key(adapted, "$ref")
        assert not _has_key(adapted, "$defs")
        # The referenced model is inlined (a pydantic-added title may remain).
        assert adapted["properties"]["inner"]["properties"]["name"]["type"] == "string"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
