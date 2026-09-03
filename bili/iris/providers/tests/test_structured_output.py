"""Tests for schema-constrained structured output across providers.

Covers:
- ``normalize_schema``: dict passthrough, Pydantic class conversion, and
  rejection of everything else (including Pydantic *instances*).
- ``openai_response_format``: payload shape, ``strict: true``, name
  derivation from the schema title, sanitization, and truncation.
- Support registry: ``supports_structured_output``,
  ``structured_output_providers``, ``register_structured_output_provider``,
  and the fail-fast ``require_structured_output_support``.
- ``parse_structured_content``: plain JSON, fenced JSON, parse failures,
  schema-validation failures, and the schema=None (parse-only) path.
- Provider wiring, with mocked SDK modules: Ollama (``format``), OpenAI and
  Azure OpenAI (``.bind(response_format=...)``), Vertex and Google GenAI
  (``response_schema`` + ``response_mime_type``), including the
  Vertex-native-vs-cross-provider conflict error.
- ``load_model``: the fail-fast gate for unsupported providers, and full
  routing of ``structured_output_schema`` through both the registry path
  (local_ollama) and the built-in dispatch (remote_openai, remote_azure_openai,
  remote_google_vertex).
"""

# pylint: disable=too-few-public-methods,duplicate-code

import json
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers import structured_output as so_module
from bili.iris.providers.azure_openai_provider import AzureOpenAIProvider
from bili.iris.providers.google_genai_provider import GoogleGenAIProvider
from bili.iris.providers.ollama_provider import OllamaProvider
from bili.iris.providers.openai_provider import OpenAIProvider
from bili.iris.providers.structured_output import (
    StructuredOutputError,
    StructuredOutputParseError,
    StructuredOutputValidationError,
    normalize_schema,
    openai_response_format,
    parse_structured_content,
    register_structured_output_provider,
    require_structured_output_support,
    structured_output_providers,
    supports_structured_output,
)
from bili.iris.providers.vertex_provider import VertexAIProvider

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

#: Nested schema shaped like the failure class this capability exists to
#: prevent: an array of objects where a bare string is the tempting-but-
#: invalid output.
NESTED_SCHEMA = {
    "title": "Outline Document",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "bullets"],
            },
        },
    },
    "required": ["title", "sections"],
}


class OutlineModel(BaseModel):
    """Minimal Pydantic model for normalize_schema conversion tests."""

    title: str
    count: int


@contextmanager
def _mock_module(module_name: str, **attrs):
    """Temporarily inject a fake module into sys.modules for lazy-import tests."""
    mod = ModuleType(module_name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    already_present = sys.modules.get(module_name)
    sys.modules[module_name] = mod
    try:
        yield mod
    finally:
        if already_present is not None:
            sys.modules[module_name] = already_present
        else:
            sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# normalize_schema
# ---------------------------------------------------------------------------


class TestNormalizeSchema:
    """Verify schema normalization accepts dicts and Pydantic classes only."""

    def test_dict_passthrough(self):
        """Verify a dict schema is returned unchanged (same object)."""
        assert normalize_schema(NESTED_SCHEMA) is NESTED_SCHEMA

    def test_pydantic_class_converts_to_json_schema(self):
        """Verify a Pydantic BaseModel subclass converts via model_json_schema."""
        schema = normalize_schema(OutlineModel)
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"title", "count"}
        assert schema == OutlineModel.model_json_schema()

    def test_pydantic_instance_rejected(self):
        """Verify a Pydantic *instance* raises TypeError (pass the class)."""
        with pytest.raises(TypeError, match="dict or a Pydantic"):
            normalize_schema(OutlineModel(title="x", count=1))

    @pytest.mark.parametrize("bad", ["{}", 42, None, ["a"], object()])
    def test_non_schema_inputs_rejected(self, bad):
        """Verify non-dict, non-model inputs raise TypeError."""
        with pytest.raises(TypeError, match="structured_output_schema"):
            normalize_schema(bad)


# ---------------------------------------------------------------------------
# openai_response_format
# ---------------------------------------------------------------------------


class TestOpenAIResponseFormat:
    """Verify the OpenAI response_format payload builder."""

    def test_payload_shape_and_strict(self):
        """Verify the json_schema payload embeds the adapted schema with strict."""
        payload = openai_response_format(NESTED_SCHEMA)
        assert payload["type"] == "json_schema"
        assert payload["json_schema"]["strict"] is True
        # The schema is adapted to OpenAI's strict subset before embedding.
        schema = payload["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    def test_name_derived_from_title_is_sanitized(self):
        """Verify the schema title becomes the name with invalid chars replaced."""
        payload = openai_response_format(NESTED_SCHEMA)
        assert payload["json_schema"]["name"] == "Outline_Document"

    def test_default_name_when_title_absent(self):
        """Verify the fallback name applies for title-less schemas."""
        payload = openai_response_format({"type": "object"})
        assert payload["json_schema"]["name"] == "structured_output"

    def test_explicit_name_overrides_title(self):
        """Verify an explicit name wins over the schema title."""
        payload = openai_response_format(NESTED_SCHEMA, name="my_doc")
        assert payload["json_schema"]["name"] == "my_doc"

    def test_name_truncated_to_64_chars(self):
        """Verify names longer than OpenAI's 64-char limit are truncated."""
        payload = openai_response_format({"type": "object"}, name="x" * 100)
        assert len(payload["json_schema"]["name"]) == 64

    def test_fully_invalid_name_falls_back_to_default(self):
        """Verify a name of only invalid chars sanitizes to underscores, not empty."""
        payload = openai_response_format({"type": "object"}, name="!!!")
        assert payload["json_schema"]["name"] == "___"


# ---------------------------------------------------------------------------
# Support registry
# ---------------------------------------------------------------------------


class TestSupportRegistry:
    """Verify the provider support map and its extension hook."""

    @pytest.mark.parametrize(
        "provider",
        [
            "local_ollama",
            "remote_openai",
            "remote_azure_openai",
            "remote_google_vertex",
            "remote_google_genai",
        ],
    )
    def test_supported_providers(self, provider):
        """Verify each wired provider reports support."""
        assert supports_structured_output(provider) is True
        # Must not raise.
        require_structured_output_support(provider)

    @pytest.mark.parametrize(
        "provider",
        [
            "remote_anthropic",
            "remote_aws_bedrock",
            "remote_mistral",
            "local_llamacpp",
            "local_huggingface",
            "cli",
            "cli_claude_code",
            "unknown_provider",
        ],
    )
    def test_unsupported_providers(self, provider):
        """Verify unwired providers report no support and fail fast."""
        assert supports_structured_output(provider) is False
        with pytest.raises(ValueError, match=provider):
            require_structured_output_support(provider)

    def test_require_error_lists_supported_types(self):
        """Verify the fail-fast error names the supported provider types."""
        with pytest.raises(ValueError, match="local_ollama"):
            require_structured_output_support("cli")

    def test_register_external_provider(self):
        """Verify external providers can declare support at startup."""
        provider_type = "remote_test_structured"
        assert not supports_structured_output(provider_type)
        register_structured_output_provider(provider_type)
        try:
            assert supports_structured_output(provider_type)
            require_structured_output_support(provider_type)
            assert provider_type in structured_output_providers()
        finally:
            # pylint: disable=protected-access
            so_module._STRUCTURED_OUTPUT_PROVIDERS.discard(provider_type)

    def test_structured_output_providers_is_frozen_snapshot(self):
        """Verify the public view is a frozenset (mutating it is impossible)."""
        assert isinstance(structured_output_providers(), frozenset)


# ---------------------------------------------------------------------------
# parse_structured_content
# ---------------------------------------------------------------------------


class TestParseStructuredContent:
    """Verify parsing and post-hoc validation of structured content."""

    VALID_DOC = '{"title": "T", "sections": [{"heading": "H", "bullets": ["a", "b"]}]}'

    def test_plain_json_parses_and_validates(self):
        """Verify schema-valid JSON parses to a dict."""
        doc = parse_structured_content(self.VALID_DOC, schema=NESTED_SCHEMA)
        assert doc["title"] == "T"
        assert doc["sections"][0]["bullets"] == ["a", "b"]

    def test_fenced_json_parses(self):
        """Verify markdown-fenced JSON (degraded/unconstrained mode) parses."""
        content = f"Here you go:\n```json\n{self.VALID_DOC}\n```\nDone."
        doc = parse_structured_content(content, schema=NESTED_SCHEMA)
        assert doc["title"] == "T"

    def test_bare_fence_without_language_tag_parses(self):
        """Verify a fence with no language tag also parses."""
        content = f"```\n{self.VALID_DOC}\n```"
        doc = parse_structured_content(content)
        assert doc["title"] == "T"

    def test_think_preamble_json_parses(self):
        """Verify a reasoning model's <think>...</think> preamble is stripped."""
        content = f"<think>Let me reason about this.</think>{self.VALID_DOC}"
        doc = parse_structured_content(content, schema=NESTED_SCHEMA)
        assert doc["title"] == "T"

    def test_multiline_think_preamble_parses(self):
        """Verify a multi-line preamble in any casing is stripped (DOTALL)."""
        content = f"<Think>\nstep one\nstep two\n</Think>\n{self.VALID_DOC}"
        doc = parse_structured_content(content)
        assert doc["title"] == "T"

    def test_think_preamble_then_fenced_json_parses(self):
        """Verify a preamble followed by a markdown fence parses (composition)."""
        content = f"<think>reason</think>\n```json\n{self.VALID_DOC}\n```"
        doc = parse_structured_content(content, schema=NESTED_SCHEMA)
        assert doc["title"] == "T"

    def test_think_preamble_without_strip_would_not_parse(self):
        """The raw <think>-prefixed content is not itself valid JSON, so the
        strip candidate is load-bearing (without it, this raises)."""
        content = f"<think>reason</think>{self.VALID_DOC}"
        # The raw text is not valid JSON; parsing only succeeds via the stripped
        # candidate.  Sanity-check the precondition so this test cannot silently
        # pass if the preamble ever became parseable on its own.
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(content)
        assert parse_structured_content(content)["title"] == "T"

    def test_invalid_json_raises_parse_error(self):
        """Verify non-JSON content raises StructuredOutputParseError."""
        with pytest.raises(StructuredOutputParseError, match="not valid JSON"):
            parse_structured_content("definitely: not json\n---\nmore", schema=None)

    def test_empty_content_raises_parse_error(self):
        """Verify empty/None-ish content raises StructuredOutputParseError."""
        with pytest.raises(StructuredOutputParseError):
            parse_structured_content("")

    def test_schema_violation_raises_validation_error(self):
        """Verify schema-invalid JSON raises StructuredOutputValidationError.

        The document gives a string where the schema requires a list of
        objects: the exact shape of the motivating failure.
        """
        bad = '{"title": "T", "sections": "just a string"}'
        with pytest.raises(StructuredOutputValidationError, match="conform"):
            parse_structured_content(bad, schema=NESTED_SCHEMA)

    def test_missing_required_key_raises_validation_error(self):
        """Verify a missing required property fails validation."""
        with pytest.raises(StructuredOutputValidationError):
            parse_structured_content('{"title": "T"}', schema=NESTED_SCHEMA)

    def test_schema_none_skips_validation(self):
        """Verify schema=None parses without validating."""
        doc = parse_structured_content('{"anything": 1}')
        assert doc == {"anything": 1}

    def test_errors_are_structured_output_errors(self):
        """Verify both error types share the StructuredOutputError base."""
        assert issubclass(StructuredOutputParseError, StructuredOutputError)
        assert issubclass(StructuredOutputValidationError, StructuredOutputError)
        assert issubclass(StructuredOutputError, ValueError)

    def test_non_dict_json_validates_against_schema(self):
        """Verify a JSON array parses and validates when the schema allows it."""
        doc = parse_structured_content(
            "[1, 2, 3]", schema={"type": "array", "items": {"type": "integer"}}
        )
        assert doc == [1, 2, 3]


# ---------------------------------------------------------------------------
# Provider wiring — Ollama
# ---------------------------------------------------------------------------


class TestOllamaStructuredOutput:
    """Verify OllamaProvider forwards the schema as ChatOllama's format."""

    def test_schema_forwarded_as_format(self):
        """Verify the schema dict lands on ChatOllama's format parameter."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(
                model_name="qwen3:8b", structured_output_schema=NESTED_SCHEMA
            )
        assert mock_cls.call_args[1]["format"] is NESTED_SCHEMA

    def test_pydantic_class_normalized_before_forwarding(self):
        """Verify a Pydantic class converts to a schema dict for format."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(
                model_name="qwen3:8b", structured_output_schema=OutlineModel
            )
        assert mock_cls.call_args[1]["format"] == OutlineModel.model_json_schema()

    def test_format_absent_without_schema(self):
        """Verify format is not passed when no schema is requested."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3:8b")
        assert "format" not in mock_cls.call_args[1]


# ---------------------------------------------------------------------------
# Provider wiring — OpenAI / Azure OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIStructuredOutput:
    """Verify the OpenAI-family providers bind response_format."""

    def test_openai_binds_response_format(self):
        """Verify OpenAIProvider returns the bound model with the payload."""
        mock_cls = MagicMock()
        instance = mock_cls.return_value
        with _mock_module("langchain_openai", ChatOpenAI=mock_cls):
            result = OpenAIProvider().load(
                model_name="gpt-4o", structured_output_schema=NESTED_SCHEMA
            )
        bind_kwargs = instance.bind.call_args[1]
        assert bind_kwargs["response_format"] == openai_response_format(NESTED_SCHEMA)
        assert result is instance.bind.return_value

    def test_openai_no_bind_without_schema(self):
        """Verify no binding wrapper is applied when no schema is requested."""
        mock_cls = MagicMock()
        instance = mock_cls.return_value
        with _mock_module("langchain_openai", ChatOpenAI=mock_cls):
            result = OpenAIProvider().load(model_name="gpt-4o")
        instance.bind.assert_not_called()
        assert result is instance

    def test_azure_binds_response_format(self):
        """Verify AzureOpenAIProvider uses the same binding mechanism."""
        mock_cls = MagicMock()
        instance = mock_cls.return_value
        with _mock_module("langchain_openai", AzureChatOpenAI=mock_cls):
            result = AzureOpenAIProvider().load(
                model_name="gpt-41",
                api_version="2025-01-01-preview",
                structured_output_schema=NESTED_SCHEMA,
            )
        bind_kwargs = instance.bind.call_args[1]
        assert bind_kwargs["response_format"]["type"] == "json_schema"
        assert bind_kwargs["response_format"]["json_schema"]["strict"] is True
        assert result is instance.bind.return_value


# ---------------------------------------------------------------------------
# Provider wiring — Vertex / Google GenAI
# ---------------------------------------------------------------------------


class TestGoogleStructuredOutput:
    """Verify the Google providers set response_schema + JSON mime type."""

    def test_vertex_maps_schema_and_mime_type(self):
        """Verify VertexAIProvider maps the cross-provider kwarg."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_vertexai", ChatVertexAI=mock_cls):
            VertexAIProvider().load(
                model_name="gemini-2.5-pro", structured_output_schema=NESTED_SCHEMA
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["response_schema"] == NESTED_SCHEMA
        assert kwargs["response_mime_type"] == "application/json"

    def test_vertex_native_params_still_work(self):
        """Verify the pre-existing response_schema path is unchanged."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_vertexai", ChatVertexAI=mock_cls):
            VertexAIProvider().load(
                model_name="gemini-2.5-pro",
                response_schema=NESTED_SCHEMA,
                response_mime_type="application/json",
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["response_schema"] is NESTED_SCHEMA

    @pytest.mark.parametrize(
        "native_kwargs",
        [
            {"response_schema": {"type": "object"}},
            {"response_mime_type": "application/json"},
        ],
    )
    def test_vertex_conflict_raises(self, native_kwargs):
        """Verify mixing cross-provider and Vertex-native spellings raises."""
        with pytest.raises(ValueError, match="not both"):
            VertexAIProvider().load(
                model_name="gemini-2.5-pro",
                structured_output_schema=NESTED_SCHEMA,
                **native_kwargs,
            )

    def test_genai_maps_schema_and_mime_type(self):
        """Verify GoogleGenAIProvider maps the cross-provider kwarg."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(
                model_name="gemini-2.5-flash",
                structured_output_schema=NESTED_SCHEMA,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["response_schema"] == NESTED_SCHEMA
        assert kwargs["response_mime_type"] == "application/json"

    def test_genai_params_absent_without_schema(self):
        """Verify no response params are set when no schema is requested."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(model_name="gemini-2.5-flash")
        kwargs = mock_cls.call_args[1]
        assert "response_schema" not in kwargs
        assert "response_mime_type" not in kwargs


# ---------------------------------------------------------------------------
# load_model — fail-fast gate and end-to-end routing
# ---------------------------------------------------------------------------


class TestLoadModelStructuredOutput:
    """Verify load_model gates and routes structured_output_schema."""

    @pytest.mark.parametrize(
        "provider",
        ["remote_aws_bedrock", "remote_anthropic", "cli", "local_llamacpp"],
    )
    def test_gate_rejects_unsupported_provider(self, provider):
        """Verify a schema request against an unsupported provider fails fast."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        with pytest.raises(ValueError, match="structured output"):
            load_model(
                provider,
                model_name="anything",
                structured_output_schema=NESTED_SCHEMA,
            )

    def test_gate_ignores_none_schema(self):
        """Verify structured_output_schema=None does not trigger the gate."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            load_model(
                "local_ollama", model_name="qwen3", structured_output_schema=None
            )
        assert "format" not in mock_cls.call_args[1]

    def test_registry_path_routes_schema_to_ollama(self):
        """Verify the registry path delivers the schema as ChatOllama format."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            load_model(
                "local_ollama",
                model_name="qwen3:8b",
                structured_output_schema=NESTED_SCHEMA,
            )
        assert mock_cls.call_args[1]["format"] is NESTED_SCHEMA

    def test_builtin_dispatch_routes_schema_to_openai(self):
        """Verify the built-in remote_openai branch binds response_format."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        instance = mock_cls.return_value
        with _mock_module("langchain_openai", ChatOpenAI=mock_cls):
            result = load_model(
                "remote_openai",
                model_name="gpt-4o",
                structured_output_schema=NESTED_SCHEMA,
            )
        bind_kwargs = instance.bind.call_args[1]
        assert bind_kwargs["response_format"]["type"] == "json_schema"
        assert result is instance.bind.return_value

    def test_builtin_dispatch_routes_schema_to_azure(self):
        """Verify the built-in remote_azure_openai branch binds response_format."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        instance = mock_cls.return_value
        with _mock_module("langchain_openai", AzureChatOpenAI=mock_cls):
            load_model(
                "remote_azure_openai",
                model_name="gpt-41",
                api_version="2025-01-01-preview",
                structured_output_schema=NESTED_SCHEMA,
            )
        bind_kwargs = instance.bind.call_args[1]
        assert bind_kwargs["response_format"]["json_schema"]["strict"] is True

    def test_builtin_dispatch_routes_schema_to_vertex(self):
        """Verify the built-in remote_google_vertex branch maps the schema."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        with _mock_module("langchain_google_vertexai", ChatVertexAI=mock_cls):
            load_model(
                "remote_google_vertex",
                model_name="gemini-2.5-pro",
                structured_output_schema=NESTED_SCHEMA,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["response_schema"] is NESTED_SCHEMA
        assert kwargs["response_mime_type"] == "application/json"

    def test_builtin_vertex_conflict_raises(self):
        """Verify the built-in Vertex loader rejects mixed schema spellings."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_cls = MagicMock()
        with _mock_module("langchain_google_vertexai", ChatVertexAI=mock_cls):
            with pytest.raises(ValueError, match="not both"):
                load_model(
                    "remote_google_vertex",
                    model_name="gemini-2.5-pro",
                    structured_output_schema=NESTED_SCHEMA,
                    response_schema={"type": "object"},
                )
