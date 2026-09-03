"""Tests for the Google Generative AI (Gemini Developer API) provider.

Covers the Developer-API-specific selection behaviour that distinguishes
``remote_google_genai`` from the Vertex AI provider:

- ``"genai:"``-sentinel prefix stripping in ``GoogleGenAIProvider.load()``
  (with and without the prefix, leading-only, and single-strip semantics).
- The resolver-to-provider contract: the resolver leaves the sentinel on
  ``model_id`` and ``load()`` strips it before the API call.
- Routing against the real (unpatched) ``LLM_MODELS`` catalog, where catalog
  lookup precedes the resolver heuristics and Vertex is declared first.
- The Developer API catalog entries: Flash Lite models are selectable and every
  entry declares native tool calling.

``GoogleGenAIProvider.load()``'s parameter mapping is covered in
``test_new_providers.py``; this module covers only the Developer-API selection
seam.
"""

# pylint: disable=too-few-public-methods,duplicate-code

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers.google_genai_provider import (
    GOOGLE_GENAI_MODEL_PREFIX,
    GoogleGenAIProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _mock_module(module_name: str, **attrs):
    """Temporarily inject a fake module into sys.modules for lazy-import tests.

    ``GoogleGenAIProvider`` imports ``langchain_google_genai`` inside ``load()``
    so the module imports without the optional dependency installed.  This
    context manager supplies a stand-in module so the ``from
    langchain_google_genai import ChatGoogleGenerativeAI`` inside ``load()``
    resolves without the real SDK.
    """
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


def _load_and_capture_model(model_name: str) -> str:
    """Run ``load()`` with a mocked SDK and return the model id it passed on."""
    mock_cls = MagicMock()
    with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
        GoogleGenAIProvider().load(model_name=model_name)
    return mock_cls.call_args[1]["model"]


def _resolve(model_name: str):
    """Call _resolve_model_full against the real LLM_MODELS catalog."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        _resolve_model_full,
    )

    return _resolve_model_full(model_name)


# ---------------------------------------------------------------------------
# "genai:" sentinel prefix
# ---------------------------------------------------------------------------


class TestGoogleGenAISentinelPrefix:
    """Verify the 'genai:' sentinel is stripped before the id reaches the API."""

    def test_sentinel_prefix_constant(self):
        """Verify the exported sentinel constant is the documented literal."""
        assert GOOGLE_GENAI_MODEL_PREFIX == "genai:"

    def test_sentinel_stripped_before_api(self):
        """Verify a 'genai:'-prefixed name reaches the SDK bare."""
        assert _load_and_capture_model("genai:gemini-2.5-flash") == "gemini-2.5-flash"

    def test_bare_name_passes_through_unchanged(self):
        """Verify a name without the sentinel is not altered."""
        assert (
            _load_and_capture_model("gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"
        )

    def test_only_leading_sentinel_is_stripped(self):
        """Verify an embedded 'genai:' substring is not stripped from the id.

        Stripping anywhere but the front would corrupt the model id sent to the
        API, so the prefix check is anchored to the start.
        """
        assert _load_and_capture_model("gemini-genai:custom") == "gemini-genai:custom"

    def test_sentinel_stripped_once_only(self):
        """Verify only a single leading sentinel is removed."""
        assert (
            _load_and_capture_model("genai:genai:gemini-2.5-flash")
            == "genai:gemini-2.5-flash"
        )

    def test_resolver_to_provider_contract(self):
        """Verify the resolver's model_id survives load() as the real API id.

        The resolver deliberately leaves the sentinel on model_id (matching the
        'ollama:' contract) and relies on load() to strip it.  This pins the
        cross-file contract end to end: a Vertex-catalogued id prefixed with the
        sentinel must reach the Developer API bare.
        """
        provider, model_id, _ = _resolve("genai:gemini-2.5-flash")
        assert provider == "remote_google_genai"
        assert _load_and_capture_model(model_id) == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Routing against the real catalog
# ---------------------------------------------------------------------------


class TestGoogleGenAICatalogRouting:
    """Verify Developer API routing against the real (unpatched) catalog.

    These exercise the resolver's full search order -- catalog lookup first,
    heuristics second -- rather than the heuristics in isolation.
    """

    def test_gemini_3_1_flash_lite_resolves_to_direct_api(self):
        """Verify the catalogued 3.1 Flash Lite entry resolves to the Developer API."""
        provider, model_id, _ = _resolve("gemini-3.1-flash-lite")
        assert provider == "remote_google_genai"
        assert model_id == "gemini-3.1-flash-lite"

    def test_sentinel_overrides_vertex_catalog_match(self):
        """Verify the sentinel reaches the Developer API for a Vertex-listed id.

        A bare 'gemini-2.5-flash' resolves to Vertex because catalog lookup
        precedes the heuristics and Vertex is declared first.  The sentinel is
        the seam that lets a downstream caller select the Developer API for that
        same id without hardcoding a display name.
        """
        provider, _, _ = _resolve("genai:gemini-2.5-flash")
        assert provider == "remote_google_genai"

    def test_bare_catalogued_id_still_routes_to_vertex(self):
        """Verify bare Vertex-catalogued ids are unchanged by the sentinel work."""
        provider, _, _ = _resolve("gemini-2.5-flash")
        assert provider == "remote_google_vertex"

    def test_direct_api_display_name_still_resolves(self):
        """Verify the '(Direct API)' display name remains a valid selector."""
        provider, model_id, _ = _resolve("Gemini 3.1 Flash Lite (Direct API)")
        assert provider == "remote_google_genai"
        assert model_id == "gemini-3.1-flash-lite"

    def test_ollama_sentinel_unaffected(self):
        """Verify the adjacent 'ollama:' sentinel still routes as before."""
        provider, _, _ = _resolve("ollama:deepseek-r1:14b")
        assert provider == "local_ollama"


# ---------------------------------------------------------------------------
# Developer API catalog entries
# ---------------------------------------------------------------------------


class TestGoogleGenAICatalogEntries:
    """Verify the remote_google_genai catalog entries downstream apps select."""

    @staticmethod
    def _entries() -> list:
        """Return the remote_google_genai model entries from LLM_MODELS."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        return LLM_MODELS["remote_google_genai"]["models"]

    @pytest.mark.parametrize(
        "model_id", ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
    )
    def test_flash_lite_models_catalogued_on_direct_api(self, model_id):
        """Verify the Flash Lite models are selectable on the Developer API."""
        assert model_id in {entry["model_id"] for entry in self._entries()}

    def test_entries_declare_native_tool_calling(self):
        """Verify every Developer API entry binds tools natively.

        Downstream agents resolve tool strategy from the catalog; an entry
        defaulting to a non-native strategy would silently route a tool-calling
        agent to the prompted ReAct loop.
        """
        for entry in self._entries():
            assert entry["tool_strategy"] == "native", entry["model_name"]
            assert entry["supports_tools"] is True, entry["model_name"]

    def test_gemini_3_1_flash_lite_declares_context_window(self):
        """Verify the 3.1 Flash Lite entry declares the documented token limits.

        https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite
        """
        entry = next(
            e for e in self._entries() if e["model_id"] == "gemini-3.1-flash-lite"
        )
        assert entry["max_input_tokens"] == 1048576
        assert entry["max_output_tokens"] == 65536

    def test_tool_strategy_resolves_native_for_flash_lite(self):
        """Verify resolve_tool_strategy reports native for the new entry."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_supports_tools,
            resolve_tool_strategy,
        )

        assert resolve_tool_strategy("gemini-3.1-flash-lite") == "native"
        assert resolve_supports_tools("gemini-3.1-flash-lite") is True
