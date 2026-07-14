"""Tests for the local Ollama server provider.

Covers:
- ``OllamaProvider.load()`` with a mocked ``langchain_ollama.ChatOllama``:
  default base_url, explicit base_url, num_predict mapping, full config,
  optional-param omission, extra-kwarg tolerance, the missing-SDK path, and
  ``"ollama:"``-sentinel prefix stripping (with and without the prefix).
- Built-in registration of ``local_ollama`` in ``PROVIDER_REGISTRY``.
- The ``local_ollama`` catalog entry: native tool-calling capability wiring
  (``tool_strategy == "native"`` and ``supports_tools is True``) and the
  ``base_url`` default carried via the entry's ``kwargs`` block.
- ``resolve_tool_strategy`` treating the entry as native.
- ``load_model("local_ollama", ...)`` routing through the provider registry.
- The resolver's ``"ollama:"`` heuristic routing an arbitrary, non-catalog
  user-pulled tag to ``local_ollama``, including the full ``create_llm()``
  path from an ``AgentSpec.model_name`` through to ``ChatOllama``.
"""

# pylint: disable=too-few-public-methods,duplicate-code

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers.base import LLMProvider
from bili.iris.providers.ollama_provider import (
    DEFAULT_OLLAMA_BASE_URL,
    OLLAMA_MODEL_PREFIX,
    OllamaProvider,
)
from bili.iris.providers.registry import PROVIDER_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _mock_module(module_name: str, **attrs):
    """Temporarily inject a fake module into sys.modules for lazy-import tests.

    ``OllamaProvider`` imports ``langchain_ollama`` inside ``load()`` so the
    module imports without the optional dependency installed.  This context
    manager supplies a stand-in module so the ``from langchain_ollama import
    ChatOllama`` inside ``load()`` resolves without the real SDK.
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


# ---------------------------------------------------------------------------
# OllamaProvider.load()
# ---------------------------------------------------------------------------


class TestOllamaProviderLoad:
    """Verify OllamaProvider.load() constructs ChatOllama correctly."""

    def test_minimal_load_uses_default_base_url(self):
        """Verify minimal config passes model and the default base_url."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3"
        assert kwargs["base_url"] == DEFAULT_OLLAMA_BASE_URL

    def test_explicit_base_url_overrides_default(self):
        """Verify a provided base_url takes precedence over the default."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(
                model_name="llama3.1", base_url="http://remote-host:11434"
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["base_url"] == "http://remote-host:11434"

    def test_max_tokens_maps_to_num_predict(self):
        """Verify max_tokens is forwarded as num_predict (Ollama's name)."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3", max_tokens=256)
        kwargs = mock_cls.call_args[1]
        assert kwargs["num_predict"] == 256
        assert "max_tokens" not in kwargs

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatOllama."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(
                model_name="mistral",
                base_url="http://localhost:22222",
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                top_k=20,
                seed=7,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "mistral"
        assert kwargs["base_url"] == "http://localhost:22222"
        assert kwargs["num_predict"] == 512
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 20
        assert kwargs["seed"] == 7

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent from config when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3")
        kwargs = mock_cls.call_args[1]
        for absent in ("num_predict", "temperature", "top_p", "top_k", "seed"):
            assert absent not in kwargs

    def test_none_base_url_falls_back_to_default(self):
        """Verify passing base_url=None applies the default endpoint."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3", base_url=None)
        kwargs = mock_cls.call_args[1]
        assert kwargs["base_url"] == DEFAULT_OLLAMA_BASE_URL

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3", unknown="x")
        assert mock_cls.called

    def test_returns_chatollama_instance(self):
        """Verify load() returns the object ChatOllama constructs."""
        sentinel = MagicMock(name="chat_ollama_instance")
        mock_cls = MagicMock(return_value=sentinel)
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            result = OllamaProvider().load(model_name="qwen3")
        assert result is sentinel

    def test_ollama_prefix_is_stripped_before_reaching_chatollama(self):
        """Verify the 'ollama:' sentinel is stripped so ChatOllama gets the bare tag.

        Reproduces the exact failing case from AETHER/MAS resolution: a
        user-pulled tag with a colon-separated size suffix (qwen3:14b).
        """
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="ollama:qwen3:14b")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3:14b"

    def test_bare_tag_without_prefix_still_works(self):
        """Verify a bare tag with no sentinel prefix is passed through unchanged.

        This is the single-turn path: an explicit provider_type is given, so
        the caller passes the bare Ollama tag directly (no "ollama:" prefix
        needed, since routing is not the resolver's job here).
        """
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name="qwen3:14b")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3:14b"

    def test_prefix_constant_matches_stripped_value(self):
        """Verify OLLAMA_MODEL_PREFIX is exactly what load() strips."""
        mock_cls = MagicMock()
        tag = "llama3.1:70b-instruct-q4_0"
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            OllamaProvider().load(model_name=f"{OLLAMA_MODEL_PREFIX}{tag}")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == tag

    def test_missing_sdk_raises_import_error(self):
        """Verify a helpful ImportError surfaces when langchain_ollama is absent.

        Setting sys.modules["langchain_ollama"] = None blocks the import
        unconditionally (Python raises ImportError for a None entry),
        regardless of whether the real package happens to be installed in
        the venv running this test -- unlike popping the module, which only
        exercises this path when the dependency is truly absent.
        """
        saved = sys.modules.pop("langchain_ollama", None)
        sys.modules["langchain_ollama"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="langchain-ollama.*required"):
                OllamaProvider().load(model_name="qwen3")
        finally:
            if saved is None:
                sys.modules.pop("langchain_ollama", None)
            else:
                sys.modules["langchain_ollama"] = saved


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


class TestOllamaBuiltinRegistration:
    """Verify local_ollama is registered as a built-in provider."""

    def test_local_ollama_registered(self):
        """Verify 'local_ollama' is present in PROVIDER_REGISTRY."""
        assert "local_ollama" in PROVIDER_REGISTRY

    def test_local_ollama_maps_to_ollama_provider(self):
        """Verify 'local_ollama' maps to OllamaProvider."""
        assert PROVIDER_REGISTRY.get("local_ollama") is OllamaProvider

    def test_ollama_provider_is_llmprovider_subclass(self):
        """Verify OllamaProvider is an LLMProvider subclass."""
        assert issubclass(OllamaProvider, LLMProvider)


# ---------------------------------------------------------------------------
# Catalog entry — native tool-calling capability wiring
# ---------------------------------------------------------------------------


def _ollama_entry() -> dict:
    """Return the single local_ollama model entry from LLM_MODELS."""
    from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
        LLM_MODELS,
    )

    models = LLM_MODELS["local_ollama"]["models"]
    assert len(models) == 1
    return models[0]


class TestOllamaCatalogEntry:
    """Verify the local_ollama LLM_MODELS entry declares native tool calling."""

    def test_entry_present_in_llm_models(self):
        """Verify local_ollama is a top-level key in LLM_MODELS."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        assert "local_ollama" in LLM_MODELS

    def test_entry_declares_native_tool_strategy(self):
        """Verify the entry enables native tool calling (unlike other local providers)."""
        entry = _ollama_entry()
        assert entry["tool_strategy"] == "native"
        assert entry["supports_tools"] is True

    def test_entry_has_required_fields(self):
        """Verify the entry carries model_name and model_id."""
        entry = _ollama_entry()
        assert "model_name" in entry
        assert "model_id" in entry

    def test_entry_carries_base_url_default_in_kwargs(self):
        """Verify the entry's kwargs default the daemon base_url."""
        entry = _ollama_entry()
        assert entry["kwargs"]["base_url"] == DEFAULT_OLLAMA_BASE_URL

    def test_entry_marked_local_only(self):
        """Verify the entry is flagged local_only (no cloud egress)."""
        entry = _ollama_entry()
        assert entry["local_only"] is True

    def test_tool_strategy_resolves_native(self):
        """Verify resolve_tool_strategy treats the entry as native tool-calling."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_supports_tools,
            resolve_tool_strategy,
        )

        entry = _ollama_entry()
        assert resolve_tool_strategy(entry["model_id"]) == "native"
        assert resolve_supports_tools(entry["model_id"]) is True

    def test_catalog_base_url_flows_as_extra_kwargs(self):
        """Verify _resolve_model_full surfaces the entry's kwargs (incl. base_url)."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            _resolve_model_full,
        )

        entry = _ollama_entry()
        provider, model_id, extra_kwargs = _resolve_model_full(entry["model_name"])
        assert provider == "local_ollama"
        assert model_id == entry["model_id"]
        assert extra_kwargs["base_url"] == DEFAULT_OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# load_model() registry-path routing
# ---------------------------------------------------------------------------


class TestLoadModelRoutesOllama:
    """Verify load_model('local_ollama', ...) reaches OllamaProvider.load()."""

    def test_load_model_delegates_to_ollama_provider(self):
        """Verify load_model constructs ChatOllama via the registry else-branch."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        sentinel = MagicMock(name="chat_ollama_instance")
        mock_cls = MagicMock(return_value=sentinel)
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            result = load_model(
                "local_ollama",
                model_name="qwen3",
                base_url="http://localhost:11434",
                temperature=0.1,
            )
        assert result is sentinel
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3"
        assert kwargs["base_url"] == "http://localhost:11434"
        assert kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Resolver heuristic — "ollama:" sentinel for arbitrary, non-catalog tags
# ---------------------------------------------------------------------------


class TestOllamaSentinelHeuristic:
    """Verify the resolver routes an "ollama:"-prefixed tag to local_ollama.

    Reproduces the AETHER/MAS resolution gap: an AgentSpec carries only
    model_name, resolved via catalog lookup + prefix heuristics. An arbitrary
    user-pulled tag (not in the catalog, matching no other heuristic) could
    not resolve at all before the "ollama:" sentinel rule was added.
    """

    def test_before_fix_repro_raises_without_sentinel(self):
        """Verify a bare non-catalog tag with no sentinel still cannot resolve.

        This is the exact shape of the original failure: passing the raw
        user-pulled tag with no routing hint raises ValueError, because it
        matches neither a catalog entry nor any heuristic pattern. The
        "ollama:" sentinel (tested below) is the fix; a bare tag legitimately
        has no way to route without either a catalog entry or the sentinel.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_model,
        )

        with pytest.raises(ValueError, match="Cannot resolve model"):
            resolve_model("qwen3:14b")

    def test_sentinel_resolves_the_exact_failing_tag(self):
        """Verify 'ollama:qwen3:14b' now resolves to local_ollama.

        This is the exact previously-failing case from AETHER/MAS
        resolution: a colon-separated user-pulled tag with a size suffix.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_model,
        )

        provider, model_id = resolve_model("ollama:qwen3:14b")
        assert provider == "local_ollama"
        # The resolver keeps the model_id unchanged (mirrors the "cli:"
        # sentinel); OllamaProvider.load() strips the prefix, not the resolver.
        assert model_id == "ollama:qwen3:14b"

    @pytest.mark.parametrize(
        "tag",
        [
            "ollama:qwen3",
            "ollama:qwen3:14b",
            "ollama:llama3.1:70b-instruct-q4_0",
            "ollama:custom-finetune-v2",
        ],
    )
    def test_sentinel_resolves_arbitrary_tags(self, tag):
        """Verify any 'ollama:'-prefixed tag routes to local_ollama."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_provider,
        )

        assert resolve_provider(tag) == "local_ollama"

    def test_sentinel_tool_strategy_is_native(self):
        """Verify a non-catalog 'ollama:' tag still resolves to native tool-calling.

        The tag is not in LLM_MODELS, so resolve_tool_strategy falls through
        to its "not found" default, which is "native". This documents that
        fallback explicitly for the sentinel-routed (non-catalog) path.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_tool_strategy,
        )

        assert resolve_tool_strategy("ollama:qwen3:14b") == "native"

    def test_create_llm_end_to_end_with_sentinel_tag(self):
        """Verify create_llm() resolves an AgentSpec's sentinel-prefixed tag
        all the way to a ChatOllama instance with the bare tag.

        End-to-end reproduction of the fixed AETHER/MAS path: an AgentSpec
        with model_name="ollama:qwen3:14b" now compiles to a working LLM
        instead of raising ValueError at resolution time.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )
        from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
            AgentSpec,
        )

        spec = AgentSpec(
            agent_id="local-model-agent",
            role="tester",
            objective="Integration test for the ollama: sentinel resolution path.",
            model_name="ollama:qwen3:14b",
        )

        sentinel = MagicMock(name="chat_ollama_instance")
        mock_cls = MagicMock(return_value=sentinel)
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            result = create_llm(spec)

        assert result is sentinel
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3:14b"
