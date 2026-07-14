"""Tests for the local Ollama server provider.

Covers:
- ``OllamaProvider.load()`` with a mocked ``langchain_ollama.ChatOllama``:
  default base_url, explicit base_url, num_predict mapping, full config,
  optional-param omission, extra-kwarg tolerance, and the missing-SDK path.
- Built-in registration of ``local_ollama`` in ``PROVIDER_REGISTRY``.
- The ``local_ollama`` catalog entry: native tool-calling capability wiring
  (``tool_strategy == "native"`` and ``supports_tools is True``) and the
  ``base_url`` default carried via the entry's ``kwargs`` block.
- ``resolve_tool_strategy`` treating the entry as native.
- ``load_model("local_ollama", ...)`` routing through the provider registry.
"""

# pylint: disable=too-few-public-methods,duplicate-code

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers.base import LLMProvider
from bili.iris.providers.ollama_provider import DEFAULT_OLLAMA_BASE_URL, OllamaProvider
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

    def test_missing_sdk_raises_import_error(self):
        """Verify a helpful ImportError surfaces when langchain_ollama is absent."""
        # Ensure no stand-in module is present so the real import is attempted.
        saved = sys.modules.pop("langchain_ollama", None)
        try:
            with pytest.raises(ImportError):
                OllamaProvider().load(model_name="qwen3")
        finally:
            if saved is not None:
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
