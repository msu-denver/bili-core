"""Tests for the new built-in API providers added in the provider-api-expansion.

Covers:
- Each concrete provider's ``load()`` method with mocked heavy dependencies:
  Anthropic, Mistral, Cohere, Google GenAI, DeepSeek, xAI, Groq
- All 7 new types are registered in PROVIDER_REGISTRY via builtin.py
- LLM_MODELS entries exist for each new provider type
- Heuristic resolution routes new model ID patterns correctly
- ``load_model()`` delegates to each new provider via the registry else-branch
"""

# pylint: disable=too-few-public-methods,duplicate-code

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers.anthropic_provider import AnthropicProvider
from bili.iris.providers.base import LLMProvider
from bili.iris.providers.cohere_provider import CohereProvider
from bili.iris.providers.deepseek_provider import DeepSeekProvider
from bili.iris.providers.google_genai_provider import GoogleGenAIProvider
from bili.iris.providers.groq_provider import GroqProvider
from bili.iris.providers.mistral_provider import MistralProvider
from bili.iris.providers.registry import PROVIDER_REGISTRY
from bili.iris.providers.xai_provider import XAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _mock_module(module_name: str, **attrs):
    """Temporarily inject a fake module into sys.modules for testing lazy imports.

    Providers import their SDK inside ``load()`` to keep the base install lean.
    When the SDK is not installed, tests need to mock the module so that the
    ``from <sdk> import <Class>`` inside ``load()`` does not raise ImportError.

    Usage::

        with _mock_module("langchain_foo", ChatFoo=MagicMock()) as mod:
            provider.load(model_name="foo")
            kwargs = mod.ChatFoo.call_args[1]
    """
    mod = ModuleType(module_name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    # Capture the real module object, not a bool: a present-flag would restore
    # the literal True into sys.modules in the finally, corrupting the module
    # for later tests when the SDK actually is installed.  Matches the helper
    # in test_ollama_provider.py.
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
# Builtin registration — new provider types
# ---------------------------------------------------------------------------

_NEW_PROVIDER_TYPES = {
    "remote_anthropic",
    "remote_mistral",
    "remote_cohere",
    "remote_google_genai",
    "remote_deepseek",
    "remote_xai",
    "remote_groq",
}


class TestNewBuiltinRegistration:
    """Verify all 7 new provider types are registered in PROVIDER_REGISTRY."""

    def test_all_new_providers_registered(self):
        """Verify each new provider type is present in PROVIDER_REGISTRY."""
        for provider_type in _NEW_PROVIDER_TYPES:
            assert (
                provider_type in PROVIDER_REGISTRY
            ), f"Expected '{provider_type}' in PROVIDER_REGISTRY"

    def test_new_providers_are_llmprovider_subclasses(self):
        """Verify every new registered provider is an LLMProvider subclass."""
        for provider_type in _NEW_PROVIDER_TYPES:
            cls = PROVIDER_REGISTRY.get(provider_type)
            assert cls is not None
            assert issubclass(
                cls, LLMProvider
            ), f"Provider '{provider_type}' → {cls} is not an LLMProvider subclass"

    def test_provider_classes_map_correctly(self):
        """Verify each type maps to its expected implementation class."""
        mapping = {
            "remote_anthropic": AnthropicProvider,
            "remote_mistral": MistralProvider,
            "remote_cohere": CohereProvider,
            "remote_google_genai": GoogleGenAIProvider,
            "remote_deepseek": DeepSeekProvider,
            "remote_xai": XAIProvider,
            "remote_groq": GroqProvider,
        }
        for provider_type, expected_cls in mapping.items():
            registered = PROVIDER_REGISTRY.get(provider_type)
            assert registered is expected_cls, (
                f"Expected '{provider_type}' → {expected_cls.__name__}, "
                f"got {registered}"
            )


# ---------------------------------------------------------------------------
# LLM_MODELS entries
# ---------------------------------------------------------------------------


class TestLLMModelsEntries:
    """Verify each new provider type has entries in LLM_MODELS."""

    def test_all_new_types_in_llm_models(self):
        """Verify each new provider type key appears in LLM_MODELS."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        for provider_type in _NEW_PROVIDER_TYPES:
            assert (
                provider_type in LLM_MODELS
            ), f"Expected '{provider_type}' in LLM_MODELS"

    def test_each_entry_has_models_list(self):
        """Verify every new LLM_MODELS entry has a non-empty models list."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        for provider_type in _NEW_PROVIDER_TYPES:
            entry = LLM_MODELS.get(provider_type, {})
            models = entry.get("models", [])
            assert len(models) > 0, f"LLM_MODELS['{provider_type}']['models'] is empty"

    def test_each_model_entry_has_required_fields(self):
        """Verify every model entry has model_name and model_id."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        for provider_type in _NEW_PROVIDER_TYPES:
            models = LLM_MODELS.get(provider_type, {}).get("models", [])
            for model in models:
                assert (
                    "model_name" in model
                ), f"Missing 'model_name' in {provider_type} entry: {model}"
                assert (
                    "model_id" in model
                ), f"Missing 'model_id' in {provider_type} entry: {model}"

    def test_anthropic_models_present(self):
        """Verify the four Anthropic model IDs are in LLM_MODELS."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        ids = {m["model_id"] for m in LLM_MODELS["remote_anthropic"]["models"]}
        assert "claude-opus-4-8" in ids
        assert "claude-sonnet-4-6" in ids
        assert "claude-haiku-4-5" in ids
        assert "claude-fable-5" in ids

    def test_groq_models_present(self):
        """Verify Groq compound-beta models are in LLM_MODELS."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        ids = {m["model_id"] for m in LLM_MODELS["remote_groq"]["models"]}
        assert "compound-beta" in ids
        assert "compound-beta-mini" in ids

    def test_google_genai_display_names_include_direct_api(self):
        """Verify remote_google_genai entries use '(Direct API)' in display names.

        The disambiguation suffix makes these entries selectable via display name
        without colliding with the identically-model_id'd Vertex entries.
        """
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        for entry in LLM_MODELS["remote_google_genai"]["models"]:
            assert "(Direct API)" in entry["model_name"], (
                f"remote_google_genai entry missing '(Direct API)' in model_name: "
                f"{entry['model_name']!r}"
            )

    def test_google_genai_display_names_are_distinct_from_vertex(self):
        """Verify no remote_google_genai display name collides with Vertex display names."""
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )

        vertex_names = {
            m["model_name"]
            for m in LLM_MODELS.get("remote_google_vertex", {}).get("models", [])
        }
        for entry in LLM_MODELS["remote_google_genai"]["models"]:
            assert entry["model_name"] not in vertex_names, (
                f"remote_google_genai display name '{entry['model_name']}' collides "
                f"with a remote_google_vertex entry"
            )


# ---------------------------------------------------------------------------
# Heuristic resolution — new patterns
# ---------------------------------------------------------------------------


class TestHeuristicResolution:
    """Verify _HEURISTIC_RULES routes new model ID patterns correctly."""

    def _resolve(self, model_name: str):
        """Import and call _resolve_model_full, ignoring LLM_MODELS (patch it)."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            _resolve_model_full,
        )

        # Patch _lookup_in_llm_models to always return None so heuristics run.
        with patch(
            "bili.aether.compiler.llm_resolver._lookup_in_llm_models",
            return_value=None,
        ):
            return _resolve_model_full(model_name)

    def test_claude_routes_to_anthropic(self):
        """Verify 'claude-' prefix routes to remote_anthropic."""
        provider, model_id, _ = self._resolve("claude-opus-4-8")
        assert provider == "remote_anthropic"
        assert model_id == "claude-opus-4-8"

    def test_anthropic_claude_prefix_still_routes_to_bedrock(self):
        """Verify 'anthropic.claude' routes to remote_aws_bedrock (Bedrock-hosted)."""
        provider, model_id, _ = self._resolve("anthropic.claude-v2")
        assert provider == "remote_aws_bedrock"
        assert model_id == "anthropic.claude-v2"

    def test_mistral_routes_to_mistral_provider(self):
        """Verify 'mistral-' prefix routes to remote_mistral."""
        provider, _, _ = self._resolve("mistral-large-latest")
        assert provider == "remote_mistral"

    def test_codestral_routes_to_mistral_provider(self):
        """Verify 'codestral' routes to remote_mistral."""
        provider, _, _ = self._resolve("codestral-latest")
        assert provider == "remote_mistral"

    def test_command_routes_to_cohere(self):
        """Verify 'command-' prefix routes to remote_cohere."""
        provider, _, _ = self._resolve("command-r-plus")
        assert provider == "remote_cohere"

    def test_gemini_routes_to_google_genai(self):
        """Verify 'gemini-' prefix routes to remote_google_genai."""
        provider, _, _ = self._resolve("gemini-2.5-flash")
        assert provider == "remote_google_genai"

    def test_genai_sentinel_routes_to_google_genai(self):
        """Verify the 'genai:' sentinel routes to remote_google_genai."""
        provider, model_id, _ = self._resolve("genai:gemini-2.5-flash")
        assert provider == "remote_google_genai"
        # The prefix stays on model_id; GoogleGenAIProvider.load() strips it.
        assert model_id == "genai:gemini-2.5-flash"

    def test_genai_sentinel_routes_non_gemini_names(self):
        """Verify the sentinel routes a name carrying no 'gemini' substring.

        The sentinel must not depend on the vendor rules to fire, or a future
        Developer API model named off-pattern would fail to resolve.
        """
        provider, _, _ = self._resolve("genai:some-future-model")
        assert provider == "remote_google_genai"

    def test_genai_sentinel_precedes_vendor_rules(self):
        """Verify the sentinel wins over an incidental vendor substring.

        Rule matching is a substring test taking the first hit, so a sentinel
        placed after a vendor rule would let that rule steal the match.
        """
        provider, _, _ = self._resolve("genai:gpt-4o-lookalike")
        assert provider == "remote_google_genai"

    def test_deepseek_routes_to_deepseek(self):
        """Verify 'deepseek-' prefix routes to remote_deepseek."""
        provider, _, _ = self._resolve("deepseek-chat")
        assert provider == "remote_deepseek"

    def test_grok_routes_to_xai(self):
        """Verify 'grok-' prefix routes to remote_xai."""
        provider, _, _ = self._resolve("grok-3-latest")
        assert provider == "remote_xai"

    def test_llama3_routes_to_groq(self):
        """Verify 'llama-3' prefix routes to remote_groq."""
        provider, _, _ = self._resolve("llama-3.3-70b-versatile")
        assert provider == "remote_groq"

    def test_compound_beta_routes_to_groq(self):
        """Verify 'compound-beta' routes to remote_groq."""
        provider, _, _ = self._resolve("compound-beta")
        assert provider == "remote_groq"

    # Backward-compat fallbacks (pre-existing rules restored in this PR)

    def test_bare_gemini_still_routes_to_vertex(self):
        """Verify bare 'gemini' (no hyphen) still falls back to remote_google_vertex.

        Pre-existing behavior: 'gemini' was a broad heuristic for Vertex AI.
        With the addition of 'gemini-' -> remote_google_genai, the hyphenated
        form goes to GenAI while the bare form still reaches Vertex.
        """
        provider, _, _ = self._resolve("gemini")
        assert (
            provider == "remote_google_vertex"
        ), "Bare 'gemini' must still fall back to remote_google_vertex"

    def test_bare_mistral_still_routes_to_bedrock(self):
        """Verify a bare 'mistral' string still falls back to remote_aws_bedrock.

        Pre-existing behavior: 'mistral' was a Bedrock catch-all.  The more
        specific 'mistral-' -> remote_mistral pattern handles direct-API IDs;
        bare 'mistral' (no hyphen) reaches the Bedrock fallback.
        """
        provider, _, _ = self._resolve("mistral")
        assert (
            provider == "remote_aws_bedrock"
        ), "Bare 'mistral' must still fall back to remote_aws_bedrock"

    def test_gemini_hyphen_takes_priority_over_bare_gemini(self):
        """Verify 'gemini-2.0-flash' hits the more-specific GenAI rule, not Vertex.

        The 'gemini-' pattern fires first because it appears before the broad
        'gemini' fallback in _HEURISTIC_RULES.
        """
        provider, _, _ = self._resolve("gemini-2.0-flash")
        assert provider == "remote_google_genai"

    def test_mistral_hyphen_takes_priority_over_bare_mistral(self):
        """Verify 'mistral-large-latest' hits direct Mistral, not Bedrock fallback.

        The 'mistral-' pattern fires first because it appears before the broad
        'mistral' fallback in _HEURISTIC_RULES.
        """
        provider, _, _ = self._resolve("mistral-large-latest")
        assert provider == "remote_mistral"


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Verify AnthropicProvider.load() constructs ChatAnthropic correctly."""

    def test_minimal_load_applies_default_max_tokens(self):
        """Verify minimal config uses 1024 as the default max_tokens."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            AnthropicProvider().load(model_name="claude-sonnet-4-6")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 1024

    def test_explicit_max_tokens_overrides_default(self):
        """Verify provided max_tokens takes precedence over the default."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            AnthropicProvider().load(model_name="claude-opus-4-8", max_tokens=4096)
        kwargs = mock_cls.call_args[1]
        assert kwargs["max_tokens"] == 4096

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatAnthropic."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            AnthropicProvider().load(
                model_name="claude-haiku-4-5",
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=10,
                max_retries=3,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 10
        assert kwargs["max_retries"] == 3

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            AnthropicProvider().load(model_name="claude-sonnet-4-6", unknown="x")
        assert mock_cls.called

    def test_none_max_tokens_uses_default(self):
        """Verify passing max_tokens=None applies the 1024 default."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            AnthropicProvider().load(model_name="claude-haiku-4-5", max_tokens=None)
        kwargs = mock_cls.call_args[1]
        assert kwargs["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# MistralProvider
# ---------------------------------------------------------------------------


class TestMistralProvider:
    """Verify MistralProvider.load() constructs ChatMistralAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatMistralAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_mistralai", ChatMistralAI=mock_cls):
            MistralProvider().load(model_name="mistral-large-latest")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "mistral-large-latest"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatMistralAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_mistralai", ChatMistralAI=mock_cls):
            MistralProvider().load(
                model_name="mistral-small-latest",
                max_tokens=1024,
                temperature=0.4,
                top_p=0.85,
                max_retries=2,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "mistral-small-latest"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.4
        assert kwargs["top_p"] == 0.85
        assert kwargs["max_retries"] == 2

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent from config when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_mistralai", ChatMistralAI=mock_cls):
            MistralProvider().load(model_name="mistral-large-latest")
        kwargs = mock_cls.call_args[1]
        for absent in ("max_tokens", "temperature", "top_p", "max_retries"):
            assert absent not in kwargs

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_mistralai", ChatMistralAI=mock_cls):
            MistralProvider().load(model_name="mistral-large-latest", seed=42)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# CohereProvider
# ---------------------------------------------------------------------------


class TestCohereProvider:
    """Verify CohereProvider.load() constructs ChatCohere correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatCohere."""
        mock_cls = MagicMock()
        with _mock_module("langchain_cohere", ChatCohere=mock_cls):
            CohereProvider().load(model_name="command-r-plus")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "command-r-plus"

    def test_top_p_maps_to_p(self):
        """Verify top_p is forwarded as 'p' (Cohere's parameter name)."""
        mock_cls = MagicMock()
        with _mock_module("langchain_cohere", ChatCohere=mock_cls):
            CohereProvider().load(model_name="command-r", top_p=0.8)
        kwargs = mock_cls.call_args[1]
        assert kwargs["p"] == 0.8

    def test_top_k_maps_to_k(self):
        """Verify top_k is forwarded as 'k' (Cohere's parameter name)."""
        mock_cls = MagicMock()
        with _mock_module("langchain_cohere", ChatCohere=mock_cls):
            CohereProvider().load(model_name="command-r", top_k=40)
        kwargs = mock_cls.call_args[1]
        assert kwargs["k"] == 40

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatCohere."""
        mock_cls = MagicMock()
        with _mock_module("langchain_cohere", ChatCohere=mock_cls):
            CohereProvider().load(
                model_name="command-a-plus-05-2026",
                max_tokens=2000,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                max_retries=3,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "command-a-plus-05-2026"
        assert kwargs["max_tokens"] == 2000
        assert kwargs["temperature"] == 0.5
        assert kwargs["p"] == 0.9
        assert kwargs["k"] == 50
        assert kwargs["max_retries"] == 3

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_cohere", ChatCohere=mock_cls):
            CohereProvider().load(model_name="command-r", seed=99)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# GoogleGenAIProvider
# ---------------------------------------------------------------------------


class TestGoogleGenAIProvider:
    """Verify GoogleGenAIProvider.load() constructs ChatGoogleGenerativeAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatGoogleGenerativeAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(model_name="gemini-2.5-flash")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gemini-2.5-flash"

    def test_max_tokens_maps_to_max_output_tokens(self):
        """Verify max_tokens is forwarded as max_output_tokens."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(model_name="gemini-2.0-flash", max_tokens=1024)
        kwargs = mock_cls.call_args[1]
        assert kwargs["max_output_tokens"] == 1024
        assert "max_tokens" not in kwargs

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatGoogleGenerativeAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(
                model_name="gemini-2.5-flash",
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
                top_k=30,
                max_retries=2,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["temperature"] == 0.6
        assert kwargs["top_p"] == 0.95
        assert kwargs["top_k"] == 30
        assert kwargs["max_retries"] == 2

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(model_name="gemini-2.0-flash-lite")
        kwargs = mock_cls.call_args[1]
        for absent in ("max_output_tokens", "temperature", "top_p", "top_k"):
            assert absent not in kwargs

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_google_genai", ChatGoogleGenerativeAI=mock_cls):
            GoogleGenAIProvider().load(model_name="gemini-2.5-flash", seed=1)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# DeepSeekProvider
# ---------------------------------------------------------------------------


class TestDeepSeekProvider:
    """Verify DeepSeekProvider.load() constructs ChatDeepSeek correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatDeepSeek."""
        mock_cls = MagicMock()
        with _mock_module("langchain_deepseek", ChatDeepSeek=mock_cls):
            DeepSeekProvider().load(model_name="deepseek-chat")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "deepseek-chat"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatDeepSeek."""
        mock_cls = MagicMock()
        with _mock_module("langchain_deepseek", ChatDeepSeek=mock_cls):
            DeepSeekProvider().load(
                model_name="deepseek-reasoner",
                max_tokens=8192,
                temperature=0.0,
                top_p=1.0,
                max_retries=3,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "deepseek-reasoner"
        assert kwargs["max_tokens"] == 8192
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["max_retries"] == 3

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_deepseek", ChatDeepSeek=mock_cls):
            DeepSeekProvider().load(model_name="deepseek-chat")
        kwargs = mock_cls.call_args[1]
        for absent in ("max_tokens", "temperature", "top_p", "max_retries"):
            assert absent not in kwargs

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_deepseek", ChatDeepSeek=mock_cls):
            DeepSeekProvider().load(model_name="deepseek-chat", seed=0)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# XAIProvider
# ---------------------------------------------------------------------------


class TestXAIProvider:
    """Verify XAIProvider.load() constructs ChatXAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatXAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_xai", ChatXAI=mock_cls):
            XAIProvider().load(model_name="grok-3-latest")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "grok-3-latest"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatXAI."""
        mock_cls = MagicMock()
        with _mock_module("langchain_xai", ChatXAI=mock_cls):
            XAIProvider().load(
                model_name="grok-beta",
                max_tokens=4096,
                temperature=0.8,
                top_p=0.9,
                max_retries=2,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "grok-beta"
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_retries"] == 2

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_xai", ChatXAI=mock_cls):
            XAIProvider().load(model_name="grok-3-latest")
        kwargs = mock_cls.call_args[1]
        for absent in ("max_tokens", "temperature", "top_p", "max_retries"):
            assert absent not in kwargs

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_xai", ChatXAI=mock_cls):
            XAIProvider().load(model_name="grok-3-latest", seed=0)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# GroqProvider
# ---------------------------------------------------------------------------


class TestGroqProvider:
    """Verify GroqProvider.load() constructs ChatGroq correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model_name to ChatGroq."""
        mock_cls = MagicMock()
        with _mock_module("langchain_groq", ChatGroq=mock_cls):
            GroqProvider().load(model_name="llama-3.3-70b-versatile")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_name"] == "llama-3.3-70b-versatile"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatGroq."""
        mock_cls = MagicMock()
        with _mock_module("langchain_groq", ChatGroq=mock_cls):
            GroqProvider().load(
                model_name="compound-beta",
                max_tokens=8000,
                temperature=0.2,
                top_p=0.95,
                max_retries=3,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_name"] == "compound-beta"
        assert kwargs["max_tokens"] == 8000
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_retries"] == 3

    def test_optional_params_absent_when_not_provided(self):
        """Verify optional params are absent when not provided."""
        mock_cls = MagicMock()
        with _mock_module("langchain_groq", ChatGroq=mock_cls):
            GroqProvider().load(model_name="llama-3.1-8b-instant")
        kwargs = mock_cls.call_args[1]
        for absent in ("max_tokens", "temperature", "top_p", "max_retries"):
            assert absent not in kwargs

    def test_extra_kwargs_ignored(self):
        """Verify unknown kwargs do not raise errors."""
        mock_cls = MagicMock()
        with _mock_module("langchain_groq", ChatGroq=mock_cls):
            GroqProvider().load(model_name="compound-beta-mini", seed=9)
        assert mock_cls.called


# ---------------------------------------------------------------------------
# load_model() registry-path integration for new providers
# ---------------------------------------------------------------------------


class TestLoadModelNewProviders:
    """Verify load_model() routes new provider types via the registry."""

    def _run_via_registry(self, provider_type: str) -> MagicMock:
        """Register a stub, call load_model(), return the stub LLM."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_llm = MagicMock()

        class _StubProvider(LLMProvider):
            def load(self, **kwargs):
                return mock_llm

        # Each test uses a unique key to avoid collisions between parallel runs.
        test_key = f"_test_{provider_type}"
        PROVIDER_REGISTRY.register(test_key, _StubProvider)
        try:
            result = load_model(test_key, model_name="any")
            assert result is mock_llm
            return result
        finally:
            PROVIDER_REGISTRY.unregister(test_key)

    def test_anthropic_registry_path(self):
        """Verify load_model delegates to a registered anthropic-style provider."""
        self._run_via_registry("remote_anthropic")

    def test_mistral_registry_path(self):
        """Verify load_model delegates to a registered mistral-style provider."""
        self._run_via_registry("remote_mistral")

    def test_cohere_registry_path(self):
        """Verify load_model delegates to a registered cohere-style provider."""
        self._run_via_registry("remote_cohere")

    def test_google_genai_registry_path(self):
        """Verify load_model delegates to a registered google-genai-style provider."""
        self._run_via_registry("remote_google_genai")

    def test_deepseek_registry_path(self):
        """Verify load_model delegates to a registered deepseek-style provider."""
        self._run_via_registry("remote_deepseek")

    def test_xai_registry_path(self):
        """Verify load_model delegates to a registered xai-style provider."""
        self._run_via_registry("remote_xai")

    def test_groq_registry_path(self):
        """Verify load_model delegates to a registered groq-style provider."""
        self._run_via_registry("remote_groq")

    @pytest.mark.parametrize(
        "model_id,expected_provider",
        [
            # Anthropic direct: claude-* IDs registered under remote_anthropic,
            # NOT under Bedrock (those use the anthropic.claude-* prefix).
            ("claude-sonnet-4-6", "remote_anthropic"),
            ("claude-opus-4-8", "remote_anthropic"),
            ("claude-haiku-4-5", "remote_anthropic"),
            # Mistral: registered under remote_mistral (not Bedrock's mistral.*)
            ("mistral-large-latest", "remote_mistral"),
            ("codestral-latest", "remote_mistral"),
            # Cohere: Command family registered under remote_cohere
            ("command-r-plus", "remote_cohere"),
            ("command-a-plus-05-2026", "remote_cohere"),
            # Google GenAI: use the disambiguation display name "(Direct API)" so
            # the LLM_MODELS lookup selects remote_google_genai, not Vertex.
            # The raw model_id (e.g. "gemini-2.5-flash") also exists in the
            # Vertex catalog; passing the raw ID resolves to Vertex (first match
            # in insertion order) -- that's tested in test_raw_gemini_resolves_to_vertex.
            ("Gemini 2.5 Flash (Direct API)", "remote_google_genai"),
            ("Gemini 2.0 Flash (Direct API)", "remote_google_genai"),
            ("Gemini 2.0 Flash Lite (Direct API)", "remote_google_genai"),
            # DeepSeek: registered under remote_deepseek
            ("deepseek-chat", "remote_deepseek"),
            ("deepseek-reasoner", "remote_deepseek"),
            # xAI: registered under remote_xai
            ("grok-3-latest", "remote_xai"),
            # Groq: registered under remote_groq
            ("llama-3.3-70b-versatile", "remote_groq"),
            ("compound-beta", "remote_groq"),
            ("compound-beta-mini", "remote_groq"),
        ],
    )
    def test_llm_models_lookup_resolves_new_providers(
        self, model_id: str, expected_provider: str
    ):
        """Verify display names in LLM_MODELS resolve to the correct new provider.

        Google GenAI entries use display names that include "(Direct API)" to
        distinguish them from the identical model_id values registered under
        remote_google_vertex.  Users who want the Google AI Developer API
        (GOOGLE_API_KEY, not GCP/Vertex credentials) must select a model by its
        display name or invoke load_model() with provider_type explicitly set.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_model,
        )

        provider, _ = resolve_model(model_id)
        assert provider == expected_provider, (
            f"Expected model '{model_id}' to resolve to '{expected_provider}', "
            f"got '{provider}'"
        )

    def test_raw_gemini_model_id_resolves_to_vertex(self):
        """Verify raw gemini-* model IDs resolve to Vertex (first catalog match).

        Because the same gemini-2.5-flash model_id exists in both
        remote_google_vertex (registered first in LLM_MODELS) and
        remote_google_genai, the raw ID resolves to Vertex.  This is expected
        and backward-compatible.  Users reach remote_google_genai via the
        '(Direct API)' display-name catalog entries or by calling load_model()
        with an explicit provider_type.
        """
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_model,
        )

        provider, model_id = resolve_model("gemini-2.5-flash")
        assert (
            provider == "remote_google_vertex"
        ), "Raw 'gemini-2.5-flash' must resolve to Vertex (first catalog match)"
        assert model_id == "gemini-2.5-flash"
