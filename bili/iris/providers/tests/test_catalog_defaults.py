"""Tests for catalog-derived max_tokens defaulting.

Without this, a provider's own fallback caps output (langchain_anthropic uses
1024), too small for an agent to emit a multi-KB document as a required tool
argument.  The catalog declares the correct per-model ``max_output_tokens``;
``load_model`` now applies it when the caller supplies no ``max_tokens``.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from bili.iris.loaders.llm_loader import load_model
from bili.iris.providers import LLMProvider, register_provider
from bili.iris.providers.catalog_defaults import model_max_output_tokens

_MAX_TOKENS_LOOKUP = "bili.iris.providers.catalog_defaults.model_max_output_tokens"
_PROVIDERS_CONST = "bili.iris.loaders.llm_loader._CATALOG_MAX_TOKENS_PROVIDERS"


def _model_stub():
    """A model object that is a no-op for temperature resilience (temperature
    is None, so the loaded-model wrapper leaves it untouched)."""
    model = MagicMock()
    model.temperature = None
    return model


def _recording_provider():
    """Return an LLMProvider subclass that records the kwargs it is loaded with."""
    seen: dict = {}

    class _Rec(LLMProvider):  # pylint: disable=too-few-public-methods
        def load(self, **kwargs):  # pylint: disable=arguments-differ
            seen.update(kwargs)
            return _model_stub()

    return _Rec, seen


# ---------------------------------------------------------------------------
# model_max_output_tokens
# ---------------------------------------------------------------------------


class TestModelMaxOutputTokens:
    """The catalog lookup returns the per-model output cap, or None."""

    def test_cataloged_anthropic_model(self):
        """A cataloged Anthropic model returns its declared max_output_tokens."""
        assert model_max_output_tokens("remote_anthropic", "claude-sonnet-4-6") == 16000

    def test_cataloged_openai_model(self):
        """A cataloged OpenAI model returns its declared max_output_tokens."""
        assert model_max_output_tokens("remote_openai", "gpt-4o") == 16384

    def test_opt_out_model_returns_none(self):
        """A model that declares supports_max_output_tokens=False opts out."""
        assert model_max_output_tokens("remote_openai", "o3-mini") is None

    def test_uncataloged_model_returns_none(self):
        """A passthrough (uncataloged) model returns None."""
        assert model_max_output_tokens("remote_anthropic", "claude-sonnet-5") is None

    def test_unknown_provider_returns_none(self):
        """An unknown provider type returns None."""
        assert model_max_output_tokens("not_a_provider", "x") is None

    def test_none_model_name_returns_none(self):
        """A missing model name returns None."""
        assert model_max_output_tokens("remote_anthropic", None) is None

    def test_missing_config_returns_none(self):
        """When the catalog cannot be imported, the lookup degrades to None."""
        with patch.dict(sys.modules, {"bili.iris.config.llm_config": None}):
            assert (
                model_max_output_tokens("remote_anthropic", "claude-sonnet-4-6") is None
            )


# ---------------------------------------------------------------------------
# load_model wiring
# ---------------------------------------------------------------------------


class TestLoadModelMaxTokensDefault:
    """load_model applies the catalog default only when the caller omits it."""

    def test_defaults_from_catalog_when_omitted(self):
        """No caller max_tokens + a cataloged value -> the provider gets it."""
        provider, seen = _recording_provider()
        register_provider("remote_maxtok_default", provider)
        with patch(_MAX_TOKENS_LOOKUP, return_value=16000), patch(
            _PROVIDERS_CONST, ("remote_maxtok_default",)
        ):
            load_model("remote_maxtok_default", model_name="x")
        assert seen.get("max_tokens") == 16000

    def test_caller_max_tokens_wins(self):
        """A caller-supplied max_tokens is not overridden by the catalog."""
        provider, seen = _recording_provider()
        register_provider("remote_maxtok_caller", provider)
        with patch(_MAX_TOKENS_LOOKUP, return_value=16000) as lookup, patch(
            _PROVIDERS_CONST, ("remote_maxtok_caller",)
        ):
            load_model("remote_maxtok_caller", model_name="x", max_tokens=512)
        assert seen.get("max_tokens") == 512
        # The lookup is not even consulted when the caller supplied a value.
        lookup.assert_not_called()

    def test_passthrough_model_gets_no_max_tokens(self):
        """A model with no cataloged value leaves max_tokens unset."""
        provider, seen = _recording_provider()
        register_provider("remote_maxtok_passthrough", provider)
        with patch(_MAX_TOKENS_LOOKUP, return_value=None), patch(
            _PROVIDERS_CONST, ("remote_maxtok_passthrough",)
        ):
            load_model("remote_maxtok_passthrough", model_name="x")
        assert "max_tokens" not in seen

    def test_provider_outside_the_set_gets_no_default(self):
        """A provider not in the catalog-default set is left untouched, even with
        a cataloged value (avoids capping providers that default sensibly)."""
        provider, seen = _recording_provider()
        register_provider("remote_maxtok_excluded", provider)
        with patch(_MAX_TOKENS_LOOKUP, return_value=16000) as lookup, patch(
            _PROVIDERS_CONST, ("remote_anthropic", "remote_aws_bedrock")
        ):
            load_model("remote_maxtok_excluded", model_name="x")
        assert "max_tokens" not in seen
        lookup.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
