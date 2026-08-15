"""Currency guards for the model catalog's current-generation entries.

These pin the per-model values the loader depends on for the newest model
families, so a later edit that drops or mistypes one fails here rather than at
runtime.  The output-budget check matters because ``load_model`` reads
``max_output_tokens`` from the catalog to fill ``max_tokens`` for Anthropic and
Bedrock: an uncataloged model falls back to the provider floor instead, which is
far below a real authoring budget.
"""

# pylint: disable=too-few-public-methods

from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.providers.catalog_defaults import model_max_output_tokens
from bili.iris.providers.temperature_resilience import model_supports_temperature


def _entry(model_type, model_id):
    """Return the catalog entry for ``model_id`` under ``model_type``, or None."""
    for entry in LLM_MODELS.get(model_type, {}).get("models", []):
        if entry.get("model_id") == model_id:
            return entry
    return None


class TestAnthropicCurrentModels:
    """The current Anthropic families resolve to a real output budget."""

    def test_opus_5_output_budget(self):
        """Opus 5 carries a multi-KB authoring budget, not the provider floor."""
        assert model_max_output_tokens("remote_anthropic", "claude-opus-5") == 32000

    def test_sonnet_5_output_budget(self):
        """Sonnet 5 carries a multi-KB authoring budget, not the provider floor."""
        assert model_max_output_tokens("remote_anthropic", "claude-sonnet-5") == 16000

    def test_haiku_dated_snapshot_resolves(self):
        """The dated Haiku snapshot resolves the same as its bare alias, so
        either string a caller passes gets a catalog budget."""
        alias = model_max_output_tokens("remote_anthropic", "claude-haiku-4-5")
        dated = model_max_output_tokens("remote_anthropic", "claude-haiku-4-5-20251001")
        assert alias is not None
        assert dated == alias

    def test_current_families_support_temperature(self):
        """The Claude 5 families accept temperature (not reasoning-restricted)."""
        for model_id in ("claude-opus-5", "claude-sonnet-5"):
            assert model_supports_temperature("remote_anthropic", model_id) is True


class TestOpenAICurrentModels:
    """The GPT-5.6 family is cataloged; o1 is corrected to reject temperature."""

    def test_gpt_5_6_alias_and_variants_present(self):
        """The alias and all three variants carry the published limits."""
        for model_id in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            entry = _entry("remote_openai", model_id)
            assert entry is not None, model_id
            assert entry["max_input_tokens"] == 1050000
            assert entry["max_output_tokens"] == 128000

    def test_gpt_5_6_family_omits_temperature(self):
        """Reasoning models reject a non-default temperature, so the loader
        must omit it (catalog flag False)."""
        for model_id in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert model_supports_temperature("remote_openai", model_id) is False

    def test_o1_rejects_temperature(self):
        """o1 is a reasoning model; the corrected flag makes the loader omit
        temperature (matching o1-mini / o3-mini)."""
        assert model_supports_temperature("remote_openai", "o1") is False


class TestGeminiCurrentModels:
    """The Gemini 3.x pro/flash/flash-lite set is cataloged on the direct API."""

    def test_gemini_3_family_present(self):
        """Pro, flash, and flash-lite resolve with a 1M input window."""
        for model_id in (
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
        ):
            entry = _entry("remote_google_genai", model_id)
            assert entry is not None, model_id
            assert entry["max_input_tokens"] == 1048576
