"""Tests for the provider allowlists and id normalisation."""

import pytest

from bili.iris.config.catalog_divergence.mapping import (
    ADVISORY_ONLY_PROVIDER_TYPES,
    LITELLM_PROVIDERS,
    MODELS_DEV_PROVIDERS,
    UNLISTED_PROVIDER_TYPES,
    id_candidates,
)
from bili.iris.config.llm_config import LLM_MODELS


class TestAllowlists:
    """The mapping tables must describe the catalog they are read against."""

    @pytest.mark.parametrize("provider_type", sorted(MODELS_DEV_PROVIDERS))
    def test_models_dev_keys_exist_in_the_catalog(self, provider_type):
        """Every mapped provider type is a real catalog provider type."""
        assert provider_type in LLM_MODELS

    @pytest.mark.parametrize("provider_type", sorted(LITELLM_PROVIDERS))
    def test_litellm_keys_exist_in_the_catalog(self, provider_type):
        """Every mapped provider type is a real catalog provider type."""
        assert provider_type in LLM_MODELS

    @pytest.mark.parametrize("provider_type", sorted(UNLISTED_PROVIDER_TYPES))
    def test_unlisted_keys_exist_in_the_catalog(self, provider_type):
        """Every unlisted provider type is a real catalog provider type."""
        assert provider_type in LLM_MODELS

    def test_unlisted_types_are_never_also_mapped(self):
        """A type is either expected to match or excluded, never both.

        Mapping a type and simultaneously declaring it unlisted would silence
        its coverage findings while still resolving records for it, which is
        two answers to one question.
        """
        mapped = set(MODELS_DEV_PROVIDERS) | set(LITELLM_PROVIDERS)
        assert not mapped & UNLISTED_PROVIDER_TYPES

    def test_every_catalog_type_is_mapped_or_unlisted(self):
        """No catalog provider type is silently unaccounted for.

        A type that is neither mapped nor declared unlisted would produce a
        coverage finding per entry with no way to tell an id-mapper gap from a
        provider that has no upstream listing at all.
        """
        mapped = set(MODELS_DEV_PROVIDERS) | set(LITELLM_PROVIDERS)
        unaccounted = set(LLM_MODELS) - mapped - UNLISTED_PROVIDER_TYPES
        assert not unaccounted, f"unaccounted provider types: {sorted(unaccounted)}"

    def test_advisory_types_are_mapped(self):
        """An advisory-only type is one that IS looked up, with capped severity."""
        mapped = set(MODELS_DEV_PROVIDERS) | set(LITELLM_PROVIDERS)
        assert ADVISORY_ONLY_PROVIDER_TYPES <= mapped


class TestIdCandidates:
    """Normalisation widens the search; it never rewrites the catalog."""

    def test_the_literal_id_is_always_first(self):
        """An exact upstream key must win over any normalised spelling."""
        assert id_candidates("remote_openai", "gpt-4")[0] == "gpt-4"

    def test_an_unmapped_provider_type_gets_only_the_literal(self):
        """A type with no normalisation rule yields exactly one candidate."""
        assert id_candidates("remote_openai", "gpt-4") == ["gpt-4"]

    def test_managed_inference_id_drops_region_vendor_and_version(self):
        """A region-, vendor- and version-qualified id widens to each form."""
        candidates = id_candidates(
            "remote_aws_bedrock", "us.anthropic.claude-sonnet-4-v1:0"
        )
        assert candidates[0] == "us.anthropic.claude-sonnet-4-v1:0"
        assert "anthropic.claude-sonnet-4-v1:0" in candidates
        assert "claude-sonnet-4-v1:0" in candidates
        assert "claude-sonnet-4" in candidates

    def test_managed_inference_id_without_a_region_still_drops_the_vendor(self):
        """The vendor strip does not depend on a region prefix being present."""
        candidates = id_candidates("remote_aws_bedrock", "amazon.nova-pro-v1:0")
        assert candidates[0] == "amazon.nova-pro-v1:0"
        assert "nova-pro-v1:0" in candidates
        assert "nova-pro" in candidates

    def test_managed_inference_id_with_no_decoration_is_unchanged(self):
        """An id with nothing to strip yields itself alone."""
        assert id_candidates("remote_aws_bedrock", "plainmodel") == ["plainmodel"]

    def test_local_runtime_tag_is_dropped(self):
        """A local runtime tag widens to the untagged name."""
        assert id_candidates("local_ollama", "qwen3:8b") == ["qwen3:8b", "qwen3"]

    def test_local_runtime_id_without_a_tag_is_unchanged(self):
        """An untagged local id yields itself alone."""
        assert id_candidates("local_ollama", "qwen3") == ["qwen3"]

    def test_generative_api_path_prefix_is_dropped(self):
        """A path-qualified generative-API id widens to the bare id."""
        assert id_candidates("remote_google_genai", "models/gemini-2.5-pro") == [
            "models/gemini-2.5-pro",
            "gemini-2.5-pro",
        ]

    def test_generative_api_id_without_a_prefix_is_unchanged(self):
        """An unqualified generative-API id yields itself alone."""
        assert id_candidates("remote_google_vertex", "gemini-2.5-pro") == [
            "gemini-2.5-pro"
        ]

    def test_candidates_are_deduplicated(self):
        """A normalisation that produces a repeat must not repeat it."""
        candidates = id_candidates("remote_aws_bedrock", "us.plainmodel")
        assert len(candidates) == len(set(candidates))
