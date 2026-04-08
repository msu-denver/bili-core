"""Tests for bili.iris.config.llm_config structure and content."""

import pytest

from bili.iris.config.llm_config import LLM_MODELS

# ---------------------------------------------------------------------------
# LLM_MODELS top-level structure
# ---------------------------------------------------------------------------


class TestLlmModelsStructure:
    """Verify that LLM_MODELS is well-formed and non-empty."""

    EXPECTED_PROVIDERS = {
        "remote_aws_bedrock",
        "remote_google_vertex",
        "remote_azure_openai",
        "remote_openai",
        "local_llamacpp",
        "local_huggingface",
    }

    def test_llm_models_is_non_empty_dict(self):
        """Verify LLM_MODELS is a non-empty dictionary."""
        assert isinstance(LLM_MODELS, dict)
        assert len(LLM_MODELS) > 0

    def test_contains_expected_providers(self):
        """Verify all expected provider keys are present."""
        assert self.EXPECTED_PROVIDERS.issubset(LLM_MODELS.keys())

    @pytest.mark.parametrize("provider_key", list(LLM_MODELS.keys()))
    def test_provider_has_required_top_level_fields(self, provider_key):
        """Verify each provider has name, description, and models fields."""
        provider = LLM_MODELS[provider_key]
        assert "name" in provider, f"'{provider_key}' missing 'name'"
        assert "description" in provider, f"'{provider_key}' missing 'description'"
        assert "models" in provider, f"'{provider_key}' missing 'models'"

    @pytest.mark.parametrize("provider_key", list(LLM_MODELS.keys()))
    def test_models_list_is_non_empty(self, provider_key):
        """Verify each provider has a non-empty models list."""
        models = LLM_MODELS[provider_key]["models"]
        assert isinstance(models, list)
        assert len(models) > 0


# ---------------------------------------------------------------------------
# Individual model entry structure
# ---------------------------------------------------------------------------


class TestModelEntryStructure:
    """Each model entry should have a consistent set of fields."""

    REQUIRED_MODEL_FIELDS = {"model_name", "model_id"}

    @pytest.fixture(
        params=[
            (pkey, idx)
            for pkey, pval in LLM_MODELS.items()
            for idx in range(len(pval["models"]))
        ],
        ids=[
            f"{pkey}:{pval['models'][idx]['model_name']}"
            for pkey, pval in LLM_MODELS.items()
            for idx in range(len(pval["models"]))
        ],
    )
    def model_entry(self, request):
        """Return a single model entry for parametrized testing."""
        pkey, idx = request.param
        return LLM_MODELS[pkey]["models"][idx]

    def test_model_has_required_fields(self, model_entry):
        """Verify each model entry contains all required fields."""
        for field in self.REQUIRED_MODEL_FIELDS:
            assert (
                field in model_entry
            ), f"Model '{model_entry.get('model_name', '?')}' missing '{field}'"

    def test_model_name_is_non_empty_string(self, model_entry):
        """Verify model_name is a non-empty string."""
        assert isinstance(model_entry["model_name"], str)
        assert len(model_entry["model_name"].strip()) > 0

    def test_model_id_is_non_empty_string(self, model_entry):
        """Verify model_id is a non-empty string."""
        assert isinstance(model_entry["model_id"], str)
        assert len(model_entry["model_id"].strip()) > 0
