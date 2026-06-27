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


# ---------------------------------------------------------------------------
# Catalog-wide hardening: uniqueness, enum correctness, Bedrock ID format
# ---------------------------------------------------------------------------

#: The Bedrock model_id prefixes that are known valid.  Every entry in
#: remote_aws_bedrock must start with one of these.  Extend this set when
#: AWS adds a new vendor or a new geo prefix (eu., global.) to the catalog.
_BEDROCK_VALID_PREFIXES = (
    "amazon.",
    "us.amazon.",
    "ai21.",
    "us.ai21.",
    "anthropic.",
    "us.anthropic.",
    "cohere.",
    "us.cohere.",
    "deepseek.",
    "us.deepseek.",
    "eu.deepseek.",
    "global.deepseek.",
    "meta.",
    "us.meta.",
    "eu.meta.",
    "mistral.",
    "us.mistral.",
    "eu.mistral.",
    "us.twelvelabs.",
)

#: Legal tool_strategy values from PR #215.
_KNOWN_TOOL_STRATEGIES = {"native", "facilitated", "mcp", "none"}


def _all_model_entries():
    """Yield (provider_key, model_dict) for every entry in LLM_MODELS."""
    for pkey, pval in LLM_MODELS.items():
        for entry in pval["models"]:
            yield pkey, entry


class TestCatalogHardening:
    """Cross-catalog invariants that CI must enforce to catch typos early."""

    def test_model_ids_are_unique_within_each_provider(self):
        """Within each provider section, every model_id must be unique.

        The same model_id can legitimately appear in more than one provider
        (e.g., gpt-4o in both remote_openai and remote_azure_openai) because
        the same upstream model may be accessible through different API paths.
        The dangerous error is a duplicate within a SINGLE provider section,
        which almost always means a copy-paste block where the model_id was
        not updated.
        """
        for pkey, pval in LLM_MODELS.items():
            ids = [entry["model_id"] for entry in pval["models"]]
            seen: set = set()
            dupes = []
            for mid in ids:
                if mid in seen:
                    dupes.append(mid)
                seen.add(mid)
            assert not dupes, f"Duplicate model_ids within provider '{pkey}': {dupes}"

    def test_tool_strategy_values_are_valid_enum(self):
        """Every model entry that declares tool_strategy uses a known value.

        Invalid values silently fall through the routing logic and route as
        though the model has no tool support — wrong behavior, hard to debug.
        """
        bad = []
        for pkey, entry in _all_model_entries():
            strat = entry.get("tool_strategy")
            if strat is not None and strat not in _KNOWN_TOOL_STRATEGIES:
                bad.append(
                    f"{pkey}/{entry['model_name']}: "
                    f"tool_strategy={strat!r} not in {_KNOWN_TOOL_STRATEGIES}"
                )
        assert not bad, "Unknown tool_strategy values:\n" + "\n".join(bad)

    def test_supports_tools_matches_tool_strategy(self):
        """supports_tools must equal (tool_strategy == 'native') when both present.

        Additionally, the two fields must appear together: an entry that declares
        one but omits the other violates the schema and would silently hide a
        real mismatch.  Entries that omit BOTH are fine (legacy entries that
        predate the tool_strategy field).

        This invariant is the backward-compat contract from PR #215.
        """
        errors = []
        for pkey, entry in _all_model_entries():
            strat = entry.get("tool_strategy")
            stated = entry.get("supports_tools")
            name = f"{pkey}/{entry['model_name']}"

            # Skip entries that declare neither field (pre-tool_strategy legacy rows).
            if strat is None and stated is None:
                continue

            # If only one field is present the schema is incomplete.
            if strat is None:
                errors.append(
                    f"{name}: supports_tools={stated!r} present but tool_strategy missing"
                )
                continue
            if stated is None:
                errors.append(
                    f"{name}: tool_strategy={strat!r} present but supports_tools missing"
                )
                continue

            # Both present — verify they agree.
            expected = strat == "native"
            if stated != expected:
                errors.append(
                    f"{name}: tool_strategy={strat!r} implies supports_tools={expected}, "
                    f"but got {stated!r}"
                )

        assert (
            not errors
        ), "supports_tools/tool_strategy invariant violations:\n" + "\n".join(errors)

    def test_bedrock_model_ids_start_with_known_vendor_prefix(self):
        """Every remote_aws_bedrock entry must start with a recognized prefix.

        Note: newer Bedrock model IDs do NOT always end with a version
        qualifier (e.g., Mistral Ministral 3 / Large 3 / Devstral 2, and the
        Claude 4 family omit -v1:0).  The prefix check catches wrong IDs
        (e.g., a GCP / OpenAI ID accidentally listed under Bedrock) without
        falsely rejecting the no-suffix convention.
        """
        bad = []
        for entry in LLM_MODELS["remote_aws_bedrock"]["models"]:
            mid = entry["model_id"]
            if not any(mid.startswith(p) for p in _BEDROCK_VALID_PREFIXES):
                bad.append(
                    f"{entry['model_name']}: model_id={mid!r} does not start "
                    f"with a known Bedrock vendor prefix"
                )
        assert not bad, (
            "Bedrock model_ids with unrecognized vendor prefix "
            "(add the new prefix to _BEDROCK_VALID_PREFIXES if correct):\n"
            + "\n".join(bad)
        )
