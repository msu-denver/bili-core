"""Tests for bili.iris.providers.modality: per-model input-modality capability.

Two halves.  The first pins the reader and the refusal, including the
tri-state that makes an undeclared model degrade to a warning rather than to
either a false refusal or a false assurance.  The second pins the catalog data
itself: a declaration is a capability claim bili-core makes to its callers, so
a new entry that arrives without one, or with a malformed one, fails the build
instead of shipping an unverifiable claim.
"""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.multimodal import image_part, text_part
from bili.iris.providers.modality import (
    AUDIO,
    CATALOG_KEY,
    IMAGE,
    KNOWN_INPUT_MODALITIES,
    TEXT,
    UnsupportedInputModalityError,
    describe_message_modalities,
    model_input_modalities,
    models_supporting_input_modality,
    require_input_modalities,
    require_input_modality,
    supports_input_modality,
)

# A cataloged model declared image-capable, and one declared text-only.  Both
# are read out of the catalog rather than hardcoded, so the tests keep meaning
# something if either entry is retired.
VISION_TYPE, VISION_MODEL = "remote_openai", "gpt-4o"
TEXT_TYPE, TEXT_MODEL = "remote_openai", "gpt-35-turbo"

# Provider types whose transport collapses a message list to one prompt
# string, so text-only is a structural fact about bili-core rather than a
# vendor capability claim.  Derived from the catalog so a new CLI preset is
# covered automatically.
CLI_PROVIDER_TYPES = frozenset(
    key for key in LLM_MODELS if key == "cli" or key.startswith("cli_")
)

# In-process local providers wrap text-completion pipelines.
IN_PROCESS_LOCAL_TYPES = frozenset({"local_llamacpp", "local_huggingface"})

#: Catalog entries that deliberately declare nothing, each because bili-core
#: has no defensible record of what the model accepts: a moving ``-latest``
#: alias, a model whose input kind is outside this vocabulary (video), or a
#: forward-dated entry whose capability is not established.  Asserted exactly
#: so a NEW entry cannot join them silently -- an omission has to be a
#: decision someone made, not a field someone forgot.
EXPECTED_UNDECLARED = {
    ("remote_aws_bedrock", "mistral.mistral-large-3-675b-instruct"),
    ("remote_aws_bedrock", "mistral.ministral-3-3b-instruct"),
    ("remote_aws_bedrock", "mistral.ministral-3-8b-instruct"),
    ("remote_aws_bedrock", "mistral.ministral-3-14b-instruct"),
    ("remote_aws_bedrock", "mistral.devstral-2-123b"),
    ("remote_aws_bedrock", "us.twelvelabs.pegasus-1-2-v1:0"),
    ("remote_mistral", "mistral-large-latest"),
    ("remote_mistral", "mistral-small-latest"),
    ("remote_cohere", "command-a-plus-05-2026"),
    ("remote_xai", "grok-3-latest"),
    ("remote_groq", "compound-beta"),
    ("remote_groq", "compound-beta-mini"),
}


def _all_entries():
    """Yield ``(provider_type, entry)`` for every cataloged model."""
    for provider_type, info in LLM_MODELS.items():
        for entry in info.get("models", []):
            yield provider_type, entry


# ---------------------------------------------------------------------------
# Reading the declaration
# ---------------------------------------------------------------------------


class TestModelInputModalities:
    """The catalog read, and its three states."""

    def test_declared_model_returns_its_modalities(self):
        """Declared model returns its modalities."""
        assert model_input_modalities(VISION_TYPE, VISION_MODEL) == frozenset(
            {TEXT, IMAGE}
        )

    def test_text_only_model_returns_text(self):
        """Text only model returns text."""
        assert model_input_modalities(TEXT_TYPE, TEXT_MODEL) == frozenset({TEXT})

    def test_uncataloged_model_is_undeclared(self):
        """Uncataloged model is undeclared."""
        assert model_input_modalities(VISION_TYPE, "no-such-model") is None

    def test_unknown_provider_type_is_undeclared(self):
        """Unknown provider type is undeclared."""
        assert model_input_modalities("remote_nowhere", VISION_MODEL) is None

    def test_missing_model_name_is_undeclared(self):
        """Missing model name is undeclared."""
        assert model_input_modalities(VISION_TYPE, None) is None

    def test_entry_without_the_key_is_undeclared(self, monkeypatch):
        """Entry without the key is undeclared."""
        entry = dict(LLM_MODELS[VISION_TYPE]["models"][0])
        entry.pop(CATALOG_KEY, None)
        monkeypatch.setitem(LLM_MODELS, "test_provider", {"models": [entry]})
        assert model_input_modalities("test_provider", entry["model_id"]) is None


class TestSupportsInputModality:
    """The tri-state predicate."""

    def test_true_when_declared_and_present(self):
        """True when declared and present."""
        assert supports_input_modality(VISION_TYPE, VISION_MODEL, IMAGE) is True

    def test_false_when_declared_and_absent(self):
        """False when declared and absent."""
        assert supports_input_modality(TEXT_TYPE, TEXT_MODEL, IMAGE) is False

    def test_none_when_undeclared(self):
        """Distinct from False: bili-core has no record, it is not asserting
        that the model refuses."""
        assert supports_input_modality(VISION_TYPE, "no-such-model", IMAGE) is None


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


class TestRequireInputModality:
    """Refusing at selection instead of at the provider call."""

    def test_declared_support_passes(self):
        """Declared support passes."""
        require_input_modality(VISION_TYPE, VISION_MODEL, IMAGE)

    def test_declared_absence_raises(self):
        """Declared absence raises."""
        with pytest.raises(UnsupportedInputModalityError) as excinfo:
            require_input_modality(TEXT_TYPE, TEXT_MODEL, IMAGE)
        message = str(excinfo.value)
        assert TEXT_MODEL in message
        assert TEXT_TYPE in message
        assert "image" in message

    def test_refusal_names_alternatives_of_the_same_provider_type(self):
        """The error is actionable: it says which models of this type do."""
        with pytest.raises(UnsupportedInputModalityError) as excinfo:
            require_input_modality(TEXT_TYPE, TEXT_MODEL, IMAGE)
        assert VISION_MODEL in str(excinfo.value)

    def test_refusal_is_a_valueerror(self):
        """Existing ``except ValueError`` around load_model keeps working."""
        assert issubclass(UnsupportedInputModalityError, ValueError)

    def test_undeclared_model_warns_and_proceeds(self, caplog):
        """Refusing here would block every passthrough model, including a
        locally-pulled vision model the catalog cannot enumerate."""
        with caplog.at_level(logging.WARNING, logger="bili.iris.providers.modality"):
            require_input_modality("local_ollama", "llava:13b", IMAGE)
        assert "declares no input modalities" in caplog.text
        assert "llava:13b" in caplog.text

    def test_unknown_modality_name_is_rejected(self):
        """Unknown modality name is rejected."""
        with pytest.raises(ValueError, match="Unknown input modality"):
            require_input_modality(VISION_TYPE, VISION_MODEL, "video")

    def test_text_is_always_requestable(self):
        """Requesting text succeeds for every declared model."""
        require_input_modality(TEXT_TYPE, TEXT_MODEL, TEXT)


class TestRequireInputModalities:
    """The plural wrapper load_model calls."""

    def test_accepts_a_bare_string(self):
        """Accepts a bare string."""
        require_input_modalities(VISION_TYPE, VISION_MODEL, IMAGE)

    def test_accepts_an_iterable(self):
        """Accepts an iterable."""
        require_input_modalities(VISION_TYPE, VISION_MODEL, [TEXT, IMAGE])

    def test_raises_on_the_first_unsupported(self):
        """Raises on the first unsupported."""
        with pytest.raises(UnsupportedInputModalityError):
            require_input_modalities(TEXT_TYPE, TEXT_MODEL, [TEXT, IMAGE])

    def test_empty_iterable_is_a_no_op(self):
        """An empty requirement checks nothing and raises nothing."""
        require_input_modalities(TEXT_TYPE, TEXT_MODEL, [])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestModelsSupportingInputModality:
    """The other direction: which models accept this kind of input."""

    def test_image_capable_models_are_grouped_by_provider(self):
        """Image capable models are grouped by provider."""
        result = models_supporting_input_modality(IMAGE)
        assert VISION_MODEL in result[VISION_TYPE]

    def test_text_only_models_are_excluded(self):
        """Text only models are excluded."""
        result = models_supporting_input_modality(IMAGE)
        assert TEXT_MODEL not in result[TEXT_TYPE]

    def test_undeclared_models_are_excluded(self):
        """This answers 'known to accept it', never 'might'."""
        result = models_supporting_input_modality(IMAGE)
        for provider_type, model_id in EXPECTED_UNDECLARED:
            assert model_id not in result.get(provider_type, [])

    def test_cli_provider_types_are_absent(self):
        """CLI provider types are absent."""
        result = models_supporting_input_modality(IMAGE)
        assert not CLI_PROVIDER_TYPES & set(result)

    def test_every_provider_declares_text(self):
        """Every provider declares text."""
        result = models_supporting_input_modality(TEXT)
        assert set(result) == set(LLM_MODELS)

    def test_unknown_modality_matches_nothing(self):
        """Unknown modality matches nothing."""
        assert models_supporting_input_modality("video") == {}

    def test_providers_with_no_match_are_omitted(self):
        """Providers with no match are omitted."""
        result = models_supporting_input_modality(AUDIO)
        assert result == {}


# ---------------------------------------------------------------------------
# Deriving the requirement from the messages
# ---------------------------------------------------------------------------


class TestDescribeMessageModalities:
    """Mapping the content-part vocabulary onto the catalog vocabulary."""

    def test_text_only_messages_need_nothing(self):
        """Text only messages need nothing."""
        assert describe_message_modalities([HumanMessage(content="hi")]) == []

    def test_text_part_list_needs_nothing(self):
        """Text part list needs nothing."""
        assert (
            describe_message_modalities([HumanMessage(content=[text_part("hi")])]) == []
        )

    def test_an_image_part_requires_image(self):
        """An image part requires image."""
        messages = [HumanMessage(content=[text_part("hi"), image_part(url="u")])]
        assert describe_message_modalities(messages) == [IMAGE]

    @pytest.mark.parametrize("kind", ["image_url", "image", "input_image"])
    def test_every_image_spelling_maps_to_image(self, kind):
        """Every image spelling maps to image."""
        messages = [HumanMessage(content=[{"type": kind}])]
        assert describe_message_modalities(messages) == [IMAGE]

    def test_audio_maps_to_audio(self):
        """Audio maps to audio."""
        messages = [HumanMessage(content=[{"type": "input_audio"}])]
        assert describe_message_modalities(messages) == [AUDIO]

    def test_result_is_deduplicated_and_sorted(self):
        """Result is deduplicated and sorted."""
        messages = [
            HumanMessage(content=[{"type": "image_url"}, {"type": "audio"}]),
            AIMessage(content=[{"type": "image"}]),
        ]
        assert describe_message_modalities(messages) == [AUDIO, IMAGE]

    def test_a_file_part_maps_to_no_modality(self):
        """Recognised as non-text (so it is never silently dropped) but outside
        the input-modality vocabulary."""
        messages = [HumanMessage(content=[{"type": "file"}])]
        assert describe_message_modalities(messages) == []


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


class TestCatalogDeclarations:
    """The declarations are capability claims; keep them well-formed."""

    def test_every_declared_value_is_a_list_of_known_modalities(self):
        """Every declared value is a list of known modalities."""
        for provider_type, entry in _all_entries():
            declared = entry.get(CATALOG_KEY)
            if declared is None:
                continue
            assert isinstance(declared, list), (provider_type, entry["model_id"])
            assert declared, (provider_type, entry["model_id"])
            assert set(declared) <= KNOWN_INPUT_MODALITIES, (
                provider_type,
                entry["model_id"],
            )

    def test_every_declared_value_includes_text(self):
        """Every model in this catalog is a chat model; one that could not
        take text would not be reachable through this seam at all."""
        for provider_type, entry in _all_entries():
            declared = entry.get(CATALOG_KEY)
            if declared is None:
                continue
            assert TEXT in declared, (provider_type, entry["model_id"])

    def test_declared_values_have_no_duplicates(self):
        """Declared values have no duplicates."""
        for provider_type, entry in _all_entries():
            declared = entry.get(CATALOG_KEY)
            if declared is None:
                continue
            assert len(declared) == len(set(declared)), (
                provider_type,
                entry["model_id"],
            )

    def test_undeclared_entries_are_exactly_the_expected_set(self):
        """A new entry must arrive with a declaration or with a deliberate
        omission recorded here -- never by default."""
        undeclared = {
            (provider_type, entry["model_id"])
            for provider_type, entry in _all_entries()
            if entry.get(CATALOG_KEY) is None
        }
        assert undeclared == EXPECTED_UNDECLARED

    def test_cli_providers_are_declared_text_only(self):
        """Structural, not a vendor claim: a CLI provider renders the message
        list to one prompt string, so it has no channel for an image."""
        assert CLI_PROVIDER_TYPES, "expected at least one CLI provider type"
        for provider_type in CLI_PROVIDER_TYPES:
            for entry in LLM_MODELS[provider_type]["models"]:
                assert entry.get(CATALOG_KEY) == [TEXT], (provider_type, entry)

    def test_in_process_local_providers_are_declared_text_only(self):
        """In process local providers are declared text only."""
        for provider_type in IN_PROCESS_LOCAL_TYPES:
            for entry in LLM_MODELS[provider_type]["models"]:
                assert entry.get(CATALOG_KEY) == [TEXT], (provider_type, entry)

    def test_at_least_one_image_capable_model_per_major_api_family(self):
        """The catalog is only useful for routing if it actually resolves; a
        family that lost every declaration would make the check vacuous."""
        result = models_supporting_input_modality(IMAGE)
        for provider_type in (
            "remote_openai",
            "remote_azure_openai",
            "remote_anthropic",
            "remote_aws_bedrock",
            "remote_google_vertex",
            "remote_google_genai",
        ):
            assert result.get(provider_type), provider_type
