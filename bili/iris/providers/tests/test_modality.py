"""Tests for bili.iris.providers.modality: per-model input-modality capability.

Two halves.  The first pins the reader and the refusal, including the
tri-state that makes an undeclared model degrade to a warning rather than to
either a false refusal or a false assurance.  The second pins the catalog data
itself: a declaration is a capability claim bili-core makes to its callers, so
a new entry that arrives without one, or with a malformed one, fails the build
instead of shipping an unverifiable claim.
"""

import logging
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import bili.iris.providers.modality as modality_module
from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.multimodal import image_part, text_part
from bili.iris.providers.cli_presets import CLI_PRESET_CATALOG
from bili.iris.providers.modality import (
    AUDIO,
    CATALOG_KEY,
    IMAGE,
    IMAGE_DELIVERY_BYTES,
    IMAGE_DELIVERY_CATALOG_KEY,
    IMAGE_DELIVERY_OFFERED_BY_PATH,
    KNOWN_IMAGE_DELIVERIES,
    KNOWN_INPUT_MODALITIES,
    TEXT,
    UnsupportedInputModalityError,
    describe_message_modalities,
    image_delivery_kind,
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
# string.  Derived from the catalog so a new CLI preset is covered
# automatically.
CLI_PROVIDER_TYPES = frozenset(
    key for key in LLM_MODELS if key == "cli" or key.startswith("cli_")
)

# The CLI types split on one question: does bili-core know how to point this
# tool at a file?  A named preset does (its harness is known, so an image is
# materialized in the subprocess's working directory and the invocation
# points at it); the generic type drives an arbitrary executable, so it does
# not, and an image part sent to it is refused.  Derived from the shipped
# preset catalog rather than listed, so the two halves cannot fall out of
# step with what actually ships.
CLI_TYPES_WITH_AN_IMAGE_ROUTE = frozenset(
    provider_type
    for provider_type in CLI_PROVIDER_TYPES
    if (CLI_PRESET_CATALOG.get(provider_type) is not None)
    and CLI_PRESET_CATALOG[provider_type].image_route is not None
)
CLI_TYPES_WITHOUT_AN_IMAGE_ROUTE = CLI_PROVIDER_TYPES - CLI_TYPES_WITH_AN_IMAGE_ROUTE

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

    def test_a_cli_type_with_no_file_read_route_is_absent(self):
        """A CLI type bili-core cannot point at a file is absent.

        The routing direction answers "known to accept it".  A CLI tool
        bili-core knows nothing about has no channel for an image at all, so
        listing it would route a caller holding one to a provider that will
        refuse the turn.
        """
        result = models_supporting_input_modality(IMAGE)
        assert CLI_TYPES_WITHOUT_AN_IMAGE_ROUTE
        assert not CLI_TYPES_WITHOUT_AN_IMAGE_ROUTE & set(result)

    def test_a_cli_type_with_a_file_read_route_is_listed(self):
        """A CLI preset whose harness can open a file is listed.

        The routing direction is only useful if it resolves; a route that
        shipped without the catalog declaring it would leave the model
        refused at selection for an image the provider could have delivered.
        """
        result = models_supporting_input_modality(IMAGE)
        assert CLI_TYPES_WITH_AN_IMAGE_ROUTE
        for provider_type in CLI_TYPES_WITH_AN_IMAGE_ROUTE:
            assert result.get(provider_type), provider_type

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

    def test_a_cli_provider_with_no_file_read_route_is_declared_text_only(self):
        """Structural, not a vendor claim.

        A CLI provider renders the message list to one prompt string, so the
        only way an image reaches the model behind it is as a file the
        harness opens.  With no known way to point that harness at a file,
        the entry declares text only and the transport refuses an image part
        by name.
        """
        assert CLI_TYPES_WITHOUT_AN_IMAGE_ROUTE, "expected at least one such type"
        for provider_type in CLI_TYPES_WITHOUT_AN_IMAGE_ROUTE:
            for entry in LLM_MODELS[provider_type]["models"]:
                assert entry.get(CATALOG_KEY) == [TEXT], (provider_type, entry)

    def test_a_cli_provider_with_a_file_read_route_declares_delivery_by_path(self):
        """The declaration says how, not just whether.

        Reading these entries as plain image support would put them in the
        same bucket as a provider handed the actual bytes.  A harness is
        offered a path and may not open it, and that difference is not
        recoverable from the response, so the catalog states it.
        """
        assert CLI_TYPES_WITH_AN_IMAGE_ROUTE, "expected at least one such type"
        for provider_type in CLI_TYPES_WITH_AN_IMAGE_ROUTE:
            for entry in LLM_MODELS[provider_type]["models"]:
                assert entry.get(CATALOG_KEY) == [TEXT, IMAGE], (provider_type, entry)
                assert (
                    image_delivery_kind(provider_type, entry["model_id"])
                    == IMAGE_DELIVERY_OFFERED_BY_PATH
                ), (provider_type, entry)

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

    def test_a_legacy_text_only_snapshot_does_not_claim_vision(self):
        """A model id whose vendor-published capability is text-only must
        not declare "image", even when a sibling entry for a *different*
        model id (a later, actually vision-capable snapshot) legitimately
        does. A declaration that outstrips the real model would let an
        image-bearing request reach a text-only endpoint undetected by the
        modality gate this module implements."""
        assert model_input_modalities("remote_openai", "gpt-4") == frozenset({TEXT})


class TestImageDeliveryKind:
    """How an image gets to the model, which is a different question from
    whether the model accepts one.

    Reading the two as one fact is what this exists to prevent: a caller
    auditing what happened to an image needs to distinguish a provider that
    was handed the bytes from a harness that was offered a path and may
    never have opened it.
    """

    def test_a_message_based_image_model_delivers_bytes(self):
        """The default, so no existing entry had to declare anything."""
        assert image_delivery_kind(VISION_TYPE, VISION_MODEL) == IMAGE_DELIVERY_BYTES

    def test_a_text_only_model_delivers_nothing(self):
        """``None`` rather than ``bytes``.

        A model that takes no image delivers no image, and reporting
        ``bytes`` there would be a positive claim about a capability the
        entry explicitly denies.
        """
        assert image_delivery_kind(TEXT_TYPE, TEXT_MODEL) is None

    def test_an_uncataloged_model_delivers_nothing(self):
        """The same "no record" state the modality reader returns.

        bili-core cannot assert how an image reaches a model it has never
        heard of, and this reader must not invent the answer the tri-state
        modality reader deliberately withholds.
        """
        assert image_delivery_kind(VISION_TYPE, "some-unlisted-model") is None
        assert image_delivery_kind("no_such_provider", VISION_MODEL) is None

    def test_an_unknown_declared_kind_degrades_to_bytes_with_a_warning(self, caplog):
        """An unreadable declaration is surfaced, not obeyed.

        Returning the unknown string would push a value the rest of the
        system cannot interpret into a caller's audit trail; failing the load
        over a metadata typo would take out a model that works.
        """
        entry = dict(LLM_MODELS[VISION_TYPE]["models"][0])
        with patch.object(
            modality_module,
            "_catalog_entry",
            return_value={**entry, IMAGE_DELIVERY_CATALOG_KEY: "carrier-pigeon"},
        ):
            with caplog.at_level(logging.WARNING):
                assert (
                    image_delivery_kind(VISION_TYPE, VISION_MODEL)
                    == IMAGE_DELIVERY_BYTES
                )
        assert "carrier-pigeon" in caplog.text

    def test_the_two_known_kinds_are_the_whole_vocabulary(self):
        """Both names, and only those two, so a caller can switch on them."""
        assert KNOWN_IMAGE_DELIVERIES == {
            IMAGE_DELIVERY_BYTES,
            IMAGE_DELIVERY_OFFERED_BY_PATH,
        }

    def test_every_declared_delivery_in_the_catalog_is_a_known_kind(self):
        """A typo in an entry would otherwise sit unread until something
        branched on it."""
        for provider_info in LLM_MODELS.values():
            for entry in provider_info.get("models", []):
                declared = entry.get(IMAGE_DELIVERY_CATALOG_KEY)
                if declared is not None:
                    assert declared in KNOWN_IMAGE_DELIVERIES, entry["model_id"]

    def test_only_an_image_capable_entry_declares_a_delivery(self):
        """Declaring how an image arrives at a model that takes none is a
        statement about nothing, and would read as image support to anyone
        scanning for the key."""
        for provider_info in LLM_MODELS.values():
            for entry in provider_info.get("models", []):
                if IMAGE_DELIVERY_CATALOG_KEY in entry:
                    assert IMAGE in (entry.get(CATALOG_KEY) or ()), entry["model_id"]
