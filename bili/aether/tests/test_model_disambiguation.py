"""Resolution of a model name several providers catalog.

The cases over the collisions are DERIVED from ``LLM_MODELS``: a catalog edge
that adds or removes a duplicated id changes what is asserted, rather than
leaving a typed list agreeing with a catalog that moved under it.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from bili.aether.compiler.llm_resolver import (
    RESOLUTION_SOURCES,
    ModelResolution,
    describe_model_resolution,
    resolve_model,
    resolve_prompt_length_limit,
    resolve_provider,
    resolve_tool_strategy,
)
from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.config.model_families import colliding_model_ids, disambiguate

COLLISIONS = colliding_model_ids()
COLLIDING_IDS = sorted(COLLISIONS)

#: Every (colliding id, provider) pair, for the qualified-name cases.
QUALIFIED_PAIRS = [
    (model_id, provider)
    for model_id in COLLIDING_IDS
    for provider in COLLISIONS[model_id]
]

#: Model ids exactly one provider catalogs.  These must not move: a family
#: preference reorders the providers that carry a model and never adds one.
SINGLE_PROVIDER_IDS = sorted(
    {
        entry["model_id"]
        for info in LLM_MODELS.values()
        for entry in info.get("models", [])
        if entry.get("model_id") and entry["model_id"] not in COLLISIONS
    }
)


def _catalog_entry(provider_type: str, model_id: str) -> dict:
    """Return one provider's entry for *model_id*."""
    for entry in LLM_MODELS[provider_type].get("models", []):
        if entry.get("model_id") == model_id:
            return entry
    raise AssertionError(  # pragma: no cover
        f"{provider_type} has no entry for {model_id}"
    )


def _sole_provider(model_id: str) -> str:
    """Return the one provider cataloging *model_id*."""
    for provider, info in LLM_MODELS.items():
        for entry in info.get("models", []):
            if entry.get("model_id") == model_id:
                return provider
    raise AssertionError(f"{model_id} is in no catalog")  # pragma: no cover


class TestABareCollidingNameIsDecidedByTheFamilyOwner:
    """The load-bearing property: catalog order no longer decides."""

    @pytest.mark.parametrize("model_id", COLLIDING_IDS)
    def test_it_resolves_to_the_declared_owner(self, model_id: str):
        """The resolver's answer is the table's answer, for every collision."""
        expected, _reason = disambiguate(model_id, COLLISIONS[model_id])
        provider, resolved_id = resolve_model(model_id)
        assert provider == expected
        assert resolved_id == model_id

    @pytest.mark.parametrize("model_id", COLLIDING_IDS)
    def test_the_answer_is_not_merely_the_first_in_catalog_order(self, model_id: str):
        """At least one collision is decided AGAINST the catalog's order.

        Asserted over the whole set rather than per id, because a family
        whose owner happens to be written first would satisfy a per-id
        version of this while proving nothing.
        """
        decided_against_order = [
            mid for mid in COLLIDING_IDS if resolve_provider(mid) != COLLISIONS[mid][0]
        ]
        assert decided_against_order, (
            "every collision resolves to the first provider in catalog order, "
            "so this suite would pass with the disambiguation step removed"
        )

    @pytest.mark.parametrize("model_id", COLLIDING_IDS)
    def test_the_resolution_reports_every_candidate(self, model_id: str):
        """The reason a name reached a provider is observable, not inferred."""
        resolution = describe_model_resolution(model_id)
        assert resolution.source == "catalog-disambiguated"
        assert resolution.candidates == COLLISIONS[model_id]
        assert resolution.reason
        assert not resolution.is_ambiguous


class TestAQualifiedNameWins:
    """``AgentSpec`` has no provider field, so the name is the only override."""

    @pytest.mark.parametrize(
        ("model_id", "provider"),
        QUALIFIED_PAIRS,
        ids=[f"{p}:{m}" for m, p in QUALIFIED_PAIRS],
    )
    def test_every_candidate_provider_is_reachable(self, model_id: str, provider: str):
        """Each provider cataloging a colliding id can be selected by name."""
        resolution = describe_model_resolution(f"{provider}:{model_id}")
        assert resolution.provider_type == provider
        assert resolution.model_id == model_id
        assert resolution.source == "qualified"

    def test_the_qualifier_is_stripped_from_the_model_id(self):
        """The provider is sent the bare id, never the prefixed string.

        Before the qualified form existed the prefixed string missed the
        catalog, matched a vendor substring rule, and was passed on as the
        model id.
        """
        provider, model_id = resolve_model("remote_azure_openai:gpt-4o")
        assert (provider, model_id) == ("remote_azure_openai", "gpt-4o")

    def test_it_carries_the_named_providers_own_kwargs(self):
        """Azure's entry supplies api_version; OpenAI's supplies none."""
        azure = describe_model_resolution("remote_azure_openai:gpt-4o")
        openai = describe_model_resolution("remote_openai:gpt-4o")
        assert "api_version" in azure.extra_kwargs
        assert openai.extra_kwargs == {}

    def test_an_uncataloged_model_passes_through_under_the_named_provider(self):
        """An explicit provider is honoured for an id the catalog lacks.

        A locally pulled tag or a model newer than the catalog is legitimate;
        the alternative left the prefix inside the id.
        """
        resolution = describe_model_resolution("remote_openai:gpt-9-unreleased")
        assert resolution.provider_type == "remote_openai"
        assert resolution.model_id == "gpt-9-unreleased"
        assert resolution.extra_kwargs == {}

    def test_a_prefix_that_is_not_a_provider_type_is_not_a_qualifier(self):
        """An ordinary id carrying a colon is left alone.

        A local tag such as 'qwen3:8b' must not be read as provider 'qwen3'.
        """
        with pytest.raises(ValueError):
            resolve_model("qwen3:8b")

    @pytest.mark.parametrize(
        ("name", "expected_provider", "expected_id"),
        [
            ("cli:custom", "cli", "cli:custom"),
            ("ollama:deepseek-r1:14b", "local_ollama", "ollama:deepseek-r1:14b"),
            (
                "genai:gemini-2.5-flash",
                "remote_google_genai",
                "genai:gemini-2.5-flash",
            ),
        ],
    )
    def test_a_sentinel_prefix_still_wins(
        self, name: str, expected_provider: str, expected_id: str
    ):
        """The sentinels keep their contract, id included.

        'cli' is both a sentinel and a provider type, and the CLI provider
        takes its model from a separate kwarg, so reading 'cli:' as a
        qualifier would strip a prefix that path expects to keep.
        """
        assert resolve_model(name) == (expected_provider, expected_id)


class TestANameOnlyOneProviderCatalogsDoesNotMove:
    """A preference must never route a model to a catalog that lacks it."""

    @pytest.mark.parametrize("model_id", SINGLE_PROVIDER_IDS)
    def test_it_keeps_its_provider(self, model_id: str):
        """Derived over every non-colliding id in the catalog."""
        assert resolve_provider(model_id) == _sole_provider(model_id)

    def test_a_claimed_family_does_not_capture_a_single_entry(self):
        """'gemini-2.5-pro' matches the gemini rule and is Vertex-only.

        This is the case that separates 'break a tie' from 'prefer a vendor'.
        """
        resolution = describe_model_resolution("gemini-2.5-pro")
        assert resolution.provider_type == "remote_google_vertex"
        assert resolution.source == "catalog"


class TestUnaffectedResolutionPaths:
    """Everything the change was not aimed at answers as it did before."""

    def test_a_display_name_still_selects_its_own_provider(self):
        """Display names are provider-distinct and were never ambiguous."""
        assert resolve_model("Azure OpenAI GPT-4o Omni") == (
            "remote_azure_openai",
            "gpt-4o",
        )
        assert resolve_model("OpenAI GPT-4o Omni") == ("remote_openai", "gpt-4o")
        assert resolve_model("Gemini 3.1 Flash Lite (Direct API)") == (
            "remote_google_genai",
            "gemini-3.1-flash-lite",
        )

    def test_an_uncataloged_name_still_routes_by_heuristic(self):
        """The fourth step is untouched, and reports itself as the source."""
        resolution = describe_model_resolution("claude-opus-4-1-20260805")
        assert resolution.provider_type == "remote_anthropic"
        assert resolution.model_id == "claude-opus-4-1-20260805"
        assert resolution.source == "heuristic"
        assert resolution.candidates == ()

    def test_bare_gemini_still_falls_back_to_vertex(self):
        """The non-hyphenated catch-all rule is unchanged."""
        assert resolve_provider("gemini") == "remote_google_vertex"

    def test_an_unresolvable_name_still_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve model"):
            resolve_model("not-a-model-anyone-ships")


class TestAnUncoveredCollisionIsReportedNotRaised:
    """A catalog edit must not break a running deployment."""

    @staticmethod
    def _no_rules(monkeypatch):
        monkeypatch.setattr("bili.iris.config.model_families.MODEL_FAMILY_OWNERS", ())

    def test_it_falls_back_to_catalog_order(self, monkeypatch):
        """With no rule, the pre-existing answer is what comes back."""
        self._no_rules(monkeypatch)
        resolution = describe_model_resolution("gpt-4o")
        assert resolution.source == "catalog-ambiguous"
        assert resolution.is_ambiguous
        assert resolution.provider_type == COLLISIONS["gpt-4o"][0]

    def test_it_warns_and_names_the_override(self, monkeypatch, caplog):
        """Silence here is what made the original routing invisible."""
        self._no_rules(monkeypatch)
        with caplog.at_level(
            logging.WARNING, logger="bili.aether.compiler.llm_resolver"
        ):
            describe_model_resolution("gpt-4o")
        assert any(
            "remote_azure_openai" in r.getMessage()
            and "remote_openai" in r.getMessage()
            and "<provider_type>:gpt-4o" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), caplog.text


class TestTheDisambiguationIsAnnounced:
    """A decision made for the caller is visible, at the right granularity."""

    def test_resolving_alone_does_not_log_at_info(self, caplog):
        """AgentSpec validation resolves on every construction.

        An INFO line inside the resolver would repeat the same sentence
        several times per agent, so the announcement belongs to the caller
        that acts on the resolution.
        """
        with caplog.at_level(logging.INFO, logger="bili.aether.compiler.llm_resolver"):
            describe_model_resolution("gpt-4o")
            describe_model_resolution("gemini-2.5-pro")
        assert [r for r in caplog.records if r.levelno >= logging.INFO] == []

    def test_creating_the_llm_names_the_provider_and_the_alternatives(self, caplog):
        """The one place a bare name's routing is announced."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )
        from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
            AgentSpec,
        )

        agent = AgentSpec(
            agent_id="a", role="r", objective="resolve a model", model_name="gpt-4o"
        )
        with caplog.at_level(logging.INFO, logger="bili.aether.compiler.llm_resolver"):
            with patch("bili.iris.loaders.llm_loader.load_model", MagicMock()):
                create_llm(agent)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert messages, caplog.text
        assert any(
            "remote_openai" in m
            and "remote_azure_openai" in m
            and "<provider_type>:gpt-4o" in m
            for m in messages
        ), messages

    def test_a_single_candidate_name_is_not_announced_as_a_decision(self, caplog):
        """Nothing was decided, so nothing about alternatives is printed."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )
        from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
            AgentSpec,
        )

        agent = AgentSpec(
            agent_id="a",
            role="r",
            objective="resolve a model",
            model_name="gemini-2.5-pro",
        )
        with caplog.at_level(logging.INFO, logger="bili.aether.compiler.llm_resolver"):
            with patch("bili.iris.loaders.llm_loader.load_model", MagicMock()):
                create_llm(agent)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert messages, caplog.text
        assert not any("Also cataloged by" in m for m in messages), messages


class TestEveryPerModelFieldComesFromTheEntryThatWillBeLoaded:
    """One answer to which catalog entry a name means.

    The two entries for a colliding id are written independently and do
    diverge, so a field read by first-catalog-match while the loader prefers
    the family owner reports one provider's model with another's declared
    limits.
    """

    @pytest.mark.parametrize("model_id", COLLIDING_IDS)
    def test_the_prompt_limit_is_the_resolved_providers(self, model_id: str):
        """Derived: whatever the resolver picks, the limit comes from there."""
        provider, _ = resolve_model(model_id)
        declared = _catalog_entry(provider, model_id).get("max_input_tokens")
        assert resolve_prompt_length_limit(model_id) == declared

    @pytest.mark.parametrize("model_id", COLLIDING_IDS)
    def test_the_tool_strategy_is_the_resolved_providers(self, model_id: str):
        provider, _ = resolve_model(model_id)
        entry = _catalog_entry(provider, model_id)
        expected = entry.get(
            "tool_strategy",
            "native" if entry.get("supports_tools", True) else "facilitated",
        )
        assert resolve_tool_strategy(model_id) == expected

    def test_the_two_entries_for_some_collision_really_do_diverge(self):
        """The premise: without it these rows pass against identical data.

        Asserted over the catalog rather than on one id, so it survives an
        edit that happens to align the pair this test used to name.
        """
        diverging = [
            model_id
            for model_id, providers in COLLISIONS.items()
            if len(
                {
                    tuple(
                        sorted(
                            (k, repr(v))
                            for k, v in _catalog_entry(provider, model_id).items()
                        )
                    )
                    for provider in providers
                }
            )
            > 1
        ]
        assert diverging, (
            "every colliding id has identical entries under both providers, so "
            "the rows above cannot distinguish reading the resolved entry from "
            "reading the first catalog match"
        )

    def test_a_qualified_name_reads_the_named_providers_entry(self):
        """The override reaches the per-model fields too, not just the load."""
        bare = resolve_prompt_length_limit("gpt-4")
        qualified = resolve_prompt_length_limit("remote_azure_openai:gpt-4")
        assert bare == _catalog_entry("remote_openai", "gpt-4")["max_input_tokens"]
        assert (
            qualified
            == _catalog_entry("remote_azure_openai", "gpt-4")["max_input_tokens"]
        )
        assert bare != qualified

    def test_an_unresolvable_name_degrades_rather_than_raising(self):
        """Both readers answered 'unknown' for an unknown name before, and do now."""
        assert resolve_prompt_length_limit("not-a-model-anyone-ships") is None
        assert resolve_tool_strategy("not-a-model-anyone-ships") == "native"

    def test_a_heuristic_routed_name_has_no_declared_limit(self):
        """No catalog entry backs it, so there is no declared value to report."""
        assert resolve_prompt_length_limit("claude-opus-4-1-20260805") is None
        assert resolve_tool_strategy("claude-opus-4-1-20260805") == "native"

    def test_a_display_name_still_reads_its_own_entry(self):
        """Display names were never ambiguous and must not move."""
        assert (
            resolve_prompt_length_limit("Azure OpenAI GPT-4 Turbo with Vision")
            == _catalog_entry("remote_azure_openai", "gpt-4")["max_input_tokens"]
        )


class TestCreateLlmUsesTheResolvedProvider:
    """The decision reaches the loader, not just the reporting helper."""

    def test_a_bare_colliding_name_loads_under_the_family_owner(self):
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )
        from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
            AgentSpec,
        )

        agent = AgentSpec(
            agent_id="a", role="r", objective="resolve a model", model_name="gpt-4o"
        )
        with patch(
            "bili.iris.loaders.llm_loader.load_model", MagicMock()
        ) as load_model:
            create_llm(agent)
        assert load_model.call_args[0][0] == "remote_openai"
        assert load_model.call_args[1]["model_name"] == "gpt-4o"

    def test_a_qualified_name_loads_under_the_named_provider(self):
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )
        from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
            AgentSpec,
        )

        agent = AgentSpec(
            agent_id="a",
            role="r",
            objective="resolve a model",
            model_name="remote_azure_openai:gpt-4o",
        )
        with patch(
            "bili.iris.loaders.llm_loader.load_model", MagicMock()
        ) as load_model:
            create_llm(agent)
        assert load_model.call_args[0][0] == "remote_azure_openai"
        assert load_model.call_args[1]["model_name"] == "gpt-4o"
        assert "api_version" in load_model.call_args[1]


class TestTheResolutionRecord:
    """The reported source is a closed vocabulary, and every value is reachable."""

    def test_every_source_the_record_reports_is_declared(self):
        observed = {
            describe_model_resolution(name).source
            for name in (
                "remote_openai:gpt-4o",
                "gemini-2.5-pro",
                "gpt-4o",
                "claude-opus-4-1-20260805",
            )
        }
        assert observed <= set(RESOLUTION_SOURCES)
        assert observed == {
            "qualified",
            "catalog",
            "catalog-disambiguated",
            "heuristic",
        }

    def test_the_ambiguous_source_is_reachable(self, monkeypatch):
        monkeypatch.setattr("bili.iris.config.model_families.MODEL_FAMILY_OWNERS", ())
        assert describe_model_resolution("gpt-4o").source in RESOLUTION_SOURCES
        assert describe_model_resolution("gpt-4o").source == "catalog-ambiguous"

    def test_the_record_is_immutable(self):
        """A resolution is a report; a caller must not edit it in place."""
        resolution = describe_model_resolution("gpt-4o")
        assert isinstance(resolution, ModelResolution)
        with pytest.raises(Exception):
            resolution.provider_type = "remote_azure_openai"  # type: ignore[misc]
