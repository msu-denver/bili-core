"""Tests for the model-family owner table.

Every case is DERIVED from ``LLM_MODELS`` rather than typed, so a catalog
edit that adds a collision, removes one, or moves a model between providers
turns a row red instead of passing against a stale literal.
"""

import pytest

from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.config.model_families import (
    MODEL_FAMILY_OWNERS,
    FamilyRule,
    catalog_providers_for,
    colliding_model_ids,
    disambiguate,
    owning_provider,
    verify_family_owner_table,
)

COLLISIONS = colliding_model_ids()


class TestTheCatalogStillCollides:
    """The table exists for a real property of the catalog, not a hypothesis."""

    def test_some_model_id_is_cataloged_by_more_than_one_provider(self):
        """A table with nothing to disambiguate would be dead code.

        This is the premise every other row rests on: if the catalog ever
        stops carrying a duplicated id, the table and the resolution step
        that reads it should go, and this says so.
        """
        assert COLLISIONS, (
            "no model_id is cataloged by more than one provider; "
            "MODEL_FAMILY_OWNERS and the disambiguation step have no subject"
        )

    def test_a_collision_names_distinct_providers(self):
        """A provider listing one id twice is not a collision."""
        for model_id, providers in COLLISIONS.items():
            assert len(set(providers)) == len(
                providers
            ), f"{model_id}: provider list {providers} repeats a provider"


class TestEveryCollisionIsDecided:
    """Completeness, derived: no collision may fall through to catalog order."""

    @pytest.mark.parametrize("model_id", sorted(COLLISIONS))
    def test_a_rule_covers_it(self, model_id: str):
        """Each colliding id has a rule, so nothing resolves by insertion order.

        A new colliding entry fails here rather than silently inheriting
        whichever provider happens to be written first in the catalog.
        """
        assert disambiguate(model_id, COLLISIONS[model_id]) is not None, (
            f"'{model_id}' is cataloged by {COLLISIONS[model_id]} and no "
            f"MODEL_FAMILY_OWNERS rule claims it; it would resolve by "
            f"catalog order"
        )

    @pytest.mark.parametrize("model_id", sorted(COLLISIONS))
    def test_the_chosen_provider_catalogs_the_model(self, model_id: str):
        """A preference may reorder the candidates and never add one.

        Routing to a provider whose catalog has no entry for the model would
        replace an arbitrary answer with a broken one.
        """
        decided = disambiguate(model_id, COLLISIONS[model_id])
        assert decided is not None
        provider, _reason = decided
        assert provider in COLLISIONS[model_id]
        assert provider in catalog_providers_for(model_id)


class TestNoRuleIsDead:
    """A rule covering nothing reads as coverage and provides none."""

    @pytest.mark.parametrize("rule", MODEL_FAMILY_OWNERS, ids=lambda r: r.prefix)
    def test_the_rule_decides_at_least_one_live_collision(self, rule: FamilyRule):
        """Each row wins a real collision in the shipped catalog."""
        decided = [
            model_id
            for model_id, providers in COLLISIONS.items()
            if disambiguate(model_id, providers) == (rule.provider_type, rule.reason)
        ]
        assert decided, (
            f"rule {rule.prefix!r} -> {rule.provider_type!r} decides no collision "
            f"in the current catalog; delete it or fix the prefix"
        )


class TestLongestPrefixWins:
    """Order in the tuple is presentation; specificity decides."""

    def test_a_narrower_rule_beats_a_broader_one(self):
        """'gpt-35-turbo' is claimed by its own row, not by the 'gpt-' row.

        Written as a first-match scan this would depend on which row happens
        to come first in the literal.
        """
        rule = owning_provider("gpt-35-turbo")
        assert rule is not None
        assert rule.prefix == "gpt-35-turbo"
        assert owning_provider("gpt-4o").prefix == "gpt-"

    def test_matching_is_case_insensitive(self):
        """A caller's capitalisation does not change which family owns an id."""
        assert owning_provider("GPT-4o") == owning_provider("gpt-4o")

    def test_an_unclaimed_id_has_no_rule(self):
        """A family nobody declared returns None rather than a nearest guess."""
        assert owning_provider("some-unlisted-vendor-model") is None


class TestDisambiguateIsBoundedByItsCandidates:
    """The function can only ever pick from what it was handed."""

    def test_an_owner_outside_the_candidates_declines(self):
        """A single-provider name keeps its provider even when a rule claims it.

        'gemini-2.5-pro' matches the 'gemini-' rule and is cataloged only by
        Vertex; preferring the Developer API would name a model that catalog
        has no entry for.
        """
        assert catalog_providers_for("gemini-2.5-pro") == ("remote_google_vertex",)
        assert owning_provider("gemini-2.5-pro") is not None
        assert disambiguate("gemini-2.5-pro", ("remote_google_vertex",)) is None

    def test_no_candidates_declines(self):
        """Nothing to choose between is not a choice."""
        assert disambiguate("gpt-4o", ()) is None


class TestCatalogProvidersFor:
    """The candidate reader matches the pair the resolver looks names up by."""

    def test_a_display_name_names_one_provider(self):
        """Display names are provider-distinct, so they never collide."""
        assert catalog_providers_for("Azure OpenAI GPT-4o Omni") == (
            "remote_azure_openai",
        )
        assert catalog_providers_for("OpenAI GPT-4o Omni") == ("remote_openai",)

    def test_a_colliding_id_names_every_provider_in_catalog_order(self):
        """Order is the catalog's, which is what 'first' meant before the table."""
        providers = catalog_providers_for("gpt-4o")
        assert providers == tuple(
            p for p in LLM_MODELS if p in ("remote_azure_openai", "remote_openai")
        )

    def test_an_unknown_name_names_nobody(self):
        assert catalog_providers_for("no-such-model") == ()


class TestTheTableIsVerifiedAtImport:
    """The guard refuses a table that would mis-route or silently not apply.

    Each case mutates the shipped table and calls the verifier the module
    runs on import, so a fault that reaches the module raises there.
    """

    @staticmethod
    def _verify(rules, monkeypatch):
        monkeypatch.setattr(
            "bili.iris.config.model_families.MODEL_FAMILY_OWNERS", tuple(rules)
        )
        verify_family_owner_table()

    def test_the_shipped_table_passes(self):
        """The guard is satisfied by what ships, so a failure means an edit."""
        verify_family_owner_table()

    def test_an_empty_prefix_is_refused(self, monkeypatch):
        """An empty prefix claims every model name."""
        with pytest.raises(ValueError, match="empty prefix"):
            self._verify([FamilyRule("", "remote_openai", "why")], monkeypatch)

    def test_an_upper_cased_prefix_is_refused(self, monkeypatch):
        """Matching lower-cases the id, so an upper-cased prefix never fires."""
        with pytest.raises(ValueError, match="not lower-cased"):
            self._verify([FamilyRule("GPT-", "remote_openai", "why")], monkeypatch)

    def test_a_duplicate_prefix_is_refused(self, monkeypatch):
        """Two rows on one prefix leave one of them unreachable."""
        with pytest.raises(ValueError, match="declared twice"):
            self._verify(
                [
                    FamilyRule("gpt-", "remote_openai", "why"),
                    FamilyRule("gpt-", "remote_azure_openai", "why"),
                ],
                monkeypatch,
            )

    def test_an_unknown_provider_is_refused(self, monkeypatch):
        """A typo'd provider would match and then never apply."""
        with pytest.raises(ValueError, match="not a key of LLM_MODELS"):
            self._verify([FamilyRule("gpt-", "remote_opneai", "why")], monkeypatch)

    def test_a_missing_reason_is_refused(self, monkeypatch):
        """The reason is what a caller logs when a name was decided for them."""
        with pytest.raises(ValueError, match="no reason"):
            self._verify([FamilyRule("gpt-", "remote_openai", "")], monkeypatch)

    def test_the_module_runs_the_guard_on_import(self):
        """Executing the module's own source with a bad table must raise.

        Calling the verifier directly proves the function; it says nothing
        about whether anything runs it.  Deleting the module-level call makes
        this pass silently, so the source is executed instead.
        """
        import pathlib  # pylint: disable=import-outside-toplevel

        import bili.iris.config.model_families as mod  # pylint: disable=import-outside-toplevel

        source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        namespace = {"__name__": "model_families_import_probe"}
        # Sanity: the real source imports and runs cleanly.
        exec(
            compile(source, mod.__file__, "exec"), namespace
        )  # pylint: disable=exec-used

        broken = source.replace(
            'FamilyRule(\n        "gpt-",',
            'FamilyRule(\n        "",',
            1,
        )
        assert broken != source, "the mutation anchor moved; re-point it"
        with pytest.raises(ValueError, match="empty prefix"):
            exec(  # pylint: disable=exec-used
                compile(broken, mod.__file__, "exec"),
                {"__name__": "model_families_import_probe"},
            )
