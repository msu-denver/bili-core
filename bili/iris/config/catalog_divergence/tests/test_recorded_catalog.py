"""What the check says about the shipped catalog and the recorded upstreams.

These are the measurements, as opposed to the classifier's unit behaviour.
They run against the recorded slices, so they measure THIS repository against
THOSE bytes and nothing else. The scheduled job is what measures the catalog
against whatever the upstreams say today; a test cannot do that job without
becoming a test that fails for reasons outside the repository.

The floors are the id-mapper's regression net. An upstream renaming a model or
changing its id scheme moves the live numbers and is the scheduled job's to
report; a change in this repository that stops resolving ids moves these, and
that is a defect in the change.
"""

# pylint: disable=redefined-outer-name  # the pytest fixture convention

import pytest

from bili.iris.config.catalog_divergence.compare import (
    ERROR,
    FIELD_COVERAGE,
    compare_catalog,
)
from bili.iris.config.catalog_divergence.datasets import MODELS_DEV
from bili.iris.config.catalog_divergence.mapping import (
    LITELLM_PROVIDERS,
    MODELS_DEV_PROVIDERS,
    UNLISTED_PROVIDER_TYPES,
)

#: Provider-scoped match counts measured against the 2026-09-03 recorded
#: slices. The count is the floor rather than the rate, because a rate hides
#: a family gaining entries: adding four unmatchable models to a family of
#: four would halve the rate while breaking nothing, and dropping four
#: matched ones would hold the rate while breaking the mapper.
MATCH_FLOORS = {
    "remote_anthropic": 7,
    "remote_aws_bedrock": 48,
    "remote_azure_openai": 11,
    "remote_cohere": 3,
    "remote_deepseek": 2,
    "remote_google_genai": 7,
    "remote_google_vertex": 5,
    "remote_groq": 2,
    "remote_mistral": 3,
    "remote_openai": 9,
    "remote_xai": 1,
    "local_ollama": 0,
}

#: Total entries resolved to at least one authoritative record, same capture.
TOTAL_MATCH_FLOOR = 98


@pytest.fixture(scope="module")
def recorded(models_dev_dataset, litellm_dataset):
    """The comparison of the shipped catalog against the recorded slices.

    :param models_dev_dataset: The parsed models.dev slice.
    :param litellm_dataset: The parsed LiteLLM slice.
    :returns: The report.
    """
    return compare_catalog(models_dev_dataset, litellm_dataset)


class TestMatchRateFloors:
    """An id-scheme regression shows up here before it reaches a user."""

    def test_the_comparison_is_complete(self, recorded):
        """Both slices must parse, or every floor below measures nothing."""
        assert recorded.any_unavailable is False

    @pytest.mark.parametrize("provider_type", sorted(MATCH_FLOORS))
    def test_a_family_still_resolves_its_recorded_count(self, recorded, provider_type):
        """Each mapped family resolves at least what it resolved at capture."""
        match = recorded.matches[provider_type]
        floor = MATCH_FLOORS[provider_type]
        assert match.matched_either >= floor, (
            f"{provider_type} resolved {match.matched_either} of {match.entries}, "
            f"below the recorded floor of {floor}: the id mapping regressed"
        )

    def test_the_whole_catalog_still_resolves_its_recorded_count(self, recorded):
        """The aggregate catches a regression spread thinly across families."""
        total = sum(m.matched_either for m in recorded.matches.values())
        assert total >= TOTAL_MATCH_FLOOR

    def test_every_floor_names_a_mapped_family(self):
        """A floor for an unlisted family would assert against nothing."""
        assert not set(MATCH_FLOORS) & UNLISTED_PROVIDER_TYPES

    def test_every_mapped_family_has_a_floor(self, recorded):
        """A family with no floor is a family whose mapper is unguarded."""
        mapped = set(recorded.matches) - UNLISTED_PROVIDER_TYPES
        assert mapped == set(MATCH_FLOORS)

    def test_a_family_with_no_upstream_listing_resolves_nothing(self, recorded):
        """The excluded families are excluded because they cannot match.

        If one of them started matching, the exclusion would be silencing
        real coverage findings rather than avoiding noise.
        """
        for provider_type in UNLISTED_PROVIDER_TYPES:
            assert recorded.matches[provider_type].matched_either == 0


class TestWhatTheCatalogSaysToday:
    """The findings the recorded upstreams produce against the shipped table."""

    def test_no_entry_over_claims_a_capability(self, recorded):
        """No declared modality is denied by a dataset that states its own.

        This is the class the check was written for, and the catalog is clean
        of it against this capture. A future entry declaring a capability
        from a display string rather than from the model's behaviour turns
        this red.
        """
        errors = [f for f in recorded.findings if f.severity == ERROR]
        assert errors == [], f"over-claimed capabilities: {errors}"

    def test_the_unverifiable_entries_are_the_expected_holdouts(self, recorded):
        """Every uncovered entry is a curated case, not an id-mapper gap.

        A deployment name, a legacy model the upstreams have dropped, or a
        local tag: none of these is something a better mapper would resolve.
        """
        uncovered = {
            f.model_id for f in recorded.findings if f.field_name == FIELD_COVERAGE
        }
        assert uncovered == {
            "qwen3",
            "gpt-41",
            "gpt-41-mini",
            "gpt-41-nano",
            "gemini-1.0-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
            "gemini-1.5-pro-002",
            "compound-beta",
            "compound-beta-mini",
            "gpt-35-turbo",
            "o1-mini",
            "grok-beta",
        }

    def test_every_finding_cites_an_authoritative_provider(self, recorded):
        """A finding sourced from a gateway would be the contamination bug.

        Every citation has to sit under a provider this repository named as
        authoritative for that catalog family; a record found anywhere else
        is not found at all.
        """
        for f in recorded.findings:
            for value in f.dataset_values:
                table = (
                    MODELS_DEV_PROVIDERS
                    if value.source == MODELS_DEV
                    else LITELLM_PROVIDERS
                )
                allowed = table.get(f.provider_type, ())
                assert value.provider_id in allowed, (
                    f"{f.provider_type}/{f.model_id} cites "
                    f"{value.source}[{value.provider_id}], which is not "
                    f"authoritative for that family"
                )

    def test_a_finding_never_cites_a_region_qualified_key(self, recorded):
        """A citation a reader cannot look up is a citation they distrust."""
        for f in recorded.findings:
            for value in f.dataset_values:
                assert value.key.count("/") <= 1, (
                    f"{f.model_id} cites {value.key}, a qualified key where a "
                    f"canonical one exists"
                )
