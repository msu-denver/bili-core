"""Tests for the comparison and its severity rules.

The catalogs and datasets here are synthetic on purpose: these tests assert
what the CLASSIFIER does with a given disagreement, so every input has to be
controlled. What the classifier says about the real catalog and the real
upstreams is measured in ``test_recorded_catalog.py`` against the recorded
payloads instead.
"""

import pytest

from bili.iris.config.catalog_divergence.compare import (
    ERROR,
    FIELD_CONTEXT_WINDOW,
    FIELD_COVERAGE,
    FIELD_INPUT_MODALITIES,
    FIELD_OUTPUT_MODALITIES,
    FIELD_TEMPERATURE,
    INFO,
    WARNING,
    DatasetValue,
    Finding,
    ProviderMatch,
    compare_catalog,
    merge_findings,
)
from bili.iris.config.catalog_divergence.datasets import (
    LITELLM,
    MODELS_DEV,
    REASON_NETWORK,
    Unavailable,
    parse_litellm,
    parse_models_dev,
)

NO_MODELS_DEV = Unavailable(MODELS_DEV, REASON_NETWORK, "not fetched for this case")
NO_LITELLM = Unavailable(LITELLM, REASON_NETWORK, "not fetched for this case")


def catalog(provider_type="remote_openai", **entry):
    """Build a one-entry catalog for *provider_type*.

    :param provider_type: The catalog provider type key.
    :param entry: The catalog entry's fields.
    :returns: A catalog mapping with a single entry.
    :rtype: dict
    """
    entry.setdefault("model_id", "m")
    entry.setdefault("model_name", "Test Model")
    return {provider_type: {"name": "T", "description": "d", "models": [entry]}}


def models_dev(provider_id="openai", key="m", **record):
    """Build a one-record models.dev dataset.

    :param provider_id: The upstream provider id.
    :param key: The upstream model key.
    :param record: The upstream record's fields.
    :returns: A parsed dataset.
    """
    return parse_models_dev({provider_id: {"models": {key: record}}})


def litellm(provider_id="openai", key="m", **record):
    """Build a one-record LiteLLM dataset.

    :param provider_id: The upstream provider id.
    :param key: The upstream model key.
    :param record: The upstream record's fields.
    :returns: A parsed dataset.
    """
    record["litellm_provider"] = provider_id
    return parse_litellm({key: record})


def only(findings, field_name):
    """Return the findings for one field.

    :param findings: The findings to filter.
    :param field_name: The field to keep.
    :returns: The matching findings.
    :rtype: list
    """
    return [f for f in findings if f.field_name == field_name]


class TestModalityOverClaim:
    """The false-assurance direction, which is what this exists to catch."""

    def test_a_claim_an_explicit_array_denies_is_an_error(self):
        """A declared modality the upstream does not list is an ERROR.

        This is the shape of a real defect: an entry whose image input had
        been declared from a display string rather than from the model's
        capability, which passes the selection gate and then fails opaquely.
        """
        report = compare_catalog(
            models_dev(modalities={"input": ["text"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        findings = only(report.findings, FIELD_INPUT_MODALITIES)
        assert len(findings) == 1
        assert findings[0].severity == ERROR
        assert "image" in findings[0].message
        assert report.has_errors is True

    def test_a_claim_an_unrecorded_dataset_cannot_deny_is_silent(self):
        """An omission is never a denial.

        The flat dataset omits fields it has no value for, so reading its
        silence as "no" would manufacture the exact false finding this check
        exists to avoid.
        """
        report = compare_catalog(
            NO_MODELS_DEV,
            litellm(max_input_tokens=1000),
            catalog(input_modalities=["text", "image"], max_input_tokens=1000),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []
        assert report.has_errors is False

    def test_a_positive_flag_that_agrees_produces_nothing(self):
        """A dataset confirming the claim is not a disagreement."""
        report = compare_catalog(
            NO_MODELS_DEV,
            litellm(supports_vision=True),
            catalog(input_modalities=["text", "image"]),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []

    def test_an_over_claim_against_a_positive_flag_is_silent(self):
        """A non-enumerative dataset licenses no verdict on an over-claim.

        The flat dataset's set is synthesised from a vision flag, so it
        confirms what it does record and enumerates nothing. It cannot say
        the model rejects audio, so there is no finding to make: reporting
        one either way would be inventing an answer from silence.
        """
        report = compare_catalog(
            NO_MODELS_DEV,
            litellm(supports_vision=True),
            catalog(input_modalities=["text", "image", "audio"]),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []

    def test_the_knowable_half_of_a_mixed_disagreement_is_still_reported(self):
        """Silence on the unknowable half must not silence the other half.

        When the catalog both over-claims something the dataset cannot deny
        and omits something the dataset positively confirms, the confirmed
        omission is a real signal and is reported.
        """
        report = compare_catalog(
            NO_MODELS_DEV,
            litellm(supports_vision=True),
            catalog(input_modalities=["text", "audio"]),
        )
        findings = only(report.findings, FIELD_INPUT_MODALITIES)
        assert [f.severity for f in findings] == [WARNING]
        assert "image" in findings[0].message


class TestModalityUnderClaim:
    """Declaring less than the upstream lists is defensible, not a defect."""

    def test_a_narrower_declaration_is_a_warning(self):
        """The framework may decline to claim an input kind it cannot build."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text", "image", "audio"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        findings = only(report.findings, FIELD_INPUT_MODALITIES)
        assert len(findings) == 1
        assert findings[0].severity == WARNING
        assert "audio" in findings[0].message
        assert report.has_errors is False

    def test_a_kind_outside_the_vocabulary_is_not_a_divergence(self):
        """A kind the catalog cannot express is not something it omits."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text", "image", "pdf", "video"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []

    def test_the_report_shows_what_the_upstream_actually_said(self):
        """The citation is verbatim, so the reader sees the untrimmed set."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text", "image", "audio", "pdf"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        finding = only(report.findings, FIELD_INPUT_MODALITIES)[0]
        assert "pdf" in finding.dataset_values[0].value

    def test_an_agreeing_declaration_produces_nothing(self):
        """Agreement is silence."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text", "image"]}),
            NO_LITELLM,
            catalog(input_modalities=["image", "text"]),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []

    def test_an_undeclared_catalog_modality_is_not_compared(self):
        """An entry that declares nothing is making no claim to check."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text", "image"]}),
            NO_LITELLM,
            catalog(),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []


class TestOutputModalities:
    """The output axis is compared wherever the catalog declares one."""

    def test_a_declared_output_over_claim_is_an_error(self):
        """The same asymmetry applies on the output side."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text"], "output": ["text"]}),
            NO_LITELLM,
            catalog(output_modalities=["text", "image"]),
        )
        findings = only(report.findings, FIELD_OUTPUT_MODALITIES)
        assert len(findings) == 1
        assert findings[0].severity == ERROR

    def test_a_declared_output_under_claim_is_a_warning(self):
        """Declaring less on the output side is a prompt to review."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text"], "output": ["text", "image"]}),
            NO_LITELLM,
            catalog(output_modalities=["text"]),
        )
        findings = only(report.findings, FIELD_OUTPUT_MODALITIES)
        assert [f.severity for f in findings] == [WARNING]


class TestTemperature:
    """A disagreement here is a semantic mismatch candidate, not a verdict."""

    @pytest.mark.parametrize("declared,dataset", [(True, False), (False, True)])
    def test_either_direction_is_a_warning(self, declared, dataset):
        """Neither side is treated as simply wrong.

        A live probe settled this axis: the parameter is accepted normally
        and constrained only alongside extended thinking, so a dataset
        boolean can encode a mode-conditional restriction flattened to an
        unconditional false.
        """
        report = compare_catalog(
            models_dev(temperature=dataset),
            NO_LITELLM,
            catalog(supports_temperature=declared),
        )
        findings = only(report.findings, FIELD_TEMPERATURE)
        assert [f.severity for f in findings] == [WARNING]
        assert "semantic mismatch candidate" in findings[0].message

    def test_agreement_produces_nothing(self):
        """Agreement is silence."""
        report = compare_catalog(
            models_dev(temperature=True),
            NO_LITELLM,
            catalog(supports_temperature=True),
        )
        assert only(report.findings, FIELD_TEMPERATURE) == []

    def test_a_dataset_with_no_answer_produces_nothing(self):
        """A dataset carrying no temperature field asserts nothing."""
        report = compare_catalog(
            models_dev(),
            litellm(max_input_tokens=10),
            catalog(supports_temperature=True, max_input_tokens=10),
        )
        assert only(report.findings, FIELD_TEMPERATURE) == []

    def test_an_entry_declaring_nothing_is_not_compared(self):
        """An entry with no declaration is making no claim to check."""
        report = compare_catalog(models_dev(temperature=False), NO_LITELLM, catalog())
        assert only(report.findings, FIELD_TEMPERATURE) == []


class TestContextWindow:
    """A window is a quantity, so it never reaches ERROR."""

    def test_over_declaring_is_a_warning(self):
        """A declared window is read as an approximate prompt budget."""
        report = compare_catalog(
            models_dev(limit={"context": 8192}),
            NO_LITELLM,
            catalog(max_input_tokens=128000),
        )
        findings = only(report.findings, FIELD_CONTEXT_WINDOW)
        assert [f.severity for f in findings] == [WARNING]
        assert "larger" in findings[0].message

    def test_under_declaring_is_info(self):
        """Declaring less capacity than exists is conservative."""
        report = compare_catalog(
            models_dev(limit={"context": 1000000}),
            NO_LITELLM,
            catalog(max_input_tokens=200000),
        )
        findings = only(report.findings, FIELD_CONTEXT_WINDOW)
        assert [f.severity for f in findings] == [INFO]
        assert "smaller" in findings[0].message

    def test_a_window_disagreement_never_reaches_error(self):
        """Not a capability, gates no call, and the datasets disagree on it."""
        report = compare_catalog(
            models_dev(limit={"context": 1}),
            NO_LITELLM,
            catalog(max_input_tokens=999999),
        )
        assert report.has_errors is False

    def test_agreement_produces_nothing(self):
        """Agreement is silence."""
        report = compare_catalog(
            models_dev(limit={"context": 8192}),
            NO_LITELLM,
            catalog(max_input_tokens=8192),
        )
        assert only(report.findings, FIELD_CONTEXT_WINDOW) == []

    @pytest.mark.parametrize("declared", [None, 0, -1, "8192", True])
    def test_an_unusable_declaration_is_not_compared(self, declared):
        """Only a positive integer is a window worth comparing."""
        entry = {} if declared is None else {"max_input_tokens": declared}
        report = compare_catalog(
            models_dev(limit={"context": 4096}), NO_LITELLM, catalog(**entry)
        )
        assert only(report.findings, FIELD_CONTEXT_WINDOW) == []


class TestCoverage:
    """An entry no dataset carries is reported, unless it never could be."""

    def test_an_unmatched_entry_is_reported_as_info(self):
        """The entry is unverifiable from these sources, which is worth saying."""
        report = compare_catalog(
            models_dev(key="something-else"), NO_LITELLM, catalog()
        )
        findings = only(report.findings, FIELD_COVERAGE)
        assert [f.severity for f in findings] == [INFO]
        assert findings[0].dataset_values == ()

    def test_a_provider_with_no_upstream_listing_is_silent(self):
        """A subprocess tool or a local weights path has no upstream to miss.

        Reporting these would put a permanent unfixable entry in every run.
        """
        report = compare_catalog(
            models_dev(key="something-else"),
            NO_LITELLM,
            catalog(provider_type="cli_claude_code"),
        )
        assert only(report.findings, FIELD_COVERAGE) == []

    def test_a_match_under_a_non_authoritative_provider_is_not_a_match(self):
        """A gateway that re-lists a model is not an authority on it.

        The record exists and carries a flatly contradictory capability; it
        must not be found at all, so the entry reads as uncovered rather than
        as disagreeing.
        """
        report = compare_catalog(
            models_dev(provider_id="some-reseller", modalities={"input": ["text"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        assert only(report.findings, FIELD_INPUT_MODALITIES) == []
        assert [f.severity for f in only(report.findings, FIELD_COVERAGE)] == [INFO]

    def test_coverage_counts_each_dataset_separately(self):
        """The per-dataset columns are what an id-mapper regression moves."""
        report = compare_catalog(models_dev(), litellm(), catalog())
        match = report.matches["remote_openai"]
        assert (match.entries, match.matched_models_dev, match.matched_litellm) == (
            1,
            1,
            1,
        )
        assert match.matched_either == 1
        assert match.match_rate == 1.0

    def test_an_entry_with_no_id_is_skipped(self):
        """An entry naming no model cannot be resolved or reported against."""
        broken = {"remote_openai": {"models": [{"model_name": "nameless"}]}}
        report = compare_catalog(models_dev(), NO_LITELLM, broken)
        assert report.findings == ()
        assert report.catalog_entries == 1

    def test_an_empty_family_has_a_zero_match_rate(self):
        """A family with no entries divides by nothing."""
        assert ProviderMatch("remote_openai", 0, 0, 0, 0).match_rate == 0.0


class TestAdvisoryCap:
    """A deployment name has no vendor model behind it."""

    def test_an_over_claim_on_an_advisory_family_cannot_be_an_error(self):
        """The upstream record with the same string is about something else.

        A deployment name is chosen by the operator and may point at any
        underlying model, so an upstream record that happens to share the
        string cannot establish that the catalog is wrong. The finding is
        still reported; it just cannot fail the run.
        """
        report = compare_catalog(
            models_dev(provider_id="azure", modalities={"input": ["text"]}),
            NO_LITELLM,
            catalog(
                provider_type="remote_azure_openai",
                input_modalities=["text", "image"],
            ),
        )
        findings = only(report.findings, FIELD_INPUT_MODALITIES)
        assert [f.severity for f in findings] == [WARNING]
        assert report.has_errors is False

    def test_the_same_disagreement_on_a_vendor_family_is_an_error(self):
        """The cap is what differs, not the disagreement.

        Without this pair the cap could be a rule that never fires, or one
        that swallows every ERROR everywhere, and both would look identical.
        """
        report = compare_catalog(
            models_dev(modalities={"input": ["text"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        assert [f.severity for f in only(report.findings, FIELD_INPUT_MODALITIES)] == [
            ERROR
        ]


class TestMergingFindings:
    """Corroboration reads as one stronger finding; disagreement stays two."""

    def test_two_datasets_agreeing_merge_into_one_finding(self):
        """The same fact stated twice is noise, and it hides the agreement."""
        report = compare_catalog(
            models_dev(limit={"context": 1000000}),
            litellm(max_input_tokens=1000000),
            catalog(max_input_tokens=200000),
        )
        findings = only(report.findings, FIELD_CONTEXT_WINDOW)
        assert len(findings) == 1
        assert {v.source for v in findings[0].dataset_values} == {MODELS_DEV, LITELLM}

    def test_two_datasets_disagreeing_stay_two_findings(self):
        """A real dataset-vs-dataset split must stay visible as one."""
        report = compare_catalog(
            models_dev(limit={"context": 100}),
            litellm(max_input_tokens=1000000),
            catalog(max_input_tokens=200000),
        )
        findings = only(report.findings, FIELD_CONTEXT_WINDOW)
        assert {f.severity for f in findings} == {WARNING, INFO}

    def test_merging_preserves_first_seen_order(self):
        """A merge must not reshuffle the report."""
        first = Finding(
            WARNING, "p", "a", "A", "f", 1, (DatasetValue("s", "p", "a", 2),), "m"
        )
        second = Finding(
            WARNING, "p", "b", "B", "f", 1, (DatasetValue("s", "p", "b", 2),), "m"
        )
        merged = merge_findings([first, second])
        assert [f.model_id for f in merged] == ["a", "b"]

    def test_findings_differing_only_in_citation_merge(self):
        """Identical facts from two sources become one finding, two citations."""
        base = {
            "severity": WARNING,
            "provider_type": "p",
            "model_id": "a",
            "model_name": "A",
            "field_name": "f",
            "catalog_value": 1,
            "message": "m",
        }
        merged = merge_findings(
            [
                Finding(dataset_values=(DatasetValue("x", "p", "a", 2),), **base),
                Finding(dataset_values=(DatasetValue("y", "p", "a", 2),), **base),
            ]
        )
        assert len(merged) == 1
        assert len(merged[0].dataset_values) == 2

    def test_findings_differing_in_message_do_not_merge(self):
        """Two different statements about one field are two findings."""
        base = {
            "severity": WARNING,
            "provider_type": "p",
            "model_id": "a",
            "model_name": "A",
            "field_name": "f",
            "catalog_value": 1,
            "dataset_values": (),
        }
        merged = merge_findings(
            [Finding(message="one", **base), Finding(message="two", **base)]
        )
        assert len(merged) == 2


class TestUnavailableDatasets:
    """A failed fetch must never read as a clean catalog."""

    def test_one_unavailable_dataset_still_compares_the_other(self):
        """A partial comparison reported as partial is useful."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text"]}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"]),
        )
        assert report.has_errors is True
        assert report.any_unavailable is True
        assert [u.source for u in report.unavailable] == [LITELLM]

    def test_both_unavailable_produces_no_findings_but_says_so(self):
        """This is the case that must not be mistaken for a clean run.

        With nothing to compare against there are no findings, which on its
        own is indistinguishable from a catalog that agrees with everything.
        The incompleteness is what carries the difference.
        """
        report = compare_catalog(
            NO_MODELS_DEV, NO_LITELLM, catalog(input_modalities=["text", "image"])
        )
        assert report.count(ERROR) == 0
        assert report.count(WARNING) == 0
        assert report.any_unavailable is True
        assert {u.source for u in report.unavailable} == {MODELS_DEV, LITELLM}

    def test_an_unavailable_dataset_contributes_no_origin(self):
        """Only a dataset that was read has provenance to report."""
        report = compare_catalog(models_dev(), NO_LITELLM, catalog())
        assert set(report.dataset_origins) == {MODELS_DEV}


class TestReportShape:
    """The report object's own accessors."""

    def test_counts_and_totals_are_reported(self):
        """The header numbers come from the findings, not from a tally."""
        report = compare_catalog(
            models_dev(modalities={"input": ["text"]}, limit={"context": 999999}),
            NO_LITELLM,
            catalog(input_modalities=["text", "image"], max_input_tokens=200000),
        )
        assert report.count(ERROR) == 1
        assert report.count(INFO) == 1
        assert report.catalog_entries == 1
        assert report.generated_at.endswith("+00:00")

    def test_a_clean_catalog_produces_no_findings(self):
        """Total agreement is total silence."""
        report = compare_catalog(
            models_dev(
                modalities={"input": ["text"]}, temperature=True, limit={"context": 10}
            ),
            litellm(max_input_tokens=10),
            catalog(
                input_modalities=["text"],
                supports_temperature=True,
                max_input_tokens=10,
            ),
        )
        assert report.findings == ()
        assert report.has_errors is False
        assert report.any_unavailable is False

    def test_the_shipped_catalog_is_the_default(self):
        """Called with no catalog, it reads the one that ships."""
        report = compare_catalog(NO_MODELS_DEV, NO_LITELLM)
        assert report.catalog_entries > 100
        assert "remote_openai" in report.matches
