"""Tests for the text, JSON, and issue-body renderings.

Both forms are rendered from one report object, so the assertions that matter
are the ones that check they cannot disagree about what was found, and that an
incomplete run says so in every form.
"""

import json

from bili.iris.config.catalog_divergence.compare import (
    ERROR,
    INFO,
    WARNING,
    DatasetValue,
    DivergenceReport,
    Finding,
    ProviderMatch,
)
from bili.iris.config.catalog_divergence.datasets import (
    LITELLM,
    MODELS_DEV,
    REASON_NETWORK,
    Unavailable,
)
from bili.iris.config.catalog_divergence.report import (
    STICKY_MARKER,
    render_issue_body,
    render_json,
    render_text,
    sorted_findings,
    to_dict,
)


def finding(severity=ERROR, model_id="m", field_name="input_modalities", cites=1):
    """Build a finding for rendering.

    :param severity: The severity to carry.
    :param model_id: The model id to carry.
    :param field_name: The field name to carry.
    :param cites: How many dataset citations to attach.
    :returns: A finding.
    :rtype: Finding
    """
    return Finding(
        severity=severity,
        provider_type="remote_openai",
        model_id=model_id,
        model_name="Test Model",
        field_name=field_name,
        catalog_value=["text", "image"],
        dataset_values=tuple(
            DatasetValue(MODELS_DEV, "openai", f"k{i}", ["text"]) for i in range(cites)
        ),
        message="catalog claims ['image'], which the dataset does not list",
    )


def report(findings=(), unavailable=()):
    """Build a report for rendering.

    :param findings: The findings to carry.
    :param unavailable: The unreadable datasets to carry.
    :returns: A report.
    :rtype: DivergenceReport
    """
    return DivergenceReport(
        findings=tuple(findings),
        matches={"remote_openai": ProviderMatch("remote_openai", 4, 3, 2, 3)},
        unavailable=tuple(unavailable),
        catalog_entries=4,
        generated_at="2026-09-03T00:00:00+00:00",
        dataset_origins={MODELS_DEV: "https://models.dev/api.json"},
    )


class TestOrdering:
    """The report reads most severe first."""

    def test_findings_are_ordered_by_severity(self):
        """A reader must meet the errors before the notes."""
        ordered = sorted_findings(
            report(
                [
                    finding(INFO, "c"),
                    finding(ERROR, "a"),
                    finding(WARNING, "b"),
                ]
            )
        )
        assert [f.severity for f in ordered] == [ERROR, WARNING, INFO]

    def test_ordering_within_a_severity_is_stable(self):
        """Two runs over one catalog must produce the same report."""
        ordered = sorted_findings(
            report([finding(ERROR, "z"), finding(ERROR, "a"), finding(ERROR, "m")])
        )
        assert [f.model_id for f in ordered] == ["a", "m", "z"]


class TestRenderText:
    """The form a maintainer reads in a job log."""

    def test_a_clean_report_says_so(self):
        """Nothing found is stated, not left as an empty section."""
        assert "No divergences found." in render_text(report())

    def test_the_counts_and_provenance_are_stated(self):
        """The header carries what was compared and where it came from."""
        text = render_text(report([finding(ERROR)]))
        assert "1 error" in text
        assert "https://models.dev/api.json" in text
        assert "2026-09-03T00:00:00+00:00" in text

    def test_coverage_is_tabulated_per_provider(self):
        """The per-family columns are what an id-mapper regression moves."""
        text = render_text(report())
        assert "remote_openai" in text
        assert "75%" in text

    def test_a_finding_names_its_field_and_both_values(self):
        """A finding a reader cannot act on is not a finding."""
        text = render_text(report([finding()]))
        assert "input_modalities" in text
        assert "models.dev[openai/k0]" in text
        assert "catalog=['text', 'image']" in text

    def test_every_citation_is_shown(self):
        """Corroboration is the point of merging; hiding one would undo it."""
        text = render_text(report([finding(cites=2)]))
        assert "openai/k0" in text and "openai/k1" in text

    def test_an_incomplete_run_is_announced_before_the_counts(self):
        """A clean summary above an unreported failure tells the reader the
        opposite of the truth, so the warning goes first."""
        text = render_text(
            report(unavailable=[Unavailable(LITELLM, REASON_NETWORK, "offline")])
        )
        assert "INCOMPLETE" in text
        assert text.index("INCOMPLETE") < text.index("catalog entries")
        assert "offline" in text

    def test_a_complete_run_makes_no_incompleteness_claim(self):
        """The warning must not be permanent furniture."""
        assert "INCOMPLETE" not in render_text(report())


class TestRenderJson:
    """The form another job consumes."""

    def test_it_is_valid_json_carrying_the_same_counts(self):
        """The two renderings are built from one object and must agree."""
        data = json.loads(render_json(report([finding(ERROR), finding(WARNING, "b")])))
        assert data["counts"] == {"error": 1, "warning": 1, "info": 0}
        assert data["catalog_entries"] == 4

    def test_completeness_is_a_top_level_key(self):
        """A consumer must be able to branch on it without parsing prose."""
        assert to_dict(report())["complete"] is True
        incomplete = to_dict(
            report(unavailable=[Unavailable(LITELLM, REASON_NETWORK, "offline")])
        )
        assert incomplete["complete"] is False
        assert incomplete["unavailable"] == [
            {"source": LITELLM, "reason": REASON_NETWORK, "detail": "offline"}
        ]

    def test_coverage_is_reported_per_provider(self):
        """The floors a scheduled job watches are machine-readable."""
        coverage = to_dict(report())["coverage"]["remote_openai"]
        assert coverage["entries"] == 4
        assert coverage["matched_models_dev"] == 3
        assert coverage["match_rate"] == 0.75

    def test_findings_carry_every_citation(self):
        """The JSON form must not be a lossy summary of the text form."""
        data = to_dict(report([finding(cites=2)]))
        assert len(data["findings"][0]["dataset_values"]) == 2
        assert data["findings"][0]["field"] == "input_modalities"

    def test_findings_are_ordered_the_same_way_the_text_is(self):
        """One order, so the two forms cannot describe different reports."""
        data = to_dict(report([finding(INFO, "c"), finding(ERROR, "a")]))
        assert [f["severity"] for f in data["findings"]] == [ERROR, INFO]


class TestRenderIssueBody:
    """The body of the sticky tracking issue."""

    def test_it_states_the_adjudication_rule(self):
        """A reader arriving cold must not read a finding as a fix to apply."""
        body = render_issue_body(report([finding()]))
        assert "may fill a gap" in body
        assert "may never overturn" in body

    def test_it_embeds_the_full_text_report(self):
        """The issue has to stand alone, without the job log."""
        assert "Coverage by provider type" in render_issue_body(report([finding()]))

    def test_an_incomplete_run_is_announced_first(self):
        """The same rule as the text form, in the surface a human reads."""
        body = render_issue_body(
            report(unavailable=[Unavailable(MODELS_DEV, REASON_NETWORK, "offline")])
        )
        assert "INCOMPLETE" in body
        assert body.index("INCOMPLETE") < body.index("catalog entries")

    def test_a_run_url_is_included_when_given(self):
        """A reader needs a way back to the job that produced this."""
        assert "https://ci.example/run/1" in render_issue_body(
            report([finding()]), run_url="https://ci.example/run/1"
        )

    def test_it_carries_the_sticky_marker(self):
        """The job that posts this finds its own issue by the marker.

        Rendering it here rather than appending it in the job is what keeps
        the marker the job searches for and the marker the body carries from
        being two separate strings that can drift.
        """
        assert STICKY_MARKER in render_issue_body(report([finding()]))

    def test_no_run_url_is_rendered_when_absent(self):
        """An empty link is worse than none."""
        assert "Produced by" not in render_issue_body(report([finding()]))
