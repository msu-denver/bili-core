"""Rendering a :class:`~.compare.DivergenceReport`, as text and as JSON.

Two forms, one report.  The text form is what a maintainer reads in a job log
or an issue body; the JSON form is what another job consumes.  Both are
rendered from the same object, so they cannot disagree about what was found.

The rendering rule that matters is that an incomplete run says so, loudly and
first.  A checker whose fetch failed and which then prints a clean summary has
told the reader the opposite of the truth, so an unreadable dataset is stated
at the top of the text form, is a top-level key in the JSON form, and is
reflected in the process exit code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .compare import ERROR, INFO, SEVERITY_ORDER, WARNING, DivergenceReport, Finding


def _finding_sort_key(finding: Finding):
    """Order findings most severe first, then stably by provider and model.

    :param finding: The finding to key.
    :returns: A sortable tuple.
    """
    return (
        SEVERITY_ORDER.index(finding.severity),
        finding.provider_type,
        finding.model_id,
        finding.field_name,
    )


def sorted_findings(report: DivergenceReport) -> List[Finding]:
    """Return the report's findings in reading order.

    :param report: The comparison result.
    :returns: Findings sorted most severe first.
    :rtype: List[Finding]
    """
    return sorted(report.findings, key=_finding_sort_key)


def _format_dataset_values(finding: Finding) -> str:
    """Render a finding's dataset answers as one parenthesised clause.

    :param finding: The finding to render.
    :returns: A rendered clause, or an empty string when no dataset spoke.
    :rtype: str
    """
    if not finding.dataset_values:
        return ""
    parts = [
        f"{v.source}[{v.provider_id}/{v.key}]={v.value!r}"
        for v in finding.dataset_values
    ]
    return "  " + "; ".join(parts)


def render_text(report: DivergenceReport) -> str:
    """Render the report as human-readable text.

    :param report: The comparison result.
    :returns: The rendered report.
    :rtype: str
    """
    lines: List[str] = []
    lines.append("Model catalog divergence report")
    lines.append(f"generated: {report.generated_at}")
    for source, origin in sorted(report.dataset_origins.items()):
        lines.append(f"source:    {source} <- {origin}")

    if report.unavailable:
        lines.append("")
        lines.append("DATASETS UNREADABLE -- this comparison is INCOMPLETE:")
        for entry in report.unavailable:
            lines.append(f"  {entry.source}: {entry.reason}: {entry.detail}")

    lines.append("")
    lines.append(
        f"catalog entries: {report.catalog_entries}   findings: "
        f"{report.count(ERROR)} error, {report.count(WARNING)} warning, "
        f"{report.count(INFO)} info"
    )

    lines.append("")
    lines.append("Coverage by provider type (entries resolved to a dataset record)")
    lines.append(
        f"  {'provider type':<24} {'entries':>7} {'models.dev':>10} "
        f"{'litellm':>7} {'either':>6} {'rate':>6}"
    )
    for provider_type in sorted(report.matches):
        match = report.matches[provider_type]
        lines.append(
            f"  {provider_type:<24} {match.entries:>7} "
            f"{match.matched_models_dev:>10} {match.matched_litellm:>7} "
            f"{match.matched_either:>6} {match.match_rate:>5.0%}"
        )

    findings = sorted_findings(report)
    if not findings:
        lines.append("")
        lines.append("No divergences found.")
    else:
        current = None
        for finding in findings:
            if finding.severity != current:
                current = finding.severity
                lines.append("")
                lines.append(f"{current}")
            lines.append(
                f"  {finding.provider_type} / {finding.model_id} "
                f"[{finding.field_name}] catalog={finding.catalog_value!r}"
                f"{_format_dataset_values(finding)}"
            )
            lines.append(f"      {finding.message}")

    return "\n".join(lines) + "\n"


def to_dict(report: DivergenceReport) -> Dict[str, Any]:
    """Render the report as a JSON-serialisable dictionary.

    :param report: The comparison result.
    :returns: The report as plain data.
    :rtype: Dict[str, Any]
    """
    return {
        "generated_at": report.generated_at,
        "dataset_origins": dict(sorted(report.dataset_origins.items())),
        "complete": not report.any_unavailable,
        "unavailable": [
            {"source": u.source, "reason": u.reason, "detail": u.detail}
            for u in report.unavailable
        ],
        "catalog_entries": report.catalog_entries,
        "counts": {
            "error": report.count(ERROR),
            "warning": report.count(WARNING),
            "info": report.count(INFO),
        },
        "coverage": {
            provider_type: {
                "entries": match.entries,
                "matched_models_dev": match.matched_models_dev,
                "matched_litellm": match.matched_litellm,
                "matched_either": match.matched_either,
                "match_rate": round(match.match_rate, 4),
            }
            for provider_type, match in sorted(report.matches.items())
        },
        "findings": [
            {
                "severity": f.severity,
                "provider_type": f.provider_type,
                "model_id": f.model_id,
                "model_name": f.model_name,
                "field": f.field_name,
                "catalog_value": f.catalog_value,
                "dataset_values": [
                    {
                        "source": v.source,
                        "provider_id": v.provider_id,
                        "key": v.key,
                        "value": v.value,
                    }
                    for v in f.dataset_values
                ],
                "message": f.message,
            }
            for f in sorted_findings(report)
        ],
    }


def render_json(report: DivergenceReport, indent: int = 2) -> str:
    """Render the report as a JSON document.

    :param report: The comparison result.
    :param indent: JSON indentation.
    :returns: The rendered JSON.
    :rtype: str
    """
    return json.dumps(to_dict(report), indent=indent, sort_keys=False) + "\n"


def render_issue_body(report: DivergenceReport, run_url: str = "") -> str:
    """Render a compact Markdown body for the sticky tracking issue.

    :param report: The comparison result.
    :param run_url: An optional link back to the job that produced this.
    :returns: Markdown suitable for an issue body.
    :rtype: str
    """
    lines: List[str] = []
    lines.append("The scheduled model-catalog divergence check reports findings.")
    lines.append("")
    if report.unavailable:
        lines.append("**This run was INCOMPLETE.** A dataset could not be read:")
        lines.append("")
        for entry in report.unavailable:
            lines.append(f"- `{entry.source}`: {entry.reason} -- {entry.detail}")
        lines.append("")
    lines.append(
        f"- generated: `{report.generated_at}`\n"
        f"- catalog entries: {report.catalog_entries}\n"
        f"- findings: **{report.count(ERROR)} error**, "
        f"{report.count(WARNING)} warning, {report.count(INFO)} info"
    )
    lines.append("")
    lines.append("A finding is a prompt to adjudicate, not a value to adopt: a")
    lines.append("dataset may fill a gap in the catalog and may never overturn a")
    lines.append("declared value.")
    lines.append("")
    lines.append("```")
    lines.append(render_text(report).rstrip("\n"))
    lines.append("```")
    if run_url:
        lines.append("")
        lines.append(f"Produced by {run_url}")
    return "\n".join(lines) + "\n"
