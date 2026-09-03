#!/usr/bin/env python
"""Command-line entry point for the model-catalog divergence check.

Usage (from the project root)::

    python -m bili.iris.config.catalog_divergence.cli
    python -m bili.iris.config.catalog_divergence.cli --json report.json
    python -m bili.iris.config.catalog_divergence.cli --issue-body body.md
    python -m bili.iris.config.catalog_divergence.cli \\
        --models-dev-file fixture.json --litellm-file fixture.json

Exit codes are three, because "the check found nothing" and "the check could
not run" are different answers and collapsing them is how a broken job reports
as a passing one:

``0``
    Every dataset was read and no ``ERROR`` finding was produced.

``1``
    At least one ``ERROR`` finding: the catalog declares a capability an
    authoritative dataset does not list.

``2``
    At least one dataset could not be read, and no ``ERROR`` was produced.
    The comparison, if any, was partial.

Nothing here writes to the catalog.  The check is report-only by construction:
it holds no code path that edits a source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from .compare import compare_catalog
from .datasets import DEFAULT_TIMEOUT_SECONDS, load_litellm, load_models_dev
from .report import render_issue_body, render_json, render_text

EXIT_OK = 0
EXIT_ERRORS = 1
EXIT_UNAVAILABLE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    :returns: The configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="catalog-divergence",
        description=(
            "Report disagreements between the declared model catalog and the "
            "community capability datasets. Report-only; edits nothing."
        ),
    )
    parser.add_argument(
        "--models-dev-file",
        type=Path,
        default=None,
        help="read models.dev from this file instead of the network",
    )
    parser.add_argument(
        "--litellm-file",
        type=Path,
        default=None,
        help="read LiteLLM from this file instead of the network",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the report as JSON to this path",
    )
    parser.add_argument(
        "--issue-body",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "also write a Markdown tracking-issue body to this path; the "
            "rendering lives here rather than in the job that posts it"
        ),
    )
    parser.add_argument(
        "--run-url",
        default="",
        help="link back to the run, included in the tracking-issue body",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the text report (the exit code still reports)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-dataset fetch timeout in seconds",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    stdout: Optional[TextIO] = None,
) -> int:
    """Run the check and return the process exit code.

    :param argv: Argument vector; defaults to ``sys.argv[1:]``.
    :param stdout: Stream to write the text report to; defaults to
        ``sys.stdout``.
    :returns: One of :data:`EXIT_OK`, :data:`EXIT_ERRORS`,
        :data:`EXIT_UNAVAILABLE`.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    out = sys.stdout if stdout is None else stdout

    models_dev = load_models_dev(args.models_dev_file, timeout=args.timeout)
    litellm = load_litellm(args.litellm_file, timeout=args.timeout)
    report = compare_catalog(models_dev, litellm)

    if not args.quiet:
        out.write(render_text(report))

    if args.json is not None:
        args.json.write_text(render_json(report), encoding="utf-8")

    if args.issue_body is not None:
        args.issue_body.write_text(
            render_issue_body(report, run_url=args.run_url), encoding="utf-8"
        )

    if report.has_errors:
        return EXIT_ERRORS
    if report.any_unavailable:
        return EXIT_UNAVAILABLE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
