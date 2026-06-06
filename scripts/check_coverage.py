#!/usr/bin/env python3
"""Per-file coverage gate for bili-core.

Enforces the project goal that every runtime file has at least 90% line
coverage. Reads coverage.json produced by
``pytest --cov=bili --cov-report=json``.

The .coveragerc already scopes measurement to runtime code (it omits the
test tree), so every file in the report is shipping code. This gate fails
the build (exit 1) and lists every offender if any runtime file is below the
threshold. A global average gate would let a 50%-covered file hide behind a
crowd of 99%-covered files; the project goal is per-file, so the gate is
per-file.

Usage:
    pytest bili/ --cov=bili --cov-report=json:coverage.json -q
    python scripts/check_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLD = 90.0
COVERAGE_JSON = "coverage.json"


def main() -> int:
    path = Path(COVERAGE_JSON)
    if not path.exists():
        print(
            f"ERROR: {COVERAGE_JSON} not found. Run pytest with "
            f"--cov=bili --cov-report=json:{COVERAGE_JSON} first.",
            file=sys.stderr,
        )
        return 2

    data = json.loads(path.read_text())
    files = data.get("files", {})

    if not files:
        print(
            f"ERROR: {COVERAGE_JSON} has no measured files. Coverage did not run "
            "correctly; re-run the test suite with coverage enabled.",
            file=sys.stderr,
        )
        return 2

    offenders = []
    for file_path, fdata in sorted(files.items()):
        summary = fdata["summary"]
        if summary["num_statements"] == 0:
            # No executable statements (e.g. an empty __init__). Nothing to cover.
            continue
        pct = summary["percent_covered"]
        if pct < THRESHOLD:
            offenders.append((file_path, pct, summary["missing_lines"]))

    total = data.get("totals", {}).get("percent_covered", 0.0)

    if offenders:
        print(
            f"FAIL: {len(offenders)} runtime file(s) below {THRESHOLD:.0f}% "
            f"line coverage (overall {total:.1f}%):\n"
        )
        for file_path, pct, missing in offenders:
            print(f"  {pct:5.1f}%  ({missing:>4} lines uncovered)  {file_path}")
        print(
            f"\nEvery runtime file must reach >= {THRESHOLD:.0f}% line coverage. "
            "Add tests for the uncovered lines (see the per-file --cov-report=term-missing "
            "output for exact line numbers), or mark genuinely unreachable lines with "
            "`# pragma: no cover` and a one-line reason."
        )
        return 1

    print(
        f"PASS: all {len(files)} runtime files >= {THRESHOLD:.0f}% line coverage "
        f"(overall {total:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
