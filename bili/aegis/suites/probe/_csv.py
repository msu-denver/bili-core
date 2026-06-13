"""PROBE-specific CSV writer.

Distinct from the shared ``bili.aegis.suites._suite_runner._write_csv``
because PROBE has 21 columns (13 cross-suite + 8 PROBE-specific) and the
shared writer hard-codes a 13-column schema. Cross-suite analysis still
composes via ``pandas.concat([df_static, df_probe], ignore_index=True)``
— pandas tolerates column unions.

Two entry points:
* ``write_probe_csv``         — overwrite/create
* ``append_probe_csv_row``    — incremental append; writes header only when
                                the file does not yet exist. Used by the
                                runner so a mid-run crash still leaves
                                partial CSV data on disk for inspection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bili.aegis.probe.schema import PROBE_CSV_COLUMNS

CSV_FILENAME: str = "probe_results_matrix.csv"


def write_probe_csv(rows: list[dict[str, Any]], results_dir: Path) -> Path:
    """Overwrite ``results_dir/probe_results_matrix.csv`` with all rows.

    Creates ``results_dir`` if it does not exist. Header is always
    written. Rows must use the keys declared in
    :data:`bili.aegis.probe.schema.PROBE_CSV_COLUMNS`; any extra keys
    are silently dropped, any missing keys default to ``""``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / CSV_FILENAME
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(PROBE_CSV_COLUMNS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in PROBE_CSV_COLUMNS})
    return target


def append_probe_csv_row(row: dict[str, Any], results_dir: Path) -> Path:
    """Append one row to ``results_dir/probe_results_matrix.csv``.

    Writes the header automatically iff the file does not yet exist.
    Long-running suite runs use this so a Ctrl-C or crash mid-suite
    still leaves the completed sessions' rows on disk.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / CSV_FILENAME
    needs_header = not target.exists()
    with open(target, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(PROBE_CSV_COLUMNS), extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in PROBE_CSV_COLUMNS})
    return target
