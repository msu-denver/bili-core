"""Tests for :mod:`bili.aegis.suites.probe._csv`."""

import csv
import tempfile
from pathlib import Path
from typing import Any

from bili.aegis.probe.schema import PROBE_CSV_COLUMNS
from bili.aegis.suites.probe._csv import (
    CSV_FILENAME,
    append_probe_csv_row,
    write_probe_csv,
)


def _row(**kwargs: Any) -> dict[str, Any]:
    """Build a row dict with all PROBE_CSV_COLUMNS populated."""
    base = {col: "" for col in PROBE_CSV_COLUMNS}
    base.update(
        {
            "payload_id": "",
            "injection_type": "test",
            "severity": "high",
            "stub_mode": "stub",
            "mas_id": "simple_chain",
            "phase": "",
            "tier1_pass": "false",
            "tier2_influenced": "[]",
            "tier2_resistant": "[]",
            "tier3_score": 1,
            "tier3_confidence": "low",
            "tier3_reasoning": "default reasoning",
            "attack_suite": "probe",
            "session_id": "sess-1",
            "objective_id": "pr_test_001",
            "policy": "pair",
            "rng_seed": 0,
            "turns_used": 2,
            "budget_used": 0.12,
            "turns_to_compromise": "",
            "terminated_reason": "budget_exceeded",
        }
    )
    base.update(kwargs)
    return base


# =========================================================================
# write_probe_csv
# =========================================================================


def test_write_probe_csv_creates_file(tmp_path: Path):
    """write_probe_csv writes the canonical filename under results_dir."""
    target = write_probe_csv([_row()], tmp_path)
    assert target == tmp_path / CSV_FILENAME
    assert target.exists()


def test_write_probe_csv_writes_header_row():
    """The first line of the CSV is the column header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_probe_csv([_row()], Path(tmpdir))
        text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0]
    for col in PROBE_CSV_COLUMNS:
        assert col in first_line


def test_write_probe_csv_writes_all_rows(tmp_path: Path):
    """Multiple rows are all written (data lines = header + rows)."""
    rows = [_row(session_id=f"sess-{i}") for i in range(5)]
    path = write_probe_csv(rows, tmp_path)
    # Read back and count
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        data_rows = list(reader)
    assert len(data_rows) == 5
    assert [r["session_id"] for r in data_rows] == [f"sess-{i}" for i in range(5)]


def test_write_probe_csv_overwrites_existing_file(tmp_path: Path):
    """Calling write_probe_csv twice replaces the file (no append)."""
    write_probe_csv([_row(session_id="A")], tmp_path)
    write_probe_csv([_row(session_id="B")], tmp_path)
    path = tmp_path / CSV_FILENAME
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        data_rows = list(reader)
    assert len(data_rows) == 1
    assert data_rows[0]["session_id"] == "B"


def test_write_probe_csv_drops_extra_keys(tmp_path: Path):
    """Extra keys in the row dict are silently dropped (no header bloat)."""
    row = _row()
    row["extra_unknown_key"] = "this should not appear in the CSV"
    write_probe_csv([row], tmp_path)
    path = tmp_path / CSV_FILENAME
    assert "extra_unknown_key" not in path.read_text(encoding="utf-8")


def test_write_probe_csv_uses_empty_string_for_missing_keys(tmp_path: Path):
    """Missing keys in the row dict are written as empty strings."""
    sparse_row = {"session_id": "sparse-1"}
    write_probe_csv([sparse_row], tmp_path)
    path = tmp_path / CSV_FILENAME
    with open(path, "r", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    assert data[0]["session_id"] == "sparse-1"
    assert data[0]["objective_id"] == ""
    assert data[0]["policy"] == ""


def test_write_probe_csv_creates_results_dir(tmp_path: Path):
    """results_dir is created on demand."""
    target = tmp_path / "nested" / "dir" / "results"
    assert not target.exists()
    write_probe_csv([_row()], target)
    assert target.exists()
    assert (target / CSV_FILENAME).exists()


# =========================================================================
# append_probe_csv_row
# =========================================================================


def test_append_probe_csv_row_creates_file_with_header_on_first_call(
    tmp_path: Path,
):
    """First append to a non-existent file writes header + the row."""
    append_probe_csv_row(_row(session_id="A"), tmp_path)
    path = tmp_path / CSV_FILENAME
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Line 0 is the header, line 1 is the data row
    assert len(lines) == 2
    for col in PROBE_CSV_COLUMNS:
        assert col in lines[0]


def test_append_probe_csv_row_does_not_duplicate_header_on_subsequent_calls(
    tmp_path: Path,
):
    """Subsequent appends do NOT re-write the header.

    Anti-cheat: a missing exists-check would duplicate the header,
    breaking csv.DictReader.
    """
    append_probe_csv_row(_row(session_id="A"), tmp_path)
    append_probe_csv_row(_row(session_id="B"), tmp_path)
    append_probe_csv_row(_row(session_id="C"), tmp_path)
    path = tmp_path / CSV_FILENAME
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["session_id"] for r in rows] == ["A", "B", "C"]


def test_append_probe_csv_row_preserves_column_order(tmp_path: Path):
    """The header row uses the documented PROBE_CSV_COLUMNS order."""
    append_probe_csv_row(_row(), tmp_path)
    path = tmp_path / CSV_FILENAME
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    # The very first column should be payload_id (cross-suite convention)
    assert first_line.startswith("payload_id,")


def test_append_probe_csv_row_creates_results_dir(tmp_path: Path):
    """results_dir is created on demand by append too."""
    target = tmp_path / "deeply" / "nested" / "results"
    append_probe_csv_row(_row(), target)
    assert (target / CSV_FILENAME).exists()


def test_append_probe_csv_row_round_trips_through_csv_reader(tmp_path: Path):
    """Appended rows parse back as the same dict (modulo string coercion)."""
    original = _row(
        session_id="round-trip",
        objective_id="pr_x",
        tier3_score=2,
    )
    append_probe_csv_row(original, tmp_path)
    path = tmp_path / CSV_FILENAME
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "round-trip"
    assert rows[0]["objective_id"] == "pr_x"
    # All CSV fields are strings on round-trip
    assert rows[0]["tier3_score"] == "2"


def test_csv_filename_constant_is_probe_results_matrix():
    """The canonical filename matches the documented cross-suite schema.

    Anti-cheat: catches an accidental rename that would break cross-suite
    pandas analysis.
    """
    assert CSV_FILENAME == "probe_results_matrix.csv"


def test_columns_constant_has_21_entries():
    """Schema sanity-check: 13 cross-suite + 8 PROBE-specific = 21."""
    assert len(PROBE_CSV_COLUMNS) == 21
