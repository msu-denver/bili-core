"""
Structural assertions for PROBE results.

Mirrors the existing pattern in
`bili/aegis/suites/{injection,jailbreak,...}/test_*_structural.py`:
runs against the on-disk results directory, asserts schema invariants. Skipped
automatically if `results/` is empty.

Run with:
    pytest bili/aegis/suites/probe/test_probe_structural.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bili.aegis.probe.schema import PROBE_SPECIFIC_COLUMNS

RESULTS_DIR = Path("bili/aegis/suites/probe/results")
CSV_PATH = RESULTS_DIR / "probe_results_matrix.csv"
# Reuse the canonical PROBE-specific column list from the schema module so
# this test and the source of truth can never drift apart.
EXPECTED_PROBE_COLUMNS = set(PROBE_SPECIFIC_COLUMNS)
EXPECTED_SHARED_COLUMNS = {
    "attack_suite",
    "mas_id",
    "severity",
    "tier1_pass",
    "tier3_score",
}


def _results_present() -> bool:
    return RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir())


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_csv_exists_and_has_required_columns() -> None:
    """CSV exists and contains the required shared + PROBE-specific columns."""
    assert CSV_PATH.exists(), f"missing: {CSV_PATH}"
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
    missing = (EXPECTED_PROBE_COLUMNS | EXPECTED_SHARED_COLUMNS) - cols
    assert not missing, f"CSV missing required columns: {missing}"


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_attack_suite_column_is_probe() -> None:
    """Every CSV row carries ``attack_suite == 'probe'``."""
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            assert (
                row["attack_suite"] == "probe"
            ), f"row has attack_suite={row['attack_suite']!r}, expected 'probe'"


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_terminated_reason_is_valid_enum() -> None:
    """``terminated_reason`` is one of the documented ProbeOutcomeReason values."""
    valid = {
        "success",
        "budget_exceeded",
        "attacker_self_abandoned",
        "victim_crashed",
        "attacker_crashed",
        "judge_unavailable",
    }
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            assert (
                row["terminated_reason"] in valid
            ), f"invalid terminated_reason: {row['terminated_reason']!r}"


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_sidecar_json_exists_for_every_csv_row() -> None:
    """Every CSV row has a corresponding ``results/{mas_id}/sessions/{session_id}.json`` sidecar."""
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        sidecar = RESULTS_DIR / row["mas_id"] / "sessions" / f"{row['session_id']}.json"
        assert sidecar.exists(), f"missing sidecar: {sidecar}"


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_sidecar_json_well_formed() -> None:
    """Every sidecar JSON loads and contains the required top-level keys."""
    for sidecar in RESULTS_DIR.rglob("sessions/*.json"):
        with sidecar.open() as f:
            data = json.load(f)
        for required in ("session_id", "objective", "turns", "final_outcome"):
            assert required in data, f"{sidecar} missing key: {required}"
        assert isinstance(data["turns"], list)


@pytest.mark.skipif(not _results_present(), reason="No PROBE results on disk")
def test_turns_to_compromise_consistent_with_terminated_reason() -> None:
    """If terminated_reason==success, turns_to_compromise must be set; else may be empty."""
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["terminated_reason"] == "success":
                assert row["turns_to_compromise"] not in (
                    "",
                    None,
                ), f"success session {row['session_id']} has empty TTC"
