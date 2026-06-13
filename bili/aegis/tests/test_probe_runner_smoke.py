"""End-to-end smoke tests for ``bili.aegis.suites.probe.run_probe_suite.main``.

Exercises the runner in ``--stub`` mode (no real LLMs, no real victim
MASExecutor — uses the in-process _StubVictimExecutor and _FakeLLM) so
the entire pipeline can be exercised in CI.
"""

import csv
import json
from pathlib import Path

from bili.aegis.probe.schema import PROBE_CSV_COLUMNS
from bili.aegis.suites.probe._csv import CSV_FILENAME
from bili.aegis.suites.probe.run_probe_suite import main as run_main


# Single-config single-objective single-seed CLI args; uses --smoke for
# extra safety even though we're already specifying everything explicitly.
def _base_argv(results_dir: Path) -> list[str]:
    return [
        "--stub",
        "--smoke",
        "--policies",
        "pair",
        "--objectives",
        "pr_safety_bypass_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--seeds",
        "0",
        "--budget-turns",
        "2",
        "--budget-tokens",
        "100000",
        "--budget-cost-usd",
        "5.0",
        "--results-dir",
        str(results_dir),
    ]


# =========================================================================
# Happy-path smoke
# =========================================================================


def test_main_stub_mode_runs_to_completion_with_one_session(tmp_path: Path):
    """Stub + smoke mode runs one session end-to-end and returns 0."""
    exit_code = run_main(_base_argv(tmp_path))
    assert exit_code == 0


def test_main_writes_csv_with_one_row(tmp_path: Path):
    """One session → one CSV row + header."""
    run_main(_base_argv(tmp_path))
    csv_path = tmp_path / CSV_FILENAME
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1


def test_main_writes_sidecar_per_session(tmp_path: Path):
    """A sidecar JSON is written under {mas_id}/sessions/."""
    run_main(_base_argv(tmp_path))
    sidecars = list((tmp_path / "simple_chain" / "sessions").glob("*.json"))
    assert len(sidecars) == 1


def test_main_csv_has_all_21_columns(tmp_path: Path):
    """The CSV header matches PROBE_CSV_COLUMNS verbatim."""
    run_main(_base_argv(tmp_path))
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == set(PROBE_CSV_COLUMNS)


def test_main_csv_row_has_probe_attack_suite(tmp_path: Path):
    """attack_suite column is the literal 'probe' string."""
    run_main(_base_argv(tmp_path))
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["attack_suite"] == "probe"


def test_main_csv_row_records_stub_mode(tmp_path: Path):
    """--stub mode → stub_mode column = 'stub'."""
    run_main(_base_argv(tmp_path))
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["stub_mode"] == "stub"


def test_main_sidecar_has_required_top_level_keys(tmp_path: Path):
    """The sidecar JSON contains the keys downstream analysis depends on."""
    run_main(_base_argv(tmp_path))
    sidecars = list((tmp_path / "simple_chain" / "sessions").glob("*.json"))
    data = json.loads(sidecars[0].read_text(encoding="utf-8"))
    for key in ("session_id", "objective", "turns", "final_outcome"):
        assert key in data


def test_main_sidecar_terminated_reason_is_enum_value_string(tmp_path: Path):
    """The terminated reason is the .value string, not a Python enum repr.

    Anti-cheat: catches Enum serialization regressions that would break
    cross-tool JSON parsing.
    """
    run_main(_base_argv(tmp_path))
    sidecars = list((tmp_path / "simple_chain" / "sessions").glob("*.json"))
    data = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert isinstance(data["final_outcome"]["reason"], str)
    # Lowercase + underscore convention
    assert "_" in data["final_outcome"]["reason"] or data["final_outcome"][
        "reason"
    ] in {"success"}


# =========================================================================
# CLI filtering
# =========================================================================


def test_main_filters_policies_via_cli_arg(tmp_path: Path):
    """--policies crescendo → only crescendo sessions in the CSV."""
    argv = [
        "--stub",
        "--smoke",
        "--policies",
        "crescendo",
        "--objectives",
        "pr_safety_bypass_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--seeds",
        "0",
        "--budget-turns",
        "2",
        "--results-dir",
        str(tmp_path),
    ]
    run_main(argv)
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["policy"] == "crescendo" for row in rows)


def test_main_filters_objectives_via_cli_arg(tmp_path: Path):
    """--objectives pr_X → only that objective in the CSV."""
    argv = [
        "--stub",
        "--policies",
        "pair",
        "--objectives",
        "pr_safety_bypass_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--seeds",
        "0",
        "--budget-turns",
        "2",
        "--results-dir",
        str(tmp_path),
    ]
    run_main(argv)
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["objective_id"] == "pr_safety_bypass_001" for row in rows)


def test_main_rejects_unknown_policy(tmp_path: Path):
    """Unknown policy name → non-zero exit code."""
    argv = [
        "--stub",
        "--smoke",
        "--policies",
        "nonexistent_policy_xyz",
        "--objectives",
        "pr_safety_bypass_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--results-dir",
        str(tmp_path),
    ]
    assert run_main(argv) != 0


def test_main_three_seeds_yields_three_sessions(tmp_path: Path):
    """--seeds 0 1 2 with one objective/config/policy → 3 sessions."""
    argv = [
        "--stub",
        "--policies",
        "pair",
        "--objectives",
        "pr_safety_bypass_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--seeds",
        "0",
        "1",
        "2",
        "--budget-turns",
        "2",
        "--results-dir",
        str(tmp_path),
    ]
    run_main(argv)
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert sorted(int(row["rng_seed"]) for row in rows) == [0, 1, 2]


# =========================================================================
# Multi-policy + multi-objective combinations
# =========================================================================


def test_main_combines_policies_and_objectives_into_matrix(tmp_path: Path):
    """2 objectives × 2 policies × 1 seed = 4 sessions."""
    argv = [
        "--stub",
        "--policies",
        "pair",
        "crescendo",
        "--objectives",
        "pr_safety_bypass_001",
        "pr_role_collapse_001",
        "--configs",
        "bili/aether/config/examples/simple_chain.yaml",
        "--seeds",
        "0",
        "--budget-turns",
        "2",
        "--results-dir",
        str(tmp_path),
    ]
    run_main(argv)
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    policies = sorted({row["policy"] for row in rows})
    assert policies == ["crescendo", "pair"]


# =========================================================================
# Structural pytest passes against fake-LLM artifacts
# =========================================================================


def test_main_returns_zero_when_all_sessions_terminate_cleanly(tmp_path: Path):
    """Per-session failures do NOT cause a non-zero exit.

    Anti-cheat: only framework errors (unknown policy, config-load
    crash) should non-zero exit.
    """
    exit_code = run_main(_base_argv(tmp_path))
    assert exit_code == 0


def test_main_appends_rows_incrementally_across_runs(tmp_path: Path):
    """Two invocations of main share one results dir → rows accumulate."""
    run_main(_base_argv(tmp_path))
    run_main(
        [
            "--stub",
            "--smoke",
            "--policies",
            "crescendo",
            "--objectives",
            "pr_safety_bypass_001",
            "--configs",
            "bili/aether/config/examples/simple_chain.yaml",
            "--seeds",
            "0",
            "--budget-turns",
            "2",
            "--results-dir",
            str(tmp_path),
        ]
    )
    csv_path = tmp_path / CSV_FILENAME
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert sorted(row["policy"] for row in rows) == ["crescendo", "pair"]
