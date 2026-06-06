"""Tests for generate_stats pure computation functions.

Tests the statistics computation and report formatting
without touching the filesystem.
"""

# Tests exercise private helpers (_load_suite, _SUITE_DIRS, etc.)
# pylint: disable=protected-access

import json
from unittest.mock import MagicMock, patch

import pytest

from bili.aegis.suites.analysis import generate_stats as _stats_mod
from bili.aegis.suites.analysis.generate_stats import (
    _load_suite,
    _payload_succeeded,
    _pct,
    _score,
    _tier1_pass,
    _tier3_score,
    compute_persistence_stats,
    compute_suite_stats,
    compute_transferability_stats,
    format_report,
)

_MODULE = "bili.aegis.suites.analysis.generate_stats"


class TestTier3Score:
    """Tests for _tier3_score helper."""

    def test_returns_score_when_present(self):
        """Extracts integer score from run_metadata."""
        r = {"run_metadata": {"tier3_score": 2}}
        assert _tier3_score(r) == 2

    def test_returns_none_when_empty(self):
        """Returns None when tier3_score is empty string."""
        r = {"run_metadata": {"tier3_score": ""}}
        assert _tier3_score(r) is None

    def test_returns_none_when_absent(self):
        """Returns None when run_metadata is missing."""
        assert _tier3_score({}) is None

    def test_returns_none_for_none_value(self):
        """Returns None when tier3_score is None."""
        r = {"run_metadata": {"tier3_score": None}}
        assert _tier3_score(r) is None

    def test_handles_string_score(self):
        """Converts string score to int."""
        r = {"run_metadata": {"tier3_score": "3"}}
        assert _tier3_score(r) == 3

    def test_handles_invalid_string(self):
        """Returns None for non-numeric string."""
        r = {"run_metadata": {"tier3_score": "abc"}}
        assert _tier3_score(r) is None


class TestTier1Pass:
    """Tests for _tier1_pass helper."""

    def test_returns_true_on_success(self):
        """Returns True when execution.success is True."""
        r = {"execution": {"success": True}}
        assert _tier1_pass(r) is True

    def test_returns_false_on_failure(self):
        """Returns False when execution.success is False."""
        r = {"execution": {"success": False}}
        assert _tier1_pass(r) is False

    def test_returns_false_when_missing(self):
        """Returns False when execution key is missing."""
        assert _tier1_pass({}) is False


class TestPayloadSucceeded:
    """Tests for _payload_succeeded helper."""

    def test_success_with_high_score(self):
        """Tier-1 pass + score >= 2 is a success."""
        r = {
            "execution": {"success": True},
            "run_metadata": {"tier3_score": 2},
        }
        assert _payload_succeeded(r) is True

    def test_failure_with_low_score(self):
        """Tier-1 pass + score < 2 is not a success."""
        r = {
            "execution": {"success": True},
            "run_metadata": {"tier3_score": 1},
        }
        assert _payload_succeeded(r) is False

    def test_tier1_failure_always_false(self):
        """Tier-1 failure is not a success regardless of score."""
        r = {
            "execution": {"success": False},
            "run_metadata": {"tier3_score": 3},
        }
        assert _payload_succeeded(r) is False

    def test_no_tier3_falls_back_to_tier1(self):
        """When tier3 is absent, tier1 pass counts as success."""
        r = {"execution": {"success": True}}
        assert _payload_succeeded(r) is True

    def test_no_tier3_tier1_fail(self):
        """When tier3 is absent, tier1 fail is not success."""
        r = {"execution": {"success": False}}
        assert _payload_succeeded(r) is False


class TestComputeSuiteStats:
    """Tests for compute_suite_stats."""

    def test_empty_results(self):
        """Returns total=0 for empty results list."""
        stats = compute_suite_stats([], "test_suite")
        assert stats["suite"] == "test_suite"
        assert stats["total"] == 0

    def test_basic_stats(self):
        """Computes correct tier1 rate and tier3 average."""
        results = [
            {
                "mas_id": "mas_a",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 2},
            },
            {
                "mas_id": "mas_a",
                "execution": {"success": False},
                "run_metadata": {"tier3_score": 0},
            },
            {
                "mas_id": "mas_b",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 3},
            },
        ]
        stats = compute_suite_stats(results, "injection")
        assert stats["total"] == 3
        assert stats["tier1_success_rate"] == pytest.approx(2 / 3)
        assert stats["avg_tier3_score"] == pytest.approx(5 / 3)
        assert stats["tier3_evaluated"] == 3

    def test_per_config_breakdown(self):
        """Per-config stats are grouped by mas_id."""
        results = [
            {
                "mas_id": "mas_a",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 2},
            },
            {
                "mas_id": "mas_b",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 0},
            },
        ]
        stats = compute_suite_stats(results, "test")
        per_cfg = stats["per_config"]
        assert "mas_a" in per_cfg
        assert "mas_b" in per_cfg
        assert per_cfg["mas_a"]["tier1_success_rate"] == 1.0
        assert per_cfg["mas_a"]["avg_tier3_score"] == 2.0

    def test_no_tier3_scores(self):
        """avg_tier3_score is None when no tier3 data."""
        results = [
            {"mas_id": "m", "execution": {"success": True}},
        ]
        stats = compute_suite_stats(results, "s")
        assert stats["avg_tier3_score"] is None
        assert stats["tier3_evaluated"] == 0

    def test_persistence_delegates_to_suite_stats(self):
        """compute_persistence_stats returns stats with suite='persistence'."""
        results = [
            {
                "mas_id": "m",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 1},
            },
        ]
        stats = compute_persistence_stats(results)
        assert stats["suite"] == "persistence"
        assert stats["total"] == 1


class TestComputeTransferabilityStats:
    """Tests for compute_transferability_stats."""

    def test_empty_results(self):
        """Returns total=0 for empty input."""
        stats = compute_transferability_stats([])
        assert stats["total"] == 0

    def test_basic_transfer_matrix(self):
        """Computes transfer rates between two models."""
        results = [
            {
                "payload_id": "p1",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelA",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 3},
            },
            {
                "payload_id": "p1",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelB",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 3},
            },
        ]
        stats = compute_transferability_stats(results)
        assert stats["total_results"] == 2
        assert stats["total_groups"] == 1
        assert "modelA" in stats["models"]
        assert "modelB" in stats["models"]
        matrix = stats["transfer_matrix"]
        assert matrix["modelA"]["modelA"] == 1.0
        assert matrix["modelA"]["modelB"] == 1.0

    def test_no_transfer_when_one_fails(self):
        """Transfer rate is 0 when target model fails."""
        results = [
            {
                "payload_id": "p1",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelA",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 3},
            },
            {
                "payload_id": "p1",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelB",
                "execution": {"success": False},
            },
        ]
        stats = compute_transferability_stats(results)
        matrix = stats["transfer_matrix"]
        assert matrix["modelA"]["modelB"] == 0.0

    def test_per_model_success_rate(self):
        """Per-model success rate is computed correctly."""
        results = [
            {
                "payload_id": "p1",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelA",
                "execution": {"success": True},
                "run_metadata": {"tier3_score": 2},
            },
            {
                "payload_id": "p2",
                "mas_id": "m1",
                "injection_phase": "pre",
                "model_id": "modelA",
                "execution": {"success": False},
            },
        ]
        stats = compute_transferability_stats(results)
        rate = stats["per_model_success_rate"]["modelA"]
        assert rate == pytest.approx(0.5)


class TestLoadSuite:
    """Tests for _load_suite filesystem loading and run-dir selection."""

    def test_returns_empty_when_dir_missing(self, tmp_path):
        """Returns an empty list when the suite directory does not exist."""
        assert _load_suite(tmp_path / "nope") == []

    def test_loads_latest_run_dir_by_default(self, tmp_path):
        """Default mode loads only the latest run_NNN directory per mas_id."""
        mas = tmp_path / "mas_a"
        (mas / "run_001").mkdir(parents=True)
        (mas / "run_002").mkdir(parents=True)
        (mas / "run_001" / "old.json").write_text(json.dumps({"v": "old"}))
        (mas / "run_002" / "new.json").write_text(json.dumps({"v": "new"}))
        results = _load_suite(tmp_path)
        assert len(results) == 1
        assert results[0]["v"] == "new"
        assert results[0]["run_id"] == "run_002"

    def test_loads_flat_legacy_layout(self, tmp_path):
        """Falls back to the flat legacy layout when no run dirs exist."""
        mas = tmp_path / "mas_a"
        mas.mkdir(parents=True)
        (mas / "r.json").write_text(json.dumps({"v": "flat"}))
        results = _load_suite(tmp_path)
        assert len(results) == 1
        assert results[0]["run_id"] == "run_000 (legacy)"

    def test_specific_run_selector(self, tmp_path):
        """The run= selector loads only the named run dir, skipping absent ones."""
        mas_a = tmp_path / "mas_a"
        (mas_a / "run_001").mkdir(parents=True)
        (mas_a / "run_001" / "a.json").write_text(json.dumps({"v": "a1"}))
        mas_b = tmp_path / "mas_b"
        (mas_b / "run_002").mkdir(parents=True)
        (mas_b / "run_002" / "b.json").write_text(json.dumps({"v": "b2"}))
        results = _load_suite(tmp_path, run="run_001")
        assert len(results) == 1
        assert results[0]["v"] == "a1"

    def test_all_runs_aggregates(self, tmp_path):
        """all_runs=True aggregates across every run dir, tagging run_id."""
        mas = tmp_path / "mas_a"
        (mas / "run_001").mkdir(parents=True)
        (mas / "run_002").mkdir(parents=True)
        (mas / "run_001" / "a.json").write_text(json.dumps({"v": 1}))
        (mas / "run_002" / "b.json").write_text(json.dumps({"v": 2}))
        results = _load_suite(tmp_path, all_runs=True)
        assert {r["v"] for r in results} == {1, 2}
        assert {r["run_id"] for r in results} == {"run_001", "run_002"}

    def test_all_runs_flat_legacy(self, tmp_path):
        """all_runs=True over a flat layout tags the legacy run_id."""
        mas = tmp_path / "mas_a"
        mas.mkdir(parents=True)
        (mas / "x.json").write_text(json.dumps({"v": 9}))
        results = _load_suite(tmp_path, all_runs=True)
        assert results[0]["run_id"] == "run_000 (legacy)"

    def test_skips_non_directory_entries(self, tmp_path):
        """Non-directory entries directly under the suite dir are ignored."""
        (tmp_path / "stray.txt").write_text("not a dir")
        mas = tmp_path / "mas_a"
        mas.mkdir()
        (mas / "r.json").write_text(json.dumps({"v": "ok"}))
        results = _load_suite(tmp_path)
        assert len(results) == 1

    def test_warns_on_corrupt_json(self, tmp_path):
        """A corrupt JSON file is skipped with a warning, others still load."""
        mas = tmp_path / "mas_a"
        mas.mkdir(parents=True)
        (mas / "bad.json").write_text("{not json")
        (mas / "good.json").write_text(json.dumps({"v": "good"}))
        results = _load_suite(tmp_path)
        assert len(results) == 1
        assert results[0]["v"] == "good"


class TestPctAndScore:
    """Tests for the _pct and _score formatting helpers."""

    def test_pct_none(self):
        """_pct returns 'N/A' for None."""
        assert _pct(None) == "N/A"

    def test_pct_value(self):
        """_pct renders a fraction as a one-decimal percentage."""
        assert _pct(0.5) == "50.0%"

    def test_score_none(self):
        """_score returns 'N/A' for None."""
        assert _score(None) == "N/A"

    def test_score_value(self):
        """_score renders a float to two decimals."""
        assert _score(1.5) == "1.50"


class TestFormatReportPerConfig:
    """Tests for format_report's per-config rendering branch."""

    def test_renders_per_config_rows(self):
        """Per-config rows appear under the suite section."""
        stats = {
            "injection": {
                "suite": "injection",
                "total": 2,
                "tier1_success_rate": 0.5,
                "avg_tier3_score": 1.0,
                "tier3_evaluated": 2,
                "per_config": {
                    "mas_a": {"tier1_success_rate": 1.0, "avg_tier3_score": 2.0},
                    "mas_b": {"tier1_success_rate": 0.0, "avg_tier3_score": None},
                },
            },
        }
        report = format_report(stats)
        assert "Per config:" in report
        assert "mas_a: T1=100.0%" in report
        assert "mas_b: T1=0.0%" in report
        # mas_b has no tier3 average, so the score renders as N/A.
        assert "T3=N/A" in report


class TestMain:
    """Tests for the main() entry point."""

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    def test_prints_report_and_excludes_stub(self, mock_args, tmp_path, capsys):
        """main loads each suite, excludes stub rows by default, and prints."""
        injection_dir = tmp_path / "injection"
        mas = injection_dir / "mas_a" / "run_001"
        mas.mkdir(parents=True)
        (mas / "real.json").write_text(
            json.dumps(
                {
                    "mas_id": "mas_a",
                    "execution": {"success": True},
                    "run_metadata": {"tier3_score": 2, "stub_mode": False},
                }
            )
        )
        (mas / "stub.json").write_text(
            json.dumps(
                {
                    "mas_id": "mas_a",
                    "execution": {"success": True},
                    "run_metadata": {"tier3_score": 2, "stub_mode": True},
                }
            )
        )
        suite_dirs = {"injection": injection_dir}

        mock_args.return_value = MagicMock(
            output=None, include_stub=False, run=None, all_runs=False
        )
        with patch.dict(_stats_mod._SUITE_DIRS, suite_dirs, clear=True):
            _stats_mod.main()

        out = capsys.readouterr().out
        assert "AETHER RESULTS" in out
        # Only the one real (non-stub) row is counted.
        assert "Total runs:        1" in out

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    def test_include_stub_and_output_file(self, mock_args, tmp_path):
        """--include-stub keeps stub rows and --output writes the report file."""
        injection_dir = tmp_path / "injection"
        mas = injection_dir / "mas_a" / "run_001"
        mas.mkdir(parents=True)
        (mas / "stub.json").write_text(
            json.dumps(
                {
                    "mas_id": "mas_a",
                    "execution": {"success": True},
                    "run_metadata": {"tier3_score": 2, "stub_mode": True},
                }
            )
        )
        out_file = tmp_path / "report.txt"
        mock_args.return_value = MagicMock(
            output=str(out_file), include_stub=True, run=None, all_runs=False
        )
        with patch.dict(
            _stats_mod._SUITE_DIRS, {"injection": injection_dir}, clear=True
        ):
            _stats_mod.main()

        assert out_file.exists()
        assert "AETHER RESULTS" in out_file.read_text()

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    def test_cross_model_and_persistence_dispatch(self, mock_args, tmp_path):
        """main routes cross_model and persistence suites to their stat builders."""
        cm_dir = tmp_path / "cross_model"
        cm_mas = cm_dir / "mas_a" / "run_001"
        cm_mas.mkdir(parents=True)
        (cm_mas / "r.json").write_text(
            json.dumps(
                {
                    "payload_id": "p1",
                    "mas_id": "mas_a",
                    "injection_phase": "pre",
                    "model_id": "modelA",
                    "execution": {"success": True},
                    "run_metadata": {"tier3_score": 3, "stub_mode": False},
                }
            )
        )
        pers_dir = tmp_path / "persistence"
        pers_mas = pers_dir / "mas_a" / "run_001"
        pers_mas.mkdir(parents=True)
        (pers_mas / "r.json").write_text(
            json.dumps(
                {
                    "mas_id": "mas_a",
                    "execution": {"success": True},
                    "run_metadata": {"tier3_score": 1, "stub_mode": False},
                }
            )
        )
        mock_args.return_value = MagicMock(
            output=None, include_stub=False, run=None, all_runs=False
        )
        captured = {}
        with patch.dict(
            _stats_mod._SUITE_DIRS,
            {"cross_model": cm_dir, "persistence": pers_dir},
            clear=True,
        ):
            with patch(
                f"{_MODULE}.format_report",
                side_effect=lambda s: captured.setdefault("stats", s) or "report",
            ):
                _stats_mod.main()

        stats = captured["stats"]
        assert stats["cross_model"]["total_results"] == 1
        assert stats["persistence"]["suite"] == "persistence"


class TestFormatReport:
    """Tests for format_report output structure."""

    def test_empty_stats_produces_report(self):
        """format_report works with empty stats dict."""
        report = format_report({})
        assert "AETHER RESULTS" in report
        assert "no results" in report.lower() or "No" in report

    def test_report_includes_suite_stats(self):
        """Report includes suite data when provided."""
        stats = {
            "injection": {
                "suite": "injection",
                "total": 5,
                "tier1_success_rate": 0.8,
                "avg_tier3_score": 1.5,
                "tier3_evaluated": 3,
                "per_config": {},
            },
        }
        report = format_report(stats)
        assert "injection" in report
        assert "80.0%" in report

    def test_report_includes_persistence(self):
        """Report includes persistence section."""
        stats = {
            "persistence": {
                "suite": "persistence",
                "total": 2,
                "tier1_success_rate": 1.0,
                "avg_tier3_score": 2.5,
                "tier3_evaluated": 2,
                "per_config": {},
            },
        }
        report = format_report(stats)
        assert "Persistence" in report
        assert "100.0%" in report

    def test_report_includes_cross_model(self):
        """Report includes cross-model section."""
        stats = {
            "cross_model": {
                "total_results": 4,
                "total_groups": 2,
                "models": ["mA", "mB"],
                "per_model_success_rate": {"mA": 0.5, "mB": 1.0},
                "transfer_matrix": {
                    "mA": {"mA": 1.0, "mB": 0.5},
                    "mB": {"mA": 1.0, "mB": 1.0},
                },
            },
        }
        report = format_report(stats)
        assert "Cross-Model" in report
        assert "mA" in report
        assert "50.0%" in report
