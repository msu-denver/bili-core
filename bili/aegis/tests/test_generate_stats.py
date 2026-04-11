"""Tests for generate_stats pure computation functions.

Tests the statistics computation and report formatting
without touching the filesystem.
"""

import pytest

from bili.aegis.suites.analysis.generate_stats import (
    _payload_succeeded,
    _tier1_pass,
    _tier3_score,
    compute_persistence_stats,
    compute_suite_stats,
    compute_transferability_stats,
    format_report,
)


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
