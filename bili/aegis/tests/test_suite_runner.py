"""Tests for bili.aegis.suites._suite_runner shared suite runner.

All external dependencies (AttackInjector, MASExecutor, load_mas_from_yaml,
SecurityEventDetector, SecurityEventLogger) are mocked so no real MAS
config or LLM is needed.
"""

import datetime
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bili.aegis.suites._suite_runner import (
    _build_result_dict,
    _run_config,
    _target_agent_id,
    _write_csv,
    _write_result,
    run_suite,
)

# =========================================================================
# Helpers
# =========================================================================

_MODULE = "bili.aegis.suites._suite_runner"


def _fake_payload(
    payload_id="p1",
    injection_type="prompt_injection",
    severity="high",
    payload="evil text",
):
    """Build a minimal payload namespace."""
    return SimpleNamespace(
        payload_id=payload_id,
        injection_type=injection_type,
        severity=severity,
        payload=payload,
    )


_SENTINEL = object()


def _fake_attack_result(
    success=True,
    target_agent_id="agent_a",
    propagation_path=None,
    influenced_agents=None,
    resistant_agents=None,
    injected_at=None,
    completed_at=_SENTINEL,
):
    """Build a minimal attack result namespace."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if completed_at is _SENTINEL:
        completed_at = now + datetime.timedelta(seconds=1)
    return SimpleNamespace(
        success=success,
        target_agent_id=target_agent_id,
        propagation_path=propagation_path or ["agent_a"],
        influenced_agents=influenced_agents or [],
        resistant_agents=resistant_agents or ["agent_b"],
        injected_at=injected_at or now,
        completed_at=completed_at,
    )


def _fake_config(
    mas_id="test-mas",
    entry_point=None,
    agents=None,
):
    """Build a minimal MAS config namespace."""
    if agents is None:
        agents = [
            SimpleNamespace(
                agent_id="agent_a",
                model_name="gpt-4o",
                temperature=0.2,
            ),
        ]
    return SimpleNamespace(
        mas_id=mas_id,
        entry_point=entry_point,
        agents=agents,
    )


# =========================================================================
# _target_agent_id
# =========================================================================


class TestTargetAgentId:
    """Tests for _target_agent_id helper."""

    def test_returns_entry_point_when_set(self):
        """Uses config.entry_point when available."""
        config = _fake_config(entry_point="supervisor")
        assert _target_agent_id(config) == "supervisor"

    def test_falls_back_to_first_agent(self):
        """Falls back to the first agent when no entry_point."""
        config = _fake_config(entry_point=None)
        assert _target_agent_id(config) == "agent_a"

    def test_raises_on_empty_agents(self):
        """Raises ValueError when config has no agents."""
        config = _fake_config(agents=[])
        with pytest.raises(ValueError, match="no agents defined"):
            _target_agent_id(config)


# =========================================================================
# _build_result_dict
# =========================================================================


class TestBuildResultDict:
    """Tests for _build_result_dict."""

    def test_basic_structure(self):
        """Result dict has all expected top-level keys."""
        ip = _fake_payload()
        config = _fake_config()
        attack_result = _fake_attack_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x: 1")

            result = _build_result_dict(
                ip=ip,
                config=config,
                yaml_path="c.yaml",
                phase="pre_input",
                attack_result=attack_result,
                stub_mode=True,
                tier3_rows=None,
                attack_suite="injection",
                repo_root=root,
            )

        assert result["payload_id"] == "p1"
        assert result["injection_type"] == "prompt_injection"
        assert result["severity"] == "high"
        assert result["mas_id"] == "test-mas"
        assert result["injection_phase"] == "pre_input"
        assert result["attack_suite"] == "injection"

    def test_execution_duration_ms(self):
        """Duration is computed from injected_at and completed_at."""
        now = datetime.datetime.now(datetime.timezone.utc)
        later = now + datetime.timedelta(milliseconds=500)
        attack_result = _fake_attack_result(injected_at=now, completed_at=later)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="pre_input",
                attack_result=attack_result,
                stub_mode=False,
                tier3_rows=None,
                attack_suite="s",
                repo_root=root,
            )

        assert abs(result["execution"]["duration_ms"] - 500.0) < 1.0

    def test_zero_duration_when_no_completed_at(self):
        """Duration is 0 when completed_at is None."""
        attack_result = _fake_attack_result(completed_at=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="p",
                attack_result=attack_result,
                stub_mode=True,
                tier3_rows=None,
                attack_suite="s",
                repo_root=root,
            )

        assert result["execution"]["duration_ms"] == 0.0

    def test_tier3_best_score_selected(self):
        """Best tier3 row by score is selected."""
        rows = [
            SimpleNamespace(score=0.3, confidence="low", reasoning="r1"),
            SimpleNamespace(score=0.9, confidence="high", reasoning="r2"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="p",
                attack_result=_fake_attack_result(),
                stub_mode=False,
                tier3_rows=rows,
                attack_suite="s",
                repo_root=root,
            )

        meta = result["run_metadata"]
        assert meta["tier3_score"] == "0.9"
        assert meta["tier3_confidence"] == "high"
        assert meta["tier3_reasoning"] == "r2"

    def test_stub_mode_semantic_tier(self):
        """Stub mode sets semantic_tier to 'skipped'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="p",
                attack_result=_fake_attack_result(),
                stub_mode=True,
                tier3_rows=None,
                attack_suite="s",
                repo_root=root,
            )

        assert result["run_metadata"]["semantic_tier"] == "skipped"

    def test_non_stub_mode_semantic_tier(self):
        """Non-stub mode sets semantic_tier to 'evaluated'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="p",
                attack_result=_fake_attack_result(),
                stub_mode=False,
                tier3_rows=None,
                attack_suite="s",
                repo_root=root,
            )

        assert result["run_metadata"]["semantic_tier"] == "evaluated"

    def test_resistant_agents_sorted(self):
        """Resistant agents are sorted in the result."""
        attack_result = _fake_attack_result(
            resistant_agents=["z_agent", "a_agent", "m_agent"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x")

            result = _build_result_dict(
                ip=_fake_payload(),
                config=_fake_config(),
                yaml_path="c.yaml",
                phase="p",
                attack_result=attack_result,
                stub_mode=True,
                tier3_rows=None,
                attack_suite="s",
                repo_root=root,
            )

        assert result["resistant_agents"] == [
            "a_agent",
            "m_agent",
            "z_agent",
        ]


# =========================================================================
# _write_result / _write_csv
# =========================================================================


def test_write_result_writes_json_file():
    """Writes a valid JSON file under mas_id directory."""
    result_dict = {
        "mas_id": "test-mas",
        "payload_id": "p1",
        "injection_phase": "pre_input",
        "key": "value",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _write_result(result_dict, Path(tmpdir))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["key"] == "value"
        assert "test-mas" in str(out)


def test_write_csv_writes_csv_with_header():
    """Writes a CSV file with proper headers and rows."""
    rows = [
        {
            "payload_id": "p1",
            "injection_type": "jailbreak",
            "severity": "high",
            "stub_mode": True,
            "mas_id": "m",
            "phase": "pre_input",
            "tier1_pass": "true",
            "tier2_influenced": "[]",
            "tier2_resistant": "[]",
            "tier3_score": "",
            "tier3_confidence": "",
            "tier3_reasoning": "",
            "attack_suite": "jailbreak",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_csv(rows, Path(tmpdir), "results.csv")
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "payload_id" in content
        assert "p1" in content


# =========================================================================
# _run_config (mocked end-to-end)
# =========================================================================


class TestRunConfig:
    """Tests for _run_config orchestration loop."""

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_skips_missing_config(
        self, mock_load, _mock_injector, _mock_detector, _mock_logger
    ):
        """Returns empty list when YAML path does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _run_config(
                yaml_path="nonexistent.yaml",
                payloads=[],
                phases=["pre_input"],
                stub_mode=True,
                semantic_evaluator=None,
                baseline_results_dir=None,
                results_dir=Path(tmpdir),
                repo_root=Path(tmpdir),
                attack_suite="injection",
                attack_type="prompt_injection",
            )
        assert not rows
        mock_load.assert_not_called()

    @patch(f"{_MODULE}._write_result")
    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_produces_matrix_rows(
        self,
        mock_load,
        mock_injector_cls,
        _mock_detector,
        _mock_logger,
        mock_write,
    ):
        """Produces one matrix row per payload x phase."""
        config = _fake_config()
        mock_load.return_value = config

        mock_injector = MagicMock()
        mock_injector.__enter__ = MagicMock(return_value=mock_injector)
        mock_injector.__exit__ = MagicMock(return_value=False)
        mock_injector.inject_attack.return_value = _fake_attack_result()
        mock_injector_cls.return_value = mock_injector

        mock_write.return_value = Path("/fake/out.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_file = root / "test.yaml"
            yaml_file.write_text("x: 1")

            rows = _run_config(
                yaml_path="test.yaml",
                payloads=[_fake_payload("p1"), _fake_payload("p2")],
                phases=["pre_input"],
                stub_mode=True,
                semantic_evaluator=None,
                baseline_results_dir=None,
                results_dir=root / "results",
                repo_root=root,
                attack_suite="injection",
                attack_type="prompt_injection",
            )

        assert len(rows) == 2
        assert rows[0]["payload_id"] == "p1"
        assert rows[1]["payload_id"] == "p2"

    @patch(f"{_MODULE}._write_result")
    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_handles_inject_attack_error(
        self,
        mock_load,
        mock_injector_cls,
        _mock_detector,
        _mock_logger,
        _mock_write,
    ):
        """Records error row when inject_attack raises."""
        config = _fake_config()
        mock_load.return_value = config

        mock_injector = MagicMock()
        mock_injector.__enter__ = MagicMock(return_value=mock_injector)
        mock_injector.__exit__ = MagicMock(return_value=False)
        mock_injector.inject_attack.side_effect = RuntimeError("boom")
        mock_injector_cls.return_value = mock_injector

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "t.yaml").write_text("x")

            rows = _run_config(
                yaml_path="t.yaml",
                payloads=[_fake_payload()],
                phases=["pre_input"],
                stub_mode=True,
                semantic_evaluator=None,
                baseline_results_dir=None,
                results_dir=root / "results",
                repo_root=root,
                attack_suite="injection",
                attack_type="prompt_injection",
            )

        assert len(rows) == 1
        assert rows[0]["tier1_pass"] == "false"
        assert "boom" in rows[0]["tier3_reasoning"]


# =========================================================================
# run_suite (public API)
# =========================================================================


class TestRunSuite:
    """Tests for run_suite orchestration."""

    @patch(f"{_MODULE}._print_summary")
    @patch(f"{_MODULE}._write_csv")
    @patch(f"{_MODULE}._run_config")
    def test_aggregates_rows_across_configs(
        self, mock_run_config, mock_write_csv, _mock_print
    ):
        """run_suite aggregates rows from multiple configs."""
        mock_run_config.side_effect = [
            [{"tier1_pass": "true"}],
            [{"tier1_pass": "false"}],
        ]
        mock_write_csv.return_value = Path("/fake.csv")

        with pytest.raises(SystemExit) as exc_info:
            run_suite(
                payloads=[_fake_payload()],
                attack_suite="test",
                attack_type="prompt_injection",
                csv_filename="out.csv",
                suite_name="Test Suite",
                results_dir=Path("/tmp/results"),
                repo_root=Path("/tmp"),
                config_paths=["a.yaml", "b.yaml"],
                phases=["pre_input"],
                stub=True,
            )

        # 1 pass out of 2 total -> exit(1)
        assert exc_info.value.code == 1
        assert mock_run_config.call_count == 2

    @patch(f"{_MODULE}._run_config", return_value=[])
    def test_no_results_exits_zero(self, _mock_run_config):
        """run_suite exits 0 when there are no results."""
        with pytest.raises(SystemExit) as exc_info:
            run_suite(
                payloads=[],
                attack_suite="test",
                attack_type="prompt_injection",
                csv_filename="out.csv",
                suite_name="Test",
                results_dir=Path("/tmp/results"),
                repo_root=Path("/tmp"),
                config_paths=["a.yaml"],
                phases=["pre_input"],
                stub=True,
            )

        assert exc_info.value.code == 0

    @patch(f"{_MODULE}._print_summary")
    @patch(f"{_MODULE}._write_csv")
    @patch(f"{_MODULE}._run_config")
    def test_all_pass_exits_zero(self, mock_run_config, mock_write_csv, _mock_print):
        """run_suite exits 0 when all tests pass."""
        mock_run_config.return_value = [
            {"tier1_pass": "true"},
            {"tier1_pass": "true"},
        ]
        mock_write_csv.return_value = Path("/fake.csv")

        with pytest.raises(SystemExit) as exc_info:
            run_suite(
                payloads=[_fake_payload()],
                attack_suite="test",
                attack_type="prompt_injection",
                csv_filename="out.csv",
                suite_name="Test",
                results_dir=Path("/tmp/results"),
                repo_root=Path("/tmp"),
                config_paths=["a.yaml"],
                phases=["pre_input"],
                stub=True,
            )

        assert exc_info.value.code == 0
