"""Tests for the persistence suite runner.

Covers main() CLI parsing, _checkpointer_is_persistent logic,
helper functions, and summary printing.
All external dependencies are mocked.
"""

# Tests exercise private helpers (_CSV_COLUMNS, _run_persistence_config, etc.)
# pylint: disable=protected-access

import csv
import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE = "bili.aegis.suites.persistence.run_persistence_suite"


def _import_module():
    """Import the persistence suite runner module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.suites.persistence import run_persistence_suite as mod

    return mod


def _fake_payload(payload_id="pe_001"):
    """Build a minimal persistence payload namespace."""
    return SimpleNamespace(
        payload_id=payload_id,
        injection_type="persistence",
        severity="high",
        payload="poisoned context",
    )


def _fake_attack_result(success=True, influenced=None, resistant=None):
    """Build a minimal attack result namespace with a 1s duration."""
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    return SimpleNamespace(
        success=success,
        target_agent_id="agent_a",
        propagation_path=["agent_a"],
        influenced_agents=influenced if influenced is not None else [],
        resistant_agents=resistant if resistant is not None else ["agent_b"],
        injected_at=now,
        completed_at=now + datetime.timedelta(seconds=1),
    )


def _fake_config():
    """Build a minimal MAS config namespace."""
    agent = SimpleNamespace(agent_id="agent_a", model_name="m", temperature=0.2)
    return SimpleNamespace(
        mas_id="test_mas",
        entry_point=None,
        agents=[agent],
        checkpoint_enabled=True,
        checkpoint_config={"type": "postgres"},
    )


def _patched_injector(attack_result=None, side_effect=None):
    """Build a context-manager MagicMock that mimics AttackInjector."""
    injector = MagicMock()
    injector.__enter__ = MagicMock(return_value=injector)
    injector.__exit__ = MagicMock(return_value=False)
    if side_effect is not None:
        injector.inject_attack.side_effect = side_effect
    else:
        injector.inject_attack.return_value = (
            attack_result if attack_result is not None else _fake_attack_result()
        )
    return injector


# =========================================================================
# _checkpointer_is_persistent
# =========================================================================


class TestCheckpointerIsPersistent:
    """Tests for _checkpointer_is_persistent."""

    def test_returns_false_when_checkpoint_disabled(self):
        """Returns (False, reason) when checkpoint_enabled=False."""
        mod = _import_module()
        config = MagicMock()
        config.checkpoint_enabled = False
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False
        assert "checkpoint_enabled" in reason

    def test_returns_false_for_memory_type(self):
        """Returns (False, reason) for memory checkpoint type."""
        mod = _import_module()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": "memory"}
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False
        assert "MemorySaver" in reason

    def test_returns_false_for_none_type(self):
        """Returns (False, reason) when type is None."""
        mod = _import_module()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": None}
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False

    @patch(
        "bili.aether.integration.checkpointer_factory"
        ".create_checkpointer_from_config",
        side_effect=RuntimeError("no db"),
    )
    def test_returns_false_when_factory_fails(self, _mock):
        """Returns (False, reason) when factory raises."""
        mod = _import_module()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": "postgres"}
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False
        assert "factory" in reason.lower() or "failed" in reason

    @patch(
        "bili.aether.integration.checkpointer_factory"
        ".create_checkpointer_from_config"
    )
    def test_returns_true_for_real_checkpointer(self, mock_factory):
        """Returns (True, '') for a non-MemorySaver checkpointer."""
        mod = _import_module()
        mock_factory.return_value = MagicMock()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": "postgres"}
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is True
        assert reason == ""

    @patch(
        "bili.aether.integration.checkpointer_factory"
        ".create_checkpointer_from_config"
    )
    def test_returns_false_when_factory_falls_back_to_memorysaver(self, mock_factory):
        """Returns (False, reason) when the factory hands back a MemorySaver."""
        # pylint: disable=import-outside-toplevel
        from langgraph.checkpoint.memory import MemorySaver

        mod = _import_module()
        mock_factory.return_value = MemorySaver()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": "postgres"}
        ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False
        assert "MemorySaver at runtime" in reason

    def test_returns_false_when_langgraph_missing(self):
        """Returns (False, reason) when langgraph MemorySaver import fails."""
        import builtins  # pylint: disable=import-outside-toplevel

        mod = _import_module()
        config = MagicMock()
        config.checkpoint_enabled = True
        config.checkpoint_config = {"type": "postgres"}

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "langgraph.checkpoint.memory":
                raise ImportError("no langgraph")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_fake_import):
            ok, reason = mod._checkpointer_is_persistent(config)
        assert ok is False
        assert "langgraph not available" in reason


# =========================================================================
# _write_csv
# =========================================================================


class TestWriteCsv:
    """Tests for _write_csv helper."""

    def test_writes_correct_csv(self, tmp_path):
        """Writes CSV with all expected columns."""
        mod = _import_module()
        rows = [{col: f"v{i}" for i, col in enumerate(mod._CSV_COLUMNS)}]
        path = mod._write_csv(rows, tmp_path, "persistence_results_matrix.csv")
        assert path.exists()
        assert path.name == "persistence_results_matrix.csv"
        with path.open() as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
        assert len(data) == 1


# =========================================================================
# _write_result
# =========================================================================


class TestWriteResult:
    """Tests for _write_result helper."""

    def test_creates_directory_and_json(self, tmp_path):
        """Creates mas_id subdirectory and writes JSON."""
        mod = _import_module()
        result_dict = {
            "mas_id": "test_mas",
            "payload_id": "pe_001",
            "injection_phase": "checkpoint_injection",
        }
        out = mod._write_result(result_dict, tmp_path)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["mas_id"] == "test_mas"


# =========================================================================
# _print_summary
# =========================================================================


class TestPrintSummary:
    """Tests for _print_summary helper."""

    def test_prints_without_error(self, capsys):
        """Summary prints without raising."""
        mod = _import_module()
        rows = [
            {
                "skipped": "false",
                "tier1_pass": "true",
                "tier2_influenced": "[]",
            },
            {
                "skipped": "true",
                "tier1_pass": "",
                "tier2_influenced": "",
            },
        ]
        mod._print_summary(rows)
        captured = capsys.readouterr()
        assert "Persistence Suite Summary" in captured.out

    def test_all_skipped_note(self, capsys):
        """Prints extra note when all configs are skipped."""
        mod = _import_module()
        rows = [
            {
                "skipped": "true",
                "tier1_pass": "",
                "tier2_influenced": "",
            }
        ]
        mod._print_summary(rows)
        captured = capsys.readouterr()
        assert "All configs skipped" in captured.out


# =========================================================================
# _load_baseline
# =========================================================================


class TestLoadBaseline:
    """Tests for _load_baseline helper."""

    def test_returns_none_when_dir_none(self):
        """Returns None when no baseline dir provided."""
        mod = _import_module()
        assert mod._load_baseline(None, "mas1") is None

    def test_returns_none_when_subdir_missing(self, tmp_path):
        """Returns None when MAS subdirectory missing."""
        mod = _import_module()
        assert mod._load_baseline(tmp_path, "missing") is None

    def test_loads_first_json(self, tmp_path):
        """Loads and returns the first JSON file content."""
        mod = _import_module()
        mas_dir = tmp_path / "mas1"
        mas_dir.mkdir()
        data = {"baseline": True}
        (mas_dir / "b.json").write_text(json.dumps(data))
        result = mod._load_baseline(tmp_path, "mas1")
        assert result == data

    def test_returns_none_when_no_json_files(self, tmp_path):
        """Returns None when the MAS directory has no JSON files."""
        mod = _import_module()
        (tmp_path / "mas1").mkdir()
        assert mod._load_baseline(tmp_path, "mas1") is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        """Returns None and warns when the baseline JSON cannot be parsed."""
        mod = _import_module()
        mas_dir = tmp_path / "mas1"
        mas_dir.mkdir()
        (mas_dir / "bad.json").write_text("{not valid json")
        assert mod._load_baseline(tmp_path, "mas1") is None


# =========================================================================
# _run_persistence_config
# =========================================================================


class TestRunPersistenceConfig:
    """Tests for the per-config persistence runner loop."""

    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_skips_missing_config(self, mock_load, tmp_path):
        """Returns empty rows and None run_dir when the YAML path is missing."""
        mod = _import_module()
        rows, run_dir = mod._run_persistence_config(
            yaml_path="missing.yaml",
            payloads=[_fake_payload()],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )
        assert rows == []
        assert run_dir is None
        mock_load.assert_not_called()

    @patch(f"{_MODULE}._checkpointer_is_persistent")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_skips_non_persistent_backend(self, mock_load, mock_persistent, tmp_path):
        """Produces skip rows (one per payload) when no persistent backend."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_persistent.return_value = (False, "MemorySaver in use")

        (tmp_path / "t.yaml").write_text("x")
        rows, run_dir = mod._run_persistence_config(
            yaml_path="t.yaml",
            payloads=[_fake_payload("pe_001"), _fake_payload("pe_002")],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )
        assert run_dir is None
        assert len(rows) == 2
        assert all(r["skipped"] == "true" for r in rows)
        assert rows[0]["skip_reason"] == "MemorySaver in use"

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}._checkpointer_is_persistent")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_stub_mode_nulls_model_and_writes_result(
        self, mock_load, mock_persistent, mock_injector_cls, _det, _log, tmp_path
    ):
        """Stub mode nulls agent model_name, writes a JSON result, returns a row."""
        mod = _import_module()
        config = _fake_config()
        mock_load.return_value = config
        mock_persistent.return_value = (True, "")
        mock_injector_cls.return_value = _patched_injector(
            _fake_attack_result(influenced=["agent_a"])
        )

        (tmp_path / "t.yaml").write_text("x")
        rows, run_dir = mod._run_persistence_config(
            yaml_path="t.yaml",
            payloads=[_fake_payload("pe_001")],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert config.agents[0].model_name is None
        assert len(rows) == 1
        assert rows[0]["skipped"] == "false"
        assert rows[0]["tier1_pass"] == "true"
        assert rows[0]["tier2_influenced"] == json.dumps(["agent_a"])
        written = list(run_dir.glob("pe_001_*.json"))
        assert written, "expected a per-case JSON result file"
        data = json.loads(written[0].read_text())
        assert data["execution"]["success"] is True
        assert data["execution"]["duration_ms"] == pytest.approx(1000.0)

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}._checkpointer_is_persistent")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_inject_error_records_failure_row(
        self, mock_load, mock_persistent, mock_injector_cls, _det, _log, tmp_path
    ):
        """A raised inject_attack records a tier1-fail row carrying the message."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_persistent.return_value = (True, "")
        mock_injector_cls.return_value = _patched_injector(
            side_effect=RuntimeError("boom")
        )

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_persistence_config(
            yaml_path="t.yaml",
            payloads=[_fake_payload("pe_001")],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert len(rows) == 1
        assert rows[0]["skipped"] == "false"
        assert rows[0]["tier1_pass"] == "false"
        assert rows[0]["tier3_reasoning"] == "boom"

    @patch(f"{_MODULE}._load_baseline")
    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}._checkpointer_is_persistent")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_tier3_evaluation_records_best_score(
        self,
        mock_load,
        mock_persistent,
        mock_injector_cls,
        _det,
        _log,
        mock_baseline,
        tmp_path,
    ):
        """Non-stub run with a baseline records the best Tier-3 score."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_persistent.return_value = (True, "")
        mock_injector_cls.return_value = _patched_injector(_fake_attack_result())
        mock_baseline.return_value = {"baseline": True}

        evaluator = MagicMock()
        evaluator.evaluate.return_value = [
            SimpleNamespace(score=1, confidence="low", reasoning="r1"),
            SimpleNamespace(score=3, confidence="high", reasoning="r3"),
        ]

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_persistence_config(
            yaml_path="t.yaml",
            payloads=[_fake_payload("pe_001")],
            stub_mode=False,
            semantic_evaluator=evaluator,
            baseline_results_dir=tmp_path / "baseline",
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        evaluator.evaluate.assert_called_once()
        assert rows[0]["tier3_score"] == "3"
        assert rows[0]["tier3_confidence"] == "high"
        assert rows[0]["tier3_reasoning"] == "r3"

    @patch(f"{_MODULE}._load_baseline")
    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}._checkpointer_is_persistent")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_tier3_evaluator_error_swallowed(
        self,
        mock_load,
        mock_persistent,
        mock_injector_cls,
        _det,
        _log,
        mock_baseline,
        tmp_path,
    ):
        """A failing evaluator leaves Tier-3 columns empty but records the row."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_persistent.return_value = (True, "")
        mock_injector_cls.return_value = _patched_injector(_fake_attack_result())
        mock_baseline.return_value = {"baseline": True}

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("judge down")

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_persistence_config(
            yaml_path="t.yaml",
            payloads=[_fake_payload("pe_001")],
            stub_mode=False,
            semantic_evaluator=evaluator,
            baseline_results_dir=tmp_path / "baseline",
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert rows[0]["tier3_score"] == ""
        assert rows[0]["tier3_reasoning"] == ""


# =========================================================================
# run_persistence_suite (programmatic API)
# =========================================================================


class TestRunPersistenceSuite:
    """Tests for the non-exiting programmatic entry point."""

    @patch(f"{_MODULE}._write_csv")
    @patch(f"{_MODULE}._run_persistence_config")
    def test_aggregates_and_writes_versioned_csv(
        self, mock_run, mock_write_csv, tmp_path
    ):
        """Aggregates rows across configs and writes a run-versioned CSV."""
        mod = _import_module()
        mock_run.side_effect = [
            (
                [
                    {
                        "skipped": "false",
                        "tier1_pass": "true",
                        "tier2_influenced": "[]",
                    }
                ],
                tmp_path / "run_005",
            ),
            (
                [
                    {
                        "skipped": "false",
                        "tier1_pass": "false",
                        "tier2_influenced": "[]",
                    }
                ],
                tmp_path / "run_006",
            ),
        ]
        mock_write_csv.return_value = tmp_path / "out.csv"

        rows, first_run = mod.run_persistence_suite(
            payloads=[_fake_payload()],
            config_paths=["a.yaml", "b.yaml"],
            stub_mode=False,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path,
            repo_root=tmp_path,
        )

        assert len(rows) == 2
        assert first_run == "run_005"
        assert (
            mock_write_csv.call_args[0][2] == "persistence_results_matrix_run_005.csv"
        )

    @patch(f"{_MODULE}._run_persistence_config", return_value=([], None))
    def test_no_rows_returns_empty(self, _mock_run, tmp_path):
        """With no rows the function returns empty results and no run dir."""
        mod = _import_module()
        rows, first_run = mod.run_persistence_suite(
            payloads=[_fake_payload()],
            config_paths=["a.yaml"],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path,
            repo_root=tmp_path,
        )
        assert rows == []
        assert first_run is None


# =========================================================================
# main() — CLI parsing
# =========================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite" "._run_persistence_config"
    )
    def test_stub_mode_runs_configs(self, mock_run, mock_args):
        """--stub mode calls _run_persistence_config per config."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=["a.yaml", "b.yaml"],
            payloads=None,
            baseline_results=None,
            log_level="WARNING",
        )
        mock_run.return_value = (
            [
                {
                    "skipped": "false",
                    "tier1_pass": "true",
                    **{
                        c: ""
                        for c in mod._CSV_COLUMNS
                        if c not in ("skipped", "tier1_pass")
                    },
                }
            ],
            MagicMock(name="run_001"),
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        assert mock_run.call_count == 2

    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite" ".PERSISTENCE_PAYLOADS",
        [],
    )
    def test_no_matching_payloads_exits(self, mock_args):
        """Exits with code 1 when no payloads match filter."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=[],
            payloads=["nonexistent"],
            baseline_results=None,
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1

    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.suites.persistence.run_persistence_suite" "._run_persistence_config"
    )
    def test_all_skipped_exits_zero(self, mock_run, mock_args):
        """Exits 0 when all rows are skipped (skip is not failure)."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=["a.yaml"],
            payloads=None,
            baseline_results=None,
            log_level="WARNING",
        )
        mock_run.return_value = (
            [
                {
                    "skipped": "true",
                    "tier1_pass": "",
                    **{
                        c: ""
                        for c in mod._CSV_COLUMNS
                        if c not in ("skipped", "tier1_pass")
                    },
                }
            ],
            None,
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_persistence_config")
    def test_missing_baseline_dir_clears_to_none(self, mock_run, mock_args, capsys):
        """A nonexistent baseline dir warns and is passed through as None."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=["a.yaml"],
            payloads=None,
            baseline_results="no/such/baseline/dir",
            log_level="WARNING",
        )
        mock_run.return_value = ([], None)
        with pytest.raises(SystemExit):
            mod.main()
        err = capsys.readouterr().err
        assert "baseline results dir not found" in err
        assert mock_run.call_args[1]["baseline_results_dir"] is None

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_persistence_config")
    def test_non_stub_builds_semantic_evaluator(self, mock_run, mock_args):
        """Non-stub mode constructs a persistence-tuned SemanticEvaluator."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=False,
            configs=["a.yaml"],
            payloads=None,
            baseline_results=None,
            log_level="WARNING",
        )
        mock_run.return_value = ([], None)
        sentinel_evaluator = MagicMock(name="evaluator")
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator",
            return_value=sentinel_evaluator,
        ) as mock_eval_cls:
            with pytest.raises(SystemExit):
                mod.main()
        mock_eval_cls.assert_called_once()
        assert mock_run.call_args[1]["semantic_evaluator"] is sentinel_evaluator

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_persistence_config")
    def test_non_stub_semantic_evaluator_init_error_tolerated(
        self, mock_run, mock_args
    ):
        """A failing SemanticEvaluator init in non-stub mode passes None through."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=False,
            configs=["a.yaml"],
            payloads=None,
            baseline_results=None,
            log_level="WARNING",
        )
        mock_run.return_value = ([], None)
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator",
            side_effect=RuntimeError("no creds"),
        ):
            with pytest.raises(SystemExit):
                mod.main()
        assert mock_run.call_args[1]["semantic_evaluator"] is None
