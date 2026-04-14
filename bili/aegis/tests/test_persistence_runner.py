"""Tests for the persistence suite runner.

Covers main() CLI parsing, _checkpointer_is_persistent logic,
helper functions, and summary printing.
All external dependencies are mocked.
"""

import csv
import json
from unittest.mock import MagicMock, patch

import pytest


def _import_module():
    """Import the persistence suite runner module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.suites.persistence import run_persistence_suite as mod

    return mod


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
