"""Tests for the cross-model transferability suite runner.

Covers main() CLI parsing, model matrix building,
provider_family derivation, helper functions, and summary printing.
All external dependencies (LLM calls, file I/O, YAML loading) are mocked.
"""

import csv
import json
from unittest.mock import MagicMock, patch

import pytest

# -----------------------------------------------------------------------
# Lazy-import helpers — the module performs sys.path manipulation and
# heavyweight imports at module level; we gate those behind patches.
# -----------------------------------------------------------------------


def _import_module():
    """Import the cross_model runner module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.tests.cross_model import run_cross_model_suite as mod

    return mod


# =========================================================================
# _provider_family
# =========================================================================


class TestProviderFamily:
    """Tests for _provider_family helper."""

    def test_anthropic_bedrock(self):
        """Anthropic Bedrock model IDs map correctly."""
        mod = _import_module()
        result = mod._provider_family("us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert result == "anthropic_bedrock"

    def test_amazon_bedrock(self):
        """Amazon Bedrock model IDs map correctly."""
        mod = _import_module()
        assert mod._provider_family("amazon.nova-pro-v1:0") == ("amazon_bedrock")

    def test_google_vertex(self):
        """Google Vertex model IDs map correctly."""
        mod = _import_module()
        assert mod._provider_family("gemini-2.0-flash") == ("google_vertex")

    def test_openai(self):
        """OpenAI model IDs map correctly."""
        mod = _import_module()
        assert mod._provider_family("gpt-4o") == "openai"
        assert mod._provider_family("o3-mini") == "openai"

    def test_anthropic_direct(self):
        """Direct Anthropic model IDs map correctly."""
        mod = _import_module()
        assert mod._provider_family("claude-3-opus") == ("anthropic_direct")

    def test_stub(self):
        """None model_id returns stub."""
        mod = _import_module()
        assert mod._provider_family(None) == "stub"

    def test_unknown(self):
        """Unrecognised prefix returns unknown."""
        mod = _import_module()
        assert mod._provider_family("something-else") == "unknown"


# =========================================================================
# _write_csv
# =========================================================================


class TestWriteCsv:
    """Tests for _write_csv helper."""

    def test_writes_valid_csv(self, tmp_path):
        """Writes a CSV with correct headers and data."""
        mod = _import_module()
        rows = [{col: f"val_{i}" for i, col in enumerate(mod._CSV_COLUMNS)}]
        csv_path = mod._write_csv(rows, tmp_path)
        assert csv_path.exists()
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            result_rows = list(reader)
        assert len(result_rows) == 1
        assert set(reader.fieldnames) == set(mod._CSV_COLUMNS)


# =========================================================================
# _write_result
# =========================================================================


class TestWriteResult:
    """Tests for _write_result helper."""

    def test_creates_nested_directories_and_json(self, tmp_path):
        """Creates model-specific subdirectory and writes JSON."""
        mod = _import_module()
        result_dict = {
            "mas_id": "test_mas",
            "model_id": "gemini-2.0-flash",
            "payload_id": "pi_001",
            "injection_phase": "pre_execution",
        }
        out_path = mod._write_result(result_dict, tmp_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
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
                "tier2_influenced": '["agent_a"]',
                "payload_id": "pi_001",
                "phase": "pre_execution",
                "provider_family": "anthropic_bedrock",
            },
            {
                "skipped": "true",
                "tier1_pass": "",
                "tier2_influenced": "",
                "payload_id": "pi_002",
                "phase": "pre_execution",
                "provider_family": "google_vertex",
            },
        ]
        mod._print_summary(rows)
        captured = capsys.readouterr()
        assert "Cross-Model" in captured.out

    def test_counts_transfers_across_families(self, capsys):
        """Counts payload/phase pairs that transfer across families."""
        mod = _import_module()
        rows = [
            {
                "skipped": "false",
                "tier1_pass": "true",
                "tier2_influenced": '["a"]',
                "payload_id": "pi_001",
                "phase": "pre",
                "provider_family": "anthropic_bedrock",
            },
            {
                "skipped": "false",
                "tier1_pass": "true",
                "tier2_influenced": '["a"]',
                "payload_id": "pi_001",
                "phase": "pre",
                "provider_family": "google_vertex",
            },
        ]
        mod._print_summary(rows)
        captured = capsys.readouterr()
        assert "1 payload/phase pairs" in captured.out


# =========================================================================
# _load_baseline
# =========================================================================


class TestLoadBaseline:
    """Tests for _load_baseline helper."""

    def test_returns_none_when_dir_is_none(self):
        """Returns None when no baseline dir provided."""
        mod = _import_module()
        assert mod._load_baseline(None, "mas1") is None

    def test_returns_none_when_dir_missing(self, tmp_path):
        """Returns None when MAS subdirectory does not exist."""
        mod = _import_module()
        assert mod._load_baseline(tmp_path, "nonexistent") is None

    def test_returns_none_when_no_json_files(self, tmp_path):
        """Returns None when directory has no JSON files."""
        mod = _import_module()
        mas_dir = tmp_path / "mas1"
        mas_dir.mkdir()
        assert mod._load_baseline(tmp_path, "mas1") is None

    def test_loads_first_json(self, tmp_path):
        """Loads and parses the first JSON file found."""
        mod = _import_module()
        mas_dir = tmp_path / "mas1"
        mas_dir.mkdir()
        data = {"key": "value"}
        (mas_dir / "result.json").write_text(json.dumps(data))
        result = mod._load_baseline(tmp_path, "mas1")
        assert result == data


# =========================================================================
# _patch_config_model
# =========================================================================


class TestPatchConfigModel:
    """Tests for _patch_config_model helper."""

    def test_patches_all_agents(self):
        """All agents get their model_name replaced."""
        mod = _import_module()
        agent1 = MagicMock()
        agent1.model_copy.return_value = MagicMock(model_name="new_model")
        agent2 = MagicMock()
        agent2.model_copy.return_value = MagicMock(model_name="new_model")
        config = MagicMock()
        config.agents = [agent1, agent2]
        config.model_copy.return_value = config

        mod._patch_config_model(config, "new_model")

        agent1.model_copy.assert_called_once_with(update={"model_name": "new_model"})
        agent2.model_copy.assert_called_once_with(update={"model_name": "new_model"})


# =========================================================================
# main() — CLI parsing
# =========================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    @patch(
        "bili.aegis.tests.cross_model.run_cross_model_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.tests.cross_model.run_cross_model_suite" "._run_config_for_model"
    )
    def test_stub_mode_uses_single_model(self, mock_run, mock_args):
        """--stub replaces model matrix with single None entry."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=["fake.yaml"],
            models=None,
            payloads=None,
            phases=["pre_execution"],
            baseline_results=None,
            log_level="WARNING",
        )
        mock_run.return_value = [
            {
                "skipped": "false",
                "tier1_pass": "true",
                **{
                    c: ""
                    for c in mod._CSV_COLUMNS
                    if c not in ("skipped", "tier1_pass")
                },
            }
        ]
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model_id"] is None

    @patch(
        "bili.aegis.tests.cross_model.run_cross_model_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.tests.cross_model.run_cross_model_suite" ".INJECTION_PAYLOADS",
        [],
    )
    def test_no_matching_payloads_exits(self, mock_args):
        """Exits with code 1 when no payloads match filter."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=[],
            models=None,
            payloads=["nonexistent_payload"],
            phases=["pre_execution"],
            baseline_results=None,
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1

    @patch(
        "bili.aegis.tests.cross_model.run_cross_model_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    def test_no_matching_models_exits(self, mock_args):
        """Exits with code 1 when no models match filter."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=False,
            configs=[],
            models=["nonexistent_model"],
            payloads=None,
            phases=["pre_execution"],
            baseline_results=None,
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1
