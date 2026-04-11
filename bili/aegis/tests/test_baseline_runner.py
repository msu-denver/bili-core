"""Tests for the baseline runner.

Covers main() CLI parsing, _run_one execution, _write_result,
_print_summary, and error handling paths.
All external dependencies (LLM calls, file I/O, YAML loading) are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


def _import_module():
    """Import the baseline runner module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.tests.baseline import run_baseline as mod

    return mod


# =========================================================================
# _write_result
# =========================================================================


class TestWriteResult:
    """Tests for _write_result helper."""

    def test_creates_directory_and_writes_json(self, tmp_path):
        """Creates mas_id subdirectory and writes JSON."""
        mod = _import_module()
        result_dict = {
            "mas_id": "test_mas",
            "prompt_id": "benign_001",
        }
        with patch.object(mod, "_RESULTS_DIR", tmp_path):
            out = mod._write_result(result_dict)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["prompt_id"] == "benign_001"


# =========================================================================
# _print_summary
# =========================================================================


class TestPrintSummary:
    """Tests for _print_summary helper."""

    def test_prints_without_error(self, capsys):
        """Summary prints success/fail grid without raising."""
        mod = _import_module()
        rows = [
            {
                "mas_id": "c1",
                "prompt_id": "p1",
                "execution": {"success": True},
            },
            {
                "mas_id": "c1",
                "prompt_id": "p2",
                "execution": {"success": False},
            },
        ]
        mod._print_summary(rows, ["c1"], ["p1", "p2"])
        captured = capsys.readouterr()
        assert "Baseline Run Summary" in captured.out
        assert "1/2" in captured.out


# =========================================================================
# _run_one
# =========================================================================


class TestRunOne:
    """Tests for _run_one helper."""

    @patch("bili.aegis.tests.baseline.run_baseline.MASExecutor")
    def test_returns_result_dict(self, mock_executor_cls):
        """Returns a well-formed result dictionary."""
        mod = _import_module()
        agent_result = MagicMock()
        agent_result.agent_id = "agent_1"
        agent_result.output = {
            "raw": "response text",
            "parsed": None,
        }

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.total_execution_time_ms = 42.0
        exec_result.agent_results = [agent_result]
        exec_result.total_messages = 3

        executor = MagicMock()
        executor.run.return_value = exec_result
        mock_executor_cls.return_value = executor

        config = MagicMock()
        config.mas_id = "test_mas"
        config.agents = []

        prompt = MagicMock()
        prompt.prompt_id = "benign_001"
        prompt.category = "benign"
        prompt.text = "Hello"

        with patch.object(
            mod,
            "_config_fingerprint",
            return_value={"yaml_hash": "abc"},
        ):
            result = mod._run_one(config, "fake.yaml", prompt, stub_mode=True)

        assert result["prompt_id"] == "benign_001"
        assert result["execution"]["success"] is True
        assert "agent_1" in result["agent_outputs"]


# =========================================================================
# main() — CLI parsing
# =========================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    @patch(
        "bili.aegis.tests.baseline.run_baseline" ".argparse.ArgumentParser.parse_args"
    )
    @patch("bili.aegis.tests.baseline.run_baseline.load_mas_from_yaml")
    @patch("bili.aegis.tests.baseline.run_baseline._run_one")
    @patch("bili.aegis.tests.baseline.run_baseline._write_result")
    def test_stub_mode_runs_all_configs(
        self,
        mock_write,
        mock_run_one,
        mock_load,
        mock_args,
        tmp_path,
    ):
        """--stub sets model_name=None and runs all configs."""
        mod = _import_module()

        config = MagicMock()
        config.mas_id = "test_mas"
        agent = MagicMock()
        config.agents = [agent]
        mock_load.return_value = config

        mock_run_one.return_value = {
            "prompt_id": "p1",
            "mas_id": "test_mas",
            "execution": {"success": True, "duration_ms": 10.0},
        }
        mock_write.return_value = tmp_path / "result.json"

        fake_yaml = tmp_path / "test.yaml"
        fake_yaml.touch()

        mock_args.return_value = MagicMock(
            stub=True,
            configs=[str(fake_yaml)],
            prompts=None,
            log_level="WARNING",
        )

        with patch.object(mod, "_REPO_ROOT", tmp_path):
            with patch.object(
                mod,
                "BASELINE_PROMPTS",
                [
                    MagicMock(
                        prompt_id="p1",
                        category="benign",
                        text="Hi",
                    )
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()

        assert exc_info.value.code == 0
        assert agent.model_name is None

    @patch(
        "bili.aegis.tests.baseline.run_baseline" ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.tests.baseline.run_baseline.BASELINE_PROMPTS",
        [],
    )
    def test_no_matching_prompts_exits(self, mock_args):
        """Exits with code 1 when no prompts match filter."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=[],
            prompts=["nonexistent"],
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1

    @patch(
        "bili.aegis.tests.baseline.run_baseline" ".argparse.ArgumentParser.parse_args"
    )
    @patch("bili.aegis.tests.baseline.run_baseline.load_mas_from_yaml")
    @patch(
        "bili.aegis.tests.baseline.run_baseline._run_one",
        side_effect=RuntimeError("boom"),
    )
    @patch("bili.aegis.tests.baseline.run_baseline._write_result")
    def test_run_one_error_writes_failed_result(
        self, mock_write, mock_run, mock_load, mock_args, tmp_path
    ):
        """Errors in _run_one produce a failed result dict."""
        mod = _import_module()

        config = MagicMock()
        config.mas_id = "err_mas"
        config.agents = []
        mock_load.return_value = config

        mock_write.return_value = tmp_path / "failed.json"

        fake_yaml = tmp_path / "test.yaml"
        fake_yaml.touch()

        mock_args.return_value = MagicMock(
            stub=True,
            configs=[str(fake_yaml)],
            prompts=None,
            log_level="WARNING",
        )

        with patch.object(mod, "_REPO_ROOT", tmp_path):
            with patch.object(
                mod,
                "BASELINE_PROMPTS",
                [
                    MagicMock(
                        prompt_id="p1",
                        category="benign",
                        text="Hi",
                    )
                ],
            ):
                with patch.object(
                    mod,
                    "_config_fingerprint",
                    return_value={"yaml_hash": "x"},
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        mod.main()

        # Failed result written — exit code 1 because success=False
        assert exc_info.value.code == 1
        # _write_result is called once for the failed result
        assert mock_write.call_count >= 1
