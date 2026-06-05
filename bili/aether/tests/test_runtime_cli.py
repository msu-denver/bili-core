"""Tests for the AETHER runtime CLI (bili/aether/runtime/cli.py).

Covers _ensure_bili_stub, _build_input_data, and the main() entry point
across its default, cross-model, and checkpoint-persistence branches. All
heavy dependencies (loader, MASExecutor) are mocked.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from bili.aether.runtime import cli as cli_module
from bili.aether.runtime.cli import (
    _build_input_data,
    _build_parser,
    _ensure_bili_stub,
    main,
)


def _result(success=True, output="formatted output"):
    """Build a fake run result exposing success and get_formatted_output."""
    res = MagicMock()
    res.success = success
    res.get_formatted_output.return_value = output
    return res


def _patched_main(argv):
    """Return a context-manager set patching argv, loader, and MASExecutor."""
    return (
        patch.object(sys, "argv", argv),
        patch("bili.aether.config.loader.load_mas_from_yaml"),
        patch("bili.aether.runtime.executor.MASExecutor"),
    )


class TestEnsureBiliStub:
    """_ensure_bili_stub inserts the project root and stubs bili."""

    def test_inserts_root_and_creates_stub(self):
        """A bili module without __path__ is replaced by a package stub."""
        expected_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(cli_module.__file__)))
            )
        )
        with patch.object(sys, "path", ["/unrelated"]), patch.dict(
            sys.modules, {"bili": types.ModuleType("bili")}
        ):
            _ensure_bili_stub()
            assert expected_root in sys.path
            assert sys.modules["bili"].__path__ == [os.path.join(expected_root, "bili")]


class TestBuildInputData:
    """_build_input_data turns CLI args into MASExecutor input state."""

    def test_input_text_wraps_human_message(self):
        """--input text is wrapped as a single HumanMessage."""
        args = _build_parser().parse_args(["config.yaml", "--input", "hello"])
        data = _build_input_data(args)
        assert data["messages"][0].content == "hello"

    def test_input_file_is_read(self, tmp_path):
        """--input-file content is read from disk and wrapped."""
        path = tmp_path / "in.txt"
        path.write_text("from file")
        args = _build_parser().parse_args(["config.yaml", "--input-file", str(path)])
        data = _build_input_data(args)
        assert data["messages"][0].content == "from file"

    def test_no_input_returns_empty_dict(self):
        """With no input text or file, an empty state dict is returned."""
        args = _build_parser().parse_args(["config.yaml"])
        assert _build_input_data(args) == {}


class TestMain:
    """main() dispatches across the run modes and exits with the right code."""

    def test_default_run_success_exits_zero(self):
        """A successful default run exits with code 0."""
        p_argv, p_load, p_exec = _patched_main(
            ["cli.py", "config.yaml", "--input", "hi"]
        )
        with p_argv, p_load, p_exec as exec_cls:
            executor = exec_cls.return_value
            executor.run.return_value = _result(success=True)
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0
        executor.run.assert_called_once()

    def test_default_run_failure_exits_one(self):
        """A failed default run exits with code 1."""
        p_argv, p_load, p_exec = _patched_main(
            ["cli.py", "config.yaml", "--input", "hi", "--no-save"]
        )
        with p_argv, p_load, p_exec as exec_cls:
            exec_cls.return_value.run.return_value = _result(success=False)
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        # --no-save disables result saving.
        assert exec_cls.return_value.run.call_args.kwargs["save_results"] is False

    def test_cross_model_success(self):
        """A passing cross-model test prints both results and exits 0."""
        p_argv, p_load, p_exec = _patched_main(
            [
                "cli.py",
                "config.yaml",
                "--input",
                "hi",
                "--test-cross-model",
                "--source-model",
                "gpt-4",
                "--target-model",
                "claude",
            ]
        )
        with p_argv, p_load, p_exec as exec_cls:
            exec_cls.return_value.run_cross_model_test.return_value = (
                _result(success=True),
                _result(success=True),
            )
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_cross_model_missing_models_exits_one(self):
        """Cross-model without both model names exits 1 before running."""
        p_argv, p_load, p_exec = _patched_main(
            ["cli.py", "config.yaml", "--input", "hi", "--test-cross-model"]
        )
        with p_argv, p_load, p_exec as exec_cls:
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        exec_cls.return_value.run_cross_model_test.assert_not_called()

    def test_checkpoint_persistence_run(self):
        """The checkpoint-persistence test runs both phases and exits on success."""
        p_argv, p_load, p_exec = _patched_main(
            ["cli.py", "config.yaml", "--input", "hi", "--test-checkpoint"]
        )
        with p_argv, p_load, p_exec as exec_cls:
            exec_cls.return_value.run_with_checkpoint_persistence.return_value = (
                _result(success=True),
                _result(success=True),
            )
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0
