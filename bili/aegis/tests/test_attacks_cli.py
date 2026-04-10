"""Tests for the AEGIS attack CLI entry point.

Covers argument parsing, payload resolution, and output formatting.
All external dependencies (MASExecutor, AttackInjector, etc.) are mocked.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from bili.aegis.attacks.cli import (
    _build_parser,
    _build_payload,
    _format_attack_result,
    _row,
)


class TestBuildParser:
    """Tests for the argument parser construction."""

    def test_required_args_present(self):
        """Parser includes config_file, --agent-id, --attack-type."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "config.yaml",
                "--agent-id",
                "agent_a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "test",
            ]
        )
        assert args.config_file == "config.yaml"
        assert args.agent_id == "agent_a"
        assert args.attack_type == "prompt_injection"

    def test_invalid_attack_type_rejected(self):
        """Invalid attack types cause parse error."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "config.yaml",
                    "--agent-id",
                    "a",
                    "--attack-type",
                    "invalid_type",
                    "--payload",
                    "x",
                ]
            )

    def test_valid_attack_types_accepted(self):
        """All four valid attack types are accepted."""
        parser = _build_parser()
        valid_types = [
            "prompt_injection",
            "memory_poisoning",
            "agent_impersonation",
            "bias_inheritance",
        ]
        for attack_type in valid_types:
            args = parser.parse_args(
                [
                    "c.yaml",
                    "--agent-id",
                    "a",
                    "--attack-type",
                    attack_type,
                    "--payload",
                    "x",
                ]
            )
            assert args.attack_type == attack_type

    def test_phase_defaults_to_pre_execution(self):
        """Phase defaults to pre_execution."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "c.yaml",
                "--agent-id",
                "a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "x",
            ]
        )
        assert args.phase == "pre_execution"

    def test_phase_mid_execution_accepted(self):
        """Mid-execution phase is accepted."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "c.yaml",
                "--agent-id",
                "a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "x",
                "--phase",
                "mid_execution",
            ]
        )
        assert args.phase == "mid_execution"

    def test_no_block_flag(self):
        """--no-block flag is captured."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "c.yaml",
                "--agent-id",
                "a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "x",
                "--no-block",
            ]
        )
        assert args.no_block is True

    def test_no_propagation_flag(self):
        """--no-propagation flag is captured."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "c.yaml",
                "--agent-id",
                "a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "x",
                "--no-propagation",
            ]
        )
        assert args.no_propagation is True


class TestBuildPayload:
    """Tests for payload resolution from args."""

    def test_payload_from_flag(self):
        """Returns payload from --payload flag."""
        args = SimpleNamespace(payload="injected text", payload_file=None)
        assert _build_payload(args) == "injected text"

    def test_payload_from_file(self):
        """Reads payload content from file."""
        args = SimpleNamespace(payload=None, payload_file="payloads/test.txt")
        file_content = "payload from file"
        with patch(
            "builtins.open",
            mock_open(read_data=file_content),
        ):
            result = _build_payload(args)
        assert result == file_content

    def test_no_payload_exits(self):
        """Exits with code 1 when neither flag is provided."""
        args = SimpleNamespace(payload=None, payload_file=None)
        with pytest.raises(SystemExit) as exc_info:
            _build_payload(args)
        assert exc_info.value.code == 1

    def test_payload_flag_takes_precedence(self):
        """--payload flag is preferred over --payload-file."""
        args = SimpleNamespace(payload="direct", payload_file="file.txt")
        assert _build_payload(args) == "direct"


class TestRow:
    """Tests for the _row formatting helper."""

    def test_formats_label_and_value(self):
        """Produces aligned label: value output."""
        result = _row("Status:", "SUCCESS")
        assert "Status:" in result
        assert "SUCCESS" in result


class TestFormatAttackResult:
    """Tests for attack result formatting."""

    def test_success_result_formatting(self):
        """Successful result includes SUCCESS and agent info."""
        result = SimpleNamespace(
            success=True,
            attack_id="atk-001",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error=None,
            completed_at="2026-01-01",
            propagation_path=["agent_a", "agent_b"],
            influenced_agents=["agent_b"],
            resistant_agents=["agent_c"],
        )
        output = _format_attack_result(result, "/tmp/log")
        assert "SUCCESS" in output
        assert "agent_a" in output
        assert "agent_b" in output
        assert "/tmp/log" in output

    def test_failed_result_includes_error(self):
        """Failed result includes FAILED status and error."""
        result = SimpleNamespace(
            success=False,
            attack_id="atk-002",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error="Timeout exceeded",
            completed_at="2026-01-01",
            propagation_path=[],
            influenced_agents=[],
            resistant_agents=[],
        )
        output = _format_attack_result(result, "/tmp/log")
        assert "FAILED" in output
        assert "Timeout exceeded" in output

    def test_async_result_shows_tracking(self):
        """Async result (no completed_at) shows tracking note."""
        result = SimpleNamespace(
            success=True,
            attack_id="atk-003",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error=None,
            completed_at=None,
            propagation_path=[],
            influenced_agents=[],
            resistant_agents=[],
        )
        output = _format_attack_result(result, "/tmp/log")
        assert "async" in output

    def test_empty_propagation_shows_none(self):
        """Empty propagation path shows (none)."""
        result = SimpleNamespace(
            success=True,
            attack_id="atk-004",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error=None,
            completed_at="2026-01-01",
            propagation_path=[],
            influenced_agents=[],
            resistant_agents=[],
        )
        output = _format_attack_result(result, "/tmp/log")
        assert "(none)" in output


class TestMainEntryPoint:
    """Tests for the main() CLI entry point."""

    @patch("bili.aegis.attacks.logger.AttackLogger")
    @patch("bili.aegis.attacks.AttackInjector")
    @patch("bili.aether.runtime.executor.MASExecutor")
    @patch("bili.aether.config.loader.load_mas_from_yaml")
    def test_main_success_exits_zero(
        self,
        mock_load,
        mock_executor_cls,
        mock_injector_cls,
        mock_logger_cls,
    ):
        """Successful attack exits with code 0."""
        mock_load.return_value = MagicMock()
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor

        mock_injector = MagicMock()
        mock_injector.__enter__ = MagicMock(return_value=mock_injector)
        mock_injector.__exit__ = MagicMock(return_value=False)
        mock_injector.inject_attack.return_value = SimpleNamespace(
            success=True,
            attack_id="atk-001",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error=None,
            completed_at="2026-01-01",
            propagation_path=["agent_a"],
            influenced_agents=[],
            resistant_agents=[],
        )
        mock_injector_cls.return_value = mock_injector

        with patch.object(
            sys,
            "argv",
            [
                "cli.py",
                "config.yaml",
                "--agent-id",
                "agent_a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "test payload",
            ],
        ):
            from bili.aegis.attacks.cli import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("bili.aegis.attacks.logger.AttackLogger")
    @patch("bili.aegis.attacks.AttackInjector")
    @patch("bili.aether.runtime.executor.MASExecutor")
    @patch("bili.aether.config.loader.load_mas_from_yaml")
    def test_main_failure_exits_one(
        self,
        mock_load,
        mock_executor_cls,
        mock_injector_cls,
        mock_logger_cls,
    ):
        """Failed attack exits with code 1."""
        mock_load.return_value = MagicMock()
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor

        mock_injector = MagicMock()
        mock_injector.__enter__ = MagicMock(return_value=mock_injector)
        mock_injector.__exit__ = MagicMock(return_value=False)
        mock_injector.inject_attack.return_value = SimpleNamespace(
            success=False,
            attack_id="atk-002",
            mas_id="test-mas",
            target_agent_id="agent_a",
            attack_type="prompt_injection",
            injection_phase="pre_execution",
            error="boom",
            completed_at="2026-01-01",
            propagation_path=[],
            influenced_agents=[],
            resistant_agents=[],
        )
        mock_injector_cls.return_value = mock_injector

        with patch.object(
            sys,
            "argv",
            [
                "cli.py",
                "config.yaml",
                "--agent-id",
                "agent_a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "test",
            ],
        ):
            from bili.aegis.attacks.cli import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("bili.aegis.attacks.logger.AttackLogger")
    @patch("bili.aegis.attacks.AttackInjector")
    @patch("bili.aether.runtime.executor.MASExecutor")
    @patch("bili.aether.config.loader.load_mas_from_yaml")
    def test_main_value_error_exits_one(
        self,
        mock_load,
        mock_executor_cls,
        mock_injector_cls,
        mock_logger_cls,
    ):
        """ValueError from injector exits with code 1."""
        mock_load.return_value = MagicMock()
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor

        mock_injector = MagicMock()
        mock_injector.__enter__ = MagicMock(return_value=mock_injector)
        mock_injector.__exit__ = MagicMock(return_value=False)
        mock_injector.inject_attack.side_effect = ValueError("bad agent")
        mock_injector_cls.return_value = mock_injector

        with patch.object(
            sys,
            "argv",
            [
                "cli.py",
                "config.yaml",
                "--agent-id",
                "agent_a",
                "--attack-type",
                "prompt_injection",
                "--payload",
                "test",
            ],
        ):
            from bili.aegis.attacks.cli import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
