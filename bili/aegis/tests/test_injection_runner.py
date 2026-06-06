"""Tests for the injection suite runner entry point.

Verifies that main() parses arguments correctly and delegates
to run_suite() with the proper configuration.
All external dependencies are mocked.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE = "bili.aegis.suites.injection.run_injection_suite"


def _fake_payload(
    payload_id="pi_direct_001",
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


class TestInjectionRunnerMain:
    """Tests for injection runner main() entry point."""

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["path/to/config.yaml"])
    def test_stub_mode_passes_correct_args(self, mock_run_suite):
        """Stub mode passes stub=True and no evaluator."""
        with patch.object(sys, "argv", ["run_injection_suite.py", "--stub"]):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        mock_run_suite.assert_called_once()
        call_kwargs = mock_run_suite.call_args[1]
        assert call_kwargs["stub"] is True
        assert call_kwargs["attack_suite"] == "prompt_injection"
        assert call_kwargs["attack_type"] == "prompt_injection"
        assert call_kwargs["semantic_evaluator"] is None
        assert call_kwargs["csv_filename"] == ("injection_results_matrix.csv")

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["path/to/config.yaml"])
    def test_phases_default_to_both(self, mock_run_suite):
        """Default phases include pre_execution and mid_execution."""
        with patch.object(sys, "argv", ["run_injection_suite.py", "--stub"]):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_suite.call_args[1]
        assert "pre_execution" in call_kwargs["phases"]
        assert "mid_execution" in call_kwargs["phases"]

    @patch(f"{_MODULE}.run_suite")
    @patch(
        f"{_MODULE}.INJECTION_PAYLOADS",
        [_fake_payload("pi_a"), _fake_payload("pi_b")],
    )
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_payload_filter_restricts_payloads(self, mock_run_suite):
        """--payloads flag filters to matching payload IDs."""
        with patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--stub",
                "--payloads",
                "pi_a",
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        payloads = mock_run_suite.call_args[1]["payloads"]
        assert len(payloads) == 1
        assert payloads[0].payload_id == "pi_a"

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_single_phase_filter(self, mock_run_suite):
        """--phases flag restricts to a single phase."""
        with patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--stub",
                "--phases",
                "pre_execution",
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_suite.call_args[1]
        assert call_kwargs["phases"] == ["pre_execution"]

    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_invalid_payload_filter_exits(self):
        """Exits with code 1 when no payloads match the filter."""
        with patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--stub",
                "--payloads",
                "nonexistent_id",
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_custom_configs_override(self, mock_run_suite):
        """--configs overrides the default config paths."""
        with patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--stub",
                "--configs",
                "custom/a.yaml",
                "custom/b.yaml",
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_suite.call_args[1]
        assert call_kwargs["config_paths"] == [
            "custom/a.yaml",
            "custom/b.yaml",
        ]

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_results_dir_is_injection_results(self, mock_run_suite):
        """Results directory is the injection results folder."""
        with patch.object(sys, "argv", ["run_injection_suite.py", "--stub"]):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        results_dir = mock_run_suite.call_args[1]["results_dir"]
        assert str(results_dir).endswith("injection/results")

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_non_stub_initializes_semantic_evaluator(self, mock_run_suite, tmp_path):
        """Non-stub mode builds a SemanticEvaluator and an existing baseline dir."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        fake_eval = MagicMock(name="SemanticEvaluator")
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator", return_value=fake_eval
        ), patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--baseline-results",
                str(baseline_dir),
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_suite.call_args[1]
        assert call_kwargs["stub"] is False
        assert call_kwargs["semantic_evaluator"] is fake_eval
        assert call_kwargs["baseline_results_dir"] == baseline_dir

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_missing_baseline_dir_warns_and_clears(self, mock_run_suite, capsys):
        """A nonexistent baseline dir is reported and reset to None."""
        fake_eval = MagicMock(name="SemanticEvaluator")
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator", return_value=fake_eval
        ), patch.object(
            sys,
            "argv",
            [
                "run_injection_suite.py",
                "--baseline-results",
                "/no/such/dir",
            ],
        ):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        err = capsys.readouterr().err
        assert "baseline results dir not found" in err
        assert mock_run_suite.call_args[1]["baseline_results_dir"] is None

    @patch(f"{_MODULE}.run_suite")
    @patch(f"{_MODULE}.INJECTION_PAYLOADS", [_fake_payload()])
    @patch(f"{_MODULE}.CONFIG_PATHS", ["c.yaml"])
    def test_semantic_evaluator_import_error_handled(self, mock_run_suite):
        """A failing SemanticEvaluator init leaves the evaluator as None."""
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator",
            side_effect=ImportError("missing dep"),
        ), patch.object(sys, "argv", ["run_injection_suite.py"]):
            from bili.aegis.suites.injection.run_injection_suite import main

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        assert mock_run_suite.call_args[1]["semantic_evaluator"] is None
