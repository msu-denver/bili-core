"""Tests for the four remaining AEGIS suite runner entry points.

Covers agent_impersonation, bias_inheritance, jailbreak, and
memory_poisoning runners.  Each runner follows the same pattern:
a main() function that parses CLI args and delegates to run_suite().
All external dependencies are mocked.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# -----------------------------------------------------------------------
# Module paths under test
# -----------------------------------------------------------------------
_AI_MOD = "bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite"
_BI_MOD = "bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite"
_JB_MOD = "bili.aegis.tests.jailbreak.run_jailbreak_suite"
_MP_MOD = "bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _fake_payload(
    payload_id="fake_001",
    injection_type="test_type",
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


# ===================================================================
# Agent Impersonation Runner
# ===================================================================


class TestAgentImpersonationRunnerMain:
    """Tests for agent impersonation runner main()."""

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_stub_mode_passes_correct_args(self, mock_run_suite):
        """Stub mode passes stub=True and no evaluator."""
        with patch.object(
            sys,
            "argv",
            ["run_agent_impersonation_suite.py", "--stub"],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["stub"] is True
        assert kw["attack_suite"] == "agent_impersonation"
        assert kw["attack_type"] == "agent_impersonation"
        assert kw["semantic_evaluator"] is None
        assert kw["csv_filename"] == ("agent_impersonation_results_matrix.csv")

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_phases_default_to_both(self, mock_run_suite):
        """Default phases include pre and mid execution."""
        with patch.object(
            sys,
            "argv",
            ["run_agent_impersonation_suite.py", "--stub"],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert "pre_execution" in kw["phases"]
        assert "mid_execution" in kw["phases"]

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload("ai_a"), _fake_payload("ai_b")],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_payload_filter_restricts(self, mock_run_suite):
        """--payloads flag filters to matching IDs."""
        with patch.object(
            sys,
            "argv",
            [
                "run_agent_impersonation_suite.py",
                "--stub",
                "--payloads",
                "ai_a",
            ],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        payloads = mock_run_suite.call_args[1]["payloads"]
        assert len(payloads) == 1
        assert payloads[0].payload_id == "ai_a"

    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_invalid_payload_filter_exits(self):
        """Exits with code 1 when no payloads match."""
        with patch.object(
            sys,
            "argv",
            [
                "run_agent_impersonation_suite.py",
                "--stub",
                "--payloads",
                "nonexistent",
            ],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_custom_configs_override(self, mock_run_suite):
        """--configs overrides default config paths."""
        with patch.object(
            sys,
            "argv",
            [
                "run_agent_impersonation_suite.py",
                "--stub",
                "--configs",
                "a.yaml",
                "b.yaml",
            ],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["config_paths"] == ["a.yaml", "b.yaml"]

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_results_dir_suffix(self, mock_run_suite):
        """Results dir ends with agent_impersonation/results."""
        with patch.object(
            sys,
            "argv",
            ["run_agent_impersonation_suite.py", "--stub"],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        results_dir = mock_run_suite.call_args[1]["results_dir"]
        assert str(results_dir).endswith("agent_impersonation/results")

    @patch(f"{_AI_MOD}.run_suite")
    @patch(
        f"{_AI_MOD}.AGENT_IMPERSONATION_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_AI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_single_phase_filter(self, mock_run_suite):
        """--phases restricts to a single phase."""
        with patch.object(
            sys,
            "argv",
            [
                "run_agent_impersonation_suite.py",
                "--stub",
                "--phases",
                "pre_execution",
            ],
        ):
            from bili.aegis.tests.agent_impersonation.run_agent_impersonation_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["phases"] == ["pre_execution"]


# ===================================================================
# Bias Inheritance Runner
# ===================================================================


class TestBiasInheritanceRunnerMain:
    """Tests for bias inheritance runner main()."""

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_stub_mode_passes_correct_args(self, mock_run_suite):
        """Stub mode passes stub=True and no evaluator."""
        with patch.object(
            sys,
            "argv",
            ["run_bias_inheritance_suite.py", "--stub"],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["stub"] is True
        assert kw["attack_suite"] == "bias_inheritance"
        assert kw["attack_type"] == "bias_inheritance"
        assert kw["semantic_evaluator"] is None
        assert kw["csv_filename"] == ("bias_inheritance_results_matrix.csv")

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_phases_default_to_both(self, mock_run_suite):
        """Default phases include pre and mid execution."""
        with patch.object(
            sys,
            "argv",
            ["run_bias_inheritance_suite.py", "--stub"],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert "pre_execution" in kw["phases"]
        assert "mid_execution" in kw["phases"]

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload("bi_a"), _fake_payload("bi_b")],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_payload_filter_restricts(self, mock_run_suite):
        """--payloads flag filters to matching IDs."""
        with patch.object(
            sys,
            "argv",
            [
                "run_bias_inheritance_suite.py",
                "--stub",
                "--payloads",
                "bi_a",
            ],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        payloads = mock_run_suite.call_args[1]["payloads"]
        assert len(payloads) == 1
        assert payloads[0].payload_id == "bi_a"

    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_invalid_payload_filter_exits(self):
        """Exits with code 1 when no payloads match."""
        with patch.object(
            sys,
            "argv",
            [
                "run_bias_inheritance_suite.py",
                "--stub",
                "--payloads",
                "nonexistent",
            ],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_custom_configs_override(self, mock_run_suite):
        """--configs overrides default config paths."""
        with patch.object(
            sys,
            "argv",
            [
                "run_bias_inheritance_suite.py",
                "--stub",
                "--configs",
                "a.yaml",
                "b.yaml",
            ],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["config_paths"] == ["a.yaml", "b.yaml"]

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_results_dir_suffix(self, mock_run_suite):
        """Results dir ends with bias_inheritance/results."""
        with patch.object(
            sys,
            "argv",
            ["run_bias_inheritance_suite.py", "--stub"],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        results_dir = mock_run_suite.call_args[1]["results_dir"]
        assert str(results_dir).endswith("bias_inheritance/results")

    @patch(f"{_BI_MOD}.run_suite")
    @patch(
        f"{_BI_MOD}.BIAS_INHERITANCE_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_BI_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_single_phase_filter(self, mock_run_suite):
        """--phases restricts to a single phase."""
        with patch.object(
            sys,
            "argv",
            [
                "run_bias_inheritance_suite.py",
                "--stub",
                "--phases",
                "mid_execution",
            ],
        ):
            from bili.aegis.tests.bias_inheritance.run_bias_inheritance_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["phases"] == ["mid_execution"]


# ===================================================================
# Jailbreak Runner
# ===================================================================


class TestJailbreakRunnerMain:
    """Tests for jailbreak runner main()."""

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_stub_mode_passes_correct_args(self, mock_run_suite):
        """Stub mode passes stub=True and no evaluator."""
        with patch.object(
            sys,
            "argv",
            ["run_jailbreak_suite.py", "--stub"],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["stub"] is True
        assert kw["attack_suite"] == "jailbreak"
        assert kw["attack_type"] == "jailbreak"
        assert kw["semantic_evaluator"] is None
        assert kw["csv_filename"] == ("jailbreak_results_matrix.csv")

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_phases_default_to_both(self, mock_run_suite):
        """Default phases include pre and mid execution."""
        with patch.object(
            sys,
            "argv",
            ["run_jailbreak_suite.py", "--stub"],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert "pre_execution" in kw["phases"]
        assert "mid_execution" in kw["phases"]

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload("jb_a"), _fake_payload("jb_b")],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_payload_filter_restricts(self, mock_run_suite):
        """--payloads flag filters to matching IDs."""
        with patch.object(
            sys,
            "argv",
            [
                "run_jailbreak_suite.py",
                "--stub",
                "--payloads",
                "jb_a",
            ],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        payloads = mock_run_suite.call_args[1]["payloads"]
        assert len(payloads) == 1
        assert payloads[0].payload_id == "jb_a"

    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_invalid_payload_filter_exits(self):
        """Exits with code 1 when no payloads match."""
        with patch.object(
            sys,
            "argv",
            [
                "run_jailbreak_suite.py",
                "--stub",
                "--payloads",
                "nonexistent",
            ],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_custom_configs_override(self, mock_run_suite):
        """--configs overrides default config paths."""
        with patch.object(
            sys,
            "argv",
            [
                "run_jailbreak_suite.py",
                "--stub",
                "--configs",
                "a.yaml",
                "b.yaml",
            ],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["config_paths"] == ["a.yaml", "b.yaml"]

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_results_dir_suffix(self, mock_run_suite):
        """Results dir ends with jailbreak/results."""
        with patch.object(
            sys,
            "argv",
            ["run_jailbreak_suite.py", "--stub"],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        results_dir = mock_run_suite.call_args[1]["results_dir"]
        assert str(results_dir).endswith("jailbreak/results")

    @patch(f"{_JB_MOD}.run_suite")
    @patch(
        f"{_JB_MOD}.JAILBREAK_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_JB_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_single_phase_filter(self, mock_run_suite):
        """--phases restricts to a single phase."""
        with patch.object(
            sys,
            "argv",
            [
                "run_jailbreak_suite.py",
                "--stub",
                "--phases",
                "pre_execution",
            ],
        ):
            from bili.aegis.tests.jailbreak.run_jailbreak_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["phases"] == ["pre_execution"]


# ===================================================================
# Memory Poisoning Runner
# ===================================================================


class TestMemoryPoisoningRunnerMain:
    """Tests for memory poisoning runner main()."""

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_stub_mode_passes_correct_args(self, mock_run_suite):
        """Stub mode passes stub=True and no evaluator."""
        with patch.object(
            sys,
            "argv",
            ["run_memory_poisoning_suite.py", "--stub"],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["stub"] is True
        assert kw["attack_suite"] == "memory_poisoning"
        assert kw["attack_type"] == "memory_poisoning"
        assert kw["semantic_evaluator"] is None
        assert kw["csv_filename"] == ("memory_poisoning_results_matrix.csv")

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_phases_default_to_both(self, mock_run_suite):
        """Default phases include pre and mid execution."""
        with patch.object(
            sys,
            "argv",
            ["run_memory_poisoning_suite.py", "--stub"],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert "pre_execution" in kw["phases"]
        assert "mid_execution" in kw["phases"]

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload("mp_a"), _fake_payload("mp_b")],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_payload_filter_restricts(self, mock_run_suite):
        """--payloads flag filters to matching IDs."""
        with patch.object(
            sys,
            "argv",
            [
                "run_memory_poisoning_suite.py",
                "--stub",
                "--payloads",
                "mp_a",
            ],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        payloads = mock_run_suite.call_args[1]["payloads"]
        assert len(payloads) == 1
        assert payloads[0].payload_id == "mp_a"

    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_invalid_payload_filter_exits(self):
        """Exits with code 1 when no payloads match."""
        with patch.object(
            sys,
            "argv",
            [
                "run_memory_poisoning_suite.py",
                "--stub",
                "--payloads",
                "nonexistent",
            ],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_custom_configs_override(self, mock_run_suite):
        """--configs overrides default config paths."""
        with patch.object(
            sys,
            "argv",
            [
                "run_memory_poisoning_suite.py",
                "--stub",
                "--configs",
                "a.yaml",
                "b.yaml",
            ],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["config_paths"] == ["a.yaml", "b.yaml"]

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_results_dir_suffix(self, mock_run_suite):
        """Results dir ends with memory_poisoning/results."""
        with patch.object(
            sys,
            "argv",
            ["run_memory_poisoning_suite.py", "--stub"],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        results_dir = mock_run_suite.call_args[1]["results_dir"]
        assert str(results_dir).endswith("memory_poisoning/results")

    @patch(f"{_MP_MOD}.run_suite")
    @patch(
        f"{_MP_MOD}.MEMORY_POISONING_PAYLOADS",
        [_fake_payload()],
    )
    @patch(f"{_MP_MOD}.CONFIG_PATHS", ["c.yaml"])
    def test_single_phase_filter(self, mock_run_suite):
        """--phases restricts to a single phase."""
        with patch.object(
            sys,
            "argv",
            [
                "run_memory_poisoning_suite.py",
                "--stub",
                "--phases",
                "mid_execution",
            ],
        ):
            from bili.aegis.tests.memory_poisoning.run_memory_poisoning_suite import (  # pylint: disable=import-outside-toplevel
                main,
            )

            mock_run_suite.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()

        kw = mock_run_suite.call_args[1]
        assert kw["phases"] == ["mid_execution"]
