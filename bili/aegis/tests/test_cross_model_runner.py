"""Tests for the cross-model transferability suite runner.

Covers main() CLI parsing, model matrix building,
provider_family derivation, helper functions, and summary printing.
All external dependencies (LLM calls, file I/O, YAML loading) are mocked.
"""

# Tests exercise private helpers (_provider_family, _write_csv, etc.)
# pylint: disable=protected-access

import csv
import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE = "bili.aegis.suites.cross_model.run_cross_model_suite"


def _fake_payload(payload_id="pi_001"):
    """Build a minimal payload namespace."""
    return SimpleNamespace(
        payload_id=payload_id,
        injection_type="prompt_injection",
        severity="high",
        payload="evil text",
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


class _FakeAgent(SimpleNamespace):
    """Agent stub that mimics ``AgentSpec.model_copy`` for patch testing."""

    def model_copy(self, update=None):
        """Return a copy with the given field updates applied."""
        data = dict(self.__dict__)
        data.update(update or {})
        return _FakeAgent(**data)


class _FakeConfig(SimpleNamespace):
    """Config stub that mimics ``MASConfig.model_copy`` for patch testing."""

    def model_copy(self, update=None):
        """Return a copy with the given field updates applied."""
        data = dict(self.__dict__)
        data.update(update or {})
        return _FakeConfig(**data)


def _fake_config():
    """Build a minimal MAS config that supports ``model_copy``."""
    agent = _FakeAgent(agent_id="agent_a", model_name="m", temperature=0.2)
    return _FakeConfig(mas_id="test_mas", entry_point=None, agents=[agent])


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


# -----------------------------------------------------------------------
# Lazy-import helpers — the module performs sys.path manipulation and
# heavyweight imports at module level; we gate those behind patches.
# -----------------------------------------------------------------------


def _import_module():
    """Import the cross_model runner module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.suites.cross_model import run_cross_model_suite as mod

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
        csv_path = mod._write_csv(rows, tmp_path, "cross_model_matrix.csv")
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

    def test_returns_none_on_corrupt_json(self, tmp_path):
        """Returns None and warns when the baseline JSON cannot be parsed."""
        mod = _import_module()
        mas_dir = tmp_path / "mas1"
        mas_dir.mkdir()
        (mas_dir / "bad.json").write_text("{not valid json")
        assert mod._load_baseline(tmp_path, "mas1") is None


# =========================================================================
# _find_repo_root
# =========================================================================


class TestFindRepoRoot:
    """Tests for the inlined bootstrap _find_repo_root helper."""

    def test_walks_up_to_git_dir(self, tmp_path, monkeypatch):
        """Returns the first ancestor directory containing a .git dir."""
        mod = _import_module()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (tmp_path / ".git").mkdir()
        fake_file = nested / "run.py"
        monkeypatch.setattr(mod, "__file__", str(fake_file))
        assert mod._find_repo_root() == tmp_path

    def test_raises_when_no_git_dir(self, tmp_path, monkeypatch):
        """Raises RuntimeError when no .git directory exists above the file."""
        mod = _import_module()
        isolated = tmp_path / "no_git_here" / "pkg"
        isolated.mkdir(parents=True)
        monkeypatch.setattr(mod, "__file__", str(isolated / "run.py"))
        # No .git is created anywhere from tmp_path up to the filesystem root,
        # so the walk exhausts and raises.
        with pytest.raises(RuntimeError, match="repo root"):
            mod._find_repo_root()


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
        "bili.aegis.suites.cross_model.run_cross_model_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch("bili.aegis.suites.cross_model.run_cross_model_suite._run_config_for_model")
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
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model_id"] is None

    @patch(
        "bili.aegis.suites.cross_model.run_cross_model_suite"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.suites.cross_model.run_cross_model_suite.INJECTION_PAYLOADS",
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
        "bili.aegis.suites.cross_model.run_cross_model_suite"
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

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_config_for_model")
    def test_default_model_matrix_used(self, mock_run, mock_args):
        """Without --models the full MODEL_MATRIX is iterated in non-stub mode."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=False,
            configs=["a.yaml"],
            models=None,
            payloads=None,
            phases=["pre_execution"],
            baseline_results=None,
            log_level="WARNING",
        )
        with patch("bili.aegis.evaluator.SemanticEvaluator", return_value=MagicMock()):
            mock_run.return_value = ([], None)
            with pytest.raises(SystemExit) as exc_info:
                mod.main()
        # 0 ran rows -> exit 0; called once per model in the default matrix.
        assert exc_info.value.code == 0
        assert mock_run.call_count == len(mod.MODEL_MATRIX)
        # The model_id of the first call is the first matrix entry's id.
        assert mock_run.call_args_list[0][1]["model_id"] == mod.MODEL_MATRIX[0][0]

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_config_for_model")
    def test_missing_baseline_dir_clears_to_none(self, mock_run, mock_args, capsys):
        """A nonexistent baseline dir warns and is passed through as None."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=True,
            configs=["a.yaml"],
            models=None,
            payloads=None,
            phases=["pre_execution"],
            baseline_results="/no/such/baseline",
            log_level="WARNING",
        )
        mock_run.return_value = ([], None)
        with pytest.raises(SystemExit):
            mod.main()
        err = capsys.readouterr().err
        assert "baseline results dir not found" in err
        assert mock_run.call_args[1]["baseline_results_dir"] is None

    @patch(f"{_MODULE}.argparse.ArgumentParser.parse_args")
    @patch(f"{_MODULE}._run_config_for_model")
    def test_semantic_evaluator_error_handled(self, mock_run, mock_args):
        """A failing SemanticEvaluator init in non-stub mode is tolerated."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            stub=False,
            configs=["a.yaml"],
            models=["us.anthropic.claude-3-5-haiku-20241022-v1:0"],
            payloads=None,
            phases=["pre_execution"],
            baseline_results=None,
            log_level="WARNING",
        )
        with patch(
            "bili.aegis.evaluator.SemanticEvaluator",
            side_effect=RuntimeError("no creds"),
        ):
            mock_run.return_value = ([], None)
            with pytest.raises(SystemExit):
                mod.main()
        assert mock_run.call_args[1]["semantic_evaluator"] is None


# =========================================================================
# _run_config_for_model
# =========================================================================


class TestRunConfigForModel:
    """Tests for the per-config per-model runner loop."""

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_skips_missing_config(self, mock_load, _inj, _det, _log, tmp_path):
        """Returns empty rows and None run_dir when the YAML path is missing."""
        mod = _import_module()
        rows, run_dir = mod._run_config_for_model(
            yaml_path="missing.yaml",
            model_id=None,
            model_display_name="stub",
            payloads=[_fake_payload()],
            phases=["pre_execution"],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )
        assert rows == []
        assert run_dir is None
        mock_load.assert_not_called()

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_writes_result_and_matrix_row(
        self, mock_load, mock_injector_cls, _det, _log, tmp_path
    ):
        """A successful injection writes a JSON result and a non-skipped row."""
        mod = _import_module()
        config = _fake_config()
        mock_load.return_value = config
        mock_injector_cls.return_value = _patched_injector(
            _fake_attack_result(influenced=["agent_a"])
        )

        (tmp_path / "t.yaml").write_text("x")
        rows, run_dir = mod._run_config_for_model(
            yaml_path="t.yaml",
            model_id=None,
            model_display_name="stub",
            payloads=[_fake_payload("pi_001")],
            phases=["pre_execution"],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert len(rows) == 1
        row = rows[0]
        assert row["skipped"] == "false"
        assert row["tier1_pass"] == "true"
        assert row["model_id"] == "stub"
        assert row["tier2_influenced"] == json.dumps(["agent_a"])
        # JSON result file was written under the run_dir/model_safe directory.
        written = list(run_dir.glob("**/pi_001_pre_execution.json"))
        assert written, "expected a per-case JSON result file"
        data = json.loads(written[0].read_text())
        assert data["execution"]["success"] is True
        assert data["execution"]["duration_ms"] == pytest.approx(1000.0)

    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_inject_error_produces_skip_row(
        self, mock_load, mock_injector_cls, _det, _log, tmp_path
    ):
        """A raised inject_attack records a skipped row with a skip_reason."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_injector_cls.return_value = _patched_injector(
            side_effect=RuntimeError("no creds")
        )

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_config_for_model(
            yaml_path="t.yaml",
            model_id="amazon.nova-pro-v1:0",
            model_display_name="Nova",
            payloads=[_fake_payload("pi_001")],
            phases=["pre_execution"],
            stub_mode=False,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert len(rows) == 1
        assert rows[0]["skipped"] == "true"
        assert "RuntimeError" in rows[0]["skip_reason"]
        assert rows[0]["provider_family"] == "amazon_bedrock"

    @patch(f"{_MODULE}._load_baseline")
    @patch(f"{_MODULE}.SecurityEventLogger")
    @patch(f"{_MODULE}.SecurityEventDetector")
    @patch(f"{_MODULE}.AttackInjector")
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_tier3_evaluation_records_best_score(
        self, mock_load, mock_injector_cls, _det, _log, mock_baseline, tmp_path
    ):
        """Non-stub run with a baseline records the best Tier-3 score."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_injector_cls.return_value = _patched_injector(_fake_attack_result())
        mock_baseline.return_value = {"baseline": True}

        evaluator = MagicMock()
        evaluator.evaluate.return_value = [
            SimpleNamespace(score=1, confidence="low", reasoning="r1"),
            SimpleNamespace(score=3, confidence="high", reasoning="r3"),
        ]

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_config_for_model(
            yaml_path="t.yaml",
            model_id="gemini-2.0-flash",
            model_display_name="Gemini",
            payloads=[_fake_payload("pi_001")],
            phases=["pre_execution"],
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
    @patch(f"{_MODULE}.load_mas_from_yaml")
    def test_tier3_evaluator_error_swallowed(
        self, mock_load, mock_injector_cls, _det, _log, mock_baseline, tmp_path
    ):
        """A failing evaluator leaves Tier-3 columns empty but records the row."""
        mod = _import_module()
        mock_load.return_value = _fake_config()
        mock_injector_cls.return_value = _patched_injector(_fake_attack_result())
        mock_baseline.return_value = {"baseline": True}

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("judge down")

        (tmp_path / "t.yaml").write_text("x")
        rows, _run_dir = mod._run_config_for_model(
            yaml_path="t.yaml",
            model_id="gemini-2.0-flash",
            model_display_name="Gemini",
            payloads=[_fake_payload("pi_001")],
            phases=["pre_execution"],
            stub_mode=False,
            semantic_evaluator=evaluator,
            baseline_results_dir=tmp_path / "baseline",
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )

        assert rows[0]["tier3_score"] == ""


# =========================================================================
# run_cross_model_suite (programmatic API)
# =========================================================================


class TestRunCrossModelSuite:
    """Tests for the non-exiting programmatic entry point."""

    @patch(f"{_MODULE}._write_csv")
    @patch(f"{_MODULE}._run_config_for_model")
    def test_aggregates_and_writes_csv(self, mock_run, mock_write_csv, tmp_path):
        """Aggregates rows across the model x config grid and writes a CSV."""
        mod = _import_module()
        mock_run.side_effect = [
            (
                [
                    {
                        "skipped": "false",
                        "tier1_pass": "true",
                        "tier2_influenced": '["agent_a"]',
                        "payload_id": "pi_001",
                        "phase": "pre_execution",
                        "provider_family": "anthropic_bedrock",
                    }
                ],
                tmp_path / "run_003",
            ),
            (
                [
                    {
                        "skipped": "false",
                        "tier1_pass": "false",
                        "tier2_influenced": "[]",
                        "payload_id": "pi_001",
                        "phase": "pre_execution",
                        "provider_family": "google_vertex",
                    }
                ],
                tmp_path / "run_004",
            ),
        ]
        mock_write_csv.return_value = tmp_path / "out.csv"

        rows, first_run = mod.run_cross_model_suite(
            payloads=[_fake_payload()],
            config_paths=["a.yaml"],
            phases=["pre_execution"],
            model_matrix=[("mA", "A"), ("mB", "B")],
            stub_mode=False,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path,
            repo_root=tmp_path,
        )

        assert len(rows) == 2
        assert first_run == "run_003"
        assert mock_write_csv.call_args[0][2] == "cross_model_matrix_run_003.csv"

    @patch(f"{_MODULE}._run_config_for_model", return_value=([], None))
    def test_no_rows_returns_empty(self, _mock_run, tmp_path):
        """With no rows the function returns empty results and no run dir."""
        mod = _import_module()
        rows, first_run = mod.run_cross_model_suite(
            payloads=[_fake_payload()],
            config_paths=["a.yaml"],
            phases=["pre_execution"],
            model_matrix=[(None, "stub")],
            stub_mode=True,
            semantic_evaluator=None,
            baseline_results_dir=None,
            results_dir=tmp_path,
            repo_root=tmp_path,
        )
        assert rows == []
        assert first_run is None
