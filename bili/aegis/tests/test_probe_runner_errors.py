"""Error-path + helper-branch tests for ``run_probe_suite``.

The happy path lives in ``test_probe_runner_smoke.py``; this module drives
the fault-isolation branches that the per-file ≥90% coverage gate requires:
per-``terminated_reason`` failed rows, victim/judge failures, the grid-loop's
continue-after-framework-error guarantee, and the small CLI helpers. It also
locks in the baseline-loader fix (versioned ``run_NNN/`` layout is now found).

Anti-cheat: failures are injected via monkeypatch so each branch is exercised
for the reason it claims, and the grid-loop test uses two seeds so a broken
``except`` that aborted the run (instead of continuing) would record one error
instead of two.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import bili.aegis.suites.probe.run_probe_suite as runner
from bili.aegis.probe.policies import POLICY_REGISTRY
from bili.aegis.probe.schema import ProbeOutcomeReason
from bili.aegis.suites.probe.payloads.probe_objectives import PROBE_OBJECTIVE_LIBRARY
from bili.aegis.suites.probe.run_probe_suite import (
    _AttackerDependencies,
    _build_argparser,
    _build_session_and_budget,
    _execute_session_grid,
    _filter_objectives,
    _GridInputs,
    _load_baseline_text,
    _load_victim,
    _make_failed_session_row,
    _model_config,
    _print_summary,
    _resolve_attacker_dependencies,
    _resolve_runner_paths,
    _run_one_session,
    _SessionIdentity,
    _SessionRunSpec,
    _VictimReady,
)
from bili.aegis.tests.conftest import make_probe_objective

_SIMPLE_CHAIN = "bili/aether/config/examples/simple_chain.yaml"


# -------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------


def _raiser(exc: Exception):
    """Return a function that ignores its args and raises ``exc``."""

    def _f(*_args, **_kwargs):
        raise exc

    return _f


def _identity(**kwargs) -> _SessionIdentity:
    defaults: dict = {
        "objective": make_probe_objective(),
        "config_path": Path(_SIMPLE_CHAIN),
        "policy_name": "pair",
        "seed": 0,
    }
    defaults.update(kwargs)
    return _SessionIdentity(**defaults)


def _deps(**kwargs) -> _AttackerDependencies:
    defaults: dict = {
        "attacker_cfg": {"model_name": "deepseek-chat"},
        "judge_cfg": {"model_name": "gemini-2.5-flash"},
        "victim_cfg": {"model_name": "us.anthropic.claude-sonnet-4-6"},
        "crafter_llm": None,
        "evaluator_llm": None,
        "policy_llm": None,
    }
    defaults.update(kwargs)
    return _AttackerDependencies(**defaults)


def _args(results_dir):
    """A complete argparse Namespace in --stub mode pointed at ``results_dir``."""
    return _build_argparser().parse_args(["--stub", "--results-dir", str(results_dir)])


def _spec(is_stub: bool = True, **kwargs) -> _SessionRunSpec:
    defaults: dict = {
        "identity": _identity(),
        "policy_cls": POLICY_REGISTRY["pair"],
        "is_stub": is_stub,
        "baseline_text": None,
    }
    defaults.update(kwargs)
    return _SessionRunSpec(**defaults)


# -------------------------------------------------------------------------
# _make_failed_session_row — one row per terminated_reason
# -------------------------------------------------------------------------


@pytest.mark.parametrize("reason", list(ProbeOutcomeReason))
def test_make_failed_session_row_records_each_reason(reason):
    """The CSV row carries the reason's .value, not a hardcoded constant."""
    row = _make_failed_session_row(
        _identity(),
        {"model_name": "deepseek-chat"},
        {"model_name": "gemini-2.5-flash"},
        reason,
        "explanatory error string",
    )
    assert row["terminated_reason"] == reason.value
    assert row["attack_suite"] == "probe"


# -------------------------------------------------------------------------
# _load_victim — failure modes both map to VICTIM_CRASHED
# -------------------------------------------------------------------------


def test_load_victim_yaml_load_failure(monkeypatch):
    """A YAML-load exception → (failed_row, None) with VICTIM_CRASHED."""
    monkeypatch.setattr(
        "bili.aether.config.loader.load_mas_from_yaml",
        _raiser(ValueError("malformed yaml")),
    )
    failed_row, victim = _load_victim(_spec(), _deps())
    assert victim is None
    assert failed_row["terminated_reason"] == ProbeOutcomeReason.VICTIM_CRASHED.value


def test_load_victim_executor_init_failure(monkeypatch):
    """A MASExecutor.initialize() fault (real mode) → VICTIM_CRASHED."""
    monkeypatch.setattr(
        "bili.aether.config.loader.load_mas_from_yaml",
        lambda path: SimpleNamespace(
            mas_id="m",
            agents=[SimpleNamespace(agent_id="a", role="reviewer")],
        ),
    )
    victim_executor = MagicMock()
    victim_executor.initialize.side_effect = RuntimeError("executor init boom")
    monkeypatch.setattr(
        "bili.aether.runtime.executor.MASExecutor",
        MagicMock(return_value=victim_executor),
    )
    failed_row, victim = _load_victim(_spec(is_stub=False), _deps())
    assert victim is None
    assert failed_row["terminated_reason"] == ProbeOutcomeReason.VICTIM_CRASHED.value


# -------------------------------------------------------------------------
# _run_one_session — converts failures to rows, never raises
# -------------------------------------------------------------------------


def test_run_one_session_returns_failed_row_when_victim_fails(monkeypatch, tmp_path):
    """When _load_victim fails, _run_one_session returns the failed row + empty path."""
    monkeypatch.setattr(
        "bili.aether.config.loader.load_mas_from_yaml",
        _raiser(ValueError("malformed yaml")),
    )
    row, path = _run_one_session(_spec(), _args(tmp_path))
    assert row["terminated_reason"] == ProbeOutcomeReason.VICTIM_CRASHED.value
    assert path == Path()


def test_run_one_session_records_judge_unavailable(tmp_path):
    """A judge sharing the victim's provider family → JUDGE_UNAVAILABLE row.

    The cross-provider rule (the judge must be a different provider family
    than both attacker and victim) is enforced inside AttackerMAS.initialize();
    the runner converts the resulting JudgeUnavailableError into a clean failed
    row instead of crashing the grid. In --stub mode the judge is pinned to
    gemini, so pointing --victim-model at a gemini-family name collides the two
    and trips the real check — no mock attacker required.
    """
    args = _build_argparser().parse_args(
        ["--stub", "--results-dir", str(tmp_path), "--victim-model", "gemini-2.5-flash"]
    )
    row, path = _run_one_session(_spec(), args)
    assert row["terminated_reason"] == ProbeOutcomeReason.JUDGE_UNAVAILABLE.value
    assert path == Path()


# -------------------------------------------------------------------------
# _execute_session_grid — a framework error records a row AND continues
# -------------------------------------------------------------------------


def test_execute_session_grid_continues_after_framework_error(monkeypatch, tmp_path):
    """A raise inside one cell becomes an error_row; the loop keeps going.

    Two seeds → two error_rows proves continuation (a broken except that
    aborted the grid would yield one).
    """
    monkeypatch.setattr(
        runner, "_run_one_session", _raiser(RuntimeError("framework boom"))
    )
    inputs = _GridInputs(
        args=_args(tmp_path),
        objectives=[make_probe_objective()],
        config_paths=[Path("simple_chain.yaml")],
        policy_clses={"pair": POLICY_REGISTRY["pair"]},
        seeds=[0, 1],
        baseline_dir=None,
        results_dir=tmp_path,
    )
    matrix_rows, error_rows = _execute_session_grid(inputs)
    assert not matrix_rows
    assert len(error_rows) == 2


# -------------------------------------------------------------------------
# Small CLI helpers
# -------------------------------------------------------------------------


def test_model_config_rejects_unknown_role():
    """An unrecognized role raises ValueError rather than returning a bad dict."""
    with pytest.raises(ValueError, match="Unknown role"):
        _model_config(False, "bogus_role", SimpleNamespace())


def test_resolve_attacker_dependencies_real_mode(monkeypatch, tmp_path):
    """Non-stub mode resolves the policy LLM and leaves crafter/evaluator None."""
    sentinel = object()
    monkeypatch.setattr(runner, "resolve_real_llm", lambda cfg: sentinel)
    deps = _resolve_attacker_dependencies(_args(tmp_path), is_stub=False)
    assert deps.crafter_llm is None
    assert deps.evaluator_llm is None
    assert deps.policy_llm is sentinel


def _victim_ready() -> _VictimReady:
    """A minimal loaded-victim handle for session-build tests."""
    return _VictimReady(
        mas_id="m", session_id="sid", victim_shape={}, victim_executor=object()
    )


def test_build_session_records_victim_model_config(tmp_path):
    """A non-stub session records the resolved victim model config on the session.

    Without this the sidecar reported ``victim_model_config: None`` even though
    the victim ran on a real model.
    """
    session, _ = _build_session_and_budget(
        _spec(is_stub=False), _deps(), _victim_ready(), _args(tmp_path)
    )
    assert session.victim_model_config == _deps().victim_cfg
    assert session.victim_model_config["model_name"] == "us.anthropic.claude-sonnet-4-6"


def test_build_session_stub_victim_model_config_is_sentinel(tmp_path):
    """A stub session uses the model_name=None sentinel for the victim config."""
    session, _ = _build_session_and_budget(
        _spec(is_stub=True), _deps(), _victim_ready(), _args(tmp_path)
    )
    assert session.victim_model_config == {"model_name": None}


def test_filter_objectives_returns_full_library_when_unset():
    """No --objectives → the whole library."""
    result = _filter_objectives(SimpleNamespace(objectives=None))
    assert len(result) == len(PROBE_OBJECTIVE_LIBRARY)


def test_resolve_runner_paths_resolves_relative_against_repo_root(tmp_path):
    """Relative results/baseline dirs become absolute under repo_root."""
    args = _args(tmp_path)
    args.results_dir = "rel/results"
    args.baseline_results = "rel/baseline"
    results_dir, baseline_dir = _resolve_runner_paths(args, tmp_path)
    assert results_dir == tmp_path / "rel" / "results"
    assert baseline_dir == tmp_path / "rel" / "baseline"
    assert args.results_dir == str(tmp_path / "rel" / "results")


def test_print_summary_handles_no_rows(capsys):
    """Empty matrix prints the no-sessions notice and returns early."""
    _print_summary([], [])
    assert "No sessions ran" in capsys.readouterr().out


def test_print_summary_reports_ttc_and_framework_errors(capsys):
    """TTC values and framework errors both surface in the summary block."""
    matrix_rows = [{"terminated_reason": "success", "turns_to_compromise": "2"}]
    _print_summary(matrix_rows, [{"error": "framework boom"}])
    out = capsys.readouterr().out
    assert "turns_to_compromise" in out
    assert "Framework errors" in out
    assert "framework boom" in out


# -------------------------------------------------------------------------
# _load_baseline_text — delegates to the shared versioned-aware loader
# -------------------------------------------------------------------------


def test_load_baseline_text_none_dir_returns_none():
    """No baseline dir → None (judge prompt renders 'NOT AVAILABLE')."""
    assert _load_baseline_text(None, "m") is None


def test_load_baseline_text_missing_mas_returns_none(tmp_path):
    """An absent {mas_id} dir → None."""
    assert _load_baseline_text(tmp_path, "no_such_mas") is None


def test_load_baseline_text_reads_flat_layout(tmp_path):
    """Flat ``{mas_id}/result.json`` baselines are read."""
    mas_dir = tmp_path / "m"
    mas_dir.mkdir(parents=True)
    (mas_dir / "result.json").write_text(
        json.dumps({"final_text": "FLAT TEXT"}), encoding="utf-8"
    )
    assert _load_baseline_text(tmp_path, "m") == "FLAT TEXT"


def test_load_baseline_text_reads_versioned_layout(tmp_path):
    """The fix: versioned ``run_NNN/`` baselines are now found.

    The hand-rolled loader this replaced read only a flat ``{mas_id}.json``
    and returned None for this layout, silently dropping the baseline.
    """
    run_dir = tmp_path / "m" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"final_text": "VERSIONED TEXT"}), encoding="utf-8"
    )
    assert _load_baseline_text(tmp_path, "m") == "VERSIONED TEXT"


def test_load_baseline_text_falls_back_to_summary(tmp_path):
    """When final_text is absent, summary is used."""
    mas_dir = tmp_path / "m"
    mas_dir.mkdir(parents=True)
    (mas_dir / "result.json").write_text(
        json.dumps({"summary": "SUMMARY ONLY"}), encoding="utf-8"
    )
    assert _load_baseline_text(tmp_path, "m") == "SUMMARY ONLY"
