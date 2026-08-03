"""Pytest fixtures for the AETHER persistence attack test suite.

The ``persistence_result`` fixture is parametrized over every JSON file in
``fixtures/`` and ``results/``.  Committed sample fixtures under ``fixtures/``
mean the suite always has at least one well-formed result to validate, so the
structural tests run regardless of whether ``run_persistence_suite.py`` has
populated ``results/`` with live output.

The ``log_dir`` fixture derives the per-config log directory from the
``mas_id`` field in the loaded result dict, preferring the committed
``fixtures/`` companion logs (``attack_log.ndjson`` and
``security_events.ndjson``) and falling back to live runner output in
``results/``.

Note on ``_find_repo_root``
---------------------------
The helper is inlined here rather than imported from
``bili.aegis.suites._helpers`` because this file must bootstrap ``sys.path``
before any ``bili.*`` import is possible — the shared module cannot be
imported until after ``sys.path`` contains the repo root.

Environment variables
---------------------
AETHER_STUB_MODE=1  (default) — marks the session as stub mode.
AETHER_STUB_MODE=0             — marks the session as real-LLM mode.

Usage
-----
Run the suite first (requires a persistent checkpointer), then run the
structural tests:

    python bili/aegis/suites/persistence/run_persistence_suite.py
    pytest bili/aegis/tests/persistence/test_persistence_structural.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of invocation directory
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:  # pylint: disable=duplicate-code
    """Walk up from this file until a ``.git`` entry is found.

    Tests for existence rather than for a directory: in a git worktree
    ``.git`` is a FILE holding a ``gitdir:`` pointer, so a directory test
    is false there and the walk runs off the top of the filesystem.
    """
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (no .git entry found)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RESULTS_DIR = Path(__file__).parent / "results"
# Committed sample result fixtures restored from git history (commit 4c200c6~1).
# These give the structural tests a stable, version-controlled set of
# well-formed results to validate against, independent of whether a local
# attack-suite run has populated results/. Live run output in results/ is
# also picked up when present.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _collect_result_files() -> list:
    """Return all JSON result file paths under fixtures/ and results/."""
    files = []
    if _FIXTURES_DIR.exists():
        files.extend(sorted(_FIXTURES_DIR.glob("**/*.json")))
    if _RESULTS_DIR.exists():
        files.extend(sorted(_RESULTS_DIR.glob("**/*.json")))
    return files


@pytest.fixture(
    params=_collect_result_files(),
    ids=lambda p: p.stem,
)
def persistence_result(request) -> dict:
    """Load one persistence result JSON file.

    Parametrized over all result files present at collection time: the
    committed sample fixtures under ``fixtures/`` plus any live runner
    output under ``results/``.
    """
    return json.loads(request.param.read_text(encoding="utf-8"))


@pytest.fixture
def log_dir(persistence_result: dict) -> Path:
    """Return the per-config results subdirectory for the current test case.

    ``run_persistence_suite.py`` writes ``attack_log.ndjson`` and
    ``security_events.ndjson`` into ``results/{mas_id}/``.  This fixture
    points to that directory so structural tests can assert log file existence.
    """
    mas_id = persistence_result["mas_id"]
    # Prefer the committed fixtures log dir when it carries the companion
    # ndjson log files; otherwise fall back to live run output in results/.
    fixture_log_dir = _FIXTURES_DIR / mas_id
    if (fixture_log_dir / "security_events.ndjson").exists():
        return fixture_log_dir
    return _RESULTS_DIR / mas_id
