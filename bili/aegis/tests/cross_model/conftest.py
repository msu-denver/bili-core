"""Pytest fixtures for the AETHER cross-model transferability test suite.

The ``cross_model_result`` fixture is parametrized over every JSON file in
``results/``.  When the directory is empty (i.e. ``run_cross_model_suite.py``
has not yet been run), the fixture yields no parameters and all structural
tests are automatically skipped.

The ``log_dir`` fixture derives the per-config/per-model log directory from
the ``mas_id`` and ``model_id`` fields in the loaded result dict.  Log files
(``attack_log.ndjson`` and ``security_events.ndjson``) are written there by
the runner.

Note on ``_find_repo_root``
---------------------------
The helper is inlined here rather than imported from
``bili.aegis.tests._helpers`` because this file must bootstrap ``sys.path``
before any ``bili.*`` import is possible — the shared module cannot be
imported until after ``sys.path`` contains the repo root.

Usage
-----
Run the suite first, then run the structural tests:

    python bili/aegis/tests/cross_model/run_cross_model_suite.py --stub
    pytest bili/aegis/tests/cross_model/test_cross_model_structural.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of invocation directory
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:  # pylint: disable=duplicate-code
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (.git directory not found)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bili.aegis.tests._helpers import (  # noqa: E402  pylint: disable=wrong-import-position
    model_id_safe as _model_id_safe,
)

_RESULTS_DIR = Path(__file__).parent / "results"


def _collect_result_files() -> list:
    """Return result file params, or a skip marker when no files exist."""
    if not _RESULTS_DIR.exists():
        return [pytest.param(None, marks=pytest.mark.skip(reason="No result files"))]
    files = sorted(_RESULTS_DIR.glob("**/*.json"))
    if not files:
        return [pytest.param(None, marks=pytest.mark.skip(reason="No result files"))]
    return files


@pytest.fixture(
    params=_collect_result_files(), ids=lambda p: p.stem if p else "no_results"
)
def cross_model_result(request) -> dict:  # pylint: disable=redefined-outer-name
    """Load one cross-model result JSON file.

    Parametrized over all result files present at collection time.
    When ``results/`` is empty the fixture provides no parameters and all
    tests that depend on it are skipped.
    """
    if request.param is None:
        pytest.skip("No result files found")
    return json.loads(request.param.read_text(encoding="utf-8"))


@pytest.fixture
def log_dir(cross_model_result: dict) -> Path:
    """Return the per-config/per-model results subdirectory for the current test case.

    ``run_cross_model_suite.py`` writes ``attack_log.ndjson`` and
    ``security_events.ndjson`` into ``results/{mas_id}/{model_id_safe}/``.
    This fixture points to that directory so structural tests can assert log
    file existence.
    """
    mas_id = cross_model_result["mas_id"]
    model_id = cross_model_result.get("model_id")
    return _RESULTS_DIR / mas_id / _model_id_safe(model_id)
