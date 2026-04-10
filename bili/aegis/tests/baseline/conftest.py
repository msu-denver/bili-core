"""Pytest fixtures for the AETHER baseline structural test suite.

The ``baseline_result`` fixture is parametrized over every JSON file in
``results/``.  When the results directory is empty (i.e. no baseline run has
been collected yet), the fixture yields no parameters and all structural tests
are automatically skipped — matching build-order step 4 ("collect baseline
results") in the plan.
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of invocation directory
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this file until a .git directory is found."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (.git directory not found)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
def baseline_result(request) -> dict:
    """Load one baseline result JSON file.

    Parametrized over all result files present at collection time.
    When ``results/`` is empty the fixture provides no parameters and all
    tests that depend on it are skipped.
    """
    if request.param is None:
        pytest.skip("No result files found")
    return json.loads(request.param.read_text(encoding="utf-8"))
