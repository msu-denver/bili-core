"""Pytest fixtures for the AETHER baseline structural test suite.

The ``baseline_result`` fixture is parametrized over every JSON file in
``results/``.  When the results directory is empty (i.e. no baseline run has
been collected yet), the fixture yields no parameters and all structural tests
are automatically skipped — matching build-order step 4 ("collect baseline
results") in the plan.

Environment variables
---------------------
AETHER_STUB_MODE=1   (default) — marks the session as stub mode.
AETHER_STUB_MODE=0             — marks the session as real-LLM mode.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of invocation directory
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RESULTS_DIR = Path(__file__).parent / "results"

STUB_MODE: bool = os.getenv("AETHER_STUB_MODE", "1") == "1"


def _collect_result_files() -> list[Path]:
    """Return all *.json files under results/, sorted for deterministic ordering."""
    if not _RESULTS_DIR.exists():
        return []
    return sorted(_RESULTS_DIR.glob("**/*.json"))


@pytest.fixture(params=_collect_result_files(), ids=lambda p: p.stem)
def baseline_result(request) -> dict:
    """Load one baseline result JSON file.

    Parametrized over all result files present at collection time.
    When ``results/`` is empty the fixture provides no parameters and all
    tests that depend on it are skipped.
    """
    return json.loads(request.param.read_text(encoding="utf-8"))
