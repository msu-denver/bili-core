"""Pytest fixtures for the AETHER injection test suite.

The ``injection_result`` fixture is parametrized over every JSON file in
``results/``.  When the directory is empty (i.e. ``run_injection_suite.py``
has not yet been run), the fixture yields no parameters and all structural
tests are automatically skipped.

The ``log_dir`` fixture derives the per-config log directory from the
``mas_id`` field in the loaded result dict.  Log files (``attack_log.ndjson``
and ``security_events.ndjson``) are written there by the runner.

Environment variables
---------------------
AETHER_STUB_MODE=1  (default) — marks the session as stub mode.
AETHER_STUB_MODE=0             — marks the session as real-LLM mode.

Usage
-----
Run the suite first, then run the structural tests:

    python bili/aether/tests/injection/run_injection_suite.py --stub
    pytest bili/aether/tests/injection/test_injection_structural.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of invocation directory
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RESULTS_DIR = Path(__file__).parent / "results"

STUB_MODE: bool = os.getenv("AETHER_STUB_MODE", "1") == "1"


def _collect_result_files() -> list[Path]:
    """Return all *.json result files under results/, sorted for deterministic ordering.

    Excludes the CSV matrix file.
    """
    if not _RESULTS_DIR.exists():
        return []
    return sorted(_RESULTS_DIR.glob("**/*.json"))


@pytest.fixture(params=_collect_result_files(), ids=lambda p: p.stem)
def injection_result(request) -> dict:
    """Load one injection result JSON file.

    Parametrized over all result files present at collection time.
    When ``results/`` is empty the fixture provides no parameters and all
    tests that depend on it are skipped.
    """
    return json.loads(request.param.read_text(encoding="utf-8"))


@pytest.fixture
def log_dir(injection_result: dict) -> Path:
    """Return the per-config results subdirectory for the current test case.

    ``run_injection_suite.py`` writes ``attack_log.ndjson`` and
    ``security_events.ndjson`` into ``results/{mas_id}/``.  This fixture
    points to that directory so structural tests can assert log file existence.
    """
    mas_id = injection_result["mas_id"]
    return _RESULTS_DIR / mas_id
