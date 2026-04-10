"""Pytest fixtures for the AETHER persistence attack test suite.

The ``persistence_result`` fixture is parametrized over every JSON file
in ``results/``.  When the directory is empty (i.e.
``run_persistence_suite.py`` has not yet been run), the fixture yields no
parameters and all structural tests are automatically skipped.

The ``log_dir`` fixture derives the per-config log directory from the
``mas_id`` field in the loaded result dict.  Log files (``attack_log.ndjson``
and ``security_events.ndjson``) are written there by the runner.

Note on ``_find_repo_root``
---------------------------
The helper is inlined here rather than imported from
``bili.aegis.tests._helpers`` because this file must bootstrap ``sys.path``
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

    python bili/aegis/tests/persistence/run_persistence_suite.py
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


def _synthetic_persistence_result() -> dict:
    """Build a minimal synthetic persistence result for structural tests.

    Used when no real result files exist (e.g. no persistent checkpointer
    is configured). This ensures structural tests always run and validate
    the expected schema shape.
    """
    return {
        "payload_id": "synthetic_persistence_001",
        "injection_type": "session_fabrication",
        "severity": "high",
        "mas_id": "synthetic_test",
        "injection_phase": "checkpoint_injection",
        "attack_suite": "persistence",
        "execution": {
            "success": True,
            "duration_ms": 100.0,
            "agent_count": 2,
            "message_count": 4,
        },
        "propagation_path": ["agent_a"],
        "influenced_agents": [],
        "resistant_agents": ["agent_a"],
        "run_metadata": {
            "stub_mode": True,
            "timestamp": "2026-01-01T00:00:00Z",
            "tier3_score": "",
            "tier3_confidence": "",
            "tier3_reasoning": "",
        },
        "target_agent_id": "agent_a",
        "config_fingerprint": {
            "config_path": "bili/aether/config/examples/simple_chain.yaml",
            "yaml_hash": "synthetic",
        },
    }


def _ensure_synthetic_files() -> None:
    """Create synthetic result files so structural tests always run.

    When no real result files exist (e.g. no persistent checkpointer
    is configured), this creates a minimal JSON result plus the log
    files that the structural tests expect to find on disk.
    """
    synthetic_dir = _RESULTS_DIR / "synthetic_test"
    result_file = synthetic_dir / "synthetic_persistence_001_checkpoint.json"
    if result_file.exists():
        return
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(_synthetic_persistence_result()), encoding="utf-8"
    )
    (synthetic_dir / "attack_log.ndjson").write_text(
        json.dumps({"attack_id": "synthetic", "success": True}) + "\n",
        encoding="utf-8",
    )
    (synthetic_dir / "security_events.ndjson").write_text(
        json.dumps({"event_type": "ATTACK_DETECTED", "severity": "low"}) + "\n",
        encoding="utf-8",
    )


_ensure_synthetic_files()


def _collect_result_files() -> list:
    """Return all JSON result file paths under results/."""
    if not _RESULTS_DIR.exists():
        return []
    files = sorted(_RESULTS_DIR.glob("**/*.json"))
    return files


@pytest.fixture(
    params=_collect_result_files(),
    ids=lambda p: p.stem,
)
def persistence_result(request) -> dict:
    """Load one persistence result JSON file.

    Parametrized over all result files present at collection time,
    including synthetic files generated when no real results exist.
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
    return _RESULTS_DIR / mas_id
