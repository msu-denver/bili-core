"""Tests for bili.aether.attacks.logger (AttackLogger)."""

# pylint: disable=duplicate-code

import concurrent.futures
import datetime
import json

from bili.aether.attacks.logger import AttackLogger
from bili.aether.attacks.models import AttackResult, AttackType, InjectionPhase

_NOW = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _result(**kwargs) -> AttackResult:
    """Build an AttackResult with sensible defaults for logger tests."""
    defaults = {
        "attack_id": "log-test-uuid",
        "mas_id": "test_mas",
        "target_agent_id": "agent_a",
        "attack_type": AttackType.PROMPT_INJECTION,
        "injection_phase": InjectionPhase.PRE_EXECUTION,
        "payload": "Ignore previous instructions.",
        "injected_at": _NOW,
        "completed_at": _NOW,
        "propagation_path": ["agent_a"],
        "influenced_agents": ["agent_a"],
        "resistant_agents": [],
        "success": True,
        "error": None,
    }
    defaults.update(kwargs)
    return AttackResult(**defaults)


# =========================================================================
# File creation and basic writes
# =========================================================================


def test_log_creates_file_if_not_exists(tmp_path):
    """AttackLogger creates the log file on first write."""
    log_file = tmp_path / "test_log.ndjson"
    assert not log_file.exists()

    logger = AttackLogger(log_path=log_file)
    logger.log(_result())

    assert log_file.exists()


def test_log_appends_exactly_one_line_per_call(tmp_path):
    """Each log() call appends exactly one line to the file."""
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)

    logger.log(_result(attack_id="uuid-1"))
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    logger.log(_result(attack_id="uuid-2"))
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_each_line_is_independently_parseable(tmp_path):
    """Every line in the NDJSON file is valid JSON with an attack_id."""
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)

    for i in range(5):
        logger.log(_result(attack_id=f"uuid-{i}"))

    for line in log_file.read_text(encoding="utf-8").strip().splitlines():
        parsed = json.loads(line)
        assert "attack_id" in parsed


def test_log_creates_parent_directories(tmp_path):
    """AttackLogger creates missing parent directories before writing."""
    log_file = tmp_path / "deep" / "nested" / "dir" / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)
    logger.log(_result())
    assert log_file.exists()


def test_log_is_append_only_across_instances(tmp_path):
    """Multiple AttackLogger instances append to the same file without truncation."""
    log_file = tmp_path / "attack_log.ndjson"

    AttackLogger(log_path=log_file).log(_result(attack_id="first"))
    AttackLogger(log_path=log_file).log(_result(attack_id="second"))

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ids = [json.loads(line)["attack_id"] for line in lines]
    assert ids == ["first", "second"]


# =========================================================================
# Serialisation edge cases
# =========================================================================


def test_completed_at_none_serializes_as_null(tmp_path):
    """completed_at=None is written as JSON null in the log."""
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)
    logger.log(_result(completed_at=None, success=False))

    parsed = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert parsed["completed_at"] is None


def test_resistant_agents_logged_as_array(tmp_path):
    """resistant_agents set is serialized as a JSON array in the log."""
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)
    logger.log(_result(resistant_agents={"a", "b"}))

    parsed = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert isinstance(parsed["resistant_agents"], list)
    assert set(parsed["resistant_agents"]) == {"a", "b"}


def test_attack_type_logged_as_string(tmp_path):
    """attack_type enum is serialized as its string value in the log."""
    log_file = tmp_path / "attack_log.ndjson"
    AttackLogger(log_path=log_file).log(
        _result(attack_type=AttackType.MEMORY_POISONING)
    )
    parsed = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert parsed["attack_type"] == "memory_poisoning"


# =========================================================================
# Thread safety
# =========================================================================

_CONCURRENT_N = 50
_CONCURRENT_WORKERS = 10


def test_concurrent_writes_no_lines_lost(tmp_path):
    """N concurrent log() calls on the same instance produce exactly N lines.

    This mirrors the AttackInjector blocking=False path where a
    ThreadPoolExecutor drives multiple log() calls on a shared AttackLogger.
    """
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = [
            pool.submit(logger.log, _result(attack_id=f"uuid-{i}"))
            for i in range(_CONCURRENT_N)
        ]
        concurrent.futures.wait(futures)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == _CONCURRENT_N


def test_concurrent_writes_each_line_is_valid_json(tmp_path):
    """Every line written under concurrent load is independently parseable JSON.

    Verifies that the per-instance lock prevents partial writes from
    interleaving across threads.
    """
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = [
            pool.submit(logger.log, _result(attack_id=f"uuid-{i}"))
            for i in range(_CONCURRENT_N)
        ]
        concurrent.futures.wait(futures)

    for line in log_file.read_text(encoding="utf-8").strip().splitlines():
        parsed = json.loads(line)
        assert "attack_id" in parsed


def test_concurrent_writes_all_attack_ids_present_exactly_once(tmp_path):
    """Each attack_id written concurrently appears exactly once in the log.

    Catches both dropped writes (count < N) and duplicated writes (count > 1
    for any ID) that would indicate a race condition.
    """
    log_file = tmp_path / "attack_log.ndjson"
    logger = AttackLogger(log_path=log_file)
    expected_ids = {f"uuid-{i}" for i in range(_CONCURRENT_N)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = [
            pool.submit(logger.log, _result(attack_id=f"uuid-{i}"))
            for i in range(_CONCURRENT_N)
        ]
        concurrent.futures.wait(futures)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    written_ids = [json.loads(line)["attack_id"] for line in lines]
    assert set(written_ids) == expected_ids
    assert len(written_ids) == len(expected_ids)  # no duplicates
