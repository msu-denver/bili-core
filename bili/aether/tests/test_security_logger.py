"""Tests for bili.aether.security.logger."""

import json
import threading

from bili.aether.security.logger import SecurityEventLogger
from bili.aether.security.models import SecurityEvent, SecurityEventType


def _event(
    event_type: SecurityEventType = SecurityEventType.ATTACK_DETECTED,
    severity: str = "high",
    mas_id: str = "test_mas",
    **kwargs,
) -> SecurityEvent:
    """Build a minimal SecurityEvent for logging tests."""
    defaults = {
        "event_type": event_type,
        "severity": severity,
        "mas_id": mas_id,
        "attack_id": "attack-1234",
        "target_agent_id": "agent_a",
        "attack_type": "prompt_injection",
        "success": True,
    }
    defaults.update(kwargs)
    return SecurityEvent(**defaults)


# =========================================================================
# File creation and basic append
# =========================================================================


def test_logger_creates_file_on_first_write(tmp_path):
    """Logging an event creates the NDJSON file if it does not exist."""
    log_path = tmp_path / "events.ndjson"
    assert not log_path.exists()
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event())
    assert log_path.exists()


def test_logger_creates_parent_directories(tmp_path):
    """The logger creates missing parent directories before the first write."""
    log_path = tmp_path / "deep" / "nested" / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event())
    assert log_path.exists()


def test_logger_appends_one_line_per_event(tmp_path):
    """Each call to log() appends exactly one non-empty line."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event())
    logger.log(_event(severity="low"))
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_logger_does_not_truncate_on_second_write(tmp_path):
    """Existing content is preserved when a second event is appended."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    e1 = _event(attack_id="first-attack")
    e2 = _event(attack_id="second-attack")
    logger.log(e1)
    logger.log(e2)

    content = log_path.read_text(encoding="utf-8")
    assert "first-attack" in content
    assert "second-attack" in content


# =========================================================================
# NDJSON format correctness
# =========================================================================


def test_logged_line_is_valid_json(tmp_path):
    """Each line written to the log file is independently parseable JSON."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    event = _event()
    logger.log(event)

    line = log_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["mas_id"] == "test_mas"
    assert parsed["attack_id"] == "attack-1234"


def test_logged_event_id_matches_source_event(tmp_path):
    """The event_id in the log matches the original SecurityEvent."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    event = _event()
    logger.log(event)

    parsed = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert parsed["event_id"] == event.event_id


def test_logged_event_type_is_string_value(tmp_path):
    """event_type is serialised as its string value, not an enum repr."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event(event_type=SecurityEventType.AGENT_RESISTED))

    parsed = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert parsed["event_type"] == "agent_resisted"


# =========================================================================
# export_json()
# =========================================================================


def test_export_json_returns_valid_json_array(tmp_path):
    """export_json() returns a string that parses as a JSON array."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event())
    logger.log(_event(severity="medium"))

    result = logger.export_json()
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_export_json_empty_when_no_events(tmp_path):
    """export_json() returns '[]' when the log file does not exist."""
    log_path = tmp_path / "no_events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    result = logger.export_json()
    assert json.loads(result) == []


def test_export_json_writes_to_path_when_provided(tmp_path):
    """export_json(path=...) writes the JSON array to the given path."""
    log_path = tmp_path / "events.ndjson"
    export_path = tmp_path / "exported" / "all_events.json"
    logger = SecurityEventLogger(log_path=log_path)
    logger.log(_event())

    result = logger.export_json(path=export_path)
    assert export_path.exists()
    on_disk = json.loads(export_path.read_text(encoding="utf-8"))
    in_memory = json.loads(result)
    assert on_disk == in_memory
    assert len(on_disk) == 1


def test_export_json_preserves_all_fields(tmp_path):
    """export_json() array entries contain the expected SecurityEvent fields."""
    log_path = tmp_path / "events.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    event = _event(run_id="run-xyz", severity="medium")
    logger.log(event)

    parsed = json.loads(logger.export_json())
    entry = parsed[0]
    assert entry["event_id"] == event.event_id
    assert entry["run_id"] == "run-xyz"
    assert entry["severity"] == "medium"
    assert entry["mas_id"] == "test_mas"


# =========================================================================
# Thread-safety
# =========================================================================


def test_logger_thread_safety(tmp_path):
    """Concurrent log() calls from multiple threads produce uncorrupted NDJSON."""
    log_path = tmp_path / "concurrent.ndjson"
    logger = SecurityEventLogger(log_path=log_path)
    n_threads = 20

    threads = [
        threading.Thread(target=logger.log, args=(_event(attack_id=f"attack-{i}"),))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == n_threads
    # Every line must be valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "event_id" in obj
