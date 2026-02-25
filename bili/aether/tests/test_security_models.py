"""Tests for bili.aether.security.models."""

import datetime
import json

import pytest

from bili.aether.security.models import SecurityEvent, SecurityEventType

_MAS_ID = "test_mas"


def _event(**kwargs) -> SecurityEvent:
    """Build a SecurityEvent with sensible defaults."""
    defaults = {
        "event_type": SecurityEventType.ATTACK_DETECTED,
        "severity": "high",
        "mas_id": _MAS_ID,
        "attack_id": "attack-uuid-1234",
        "target_agent_id": "agent_a",
        "attack_type": "prompt_injection",
        "success": True,
    }
    defaults.update(kwargs)
    return SecurityEvent(**defaults)


# =========================================================================
# SecurityEventType enum tests
# =========================================================================


def test_security_event_type_attack_detected_value():
    """ATTACK_DETECTED enum member has the correct string value."""
    assert SecurityEventType.ATTACK_DETECTED.value == "attack_detected"


def test_security_event_type_agent_compromised_value():
    """AGENT_COMPROMISED enum member has the correct string value."""
    assert SecurityEventType.AGENT_COMPROMISED.value == "agent_compromised"


def test_security_event_type_agent_resisted_value():
    """AGENT_RESISTED enum member has the correct string value."""
    assert SecurityEventType.AGENT_RESISTED.value == "agent_resisted"


def test_security_event_type_payload_propagated_value():
    """PAYLOAD_PROPAGATED enum member has the correct string value."""
    assert SecurityEventType.PAYLOAD_PROPAGATED.value == "payload_propagated"


def test_security_event_type_rejects_unknown():
    """SecurityEventType raises ValueError for unknown values."""
    with pytest.raises(ValueError):
        SecurityEventType("unknown_type")


# =========================================================================
# SecurityEvent field defaults and auto-generation
# =========================================================================


def test_security_event_event_id_is_generated():
    """event_id is auto-populated with a non-empty UUID string."""
    event = _event()
    assert isinstance(event.event_id, str)
    assert len(event.event_id) == 36  # UUID4 canonical form


def test_security_event_event_ids_are_unique():
    """Two SecurityEvent instances get distinct event_id values."""
    e1 = _event()
    e2 = _event()
    assert e1.event_id != e2.event_id


def test_security_event_detected_at_is_utc_datetime():
    """detected_at defaults to an aware UTC datetime."""
    event = _event()
    assert isinstance(event.detected_at, datetime.datetime)
    assert event.detected_at.tzinfo is not None


def test_security_event_run_id_defaults_to_none():
    """run_id is None when not supplied."""
    event = _event()
    assert event.run_id is None


def test_security_event_run_id_can_be_set():
    """run_id is stored when explicitly provided."""
    event = _event(run_id="run-uuid-5678")
    assert event.run_id == "run-uuid-5678"


def test_security_event_affected_agent_id_defaults_to_none():
    """affected_agent_id is None for summary-level events by default."""
    event = _event()
    assert event.affected_agent_id is None


def test_security_event_details_defaults_to_empty_dict():
    """details defaults to an empty dict."""
    event = _event()
    assert event.details == {}


# =========================================================================
# SecurityEvent Literal severity validation
# =========================================================================


def test_security_event_severity_low_accepted():
    """Severity 'low' is accepted by Pydantic."""
    event = _event(severity="low")
    assert event.severity == "low"


def test_security_event_severity_medium_accepted():
    """Severity 'medium' is accepted by Pydantic."""
    event = _event(severity="medium")
    assert event.severity == "medium"


def test_security_event_severity_high_accepted():
    """Severity 'high' is accepted by Pydantic."""
    event = _event(severity="high")
    assert event.severity == "high"


def test_security_event_severity_invalid_rejected():
    """Severity values outside the Literal union raise a ValidationError."""
    with pytest.raises(Exception):  # pydantic ValidationError
        _event(severity="critical")


# =========================================================================
# SecurityEvent JSON serialisation
# =========================================================================


def test_security_event_serializes_to_valid_json():
    """model_dump(mode='json') produces a JSON-serialisable dict."""
    event = _event()
    dumped = event.model_dump(mode="json")
    line = json.dumps(dumped)
    assert isinstance(line, str)
    parsed = json.loads(line)
    assert parsed["mas_id"] == _MAS_ID
    assert parsed["attack_id"] == "attack-uuid-1234"


def test_security_event_use_enum_values_in_dump():
    """event_type serialises as a string value, not an enum object."""
    event = _event(event_type=SecurityEventType.AGENT_COMPROMISED)
    dumped = event.model_dump(mode="json")
    assert dumped["event_type"] == "agent_compromised"


def test_security_event_detected_at_serializes_as_string():
    """detected_at serialises as a datetime string (not a raw datetime object)."""
    event = _event()
    dumped = event.model_dump(mode="json")
    assert isinstance(dumped["detected_at"], str)


def test_security_event_run_id_serializes_as_null_when_none():
    """run_id serialises as JSON null when not set."""
    event = _event()
    line = json.dumps(event.model_dump(mode="json"))
    parsed = json.loads(line)
    assert parsed["run_id"] is None


def test_security_event_run_id_serializes_when_set():
    """run_id serialises as the provided string."""
    event = _event(run_id="my-run-id")
    dumped = event.model_dump(mode="json")
    assert dumped["run_id"] == "my-run-id"


def test_security_event_details_dict_serializes_correctly():
    """details dict is included intact in the serialised output."""
    details = {"propagation_path": ["agent_a", "agent_b"], "spread_to": ["agent_b"]}
    event = _event(details=details)
    dumped = event.model_dump(mode="json")
    assert dumped["details"]["propagation_path"] == ["agent_a", "agent_b"]
    assert dumped["details"]["spread_to"] == ["agent_b"]
