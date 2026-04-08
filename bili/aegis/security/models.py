"""Data models for the AETHER Security Event Detection & Logging system.

``SecurityEvent`` is the primary research record emitted by
``SecurityEventDetector``.  Every detected event is serialised via
``model_dump(mode="json")`` and appended as a single line to the NDJSON
security log.

``run_id`` links security events back to the ``MASExecutionResult`` that
produced them (and to the ``AttackResult`` in the attack log) so that all
three log files can be joined on a single key.
"""

import datetime
import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SecurityEventType(str, Enum):  # pylint: disable=too-few-public-methods
    """Taxonomy of security event types produced by the detector."""

    ATTACK_DETECTED = "attack_detected"
    AGENT_COMPROMISED = "agent_compromised"
    AGENT_RESISTED = "agent_resisted"
    PAYLOAD_PROPAGATED = "payload_propagated"
    REPEATED_TARGET = "repeated_target"


class SecurityEvent(BaseModel):
    """A single security event detected during or after a MAS execution.

    Attributes:
        event_id: UUID4 string uniquely identifying this event.
        run_id: Optional run_id linking to ``MASExecutionResult.run_id``
            and ``AttackResult.run_id`` for cross-log correlation.
        event_type: The category of security event.
        severity: Severity level — one of ``"low"``, ``"medium"``, or
            ``"high"``.  Validated by Pydantic via ``Literal`` typing.
        detected_at: UTC timestamp when the event was detected.  Defaults
            to ``datetime.datetime.now(utc)`` at construction time.
        mas_id: Identifier of the MAS involved in the attack.
        attack_id: Optional link back to ``AttackResult.attack_id``.
        target_agent_id: The agent that was the initial injection target.
        affected_agent_id: For per-agent events (AGENT_COMPROMISED /
            AGENT_RESISTED), the specific agent affected.  ``None`` for
            summary-level events.
        attack_type: The ``AttackType`` value string from the source
            ``AttackResult`` (e.g. ``"prompt_injection"``).
        success: Whether the underlying attack succeeded.
        details: Flexible dict for additional event-specific metadata
            (e.g. ``propagation_path``, ``prior_attack_count``).
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: Optional[str] = None
    event_type: SecurityEventType
    severity: Literal["low", "medium", "high"]
    detected_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    mas_id: str
    attack_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    affected_agent_id: Optional[str] = None
    attack_type: Optional[str] = None
    success: Optional[bool] = None
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)
