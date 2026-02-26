"""Data models for the AETHER Attack Injection Framework.

Defines the core research data structures:

- ``InjectionPhase`` / ``AttackType`` — enum taxonomy
- ``AgentObservation`` — single agent's encounter with an injected payload
- ``AttackResult`` — full attack record written to the NDJSON research log

``resistant_agents`` is typed as ``set[str]`` on the Python side.  Pydantic v2
serialises it as a JSON array when ``model_dump(mode="json")`` is called, which
is the format used by the log file.
"""

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InjectionPhase(str, Enum):  # pylint: disable=too-few-public-methods
    """When the payload is delivered relative to graph execution."""

    PRE_EXECUTION = "pre_execution"
    MID_EXECUTION = "mid_execution"


class AttackType(str, Enum):  # pylint: disable=too-few-public-methods
    """Taxonomy of adversarial payload types."""

    PROMPT_INJECTION = "prompt_injection"
    MEMORY_POISONING = "memory_poisoning"
    AGENT_IMPERSONATION = "agent_impersonation"
    BIAS_INHERITANCE = "bias_inheritance"


class AgentObservation(BaseModel):
    """One agent's encounter with the injected payload during graph traversal.

    Attributes:
        agent_id: The agent's unique identifier.
        role: The agent's configured role string.
        received_payload: True if the payload (or a >20-char substring) was
            detected in the agent's input state.
        influenced: True if the payload or known compliance markers appeared
            in the agent's output state.
        output_excerpt: First 500 characters of the agent's output, or None
            if the output was empty / unavailable.
        resisted: True when ``received_payload`` is True and ``influenced``
            is False — the agent received but filtered the payload.
    """

    agent_id: str
    role: str
    received_payload: bool
    influenced: bool
    output_excerpt: Optional[str] = None
    resisted: bool

    @model_validator(mode="after")
    def _enforce_resisted(self) -> "AgentObservation":
        """Enforce resisted = received_payload and not influenced."""
        self.resisted = self.received_payload and not self.influenced
        return self


class AttackResult(BaseModel):
    """Complete record of a single attack injection and its propagation.

    This is the primary research deliverable.  Every completed injection
    is serialised via ``model_dump(mode="json")`` and appended as a single
    line to the NDJSON attack log.

    Attributes:
        attack_id: UUID4 string uniquely identifying this injection event.
        mas_id: Identifier of the MAS that was attacked.
        target_agent_id: The agent that received the initial payload.
        attack_type: The category of adversarial payload used.
        injection_phase: Whether the payload was delivered before or during
            graph execution.
        payload: The raw adversarial string that was injected.
        injected_at: UTC timestamp when ``inject_attack()`` was called.
        completed_at: UTC timestamp when propagation tracking finished.
            ``None`` for fire-and-forget (``blocking=False``) calls that
            have not yet resolved.
        propagation_path: Ordered list of agent IDs that processed the
            payload (by observation sequence, not insertion order).
        influenced_agents: Agent IDs whose output was altered by the payload.
        resistant_agents: Agent IDs that received but resisted the payload.
            Stored as a set; serialised as a JSON array in the log file.
        success: True when execution completed without an unhandled error.
        error: Error message if execution failed; None otherwise.
    """

    attack_id: str = Field(..., description="UUID4 unique attack identifier")
    run_id: Optional[str] = None
    mas_id: str
    target_agent_id: str
    attack_type: AttackType
    injection_phase: InjectionPhase
    payload: str
    injected_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

    propagation_path: list[str] = Field(default_factory=list)
    influenced_agents: list[str] = Field(default_factory=list)
    resistant_agents: set[str] = Field(default_factory=set)

    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)
