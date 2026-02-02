"""
Enumerations for AETHER schema.

This module defines STRUCTURAL enum types only - not domain-specific ones.

Design Decision:
- Agent roles are FREE-FORM STRINGS, not enums (allows any domain)
- Agent capabilities are FREE-FORM STRINGS (extensible without code changes)
- Only workflow structure enums are defined here

This follows bili-core's registry-based patterns where extensibility
is achieved through configuration, not code modification.
"""

from enum import Enum


class OutputFormat(str, Enum):
    """
    Agent output format types.

    Determines how agent responses are structured.
    """

    TEXT = "text"  # Plain text response
    JSON = "json"  # JSON object (flexible schema)
    STRUCTURED = "structured"  # JSON with enforced schema validation


class WorkflowType(str, Enum):
    """
    MAS workflow execution patterns.

    These are structural patterns that define HOW agents interact,
    not WHAT domain they operate in.

    Patterns:
    - SEQUENTIAL: Linear chain (A -> B -> C)
    - PARALLEL: Concurrent execution (A, B, C run simultaneously)
    - HIERARCHICAL: Tree structure with tiers and voting
    - SUPERVISOR: Hub-and-spoke with dynamic routing
    - CONSENSUS: Network/deliberation until agreement
    - DELIBERATIVE: Custom workflow with conditional edges
    - CUSTOM: Fully custom graph definition
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    SUPERVISOR = "supervisor"
    CONSENSUS = "consensus"
    DELIBERATIVE = "deliberative"
    CUSTOM = "custom"


class CommunicationProtocol(str, Enum):
    """
    Communication protocols for agent channels.

    Defines how agents exchange messages (structural concern).
    """

    DIRECT = "direct"  # Point-to-point: A -> B
    BROADCAST = "broadcast"  # One-to-many: A -> All
    REQUEST_RESPONSE = "request_response"  # Bidirectional: A <-> B
    PUBSUB = "pubsub"  # Publish-subscribe pattern
    COMPETITIVE = "competitive"  # Adversarial/debate (both parties argue)
    CONSENSUS = "consensus"  # Peer deliberation toward agreement
