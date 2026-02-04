"""
Enumerations for AETHER schema.

This module defines structural enum types used across MAS configurations.
Note: Agent roles and capabilities are intentionally NOT enums - they are
free-form strings to support any domain without code changes.
"""

from enum import Enum


class OutputFormat(str, Enum):
    """
    Agent output format types.

    Determines how agent responses are structured.
    """

    TEXT = "text"  # Plain text response
    JSON = "json"  # JSON object (flexible)
    STRUCTURED = "structured"  # JSON with enforced schema


class WorkflowType(str, Enum):
    """
    MAS workflow execution patterns.

    Maps to common multi-agent patterns:
    - SEQUENTIAL: Chain pattern (A -> B -> C)
    - HIERARCHICAL: Tree/voting pattern with tiers
    - SUPERVISOR: Hub-and-spoke pattern
    - CONSENSUS: Network/deliberation pattern
    - DELIBERATIVE: Custom with escalation
    - PARALLEL: All agents execute simultaneously
    - CUSTOM: User-defined workflow with explicit edges
    """

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    SUPERVISOR = "supervisor"
    CONSENSUS = "consensus"
    DELIBERATIVE = "deliberative"
    PARALLEL = "parallel"
    CUSTOM = "custom"


class CommunicationProtocol(str, Enum):
    """
    Communication protocols for agent channels.

    Defines how agents exchange messages.
    """

    DIRECT = "direct"  # Point-to-point: A -> B
    BROADCAST = "broadcast"  # One-to-many: A -> All
    REQUEST_RESPONSE = "request_response"  # Bidirectional: A <-> B
    PUBSUB = "pubsub"  # Publish-subscribe pattern
    COMPETITIVE = "competitive"  # Adversarial debate pattern
    CONSENSUS = "consensus"  # Peer deliberation pattern
