"""
Enumerations for AETHER schema.

This module defines all enum types used across agent and MAS configurations.
"""

from enum import Enum


class AgentRole(str, Enum):
    """
    Agent roles for content moderation MAS.

    Includes bili-core standard roles and new roles for your 5 MAS patterns.
    """

    # bili-core standard roles (Task 7 integration)
    CONTENT_REVIEWER = "content_reviewer"
    POLICY_EXPERT = "policy_expert"
    JUDGE = "judge"
    APPEALS_SPECIALIST = "appeals_specialist"
    COMMUNITY_MANAGER = "community_manager"

    # NEW: Adversarial/debate roles (MAS #2 - Hierarchical)
    ALLOW_ADVOCATE = "allow_advocate"
    BLOCK_ADVOCATE = "block_advocate"

    # Catch-all for custom roles
    CUSTOM = "custom"


class AgentCapability(str, Enum):
    """
    Agent capabilities (features an agent can use).

    These correspond to bili-core features and LangGraph capabilities.
    """

    # bili-core integration
    RAG_RETRIEVAL = "rag_retrieval"
    POLICY_LOOKUP = "policy_lookup"
    MEMORY_ACCESS = "memory_access"

    # LangGraph features
    CHECKPOINT_PERSISTENCE = "checkpoint_persistence"
    TOOL_CALLING = "tool_calling"

    # Communication
    INTER_AGENT_COMMUNICATION = "inter_agent_communication"


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

    Maps to your 5 MAS configurations:
    - SEQUENTIAL: MAS #1 (Chain)
    - HIERARCHICAL: MAS #2 (Tree/voting)
    - SUPERVISOR: MAS #3 (Hub-and-spoke)
    - CONSENSUS: MAS #4 (Network/deliberation)
    - DELIBERATIVE: MAS #5 (Custom with escalation)
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

    DIRECT = "direct"  # Point-to-point: A → B
    BROADCAST = "broadcast"  # One-to-many: A → All
    REQUEST_RESPONSE = "request_response"  # Bidirectional: A ↔ B
    PUBSUB = "pubsub"  # Publish-subscribe pattern

    # NEW: For your MAS patterns
    COMPETITIVE = "competitive"  # MAS #2: Adversarial debate
    CONSENSUS = "consensus"  # MAS #4: Peer deliberation
