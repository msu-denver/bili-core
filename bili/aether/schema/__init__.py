"""
AETHER Schema Module

Pydantic models for defining multi-agent system configurations,
agent specifications, and communication channel definitions.

Design Philosophy:
- Agent roles and capabilities are free-form strings (not enums)
- Use presets for convenience without restrictions
- Structural enums (WorkflowType, OutputFormat, etc.) are still used

Modules:
- agent_spec.py: Domain-agnostic agent configuration schema
- agent_presets.py: Registry-based preset system for common patterns
- mas_config.py: Multi-agent system configuration schema
- enums.py: Structural enumerations (workflow types, protocols, etc.)
"""

# Agent specification
from .agent_spec import AgentSpec

# Agent presets
from .agent_presets import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    list_presets,
    register_preset,
)

# Structural enums only (no domain-specific enums)
from .enums import (
    CommunicationProtocol,
    OutputFormat,
    WorkflowType,
)

# MAS configuration
from .mas_config import Channel, MASConfig, WorkflowEdge

__all__ = [
    # Structural enums
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    # Agent
    "AgentSpec",
    # Presets
    "AGENT_PRESETS",
    "create_agent_from_preset",
    "get_preset",
    "list_presets",
    "register_preset",
    # MAS
    "MASConfig",
    "Channel",
    "WorkflowEdge",
]
