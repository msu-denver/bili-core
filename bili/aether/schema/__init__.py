"""
AETHER Schema Module

Pydantic models for defining multi-agent system configurations,
agent specifications, and communication channel definitions.

Modules:
- agent_spec.py: Agent configuration schema
- mas_config.py: Multi-agent system configuration schema
- enums.py: Shared enumerations
"""

# Agent specification
from .agent_spec import AgentSpec

# Enums
from .enums import (
    AgentCapability,
    AgentRole,
    CommunicationProtocol,
    OutputFormat,
    WorkflowType,
)

# MAS configuration (Task 3, but available now)
from .mas_config import Channel, MASConfig, WorkflowEdge

__all__ = [
    # Enums
    "AgentRole",
    "AgentCapability",
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    # Agent
    "AgentSpec",
    # MAS
    "MASConfig",
    "Channel",
    "WorkflowEdge",
]
