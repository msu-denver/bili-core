"""
AETHER Schema Module

This module contains Pydantic models and data structures for defining
multi-agent system configurations, agent specifications, and communication
channel definitions.

Modules:
- agent_spec.py: Agent configuration schema (Task 2)
- mas_config.py: Multi-agent system configuration schema (Task 3)
- examples.py: Example agent and MAS configurations

This module will be fully implemented in Task 2.
"""

# Placeholder for Task 2 imports
# from .agent_spec import AgentSpec, AgentRole, OutputFormat, AgentCapability
# from .examples import CONTENT_REVIEWER_SPEC, POLICY_EXPERT_SPEC

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
