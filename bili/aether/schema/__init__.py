"""
AETHER Schema Module

Pydantic models for defining multi-agent system configurations,
agent specifications, communication channel definitions, and
agent pipeline sub-graph specifications.

Design Philosophy:
- Agent roles and capabilities are free-form strings (not enums)
- Use presets for convenience without restrictions
- Structural enums (WorkflowType, OutputFormat, etc.) are still used
- Pipeline support is optional and backwards compatible

Modules:
- agent_spec.py: Domain-agnostic agent configuration schema
- agent_presets.py: Registry-based preset system for common patterns
- mas_config.py: Multi-agent system configuration schema
- pipeline_spec.py: Internal pipeline sub-graph specification
- enums.py: Structural enumerations (workflow types, protocols, etc.)
"""

# Agent presets
from .agent_presets import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    list_presets,
    register_preset,
)

# Agent specification
from .agent_spec import AgentSpec

# Structural enums only (no domain-specific enums)
from .enums import CommunicationProtocol, OutputFormat, WorkflowType

# MAS configuration
from .mas_config import Channel, MASConfig, WorkflowEdge

# Pipeline specification
from .pipeline_spec import PipelineEdgeSpec, PipelineNodeSpec, PipelineSpec

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
    # Pipeline
    "PipelineSpec",
    "PipelineNodeSpec",
    "PipelineEdgeSpec",
]
