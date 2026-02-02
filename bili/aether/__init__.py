"""
AETHER - Agent Execution & Task Handling for Enhanced Reasoning.

A domain-agnostic multi-agent system framework for bili-core.

This module provides:
- AgentSpec: Flexible agent specification with free-form roles
- MASConfig: Multi-agent system configuration
- AgentPresets: Registry-based preset system for common patterns
- Workflow types: Sequential, hierarchical, supervisor, consensus, etc.

Design Philosophy:
- Domain-agnostic: Roles are free-form strings, not enums
- Extensible: Presets provide convenience without restrictions
- Registry-based: Consistent with bili-core patterns (tools, nodes, middleware)

Example:
    >>> from bili.aether import AgentSpec, MASConfig, get_preset
    >>>
    >>> # Create agent from scratch
    >>> agent = AgentSpec(
    ...     agent_id="my_agent",
    ...     role="custom_analyzer",
    ...     objective="Analyze data patterns"
    ... )
    >>>
    >>> # Or use a preset as starting point
    >>> researcher = get_preset("researcher", agent_id="research_bot")
"""

from .schema.agent_presets import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    register_preset,
)
from .schema.agent_spec import AgentSpec
from .schema.enums import (
    CommunicationProtocol,
    OutputFormat,
    WorkflowType,
)
from .schema.mas_config import Channel, MASConfig, WorkflowEdge

__all__ = [
    # Core schemas
    "AgentSpec",
    "MASConfig",
    "Channel",
    "WorkflowEdge",
    # Enums (structural only, not domain-specific)
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    # Preset system
    "AGENT_PRESETS",
    "get_preset",
    "create_agent_from_preset",
    "register_preset",
]
