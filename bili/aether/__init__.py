"""
AETHER: Agent Ecosystems for Testing, Hardening, Evaluation, and Research

AETHER is a multi-agent system (MAS) framework built as an extension to
bili-core. It enables declarative configuration of multi-agent systems
for any domain - research, content moderation, code review, and more.

Design Philosophy:
- Domain-agnostic: Agent roles and capabilities are free-form strings
- Preset system: Common patterns available without restrictions
- Extensible: Register custom presets at runtime
- bili-core integration: Inherit LLM, tool, and state management features

Core Capabilities:
- Multi-agent system configuration and orchestration
- Agent-to-agent communication protocols
- Multiple workflow patterns (sequential, hierarchical, supervisor, consensus)
- Transparent agent communication logging
- Full bili-core integration (auth, checkpoints, tools, LLMs)

Author: MSU Denver Cybersecurity Research
License: MIT
"""

__version__ = "0.1.0"
__author__ = "MSU Denver Cybersecurity Research, MonRos3"

from .config import load_mas_from_dict, load_mas_from_yaml
from .schema import (
    AGENT_PRESETS,
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    OutputFormat,
    WorkflowEdge,
    WorkflowType,
    create_agent_from_preset,
    get_preset,
    list_presets,
    register_preset,
)

__all__ = [
    "__version__",
    "__author__",
    # Schema
    "AgentSpec",
    "MASConfig",
    "Channel",
    "WorkflowEdge",
    # Structural enums
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    # Preset system
    "AGENT_PRESETS",
    "create_agent_from_preset",
    "get_preset",
    "list_presets",
    "register_preset",
    # Config loaders
    "load_mas_from_yaml",
    "load_mas_from_dict",
]
