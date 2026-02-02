"""
AETHER schema module.

Provides dataclasses and types for multi-agent system configuration.
"""

from .agent_presets import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    register_preset,
)
from .agent_spec import AgentSpec
from .enums import CommunicationProtocol, OutputFormat, WorkflowType
from .mas_config import Channel, MASConfig, WorkflowEdge

__all__ = [
    "AgentSpec",
    "MASConfig",
    "Channel",
    "WorkflowEdge",
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    "AGENT_PRESETS",
    "get_preset",
    "create_agent_from_preset",
    "register_preset",
]
