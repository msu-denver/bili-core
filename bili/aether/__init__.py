"""
AETHER: Agent Ecosystems for Testing, Hardening, Evaluation, and Research

AETHER is a multi-agent system (MAS) security testing framework built as an
extension to bili-core. It enables declarative configuration of multi-agent
systems for testing security vulnerabilities in LLM-based agent architectures.

Core Capabilities:
- Multi-agent system configuration and orchestration
- Agent-to-agent communication protocols
- Security attack vector testing (prompt injection, memory poisoning, etc.)
- Transparent agent communication logging
- Integration with bili-core's LLM, tool, and state management features

AETHER extends bili-core's single-agent RAG workflows to support complex
multi-agent security testing scenarios while inheriting all bili-core
capabilities (authentication, checkpointing, tool access, LLM configuration).

Author: MSU Denver Cybersecurity Research
License: MIT
"""

__version__ = "0.0.7"
__author__ = "MSU Denver Cybersecurity Research, MonRos3"

from .config import load_mas_from_dict, load_mas_from_yaml
from .schema import (
    AgentCapability,
    AgentRole,
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    OutputFormat,
    WorkflowEdge,
    WorkflowType,
)
from .validation import MASValidator, ValidationResult, validate_mas

__all__ = [
    "__version__",
    "__author__",
    # Schema
    "AgentSpec",
    "MASConfig",
    "Channel",
    "WorkflowEdge",
    # Enums
    "AgentRole",
    "AgentCapability",
    "OutputFormat",
    "WorkflowType",
    "CommunicationProtocol",
    # Config loaders
    "load_mas_from_yaml",
    "load_mas_from_dict",
    # Validation
    "MASValidator",
    "ValidationResult",
    "validate_mas",
]
