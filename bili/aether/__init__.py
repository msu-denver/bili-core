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

Submodules are loaded lazily (PEP 562) so that importing bili.aether does not
eagerly pull in compiler, runtime, or validation dependencies.

Author: MSU Denver Cybersecurity Research
License: MIT
"""

__version__ = "0.0.7"
__author__ = "MSU Denver Cybersecurity Research, MonRos3"

# Lazy attribute mapping: name -> (relative_module, attribute_name)
_LAZY_IMPORTS = {
    # Compiler
    "CompiledMAS": (".compiler", "CompiledMAS"),
    "compile_mas": (".compiler", "compile_mas"),
    # Config loaders
    "load_mas_from_dict": (".config", "load_mas_from_dict"),
    "load_mas_from_yaml": (".config", "load_mas_from_yaml"),
    # Runtime / Execution
    "AgentExecutionResult": (".runtime", "AgentExecutionResult"),
    "MASExecutionResult": (".runtime", "MASExecutionResult"),
    "MASExecutor": (".runtime", "MASExecutor"),
    "execute_mas": (".runtime", "execute_mas"),
    # Schema
    "AGENT_PRESETS": (".schema", "AGENT_PRESETS"),
    "AgentSpec": (".schema", "AgentSpec"),
    "Channel": (".schema", "Channel"),
    "CommunicationProtocol": (".schema", "CommunicationProtocol"),
    "MASConfig": (".schema", "MASConfig"),
    "OutputFormat": (".schema", "OutputFormat"),
    "WorkflowEdge": (".schema", "WorkflowEdge"),
    "WorkflowType": (".schema", "WorkflowType"),
    "create_agent_from_preset": (".schema", "create_agent_from_preset"),
    "get_preset": (".schema", "get_preset"),
    "list_presets": (".schema", "list_presets"),
    "register_preset": (".schema", "register_preset"),
    # Validation
    "MASValidator": (".validation", "MASValidator"),
    "ValidationResult": (".validation", "ValidationResult"),
    "validate_mas": (".validation", "validate_mas"),
}

# Submodule names that can be accessed as attributes
_LAZY_SUBMODULES = {
    "compiler",
    "config",
    "integration",
    "runtime",
    "schema",
    "ui",
    "validation",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return (
        list(_LAZY_IMPORTS.keys())
        + list(_LAZY_SUBMODULES)
        + [
            "__version__",
            "__author__",
        ]
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
    # Validation
    "MASValidator",
    "ValidationResult",
    "validate_mas",
    # Compiler
    "CompiledMAS",
    "compile_mas",
    # Runtime / Execution
    "AgentExecutionResult",
    "MASExecutionResult",
    "MASExecutor",
    "execute_mas",
]
