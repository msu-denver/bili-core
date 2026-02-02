"""
Agent presets registry.

Provides pre-configured agent templates for common use cases.
These are OPTIONAL conveniences, not restrictions.

Design Philosophy:
- Presets are starting points that can be overridden
- Users can define custom agents without using presets
- New presets can be registered at runtime without code changes
- Consistent with bili-core's registry pattern (tools, nodes, middleware)

Usage:
    >>> from bili.aether.schema.agent_presets import get_preset, register_preset
    >>>
    >>> # Use a built-in preset
    >>> agent = get_preset("researcher", agent_id="my_researcher")
    >>>
    >>> # Override preset defaults
    >>> agent = get_preset("researcher",
    ...     agent_id="custom_researcher",
    ...     temperature=0.8,
    ...     tools=["custom_tool"]
    ... )
    >>>
    >>> # Register a custom preset
    >>> register_preset("my_domain_expert", {
    ...     "role": "domain_expert",
    ...     "objective": "Provide domain expertise",
    ...     "temperature": 0.3,
    ...     "capabilities": ["domain_knowledge", "explanation"],
    ... })
"""

from typing import Any, Dict, Optional

# =============================================================================
# PRESET REGISTRY
# =============================================================================

AGENT_PRESETS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # RESEARCH & ANALYSIS
    # -------------------------------------------------------------------------
    "researcher": {
        "role": "researcher",
        "objective": "Research topics thoroughly and provide comprehensive analysis",
        "temperature": 0.5,
        "capabilities": ["web_search", "document_analysis", "summarization"],
        "tools": ["serp_api_tool"],
        "output_format": "text",
    },
    "analyst": {
        "role": "analyst",
        "objective": "Analyze data and provide insights with supporting evidence",
        "temperature": 0.3,
        "capabilities": ["data_analysis", "pattern_recognition", "reporting"],
        "tools": [],
        "output_format": "structured",
    },
    # -------------------------------------------------------------------------
    # CONTENT MODERATION (example domain)
    # -------------------------------------------------------------------------
    "content_reviewer": {
        "role": "content_reviewer",
        "objective": "Review content for policy compliance",
        "temperature": 0.0,
        "capabilities": ["text_analysis", "policy_evaluation"],
        "tools": ["policy_lookup"],
        "output_format": "json",
    },
    "moderator_judge": {
        "role": "judge",
        "objective": "Make final moderation decisions based on evidence",
        "temperature": 0.0,
        "capabilities": ["decision_making", "evidence_synthesis"],
        "tools": [],
        "output_format": "structured",
    },
    # -------------------------------------------------------------------------
    # CUSTOMER SUPPORT (example domain)
    # -------------------------------------------------------------------------
    "support_agent": {
        "role": "support_agent",
        "objective": "Assist customers with their inquiries professionally",
        "temperature": 0.7,
        "capabilities": ["conversation", "faq_lookup", "ticket_creation"],
        "tools": ["knowledge_base"],
        "output_format": "text",
    },
    "escalation_handler": {
        "role": "escalation_handler",
        "objective": "Handle escalated issues requiring special attention",
        "temperature": 0.5,
        "capabilities": ["complex_resolution", "policy_exception"],
        "tools": ["knowledge_base", "ticket_system"],
        "output_format": "text",
    },
    # -------------------------------------------------------------------------
    # CODE & DEVELOPMENT (example domain)
    # -------------------------------------------------------------------------
    "code_reviewer": {
        "role": "code_reviewer",
        "objective": "Review code for quality, security, and best practices",
        "temperature": 0.2,
        "capabilities": ["code_analysis", "security_review", "style_check"],
        "tools": [],
        "output_format": "structured",
    },
    "architect": {
        "role": "architect",
        "objective": "Design system architecture and provide technical guidance",
        "temperature": 0.4,
        "capabilities": ["system_design", "pattern_recommendation"],
        "tools": [],
        "output_format": "text",
    },
    # -------------------------------------------------------------------------
    # WORKFLOW ROLES (structural patterns)
    # -------------------------------------------------------------------------
    "supervisor": {
        "role": "supervisor",
        "objective": "Coordinate and route tasks to appropriate specialist agents",
        "temperature": 0.3,
        "capabilities": ["task_routing", "coordination", "status_tracking"],
        "tools": [],
        "is_supervisor": True,
        "output_format": "json",
    },
    "voter": {
        "role": "voter",
        "objective": "Evaluate options and provide a vote with reasoning",
        "temperature": 0.2,
        "capabilities": ["evaluation", "voting"],
        "tools": [],
        "voting_weight": 1.0,
        "output_format": "structured",
    },
    "advocate": {
        "role": "advocate",
        "objective": "Argue a specific position with supporting evidence",
        "temperature": 0.6,
        "capabilities": ["argumentation", "evidence_gathering"],
        "tools": [],
        "output_format": "text",
    },
    # -------------------------------------------------------------------------
    # GENERIC UTILITY
    # -------------------------------------------------------------------------
    "summarizer": {
        "role": "summarizer",
        "objective": "Summarize information concisely while preserving key points",
        "temperature": 0.3,
        "capabilities": ["summarization", "key_extraction"],
        "tools": [],
        "output_format": "text",
    },
    "validator": {
        "role": "validator",
        "objective": "Validate inputs against specified criteria",
        "temperature": 0.0,
        "capabilities": ["validation", "error_detection"],
        "tools": [],
        "output_format": "structured",
    },
}


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================


def register_preset(name: str, preset: Dict[str, Any]) -> None:
    """
    Register a new agent preset.

    This allows runtime extension of the preset system without code changes.
    Consistent with bili-core's dynamic registration patterns.

    Args:
        name: Unique preset name
        preset: Preset configuration dictionary

    Raises:
        ValueError: If preset name already exists (use update_preset to override)

    Example:
        >>> register_preset("my_specialist", {
        ...     "role": "specialist",
        ...     "objective": "Handle specialized tasks",
        ...     "temperature": 0.4,
        ...     "capabilities": ["specialized_analysis"],
        ... })
    """
    if name in AGENT_PRESETS:
        raise ValueError(
            f"Preset '{name}' already exists. Use update_preset() to override."
        )

    # Validate required fields
    required_fields = ["role", "objective"]
    missing = [f for f in required_fields if f not in preset]
    if missing:
        raise ValueError(f"Preset missing required fields: {missing}")

    AGENT_PRESETS[name] = preset


def update_preset(name: str, preset: Dict[str, Any]) -> None:
    """
    Update an existing preset or create a new one.

    Args:
        name: Preset name
        preset: New preset configuration
    """
    AGENT_PRESETS[name] = preset


def get_preset_config(name: str) -> Optional[Dict[str, Any]]:
    """
    Get the raw preset configuration dictionary.

    Args:
        name: Preset name

    Returns:
        Preset configuration or None if not found
    """
    return AGENT_PRESETS.get(name)


def list_presets() -> list:
    """
    List all available preset names.

    Returns:
        List of preset names
    """
    return list(AGENT_PRESETS.keys())


def get_preset(preset_name: str, agent_id: str, **overrides) -> "AgentSpec":
    """
    Create an AgentSpec from a preset with optional overrides.

    This is the primary way to use presets - get a starting configuration
    and optionally override any fields.

    Args:
        preset_name: Name of the preset to use
        agent_id: Unique agent ID (required)
        **overrides: Any AgentSpec fields to override

    Returns:
        AgentSpec instance

    Raises:
        ValueError: If preset not found

    Example:
        >>> # Use preset as-is
        >>> agent = get_preset("researcher", agent_id="research_1")
        >>>
        >>> # Override specific fields
        >>> agent = get_preset("researcher",
        ...     agent_id="custom_researcher",
        ...     temperature=0.8,
        ...     objective="Research AI safety topics"
        ... )
    """
    # Import here to avoid circular imports
    # pylint: disable=import-outside-toplevel
    from .agent_spec import AgentSpec

    preset = AGENT_PRESETS.get(preset_name)
    if preset is None:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {available}"
        )

    # Merge preset with overrides (overrides take precedence)
    config = {**preset, **overrides, "agent_id": agent_id}

    return AgentSpec(**config)


def create_agent_from_preset(
    preset_name: str, agent_id: str, **overrides
) -> "AgentSpec":
    """
    Alias for get_preset() for backwards compatibility.

    See get_preset() for documentation.
    """
    return get_preset(preset_name, agent_id, **overrides)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # List available presets
    print("Available presets:")
    for name in list_presets():
        config = get_preset_config(name)
        print(f"  - {name}: {config['role']}")

    print()

    # Create agent from preset
    agent = get_preset("researcher", agent_id="test_researcher")
    print(f"Created agent: {agent}")
    print(f"  Role: {agent.role}")
    print(f"  Objective: {agent.objective}")

    print()

    # Create with overrides
    custom_agent = get_preset(
        "researcher",
        agent_id="custom_researcher",
        temperature=0.8,
        objective="Research quantum computing advances",
    )
    print(f"Custom agent: {custom_agent}")
    print(f"  Temperature: {custom_agent.temperature}")
    print(f"  Objective: {custom_agent.objective}")

    print()

    # Register a new preset
    register_preset(
        "my_specialist",
        {
            "role": "specialist",
            "objective": "Handle specialized tasks",
            "temperature": 0.4,
            "capabilities": ["specialized_analysis"],
        },
    )
    print(f"Registered new preset. Total presets: {len(list_presets())}")
