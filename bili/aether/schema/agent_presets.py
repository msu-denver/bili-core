"""
Agent preset system for AETHER.

Provides pre-defined configurations for common agent patterns.
Presets offer convenience without restrictions - users can always
define custom agents with any role and capabilities.
"""

from typing import Any, Dict, Optional

from .agent_spec import AgentSpec


# =============================================================================
# PRESET REGISTRY
# =============================================================================

AGENT_PRESETS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Content Moderation Presets
    # -------------------------------------------------------------------------
    "content_reviewer": {
        "role": "content_reviewer",
        "capabilities": ["text_analysis", "policy_enforcement"],
        "temperature": 0.3,
    },
    "policy_expert": {
        "role": "policy_expert",
        "capabilities": ["policy_lookup", "rule_interpretation"],
        "temperature": 0.2,
    },
    "moderator_judge": {
        "role": "judge",
        "capabilities": ["decision_making", "evidence_synthesis"],
        "temperature": 0.1,
    },
    "appeals_specialist": {
        "role": "appeals_specialist",
        "capabilities": ["context_analysis", "precedent_lookup"],
        "temperature": 0.3,
    },
    # -------------------------------------------------------------------------
    # Research & Analysis Presets
    # -------------------------------------------------------------------------
    "researcher": {
        "role": "researcher",
        "capabilities": ["web_search", "document_analysis", "summarization"],
        "tools": ["serp_api_tool"],
        "temperature": 0.5,
    },
    "analyst": {
        "role": "analyst",
        "capabilities": ["data_analysis", "pattern_recognition", "reporting"],
        "temperature": 0.3,
    },
    "fact_checker": {
        "role": "fact_checker",
        "capabilities": ["web_search", "source_verification", "claim_analysis"],
        "tools": ["serp_api_tool"],
        "temperature": 0.2,
    },
    # -------------------------------------------------------------------------
    # Code & Technical Presets
    # -------------------------------------------------------------------------
    "code_reviewer": {
        "role": "code_reviewer",
        "capabilities": ["static_analysis", "best_practices", "security_scanning"],
        "temperature": 0.2,
    },
    "security_auditor": {
        "role": "security_auditor",
        "capabilities": ["vulnerability_detection", "threat_modeling"],
        "temperature": 0.1,
    },
    "documentation_writer": {
        "role": "documentation_writer",
        "capabilities": ["technical_writing", "code_explanation"],
        "temperature": 0.4,
    },
    # -------------------------------------------------------------------------
    # Customer Support Presets
    # -------------------------------------------------------------------------
    "support_agent": {
        "role": "support_agent",
        "capabilities": ["ticket_handling", "knowledge_base_search"],
        "tools": ["faiss_retriever"],
        "temperature": 0.4,
    },
    "escalation_specialist": {
        "role": "escalation_specialist",
        "capabilities": ["complex_issue_resolution", "customer_advocacy"],
        "temperature": 0.3,
    },
    # -------------------------------------------------------------------------
    # Workflow Pattern Presets
    # -------------------------------------------------------------------------
    "supervisor": {
        "role": "supervisor",
        "capabilities": ["task_routing", "agent_coordination", "quality_control"],
        "is_supervisor": True,
        "temperature": 0.2,
    },
    "consensus_voter": {
        "role": "voter",
        "capabilities": ["evaluation", "voting"],
        "output_format": "json",
        "temperature": 0.3,
    },
    "debate_advocate": {
        "role": "advocate",
        "capabilities": ["argumentation", "evidence_presentation"],
        "temperature": 0.5,
    },
}


# =============================================================================
# PRESET FUNCTIONS
# =============================================================================


def register_preset(name: str, preset: Dict[str, Any]) -> None:
    """
    Register a custom preset at runtime.

    Args:
        name: Unique preset name
        preset: Dict with AgentSpec fields to use as defaults

    Examples:
        >>> register_preset("my_agent", {
        ...     "role": "custom_role",
        ...     "capabilities": ["cap1", "cap2"],
        ...     "temperature": 0.5,
        ... })
    """
    AGENT_PRESETS[name] = preset


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a preset by name.

    Args:
        name: Preset name

    Returns:
        Preset dict or None if not found
    """
    return AGENT_PRESETS.get(name)


def list_presets() -> list:
    """
    List all available preset names.

    Returns:
        List of preset names
    """
    return list(AGENT_PRESETS.keys())


def create_agent_from_preset(
    preset_name: str,
    agent_id: str,
    objective: str,
    **overrides,
) -> AgentSpec:
    """
    Create an AgentSpec from a preset with optional overrides.

    Args:
        preset_name: Name of preset to use
        agent_id: Unique agent ID (required)
        objective: Agent objective (required)
        **overrides: Any AgentSpec fields to override

    Returns:
        Configured AgentSpec instance

    Raises:
        ValueError: If preset not found

    Examples:
        >>> # Create researcher from preset
        >>> agent = create_agent_from_preset(
        ...     preset_name="researcher",
        ...     agent_id="my_researcher",
        ...     objective="Research quantum computing advances"
        ... )

        >>> # Create with overrides
        >>> agent = create_agent_from_preset(
        ...     preset_name="code_reviewer",
        ...     agent_id="senior_reviewer",
        ...     objective="Review Python code for security",
        ...     temperature=0.1,
        ...     capabilities=["python_analysis", "security_focus"]
        ... )
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list_presets()}"
        )

    # Merge preset with overrides
    config = {
        **preset,
        "agent_id": agent_id,
        "objective": objective,
        **overrides,
    }

    return AgentSpec(**config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Available presets:", list_presets())

    # Create agent from preset
    researcher = create_agent_from_preset(
        preset_name="researcher",
        agent_id="tech_researcher",
        objective="Research and summarize AI developments",
    )
    print(f"\nResearcher: {researcher}")
    print(f"  Role: {researcher.role}")
    print(f"  Capabilities: {researcher.capabilities}")
    print(f"  Temperature: {researcher.temperature}")

    # Create with overrides
    custom_reviewer = create_agent_from_preset(
        preset_name="code_reviewer",
        agent_id="python_expert",
        objective="Review Python code for best practices",
        temperature=0.1,
        capabilities=["python_analysis", "pep8_compliance", "type_checking"],
    )
    print(f"\nCustom Reviewer: {custom_reviewer}")
    print(f"  Capabilities: {custom_reviewer.capabilities}")

    # Register custom preset
    register_preset("my_custom_agent", {
        "role": "custom_role",
        "capabilities": ["custom_cap"],
        "temperature": 0.4,
    })
    print(f"\nAfter registering custom preset: {list_presets()}")
