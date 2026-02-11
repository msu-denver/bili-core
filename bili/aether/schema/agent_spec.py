"""
Agent specification schema.

Defines a domain-agnostic structure for individual agents in a multi-agent system.
Agent roles and capabilities are free-form strings to support any use case.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import OutputFormat


class AgentSpec(BaseModel):
    """
    Domain-agnostic specification for a single agent in a multi-agent system.

    This schema is intentionally flexible:
    - `role` is a free-form string (not an enum) - use any role you need
    - `capabilities` are free-form strings - define your own capabilities
    - Use the preset system for convenience without restrictions

    Examples:
        >>> # Content moderation agent
        >>> agent = AgentSpec(
        ...     agent_id="reviewer",
        ...     role="content_reviewer",
        ...     objective="Review content for policy violations"
        ... )

        >>> # Research agent
        >>> agent = AgentSpec(
        ...     agent_id="researcher",
        ...     role="researcher",
        ...     objective="Research and analyze topics",
        ...     capabilities=["web_search", "document_analysis"]
        ... )

        >>> # Custom domain agent
        >>> agent = AgentSpec(
        ...     agent_id="code_reviewer",
        ...     role="senior_engineer",
        ...     objective="Review code for quality and security",
        ...     capabilities=["static_analysis", "security_scanning"]
        ... )
    """

    # =========================================================================
    # CORE IDENTITY
    # =========================================================================

    agent_id: str = Field(
        ...,
        description="Unique identifier for this agent",
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    )

    role: str = Field(
        ...,
        description="Agent's role (free-form string - use any role you need)",
        min_length=1,
        max_length=100,
    )

    objective: str = Field(
        ...,
        description="Agent's purpose/objective (used in prompts)",
        min_length=10,
        max_length=1000,
    )

    # =========================================================================
    # BEHAVIOR CONFIGURATION
    # =========================================================================

    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt (overrides bili-core prompt if inherit=True)",
        max_length=10000,
    )

    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0.0=deterministic, 2.0=creative)",
    )

    model_name: Optional[str] = Field(
        None,
        description="LLM model name (e.g., 'gpt-4', 'claude-sonnet-3-5-20241022')",
        examples=["gpt-4", "gpt-3.5-turbo", "claude-sonnet-3-5-20241022"],
    )

    max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum tokens in LLM response"
    )

    # =========================================================================
    # CAPABILITIES & TOOLS
    # =========================================================================

    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent capabilities (free-form strings - define your own)",
    )

    tools: List[str] = Field(
        default_factory=list,
        description="Tool names this agent can use (must exist in tool registry)",
    )

    # =========================================================================
    # OUTPUT CONFIGURATION
    # =========================================================================

    output_format: OutputFormat = Field(
        OutputFormat.TEXT, description="Format of agent's output"
    )

    output_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for structured output (when output_format='structured')",
    )

    # =========================================================================
    # MIDDLEWARE CONFIGURATION
    # =========================================================================

    middleware: List[str] = Field(
        default_factory=list,
        description=(
            "Middleware names to apply to this agent's execution. "
            "Available: 'summarization', 'model_call_limit'. "
            "Middleware only applies to tool-enabled agents (via create_agent)."
        ),
    )

    middleware_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters for middleware, keyed by middleware name. "
            "E.g. {'summarization': {'max_tokens_before_summary': 4000}, "
            "'model_call_limit': {'run_limit': 10}}"
        ),
    )

    # =========================================================================
    # bili-core INTEGRATION
    # =========================================================================

    inherit_from_bili_core: bool = Field(
        False,
        description="Master toggle: inherit config from bili-core",
    )

    inherit_llm_config: bool = Field(
        True,
        description=(
            "Inherit LLM model/temperature from bili-core "
            "(only applies when inherit_from_bili_core=True)"
        ),
    )

    inherit_tools: bool = Field(
        True,
        description=(
            "Inherit tool configuration from bili-core "
            "(only applies when inherit_from_bili_core=True)"
        ),
    )

    inherit_system_prompt: bool = Field(
        True,
        description=(
            "Inherit system prompt from bili-core "
            "(only applies when inherit_from_bili_core=True)"
        ),
    )

    inherit_memory: bool = Field(
        True,
        description=(
            "Inherit memory management config from bili-core "
            "(only applies when inherit_from_bili_core=True)"
        ),
    )

    inherit_checkpoint: bool = Field(
        True,
        description=(
            "Inherit checkpoint/state persistence from bili-core "
            "(only applies when inherit_from_bili_core=True)"
        ),
    )

    # =========================================================================
    # HIERARCHICAL WORKFLOWS
    # =========================================================================

    tier: Optional[int] = Field(
        None,
        ge=1,
        description="Agent tier in hierarchical workflows (1=highest authority)",
    )

    voting_weight: float = Field(
        1.0,
        ge=0.0,
        description="Weight in voting/consensus workflows (default=1.0 = equal weight)",
    )

    # =========================================================================
    # SUPERVISOR PATTERN
    # =========================================================================

    is_supervisor: bool = Field(
        False,
        description="Whether this agent can dynamically route to specialist agents",
    )

    # =========================================================================
    # CONSENSUS WORKFLOWS
    # =========================================================================

    consensus_vote_field: Optional[str] = Field(
        None,
        description="Field name in output containing vote (e.g., 'decision', 'verdict')",
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for custom use"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @model_validator(mode="after")
    def validate_structured_output(self):
        """Validate that structured output has a schema."""
        if self.output_format == OutputFormat.STRUCTURED and not self.output_schema:
            raise ValueError(
                "Agents with output_format='structured' must specify output_schema"
            )
        return self

    @model_validator(mode="after")
    def validate_inheritance_flags(self):
        """Reset sub-flags when master inheritance toggle is off."""
        if not self.inherit_from_bili_core:
            self.inherit_llm_config = True
            self.inherit_tools = True
            self.inherit_system_prompt = True
            self.inherit_memory = True
            self.inherit_checkpoint = True
        return self

    @model_validator(mode="after")
    def validate_consensus_field(self):
        """Validate consensus vote field if specified."""
        if self.consensus_vote_field:
            if self.output_format not in [OutputFormat.JSON, OutputFormat.STRUCTURED]:
                raise ValueError(
                    "consensus_vote_field requires output_format='json' or 'structured'"
                )
        return self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_display_name(self) -> str:
        """Get human-readable agent name."""
        return self.role.replace("_", " ").title()

    def __str__(self) -> str:
        """String representation."""
        return f"Agent({self.agent_id}, role={self.role})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"AgentSpec(agent_id='{self.agent_id}', "
            f"role='{self.role}', "
            f"objective='{self.objective[:50]}...')"
        )
