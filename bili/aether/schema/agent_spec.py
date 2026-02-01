"""
Agent specification schema.

Defines the structure for individual agents in a multi-agent system.
Supports all 5 MAS patterns through optional fields.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import AgentCapability, AgentRole, OutputFormat


class AgentSpec(BaseModel):
    """
    Specification for a single agent in a multi-agent system.

    This schema supports:
    - Sequential workflows (MAS #1: Chain)
    - Hierarchical workflows (MAS #2: Tree/voting)
    - Supervisor patterns (MAS #3: Hub-and-spoke)
    - Consensus workflows (MAS #4: Network)
    - Custom workflows (MAS #5: Deliberative with escalation)

    Examples:
        >>> # Simple agent
        >>> agent = AgentSpec(
        ...     agent_id="reviewer",
        ...     role="content_reviewer",
        ...     objective="Review content for violations"
        ... )

        >>> # Agent with bili-core inheritance
        >>> agent = AgentSpec(
        ...     agent_id="judge",
        ...     role="judge",
        ...     objective="Make final decision",
        ...     inherit_from_bili_core=True
        ... )

        >>> # Hierarchical agent (MAS #2)
        >>> agent = AgentSpec(
        ...     agent_id="vote_agent",
        ...     role="judge",
        ...     objective="Vote on content",
        ...     tier=1,
        ...     voting_weight=2.0
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
        pattern="^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
    )

    role: AgentRole = Field(
        ..., description="Agent's role (determines behavior and capabilities)"
    )

    custom_role_name: Optional[str] = Field(
        None, description="Human-readable name if role='custom'", max_length=200
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

    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Agent capabilities (e.g., RAG, memory, tools)",
    )

    tools: List[str] = Field(
        default_factory=list,
        description="Tool names this agent can use (e.g., 'rag_retrieval', 'web_search')",
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
    # HIERARCHICAL WORKFLOWS (MAS #2)
    # =========================================================================

    tier: Optional[int] = Field(
        None,
        ge=1,
        description="Agent tier in hierarchical workflows (1=highest authority, 3=lowest)",
    )

    voting_weight: float = Field(
        1.0,
        ge=0.0,
        description="Weight in voting/consensus workflows (default=1.0 = equal weight)",
    )

    # =========================================================================
    # SUPERVISOR PATTERN (MAS #3)
    # =========================================================================

    is_supervisor: bool = Field(
        False,
        description="Whether this agent can dynamically route to specialist agents",
    )

    # =========================================================================
    # CONSENSUS WORKFLOWS (MAS #4)
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
    def validate_custom_role(self):
        """Validate that custom roles have a name."""
        if self.role == AgentRole.CUSTOM and not self.custom_role_name:
            raise ValueError("Agents with role='custom' must specify custom_role_name")
        return self

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
            # Sub-flags are meaningless when master toggle is off;
            # normalise them so serialised output is unambiguous.
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

    def is_bili_core_role(self) -> bool:
        """Check if this agent uses a bili-core role."""
        bili_core_roles = {
            AgentRole.CONTENT_REVIEWER,
            AgentRole.POLICY_EXPERT,
            AgentRole.JUDGE,
            AgentRole.APPEALS_SPECIALIST,
            AgentRole.COMMUNITY_MANAGER,
        }
        return self.role in bili_core_roles

    def get_display_name(self) -> str:
        """Get human-readable agent name."""
        if self.role == AgentRole.CUSTOM and self.custom_role_name:
            return self.custom_role_name
        # pylint: disable=no-member
        return self.role.value.replace("_", " ").title()

    def __str__(self) -> str:
        """String representation."""
        # pylint: disable=no-member
        return f"Agent({self.agent_id}, role={self.role.value})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"AgentSpec(agent_id='{self.agent_id}', "
            # pylint: disable=no-member
            f"role={self.role.value}, "
            f"objective='{self.objective[:50]}...')"
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Simple agent
    simple_agent = AgentSpec(
        agent_id="reviewer",
        role=AgentRole.CONTENT_REVIEWER,
        objective="Review content for policy violations",
    )
    print(simple_agent)

    # Example 2: Agent with full bili-core inheritance
    bili_agent = AgentSpec(
        agent_id="judge",
        role=AgentRole.JUDGE,
        objective="Make final moderation decision",
        inherit_from_bili_core=True,
    )
    print(f"Is bili-core role: {bili_agent.is_bili_core_role()}")

    # Example 2b: Selective bili-core inheritance (custom prompt, inherit rest)
    selective_agent = AgentSpec(
        agent_id="custom_judge",
        role=AgentRole.JUDGE,
        objective="Make final moderation decision with custom prompt",
        inherit_from_bili_core=True,
        inherit_system_prompt=False,  # Use own prompt, inherit everything else
        system_prompt="You are a strict content moderator.",
    )
    print(f"Inherits tools: {selective_agent.inherit_tools}")
    print(f"Inherits prompt: {selective_agent.inherit_system_prompt}")

    # Example 3: Hierarchical agent (MAS #2)
    vote_agent = AgentSpec(
        agent_id="tier1_vote",
        role=AgentRole.JUDGE,
        objective="Aggregate votes from tier 2",
        tier=1,
        voting_weight=2.0,
    )
    print(vote_agent)

    # Example 4: Custom agent with structured output
    custom_agent = AgentSpec(
        agent_id="analyzer",
        role=AgentRole.CUSTOM,
        custom_role_name="Sentiment Analyzer",
        objective="Analyze sentiment of content",
        output_format=OutputFormat.STRUCTURED,
        output_schema={
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
                "confidence": {"type": "number"},
            },
        },
    )
    print(f"Display name: {custom_agent.get_display_name()}")
