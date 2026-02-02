"""
Agent specification schema.

Defines the structure for individual agents in a multi-agent system.
Supports any domain through free-form role and capability strings.

Design Philosophy:
- Roles are FREE-FORM STRINGS: "content_reviewer", "data_analyst", "my_custom_role"
- Capabilities are FREE-FORM STRINGS: "web_search", "code_analysis", "custom_skill"
- No domain-specific enums that would limit framework applicability
- Presets provide convenience without restrictions (see agent_presets.py)

This is consistent with bili-core's extensible patterns where tools, nodes,
and middleware are registered dynamically rather than hardcoded.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import OutputFormat


class AgentSpec(BaseModel):
    """
    Specification for a single agent in a multi-agent system.

    Key Design Decision: Roles and capabilities are free-form strings,
    not enums. This allows AETHER to be used for ANY domain without
    requiring code changes.

    Examples:
        >>> # Content moderation domain
        >>> agent = AgentSpec(
        ...     agent_id="reviewer",
        ...     role="content_reviewer",
        ...     objective="Review content for policy violations"
        ... )

        >>> # Research domain
        >>> agent = AgentSpec(
        ...     agent_id="researcher",
        ...     role="research_analyst",
        ...     objective="Analyze scientific papers",
        ...     capabilities=["literature_review", "citation_analysis"]
        ... )

        >>> # Customer support domain
        >>> agent = AgentSpec(
        ...     agent_id="support_bot",
        ...     role="support_agent",
        ...     objective="Help customers resolve issues",
        ...     capabilities=["faq_lookup", "ticket_creation"]
        ... )

        >>> # Custom domain - no restrictions!
        >>> agent = AgentSpec(
        ...     agent_id="my_agent",
        ...     role="my_custom_role",
        ...     objective="Do something unique",
        ...     capabilities=["custom_capability_1", "custom_capability_2"]
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
        description=(
            "Agent's role - any string that describes its function. "
            "Examples: 'content_reviewer', 'data_analyst', 'support_agent', 'my_role'"
        ),
        min_length=1,
        max_length=100,
    )

    objective: str = Field(
        ...,
        description="Agent's purpose/objective (used in system prompts)",
        min_length=10,
        max_length=2000,
    )

    # =========================================================================
    # BEHAVIOR CONFIGURATION
    # =========================================================================

    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt (overrides bili-core prompt if inheriting)",
        max_length=20000,
    )

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0.0=deterministic, 2.0=creative)",
    )

    model_name: Optional[str] = Field(
        None,
        description="LLM model name (e.g., 'gpt-4', 'claude-sonnet-4-20250514')",
    )

    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=200000,
        description="Maximum tokens in LLM response",
    )

    # =========================================================================
    # CAPABILITIES & TOOLS
    # =========================================================================

    capabilities: List[str] = Field(
        default_factory=list,
        description=(
            "Agent capabilities as free-form strings. "
            "Examples: ['web_search', 'code_analysis', 'summarization']"
        ),
    )

    tools: List[str] = Field(
        default_factory=list,
        description=(
            "Tool names this agent can use. "
            "Must match tools registered in bili-core's TOOLS registry."
        ),
    )

    # =========================================================================
    # OUTPUT CONFIGURATION
    # =========================================================================

    output_format: OutputFormat = Field(
        OutputFormat.TEXT,
        description="Format of agent's output",
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
        description="Master toggle: inherit configuration from bili-core runtime",
    )

    inherit_llm_config: bool = Field(
        True,
        description="Inherit LLM model/temperature from bili-core (requires master toggle)",
    )

    inherit_tools: bool = Field(
        True,
        description="Inherit tool configuration from bili-core (requires master toggle)",
    )

    inherit_system_prompt: bool = Field(
        True,
        description="Inherit system prompt from bili-core (requires master toggle)",
    )

    inherit_memory: bool = Field(
        True,
        description="Inherit memory management from bili-core (requires master toggle)",
    )

    inherit_checkpoint: bool = Field(
        True,
        description="Inherit checkpoint/state persistence from bili-core (requires master toggle)",
    )

    # =========================================================================
    # WORKFLOW PROPERTIES
    # =========================================================================

    tier: Optional[int] = Field(
        None,
        ge=1,
        description="Agent tier in hierarchical workflows (1=highest authority)",
    )

    voting_weight: float = Field(
        1.0,
        ge=0.0,
        description="Weight in voting/consensus workflows (default=1.0)",
    )

    is_supervisor: bool = Field(
        False,
        description="Whether this agent can dynamically route to other agents",
    )

    consensus_vote_field: Optional[str] = Field(
        None,
        description="Field name in output containing vote (for consensus workflows)",
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    preset: Optional[str] = Field(
        None,
        description="Name of preset used to create this agent (for tracking)",
    )

    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for custom use",
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @field_validator("role")
    @classmethod
    def validate_role_format(cls, v: str) -> str:
        """Validate role is a reasonable identifier."""
        # Allow alphanumeric, underscore, hyphen, spaces
        import re

        if not re.match(r"^[a-zA-Z0-9_\- ]+$", v):
            raise ValueError(
                f"Role '{v}' contains invalid characters. "
                "Use alphanumeric, underscore, hyphen, or space."
            )
        return v.strip()

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: List[str]) -> List[str]:
        """Validate capabilities are reasonable identifiers."""
        import re

        for cap in v:
            if not re.match(r"^[a-zA-Z0-9_\-]+$", cap):
                raise ValueError(
                    f"Capability '{cap}' contains invalid characters. "
                    "Use alphanumeric, underscore, or hyphen."
                )
        return v

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
        """Normalize sub-flags when master inheritance is off."""
        if not self.inherit_from_bili_core:
            # Sub-flags are meaningless when master toggle is off
            # Normalize them for consistent serialization
            self.inherit_llm_config = True
            self.inherit_tools = True
            self.inherit_system_prompt = True
            self.inherit_memory = True
            self.inherit_checkpoint = True
        return self

    @model_validator(mode="after")
    def validate_consensus_field(self):
        """Validate consensus vote field configuration."""
        if self.consensus_vote_field:
            if self.output_format not in [OutputFormat.JSON, OutputFormat.STRUCTURED]:
                raise ValueError(
                    "consensus_vote_field requires output_format='json' or 'structured'"
                )
        return self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def get_display_name(self) -> str:
        """Get human-readable agent name."""
        return self.role.replace("_", " ").replace("-", " ").title()

    def to_node_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs suitable for bili-core node initialization.

        This bridges AETHER agents to bili-core's node-based architecture.
        """
        kwargs = {
            "agent_id": self.agent_id,
            "role": self.role,
            "objective": self.objective,
            "temperature": self.temperature,
            "tools": self.tools,
        }

        if self.system_prompt:
            kwargs["system_prompt"] = self.system_prompt
        if self.model_name:
            kwargs["model_name"] = self.model_name
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        return kwargs

    def __str__(self) -> str:
        """String representation."""
        return f"Agent({self.agent_id}, role='{self.role}')"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"AgentSpec(agent_id='{self.agent_id}', "
            f"role='{self.role}', "
            f"objective='{self.objective[:50]}...')"
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Content moderation agent
    reviewer = AgentSpec(
        agent_id="reviewer",
        role="content_reviewer",
        objective="Review content for policy violations",
        temperature=0.0,
        capabilities=["text_analysis", "policy_evaluation"],
    )
    print(f"Created: {reviewer}")
    print(f"Display name: {reviewer.get_display_name()}")

    print()

    # Example 2: Research agent (different domain)
    researcher = AgentSpec(
        agent_id="researcher",
        role="research_analyst",
        objective="Analyze scientific papers and extract key findings",
        temperature=0.5,
        capabilities=["literature_review", "citation_analysis", "summarization"],
        tools=["serp_api_tool"],
    )
    print(f"Created: {researcher}")
    print(f"Has web capability: {researcher.has_capability('literature_review')}")

    print()

    # Example 3: Custom role - framework doesn't care what domain you use
    custom = AgentSpec(
        agent_id="my_agent",
        role="quantum_optimizer",
        objective="Optimize quantum circuit parameters for maximum fidelity",
        capabilities=["quantum_simulation", "parameter_tuning"],
    )
    print(f"Created: {custom}")

    print()

    # Example 4: Agent with bili-core inheritance
    inherited = AgentSpec(
        agent_id="inherited_agent",
        role="assistant",
        objective="General assistant inheriting bili-core config",
        inherit_from_bili_core=True,
        inherit_system_prompt=False,  # Use own prompt
        system_prompt="You are a specialized assistant.",
    )
    print(f"Created: {inherited}")
    print(f"Inherits tools: {inherited.inherit_tools}")
    print(f"Inherits prompt: {inherited.inherit_system_prompt}")
