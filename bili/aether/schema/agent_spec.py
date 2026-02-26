"""
Agent specification schema.

Defines a domain-agnostic structure for individual agents in a multi-agent system.
Agent roles and capabilities are free-form strings to support any use case.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import OutputFormat

LOGGER = logging.getLogger(__name__)


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

    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description=(
            "LLM sampling temperature (0.0=deterministic, 2.0=creative). "
            "Defaults to model provider's default if not specified."
        ),
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
    # PIPELINE (RICH AGENT SUB-GRAPHS)
    # =========================================================================

    pipeline: Optional["PipelineSpec"] = Field(
        None,
        description=(
            "Optional internal node pipeline for this agent. When set, the agent "
            "is compiled as a LangGraph sub-graph instead of a simple react agent. "
            "Use 'bili_core_default' shorthand (via metadata) or define nodes/edges "
            "explicitly. Backwards compatible: when absent, agent behaves as before."
        ),
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

    @model_validator(mode="after")
    def validate_pipeline_model_coexistence(self):
        """Warn when both pipeline and model_name are set.

        When an agent has a pipeline, model_name at the agent level becomes
        metadata only — the actual models are configured on the pipeline's
        internal nodes. This is a soft warning, not a hard failure.
        """
        if self.pipeline and self.model_name:
            warnings.warn(
                f"Agent '{self.agent_id}' has both 'pipeline' and 'model_name' set. "
                "When pipeline is present, model_name at the agent level is treated "
                "as metadata only. Models should be configured on pipeline nodes.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_pipeline_depth(self):
        """Validate that pipeline nesting doesn't exceed max depth.

        Prevents infinite recursion when pipeline nodes contain inline
        agent specs that themselves have pipelines.
        """
        if self.pipeline:
            self._check_pipeline_depth(self.pipeline, depth=1)
        return self

    @staticmethod
    def _check_pipeline_depth(pipeline: "PipelineSpec", depth: int):
        """Recursively check pipeline nesting depth."""
        from .pipeline_spec import (  # pylint: disable=import-outside-toplevel
            MAX_PIPELINE_DEPTH,
        )

        if depth > MAX_PIPELINE_DEPTH:
            raise ValueError(
                f"Pipeline nesting depth exceeds maximum of {MAX_PIPELINE_DEPTH}. "
                "Consider flattening the pipeline or using MAS-level agent orchestration."
            )
        for node in pipeline.nodes:
            if node.agent_spec and isinstance(node.agent_spec, dict):
                inner_pipeline = node.agent_spec.get("pipeline")
                if inner_pipeline:
                    # Inner pipeline is still a dict at validation time
                    from .pipeline_spec import (  # pylint: disable=import-outside-toplevel,redefined-outer-name
                        PipelineSpec,
                    )

                    inner = PipelineSpec(**inner_pipeline)
                    AgentSpec._check_pipeline_depth(inner, depth + 1)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_display_name(self) -> str:
        """Get human-readable agent name."""
        return self.role.replace("_", " ").title()  # pylint: disable=no-member

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


# Resolve forward reference for PipelineSpec
from .pipeline_spec import (  # noqa: E402  # pylint: disable=wrong-import-position
    PipelineSpec,
)

AgentSpec.model_rebuild()
