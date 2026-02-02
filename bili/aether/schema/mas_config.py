"""
Multi-agent system configuration schema.

Defines the complete structure for a MAS including agents, channels, and workflow.
This is the main configuration format for AETHER.

Design Philosophy:
- Focus on STRUCTURE (how agents connect), not DOMAIN (what task they perform)
- Domain-agnostic: works for content moderation, research, customer support, etc.
- YAML-friendly: all fields designed for easy serialization
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .agent_spec import AgentSpec
from .enums import CommunicationProtocol, WorkflowType


class Channel(BaseModel):
    """
    Communication channel between agents.

    Channels define HOW agents communicate, not WHAT they communicate about.

    Examples:
        >>> # Direct channel: reviewer sends to judge
        >>> channel = Channel(
        ...     channel_id="reviewer_to_judge",
        ...     protocol="direct",
        ...     source="reviewer",
        ...     target="judge"
        ... )

        >>> # Broadcast channel: supervisor announces to all
        >>> channel = Channel(
        ...     channel_id="announcements",
        ...     protocol="broadcast",
        ...     source="supervisor",
        ...     target="all"
        ... )
    """

    channel_id: str = Field(
        ...,
        description="Unique channel identifier",
        pattern="^[a-zA-Z0-9_-]+$",
    )

    protocol: CommunicationProtocol = Field(
        ...,
        description="Communication protocol for this channel",
    )

    source: str = Field(
        ...,
        description="Source agent ID (or 'any' for messages from anyone)",
    )

    target: str = Field(
        ...,
        description="Target agent ID (or 'all' for broadcast)",
    )

    description: str = Field(
        "",
        description="Human-readable channel description",
    )

    bidirectional: bool = Field(
        False,
        description="Whether messages can flow both directions",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional channel metadata",
    )


class WorkflowEdge(BaseModel):
    """
    Edge in workflow graph (for custom workflows).

    Edges define the flow between agents and can include conditions.

    Examples:
        >>> # Simple edge
        >>> edge = WorkflowEdge(
        ...     from_agent="reviewer",
        ...     to_agent="judge"
        ... )

        >>> # Conditional edge
        >>> edge = WorkflowEdge(
        ...     from_agent="reviewer",
        ...     to_agent="escalation",
        ...     condition="state.confidence < 0.5"
        ... )
    """

    from_agent: str = Field(
        ...,
        description="Source agent ID",
    )

    to_agent: str = Field(
        ...,
        description="Target agent ID (or 'END' for terminal)",
    )

    condition: Optional[str] = Field(
        None,
        description="Python expression for conditional routing",
    )

    label: str = Field(
        "",
        description="Edge label for visualization",
    )

    priority: int = Field(
        0,
        description="Edge priority when multiple conditions match (higher = first)",
    )


class MASConfig(BaseModel):
    """
    Complete multi-agent system configuration.

    This is the main entry point for defining a multi-agent system.
    It is DOMAIN-AGNOSTIC - the same structure works for:
    - Content moderation
    - Research workflows
    - Customer support
    - Data processing pipelines
    - Any other multi-agent use case

    Workflow Types:
    - SEQUENTIAL: Linear chain (A -> B -> C)
    - PARALLEL: Concurrent execution
    - HIERARCHICAL: Tree structure with tiers
    - SUPERVISOR: Hub-and-spoke with dynamic routing
    - CONSENSUS: Network deliberation
    - DELIBERATIVE: Custom with conditional edges
    - CUSTOM: Fully custom graph

    Examples:
        >>> # Simple sequential workflow
        >>> config = MASConfig(
        ...     mas_id="simple_chain",
        ...     name="Simple Processing Chain",
        ...     workflow_type="sequential",
        ...     agents=[agent1, agent2, agent3],
        ... )

        >>> # Supervisor pattern
        >>> config = MASConfig(
        ...     mas_id="support_system",
        ...     name="Customer Support System",
        ...     workflow_type="supervisor",
        ...     agents=[supervisor, specialist1, specialist2],
        ... )
    """

    # =========================================================================
    # CORE IDENTITY
    # =========================================================================

    mas_id: str = Field(
        ...,
        description="Unique MAS identifier",
        pattern="^[a-zA-Z0-9_-]+$",
    )

    name: str = Field(
        ...,
        description="Human-readable MAS name",
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        "",
        description="Detailed MAS description",
        max_length=5000,
    )

    version: str = Field(
        "1.0.0",
        description="MAS configuration version (semver)",
    )

    # =========================================================================
    # AGENTS & CHANNELS
    # =========================================================================

    agents: List[AgentSpec] = Field(
        ...,
        min_length=1,
        description="List of agents in this MAS",
    )

    channels: List[Channel] = Field(
        default_factory=list,
        description="Communication channels between agents",
    )

    # =========================================================================
    # WORKFLOW CONFIGURATION
    # =========================================================================

    workflow_type: WorkflowType = Field(
        ...,
        description="Workflow execution pattern",
    )

    entry_point: Optional[str] = Field(
        None,
        description="Starting agent ID (defaults to first agent)",
    )

    workflow_edges: List[WorkflowEdge] = Field(
        default_factory=list,
        description="Custom workflow edges (for CUSTOM/DELIBERATIVE workflows)",
    )

    # =========================================================================
    # HIERARCHICAL WORKFLOWS
    # =========================================================================

    hierarchical_voting: bool = Field(
        False,
        description="Whether to use hierarchical voting pattern",
    )

    min_debate_rounds: int = Field(
        0,
        ge=0,
        description="Minimum debate rounds for competitive workflows",
    )

    # =========================================================================
    # CONSENSUS WORKFLOWS
    # =========================================================================

    consensus_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of agents needed for consensus (e.g., 0.66 = 2/3)",
    )

    max_consensus_rounds: int = Field(
        10,
        gt=0,
        description="Maximum deliberation rounds before timeout",
    )

    consensus_detection: str = Field(
        "majority",
        description="Consensus detection method: 'majority', 'similarity', 'explicit', 'any'",
    )

    # =========================================================================
    # HUMAN-IN-LOOP
    # =========================================================================

    human_in_loop: bool = Field(
        False,
        description="Whether MAS can escalate to human review",
    )

    human_escalation_condition: Optional[str] = Field(
        None,
        description="Python expression for human escalation",
    )

    # =========================================================================
    # RUNTIME CONFIGURATION
    # =========================================================================

    checkpoint_enabled: bool = Field(
        True,
        description="Whether to enable checkpoint persistence",
    )

    checkpoint_config: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "memory"},
        description="Checkpoint configuration",
    )

    max_execution_time: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum execution time in seconds",
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @model_validator(mode="after")
    def validate_agent_ids_unique(self):
        """Validate that all agent IDs are unique."""
        agent_ids = [a.agent_id for a in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            duplicates = [aid for aid in agent_ids if agent_ids.count(aid) > 1]
            raise ValueError(f"Duplicate agent IDs found: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_entry_point(self):
        """Validate that entry_point exists in agents."""
        if self.entry_point:
            agent_ids = [a.agent_id for a in self.agents]
            if self.entry_point not in agent_ids:
                raise ValueError(
                    f"entry_point '{self.entry_point}' not found in agents: {agent_ids}"
                )
        return self

    @model_validator(mode="after")
    def validate_channel_agents(self):
        """Validate that channel source/target agents exist."""
        agent_ids = {a.agent_id for a in self.agents}

        for chan in self.channels:
            if chan.source not in agent_ids and chan.source != "any":
                raise ValueError(
                    f"Channel '{chan.channel_id}' source '{chan.source}' not in agents"
                )
            if chan.target not in agent_ids and chan.target != "all":
                raise ValueError(
                    f"Channel '{chan.channel_id}' target '{chan.target}' not in agents"
                )
        return self

    @model_validator(mode="after")
    def validate_workflow_edges(self):
        """Validate that workflow edge agents exist."""
        agent_ids = {a.agent_id for a in self.agents}

        for edge in self.workflow_edges:
            if edge.from_agent not in agent_ids:
                raise ValueError(
                    f"Workflow edge from_agent '{edge.from_agent}' not in agents"
                )
            if edge.to_agent not in agent_ids and edge.to_agent != "END":
                raise ValueError(
                    f"Workflow edge to_agent '{edge.to_agent}' not in agents"
                )
        return self

    @model_validator(mode="after")
    def validate_consensus_config(self):
        """Validate consensus configuration."""
        if self.workflow_type == WorkflowType.CONSENSUS:
            if self.consensus_threshold is None:
                raise ValueError("Consensus workflow requires consensus_threshold")
        return self

    @model_validator(mode="after")
    def validate_hierarchical_tiers(self):
        """Validate hierarchical tier configuration."""
        if self.workflow_type == WorkflowType.HIERARCHICAL or self.hierarchical_voting:
            tiers = [a.tier for a in self.agents if a.tier is not None]
            if not tiers:
                raise ValueError(
                    "Hierarchical workflow requires agents to have tier values"
                )
        return self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_agent(self, agent_id: str) -> Optional[AgentSpec]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def get_entry_agent(self) -> AgentSpec:
        """Get entry point agent."""
        entry_id = self.entry_point or self.agents[0].agent_id
        agent = self.get_agent(entry_id)
        if not agent:
            raise ValueError(f"Entry agent '{entry_id}' not found")
        return agent

    def get_agents_by_tier(self, tier: int) -> List[AgentSpec]:
        """Get all agents at a specific tier (for hierarchical workflows)."""
        return [a for a in self.agents if a.tier == tier]

    def get_agents_by_role(self, role: str) -> List[AgentSpec]:
        """Get all agents with a specific role."""
        return [a for a in self.agents if a.role == role]

    def get_supervisors(self) -> List[AgentSpec]:
        """Get all supervisor agents."""
        return [a for a in self.agents if a.is_supervisor]

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for YAML serialization."""
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MASConfig({self.mas_id}, "
            f"{len(self.agents)} agents, "
            f"{self.workflow_type.value})"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_sequential_mas(
    mas_id: str,
    name: str,
    agents: List[AgentSpec],
    **kwargs,
) -> MASConfig:
    """
    Create a sequential (chain) MAS.

    Agents execute in order: agent1 -> agent2 -> agent3 -> ...

    Args:
        mas_id: Unique identifier
        name: Human-readable name
        agents: List of agents in execution order
        **kwargs: Additional MASConfig fields

    Returns:
        Configured MASConfig
    """
    return MASConfig(
        mas_id=mas_id,
        name=name,
        agents=agents,
        workflow_type=WorkflowType.SEQUENTIAL,
        **kwargs,
    )


def create_supervisor_mas(
    mas_id: str,
    name: str,
    supervisor: AgentSpec,
    specialists: List[AgentSpec],
    **kwargs,
) -> MASConfig:
    """
    Create a supervisor (hub-and-spoke) MAS.

    Supervisor routes tasks to appropriate specialists.

    Args:
        mas_id: Unique identifier
        name: Human-readable name
        supervisor: The supervisor agent (should have is_supervisor=True)
        specialists: List of specialist agents
        **kwargs: Additional MASConfig fields

    Returns:
        Configured MASConfig
    """
    # Ensure supervisor flag is set
    supervisor.is_supervisor = True

    return MASConfig(
        mas_id=mas_id,
        name=name,
        agents=[supervisor] + specialists,
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point=supervisor.agent_id,
        **kwargs,
    )


def create_consensus_mas(
    mas_id: str,
    name: str,
    agents: List[AgentSpec],
    consensus_threshold: float = 0.66,
    **kwargs,
) -> MASConfig:
    """
    Create a consensus-based MAS.

    Agents deliberate until consensus is reached.

    Args:
        mas_id: Unique identifier
        name: Human-readable name
        agents: List of deliberating agents
        consensus_threshold: Fraction needed for consensus (default 2/3)
        **kwargs: Additional MASConfig fields

    Returns:
        Configured MASConfig
    """
    return MASConfig(
        mas_id=mas_id,
        name=name,
        agents=agents,
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=consensus_threshold,
        **kwargs,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Content moderation (one domain)
    from .agent_presets import get_preset

    reviewer = get_preset("content_reviewer", agent_id="reviewer")
    judge = get_preset("moderator_judge", agent_id="judge")

    moderation_mas = create_sequential_mas(
        mas_id="content_moderation",
        name="Content Moderation Pipeline",
        agents=[reviewer, judge],
        description="Review and moderate user content",
    )
    print(f"Created: {moderation_mas}")

    print()

    # Example 2: Research workflow (different domain)
    researcher = get_preset("researcher", agent_id="researcher")
    analyst = get_preset("analyst", agent_id="analyst")
    summarizer = get_preset("summarizer", agent_id="summarizer")

    research_mas = create_sequential_mas(
        mas_id="research_pipeline",
        name="Research Analysis Pipeline",
        agents=[researcher, analyst, summarizer],
        description="Research, analyze, and summarize findings",
    )
    print(f"Created: {research_mas}")

    print()

    # Example 3: Custom domain with custom roles
    custom_agent1 = AgentSpec(
        agent_id="data_ingester",
        role="data_ingester",
        objective="Ingest and validate incoming data streams",
        capabilities=["data_validation", "format_conversion"],
    )

    custom_agent2 = AgentSpec(
        agent_id="anomaly_detector",
        role="anomaly_detector",
        objective="Detect anomalies in processed data",
        capabilities=["statistical_analysis", "pattern_detection"],
    )

    custom_mas = create_sequential_mas(
        mas_id="data_pipeline",
        name="Data Processing Pipeline",
        agents=[custom_agent1, custom_agent2],
        description="Custom data processing workflow",
    )
    print(f"Created: {custom_mas}")
