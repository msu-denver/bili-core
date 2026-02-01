"""
Multi-agent system configuration schema.

Defines the complete structure for a MAS including agents, channels, and workflow.
This is the main configuration format for AETHER.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .agent_spec import AgentSpec
from .enums import CommunicationProtocol, WorkflowType


class Channel(BaseModel):
    """
    Communication channel between agents.

    Examples:
        >>> # Direct channel
        >>> channel = Channel(
        ...     channel_id="reviewer_to_judge",
        ...     protocol="direct",
        ...     source="reviewer",
        ...     target="judge"
        ... )

        >>> # Broadcast channel
        >>> channel = Channel(
        ...     channel_id="announcements",
        ...     protocol="broadcast",
        ...     source="supervisor",
        ...     target="all"
        ... )
    """

    channel_id: str = Field(
        ..., description="Unique channel identifier", pattern="^[a-zA-Z0-9_-]+$"
    )

    protocol: CommunicationProtocol = Field(
        ..., description="Communication protocol for this channel"
    )

    source: str = Field(
        ..., description="Source agent ID (or 'any' for broadcast from anyone)"
    )

    target: str = Field(
        ..., description="Target agent ID (or 'all' for broadcast to everyone)"
    )

    description: str = Field("", description="Human-readable channel description")

    bidirectional: bool = Field(
        False, description="Whether messages can flow both directions"
    )


class WorkflowEdge(BaseModel):
    """
    Edge in workflow graph (for custom workflows).

    Examples:
        >>> # Simple edge
        >>> edge = WorkflowEdge(
        ...     from_agent="reviewer",
        ...     to_agent="judge"
        ... )

        >>> # Conditional edge
        >>> edge = WorkflowEdge(
        ...     from_agent="reviewer",
        ...     to_agent="policy_expert",
        ...     condition="state.needs_escalation == True"
        ... )
    """

    from_agent: str = Field(..., description="Source agent ID")

    to_agent: str = Field(..., description="Target agent ID (or 'END' for terminal)")

    condition: Optional[str] = Field(
        None,
        description="Python expression for conditional routing (e.g., 'state.score > 0.5')",
    )

    label: str = Field("", description="Edge label for visualization")


class MASConfig(BaseModel):
    """
    Complete multi-agent system configuration.

    Supports 5 workflow patterns:
    1. Sequential (MAS #1: Chain)
    2. Hierarchical (MAS #2: Tree/voting)
    3. Supervisor (MAS #3: Hub-and-spoke)
    4. Consensus (MAS #4: Network/deliberation)
    5. Deliberative (MAS #5: Custom with escalation)

    Examples:
        >>> # Simple sequential MAS
        >>> config = MASConfig(
        ...     mas_id="simple_chain",
        ...     name="Simple Chain MAS",
        ...     agents=[agent1, agent2, agent3],
        ...     workflow_type="sequential"
        ... )
    """

    # =========================================================================
    # CORE IDENTITY
    # =========================================================================

    mas_id: str = Field(
        ..., description="Unique MAS identifier", pattern="^[a-zA-Z0-9_-]+$"
    )

    name: str = Field(
        ..., description="Human-readable MAS name", min_length=1, max_length=200
    )

    description: str = Field(
        "", description="Detailed MAS description", max_length=2000
    )

    version: str = Field("1.0.0", description="MAS configuration version")

    # =========================================================================
    # AGENTS & CHANNELS
    # =========================================================================

    agents: List[AgentSpec] = Field(
        ..., min_length=1, description="List of agents in this MAS"
    )

    channels: List[Channel] = Field(
        default_factory=list, description="Communication channels between agents"
    )

    # =========================================================================
    # WORKFLOW CONFIGURATION
    # =========================================================================

    workflow_type: WorkflowType = Field(..., description="Workflow execution pattern")

    entry_point: Optional[str] = Field(
        None, description="Starting agent ID (defaults to first agent)"
    )

    workflow_edges: List[WorkflowEdge] = Field(
        default_factory=list,
        description="Custom workflow edges (for CUSTOM workflow_type)",
    )

    # =========================================================================
    # HIERARCHICAL WORKFLOWS (MAS #2)
    # =========================================================================

    hierarchical_voting: bool = Field(
        False, description="Whether to use hierarchical voting pattern"
    )

    min_debate_rounds: int = Field(
        0, ge=0, description="Minimum rounds of debate for competitive agents (MAS #2)"
    )

    # =========================================================================
    # CONSENSUS WORKFLOWS (MAS #4)
    # =========================================================================

    consensus_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of agents needed for consensus (e.g., 0.66 = 2/3 majority)",
    )

    max_consensus_rounds: int = Field(
        10, gt=0, description="Maximum deliberation rounds before timeout"
    )

    consensus_detection: str = Field(
        "majority",
        description="How to detect consensus: 'majority', 'similarity', 'explicit', 'any'",
    )

    # =========================================================================
    # HUMAN-IN-LOOP (MAS #5 - MVP)
    # =========================================================================

    human_in_loop: bool = Field(
        False, description="Whether MAS can escalate to human review"
    )

    human_escalation_condition: Optional[str] = Field(
        None,
        description="Python expression for escalation (e.g., 'state.tie_breaker_needed')",
    )

    # =========================================================================
    # CHECKPOINT CONFIGURATION
    # =========================================================================

    checkpoint_enabled: bool = Field(
        True, description="Whether to enable checkpoint persistence"
    )

    checkpoint_config: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "sqlite", "path": "checkpoints.db"},
        description="Checkpoint configuration (type, path, etc.)",
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
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
        agent_ids = [a.agent_id for a in self.agents]

        for chan in self.channels:
            if chan.source not in agent_ids and chan.source != "any":
                raise ValueError(
                    f"Channel '{chan.channel_id}' source '{chan.source}' "
                    f"not found in agents"
                )
            if chan.target not in agent_ids and chan.target != "all":
                raise ValueError(
                    f"Channel '{chan.channel_id}' target '{chan.target}' "
                    f"not found in agents"
                )
        return self

    @model_validator(mode="after")
    def validate_workflow_edges(self):
        """Validate that workflow edge agents exist."""
        agent_ids = [a.agent_id for a in self.agents]

        for edge in self.workflow_edges:
            if edge.from_agent not in agent_ids:
                raise ValueError(
                    f"Workflow edge from_agent '{edge.from_agent}' not found in agents"
                )
            if edge.to_agent not in agent_ids and edge.to_agent != "END":
                raise ValueError(
                    f"Workflow edge to_agent '{edge.to_agent}' not found in agents"
                )
        return self

    @model_validator(mode="after")
    def validate_consensus_config(self):
        """Validate consensus configuration."""
        if self.workflow_type == WorkflowType.CONSENSUS:
            if self.consensus_threshold is None:
                raise ValueError("Consensus workflow requires consensus_threshold")
            if self.consensus_detection not in [
                "majority",
                "similarity",
                "explicit",
                "any",
            ]:
                raise ValueError(
                    f"Invalid consensus_detection: {self.consensus_detection}"
                )
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

    def __str__(self) -> str:
        """String representation."""
        # pylint: disable=no-member
        return f"MASConfig({self.mas_id}, {len(self.agents)} agents, {self.workflow_type.value})"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create example agents
    agent1 = AgentSpec(
        agent_id="reviewer", role="content_reviewer", objective="Review content"
    )

    agent2 = AgentSpec(agent_id="judge", role="judge", objective="Make decision")

    # Create channel
    channel = Channel(
        channel_id="reviewer_to_judge",
        protocol="direct",
        source="reviewer",
        target="judge",
    )

    # Create MAS config
    config = MASConfig(
        mas_id="simple_test",
        name="Simple Test MAS",
        description="Test configuration",
        agents=[agent1, agent2],
        channels=[channel],
        workflow_type="sequential",
    )

    print(config)
    print(f"Entry agent: {config.get_entry_agent()}")
