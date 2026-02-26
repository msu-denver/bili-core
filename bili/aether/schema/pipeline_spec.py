"""
Pipeline specification schema for rich agent sub-graphs.

Defines the internal node pipeline structure that enables AETHER agents
to run full multi-node LangGraph sub-graphs instead of simple react agents.
When an AgentSpec has a pipeline, it is compiled into a CompiledStateGraph
and embedded as a single node in the parent MAS graph.

Design Principles:
- Backwards compatible: pipeline is Optional on AgentSpec; absence = simple react agent
- Node types reference bili-core's GRAPH_NODE_REGISTRY or inline agent specs
- LangGraph sub-graph compilation handles the rest
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

LOGGER = logging.getLogger(__name__)

# Maximum nesting depth for pipelines containing agents with their own pipelines
MAX_PIPELINE_DEPTH = 3


class PipelineNodeSpec(BaseModel):
    """
    Specification for a single node within an agent's internal pipeline.

    Nodes can reference:
    - bili-core registry nodes by name (e.g., "react_agent", "add_persona_and_summary")
    - Inline AETHER agent specs (for sub-agents within the pipeline)

    Examples:
        >>> # Registry node
        >>> node = PipelineNodeSpec(
        ...     node_id="persona",
        ...     node_type="add_persona_and_summary"
        ... )

        >>> # Node with configuration
        >>> node = PipelineNodeSpec(
        ...     node_id="react",
        ...     node_type="react_agent",
        ...     config={"temperature": 0.7}
        ... )
    """

    node_id: str = Field(
        ...,
        description="Unique identifier for this node within the pipeline",
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
    )

    node_type: str = Field(
        ...,
        description=(
            "Node type: a bili-core GRAPH_NODE_REGISTRY name "
            "(e.g., 'react_agent', 'add_persona_and_summary') "
            "or 'agent' for an inline AETHER agent"
        ),
        min_length=1,
    )

    agent_spec: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Inline agent specification when node_type='agent'. "
            "Parsed as an AgentSpec at compile time to avoid circular imports."
        ),
    )

    config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Configuration passed to the node builder as kwargs. "
            "For registry nodes, these are merged with defaults from the "
            "parent AgentSpec (e.g., llm_model, tools, middleware)."
        ),
    )

    @model_validator(mode="after")
    def validate_agent_node(self):
        """Validate that agent nodes have an agent_spec."""
        if self.node_type == "agent" and not self.agent_spec:
            raise ValueError(
                f"Pipeline node '{self.node_id}' has node_type='agent' "
                "but no agent_spec provided"
            )
        if self.node_type != "agent" and self.agent_spec:
            warnings.warn(
                f"Pipeline node '{self.node_id}' has agent_spec but "
                f"node_type='{self.node_type}' (not 'agent'). "
                "The agent_spec will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self


class PipelineEdgeSpec(BaseModel):
    """
    Edge between two nodes in an agent's internal pipeline.

    Examples:
        >>> # Simple edge
        >>> edge = PipelineEdgeSpec(
        ...     from_node="persona",
        ...     to_node="react"
        ... )

        >>> # Conditional edge
        >>> edge = PipelineEdgeSpec(
        ...     from_node="react",
        ...     to_node="escalation",
        ...     condition="state.needs_escalation == True",
        ...     label="escalation path"
        ... )

        >>> # Terminal edge
        >>> edge = PipelineEdgeSpec(
        ...     from_node="summarize",
        ...     to_node="END"
        ... )
    """

    from_node: str = Field(..., description="Source node ID within the pipeline")

    to_node: str = Field(
        ..., description="Target node ID within the pipeline (or 'END' for terminal)"
    )

    condition: Optional[str] = Field(
        None,
        description="Python expression for conditional routing",
    )

    label: str = Field("", description="Edge label for visualization")


class PipelineSpec(BaseModel):
    """
    Complete internal pipeline specification for an agent.

    Defines the sub-graph that an agent executes internally, compiled
    into a LangGraph CompiledStateGraph and embedded as a single node
    in the parent MAS graph.

    Examples:
        >>> pipeline = PipelineSpec(
        ...     nodes=[
        ...         PipelineNodeSpec(node_id="persona", node_type="add_persona_and_summary"),
        ...         PipelineNodeSpec(node_id="react", node_type="react_agent"),
        ...     ],
        ...     edges=[
        ...         PipelineEdgeSpec(from_node="persona", to_node="react"),
        ...         PipelineEdgeSpec(from_node="react", to_node="END"),
        ...     ]
        ... )
    """

    nodes: List[PipelineNodeSpec] = Field(
        ...,
        min_length=1,
        description="Nodes in the agent's internal pipeline",
    )

    edges: List[PipelineEdgeSpec] = Field(
        ...,
        min_length=1,
        description="Edges connecting nodes in the pipeline",
    )

    entry_point: Optional[str] = Field(
        None,
        description="Starting node ID (defaults to first node in the list)",
    )

    @model_validator(mode="after")
    def validate_node_ids_unique(self):
        """Validate that all node IDs within the pipeline are unique."""
        node_ids = [n.node_id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            raise ValueError(f"Duplicate node IDs in pipeline: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_edge_references(self):
        """Validate that all edges reference valid node IDs."""
        node_ids = {n.node_id for n in self.nodes}

        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(
                    f"Pipeline edge from_node '{edge.from_node}' "
                    f"not found in pipeline nodes: {node_ids}"
                )
            if edge.to_node not in node_ids and edge.to_node != "END":
                raise ValueError(
                    f"Pipeline edge to_node '{edge.to_node}' "
                    f"not found in pipeline nodes: {node_ids}"
                )
        return self

    @model_validator(mode="after")
    def validate_entry_point(self):
        """Validate that entry_point references a valid node."""
        if self.entry_point:
            node_ids = {n.node_id for n in self.nodes}
            if self.entry_point not in node_ids:
                raise ValueError(
                    f"Pipeline entry_point '{self.entry_point}' "
                    f"not found in pipeline nodes: {node_ids}"
                )
        return self

    @model_validator(mode="after")
    def validate_path_to_end(self):
        """Validate that at least one path leads to END."""
        has_end = any(e.to_node == "END" for e in self.edges)
        if not has_end:
            raise ValueError("Pipeline must have at least one edge leading to 'END'")
        return self

    def get_entry_node(self) -> str:
        """Get the entry point node ID."""
        return self.entry_point or self.nodes[0].node_id
