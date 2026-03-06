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

import ast
import logging
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)

# Maximum nesting depth for pipelines containing agents with their own pipelines
MAX_PIPELINE_DEPTH = 3

# Reserved state field names that cannot be used in state_fields
_RESERVED_STATE_FIELDS = frozenset({"messages", "current_agent", "agent_outputs"})

# Safe type string → actual Python type mapping
_SAFE_TYPE_MAP: Dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "Any": Any,
    "List[str]": List[str],
    "List[int]": List[int],
    "List[float]": List[float],
    "List[dict]": List[dict],
    "Dict[str, Any]": Dict[str, Any],
    "Dict[str, str]": Dict[str, str],
    "Optional[str]": Optional[str],
    "Optional[int]": Optional[int],
    "Optional[float]": Optional[float],
    "Optional[bool]": Optional[bool],
}

# Safe reducer strategy name → reducer callable mapping
_REDUCER_MAP: Dict[str, Any] = {
    "replace": lambda _old, new: new,
    "append": lambda old, new: (old or []) + (new if isinstance(new, list) else [new]),
}


class PipelineStateField(BaseModel):
    """Declares a custom state field for a pipeline's inner state.

    Custom fields are added alongside the three built-in fields
    (``messages``, ``current_agent``, ``agent_outputs``) and are
    accessible within pipeline nodes and conditional edge expressions.

    Examples:
        >>> field = PipelineStateField(
        ...     name="sentiment_score", type="float", default=0.0, reducer="replace"
        ... )
    """

    name: str = Field(
        ...,
        description="Field name (must be a valid Python identifier)",
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z_][a-zA-Z0-9_]*$",
    )

    type: str = Field(
        "Any",
        description=(
            "Python type hint as string. Supported: "
            + ", ".join(sorted(_SAFE_TYPE_MAP.keys()))
        ),
    )

    default: Optional[Any] = Field(
        None,
        description="Default value for this field when not present in outer state",
    )

    reducer: Optional[str] = Field(
        None,
        description=(
            "Reducer strategy for concurrent writes: "
            "'replace' (last-writer-wins), 'append' (list concatenation), "
            "or None (no reducer — standard assignment)"
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_not_reserved(cls, v):
        """Ensure custom field names don't shadow built-in state fields."""
        if v in _RESERVED_STATE_FIELDS:
            raise ValueError(
                f"Field name '{v}' is reserved. Built-in state fields "
                f"({', '.join(sorted(_RESERVED_STATE_FIELDS))}) cannot be overridden."
            )
        return v

    @field_validator("type")
    @classmethod
    def validate_type_string(cls, v):
        """Ensure the type string maps to a known Python type."""
        if v not in _SAFE_TYPE_MAP:
            raise ValueError(
                f"Unsupported type '{v}'. Supported types: "
                f"{', '.join(sorted(_SAFE_TYPE_MAP.keys()))}"
            )
        return v

    @field_validator("reducer")
    @classmethod
    def validate_reducer_string(cls, v):
        """Ensure the reducer string maps to a known strategy."""
        if v is not None and v not in _REDUCER_MAP:
            raise ValueError(
                f"Unknown reducer '{v}'. Supported: "
                f"{', '.join(sorted(_REDUCER_MAP.keys()))}"
            )
        return v

    def resolve_type(self) -> Any:
        """Resolve the type string to an actual Python type."""
        return _SAFE_TYPE_MAP[self.type]

    def resolve_reducer(self) -> Any:
        """Resolve the reducer string to a callable, or None."""
        if self.reducer is None:
            return None
        return _REDUCER_MAP[self.reducer]


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
        description=(
            "Python expression for conditional routing. Evaluated by "
            "SafeConditionEvaluator with ``state.<field>`` access. "
            "Pipeline inner state exposes ``messages``, "
            "``current_agent``, ``agent_outputs``, and any custom "
            "fields declared in ``PipelineSpec.state_fields``. "
            "Function calls (e.g. ``len()``, ``.get()``) are blocked "
            "by the safe evaluator."
        ),
    )

    label: str = Field("", description="Edge label for visualization")

    @field_validator("condition")
    @classmethod
    def validate_condition_syntax(cls, v):
        """Validate that condition is a parseable Python expression."""
        if v is not None:
            try:
                ast.parse(v, mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"Invalid condition expression: {v!r} — {exc.msg}"
                ) from exc
        return v


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

    state_fields: List[PipelineStateField] = Field(
        default_factory=list,
        description=(
            "Custom state fields for the pipeline's inner state. "
            "Added alongside the built-in fields (messages, current_agent, "
            "agent_outputs). Accessible in pipeline nodes and conditional "
            "edge expressions via state.<field_name>."
        ),
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
