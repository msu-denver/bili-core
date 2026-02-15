from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from bili.graph_builder.classes.conditional_edge import ConditionalEdge


@dataclass(eq=False)
class Node:
    """Defines a graph node."""

    name: str
    function: Callable  # The actual node function (not a builder)

    # Edges
    edges: List[str] = field(default_factory=list)  # Normal edges to other nodes
    conditional_edges: List[ConditionalEdge] = field(default_factory=list)

    # Terminal/Entry properties
    is_entry: bool = False  # Is this a standard entry point?
    routes_to_end: bool = False  # Can this node route to END via normal edge?
    conditional_entry: Optional[ConditionalEdge] = None  # Conditional entry from START

    # Optional features
    cache_policy: Optional[Dict[str, Any]] = None  # ttl, key_func
    return_type_annotation: Optional[str] = None  # For Command routing

    def __eq__(self, name):
        if isinstance(name, str):
            return name == self.name
        if isinstance(name, type(self)):
            return self.name == name.name
        # Return NotImplemented for unsupported types (Python protocol)
        # This allows Python to try reverse comparison and provides better error messages
        return NotImplemented

    def __call__(self, **kwargs):
        return self.function(**kwargs)
