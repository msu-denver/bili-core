from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ConditionalEdge:
    """Defines a conditional edge from a source node."""

    source_node: str
    routing_function: Callable[
        [dict], str | List[str]
    ]  # Takes state, returns node name(s)
    path_map: Optional[Dict[bool, str]] = (
        None  # Optional mapping of routing output to node names
    )
