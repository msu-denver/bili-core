from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ConditionalEntryPoint:
    """Defines conditional entry point logic."""

    routing_function: Callable[[dict], str | List[str]]
    path_map: Optional[Dict[str, str]] = None
