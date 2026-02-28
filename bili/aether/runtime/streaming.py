"""Streaming event types and helpers for AETHER MAS execution.

Provides ``StreamEvent`` — a structured envelope for events emitted
during async graph execution — and ``StreamFilter`` — a declarative
filter for selecting which events to yield.

Usage::

    from bili.aether.runtime.streaming import StreamEvent, StreamFilter

    async for event in executor.astream(input_data):
        if event.event_type == "token":
            print(event.data["content"], end="", flush=True)
"""

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

LOGGER = logging.getLogger(__name__)


class StreamEventType(str, enum.Enum):
    """Types of events emitted during streaming execution."""

    # Token-level events (from LLM generation)
    TOKEN = "token"

    # Node lifecycle
    NODE_START = "node_start"
    NODE_END = "node_end"

    # Agent lifecycle (wraps one or more nodes)
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # Execution lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Error
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single event emitted during streaming MAS execution.

    Attributes:
        event_type: The kind of event (token, node_start, etc.).
        data: Event-specific payload.
        timestamp: ISO-8601 UTC timestamp when the event was created.
        node_name: Name of the graph node that produced this event.
        agent_id: Agent that owns the current execution context.
        run_id: Unique identifier for the overall execution run.
    """

    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    node_name: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "node_name": self.node_name,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }

    def to_sse(self) -> str:
        """Format as a Server-Sent Events message.

        Returns:
            SSE-formatted string: ``event: <type>\\ndata: <json>\\n\\n``
        """
        import json  # pylint: disable=import-outside-toplevel

        etype = (
            self.event_type.value
            if isinstance(self.event_type, StreamEventType)
            else self.event_type
        )
        return f"event: {etype}\ndata: {json.dumps(self.data)}\n\n"


@dataclass
class StreamFilter:
    """Declarative filter for selecting which stream events to yield.

    By default all event types are included.  Set ``include_types`` to
    restrict to specific types, or ``exclude_types`` to block specific
    types.  ``include_types`` takes precedence if both are set.

    Attributes:
        include_types: If non-empty, only yield events matching these types.
        exclude_types: If non-empty, exclude events matching these types.
        include_agents: If non-empty, only yield events from these agents.
        include_nodes: If non-empty, only yield events from these nodes.
        pass_lifecycle: When ``True`` (the default), ``RUN_START`` and
            ``RUN_END`` events always pass through ``include_agents``
            and ``include_nodes`` filters even though they carry no
            agent or node attribution.
    """

    include_types: Set[str] = field(default_factory=set)
    exclude_types: Set[str] = field(default_factory=set)
    include_agents: Set[str] = field(default_factory=set)
    include_nodes: Set[str] = field(default_factory=set)
    pass_lifecycle: bool = True

    _LIFECYCLE_TYPES: Set[str] = field(
        default_factory=lambda: {StreamEventType.RUN_START, StreamEventType.RUN_END},
        init=False,
        repr=False,
    )

    def accepts(self, event: StreamEvent) -> bool:
        """Return True if the event passes this filter."""
        # Type filtering
        if self.include_types:
            if event.event_type not in self.include_types:
                return False
        elif self.exclude_types:
            if event.event_type in self.exclude_types:
                return False

        # Lifecycle events pass through agent/node filters when pass_lifecycle is True
        is_lifecycle = event.event_type in self._LIFECYCLE_TYPES

        # Agent filtering — reject events with no agent_id when filter is active
        if self.include_agents:
            if not event.agent_id or event.agent_id not in self.include_agents:
                if not (self.pass_lifecycle and is_lifecycle):
                    return False

        # Node filtering — reject events with no node_name when filter is active
        if self.include_nodes:
            if not event.node_name or event.node_name not in self.include_nodes:
                if not (self.pass_lifecycle and is_lifecycle):
                    return False

        return True

    @classmethod
    def tokens_only(cls) -> "StreamFilter":
        """Convenience: only yield token events."""
        return cls(include_types={StreamEventType.TOKEN})

    @classmethod
    def lifecycle_only(cls) -> "StreamFilter":
        """Convenience: only yield lifecycle events (no tokens)."""
        return cls(
            include_types={
                StreamEventType.NODE_START,
                StreamEventType.NODE_END,
                StreamEventType.AGENT_START,
                StreamEventType.AGENT_END,
                StreamEventType.RUN_START,
                StreamEventType.RUN_END,
            }
        )
