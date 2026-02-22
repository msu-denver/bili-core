"""Compiled MAS container — wraps a LangGraph StateGraph built from a MASConfig."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

from bili.aether.schema import MASConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class CompiledMAS:
    """Result of compiling a MASConfig into a LangGraph StateGraph.

    Attributes:
        config: The original MASConfig used to build the graph.
        graph: The uncompiled StateGraph (can be inspected/modified).
        state_schema: The generated TypedDict state class.
        agent_nodes: Mapping of agent_id to its callable node function.
        checkpoint_config: Checkpoint backend configuration dict.
    """

    config: MASConfig
    graph: Any  # langgraph.graph.StateGraph (lazy import)
    state_schema: Type
    agent_nodes: Dict[str, Callable]
    checkpoint_config: Dict[str, Any] = field(default_factory=dict)

    def compile_graph(self, checkpointer=None):
        """Compile the StateGraph into an executable CompiledStateGraph.

        Args:
            checkpointer: Optional checkpoint saver.  If *None* and
                ``config.checkpoint_enabled`` is True, creates a
                checkpointer from ``config.checkpoint_config`` using
                the bili-core factory (falls back to MemorySaver).

        Returns:
            A ``CompiledStateGraph`` ready for ``.invoke()`` or ``.stream()``.
        """
        if checkpointer is None and self.config.checkpoint_enabled:
            checkpointer = self._create_checkpointer()

        return self.graph.compile(checkpointer=checkpointer)

    def _create_checkpointer(self) -> Any:
        """Create a checkpointer from ``config.checkpoint_config``.

        Uses the bili-core checkpointer factory when available;
        falls back to ``MemorySaver`` otherwise.
        """
        try:
            from bili.aether.integration.checkpointer_factory import (  # pylint: disable=import-outside-toplevel
                create_checkpointer_from_config,
            )

            return create_checkpointer_from_config(self.config.checkpoint_config)
        except ImportError:
            LOGGER.debug("Checkpointer factory not available; using MemorySaver")

        from langgraph.checkpoint.memory import (  # pylint: disable=import-error,import-outside-toplevel
            MemorySaver,
        )

        return MemorySaver()

    def get_agent_node(self, agent_id: str) -> Optional[Callable]:
        """Look up the callable for a specific agent node."""
        return self.agent_nodes.get(agent_id)

    def __str__(self) -> str:
        return (
            f"CompiledMAS({self.config.mas_id}, "
            f"{len(self.agent_nodes)} agents, "
            f"workflow={self.config.workflow_type.value})"  # pylint: disable=no-member
        )
