"""Graph builder — converts a validated MASConfig into a LangGraph StateGraph.

Supports all seven ``WorkflowType`` values defined in the AETHER schema.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set

from bili.aether.schema import MASConfig, WorkflowType

from .agent_generator import generate_agent_node, wrap_agent_node
from .compiled_mas import CompiledMAS
from .state_generator import generate_state_schema

LOGGER = logging.getLogger(__name__)


class GraphBuilder:  # pylint: disable=too-few-public-methods
    """Builds a LangGraph ``StateGraph`` from a validated ``MASConfig``.

    Usage::

        compiled = GraphBuilder(config).build()
        graph = compiled.compile_graph()
    """

    def __init__(self, config: MASConfig) -> None:
        self._config = config
        self._agent_nodes: Dict[str, Callable] = {}
        self._state_schema = None
        self._graph: Any = None
        self._end_node: Any = None
        self._start_node: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> CompiledMAS:
        """Build the complete ``CompiledMAS`` from the stored config."""
        from langgraph.constants import (  # pylint: disable=import-error,import-outside-toplevel
            END,
            START,
        )
        from langgraph.graph import (  # pylint: disable=import-error,import-outside-toplevel
            StateGraph,
        )

        self._end_node = END
        self._start_node = START

        # 1. Generate state schema
        self._state_schema = generate_state_schema(self._config)

        # 2. Generate agent node callables
        for agent in self._config.agents:
            node_fn = generate_agent_node(agent)
            wrapped = wrap_agent_node(node_fn, agent.agent_id)
            self._agent_nodes[agent.agent_id] = wrapped

        # 3. Create StateGraph
        self._graph = StateGraph(self._state_schema)

        # 4. Add all agent nodes
        for agent_id, node_fn in self._agent_nodes.items():
            self._graph.add_node(agent_id, node_fn)

        # 5. Build workflow-specific edges
        builder = self._get_workflow_builder()
        builder()

        LOGGER.info(
            "Built %s graph for '%s' with %d agent nodes",
            self._config.workflow_type.value,
            self._config.mas_id,
            len(self._agent_nodes),
        )

        # 6. Initialise ChannelManager if channels are configured
        channel_manager = None
        if self._config.channels:
            from bili.aether.runtime.channel_manager import (  # pylint: disable=import-outside-toplevel
                ChannelManager,
            )

            channel_manager = ChannelManager.initialize_from_config(self._config)

        # 7. Return CompiledMAS
        return CompiledMAS(
            config=self._config,
            graph=self._graph,
            state_schema=self._state_schema,
            agent_nodes=dict(self._agent_nodes),
            checkpoint_config=self._config.checkpoint_config,
            channel_manager=channel_manager,
        )

    # ------------------------------------------------------------------
    # Workflow dispatch
    # ------------------------------------------------------------------

    def _get_workflow_builder(self) -> Callable:
        dispatch = {
            WorkflowType.SEQUENTIAL: self._build_sequential,
            WorkflowType.HIERARCHICAL: self._build_hierarchical,
            WorkflowType.SUPERVISOR: self._build_supervisor,
            WorkflowType.CONSENSUS: self._build_consensus,
            WorkflowType.DELIBERATIVE: self._build_deliberative,
            WorkflowType.PARALLEL: self._build_parallel,
            WorkflowType.CUSTOM: self._build_custom,
        }
        builder = dispatch.get(self._config.workflow_type)
        if builder is None:
            raise ValueError(f"Unsupported workflow type: {self._config.workflow_type}")
        return builder

    # ==================================================================
    # SEQUENTIAL — linear chain
    # ==================================================================

    def _build_sequential(self) -> None:
        """``START -> A -> B -> C -> END`` in agent-list order.

        If ``workflow_edges`` are defined, uses those instead.
        """
        if self._config.workflow_edges:
            self._build_from_explicit_edges()
            return

        entry = self._config.get_entry_agent()
        agent_ids = [a.agent_id for a in self._config.agents]

        # Reorder so entry agent comes first
        if entry.agent_id in agent_ids:
            idx = agent_ids.index(entry.agent_id)
            ordered = agent_ids[idx:] + agent_ids[:idx]
        else:
            ordered = agent_ids

        self._graph.add_edge(self._start_node, ordered[0])
        for i in range(len(ordered) - 1):
            self._graph.add_edge(ordered[i], ordered[i + 1])
        self._graph.add_edge(ordered[-1], self._end_node)

    # ==================================================================
    # HIERARCHICAL — tier-based fan-out
    # ==================================================================

    def _build_hierarchical(self) -> None:  # pylint: disable=too-many-branches
        """Process tiers from highest number (leaves) to tier 1 (root).

        Uses channel definitions for specific inter-tier routing when
        available; falls back to full cross-tier connectivity otherwise.
        """
        tiers = sorted(
            {a.tier for a in self._config.agents if a.tier is not None},
            reverse=True,
        )
        if not tiers:
            self._build_sequential()
            return

        # Build channel-based adjacency: source -> {targets}
        channel_targets: Dict[str, Set[str]] = defaultdict(set)
        for ch in self._config.channels:
            channel_targets[ch.source].add(ch.target)
            if ch.bidirectional:
                channel_targets[ch.target].add(ch.source)

        previous_tier_ids: List[str] = []

        for tier_num in tiers:
            tier_ids = [a.agent_id for a in self._config.get_agents_by_tier(tier_num)]

            if tier_num == max(tiers):
                # Leaf tier: START fans out to all agents
                for aid in tier_ids:
                    self._graph.add_edge(self._start_node, aid)
            else:
                # Connect from previous tier
                for prev_id in previous_tier_ids:
                    targets = channel_targets.get(prev_id, set())
                    connected = targets & set(tier_ids)
                    if connected:
                        for curr_id in connected:
                            self._graph.add_edge(prev_id, curr_id)
                    else:
                        for curr_id in tier_ids:
                            self._graph.add_edge(prev_id, curr_id)

            if tier_num == 1:
                for aid in tier_ids:
                    self._graph.add_edge(aid, self._end_node)

            previous_tier_ids = tier_ids

    # ==================================================================
    # SUPERVISOR — hub-and-spoke
    # ==================================================================

    def _build_supervisor(self) -> None:
        """Supervisor receives input, conditionally routes to workers,
        workers return to supervisor, supervisor can route to END.
        """
        entry = self._config.get_entry_agent()
        supervisor_id = entry.agent_id

        worker_ids = [
            a.agent_id for a in self._config.agents if a.agent_id != supervisor_id
        ]

        self._graph.add_edge(self._start_node, supervisor_id)

        # Supervisor -> conditional routing to workers or END
        path_map = {wid: wid for wid in worker_ids}
        path_map["END"] = self._end_node

        def supervisor_router(state: dict) -> str:
            next_agent = state.get("next_agent", "END")
            if next_agent in path_map:
                return next_agent
            return "END"

        self._graph.add_conditional_edges(
            supervisor_id,
            supervisor_router,
            path_map,
        )

        for wid in worker_ids:
            self._graph.add_edge(wid, supervisor_id)

    # ==================================================================
    # CONSENSUS — round-based deliberation
    # ==================================================================

    def _build_consensus(self) -> None:
        """All agents deliberate in rounds until consensus or max rounds.

        Topology::

            START -> __round_start__ -> [all agents] -> __consensus_checker__
                         ^  (continue)                        |
                         +------------------------------------+
                                                     (end) -> END
        """
        agent_ids = [a.agent_id for a in self._config.agents]
        max_rounds = self._config.max_consensus_rounds

        def round_start(_state: dict) -> dict:
            return {}

        self._graph.add_node("__round_start__", round_start)

        def consensus_checker(state: dict) -> dict:
            current_round = (state.get("current_round") or 0) + 1
            consensus_reached = current_round >= max_rounds
            return {
                "current_round": current_round,
                "consensus_reached": consensus_reached,
            }

        self._graph.add_node("__consensus_checker__", consensus_checker)

        self._graph.add_edge(self._start_node, "__round_start__")

        for aid in agent_ids:
            self._graph.add_edge("__round_start__", aid)

        for aid in agent_ids:
            self._graph.add_edge(aid, "__consensus_checker__")

        def consensus_router(state: dict) -> str:
            if state.get("consensus_reached", False):
                return "end"
            current_round = state.get("current_round", 0)
            if current_round >= max_rounds:
                return "end"
            return "continue"

        self._graph.add_conditional_edges(
            "__consensus_checker__",
            consensus_router,
            {"continue": "__round_start__", "end": self._end_node},
        )

    # ==================================================================
    # PARALLEL — fan-out / fan-in
    # ==================================================================

    def _build_parallel(self) -> None:
        """``START -> [all agents] -> END`` (parallel execution)."""
        for agent in self._config.agents:
            self._graph.add_edge(self._start_node, agent.agent_id)
            self._graph.add_edge(agent.agent_id, self._end_node)

    # ==================================================================
    # DELIBERATIVE — delegates to CUSTOM or SEQUENTIAL
    # ==================================================================

    def _build_deliberative(self) -> None:
        """Deliberative workflows use explicit edges when available."""
        if self._config.workflow_edges:
            self._build_custom()
        else:
            self._build_sequential()

    # ==================================================================
    # CUSTOM — explicit edges from workflow_edges
    # ==================================================================

    def _build_custom(self) -> None:
        """Build graph from explicit ``workflow_edges``.

        Handles unconditional edges, conditional edges, and fan-out
        (multiple unconditional edges from the same source).
        """
        entry = self._config.get_entry_agent()
        self._graph.add_edge(self._start_node, entry.agent_id)

        # Group edges by source agent
        edges_by_source: Dict[str, list] = defaultdict(list)
        for edge in self._config.workflow_edges:
            edges_by_source[edge.from_agent].append(edge)

        for from_agent, edges in edges_by_source.items():
            conditional = [e for e in edges if e.condition]
            unconditional = [e for e in edges if not e.condition]

            if conditional:
                self._add_conditional_group(from_agent, conditional, unconditional)
            elif len(unconditional) == 1:
                target = (
                    self._end_node
                    if unconditional[0].to_agent == "END"
                    else unconditional[0].to_agent
                )
                self._graph.add_edge(from_agent, target)
            else:
                self._add_fan_out(from_agent, unconditional)

    def _add_conditional_group(
        self,
        from_agent: str,
        conditional: list,
        unconditional: list,
    ) -> None:
        """Add conditional edges from a single source agent."""
        path_map: Dict[str, Any] = {}
        all_edges = conditional + unconditional

        for edge in all_edges:
            target = self._end_node if edge.to_agent == "END" else edge.to_agent
            key = edge.label or edge.to_agent
            path_map[key] = target

        captured_edges = list(all_edges)

        def _make_router(edges_list: list) -> Callable:
            def router(state: dict) -> str:
                for e in edges_list:
                    if e.condition:
                        try:
                            if eval(  # noqa: S307  pylint: disable=eval-used
                                e.condition,
                                {"__builtins__": {}},
                                {"state": _StateProxy(state)},
                            ):
                                return e.label or e.to_agent
                        except Exception:  # pylint: disable=broad-exception-caught
                            continue
                # Fallback: first unconditional edge
                for e in edges_list:
                    if not e.condition:
                        return e.label or e.to_agent
                return edges_list[-1].label or edges_list[-1].to_agent

            return router

        self._graph.add_conditional_edges(
            from_agent,
            _make_router(captured_edges),
            path_map,
        )

    def _add_fan_out(self, from_agent: str, edges: list) -> None:
        """Add fan-out (multiple unconditional edges) from a single source."""
        targets = []
        path_map: Dict[str, Any] = {}
        for edge in edges:
            target = self._end_node if edge.to_agent == "END" else edge.to_agent
            key = edge.label or edge.to_agent
            path_map[key] = target
            targets.append(key)

        captured_targets = list(targets)

        def fan_out_router(_state: dict) -> list:
            return captured_targets

        self._graph.add_conditional_edges(from_agent, fan_out_router, path_map)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_from_explicit_edges(self) -> None:
        """Build graph from explicit edges (used by sequential fallback)."""
        entry = self._config.get_entry_agent()
        self._graph.add_edge(self._start_node, entry.agent_id)

        has_end = False
        for edge in self._config.workflow_edges:
            target = self._end_node if edge.to_agent == "END" else edge.to_agent
            self._graph.add_edge(edge.from_agent, target)
            if edge.to_agent == "END":
                has_end = True

        if not has_end:
            last_agent = self._config.agents[-1].agent_id
            self._graph.add_edge(last_agent, self._end_node)


class _StateProxy:  # pylint: disable=too-few-public-methods
    """Allows ``state.field`` attribute access for condition evaluation."""

    def __init__(self, state: dict) -> None:
        self._state = state

    def __getattr__(self, name: str) -> Any:
        try:
            return self._state[name]
        except KeyError:
            return None
