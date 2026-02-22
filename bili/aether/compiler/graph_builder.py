"""Graph builder — converts a validated MASConfig into a LangGraph StateGraph.

Supports all seven ``WorkflowType`` values defined in the AETHER schema.
"""

import ast
import logging
import operator
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Set

from bili.aether.schema import MASConfig, WorkflowType

from .agent_generator import generate_agent_node, wrap_agent_node
from .compiled_mas import CompiledMAS
from .state_generator import generate_state_schema

LOGGER = logging.getLogger(__name__)


# Safe expression evaluator for workflow conditions
# Allows: comparisons, boolean ops, attribute access, constants
# Blocks: imports, exec, function calls, comprehensions
class SafeConditionEvaluator(ast.NodeVisitor):
    """AST-based safe evaluator for workflow condition expressions.

    Only allows a restricted subset of Python expressions:
    - Attribute access (state.field)
    - Comparisons (==, !=, <, >, <=, >=, in, not in)
    - Boolean operations (and, or, not)
    - Constants (True, False, None, numbers, strings)
    - Basic arithmetic (+, -, *, /, //, %, **)

    Blocks all dangerous operations like imports, exec, function calls, etc.
    """

    # Mapping of AST comparison operators to Python operators
    _COMPARISON_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    # Mapping of AST binary operators to Python operators
    _BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Mapping of AST boolean operators to Python operators
    _BOOL_OPS = {
        ast.And: lambda values: all(values),
        ast.Or: lambda values: any(values),
    }

    # Mapping of AST unary operators to Python operators
    _UNARY_OPS = {
        ast.Not: operator.not_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def eval(self, expression: str) -> Any:
        """Safely evaluate a Python expression string."""
        try:
            tree = ast.parse(expression, mode="eval")
            return self.visit(tree.body)
        except (SyntaxError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise ValueError(f"Invalid condition expression: {expression}") from exc

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Allow constants (True, False, None, numbers, strings)."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        """Allow name lookups from context."""
        try:
            return self.context[node.id]
        except KeyError as exc:
            raise ValueError(f"Undefined variable: {node.id}") from exc

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Allow attribute access (e.g., state.field)."""
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def visit_Compare(self, node: ast.Compare) -> bool:
        """Allow comparison operations."""
        left = self.visit(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_func = self._COMPARISON_OPS.get(type(op))
            if op_func is None:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            result = result and op_func(left, right)
            left = right
        return result

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        """Allow boolean operations (and, or)."""
        op_func = self._BOOL_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        values = [self.visit(val) for val in node.values]
        return op_func(values)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Allow unary operations (not, +, -)."""
        op_func = self._UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = self.visit(node.operand)
        return op_func(operand)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Allow binary arithmetic operations."""
        op_func = self._BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return op_func(left, right)

    def generic_visit(self, node: ast.AST) -> None:
        """Block all other node types (function calls, imports, etc.)."""
        raise ValueError(
            f"Unsupported expression type: {type(node).__name__}. "
            f"Only comparisons, boolean operations, and attribute access are allowed."
        )


def safe_eval_condition(condition: str, context: Dict[str, Any]) -> bool:
    """Safely evaluate a condition expression.

    Args:
        condition: Python expression string (e.g., "state.field == True")
        context: Variable context for evaluation (e.g., {"state": state_proxy})

    Returns:
        Boolean result of the condition evaluation

    Raises:
        ValueError: If the expression contains unsafe operations
    """
    evaluator = SafeConditionEvaluator(context)
    result = evaluator.eval(condition)
    # Ensure result is boolean
    return bool(result)


class GraphBuilder:  # pylint: disable=too-few-public-methods
    """Builds a LangGraph ``StateGraph`` from a validated ``MASConfig``.

    Usage::

        compiled = GraphBuilder(config).build()
        graph = compiled.compile_graph()
    """

    def __init__(self, config: MASConfig) -> None:
        # Make a deep copy to avoid mutating caller's config during compilation
        self._config = config.model_copy(deep=True)
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

        # 1b. Apply bili-core inheritance to agents that opted in
        self._apply_inheritance()

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

        # 6. Return CompiledMAS (ChannelManager removed - using state-based communication)
        return CompiledMAS(
            config=self._config,
            graph=self._graph,
            state_schema=self._state_schema,
            agent_nodes=dict(self._agent_nodes),
            checkpoint_config=self._config.checkpoint_config,
        )

    # ------------------------------------------------------------------
    # Inheritance
    # ------------------------------------------------------------------

    def _apply_inheritance(self) -> None:
        """Apply bili-core inheritance to agents with ``inherit_from_bili_core=True``.

        Replaces ``self._config.agents`` with enriched copies where
        inheritance is enabled.  Gracefully skips if the integration
        package is unavailable.
        """
        has_inheritance = any(a.inherit_from_bili_core for a in self._config.agents)
        if not has_inheritance:
            return

        try:
            from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
                apply_inheritance_to_all,
            )
        except ImportError:
            LOGGER.warning(
                "bili.aether.integration not available; "
                "skipping inheritance resolution for %d agent(s)",
                sum(1 for a in self._config.agents if a.inherit_from_bili_core),
            )
            return

        enriched = apply_inheritance_to_all(self._config.agents, self._config)
        self._config.agents = enriched
        LOGGER.info("Applied bili-core inheritance to agents")

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
            """Check if agents have reached consensus based on their votes.

            Extracts votes from agent outputs and checks if the agreement ratio
            meets the configured consensus_threshold. Falls back to round limit
            if threshold not met.
            """
            current_round = (state.get("current_round") or 0) + 1
            agent_outputs = state.get("agent_outputs", {})
            threshold = self._config.consensus_threshold or 0.66

            # Extract votes from agent outputs
            votes = {}
            for agent_id, output in agent_outputs.items():
                vote = None

                # Try to get vote from consensus_vote_field if configured
                agent_spec = next(
                    (a for a in self._config.agents if a.agent_id == agent_id), None
                )
                if agent_spec and agent_spec.consensus_vote_field:
                    # Try to extract from parsed JSON output
                    if "parsed" in output and isinstance(output["parsed"], dict):
                        vote = output["parsed"].get(agent_spec.consensus_vote_field)

                # Fallback: look for common vote patterns in message
                if not vote and "message" in output:
                    message = output["message"].lower()
                    if "vote:" in message or "decision:" in message:
                        # Extract vote after "vote:" or "decision:"
                        match = re.search(
                            r"(?:vote|decision):\s*(\w+)", message, re.IGNORECASE
                        )
                        if match:
                            vote = match.group(1)

                if vote:
                    votes[agent_id] = str(vote).lower()

            # Check if consensus reached
            consensus_reached = False
            if len(votes) >= 2:  # Need at least 2 agents to have consensus
                # Count votes for each option
                vote_counts = Counter(votes.values())
                most_common_vote, count = vote_counts.most_common(1)[0]

                # Check if agreement ratio meets threshold
                agreement_ratio = count / len(votes)
                consensus_reached = agreement_ratio >= threshold

                LOGGER.info(
                    "Consensus check (round %d): %d/%d agents agree on '%s' (%.1f%%, threshold %.1f%%)",
                    current_round,
                    count,
                    len(votes),
                    most_common_vote,
                    agreement_ratio * 100,
                    threshold * 100,
                )

            # Fall back to round limit if no consensus yet
            if not consensus_reached and current_round >= max_rounds:
                consensus_reached = True
                LOGGER.info(
                    "Consensus reached by round limit (round %d/%d)",
                    current_round,
                    max_rounds,
                )

            return {
                "current_round": current_round,
                "consensus_reached": consensus_reached,
                "votes": votes,
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
                            if safe_eval_condition(
                                e.condition,
                                {"state": _StateProxy(state)},
                            ):
                                return e.label or e.to_agent
                        except ValueError as exc:
                            # Log the error for debugging but continue to next condition
                            LOGGER.warning(
                                "Condition evaluation failed for edge to %s: %s",
                                e.to_agent,
                                exc,
                            )
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
    """Allows ``state.field`` attribute access for condition evaluation.

    Raises AttributeError for missing state fields to help catch typos in
    condition expressions rather than silently returning None.
    """

    def __init__(self, state: dict) -> None:
        self._state = state

    def __getattr__(self, name: str) -> Any:
        try:
            return self._state[name]
        except KeyError as exc:
            raise AttributeError(
                f"State field '{name}' not found. Available fields: "
                f"{', '.join(sorted(self._state.keys()))}"
            ) from exc
