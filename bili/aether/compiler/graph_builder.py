"""Graph builder — converts a validated MASConfig into a LangGraph StateGraph.

Supports all seven ``WorkflowType`` values defined in the AETHER schema.
Supports agent pipeline sub-graphs (PipelineSpec) compiled as inner
StateGraphs and embedded as single nodes in the parent MAS graph.
"""

import ast
import logging
import operator
import re
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Set

from bili.aether.schema import MASConfig, WorkflowType

from .agent_generator import generate_agent_node, wrap_agent_node
from .compiled_mas import CompiledMAS
from .state_generator import _merge_dicts, generate_state_schema

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

    def __init__(
        self,
        config: MASConfig,
        custom_node_registry: Dict[str, Any] | None = None,
    ) -> None:
        # Make a deep copy to avoid mutating caller's config during compilation
        self._config = config.model_copy(deep=True)
        self._custom_node_registry: Dict[str, Any] = custom_node_registry or {}
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
            if agent.pipeline:
                node_fn = self._compile_pipeline_node(agent)
            else:
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
    # Pipeline sub-graph compilation
    # ------------------------------------------------------------------

    def _compile_pipeline_node(self, agent) -> Callable[[dict], dict]:
        """Compile an agent's pipeline into a sub-graph node callable.

        When an ``AgentSpec`` has a ``pipeline``, its internal nodes/edges
        are compiled into a LangGraph ``CompiledStateGraph`` and wrapped
        in a state adapter that maps between the outer MAS state and
        the inner pipeline state.

        The sub-graph is compiled with ``checkpointer=None`` to avoid
        the checkpointer propagation bug (LangGraph issue #5639) where
        ``compile_graph()`` auto-attaches a checkpointer to sub-graphs.

        Args:
            agent: An ``AgentSpec`` whose ``pipeline`` is not ``None``.

        Returns:
            A callable ``(state: dict) -> dict`` suitable for
            ``StateGraph.add_node``.
        """
        from langgraph.constants import (  # pylint: disable=import-error,import-outside-toplevel
            END,
            START,
        )
        from langgraph.graph import (  # pylint: disable=import-error,import-outside-toplevel
            StateGraph,
            add_messages,
        )
        from typing_extensions import (  # pylint: disable=import-error,import-outside-toplevel
            Annotated,
            TypedDict,
        )

        pipeline = agent.pipeline

        # 1. Build inner state schema (base fields + custom state_fields)
        inner_annotations: Dict[str, Any] = {
            "messages": Annotated[list, add_messages],
            "current_agent": Annotated[str, lambda _old, new: new],
            "agent_outputs": Annotated[Dict[str, Any], _merge_dicts],
        }
        for field in pipeline.state_fields:
            type_hint = field.resolve_type()
            reducer = field.resolve_reducer()
            if reducer is not None:
                inner_annotations[field.name] = Annotated[type_hint, reducer]
            else:
                inner_annotations[field.name] = type_hint
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", agent.agent_id)
        inner_state_cls: type = TypedDict(
            f"{safe_name}_pipeline_State", inner_annotations
        )  # type: ignore[call-overload]

        # 2. Resolve inner node callables
        inner_nodes: Dict[str, Callable] = {}
        for node_spec in pipeline.nodes:
            inner_nodes[node_spec.node_id] = self._resolve_pipeline_node(
                node_spec, agent
            )

        # 3. Build inner StateGraph
        inner_graph = StateGraph(inner_state_cls)
        for node_id, node_fn in inner_nodes.items():
            inner_graph.add_node(node_id, node_fn)

        # 4. Wire edges
        entry = pipeline.get_entry_node()
        inner_graph.add_edge(START, entry)

        # Group edges by source for conditional edge handling
        edges_by_source: Dict[str, list] = defaultdict(list)
        for edge in pipeline.edges:
            edges_by_source[edge.from_node].append(edge)

        for from_node, edges in edges_by_source.items():
            conditional = [e for e in edges if e.condition]
            unconditional = [e for e in edges if not e.condition]

            if conditional:
                self._add_pipeline_conditional_edges(
                    inner_graph, from_node, conditional, unconditional, END
                )
            else:
                for edge in unconditional:
                    target = END if edge.to_node == "END" else edge.to_node
                    inner_graph.add_edge(from_node, target)

        # 5. Compile with checkpointer=None (Monica's catch — LangGraph #5639)
        compiled_subgraph = inner_graph.compile(checkpointer=None)

        LOGGER.info(
            "Compiled pipeline sub-graph for agent '%s' with %d nodes",
            agent.agent_id,
            len(pipeline.nodes),
        )

        # 6. Wrap in state adapter
        return self._wrap_pipeline_as_agent_node(compiled_subgraph, agent)

    def _resolve_pipeline_node(self, node_spec, parent_agent) -> Callable[[dict], dict]:
        """Resolve a pipeline node spec to a callable.

        - ``node_type="agent"``: generates a node from the inline agent spec.
        - Other values: looks up from bili-core's ``GRAPH_NODE_REGISTRY``.
        """
        if node_spec.node_type == "agent":
            from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
                AgentSpec,
            )

            inner_agent = AgentSpec(**node_spec.agent_spec)
            return generate_agent_node(inner_agent)

        return self._resolve_registry_node(node_spec, parent_agent)

    def _resolve_registry_node(self, node_spec, parent_agent) -> Callable[[dict], dict]:
        """Resolve a registry-based pipeline node.

        Checks the per-compilation ``custom_node_registry`` first, then
        falls back to bili-core's global ``GRAPH_NODE_REGISTRY``.
        Instantiates via the ``functools.partial(Node, name, builder)``
        pattern and calls ``node_instance.function(**kwargs)`` to obtain
        the actual executor function.

        Args:
            node_spec: The ``PipelineNodeSpec`` to resolve.
            parent_agent: The parent ``AgentSpec`` (provides fallback config).

        Returns:
            A callable ``(state: dict) -> dict``.
        """
        # 1. Check per-compilation custom registry first
        factory = self._custom_node_registry.get(node_spec.node_type)

        # 2. Fall back to global bili-core registry
        if factory is None:
            try:
                from bili.loaders.langchain_loader import (  # pylint: disable=import-outside-toplevel
                    GRAPH_NODE_REGISTRY,
                )
            except ImportError as exc:
                raise ValueError(
                    f"Pipeline node '{node_spec.node_id}' references registry "
                    f"type '{node_spec.node_type}' but "
                    "bili.loaders.langchain_loader is not available."
                ) from exc
            factory = GRAPH_NODE_REGISTRY.get(node_spec.node_type)

        if factory is None:
            # Build combined list of available node types for the error message
            available_keys: Set[str] = set(self._custom_node_registry.keys())
            try:
                from bili.loaders.langchain_loader import (
                    GRAPH_NODE_REGISTRY as _global_reg,  # pylint: disable=import-outside-toplevel
                )

                available_keys |= set(_global_reg.keys())
            except ImportError:
                pass
            raise ValueError(
                f"Pipeline node '{node_spec.node_id}' references unknown "
                f"registry type '{node_spec.node_type}'. "
                f"Available: {sorted(available_keys)}"
            )

        from bili.graph_builder.classes.node import (  # pylint: disable=import-outside-toplevel
            Node,
        )

        # Call partial to get Node instance, then call builder with kwargs
        if callable(factory) and not isinstance(factory, Node):
            node_instance = factory()
        else:
            node_instance = factory

        # Build kwargs from parent agent context + node-specific config
        kwargs = self._build_registry_node_kwargs(parent_agent, node_spec)

        return node_instance.function(**kwargs)

    def _build_registry_node_kwargs(self, parent_agent, node_spec) -> Dict[str, Any]:
        """Build kwargs for a registry node builder function.

        Merges configuration from the parent agent (LLM, tools, persona)
        with node-specific config overrides from the ``PipelineNodeSpec``.
        """
        kwargs: Dict[str, Any] = {}

        # Resolve LLM from parent agent if available
        if parent_agent.model_name:
            try:
                from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
                    create_llm,
                    resolve_tools,
                )

                kwargs["llm_model"] = create_llm(parent_agent)

                # Also resolve tools if the parent agent has them
                if parent_agent.tools:
                    kwargs["tools"] = resolve_tools(parent_agent)
            except Exception:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    "Could not resolve LLM/tools for pipeline node '%s'",
                    node_spec.node_id,
                    exc_info=True,
                )

        # Use agent objective as fallback persona
        if parent_agent.system_prompt:
            kwargs["persona"] = parent_agent.system_prompt
        elif parent_agent.objective:
            kwargs["persona"] = parent_agent.objective

        # Merge node-specific config (overrides parent-level kwargs)
        kwargs.update(node_spec.config)

        return kwargs

    @staticmethod
    def _add_pipeline_conditional_edges(
        graph, from_node: str, conditional: list, unconditional: list, end_node
    ) -> None:
        """Add conditional edges within a pipeline sub-graph.

        Uses the same ``safe_eval_condition`` mechanism as the parent
        MAS graph for consistent condition evaluation.
        """
        path_map: Dict[str, Any] = {}
        all_edges = conditional + unconditional

        for edge in all_edges:
            target = end_node if edge.to_node == "END" else edge.to_node
            key = edge.label or edge.to_node
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
                                return e.label or e.to_node
                        except ValueError as exc:
                            LOGGER.warning(
                                "Pipeline condition evaluation failed for "
                                "edge to %s: %s",
                                e.to_node,
                                exc,
                            )
                            continue
                # Fallback: first unconditional edge
                for e in edges_list:
                    if not e.condition:
                        return e.label or e.to_node
                return edges_list[-1].label or edges_list[-1].to_node

            return router

        graph.add_conditional_edges(
            from_node,
            _make_router(captured_edges),
            path_map,
        )

    @staticmethod
    def _wrap_pipeline_as_agent_node(
        compiled_subgraph, agent
    ) -> Callable[[dict], dict]:
        """Wrap a compiled sub-graph as an agent node callable.

        Creates a function that:

        1. Maps outer MAS state → inner pipeline state (messages only)
        2. Invokes the compiled sub-graph
        3. Maps inner result → outer state update (explicit output mapping:
           only ``messages`` and ``agent_outputs`` flow back)
        4. Handles errors with attribution (``agent_id: pipeline error``)

        This explicit output mapping prevents the "blind merge" danger
        that Monica identified — inner pipeline state does NOT overwrite
        arbitrary outer MAS state fields.
        """
        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
        )

        agent_id = agent.agent_id
        agent_role = agent.role
        pipeline_node_count = len(agent.pipeline.nodes)
        custom_state_fields = agent.pipeline.state_fields

        def _pipeline_node(state: dict) -> dict:
            start = time.time()

            # Input mapping: outer → inner (messages + custom state fields)
            inner_state: Dict[str, Any] = {
                "messages": list(state.get("messages", [])),
                "current_agent": agent_id,
                "agent_outputs": {},
            }
            # Carry custom state fields from outer state into inner state
            for field in custom_state_fields:
                if field.name in state:
                    inner_state[field.name] = state[field.name]
                elif field.default is not None:
                    inner_state[field.name] = field.default

            try:
                result = compiled_subgraph.invoke(inner_state)
            except Exception as exc:
                # Error attribution: clearly identify which agent's pipeline failed
                error_msg = f"[{agent_id}] Pipeline error: {exc}"
                LOGGER.error(
                    "Pipeline execution failed for agent '%s': %s",
                    agent_id,
                    exc,
                    exc_info=True,
                )

                agent_outputs = dict(state.get("agent_outputs") or {})
                agent_outputs[agent_id] = {
                    "agent_id": agent_id,
                    "role": agent_role,
                    "status": "error",
                    "message": error_msg,
                }
                return {
                    "messages": [AIMessage(content=error_msg, name=agent_id)],
                    "current_agent": agent_id,
                    "agent_outputs": agent_outputs,
                }

            # Output mapping: inner → outer (explicit — only messages + agent_outputs)
            inner_messages = result.get("messages", [])
            inner_outputs = result.get("agent_outputs", {})

            # Extract final content from the last message in the pipeline
            final_content = f"[{agent_id}] Pipeline completed."
            for msg in reversed(inner_messages):
                if hasattr(msg, "content") and msg.content:
                    final_content = msg.content
                    break

            # Build agent output entry
            agent_outputs = dict(state.get("agent_outputs") or {})
            agent_outputs[agent_id] = {
                "agent_id": agent_id,
                "role": agent_role,
                "status": "completed",
                "message": final_content,
                "pipeline_outputs": inner_outputs,
            }

            execution_ms = (time.time() - start) * 1000
            LOGGER.info(
                "Agent node '%s' executed in %.2f ms (pipeline, %d inner nodes)",
                agent_id,
                execution_ms,
                pipeline_node_count,
            )

            output = {
                "messages": [AIMessage(content=final_content, name=agent_id)],
                "current_agent": agent_id,
                "agent_outputs": agent_outputs,
            }
            # Propagate custom state fields from inner result back to outer state
            for field in custom_state_fields:
                if field.name in result:
                    output[field.name] = result[field.name]
            return output

        _pipeline_node.__name__ = f"pipeline_{agent_id}"
        _pipeline_node.__qualname__ = f"pipeline_{agent_id}"

        return _pipeline_node

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

    Used by both MAS-level and pipeline-level conditional edge routers.
    Note that the available state fields differ between contexts: outer MAS
    state includes fields like ``current_agent`` and ``agent_outputs``, while
    inner pipeline state has a simpler schema (``messages`` and
    ``agent_outputs`` only).

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
