"""MAS Execution Controller — runs a compiled MAS graph and collects results.

Wraps the AETHER compiler and LangGraph execution pipeline to provide
a structured ``MASExecutionResult`` with per-agent outputs, timing,
communication statistics, and checkpoint metadata.

Supports four execution modes:

- **Synchronous** (``run()``): blocking execution, returns a result.
- **Sync streaming** (``stream()``): yields ``StreamEvent`` objects
  using LangGraph's synchronous ``.stream()`` method.
- **Async streaming** (``astream()``): yields ``StreamEvent`` objects
  using LangGraph's asynchronous ``.astream_events()`` method for
  token-level granularity.
- **UI streaming** (``run_streaming()``): yields ``(node_name, state_update)``
  tuples as each graph node completes, for lightweight UI consumers.

Usage::

    from bili.aether.runtime.executor import MASExecutor, execute_mas

    executor = MASExecutor(config, log_dir="logs")
    executor.initialize()
    result = executor.run({"messages": [HumanMessage(content="Hello")]})

    # Synchronous streaming:
    for event in executor.stream({"messages": [HumanMessage(content="Hi")]}):
        if event.event_type == "token":
            print(event.data["content"], end="", flush=True)

    # Async streaming:
    async for event in executor.astream({"messages": [HumanMessage(content="Hi")]}):
        if event.event_type == "token":
            print(event.data["content"], end="", flush=True)

    # UI streaming — yields (node_name, state_update) tuples:
    for node_name, state_update in executor.run_streaming(
        {"messages": [HumanMessage(content="Hi")]}, thread_id="my-thread"
    ):
        print(f"{node_name}: {state_update}")

    # UI token streaming — interleaves per-token chunks with node-completion sentinels:
    for event_type, event_data in executor.run_streaming_tokens(
        {"messages": [HumanMessage(content="Hi")]}, thread_id="my-thread"
    ):
        if event_type == "__token__":
            print(event_data["token"], end="", flush=True)
        elif event_type == "__node_complete__":
            print(f"\n[{event_data['node']} complete]")

    # ask_user (or any tool calling langgraph.types.interrupt()) pauses the
    # run_streaming() generator with a __ask_user_pending__ sentinel;
    # resume_with_value() supplies the answer and continues execution:
    for node_name, state_update in executor.run_streaming(
        {"messages": [HumanMessage(content="Deploy the app.")]}, thread_id="my-thread"
    ):
        if node_name == "__ask_user_pending__":
            question = state_update["interrupts"][0]["question"]
            for node_name2, state_update2 in executor.resume_with_value(
                "staging", thread_id="my-thread"
            ):
                print(f"{node_name2}: {state_update2}")

    # Or use the convenience function:
    result = execute_mas(config, {"messages": [HumanMessage(content="Hello")]})

Multimodal input
----------------
``input_data["messages"]`` is a list of ``BaseMessage``, so an image reaches a
run by building the message with an image content part::

    from bili.iris.multimodal import build_human_message

    result = executor.run({
        "messages": [build_human_message(
            text="What does this diagram show?",
            images=["https://example.invalid/diagram.png"],
        )]
    })

Whether the agents' bound models accept an image is a per-model question;
``bili.iris.providers.modality`` answers it from the catalog.
"""

import json
import logging
import os
import queue
import time
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from bili.aether.runtime.context import RuntimeContext
from bili.aether.runtime.execution_result import (
    AgentExecutionResult,
    MASExecutionResult,
)
from bili.aether.runtime.streaming import StreamEvent, StreamEventType, StreamFilter
from bili.aether.schema import MASConfig, WorkflowType
from bili.iris.multimodal import normalise_prompt

LOGGER = logging.getLogger(__name__)


class MASExecutor:  # pylint: disable=too-many-instance-attributes
    """Executes a MAS configuration end-to-end and collects results.

    Attributes:
        config: The ``MASConfig`` being executed.
        log_dir: Directory for logs and result files.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        config: MASConfig,
        log_dir: Optional[str] = None,
        validate_config: bool = True,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        custom_node_registry: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[RuntimeContext] = None,
        enable_steering: bool = False,
    ) -> None:
        """Initialize the executor.

        Args:
            config: A ``MASConfig`` instance.
            log_dir: Directory for communication logs and result files.
                Defaults to the current working directory.
            validate_config: Whether ``compile_mas()`` should validate
                the config (it always does; this flag is reserved for
                future use).
            user_id: Optional user identifier for multi-tenant security.
                If provided, checkpointer will enforce thread ownership
                validation and thread_ids will follow the pattern
                ``{user_id}_{conversation_id}``.
            conversation_id: Optional conversation identifier for
                multi-conversation support. Used with ``user_id`` to
                construct unique thread_ids.
            custom_node_registry: Optional mapping of node names to
                factory callables for pipeline ``node_type`` resolution.
                Checked before the global ``GRAPH_NODE_REGISTRY``.
            runtime_context: Optional :class:`RuntimeContext` holding
                named services injected into pipeline node builders.
            enable_steering: When ``True``, an operator may inject a
                directive into a running graph that the next node observes
                at the next superstep boundary (see :meth:`submit_steer`,
                :meth:`steer`, :meth:`run_streaming_steerable`). This makes
                :meth:`initialize` pause after every agent node
                (``interrupt_after``) and guarantees a checkpointer is
                attached, so directives can be applied via ``update_state``
                and picked up on resume. Defaults to ``False``: when unset,
                compilation and every existing run/stream path are
                unchanged (no interrupt points, no directive queue).
        """
        self._config = config
        self._log_dir = log_dir or os.getcwd()
        self._validate_config = validate_config
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._custom_node_registry = custom_node_registry
        self._runtime_context = runtime_context
        self._enable_steering = enable_steering
        # Thread-safe queue of operator directives drained at each superstep
        # boundary by run_streaming_steerable(). Only allocated when steering
        # is enabled, so the feature is inert (and allocates nothing) by
        # default.
        self._steer_queue: Optional["queue.Queue[str]"] = (
            queue.Queue() if enable_steering else None
        )
        self._compiled_mas = None
        self._compiled_graph = None
        # Populated by initialize(); used by run/stream methods to decide
        # whether to inject a thread_id into invoke_config.
        self._checkpointer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> MASConfig:
        """The MAS configuration."""
        return self._config

    @property
    def log_dir(self) -> str:
        """Log output directory."""
        return self._log_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Compile the MAS config into an executable LangGraph.

        Calls ``compile_mas()`` (which validates the config) and then
        ``compile_graph()`` to produce the executable graph.

        Checkpointer attachment (mirrors IRIS always-on behaviour):
        - ``checkpoint_enabled=True`` (the default) always attaches a
          checkpointer, even without ``user_id``:
            - With ``user_id``: ownership-validating saver keyed to that user.
            - Without ``user_id``: configured backend (default: memory).
        - ``checkpoint_enabled=False``: explicit opt-out; no saver attached
          (HITL override still applies).
        - HITL: a ``MemorySaver`` is always attached when human-interrupt
          nodes are present, even when ``checkpoint_enabled=False``.

        After initialization, ``self._checkpointer`` reflects whether a
        checkpointer is actually attached (``None`` = no persistence).

        Raises:
            ValueError: If config validation fails.
        """
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        self._compiled_mas = compile_mas(
            self._config,
            custom_node_registry=self._custom_node_registry,
            runtime_context=self._runtime_context,
        )

        # Determine human-interrupt nodes for HITL configs
        human_nodes = []
        if self._config.human_in_loop:
            human_nodes = [
                a.agent_id for a in self._config.agents if getattr(a, "is_human", False)
            ]

        # Always-on checkpointing: attach a checkpointer whenever
        # checkpoint_enabled=True, regardless of whether user_id is set.
        # checkpoint_enabled=False is the explicit opt-out.
        checkpointer = None
        if self._config.checkpoint_enabled:
            if self._user_id:
                # Multi-tenant: ownership-validating saver keyed to this user
                checkpointer = self._create_checkpointer_with_user_id()
            else:
                # Local / single-instance: configured backend, no ownership
                # validation
                checkpointer = self._create_checkpointer_local()

        # HITL override: state resumption requires a checkpointer even when
        # checkpoint_enabled=False.  Fall back to an in-process MemorySaver.
        if human_nodes and checkpointer is None:
            from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
                MemorySaver,
            )

            checkpointer = MemorySaver()
            LOGGER.info(
                "HITL mode: attaching MemorySaver for state resumption "
                "(checkpoint_enabled=False)"
            )

        # Steering override: injecting a directive mid-run uses the same
        # update_state + resume seam as HITL, so it likewise requires a
        # checkpointer even when checkpoint_enabled=False.
        if self._enable_steering and checkpointer is None:
            from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
                MemorySaver,
            )

            checkpointer = MemorySaver()
            LOGGER.info(
                "Steering enabled: attaching MemorySaver for directive "
                "injection and resume (checkpoint_enabled=False)"
            )

        # When steering is enabled, pause after every agent node so a
        # steerable run has a guaranteed boundary at which to drain the
        # directive queue and apply pending directives before resuming.
        # Empty (the default) means no interrupt points are added, so
        # compilation is byte-for-byte identical to the non-steering path.
        interrupt_after_nodes: List[str] = []
        if self._enable_steering:
            interrupt_after_nodes = list(self._compiled_mas.agent_nodes)

        self._checkpointer = checkpointer
        self._compiled_graph = self._compiled_mas.compile_graph(
            checkpointer=checkpointer,
            interrupt_before=human_nodes,
            interrupt_after=interrupt_after_nodes or None,
        )

        LOGGER.info(
            "MASExecutor initialized for '%s' (%d agents, %s workflow%s%s)",
            self._config.mas_id,
            len(self._config.agents),
            self._config.workflow_type.value,
            f", user_id={self._user_id}" if self._user_id else "",
            (
                ", checkpointer={}".format(type(checkpointer).__name__)
                if checkpointer
                else ", checkpointer=None"
            ),
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(  # pylint: disable=too-many-locals
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        save_results: bool = True,
    ) -> MASExecutionResult:
        """Execute the MAS graph and return structured results.

        Args:
            input_data: Initial state overrides. May include a
                ``"messages"`` key with LangChain message objects.
            thread_id: Thread ID for checkpointed execution. If
                ``None`` and checkpointing is enabled, one is
                auto-generated. When ``user_id`` is set, this is
                treated as the conversation_id and the effective
                thread_id becomes ``{user_id}_{conversation_id}``.
            save_results: Whether to persist results as a JSON file
                in ``log_dir``.

        Returns:
            A ``MASExecutionResult`` with all agent outputs and stats.
            On failure, ``result.success`` is ``False`` and
            ``result.error`` contains the error message.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before run()."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        start_ts = time.time()
        start_time = datetime.now(timezone.utc).isoformat()

        LOGGER.info("Starting MAS execution: %s", execution_id)

        # Build initial state
        initial_state = self._build_initial_state(input_data)

        # Build invoke config
        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        if self._checkpointer is not None:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config["configurable"] = {"thread_id": effective_thread_id}

        try:
            final_state = self._compiled_graph.invoke(
                initial_state, config=invoke_config
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("MAS execution failed: %s", exc, exc_info=True)
            end_time = datetime.now(timezone.utc).isoformat()
            return MASExecutionResult(
                mas_id=self._config.mas_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                total_execution_time_ms=(time.time() - start_ts) * 1000,
                # success is computed property (False when error is set)
                error=str(exc),
            )

        # Collect results
        end_time = datetime.now(timezone.utc).isoformat()
        total_ms = (time.time() - start_ts) * 1000

        agent_results = self._extract_agent_results(final_state)
        total_messages, messages_by_channel = self._compute_communication_stats(
            final_state
        )

        checkpoint_saved = self._checkpointer is not None
        # Communication now persists in checkpointer state
        comm_log_path = None

        result = MASExecutionResult(
            mas_id=self._config.mas_id,
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            total_execution_time_ms=total_ms,
            agent_results=agent_results,
            final_state=self._serialize_state(final_state),
            total_messages=total_messages,
            messages_by_channel=messages_by_channel,
            communication_log_path=comm_log_path,
            checkpoint_saved=checkpoint_saved,
            # success is computed property (True when error=None)
            metadata={
                "thread_id": invoke_config.get("configurable", {}).get("thread_id")
            },
        )

        LOGGER.info(
            "MAS execution complete: %s (%.2f ms, %d agents)",
            execution_id,
            total_ms,
            len(agent_results),
        )

        if save_results:
            os.makedirs(self._log_dir, exist_ok=True)
            result_path = os.path.join(self._log_dir, f"{execution_id}.json")
            result.save_to_file(result_path)

        return result

    # ------------------------------------------------------------------
    # Streaming execution
    # ------------------------------------------------------------------

    def stream(  # pylint: disable=too-many-locals
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_filter: Optional[StreamFilter] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Stream execution events synchronously from the MAS graph.

        Uses LangGraph's ``.stream()`` with ``stream_mode="updates"``
        to yield node-level state updates wrapped as ``StreamEvent``
        objects.

        Args:
            input_data: Initial state overrides.
            thread_id: Thread ID for checkpointed execution.
            stream_filter: Optional filter to select event types.

        Yields:
            ``StreamEvent`` objects for each graph execution step.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before stream()."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        effective_filter = stream_filter or StreamFilter()
        initial_state = self._build_initial_state(input_data)

        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        if self._checkpointer is not None:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config["configurable"] = {"thread_id": effective_thread_id}

        # Emit run_start
        start_event = StreamEvent(
            event_type=StreamEventType.RUN_START,
            data={"execution_id": execution_id, "mas_id": self._config.mas_id},
            run_id=execution_id,
        )
        if effective_filter.accepts(start_event):
            yield start_event

        try:
            for chunk in self._compiled_graph.stream(
                initial_state,
                config=invoke_config,
                stream_mode="updates",
            ):
                # chunk is a dict {node_name: state_update}
                for node_name, state_update in chunk.items():
                    node_event = StreamEvent(
                        event_type=StreamEventType.NODE_END,
                        data={"state_update": state_update},
                        node_name=node_name,
                        agent_id=self._resolve_agent_for_node(node_name),
                        run_id=execution_id,
                    )
                    if effective_filter.accepts(node_event):
                        yield node_event

        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Streaming execution failed: %s", exc, exc_info=True)
            err_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(exc)},
                run_id=execution_id,
            )
            if effective_filter.accepts(err_event):
                yield err_event

        # Emit run_end
        end_event = StreamEvent(
            event_type=StreamEventType.RUN_END,
            data={"execution_id": execution_id},
            run_id=execution_id,
        )
        if effective_filter.accepts(end_event):
            yield end_event

    def run_streaming(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Stream node outputs as ``(node_name, state_update)`` tuples.

        A lightweight streaming API for UI consumers. Yields one tuple per
        agent node as it completes execution, in execution order.

        Requires the executor to be initialized via ``initialize()`` before
        calling.

        Args:
            input_data: Initial state overrides (may include a ``"messages"``
                key with a list of LangChain messages).
            thread_id: Thread ID for checkpointed execution. Pass the same
                ID across calls to maintain conversation context.

        Yields:
            ``(node_name, state_update)`` tuples where *node_name* is the raw
            graph node name (typically matching the ``agent_id`` field in the
            MAS config for agent nodes, but may also include internal routing
            or pipeline nodes) and *state_update* is the raw state dict
            produced by that node.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
            Exception: Any exception raised by the graph is logged and
                re-raised so the caller receives a clean log entry.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before run_streaming()."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        LOGGER.info("Starting MAS streaming execution: %s", execution_id)
        initial_state = self._build_initial_state(input_data)

        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        effective_thread_id: Optional[str] = None
        if self._checkpointer is not None:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config["configurable"] = {"thread_id": effective_thread_id}

        try:
            for chunk in self._compiled_graph.stream(
                initial_state,
                config=invoke_config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    yield (node_name, state_update)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("run_streaming execution failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("MAS streaming execution complete: %s", execution_id)

        # After the stream exhausts, check whether the graph paused at a
        # human-interrupt node.  If pending nodes remain, yield a sentinel so
        # the caller can present a human-input UI and later call resume_streaming().
        if invoke_config and self._config.human_in_loop:
            try:
                graph_state = self._compiled_graph.get_state(invoke_config)
                if graph_state.next:
                    yield (
                        "__human_interrupt__",
                        {
                            "next": list(graph_state.next),
                            "thread_id": effective_thread_id,
                        },
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("Could not inspect graph state for HITL check: %s", exc)

        # Separately, check for a langgraph.types.interrupt() pause (e.g. the
        # ask_user tool) — distinct from the human_in_loop / is_human whole-
        # agent-slot mechanism above, and not gated on human_in_loop, since a
        # tool-level interrupt can occur in any MAS regardless of that flag.
        # Additive: existing __human_interrupt__ callers are unaffected.
        if invoke_config:
            try:
                graph_state = self._compiled_graph.get_state(invoke_config)
                pending_interrupts = [
                    interrupt_obj
                    for task in graph_state.tasks
                    for interrupt_obj in task.interrupts
                ]
                if pending_interrupts:
                    yield (
                        "__ask_user_pending__",
                        {
                            # Raw interrupt payload(s) as passed to interrupt(...),
                            # e.g. {"type": "ask_user", "question": ..., "options": ...}.
                            "interrupts": [i.value for i in pending_interrupts],
                            "thread_id": effective_thread_id,
                        },
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    "Could not inspect graph state for ask_user interrupt check: %s",
                    exc,
                )

    def resume_with_value(
        self,
        value: Any,
        thread_id: str,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Resume a graph paused at a ``langgraph.types.interrupt()`` call.

        Unlike :meth:`resume_streaming` (which injects a ``HumanMessage`` and
        lets the graph continue routing normally — the ``human_in_loop`` /
        ``is_human`` whole-agent-slot mechanism), this supplies *value*
        directly as the return value of the ``interrupt(...)`` call that
        paused execution, via ``Command(resume=value)``. Use this to resume
        after a ``__ask_user_pending__`` sentinel from :meth:`run_streaming`.

        The outer AETHER/IRIS node that called into the tool-calling agent
        re-executes from its own start on resume (not just the single
        interrupted tool call) — this is ``langgraph.types.interrupt()``'s
        documented behavior, not a bili-core choice. Verified against
        ``create_agent``'s tool-calling subgraph specifically: its own
        internal LLM-call node is NOT re-invoked on resume (LangGraph tracks
        that node's own already-completed task independently, even without
        an explicit checkpointer passed to ``create_agent``, when the
        subgraph is invoked from inside an already-checkpointed outer node —
        which is how every bili-core tool-calling node uses it). Code in the
        outer node before the tool-calling agent is invoked (e.g. system
        prompt / comm-context assembly) does re-run; this is harmless as
        long as it is pure computation with no external side effect, which
        holds for AETHER's own pre-invoke bookkeeping today.

        Args:
            value: The value returned to the paused ``interrupt(...)`` call
                (e.g. the human's answer to an ``ask_user`` question).
            thread_id: Thread ID originally reported in the
                ``__ask_user_pending__`` sentinel.

        Yields:
            ``(node_name, state_update)`` tuples for every node that executes
            after the interrupt, in execution order.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before resume_with_value()."
            )

        from langgraph.types import Command  # pylint: disable=import-outside-toplevel

        invoke_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_iterations,
        }

        LOGGER.info("Resuming interrupt()-paused execution for thread '%s'", thread_id)
        try:
            for chunk in self._compiled_graph.stream(
                Command(resume=value),
                config=invoke_config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    yield (node_name, state_update)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("resume_with_value failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("interrupt() resume complete for thread '%s'", thread_id)

    def run_streaming_tokens(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Stream token-level output with node-completion sentinels.

        A token-granularity streaming API for UI consumers.  Uses LangGraph's
        ``stream_mode=["messages", "updates"]`` to interleave per-token chunks
        with per-node state updates in a single pass.

        Yields three event types in order:

        - ``("__token__", {"node": node_name, "token": content})``:
          one yield per non-empty ``AIMessageChunk`` content string as tokens
          arrive from the LLM.
        - ``("__node_complete__", {"node": node_name, "state_update": state_update})``:
          one yield per agent node after all its tokens are exhausted and the
          node's full state update is available.
        - ``("__human_interrupt__", {"next": [...], "thread_id": ...})``:
          yielded once after the stream exhausts if the graph paused at a
          human-in-the-loop node (only when ``human_in_loop=True`` in config).

        Internal routing and pipeline sub-nodes are included in
        ``__node_complete__`` events — callers should filter by whether the
        node name is a known ``agent_id``.

        Args:
            input_data: Initial state overrides (may include a ``"messages"``
                key with a list of LangChain messages).
            thread_id: Thread ID for checkpointed execution. Pass the same
                ID across calls to maintain conversation context.

        Yields:
            ``(event_type, event_data)`` tuples as described above.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
            Exception: Any exception raised by the graph is logged and
                re-raised so the caller receives a clean log entry.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. "
                "Call initialize() before run_streaming_tokens()."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        LOGGER.info("Starting MAS token streaming: %s", execution_id)
        initial_state = self._build_initial_state(input_data)

        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        effective_thread_id: Optional[str] = None
        if self._checkpointer is not None:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config["configurable"] = {"thread_id": effective_thread_id}

        try:
            for mode, data in self._compiled_graph.stream(
                initial_state,
                config=invoke_config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, metadata = data
                    if not hasattr(chunk, "content") or not chunk.content:
                        continue
                    node_name = metadata.get("langgraph_node", "")
                    if not node_name:
                        continue
                    content = chunk.content
                    if isinstance(content, list):
                        # Structured content blocks (e.g. tool-use) — extract text only
                        content = "".join(
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict) and "text" in c
                        )
                    if content:
                        yield ("__token__", {"node": node_name, "token": content})
                elif mode == "updates":
                    for node_name, state_update in data.items():
                        yield (
                            "__node_complete__",
                            {"node": node_name, "state_update": state_update},
                        )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("run_streaming_tokens failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("MAS token streaming complete: %s", execution_id)

        # After the stream exhausts, check whether the graph paused at a
        # human-interrupt node (mirrors the same check in run_streaming()).
        if invoke_config and self._config.human_in_loop:
            try:
                graph_state = self._compiled_graph.get_state(invoke_config)
                if graph_state.next:
                    yield (
                        "__human_interrupt__",
                        {
                            "next": list(graph_state.next),
                            "thread_id": effective_thread_id,
                        },
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("Could not inspect graph state for HITL check: %s", exc)

    def resume_streaming(
        self,
        human_input: Union[str, Sequence[Any]],
        thread_id: str,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Resume a graph that was paused at a human-interrupt node.

        Injects *human_input* as a ``HumanMessage`` into the graph state and
        then continues streaming from where execution left off.  The graph
        must have been paused via ``run_streaming()`` after a
        ``__human_interrupt__`` sentinel was yielded.

        Args:
            human_input: The human reviewer's response.  A plain string for a
                text reply, or a list of content parts (text plus image) to
                answer with an image; see
                :func:`bili.iris.multimodal.build_human_message`.  A string
                builds exactly the message it always did.
            thread_id: Thread ID originally reported in the
                ``__human_interrupt__`` sentinel.

        Yields:
            ``(node_name, state_update)`` tuples for every node that executes
            after the interrupt, in execution order.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before resume_streaming()."
            )

        from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
            HumanMessage,
        )

        invoke_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_iterations,
        }
        self._compiled_graph.update_state(
            invoke_config,
            {"messages": [HumanMessage(content=normalise_prompt(human_input))]},
        )

        LOGGER.info("Resuming HITL execution for thread '%s'", thread_id)
        try:
            for chunk in self._compiled_graph.stream(
                None,
                config=invoke_config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    yield (node_name, state_update)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("resume_streaming failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("HITL resume complete for thread '%s'", thread_id)

    # ------------------------------------------------------------------
    # Operator steering (user-initiated mid-run redirect)
    # ------------------------------------------------------------------
    #
    # HITL and ``ask_user`` are the agent->user direction: an agent pauses to
    # ask, a human answers.  Steering is the opposite direction: a human
    # supervising a long-running run injects a mid-run redirect ("emphasize
    # X", "stop pursuing that branch") that the next agent observes at the
    # next natural boundary, without killing the run and starting over.
    #
    # It reuses the existing update_state + resume seam (the same one HITL's
    # resume_streaming uses): a directive is written into ``messages`` via
    # ``update_state``, and the run resumes with ``.stream(None)``.  Every
    # agent node rebuilds its prompt from state at the start of its step
    # (reading ``messages`` and its pending-message context), so an injected
    # directive is picked up with no agent, compiler, or schema change.

    def submit_steer(self, message: Union[str, Sequence[Any]]) -> None:
        """Enqueue an operator directive for a steerable run to pick up.

        Thread-safe: intended to be called from a different thread than the
        one driving :meth:`run_streaming_steerable`, so a supervising
        operator can inject a directive while the run is in flight. The
        directive is drained and applied at the next superstep boundary (the
        pause after the currently executing agent node completes), then
        observed by the next node to start.

        Args:
            message: The operator's steering directive: free text, or a list
                of content parts to redirect the run with an image.

        Raises:
            RuntimeError: If the executor was not constructed with
                ``enable_steering=True``.
        """
        if self._steer_queue is None:
            raise RuntimeError(
                "Steering is not enabled. Construct "
                "MASExecutor(..., enable_steering=True) to submit directives."
            )
        self._steer_queue.put(message)

    def _drain_steer_queue(self) -> List[Union[str, Sequence[Any]]]:
        """Remove and return every currently-queued operator directive.

        Non-blocking: returns only what is already queued at call time (an
        empty list when nothing is pending), so a steerable run never waits
        on the queue.
        """
        directives: List[Union[str, Sequence[Any]]] = []
        if self._steer_queue is None:
            return directives
        while True:
            try:
                directives.append(self._steer_queue.get_nowait())
            except queue.Empty:
                break
        return directives

    def _apply_steer_directives(
        self,
        invoke_config: Dict[str, Any],
        messages: Sequence[Union[str, Sequence[Any]]],
    ) -> None:
        """Write operator directives into graph state as ``HumanMessage`` s.

        This is the single point at which a steer directive lands in state,
        shared by :meth:`steer` and :meth:`run_streaming_steerable` so the
        two cannot drift on how a directive is applied. The next agent node
        reads ``messages`` at the start of its step and observes them.

        A directive is a plain string, or a list of content parts when the
        operator is redirecting the run with an image.
        """
        from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
            HumanMessage,
        )

        self._compiled_graph.update_state(
            invoke_config,
            {"messages": [HumanMessage(content=normalise_prompt(m)) for m in messages]},
        )

    def steer(
        self,
        message: Union[str, Sequence[Any]],
        thread_id: str,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Inject one operator directive into a paused run and resume it.

        The explicit counterpart to :meth:`run_streaming_steerable`'s queue:
        use this when the caller is itself driving the run (e.g. it broke out
        of a streaming loop at a boundary) and wants to inject a single
        directive and continue. Generalises :meth:`resume_streaming` (which
        answers an agent's question) to the operator->run direction: the
        directive is applied via ``update_state`` and execution resumes from
        the checkpoint, so the next node observes it.

        Requires the run to be paused at a boundary (steering attaches an
        ``interrupt_after`` on every agent node, so a steerable run pauses
        after each one) and a checkpointer to be attached (guaranteed when
        ``enable_steering=True``).

        Args:
            message: The operator's steering directive: free text, or a list
                of content parts to redirect the run with an image.
            thread_id: Thread ID of the run to steer.

        Yields:
            ``(node_name, state_update)`` tuples for every node that executes
            after the directive is applied, in execution order.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before steer()."
            )

        invoke_config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_iterations,
        }
        self._apply_steer_directives(invoke_config, [message])

        LOGGER.info("Applying operator steer directive to thread '%s'", thread_id)
        try:
            for chunk in self._compiled_graph.stream(
                None,
                config=invoke_config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    yield (node_name, state_update)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("steer failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("Steer resume complete for thread '%s'", thread_id)

    def run_streaming_steerable(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Stream node outputs, draining operator directives at each boundary.

        A steer-aware variant of :meth:`run_streaming`. Because steering
        compiles the graph to pause after every agent node
        (``interrupt_after``), this drives the graph one superstep at a time:
        it yields each node's ``(node_name, state_update)`` as it completes,
        and at each pause it drains the directive queue (see
        :meth:`submit_steer`) and applies any pending directives via
        ``update_state`` before resuming. A directive submitted while an
        agent is executing is therefore observed by the next agent to start.

        Directives are drained non-blocking, so an empty queue makes this a
        plain streamed run: it yields the same nodes in the same order and
        produces the same final state, just paused and resumed at each
        boundary. That equivalence is what makes steering safe to leave
        enabled: an unused steer channel changes nothing.

        Args:
            input_data: Initial state overrides (may include a ``"messages"``
                key with a list of LangChain messages).
            thread_id: Thread ID for checkpointed execution.

        Yields:
            ``(node_name, state_update)`` tuples in execution order.

        Raises:
            RuntimeError: If ``initialize()`` has not been called, or the
                executor was not constructed with ``enable_steering=True``.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. "
                "Call initialize() before run_streaming_steerable()."
            )
        if not self._enable_steering:
            raise RuntimeError(
                "Steering is not enabled. Construct "
                "MASExecutor(..., enable_steering=True) for a steerable run."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        LOGGER.info("Starting steerable MAS streaming execution: %s", execution_id)
        initial_state = self._build_initial_state(input_data)

        # Steering guarantees a checkpointer (see initialize()), so a
        # thread_id is always constructed for the run.
        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        effective_thread_id = self._construct_thread_id(thread_id, execution_id)
        invoke_config["configurable"] = {"thread_id": effective_thread_id}

        stream_input: Any = initial_state
        try:
            while True:
                for chunk in self._compiled_graph.stream(
                    stream_input,
                    config=invoke_config,
                    stream_mode="updates",
                ):
                    for node_name, state_update in chunk.items():
                        yield (node_name, state_update)

                # The stream exhausted: the graph either completed or paused
                # at an interrupt_after boundary. If nothing is left to run,
                # the run is done; otherwise drain any pending directives,
                # apply them, and resume.
                graph_state = self._compiled_graph.get_state(invoke_config)
                if not graph_state.next:
                    break
                directives = self._drain_steer_queue()
                if directives:
                    self._apply_steer_directives(invoke_config, directives)
                stream_input = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("run_streaming_steerable failed: %s", exc, exc_info=True)
            raise
        finally:
            LOGGER.info("Steerable MAS streaming complete: %s", execution_id)

    async def astream(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_filter: Optional[StreamFilter] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream execution events asynchronously with token granularity.

        Uses LangGraph's ``.astream_events(version="v2")`` to yield
        fine-grained events including token-level LLM output wrapped
        as ``StreamEvent`` objects.

        Args:
            input_data: Initial state overrides.
            thread_id: Thread ID for checkpointed execution.
            stream_filter: Optional filter to select event types.

        Yields:
            ``StreamEvent`` objects including token-level events.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._compiled_graph is None:
            raise RuntimeError(
                "Executor not initialized. Call initialize() before astream()."
            )

        execution_id = f"{self._config.mas_id}_{uuid.uuid4().hex[:8]}"
        effective_filter = stream_filter or StreamFilter()
        initial_state = self._build_initial_state(input_data)

        invoke_config: Dict[str, Any] = {"recursion_limit": self._config.max_iterations}
        if self._checkpointer is not None:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config["configurable"] = {"thread_id": effective_thread_id}

        # Emit run_start
        start_event = StreamEvent(
            event_type=StreamEventType.RUN_START,
            data={"execution_id": execution_id, "mas_id": self._config.mas_id},
            run_id=execution_id,
        )
        if effective_filter.accepts(start_event):
            yield start_event

        try:
            async for event in self._compiled_graph.astream_events(
                initial_state,
                config=invoke_config,
                version="v2",
            ):
                stream_event = self._map_langgraph_event(event, execution_id)
                if stream_event and effective_filter.accepts(stream_event):
                    yield stream_event

        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Async streaming execution failed: %s", exc, exc_info=True)
            err_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(exc)},
                run_id=execution_id,
            )
            if effective_filter.accepts(err_event):
                yield err_event

        # Emit run_end
        end_event = StreamEvent(
            event_type=StreamEventType.RUN_END,
            data={"execution_id": execution_id},
            run_id=execution_id,
        )
        if effective_filter.accepts(end_event):
            yield end_event

    def _map_langgraph_event(
        self, event: Dict[str, Any], execution_id: str
    ) -> Optional[StreamEvent]:
        """Map a LangGraph v2 stream event to a StreamEvent.

        Args:
            event: Raw event dict from ``astream_events(version="v2")``.
            execution_id: The current execution run ID.

        Returns:
            A ``StreamEvent``, or ``None`` to skip the event.
        """
        event_kind = event.get("event", "")
        event_data = event.get("data", {})
        event_name = event.get("name", "")

        if event_kind == "on_chat_model_stream":
            # Token-level streaming from LLM
            chunk = event_data.get("chunk")
            content = ""
            if chunk is not None:
                content = getattr(chunk, "content", str(chunk))
            if content:
                return StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    data={"content": content},
                    node_name=event_name,
                    agent_id=self._resolve_agent_for_node(event_name),
                    run_id=execution_id,
                )

        elif event_kind == "on_chain_start":
            node_name = event_name
            if node_name and node_name != "LangGraph":
                return StreamEvent(
                    event_type=StreamEventType.NODE_START,
                    data={"name": node_name},
                    node_name=node_name,
                    agent_id=self._resolve_agent_for_node(node_name),
                    run_id=execution_id,
                )

        elif event_kind == "on_chain_end":
            node_name = event_name
            output = event_data.get("output", {})
            if node_name and node_name != "LangGraph":
                return StreamEvent(
                    event_type=StreamEventType.NODE_END,
                    data={"output": output},
                    node_name=node_name,
                    agent_id=self._resolve_agent_for_node(node_name),
                    run_id=execution_id,
                )

        return None

    def _resolve_agent_for_node(self, node_name: str) -> Optional[str]:
        """Try to match a graph node name to an agent_id."""
        if self._compiled_mas is None:
            return None
        if node_name in self._compiled_mas.agent_nodes:
            return node_name
        # Check if node_name is a prefix match (e.g. pipeline sub-nodes)
        for agent_id in self._compiled_mas.agent_nodes:
            if node_name.startswith(agent_id):
                return agent_id
        return None

    # ------------------------------------------------------------------
    # Checkpoint persistence testing
    # ------------------------------------------------------------------

    def run_with_checkpoint_persistence(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Tuple[MASExecutionResult, MASExecutionResult]:
        """Run MAS twice with checkpoint save/restore.

        1. Execute the graph (saves checkpoint via LangGraph).
        2. Re-initialize the executor (clears runtime state).
        3. Execute again with the same ``thread_id`` (restores checkpoint).

        Args:
            input_data: Initial state overrides for the first run.
            thread_id: Thread ID for checkpoint continuity.

        Returns:
            Tuple of ``(original_result, restored_result)``.
        """
        effective_thread_id = thread_id or f"cp_{uuid.uuid4().hex[:8]}"

        # First run — saves checkpoint
        original_result = self.run(
            input_data=input_data,
            thread_id=effective_thread_id,
            save_results=False,
        )
        original_result.checkpoint_saved = True
        original_result.metadata["checkpoint_test"] = "original"

        # Re-initialize (simulates restart)
        self.initialize()

        # Second run — restores from checkpoint
        restored_result = self.run(
            input_data=input_data,
            thread_id=effective_thread_id,
            save_results=False,
        )
        restored_result.metadata["checkpoint_test"] = "restored"

        LOGGER.info(
            "Checkpoint persistence test complete for thread '%s'",
            effective_thread_id,
        )

        return original_result, restored_result

    # ------------------------------------------------------------------
    # Cross-model transfer testing
    # ------------------------------------------------------------------

    def run_cross_model_test(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        source_model: Optional[str] = None,
        target_model: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Tuple[MASExecutionResult, MASExecutionResult]:
        """Run MAS with two different model configurations.

        1. Set all agents to ``source_model``, execute (saves checkpoint).
        2. Set all agents to ``target_model``, re-initialize, execute
           with the same ``thread_id`` (restores checkpoint).

        Args:
            input_data: Initial state overrides.
            source_model: Model name for the first run (e.g. ``"gpt-4"``).
                ``None`` means stub mode.
            target_model: Model name for the second run.
                ``None`` means stub mode.
            thread_id: Thread ID for checkpoint continuity.

        Returns:
            Tuple of ``(source_result, target_result)``.
        """
        effective_thread_id = thread_id or f"xm_{uuid.uuid4().hex[:8]}"

        # Save original config to restore afterward (avoid permanent mutation)
        original_config = self._config

        # --- Source model run ---
        source_agents = [
            agent.model_copy(update={"model_name": source_model})
            for agent in original_config.agents
        ]
        source_config = original_config.model_copy(update={"agents": source_agents})

        self._config = source_config
        self.initialize()

        source_result = self.run(
            input_data=input_data,
            thread_id=effective_thread_id,
            save_results=False,
        )
        source_result.metadata["cross_model_test"] = "source"
        source_result.metadata["model"] = source_model

        # --- Target model run ---
        target_agents = [
            agent.model_copy(update={"model_name": target_model})
            for agent in source_config.agents
        ]
        target_config = source_config.model_copy(update={"agents": target_agents})

        self._config = target_config
        self.initialize()

        target_result = self.run(
            input_data=input_data,
            thread_id=effective_thread_id,
            save_results=False,
        )
        target_result.metadata["cross_model_test"] = "target"
        target_result.metadata["model"] = target_model

        # Restore original config
        self._config = original_config
        self.initialize()

        LOGGER.info(
            "Cross-model transfer test complete: %s -> %s (thread '%s')",
            source_model,
            target_model,
            effective_thread_id,
        )

        return source_result, target_result

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _create_checkpointer_local(self) -> Any:
        """Create a checkpointer for local/single-instance runs (no user_id).

        Uses the MASConfig's ``checkpoint_config`` to select the backend but
        does NOT set ``user_id``, so thread ownership validation is disabled.
        Falls back to ``MemorySaver`` if the factory is unavailable.

        Returns:
            A checkpointer instance without ownership validation.
        """
        try:
            from bili.aether.integration.checkpointer_factory import (  # pylint: disable=import-outside-toplevel
                create_checkpointer_from_config,
            )

            checkpointer = create_checkpointer_from_config(
                self._config.checkpoint_config, user_id=None
            )
            LOGGER.info(
                "Created local checkpointer (type=%s) for always-on persistence",
                type(checkpointer).__name__,
            )
            return checkpointer

        except ImportError:
            LOGGER.warning(
                "Checkpointer factory not available; "
                "falling back to MemorySaver for local checkpointing"
            )

        from langgraph.checkpoint.memory import (  # pylint: disable=import-error,import-outside-toplevel
            MemorySaver,
        )

        return MemorySaver()

    def _create_checkpointer_with_user_id(self) -> Any:
        """Create a checkpointer with multi-tenant security enabled.

        Uses the bili-core checkpointer factory with user_id parameter
        for thread ownership validation and on-demand schema migration.

        Returns:
            A checkpointer instance with user_id configured.
        """
        try:
            from bili.aether.integration.checkpointer_factory import (  # pylint: disable=import-outside-toplevel
                create_checkpointer_from_config,
            )

            # Create checkpointer from config with user_id
            checkpointer = create_checkpointer_from_config(
                self._config.checkpoint_config, user_id=self._user_id
            )

            LOGGER.info(
                "Created checkpointer with user_id='%s' for multi-tenant security",
                self._user_id,
            )
            return checkpointer

        except ImportError:
            LOGGER.warning(
                "Checkpointer factory not available; "
                "falling back to MemorySaver without user_id support"
            )

            from langgraph.checkpoint.memory import (  # pylint: disable=import-error,import-outside-toplevel
                MemorySaver,
            )

            return MemorySaver()

    def _validate_thread_ownership(self, thread_id: str) -> None:
        """Validate that thread_id matches the expected pattern for user_id.

        In multi-tenant mode (when user_id is set), thread_id must follow
        the pattern: ``{user_id}`` or ``{user_id}_*``.

        Args:
            thread_id: The thread_id to validate.

        Raises:
            PermissionError: If thread_id doesn't belong to the configured user_id.
        """
        if self._user_id is None:
            return  # No validation in non-multi-tenant mode

        # Thread ID must either exactly match user_id or start with "{user_id}_"
        if not (
            thread_id == self._user_id or thread_id.startswith(f"{self._user_id}_")
        ):
            raise PermissionError(
                f"Access denied: thread_id '{thread_id}' does not belong to "
                f"user '{self._user_id}'. Thread IDs must match pattern: "
                f"'{self._user_id}' or '{self._user_id}_*'"
            )

    def _construct_thread_id(self, thread_id: Optional[str], execution_id: str) -> str:
        """Construct thread_id for checkpointer, handling multi-tenant pattern.

        When ``user_id`` is set, constructs thread_id in the format
        ``{user_id}_{conversation_id}`` to ensure thread ownership validation.

        Args:
            thread_id: Optional explicit thread_id from caller.
            execution_id: Auto-generated execution_id as fallback.

        Returns:
            Effective thread_id for this execution.
        """
        if self._user_id:
            # Multi-tenant mode: enforce {user_id}_{conversation_id} pattern
            if self._conversation_id:
                # Use provided conversation_id
                return f"{self._user_id}_{self._conversation_id}"
            if thread_id:
                # Validate if thread_id already has user_id prefix (reuse case)
                # Otherwise treat as conversation_id and prepend user_id
                if (
                    thread_id.startswith(f"{self._user_id}_")
                    or thread_id == self._user_id
                ):
                    self._validate_thread_ownership(thread_id)
                    return thread_id  # Already has user_id prefix
                # Treat as conversation_id and construct full thread_id
                return f"{self._user_id}_{thread_id}"
            # Generate new conversation_id from execution_id
            conversation_id = execution_id.split("_", 1)[-1]  # Strip mas_id prefix
            return f"{self._user_id}_{conversation_id}"

        # Non-multi-tenant mode: use thread_id or execution_id as-is
        return thread_id or execution_id

    def _build_initial_state(
        self, input_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct initial state dict with workflow-type-specific fields.

        Mirrors the fields defined in
        ``bili.aether.compiler.state_generator.generate_state_schema()``.
        """
        state: Dict[str, Any] = {
            "messages": [],
            "current_agent": "",
            "agent_outputs": {},
            "mas_id": self._config.mas_id,
        }

        wtype = self._config.workflow_type

        if wtype == WorkflowType.CONSENSUS:
            state["current_round"] = 0
            state["votes"] = {}
            state["consensus_reached"] = False
            state["max_rounds"] = self._config.max_consensus_rounds

        if wtype == WorkflowType.HIERARCHICAL:
            state["current_tier"] = 0
            state["tier_results"] = {}

        if wtype == WorkflowType.SUPERVISOR:
            state["next_agent"] = ""
            state["pending_tasks"] = []
            state["completed_tasks"] = []

        if wtype == WorkflowType.CUSTOM and self._config.human_in_loop:
            state["needs_human_review"] = False

        # communication_log is always initialised so per-agent provenance is
        # captured for every run, regardless of whether explicit channels are
        # configured.
        state["communication_log"] = []

        # channel_messages and pending_messages are only needed for MAS
        # configurations that declare explicit inter-agent channels.
        if self._config.channels:
            state["channel_messages"] = {}
            state["pending_messages"] = {}

        # Merge user-provided data (overrides defaults)
        if input_data:
            state.update(input_data)

        return state

    def _extract_agent_results(
        self, final_state: Dict[str, Any]
    ) -> List[AgentExecutionResult]:
        """Extract per-agent results from the final graph state."""
        results = []
        agent_outputs = final_state.get("agent_outputs") or {}
        comm_log = final_state.get("communication_log") or []

        for agent in self._config.agents:
            output = agent_outputs.get(agent.agent_id, {})

            # Count messages sent/received from communication log
            sent = sum(1 for entry in comm_log if entry.get("sender") == agent.agent_id)
            received = sum(
                1
                for entry in comm_log
                if entry.get("receiver") in (agent.agent_id, "__all__")
                and entry.get("sender") != agent.agent_id
            )

            results.append(
                AgentExecutionResult(
                    agent_id=agent.agent_id,
                    role=agent.role,
                    output=output,
                    error=output.get("error"),
                    tools_used=output.get("tools_used", []),
                    messages_sent=sent,
                    messages_received=received,
                )
            )

        return results

    def _compute_communication_stats(
        self, final_state: Dict[str, Any]
    ) -> Tuple[int, Dict[str, int]]:
        """Compute message counts from the final state.

        Returns:
            Tuple of ``(total_messages, messages_by_channel)``.
        """
        comm_log = final_state.get("communication_log") or []
        total = len(comm_log)

        by_channel: Dict[str, int] = {}
        for entry in comm_log:
            channel = entry.get("channel", "__unknown__")
            by_channel[channel] = by_channel.get(channel, 0) + 1

        return total, by_channel

    def _get_communication_log_path(self) -> Optional[str]:
        """DEPRECATED: JSONL communication logging is deprecated.

        Communication now persists in LangGraph state via checkpointers,
        making it cloud-ready (survives pod restarts, works in K8s).

        Returns:
            Always returns None. Communication log is in state["communication_log"].
        """
        return None

    @staticmethod
    def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-serializable objects in state to JSON-safe values.

        LangChain ``AIMessage`` / ``HumanMessage`` objects in the
        ``messages`` list are converted to content-only dicts.
        """
        serialized = {}
        for key, value in state.items():
            if key == "messages":
                serialized[key] = [_serialize_message(m) for m in (value or [])]
            else:
                try:
                    # Quick serialization check
                    json.dumps(value, default=str)
                    serialized[key] = value
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        return serialized


# ======================================================================
# Convenience function
# ======================================================================


def execute_mas(
    config: MASConfig,
    input_data: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> MASExecutionResult:
    """Compile and execute a MAS in one call.

    Convenience wrapper around ``MASExecutor`` for simple use cases.

    Args:
        config: A ``MASConfig`` instance.
        input_data: Initial state with messages.
        log_dir: Directory for logs and results.

    Returns:
        A ``MASExecutionResult`` from the execution.
    """
    executor = MASExecutor(config, log_dir=log_dir)
    executor.initialize()
    return executor.run(input_data)


# ======================================================================
# Private helpers
# ======================================================================


def _serialize_message(message: Any) -> Dict[str, Any]:
    """Convert a LangChain message object to a JSON-safe dict."""
    if hasattr(message, "content"):
        result: Dict[str, Any] = {
            "type": type(message).__name__,
            "content": message.content,
        }
        if hasattr(message, "name") and message.name:
            result["name"] = message.name
        return result
    return {"type": "unknown", "content": str(message)}
