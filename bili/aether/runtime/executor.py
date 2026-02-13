"""MAS Execution Controller — runs a compiled MAS graph and collects results.

Wraps the AETHER compiler and LangGraph execution pipeline to provide
a structured ``MASExecutionResult`` with per-agent outputs, timing,
communication statistics, and checkpoint metadata.

Usage::

    from bili.aether.runtime.executor import MASExecutor, execute_mas

    executor = MASExecutor(config, log_dir="logs")
    executor.initialize()
    result = executor.run({"messages": [HumanMessage(content="Hello")]})

    # Or use the convenience function:
    result = execute_mas(config, {"messages": [HumanMessage(content="Hello")]})
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bili.aether.runtime.execution_result import (
    AgentExecutionResult,
    MASExecutionResult,
)
from bili.aether.schema import MASConfig, WorkflowType

LOGGER = logging.getLogger(__name__)


class MASExecutor:
    """Executes a MAS configuration end-to-end and collects results.

    Attributes:
        config: The ``MASConfig`` being executed.
        log_dir: Directory for logs and result files.
    """

    def __init__(
        self,
        config: MASConfig,
        log_dir: Optional[str] = None,
        validate_config: bool = True,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
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
        """
        self._config = config
        self._log_dir = log_dir or os.getcwd()
        self._validate_config = validate_config
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._compiled_mas = None
        self._compiled_graph = None

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

        If ``user_id`` was provided to ``__init__()``, creates a checkpointer
        with multi-tenant security enabled.

        Raises:
            ValueError: If config validation fails.
        """
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        self._compiled_mas = compile_mas(self._config)

        # Create checkpointer with user_id if multi-tenant mode enabled
        checkpointer = None
        if self._user_id and self._config.checkpoint_enabled:
            checkpointer = self._create_checkpointer_with_user_id()

        self._compiled_graph = self._compiled_mas.compile_graph(
            checkpointer=checkpointer
        )

        LOGGER.info(
            "MASExecutor initialized for '%s' (%d agents, %s workflow%s)",
            self._config.mas_id,
            len(self._config.agents),
            self._config.workflow_type.value,
            f", user_id={self._user_id}" if self._user_id else "",
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
        invoke_config: Dict[str, Any] = {}
        if self._config.checkpoint_enabled:
            effective_thread_id = self._construct_thread_id(thread_id, execution_id)
            invoke_config = {"configurable": {"thread_id": effective_thread_id}}

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

        checkpoint_saved = self._config.checkpoint_enabled
        # JSONL logging deprecated - communication now persists in checkpointer state
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
                # Use provided thread_id (assume it's conversation_id)
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

        # Communication fields (only when channels are configured)
        if self._config.channels:
            state["channel_messages"] = {}
            state["pending_messages"] = {}
            state["communication_log"] = []

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
