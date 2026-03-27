"""Public API for the AETHER Attack Injection Framework.

``AttackInjector`` is the single entry point for all adversarial payload
injections.  It wraps a ``MASExecutor`` (does not subclass it) and routes
each call to the appropriate strategy based on ``injection_phase``.

Pre-execution injections:
    The MASConfig is deep-copied and patched with the chosen strategy, a
    fresh ``MASExecutor`` is created for the patched config, and the full
    MAS run is executed.  ``PropagationTracker`` observes the results via
    ``MASExecutionResult.agent_results``.

Mid-execution injections:
    A fresh ``CompiledMAS`` is built from the original config.  The graph is
    compiled with ``interrupt_before=[target_agent_id]``, which causes
    LangGraph to pause before the target node.  The payload is injected into
    the interrupted state, and execution resumes via
    ``Command(resume=modified_state)``.  ``PropagationTracker`` observes each
    downstream node through the ``stream(stream_mode="updates")`` generator.

Blocking vs. fire-and-forget:
    ``blocking=True`` (default) blocks until propagation is fully tracked.
    ``blocking=False`` submits the work to a ``ThreadPoolExecutor`` and
    returns a skeleton ``AttackResult`` with ``completed_at=None``
    immediately.  The background thread logs the completed result when done.
"""

import datetime
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bili.aether.attacks.logger import AttackLogger
from bili.aether.attacks.models import AttackResult, AttackType, InjectionPhase
from bili.aether.attacks.propagation import PropagationTracker
from bili.aether.schema import MASConfig

if TYPE_CHECKING:
    from bili.aether.runtime.executor import MASExecutor
    from bili.aether.security.detector import SecurityEventDetector

LOGGER = logging.getLogger(__name__)

_STRATEGY_MAP = {
    AttackType.PROMPT_INJECTION: "inject_prompt_injection",
    AttackType.MEMORY_POISONING: "inject_memory_poisoning",
    AttackType.AGENT_IMPERSONATION: "inject_agent_impersonation",
    AttackType.BIAS_INHERITANCE: "inject_bias_inheritance",
    AttackType.JAILBREAK: "inject_prompt_injection",
    # AttackType.PERSISTENCE is intentionally absent: it uses checkpoint-phase
    # execution (_run_checkpoint_execution / strategies/persistence.py) and is
    # never dispatched through _run_pre_execution / getattr(pre_execution, ...).
}


class AttackInjector:
    """Inject adversarial payloads into a MAS and track propagation.

    Args:
        config: The ``MASConfig`` to inject into.
        executor: A ``MASExecutor`` instance, or ``None``.  Pass ``None``
            when only pre-execution phases are requested —
            ``_run_pre_execution`` creates its own executor internally.
            Required (and must be initialised) for mid-execution injections.
        log_path: Optional path to the NDJSON attack log file.  Defaults to
            ``AttackLogger.DEFAULT_PATH``.
        max_workers: Thread-pool size for ``blocking=False`` calls.
    """

    def __init__(
        self,
        config: MASConfig,
        executor: "MASExecutor | None",
        log_path: Path | None = None,
        max_workers: int = 4,
        security_detector: "SecurityEventDetector | None" = None,
    ) -> None:
        self._config = config
        self._executor = executor
        self._attack_logger = AttackLogger(log_path)
        self._security_detector = security_detector
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def inject_attack(
        self,
        agent_id: str,
        attack_type: str | AttackType,
        payload: str,
        injection_phase: str | InjectionPhase = InjectionPhase.PRE_EXECUTION,
        blocking: bool = True,
        track_propagation: bool = True,
    ) -> AttackResult:
        """Inject an adversarial payload into the MAS.

        Args:
            agent_id: Target agent identifier.  Must exist in the config.
            attack_type: The category of adversarial payload.
            payload: The raw adversarial string to inject.
            injection_phase: When to deliver the payload (``PRE_EXECUTION``
                or ``MID_EXECUTION``).
            blocking: If ``True`` (default), block until full propagation is
                tracked before returning.  If ``False``, submit to a
                background thread and return immediately with
                ``completed_at=None``.
            track_propagation: If ``True`` (default), populate
                ``propagation_path``, ``influenced_agents``, and
                ``resistant_agents`` on the returned ``AttackResult``.

        Returns:
            An ``AttackResult``.  If ``blocking=False``, the result is a
            skeleton with ``completed_at=None``; the completed result is
            written to the log by the background thread.

        Raises:
            ValueError: If *agent_id* is not found in the config, or if
                *attack_type* is not a valid ``AttackType``.
        """
        # --- Validate inputs ---
        if self._config.get_agent(agent_id) is None:
            raise ValueError(
                f"Agent '{agent_id}' not found in MAS '{self._config.mas_id}'"
            )
        try:
            attack_type_enum = AttackType(attack_type)
        except ValueError as exc:
            raise ValueError(
                f"Unknown attack_type '{attack_type}'. "
                f"Valid values: {[t.value for t in AttackType]}"
            ) from exc
        try:
            phase_enum = InjectionPhase(injection_phase)
        except ValueError as exc:
            raise ValueError(
                f"Unknown injection_phase '{injection_phase}'. "
                f"Valid values: {[p.value for p in InjectionPhase]}"
            ) from exc

        now = datetime.datetime.now(datetime.timezone.utc)
        attack_id = str(uuid.uuid4())

        skeleton = AttackResult(
            attack_id=attack_id,
            mas_id=self._config.mas_id,
            target_agent_id=agent_id,
            attack_type=attack_type_enum,
            injection_phase=phase_enum,
            payload=payload,
            injected_at=now,
            completed_at=None,
            success=False,
        )

        if not blocking:
            future: Future = self._thread_pool.submit(
                self._run_attack,
                attack_id,
                agent_id,
                attack_type_enum,
                payload,
                phase_enum,
                now,
                track_propagation,
            )
            future.add_done_callback(self._log_future_exception)
            LOGGER.info(
                "AttackInjector: submitted non-blocking attack %s for agent '%s'",
                attack_id,
                agent_id,
            )
            return skeleton

        return self._run_attack(
            attack_id,
            agent_id,
            attack_type_enum,
            payload,
            phase_enum,
            now,
            track_propagation,
        )

    def _run_attack(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        attack_id: str,
        agent_id: str,
        attack_type: AttackType,
        payload: str,
        phase: InjectionPhase,
        injected_at: datetime.datetime,
        track_propagation: bool,
    ) -> AttackResult:
        """Execute the attack and return a completed ``AttackResult``.

        This method is called either synchronously or from the thread pool.
        It always logs the result via ``AttackLogger`` before returning.
        """
        tracker = PropagationTracker(payload, agent_id) if track_propagation else None
        error: Optional[str] = None
        completed_at: Optional[datetime.datetime] = None
        run_id: Optional[str] = None

        try:
            if phase == InjectionPhase.PRE_EXECUTION:
                run_id = self._run_pre_execution(
                    agent_id, attack_type, payload, tracker
                )
            elif phase == InjectionPhase.CHECKPOINT:
                run_id = self._run_checkpoint_execution(
                    agent_id, attack_type, payload, tracker
                )
            else:
                self._run_mid_execution(agent_id, attack_type, payload, tracker)
            completed_at = datetime.datetime.now(datetime.timezone.utc)
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
            completed_at = datetime.datetime.now(datetime.timezone.utc)
            LOGGER.error("AttackInjector: attack %s failed: %s", attack_id, exc)

        result = AttackResult(
            attack_id=attack_id,
            run_id=run_id,
            mas_id=self._config.mas_id,
            target_agent_id=agent_id,
            attack_type=attack_type,
            injection_phase=phase,
            payload=payload,
            injected_at=injected_at,
            completed_at=completed_at,
            propagation_path=tracker.propagation_path() if tracker else [],
            influenced_agents=tracker.influenced_agents() if tracker else [],
            resistant_agents=tracker.resistant_agents() if tracker else set(),
            success=error is None,
            error=error,
        )

        try:
            self._attack_logger.log(result)
        except Exception as log_exc:  # pylint: disable=broad-except
            LOGGER.error(
                "AttackInjector: failed to log result for attack %s: %s",
                attack_id,
                log_exc,
            )
        if self._security_detector is not None:
            self._security_detector.detect(result)
        return result

    def _run_pre_execution(  # pylint: disable=too-many-locals
        self,
        agent_id: str,
        attack_type: AttackType,
        payload: str,
        tracker: Optional[PropagationTracker],
    ) -> Optional[str]:
        """Apply a pre-execution strategy and run the MAS.

        A deep-copied patched config is created and fed to a fresh
        ``MASExecutor``.  ``PropagationTracker.observe()`` is called for each
        agent result after execution completes.

        Returns:
            The ``run_id`` from the ``MASExecutionResult``, for correlation
            with security event logs.
        """
        from bili.aether.attacks.strategies import (  # pylint: disable=import-outside-toplevel
            pre_execution,
        )
        from bili.aether.runtime.executor import (  # pylint: disable=import-outside-toplevel
            MASExecutor,
        )

        if attack_type == AttackType.PERSISTENCE:
            raise ValueError(
                "AttackType.PERSISTENCE must use InjectionPhase.CHECKPOINT "
                "and is dispatched via _run_checkpoint_execution, not "
                "_run_pre_execution.  Check the injection_phase argument."
            )
        strategy_fn_name = _STRATEGY_MAP[attack_type]
        strategy_fn = getattr(pre_execution, strategy_fn_name)
        patched_config = strategy_fn(self._config, agent_id, payload)

        attack_executor = MASExecutor(patched_config)
        attack_executor.initialize()
        mas_result = attack_executor.run(save_results=False)

        if tracker is not None:
            # Observe each agent using the available result data
            final_state = mas_result.final_state or {}
            agent_specs = {a.agent_id: a for a in patched_config.agents}

            for agent_result in mas_result.agent_results:
                spec = agent_specs.get(agent_result.agent_id)
                role = spec.role if spec else agent_result.role
                input_state = {
                    "messages": final_state.get("messages", []),
                    "objective": spec.objective if spec else "",
                }
                tracker.observe(
                    agent_id=agent_result.agent_id,
                    role=role,
                    input_state=input_state,
                    output_state=agent_result.output,
                    attack_type=attack_type.value,
                )

        return mas_result.run_id

    def _run_checkpoint_execution(  # pylint: disable=too-many-locals
        self,
        agent_id: str,
        attack_type: AttackType,
        payload: str,
        tracker: Optional[PropagationTracker],
    ) -> Optional[str]:
        """Inject a poisoned message via the checkpointer and re-run the MAS.

        Execution flow:

        1. Compile the graph with a checkpointer.  The checkpointer is created
           from ``self._config.checkpoint_config`` via the checkpointer factory;
           if the factory is unavailable, ``MemorySaver`` is used as a fallback
           (suitable for in-process simulation but not true cross-session persistence).
        2. Run the graph once under a fresh ``thread_id`` to establish an initial
           checkpoint state.
        3. Inject the poisoned ``HumanMessage`` via
           ``compiled_graph.update_state()`` — which internally calls the
           checkpointer's ``put()`` API — bypassing normal agent input validation.
        4. Simulate session teardown by recompiling the graph from the same
           ``MASConfig`` while retaining the same checkpointer instance.
        5. Invoke the recompiled graph under the same ``thread_id``.  The
           checkpointer loads the poisoned state, making it appear as legitimate
           prior-session context to all agents.
        6. Observe each agent's output via ``PropagationTracker``.

        Returns:
            The ``thread_id`` used as a surrogate ``run_id`` for log correlation.

        Raises:
            RuntimeError: If the checkpointer cannot be created.
        """
        from bili.aether.attacks.strategies import (
            persistence as checkpoint_strategy,  # pylint: disable=import-outside-toplevel
        )
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        # Create checkpointer — try factory first, fall back to MemorySaver.
        checkpointer = self._create_checkpointer()

        thread_id = str(uuid.uuid4())
        invoke_config = {"configurable": {"thread_id": thread_id}}

        # Guard: refuse to run if the resolved checkpointer is MemorySaver.
        # MemorySaver is in-process only and does not demonstrate cross-session
        # persistence.  The persistence suite runner should have already skipped
        # this config, but this guard catches direct API calls.
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
            MemorySaver,
        )

        if isinstance(checkpointer, MemorySaver):
            raise RuntimeError(
                "inject_persistence: resolved checkpointer is MemorySaver, which is "
                "in-process only and does not survive session teardown. "
                "Configure a postgres or mongo checkpointer to run persistence attacks."
            )

        # Phase 1: initial run to establish checkpoint state.
        compiled_mas = compile_mas(self._config)
        compiled_graph = compiled_mas.compile_graph(checkpointer=checkpointer)
        compiled_graph.invoke({"messages": []}, config=invoke_config)

        # Phase 2: inject via checkpointer's put() API.
        checkpoint_strategy.inject_persistence(compiled_graph, thread_id, payload)

        # Phase 3: session teardown simulation — recompile with the same
        # checkpointer instance so the poisoned state is loaded on next invoke.
        compiled_mas_2 = compile_mas(self._config)
        compiled_graph_2 = compiled_mas_2.compile_graph(checkpointer=checkpointer)

        # Phase 4: re-invoke; agents receive the poisoned message as prior context.
        result_state = compiled_graph_2.invoke({"messages": []}, config=invoke_config)

        # Phase 5: observe each agent.
        if tracker is not None:
            messages = result_state.get("messages", []) if result_state else []
            for agent_spec in self._config.agents:
                role = agent_spec.role if agent_spec.role else agent_spec.agent_id
                input_state = {
                    "messages": messages,
                    "objective": agent_spec.objective if agent_spec.objective else "",
                }
                # Known limitation: the reload invoke returns a flat message list,
                # not per-agent results, so all agents share the same output_excerpt
                # (the last AIMessage in the reloaded state).  This differs from
                # _run_pre_execution which has per-agent agent_result.output.
                output_excerpt = ""
                for msg in reversed(messages):
                    msg_type = type(msg).__name__
                    if msg_type in ("AIMessage", "AIMessageChunk"):
                        content = getattr(msg, "content", "")
                        output_excerpt = content[:500] if content else ""
                        break
                output_state = {
                    "messages": messages,
                    "output": output_excerpt,
                }
                tracker.observe(
                    agent_id=agent_spec.agent_id,
                    role=role,
                    input_state=input_state,
                    output_state=output_state,
                    attack_type=attack_type.value,
                )

        return thread_id

    def _create_checkpointer(self) -> object:
        """Create a checkpointer from config, falling back to MemorySaver.

        Returns:
            A LangGraph checkpointer instance.
        """
        checkpoint_type = (self._config.checkpoint_config or {}).get("type", "memory")
        if self._config.checkpoint_enabled and checkpoint_type not in (
            "memory",
            "auto",
        ):
            try:
                # pylint: disable=import-outside-toplevel
                from bili.aether.integration.checkpointer_factory import (
                    create_checkpointer_from_config,
                )

                # pylint: enable=import-outside-toplevel
                return create_checkpointer_from_config(
                    self._config.checkpoint_config, user_id=None
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "AttackInjector: could not create configured checkpointer "
                    "(%s: %s); falling back to MemorySaver",
                    type(exc).__name__,
                    exc,
                )
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
            MemorySaver,
        )

        return MemorySaver()

    def _run_mid_execution(
        self,
        agent_id: str,
        attack_type: AttackType,
        payload: str,
        tracker: Optional[PropagationTracker],
    ) -> None:
        """Compile the graph with interrupt flags and run mid-execution injection.

        A fresh ``CompiledMAS`` is built from the original config so the
        ``interrupt_before`` flag can be applied without affecting the
        executor's own compiled graph.
        """
        from bili.aether.attacks.strategies import (  # pylint: disable=import-outside-toplevel
            mid_execution,
        )
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        compiled_mas = compile_mas(self._config)
        # When track_propagation=False, tracker is None so a throwaway
        # PropagationTracker is created here to satisfy the required signature
        # of run_with_mid_execution_injection.  Its observations are discarded
        # at the _run_attack call site, which checks `tracker` (None), not
        # `effective_tracker`.  This is intentional.
        effective_tracker = tracker or PropagationTracker(payload, agent_id)
        invoke_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        mid_execution.run_with_mid_execution_injection(
            compiled_mas=compiled_mas,
            input_data={"messages": []},
            target_agent_id=agent_id,
            payload=payload,
            tracker=effective_tracker,
            invoke_config=invoke_config,
            attack_type=attack_type.value,
        )

    def _log_future_exception(self, future: Future) -> None:
        """Done-callback: log any exception raised by a background attack thread."""
        exc = future.exception()
        if exc:
            LOGGER.error("AttackInjector: background attack thread failed: %s", exc)

    def close(self) -> None:
        """Shut down the background thread pool, waiting for pending attacks to finish."""
        self._thread_pool.shutdown(wait=True)

    def __enter__(self) -> "AttackInjector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
