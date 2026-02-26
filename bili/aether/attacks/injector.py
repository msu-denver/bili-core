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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from bili.aether.attacks.logger import AttackLogger
from bili.aether.attacks.models import AttackResult, AttackType, InjectionPhase
from bili.aether.attacks.propagation import PropagationTracker
from bili.aether.schema import MASConfig

LOGGER = logging.getLogger(__name__)

_STRATEGY_MAP = {
    AttackType.PROMPT_INJECTION: "inject_prompt_injection",
    AttackType.MEMORY_POISONING: "inject_memory_poisoning",
    AttackType.AGENT_IMPERSONATION: "inject_agent_impersonation",
    AttackType.BIAS_INHERITANCE: "inject_bias_inheritance",
}


class AttackInjector:
    """Inject adversarial payloads into a MAS and track propagation.

    Args:
        config: The ``MASConfig`` to inject into.
        executor: An initialised (or uninitialised) ``MASExecutor`` instance.
            The executor is used to retrieve the config and, for mid-execution
            injections, is not directly invoked (a fresh graph is compiled).
        log_path: Optional path to the NDJSON attack log file.  Defaults to
            ``AttackLogger.DEFAULT_PATH``.
        max_workers: Thread-pool size for ``blocking=False`` calls.
    """

    def __init__(
        self,
        config: MASConfig,
        executor,
        log_path: Optional[Path] = None,
        max_workers: int = 4,
    ) -> None:
        self._config = config
        self._executor = executor
        self._logger = AttackLogger(log_path)
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
            self._thread_pool.submit(
                self._run_attack,
                attack_id,
                agent_id,
                attack_type_enum,
                payload,
                phase_enum,
                now,
                track_propagation,
            )
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

        try:
            if phase == InjectionPhase.PRE_EXECUTION:
                self._run_pre_execution(agent_id, attack_type, payload, tracker)
            else:
                self._run_mid_execution(agent_id, attack_type, payload, tracker)
            completed_at = datetime.datetime.now(datetime.timezone.utc)
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
            completed_at = datetime.datetime.now(datetime.timezone.utc)
            LOGGER.error("AttackInjector: attack %s failed: %s", attack_id, exc)

        result = AttackResult(
            attack_id=attack_id,
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

        self._logger.log(result)
        return result

    def _run_pre_execution(  # pylint: disable=too-many-locals
        self,
        agent_id: str,
        attack_type: AttackType,
        payload: str,
        tracker: Optional[PropagationTracker],
    ) -> None:
        """Apply a pre-execution strategy and run the MAS.

        A deep-copied patched config is created and fed to a fresh
        ``MASExecutor``.  ``PropagationTracker.observe()`` is called for each
        agent result after execution completes.
        """
        from bili.aether.attacks.strategies import (  # pylint: disable=import-outside-toplevel
            pre_execution,
        )
        from bili.aether.runtime.executor import (  # pylint: disable=import-outside-toplevel
            MASExecutor,
        )

        strategy_fn_name = _STRATEGY_MAP[attack_type]
        strategy_fn = getattr(pre_execution, strategy_fn_name)
        patched_config = strategy_fn(self._config, agent_id, payload)

        attack_executor = MASExecutor(patched_config)
        attack_executor.initialize()
        result = attack_executor.run()

        if tracker is None:
            return

        # Observe each agent using the available result data
        final_state = result.final_state or {}
        agent_specs = {a.agent_id: a for a in patched_config.agents}

        for agent_result in result.agent_results:
            spec = agent_specs.get(agent_result.agent_id)
            role = spec.role if spec else agent_result.role
            # Approximate input state: messages list + agent's configured objective
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

    def close(self) -> None:
        """Shut down the background thread pool."""
        self._thread_pool.shutdown(wait=False)

    def __enter__(self) -> "AttackInjector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
