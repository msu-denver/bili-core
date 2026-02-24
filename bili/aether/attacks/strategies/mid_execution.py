"""Mid-execution injection strategy for the Attack Injection Framework.

Uses LangGraph's ``interrupt_before`` compile-time flag and the
``Command(resume=...)`` pattern to pause execution before the target node,
inject the payload into the graph state, then resume.

After resuming, ``graph.stream(stream_mode="updates")`` is used to observe
each downstream node's state delta without pausing execution (unlike the
``interrupt_after``-on-all-nodes approach, which would require catching and
handling a NodeInterrupt for every node — complex and error-prone).

--- Research Note ---
``interrupt_before`` pauses the graph before the target node runs.  Streaming
with ``stream_mode="updates"`` yields ``{node_name: state_delta}`` chunks.
This gives us per-node input→output visibility for PropagationTracker without
interrupting execution flow after injection.  The tradeoff is that the
"input_state" seen by observe() is the *accumulated* state just before the
node, which is a good approximation for research purposes.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage  # pylint: disable=import-error

LOGGER = logging.getLogger(__name__)


def run_with_mid_execution_injection(  # pylint: disable=too-many-locals
    compiled_mas: Any,
    input_data: dict,
    target_agent_id: str,
    payload: str,
    tracker: Any,
    invoke_config: dict | None = None,
) -> dict:
    """Run the MAS with a mid-execution payload injection at *target_agent_id*.

    Execution flow:
        1. Compile the graph with ``interrupt_before=[target_agent_id]``.
        2. ``invoke()`` — graph pauses before the target node, raising
           ``NodeInterrupt``.
        3. Catch ``NodeInterrupt``, inject payload into the interrupted state
           via :func:`_apply_payload_to_state`.
        4. ``stream(Command(resume=modified_state), stream_mode="updates")``
           — yields ``{node_name: delta}`` per node so PropagationTracker can
           observe each agent's input→output transition.
        5. Accumulate the streamed deltas into a final state dict and return.

    Args:
        compiled_mas: A ``CompiledMAS`` instance (not yet compiled with
            interrupt flags).
        input_data: Initial graph input (e.g. ``{"messages": [...]}``)
        target_agent_id: The agent node to pause before and inject into.
        payload: The adversarial string to inject.
        tracker: A ``PropagationTracker`` instance.  ``observe()`` is called
            for each agent node encountered during streaming.
        invoke_config: LangGraph config dict (e.g. ``{"configurable":
            {"thread_id": "..."}}``) for checkpoint threading.

    Returns:
        The accumulated final state dict after all nodes have executed.

    Raises:
        RuntimeError: If ``NodeInterrupt`` fires at a node other than
            *target_agent_id*, or if ``NodeInterrupt`` is never raised
            (meaning the target agent was not reached during execution).
    """
    from langgraph.errors import (  # pylint: disable=import-outside-toplevel,import-error
        NodeInterrupt,
    )
    from langgraph.types import (  # pylint: disable=import-outside-toplevel,import-error
        Command,
    )

    invoke_config = invoke_config or {}

    # Build lookup: agent_id -> role for PropagationTracker
    agent_roles: dict[str, str] = {
        a.agent_id: a.role for a in compiled_mas.config.agents
    }
    attack_type = "prompt_injection"  # default; caller may pass via tracker if needed

    # Step 1: Compile with interrupt_before at the target node
    graph = compiled_mas.compile_graph(
        interrupt_before=[target_agent_id],
    )

    # Step 2: Run until NodeInterrupt
    interrupted_state: dict | None = None
    try:
        graph.invoke(input_data, config=invoke_config)
        # If we reach here, NodeInterrupt was never raised
        raise RuntimeError(
            f"NodeInterrupt was never raised — target agent '{target_agent_id}' "
            "was not reached during execution."
        )
    except NodeInterrupt as exc:
        node_name = getattr(exc, "node", None)
        # LangGraph may encode the interrupted node name differently across
        # versions; fall back to checking the exception message.
        if node_name is not None and node_name != target_agent_id:
            raise RuntimeError(
                f"Expected NodeInterrupt at '{target_agent_id}', " f"got '{node_name}'"
            ) from exc
        # Retrieve the interrupted graph state
        interrupted_state = _get_interrupt_state(graph, invoke_config)
        LOGGER.debug(
            "run_with_mid_execution_injection: paused before '%s'",
            target_agent_id,
        )

    # Step 3: Inject payload into the interrupted state
    modified_state = _apply_payload_to_state(interrupted_state or input_data, payload)

    # Step 4 + 5: Stream with Command(resume=...) and observe each node
    accumulated: dict = {}
    all_agent_ids = {a.agent_id for a in compiled_mas.config.agents}

    for chunk in graph.stream(
        Command(resume=modified_state),
        config=invoke_config,
        stream_mode="updates",
    ):
        for node_name, delta in chunk.items():
            if node_name in all_agent_ids:
                role = agent_roles.get(node_name, node_name)
                tracker.observe(
                    agent_id=node_name,
                    role=role,
                    input_state=dict(accumulated),
                    output_state=delta if isinstance(delta, dict) else {},
                    attack_type=attack_type,
                )
            if isinstance(delta, dict):
                accumulated.update(delta)

    return accumulated


def _apply_payload_to_state(state: dict, payload: str) -> dict:
    """Return a copy of *state* with *payload* appended as a ``HumanMessage``.

    This inserts the adversarial payload into the LangGraph message channel
    so it is visible to the target node (and any downstream nodes that read
    the message history).

    Args:
        state: The graph state dict at the point of interruption.
        payload: The adversarial string to inject.

    Returns:
        A shallow copy of *state* with the messages list extended.
    """
    new_state = dict(state)
    existing_messages = list(new_state.get("messages", []))
    existing_messages.append(HumanMessage(content=payload))
    new_state["messages"] = existing_messages
    return new_state


def _get_interrupt_state(graph: Any, invoke_config: dict) -> dict:
    """Retrieve the current graph state after a NodeInterrupt.

    Uses ``graph.get_state(config)`` when available (LangGraph 1.x).
    Falls back to an empty dict if unavailable.

    Args:
        graph: The compiled LangGraph ``CompiledStateGraph``.
        invoke_config: The invoke config dict used for checkpoint threading.

    Returns:
        The current state dict, or an empty dict on failure.
    """
    try:
        snapshot = graph.get_state(invoke_config)
        return dict(snapshot.values) if hasattr(snapshot, "values") else {}
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Could not retrieve interrupt state: %s", exc)
        return {}
