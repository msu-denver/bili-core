"""Checkpoint-phase injection strategy for the Attack Injection Framework.

The persistence attack differs fundamentally from pre-execution and mid-execution
strategies: rather than patching an agent's ``objective`` or interrupting the
graph mid-run, it writes adversarial content **directly into the checkpointer's
persisted state** via the graph's ``update_state()`` method (which internally
calls the checkpointer's ``put()`` API).

Threat model
------------
An attacker with write access to the checkpointer backend (e.g. a compromised
PostgreSQL or MongoDB instance) can plant adversarial messages in a conversation
thread's history.  When the MAS resumes that thread in a subsequent session, it
loads the poisoned state as legitimate prior context and processes it without any
indication that it was externally injected.

This is the only attack type that survives a full session teardown and reload —
the poisoned content is not in the agent's working memory (memory poisoning) or
current input (prompt injection), but in the durable checkpoint store.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage  # pylint: disable=import-error

LOGGER = logging.getLogger(__name__)


def inject_persistence(compiled_graph: Any, thread_id: str, payload: str) -> None:
    """Write a poisoned HumanMessage directly into the checkpointer under *thread_id*.

    Uses ``compiled_graph.update_state()`` — the public graph API that
    internally calls the checkpointer's ``put()`` method — to inject the payload
    into the persisted message history.  The injection bypasses normal graph
    execution: the message is written to the checkpoint store without passing
    through any agent node or input validation.

    The injected message is wrapped as::

        HumanMessage(content="[Persisted context: {payload}]")

    This framing makes the message appear to be legitimate prior-session user
    input when loaded by the graph in a subsequent run.

    Args:
        compiled_graph: A compiled LangGraph graph with a non-``None``
            checkpointer.  The checkpointer must already hold state for
            *thread_id* (i.e. the graph must have been invoked at least once
            under this thread before injection).
        thread_id: The conversation thread identifier to poison.
        payload: The adversarial string to embed in the persisted history.

    Raises:
        RuntimeError: If *compiled_graph* has no checkpointer attached.
    """
    if compiled_graph.checkpointer is None:
        raise RuntimeError(
            "inject_persistence requires a non-None checkpointer. "
            "Ensure checkpoint_enabled=True and a persistent backend "
            "(postgres or mongo) is configured in the MASConfig."
        )

    invoke_config = {"configurable": {"thread_id": thread_id}}
    poisoned_message = HumanMessage(content=f"[Persisted context: {payload}]")

    LOGGER.debug(
        "inject_persistence: writing poisoned HumanMessage to thread_id=%r",
        thread_id,
    )
    # update_state appends to the message reducer, which internally calls
    # checkpointer.put() with the modified checkpoint.
    compiled_graph.update_state(invoke_config, {"messages": [poisoned_message]})
    LOGGER.info(
        "inject_persistence: payload written to checkpointer for thread_id=%r",
        thread_id,
    )
