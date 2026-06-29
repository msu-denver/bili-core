"""
Module: audit

Human-readable audit view over a multi-agent checkpoint history.

Given a checkpointer and a thread_id, ``audit_view()`` iterates the
checkpoint sequence produced by a single MAS run and returns a structured
timeline showing which agent acted at each superstep, what it produced, and
which inter-agent messages it sent.

This is a read-only, zero-side-effect utility — it calls only
``checkpointer.list()`` and deserializes the returned tuples.

Functions:
    - audit_view(checkpointer, thread_id, checkpoint_ns)

Example::

    from bili.iris.checkpointers.jsonl_checkpointer import get_jsonl_checkpointer
    from bili.aether.runtime.audit import audit_view

    saver = get_jsonl_checkpointer(path="run.jsonl")
    timeline = audit_view(saver, thread_id="run-001")
    for step in timeline:
        print(step["ts"], step["agent_id"], step["output_summary"])

Timeline entries
----------------
Only supersteps where at least one agent acted (i.e. ``agent_outputs``
changed or a ``communication_log`` entry was appended) are included in
the returned list.  The initial LangGraph checkpoint (written before any
node runs) and the empty state-initialisation superstep are skipped so
the timeline starts cleanly with the first agent turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def audit_view(
    checkpointer: Any,
    thread_id: str,
    checkpoint_ns: str = "",
) -> List[Dict[str, Any]]:
    """Build a human-readable timeline from a MAS run's checkpoint history.

    Iterates ``checkpointer.list()`` for *thread_id* in chronological order,
    diffs ``agent_outputs`` and ``communication_log`` between consecutive
    supersteps, and emits one entry per superstep where something changed.

    Args:
        checkpointer: Any object with a ``list(config)`` method that yields
            ``CheckpointTuple`` objects (e.g. ``JSONLCheckpointSaver``,
            ``QueryableMemorySaver``, ``PruningPostgresSaver``).
        thread_id: Thread ID of the run to inspect.
        checkpoint_ns: Checkpoint namespace (empty string for MAS runs).

    Returns:
        A list of step dicts in chronological order (oldest first):

        .. code-block:: python

            [
                {
                    "step":           int,          # 1-based superstep index
                    "ts":             str | None,   # ISO-8601 timestamp from checkpoint
                    "checkpoint_id":  str,
                    "agent_id":       str | None,   # which agent acted this step
                    "output_summary": str | None,   # first 200 chars of agent output
                    "messages_sent":  list[dict],   # new communication_log entries
                    "raw_agent_outputs": dict,      # full agent_outputs delta
                },
                ...
            ]

        Returns an empty list if no checkpoints exist for the thread_id.

    Raises:
        TypeError: If *checkpointer* does not expose a ``list()`` method.
    """
    config: Dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }

    # list() yields most-recent-first; reverse for chronological order
    try:
        tuples = list(checkpointer.list(config))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error(
            "audit_view: failed to list checkpoints for thread '%s': %s",
            thread_id,
            exc,
        )
        raise

    if not tuples:
        LOGGER.debug("audit_view: no checkpoints found for thread '%s'", thread_id)
        return []

    # Reverse to chronological order (oldest superstep first)
    tuples = list(reversed(tuples))

    timeline: List[Dict[str, Any]] = []
    prev_agent_outputs: Dict[str, Any] = {}
    prev_comm_log: List[Any] = []

    for step_idx, tup in enumerate(tuples, start=1):
        checkpoint = tup.checkpoint or {}
        channel_values: Dict[str, Any] = checkpoint.get("channel_values", {})

        current_agent_outputs: Dict[str, Any] = channel_values.get("agent_outputs", {})
        current_comm_log: List[Any] = channel_values.get("communication_log", [])

        # Diff agent_outputs: find keys that are new or changed this step
        changed_outputs: Dict[str, Any] = {}
        for agent_id, output in current_agent_outputs.items():
            if (
                agent_id not in prev_agent_outputs
                or prev_agent_outputs[agent_id] != output
            ):
                changed_outputs[agent_id] = output

        # Diff communication_log: find entries appended this step
        new_messages: List[Any] = current_comm_log[len(prev_comm_log) :]

        # Identify the acting agent (the one that changed outputs this step).
        # ``current_agent`` is set by every agent node; fall back to the first
        # key in changed_outputs when the channel is absent (e.g. pre-fix
        # checkpoints).
        acting_agent: Optional[str] = channel_values.get("current_agent") or None
        if not acting_agent and changed_outputs:
            acting_agent = next(iter(changed_outputs))

        # Skip supersteps where no agent activity occurred.  These are the
        # LangGraph-internal initial/state-seed checkpoints written before the
        # first agent node runs (current_agent is None or '', no new
        # agent_outputs, no new communication_log entries).
        if not acting_agent and not changed_outputs and not new_messages:
            prev_agent_outputs = dict(current_agent_outputs)
            prev_comm_log = list(current_comm_log)
            continue

        # Summarise the acting agent's output
        output_summary: Optional[str] = None
        if acting_agent and acting_agent in changed_outputs:
            raw_output = changed_outputs[acting_agent]
            if isinstance(raw_output, dict):
                text = (
                    raw_output.get("message")
                    or raw_output.get("content")
                    or raw_output.get("response")
                    or str(raw_output)
                )
            else:
                text = str(raw_output)
            output_summary = text[:200] if text else None

        # Serialize messages_sent to plain dicts (they may be dataclasses)
        serialized_messages: List[Dict[str, Any]] = []
        for msg in new_messages:
            if isinstance(msg, dict):
                serialized_messages.append(msg)
            elif hasattr(msg, "to_log_dict"):
                serialized_messages.append(msg.to_log_dict())
            elif hasattr(msg, "__dict__"):
                serialized_messages.append(vars(msg))
            else:
                serialized_messages.append({"raw": str(msg)})

        # Extract timestamp from the checkpoint dict.
        # LangGraph metadata does not carry a "ts" field; the source of truth
        # is checkpoint["ts"] set at put() time.
        ts: Optional[str] = checkpoint.get("ts")

        timeline.append(
            {
                "step": step_idx,
                "ts": ts,
                "checkpoint_id": tup.config["configurable"].get("checkpoint_id"),
                "agent_id": acting_agent,
                "output_summary": output_summary,
                "messages_sent": serialized_messages,
                "raw_agent_outputs": changed_outputs,
            }
        )

        prev_agent_outputs = dict(current_agent_outputs)
        prev_comm_log = list(current_comm_log)

    return timeline
