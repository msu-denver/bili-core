"""State integration helpers for agent communication.

Provides pure-function helpers that agent nodes call to interact with
the communication layer through LangGraph state, without needing a
direct reference to the ``ChannelManager``.

State fields used:
    ``channel_messages``  — ``Dict[str, list]``  channel_id -> list of message dicts
                            (uses _merge_dicts reducer for parallel execution safety;
                            present only when the MAS declares explicit channels)
    ``pending_messages``  — ``Dict[str, list]``  agent_id -> list of message dicts
                            (uses _merge_dicts reducer for parallel execution safety;
                            present only when the MAS declares explicit channels)
    ``communication_log`` — ``list``             flat list of all message dicts
                            (uses operator.add reducer, preserves order by completion;
                            ALWAYS present — even without explicit channels — so
                            per-agent provenance is durably checkpointed for every run)

Reducer contract for ``communication_log``
------------------------------------------
The state schema uses ``operator.add`` as the reducer, which *concatenates*
lists.  This means the state update returned by :func:`send_message_in_state`
must contain ONLY the *new* message(s) as a single-element list — NOT the
full accumulated log read from state.  If the full log were returned, the
reducer would compute ``existing_log + (existing_log + new_msg)``, doubling
every previous entry on each superstep.
"""

import logging
from typing import Any, Dict, List, Optional

from bili.aether.runtime.messages import Message, MessageType

LOGGER = logging.getLogger(__name__)


def send_message_in_state(
    state: dict,
    channel_id: str,
    sender: str,
    content: str,
    receiver: str = "__all__",
    message_type: MessageType = MessageType.DIRECT,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """Create a message and return the state-update delta for communication fields.

    This is a *pure* helper — it does not mutate *state* in place.
    The caller merges the returned dict into the LangGraph state update.

    ``communication_log`` in the returned dict contains ONLY the new message
    as a single-element list.  The ``operator.add`` reducer in the state
    schema concatenates this list onto the accumulated log, producing the
    correct append-only history.  Do NOT pre-populate it with the existing
    log from *state* — that would cause the reducer to double-count every
    prior entry on each subsequent superstep.

    Args:
        state: Current LangGraph state dict.
        channel_id: Channel to send on.
        sender: Sending agent ID.
        content: Message body.
        receiver: Receiving agent ID or ``__all__``.
        message_type: Message category.
        metadata: Optional metadata.

    Returns:
        Dict with updated ``channel_messages``, ``pending_messages``,
        and ``communication_log`` (delta-only single-element list).
    """
    msg = Message(
        sender=sender,
        receiver=receiver,
        channel=channel_id,
        content=content,
        message_type=message_type,
        metadata=metadata or {},
    )
    msg_dict = msg.to_log_dict()

    channel_messages = _update_channel_messages(state, channel_id, msg_dict)
    pending = _update_pending_messages(state, sender, receiver, msg_dict)

    # Return only the new message as the communication_log delta.
    # The operator.add reducer appends this list to the accumulated log.
    return {
        "channel_messages": channel_messages,
        "pending_messages": pending,
        "communication_log": [msg_dict],
    }


def _update_channel_messages(
    state: dict, channel_id: str, msg_dict: dict
) -> Dict[str, list]:
    """Append *msg_dict* to the channel_messages entry for *channel_id*."""
    channel_messages = dict(state.get("channel_messages") or {})
    channel_msgs = list(channel_messages.get(channel_id, []))
    channel_msgs.append(msg_dict)
    channel_messages[channel_id] = channel_msgs
    return channel_messages


def _update_pending_messages(
    state: dict, sender: str, receiver: str, msg_dict: dict
) -> Dict[str, list]:
    """Append *msg_dict* to the pending_messages for the appropriate agent(s).

    Broadcast messages (``receiver == "__all__"``) are stored under the
    ``"__all__"`` key rather than expanded to individual agents.  In
    sequential workflows, downstream agents have not yet run and are
    therefore absent from ``agent_outputs``; using a shared key means
    future agents can retrieve the broadcast via ``get_pending_messages``.
    """
    pending = dict(state.get("pending_messages") or {})
    if receiver == "__all__":
        # Store under the shared broadcast key so agents that have not yet
        # executed (and are therefore not in agent_outputs) can still read it.
        broadcast_pending = list(pending.get("__all__", []))
        broadcast_pending.append(msg_dict)
        pending["__all__"] = broadcast_pending
    else:
        agent_pending = list(pending.get(receiver, []))
        agent_pending.append(msg_dict)
        pending[receiver] = agent_pending
    return pending


def get_pending_messages(state: dict, agent_id: str) -> List[Dict[str, Any]]:
    """Return pending message dicts for *agent_id* without modifying state.

    Returns both direct messages addressed to *agent_id* and broadcast
    messages stored under the ``"__all__"`` key (excluding any sent by
    *agent_id* itself to avoid self-feedback loops).

    Returns:
        A list of message dicts (may be empty).
    """
    pending = state.get("pending_messages") or {}
    direct = list(pending.get(agent_id, []))
    broadcasts = [m for m in pending.get("__all__", []) if m.get("sender") != agent_id]
    return broadcasts + direct


def format_messages_for_context(messages: List[Dict[str, Any]]) -> str:
    """Format message dicts as human-readable text for LLM context injection.

    Example output::

        [From reviewer via reviewer_to_judge]: Content analysis looks good.
        [From policy_expert via policy_channel]: No policy violations found.

    Returns:
        A newline-separated string, or empty string if no messages.
    """
    if not messages:
        return ""

    lines = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        channel = msg.get("channel", "unknown")
        content = msg.get("content", "")
        lines.append(f"[From {sender} via {channel}]: {content}")

    return "\n".join(lines)
