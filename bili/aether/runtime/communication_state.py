"""State integration helpers for agent communication.

Provides pure-function helpers that agent nodes call to interact with
the communication layer through LangGraph state, without needing a
direct reference to the ``ChannelManager``.

State fields used:
    ``channel_messages``  — ``Dict[str, list]``  channel_id -> list of message dicts
                            (uses _merge_dicts reducer for parallel execution safety)
    ``pending_messages``  — ``Dict[str, list]``  agent_id -> list of message dicts
                            (uses _merge_dicts reducer for parallel execution safety)
    ``communication_log`` — ``list``             flat list of all message dicts
                            (uses operator.add reducer, preserves order by completion)
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
    """Create a message and return updated communication state fields.

    This is a *pure* helper — it does not mutate *state* in place.
    The caller merges the returned dict into the LangGraph state update.

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
        and ``communication_log`` fields.
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
    comm_log = list(state.get("communication_log") or [])
    comm_log.append(msg_dict)

    return {
        "channel_messages": channel_messages,
        "pending_messages": pending,
        "communication_log": comm_log,
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
