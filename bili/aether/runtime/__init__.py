"""AETHER runtime — agent communication protocol.

Provides structured inter-agent messaging through declared channels,
with JSONL logging and LangGraph state integration.

Key classes:
    ``Message``              — Pydantic model for a single message.
    ``MessageType``          — Enum (DIRECT, BROADCAST, REQUEST, RESPONSE).
    ``MessageHistory``       — Ordered message collection with query helpers.
    ``CommunicationLogger``  — JSONL file writer for message audit trails.
    ``CommunicationChannel`` — ABC for channel implementations.
    ``DirectChannel``        — Point-to-point messaging.
    ``BroadcastChannel``     — One-to-many messaging.
    ``RequestResponseChannel`` — Bidirectional request/response.
    ``ChannelManager``       — Top-level orchestrator for all channels.
"""

from bili.aether.runtime.channel_manager import ChannelManager
from bili.aether.runtime.channels import (
    BroadcastChannel,
    CommunicationChannel,
    DirectChannel,
    RequestResponseChannel,
    create_channel,
)
from bili.aether.runtime.communication_state import (
    format_messages_for_context,
    get_pending_messages,
    send_message_in_state,
)
from bili.aether.runtime.logger import CommunicationLogger
from bili.aether.runtime.messages import Message, MessageHistory, MessageType

__all__ = [
    "BroadcastChannel",
    "ChannelManager",
    "CommunicationChannel",
    "CommunicationLogger",
    "DirectChannel",
    "Message",
    "MessageHistory",
    "MessageType",
    "RequestResponseChannel",
    "create_channel",
    "format_messages_for_context",
    "get_pending_messages",
    "send_message_in_state",
]
