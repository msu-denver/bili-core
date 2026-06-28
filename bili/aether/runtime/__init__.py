"""AETHER runtime — agent communication protocol.

Provides structured inter-agent messaging through declared channels,
with LangGraph state integration and checkpoint-backed audit views.

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

Key functions:
    ``audit_view``           — Human-readable timeline from checkpoint history.
"""

from bili.aether.runtime.audit import audit_view
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
from bili.aether.runtime.context import RuntimeContext
from bili.aether.runtime.execution_result import (
    AgentExecutionResult,
    MASExecutionResult,
)
from bili.aether.runtime.executor import MASExecutor, execute_mas
from bili.aether.runtime.logger import CommunicationLogger
from bili.aether.runtime.messages import Message, MessageHistory, MessageType
from bili.aether.runtime.streaming import StreamEvent, StreamEventType, StreamFilter

__all__ = [
    "AgentExecutionResult",
    "BroadcastChannel",
    "audit_view",
    "ChannelManager",
    "CommunicationChannel",
    "CommunicationLogger",
    "DirectChannel",
    "MASExecutionResult",
    "MASExecutor",
    "Message",
    "RuntimeContext",
    "MessageHistory",
    "MessageType",
    "RequestResponseChannel",
    "StreamEvent",
    "StreamEventType",
    "StreamFilter",
    "create_channel",
    "execute_mas",
    "format_messages_for_context",
    "get_pending_messages",
    "send_message_in_state",
]
