"""Structured message schema for inter-agent communication.

Defines the ``Message`` model (Pydantic), ``MessageType`` enum, and
``MessageHistory`` container used by channels and the communication logger.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    REQUEST = "request"
    RESPONSE = "response"


class Message(BaseModel):
    """A single inter-agent message.

    Attributes:
        message_id: Unique identifier (auto-generated UUID4).
        timestamp: ISO-8601 UTC timestamp (auto-generated).
        sender: Agent ID of the sender.
        receiver: Agent ID of the receiver, or ``__all__`` for broadcasts.
        channel: Channel ID this message was sent on.
        content: Message body text.
        message_type: Categorisation of the message.
        metadata: Arbitrary key-value metadata.
        in_reply_to: Optional message_id this is responding to.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sender: str
    receiver: str
    channel: str
    content: str
    message_type: MessageType = MessageType.DIRECT
    metadata: Dict[str, Any] = Field(default_factory=dict)
    in_reply_to: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        """Return a flat dict suitable for JSONL logging."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "channel": self.channel,
            "content": self.content,
            "message_type": self.message_type.value,
            "metadata": self.metadata,
            "in_reply_to": self.in_reply_to,
        }


class MessageHistory:
    """Ordered collection of ``Message`` objects with query helpers.

    Thread-safe for single-writer scenarios (one graph execution).
    """

    def __init__(self) -> None:
        self._messages: List[Message] = []

    @property
    def messages(self) -> List[Message]:
        """Return a shallow copy of the message list."""
        return list(self._messages)

    def add_message(self, message: Message) -> None:
        """Append a message to the history."""
        self._messages.append(message)

    def get_messages_for(self, agent_id: str) -> List[Message]:
        """Return messages where *agent_id* is the receiver or receiver is ``__all__``."""
        return [m for m in self._messages if m.receiver in (agent_id, "__all__")]

    def get_messages_from(self, agent_id: str) -> List[Message]:
        """Return messages sent by *agent_id*."""
        return [m for m in self._messages if m.sender == agent_id]

    def get_messages_on_channel(self, channel_id: str) -> List[Message]:
        """Return messages sent on a specific channel."""
        return [m for m in self._messages if m.channel == channel_id]

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Serialise all messages to a list of dicts."""
        return [m.to_log_dict() for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)
