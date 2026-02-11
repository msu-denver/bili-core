"""Communication channel implementations.

Provides ``DirectChannel``, ``BroadcastChannel``, and
``RequestResponseChannel`` — each backed by the ``Channel`` schema
model from ``bili.aether.schema``.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from bili.aether.runtime.messages import Message, MessageHistory, MessageType
from bili.aether.schema import CommunicationProtocol

LOGGER = logging.getLogger(__name__)


class CommunicationChannel(ABC):
    """Abstract base for all channel types.

    Subclasses implement ``send`` and ``receive`` with protocol-specific
    validation and delivery semantics.

    Args:
        channel_config: The ``Channel`` schema model describing this channel.
        agent_ids: All agent IDs in the MAS (used by broadcast).
    """

    def __init__(
        self, channel_config: Any, agent_ids: Optional[List[str]] = None
    ) -> None:
        self.channel_config = channel_config
        self.history = MessageHistory()
        self._agent_ids = agent_ids or []
        self._delivered: Dict[str, List[str]] = (
            {}
        )  # agent_id -> list of message_ids already read

    @property
    def channel_id(self) -> str:
        """Shortcut to the underlying config's channel_id."""
        return self.channel_config.channel_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def send(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Send a message on this channel. Returns the created ``Message``."""

    def receive(self, agent_id: str) -> List[Message]:
        """Return unread messages for *agent_id* on this channel."""
        all_msgs = self.history.get_messages_for(agent_id)
        already = set(self._delivered.get(agent_id, []))
        new_msgs = [m for m in all_msgs if m.message_id not in already]
        # Mark as delivered
        self._delivered.setdefault(agent_id, []).extend(m.message_id for m in new_msgs)
        return new_msgs

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_sender(self, sender: str) -> None:
        """Raise ``ValueError`` if *sender* is not allowed on this channel."""
        source = self.channel_config.source
        if source == "any":
            return
        if self.channel_config.bidirectional:
            if sender not in (source, self.channel_config.target):
                raise ValueError(
                    f"Sender '{sender}' not allowed on bidirectional channel "
                    f"'{self.channel_id}' (source={source}, target={self.channel_config.target})"
                )
            return
        if sender != source:
            raise ValueError(
                f"Sender '{sender}' != expected source '{source}' "
                f"on channel '{self.channel_id}'"
            )


# ======================================================================
# Concrete channel types
# ======================================================================


class DirectChannel(CommunicationChannel):
    """Point-to-point channel: source -> target.

    If ``bidirectional`` is set, either end may send.
    """

    def send(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        self._validate_sender(sender)

        # Determine receiver: if bidirectional, receiver is the *other* end
        cfg = self.channel_config
        if cfg.bidirectional:
            receiver = cfg.target if sender == cfg.source else cfg.source
        else:
            receiver = cfg.target

        msg = Message(
            sender=sender,
            receiver=receiver,
            channel=self.channel_id,
            content=content,
            message_type=MessageType.DIRECT,
            metadata=metadata or {},
        )
        self.history.add_message(msg)
        return msg


class BroadcastChannel(CommunicationChannel):
    """One-to-many channel: source -> all agents (except sender)."""

    def send(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        self._validate_sender(sender)

        msg = Message(
            sender=sender,
            receiver="__all__",
            channel=self.channel_id,
            content=content,
            message_type=MessageType.BROADCAST,
            metadata=metadata or {},
        )
        self.history.add_message(msg)
        return msg

    def receive(self, agent_id: str) -> List[Message]:
        """Broadcasts are delivered to everyone except the sender."""
        all_msgs = [
            m
            for m in self.history.messages
            if m.sender != agent_id  # exclude own broadcasts
        ]
        already = set(self._delivered.get(agent_id, []))
        new_msgs = [m for m in all_msgs if m.message_id not in already]
        self._delivered.setdefault(agent_id, []).extend(m.message_id for m in new_msgs)
        return new_msgs


class RequestResponseChannel(CommunicationChannel):
    """Bidirectional request/response channel with correlation tracking.

    Always operates between ``source`` and ``target`` in both directions.
    """

    def __init__(
        self, channel_config: Any, agent_ids: Optional[List[str]] = None
    ) -> None:
        super().__init__(channel_config, agent_ids)
        self._pending_requests: Dict[str, Message] = {}  # message_id -> request Message

    def send(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Send a request (alias for ``send_request``)."""
        return self.send_request(sender, content, metadata)

    def send_request(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Send a request message. Returns the request ``Message``."""
        self._validate_sender_reqresp(sender)
        cfg = self.channel_config
        receiver = cfg.target if sender == cfg.source else cfg.source

        msg = Message(
            sender=sender,
            receiver=receiver,
            channel=self.channel_id,
            content=content,
            message_type=MessageType.REQUEST,
            metadata=metadata or {},
        )
        self._pending_requests[msg.message_id] = msg
        self.history.add_message(msg)
        return msg

    def send_response(
        self,
        sender: str,
        request_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Send a response to a previous request.

        Args:
            sender: The responding agent ID.
            request_id: ``message_id`` of the original request.
            content: Response body.
            metadata: Optional metadata.

        Raises:
            ValueError: If *request_id* is not a pending request.
        """
        if request_id not in self._pending_requests:
            raise ValueError(
                f"No pending request '{request_id}' on channel '{self.channel_id}'"
            )

        original = self._pending_requests.pop(request_id)
        receiver = original.sender  # respond back to requester

        msg = Message(
            sender=sender,
            receiver=receiver,
            channel=self.channel_id,
            content=content,
            message_type=MessageType.RESPONSE,
            metadata=metadata or {},
            in_reply_to=request_id,
        )
        self.history.add_message(msg)
        return msg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_sender_reqresp(self, sender: str) -> None:
        """Either endpoint may send on a request/response channel."""
        cfg = self.channel_config
        if sender not in (cfg.source, cfg.target):
            raise ValueError(
                f"Sender '{sender}' not allowed on request/response channel "
                f"'{self.channel_id}' (source={cfg.source}, target={cfg.target})"
            )


# ======================================================================
# Factory
# ======================================================================

_CHANNEL_TYPE_MAP = {
    CommunicationProtocol.DIRECT: DirectChannel,
    CommunicationProtocol.BROADCAST: BroadcastChannel,
    CommunicationProtocol.REQUEST_RESPONSE: RequestResponseChannel,
}


def create_channel(
    channel_config: Any,
    agent_ids: Optional[List[str]] = None,
) -> CommunicationChannel:
    """Create the appropriate channel subclass for a ``Channel`` config.

    Falls back to ``DirectChannel`` for unimplemented protocol types
    (PUBSUB, COMPETITIVE, CONSENSUS).
    """
    cls = _CHANNEL_TYPE_MAP.get(channel_config.protocol, DirectChannel)
    if channel_config.protocol not in _CHANNEL_TYPE_MAP:
        LOGGER.warning(
            "Protocol '%s' not yet implemented; falling back to DirectChannel "
            "for channel '%s'",
            channel_config.protocol.value,
            channel_config.channel_id,
        )
    return cls(channel_config, agent_ids)
