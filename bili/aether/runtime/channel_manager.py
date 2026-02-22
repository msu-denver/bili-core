"""Channel manager — creates and routes messages across all channels in a MAS.

DEPRECATION NOTICE:
    ChannelManager's JSONL logging is deprecated. AETHER now uses state-based
    communication where messages are persisted in LangGraph state via checkpointers.
    This makes communication cloud-ready (survives pod restarts, works in K8s).

    The state-based approach uses:
    - bili.aether.runtime.communication_state.send_message_in_state()
    - bili.aether.runtime.communication_state.get_pending_messages()

    JSONL logging (enable_logging=True) is kept for backward compatibility but
    should not be used in production (files are ephemeral in cloud deployments).

Provides a single entry point for agent nodes to send and receive
messages without knowing the underlying channel topology.
"""

import logging
import os
import uuid
import warnings
from typing import Any, Dict, List, Optional

from bili.aether.runtime.channels import CommunicationChannel, create_channel
from bili.aether.runtime.logger import CommunicationLogger
from bili.aether.runtime.messages import Message, MessageHistory

LOGGER = logging.getLogger(__name__)


class ChannelManager:
    """Manages all communication channels for a compiled MAS.

    DEPRECATION NOTICE:
        ChannelManager is being transitioned to a state-based communication model.
        Agent nodes should use ``communication_state.send_message_in_state()``
        and ``communication_state.get_pending_messages()`` directly instead of
        ChannelManager. This makes communication cloud-ready by persisting
        messages in LangGraph state via checkpointers.

    Responsibilities:
        * Create channel instances from ``MASConfig.channels``.
        * Route ``send_message`` calls to the correct channel.
        * Aggregate pending messages for an agent across all channels.
        * ~~Log every message via ``CommunicationLogger`` (DEPRECATED)~~

    Legacy Usage::

        mgr = ChannelManager.initialize_from_config(mas_config)
        mgr.send_message("reviewer_to_judge", "reviewer", "Looks good.")
        pending = mgr.get_messages_for_agent("judge")
        mgr.close()

    Recommended (State-Based) Usage::

        # In agent node:
        from bili.aether.runtime.communication_state import (
            send_message_in_state,
            get_pending_messages
        )

        # Send message
        state_update = send_message_in_state(
            state, "channel_id", "sender", "content", receiver="agent_id"
        )
        # Agent returns state_update to merge into graph state

        # Receive messages
        pending = get_pending_messages(state, agent_id)
    """

    def __init__(
        self,
        channels: Optional[Dict[str, CommunicationChannel]] = None,
        logger: Optional[CommunicationLogger] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> None:
        self._channels: Dict[str, CommunicationChannel] = channels or {}
        self._logger = logger
        self._agent_ids = agent_ids or []
        self._global_history = MessageHistory()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def initialize_from_config(
        cls,
        config: Any,
        log_dir: Optional[str] = None,
        enable_logging: bool = False,
    ) -> "ChannelManager":
        """Build a ``ChannelManager`` from a ``MASConfig``.

        Args:
            config: A validated ``MASConfig`` instance.
            log_dir: Directory for the JSONL log file. Defaults to cwd.
            enable_logging: **DEPRECATED**. Whether to create a JSONL communication log.
                Defaults to ``False``. JSONL logging is deprecated in favor of
                state-based communication (messages persist in checkpointer).
                JSONL files are ephemeral in cloud deployments and should not be
                used for production communication.

        Returns:
            A fully initialised ``ChannelManager``.

        Warnings:
            DeprecationWarning: When enable_logging=True, warns that JSONL logging
                is deprecated and not suitable for production cloud deployments.
        """
        agent_ids = [a.agent_id for a in config.agents]
        channels: Dict[str, CommunicationChannel] = {}

        for chan_cfg in config.channels:
            channels[chan_cfg.channel_id] = create_channel(chan_cfg, agent_ids)

        # Set up logger only if explicitly enabled
        logger = None
        if enable_logging:
            warnings.warn(
                "JSONL logging (enable_logging=True) is deprecated. "
                "AETHER now uses state-based communication where messages are "
                "persisted in LangGraph state via checkpointers. This makes "
                "communication cloud-ready (survives pod restarts, works in K8s). "
                "JSONL files are ephemeral in cloud deployments and should not be "
                "used for production. See bili.aether.runtime.communication_state "
                "for the state-based API.",
                DeprecationWarning,
                stacklevel=2,
            )
            log_dir = log_dir or os.getcwd()
            log_filename = f"communication_{config.mas_id}_{uuid.uuid4().hex[:8]}.jsonl"
            log_path = os.path.join(log_dir, log_filename)
            logger = CommunicationLogger(log_path)
            LOGGER.warning(
                "ChannelManager initialised for '%s' with %d channels, logging to %s "
                "(DEPRECATED: use state-based communication instead)",
                config.mas_id,
                len(channels),
                log_path,
            )
        else:
            LOGGER.info(
                "ChannelManager initialised for '%s' with %d channels (state-based communication recommended)",
                config.mas_id,
                len(channels),
            )

        return cls(channels=channels, logger=logger, agent_ids=agent_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_message(
        self,
        channel_id: str,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Send a message on the specified channel.

        Args:
            channel_id: Target channel ID.
            sender: Agent ID of the sender.
            content: Message body.
            metadata: Optional metadata dict.

        Returns:
            The created ``Message``.

        Raises:
            KeyError: If *channel_id* is not registered.
        """
        channel = self._get_channel(channel_id)
        msg = channel.send(sender, content, metadata)
        self._global_history.add_message(msg)

        if self._logger is not None:
            self._logger.log_message(msg)

        LOGGER.debug(
            "Message %s sent on channel '%s': %s -> %s",
            msg.message_id,
            channel_id,
            sender,
            msg.receiver,
        )
        return msg

    def get_messages_for_agent(self, agent_id: str) -> List[Message]:
        """Collect unread messages for *agent_id* across all channels."""
        pending: List[Message] = []
        for channel in self._channels.values():
            pending.extend(channel.receive(agent_id))
        return pending

    def get_channel(self, channel_id: str) -> CommunicationChannel:
        """Look up a channel by ID.

        Raises:
            KeyError: If *channel_id* is not registered.
        """
        return self._get_channel(channel_id)

    @property
    def channel_ids(self) -> List[str]:
        """Return all registered channel IDs."""
        return list(self._channels.keys())

    @property
    def global_history(self) -> MessageHistory:
        """Return the global (cross-channel) message history."""
        return self._global_history

    def close(self) -> None:
        """Flush and close the communication logger."""
        if self._logger is not None:
            self._logger.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_channel(self, channel_id: str) -> CommunicationChannel:
        """Look up a channel, raising ``KeyError`` if not found."""
        if channel_id not in self._channels:
            raise KeyError(
                f"Channel '{channel_id}' not found. "
                f"Available: {list(self._channels.keys())}"
            )
        return self._channels[channel_id]
