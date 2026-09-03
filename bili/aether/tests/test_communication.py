"""Tests for the AETHER agent communication protocol (Task 6).

Covers:
    - Message and MessageHistory
    - CommunicationLogger (JSONL output)
    - Channel types (Direct, Broadcast, RequestResponse)
    - ChannelManager
    - Communication state helpers
    - Integration with the compiler (state schema generation)
"""

# pylint: disable=missing-function-docstring

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bili.aether.runtime.channel_manager import ChannelManager
from bili.aether.runtime.channels import (
    BroadcastChannel,
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
from bili.aether.schema import AgentSpec, CommunicationProtocol, MASConfig
from bili.aether.schema.mas_config import Channel

# ======================================================================
# Helpers
# ======================================================================


def _make_channel_config(
    channel_id: str = "ch1",
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
    source: str = "agent_a",
    target: str = "agent_b",
    bidirectional: bool = False,
) -> Channel:
    """Create a minimal Channel config for testing."""
    return Channel(
        channel_id=channel_id,
        protocol=protocol,
        source=source,
        target=target,
        bidirectional=bidirectional,
    )


def _make_simple_mas(channels=None) -> MASConfig:
    """Build a minimal 2-agent MASConfig with optional channels."""
    agents = [
        AgentSpec(agent_id="agent_a", role="writer", objective="Write content"),
        AgentSpec(agent_id="agent_b", role="reviewer", objective="Review content"),
    ]
    return MASConfig(
        mas_id="test_comm",
        name="Communication Test MAS",
        agents=agents,
        channels=channels or [],
        workflow_type="sequential",
    )


# ======================================================================
# Message tests
# ======================================================================


class TestMessage:
    """Tests for the Message Pydantic model."""

    def test_message_creation_with_defaults(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            channel="ch1",
            content="Hello",
        )
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"
        assert msg.message_id  # UUID auto-generated
        assert msg.timestamp  # timestamp auto-generated
        assert msg.message_type == MessageType.DIRECT

    def test_message_to_log_dict(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            channel="ch1",
            content="Hello",
            message_type=MessageType.BROADCAST,
            metadata={"key": "value"},
        )
        d = msg.to_log_dict()
        assert d["sender"] == "agent_a"
        assert d["message_type"] == "broadcast"
        assert d["metadata"] == {"key": "value"}
        assert "message_id" in d
        assert "timestamp" in d

    def test_message_uuid_uniqueness(self):
        msgs = [
            Message(sender="a", receiver="b", channel="c", content="x")
            for _ in range(10)
        ]
        ids = {m.message_id for m in msgs}
        assert len(ids) == 10


# ======================================================================
# MessageHistory tests
# ======================================================================


class TestMessageHistory:
    """Tests for MessageHistory container."""

    def test_add_and_query_by_agent(self):
        history = MessageHistory()
        msg1 = Message(sender="a", receiver="b", channel="ch", content="hi")
        msg2 = Message(sender="b", receiver="a", channel="ch", content="hello")
        history.add_message(msg1)
        history.add_message(msg2)

        assert len(history) == 2
        assert len(history.get_messages_for("b")) == 1
        assert history.get_messages_for("b")[0].content == "hi"

    def test_query_by_channel(self):
        history = MessageHistory()
        history.add_message(
            Message(sender="a", receiver="b", channel="ch1", content="one")
        )
        history.add_message(
            Message(sender="a", receiver="b", channel="ch2", content="two")
        )
        assert len(history.get_messages_on_channel("ch1")) == 1
        assert history.get_messages_on_channel("ch1")[0].content == "one"

    def test_to_dicts(self):
        history = MessageHistory()
        history.add_message(
            Message(sender="a", receiver="b", channel="ch", content="data")
        )
        dicts = history.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["content"] == "data"

    def test_get_messages_from(self):
        history = MessageHistory()
        history.add_message(
            Message(sender="a", receiver="b", channel="ch", content="from_a")
        )
        history.add_message(
            Message(sender="b", receiver="a", channel="ch", content="from_b")
        )
        from_a = history.get_messages_from("a")
        assert len(from_a) == 1
        assert from_a[0].content == "from_a"


# ======================================================================
# CommunicationLogger tests
# ======================================================================


class TestCommunicationLogger:
    """Tests for JSONL logging."""

    def test_log_message_writes_jsonl(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            log_path = tmp.name

        try:
            with CommunicationLogger(log_path) as logger:
                msg = Message(sender="a", receiver="b", channel="ch1", content="logged")
                logger.log_message(msg)

            with open(log_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["sender"] == "a"
            assert data["content"] == "logged"
        finally:
            os.unlink(log_path)

    def test_context_manager_closes_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            log_path = tmp.name

        try:
            logger = CommunicationLogger(log_path)
            with logger:
                logger.log_message(
                    Message(sender="a", receiver="b", channel="c", content="x")
                )
            # After __exit__, file handle should be None
            assert logger._file_handle is None  # pylint: disable=protected-access
        finally:
            os.unlink(log_path)

    def test_log_message_opens_file_lazily(self):
        """log_message opens the file when no handle has been created yet."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            log_path = tmp.name
        os.unlink(log_path)
        try:
            logger = CommunicationLogger(log_path)
            assert logger._file_handle is None  # pylint: disable=protected-access
            logger.log_message(
                Message(sender="a", receiver="b", channel="c", content="lazy")
            )
            assert logger._file_handle is not None  # pylint: disable=protected-access
            logger.close()
            with open(log_path, encoding="utf-8") as fh:
                assert json.loads(fh.readline())["content"] == "lazy"
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_flush_flushes_open_handle(self):
        """flush forwards to the underlying file handle when one is open."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            log_path = tmp.name
        try:
            logger = CommunicationLogger(log_path)
            logger.log_message(
                Message(sender="a", receiver="b", channel="c", content="x")
            )
            handle = logger._file_handle  # pylint: disable=protected-access
            handle.flush = MagicMock()
            logger.flush()
            handle.flush.assert_called_once()
            logger.close()
        finally:
            os.unlink(log_path)

    def test_flush_noop_without_open_handle(self):
        """flush is a no-op when no file handle has been opened."""
        logger = CommunicationLogger("/nonexistent/never-opened.jsonl")
        # Should not raise even though no handle exists.
        logger.flush()
        assert logger._file_handle is None  # pylint: disable=protected-access

    def test_close_logs_warning_on_oserror(self):
        """A close failure is caught, logged, and the handle is cleared."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            log_path = tmp.name
        try:
            logger = CommunicationLogger(log_path)
            logger.log_message(
                Message(sender="a", receiver="b", channel="c", content="x")
            )
            handle = logger._file_handle  # pylint: disable=protected-access
            handle.flush = MagicMock(side_effect=OSError("disk gone"))
            with patch("bili.aether.runtime.logger.LOGGER.warning") as mock_warning:
                logger.close()
            mock_warning.assert_called_once()
            assert logger._file_handle is None  # pylint: disable=protected-access
        finally:
            os.unlink(log_path)


# ======================================================================
# Channel tests
# ======================================================================


class TestDirectChannel:
    """Tests for DirectChannel."""

    def test_send_and_receive(self):
        cfg = _make_channel_config()
        ch = DirectChannel(cfg)

        msg = ch.send("agent_a", "test content")
        assert msg.receiver == "agent_b"
        assert msg.message_type == MessageType.DIRECT

        received = ch.receive("agent_b")
        assert len(received) == 1
        assert received[0].content == "test content"

        # Second receive returns nothing (already delivered)
        assert len(ch.receive("agent_b")) == 0

    def test_bidirectional_send(self):
        cfg = _make_channel_config(bidirectional=True)
        ch = DirectChannel(cfg)

        msg1 = ch.send("agent_a", "from a")
        assert msg1.receiver == "agent_b"

        msg2 = ch.send("agent_b", "from b")
        assert msg2.receiver == "agent_a"

    def test_invalid_sender_raises(self):
        cfg = _make_channel_config()
        ch = DirectChannel(cfg)
        with pytest.raises(ValueError, match="Sender 'agent_c'"):
            ch.send("agent_c", "not allowed")


class TestBroadcastChannel:  # pylint: disable=too-few-public-methods
    """Tests for BroadcastChannel."""

    def test_broadcast_delivery(self):
        cfg = _make_channel_config(
            protocol=CommunicationProtocol.BROADCAST,
            target="all",
        )
        ch = BroadcastChannel(cfg, agent_ids=["agent_a", "agent_b", "agent_c"])

        ch.send("agent_a", "announcement")

        # agent_b and agent_c should receive; agent_a (sender) should not
        assert len(ch.receive("agent_b")) == 1
        assert len(ch.receive("agent_c")) == 1
        assert len(ch.receive("agent_a")) == 0


class TestRequestResponseChannel:
    """Tests for RequestResponseChannel."""

    def test_request_response_flow(self):
        cfg = _make_channel_config(
            protocol=CommunicationProtocol.REQUEST_RESPONSE,
        )
        ch = RequestResponseChannel(cfg)

        request = ch.send_request("agent_a", "What is the policy?")
        assert request.message_type == MessageType.REQUEST

        response = ch.send_response("agent_b", request.message_id, "Policy is X.")
        assert response.message_type == MessageType.RESPONSE
        assert response.in_reply_to == request.message_id

    def test_response_to_unknown_request_raises(self):
        cfg = _make_channel_config(
            protocol=CommunicationProtocol.REQUEST_RESPONSE,
        )
        ch = RequestResponseChannel(cfg)
        with pytest.raises(ValueError, match="No pending request"):
            ch.send_response("agent_b", "fake_id", "response")


# ======================================================================
# create_channel factory tests
# ======================================================================


class TestCreateChannel:
    """Tests for the channel factory function."""

    def test_creates_direct_channel(self):
        cfg = _make_channel_config(protocol=CommunicationProtocol.DIRECT)
        ch = create_channel(cfg)
        assert isinstance(ch, DirectChannel)

    def test_creates_broadcast_channel(self):
        cfg = _make_channel_config(
            protocol=CommunicationProtocol.BROADCAST, target="all"
        )
        ch = create_channel(cfg)
        assert isinstance(ch, BroadcastChannel)

    def test_creates_request_response_channel(self):
        cfg = _make_channel_config(protocol=CommunicationProtocol.REQUEST_RESPONSE)
        ch = create_channel(cfg)
        assert isinstance(ch, RequestResponseChannel)

    def test_fallback_for_unimplemented_protocol(self):
        cfg = _make_channel_config(protocol=CommunicationProtocol.PUBSUB)
        ch = create_channel(cfg)
        # Falls back to DirectChannel
        assert isinstance(ch, DirectChannel)


# ======================================================================
# ChannelManager tests
# ======================================================================


class TestChannelManager:
    """Tests for ChannelManager."""

    def test_initialize_from_config(self):
        channels = [
            _make_channel_config("direct_ch", CommunicationProtocol.DIRECT),
            _make_channel_config(
                "broadcast_ch", CommunicationProtocol.BROADCAST, target="all"
            ),
        ]
        config = _make_simple_mas(channels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ChannelManager.initialize_from_config(config, log_dir=tmp_dir)
            assert len(mgr.channel_ids) == 2
            assert "direct_ch" in mgr.channel_ids
            assert "broadcast_ch" in mgr.channel_ids
            mgr.close()

    def test_send_and_receive(self):
        channels = [_make_channel_config("ch1", CommunicationProtocol.DIRECT)]
        config = _make_simple_mas(channels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ChannelManager.initialize_from_config(config, log_dir=tmp_dir)
            mgr.send_message("ch1", "agent_a", "hello agent_b")

            pending = mgr.get_messages_for_agent("agent_b")
            assert len(pending) == 1
            assert pending[0].content == "hello agent_b"

            # Global history should also have the message
            assert len(mgr.global_history) == 1
            mgr.close()

    def test_unknown_channel_raises(self):
        mgr = ChannelManager()
        with pytest.raises(KeyError, match="Channel 'missing'"):
            mgr.send_message("missing", "a", "content")

    def test_enable_logging_warns_and_writes_jsonl(self):
        """enable_logging=True warns (deprecated) and persists sent messages to JSONL."""
        channels = [_make_channel_config("ch1", CommunicationProtocol.DIRECT)]
        config = _make_simple_mas(channels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.warns(DeprecationWarning, match="JSONL logging"):
                mgr = ChannelManager.initialize_from_config(
                    config, log_dir=tmp_dir, enable_logging=True
                )
            # Sending a message exercises the logger.log_message branch.
            mgr.send_message("ch1", "agent_a", "logged message")
            mgr.close()

            jsonl_files = [f for f in os.listdir(tmp_dir) if f.endswith(".jsonl")]
            assert len(jsonl_files) == 1
            contents = os.path.join(tmp_dir, jsonl_files[0])
            with open(contents, encoding="utf-8") as fh:
                logged = json.loads(fh.readline())
            assert logged["content"] == "logged message"

    def test_get_channel_returns_registered_channel(self):
        """get_channel looks up a channel by its registered id."""
        channels = [_make_channel_config("ch1", CommunicationProtocol.DIRECT)]
        config = _make_simple_mas(channels)
        mgr = ChannelManager.initialize_from_config(config)
        channel = mgr.get_channel("ch1")
        assert channel is not None
        with pytest.raises(KeyError):
            mgr.get_channel("does_not_exist")
        mgr.close()


# ======================================================================
# Communication state helpers tests
# ======================================================================


class TestCommunicationStateHelpers:
    """Tests for send_message_in_state, get_pending_messages, format_messages_for_context."""

    def test_send_message_in_state(self):
        state = {
            "channel_messages": {},
            "pending_messages": {},
            "communication_log": [],
            "agent_outputs": {"agent_a": {}, "agent_b": {}},
        }

        update = send_message_in_state(
            state,
            channel_id="ch1",
            sender="agent_a",
            content="hello",
            receiver="agent_b",
        )

        assert "ch1" in update["channel_messages"]
        assert len(update["channel_messages"]["ch1"]) == 1
        assert len(update["pending_messages"]["agent_b"]) == 1
        assert len(update["communication_log"]) == 1

    def test_get_pending_messages(self):
        state = {
            "pending_messages": {
                "agent_b": [{"sender": "a", "content": "hi"}],
            }
        }
        msgs = get_pending_messages(state, "agent_b")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_get_pending_messages_empty(self):
        assert not get_pending_messages({}, "agent_x")

    def test_format_messages_for_context(self):
        msgs = [
            {"sender": "reviewer", "channel": "review_ch", "content": "Looks good"},
            {"sender": "expert", "channel": "expert_ch", "content": "No issues"},
        ]
        text = format_messages_for_context(msgs)
        assert "[From reviewer via review_ch]: Looks good" in text
        assert "[From expert via expert_ch]: No issues" in text

    def test_format_messages_empty(self):
        assert format_messages_for_context([]) == ""


# ======================================================================
# Integration: state schema with communication fields
# ======================================================================


class TestStateSchemaIntegration:
    """Test that communication fields appear in generated state schemas."""

    def test_state_schema_includes_communication_fields(self):
        channels = [_make_channel_config()]
        config = _make_simple_mas(channels)

        from bili.aether.compiler.state_generator import (  # pylint: disable=import-outside-toplevel
            generate_state_schema,
        )

        schema = generate_state_schema(config)
        annotations = schema.__annotations__

        assert "channel_messages" in annotations
        assert "pending_messages" in annotations
        assert "communication_log" in annotations

    def test_state_schema_without_channels(self):
        config = _make_simple_mas(channels=[])

        from bili.aether.compiler.state_generator import (  # pylint: disable=import-outside-toplevel
            generate_state_schema,
        )

        schema = generate_state_schema(config)
        annotations = schema.__annotations__

        # channel_messages and pending_messages are routing auxiliaries — present
        # only when the MAS declares explicit inter-agent channels.
        assert "channel_messages" not in annotations
        assert "pending_messages" not in annotations

        # communication_log is always present regardless of channel configuration
        # so per-agent provenance is captured for every run.
        assert "communication_log" in annotations
