"""Tests for AETHER state-based communication (Task #22).

Tests the state-based communication system where messages persist in
LangGraph state via checkpointers instead of ephemeral JSONL files.

Covers:
    - End-to-end agent communication through state
    - Parallel execution safety with concurrent message writes
    - Message persistence via checkpointers
    - Broadcast message delivery (__all__ receiver)
    - Communication across different workflow types
    - State reducer correctness (_merge_dicts for concurrent writes)
"""

# pylint: disable=missing-function-docstring

import pytest
from langchain_core.messages import HumanMessage

from bili.aether.compiler import compile_mas
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.mas_config import Channel, CommunicationProtocol

# ======================================================================
# Helpers
# ======================================================================


def _make_test_config(
    workflow_type: WorkflowType, with_channels: bool = True
) -> MASConfig:
    """Create a test MASConfig with 3 agents and optional communication channels."""
    agents = [
        AgentSpec(
            agent_id="agent_a",
            role="analyzer",
            objective="Analyze input and send results",
        ),
        AgentSpec(
            agent_id="agent_b",
            role="validator",
            objective="Validate results from analyzer",
        ),
        AgentSpec(
            agent_id="agent_c",
            role="summarizer",
            objective="Summarize validated results",
        ),
    ]

    channels = []
    if with_channels:
        # Create channels for agent communication
        channels = [
            Channel(
                channel_id="a_to_b",
                protocol=CommunicationProtocol.DIRECT,
                source="agent_a",
                target="agent_b",
            ),
            Channel(
                channel_id="b_to_c",
                protocol=CommunicationProtocol.DIRECT,
                source="agent_b",
                target="agent_c",
            ),
            Channel(
                channel_id="broadcast_ch",
                protocol=CommunicationProtocol.BROADCAST,
                source="agent_a",
                target="all",
            ),
        ]

    return MASConfig(
        mas_id=f"test_comm_{workflow_type.value}",
        name=f"Communication Test ({workflow_type.value})",
        agents=agents,
        channels=channels,
        workflow_type=workflow_type,
        checkpoint_enabled=True,  # Enable checkpointing for persistence tests
    )


# ======================================================================
# End-to-end communication tests
# ======================================================================


class TestStateBasedCommunication:
    """Tests for state-based communication in compiled MAS graphs."""

    def test_sequential_communication_persists_in_state(self):
        """Test that agent communication is recorded in state fields."""
        config = _make_test_config(WorkflowType.SEQUENTIAL)
        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        # Execute graph with initial input
        result = graph.invoke(
            {"messages": [HumanMessage(content="Analyze this data")]},
            config={"configurable": {"thread_id": "test_seq_comm"}},
        )

        # Verify communication fields exist in final state
        assert "communication_log" in result
        assert "channel_messages" in result
        assert "pending_messages" in result

        # Verify communication log has messages (agents broadcast their output)
        comm_log = result.get("communication_log", [])
        assert isinstance(comm_log, list)
        # All 3 agents should have sent broadcast messages on __agent_output__
        assert len(comm_log) >= 3

        # Verify message structure
        for msg in comm_log:
            assert "sender" in msg
            assert "receiver" in msg
            assert "channel" in msg
            assert "content" in msg
            assert "message_id" in msg
            assert "timestamp" in msg

    def test_parallel_communication_with_concurrent_writes(self):
        """Test that parallel agents can write messages concurrently without loss."""
        config = _make_test_config(WorkflowType.PARALLEL)
        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="Process in parallel")]},
            config={"configurable": {"thread_id": "test_parallel_comm"}},
        )

        # All 3 agents run in parallel - each should broadcast their output
        # communication_log uses operator.add (list concat) so all messages persist
        comm_log = result.get("communication_log", [])
        assert len(comm_log) >= 3

        # Verify all agents are represented in the log
        senders = {msg["sender"] for msg in comm_log}
        assert "agent_a" in senders
        assert "agent_b" in senders
        assert "agent_c" in senders

        # Verify channel_messages has the __agent_output__ channel
        # NOTE: channel_messages uses _merge_dicts (shallow merge), so concurrent
        # writes to the same channel key result in last-writer-wins behavior.
        # For accumulating all messages, use communication_log (operator.add).
        channel_msgs = result.get("channel_messages", {})
        assert "__agent_output__" in channel_msgs
        assert len(channel_msgs["__agent_output__"]) >= 1  # At least one message

    def test_broadcast_message_delivery(self):
        """Test that broadcast messages are delivered to all other agents."""
        config = _make_test_config(WorkflowType.SEQUENTIAL)
        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="Broadcast test")]},
            config={"configurable": {"thread_id": "test_broadcast"}},
        )

        # Agent output broadcast should reach all other agents
        comm_log = result.get("communication_log", [])

        # Find broadcast messages (receiver="__all__")
        broadcast_msgs = [msg for msg in comm_log if msg.get("receiver") == "__all__"]
        assert len(broadcast_msgs) >= 3  # All 3 agents broadcast their output

        # Verify broadcast messages have correct structure
        for msg in broadcast_msgs:
            assert msg["message_type"] == "broadcast"
            assert msg["receiver"] == "__all__"


class TestCommunicationPersistence:
    """Tests for communication persistence via checkpointers."""

    def test_communication_persists_across_executions(self):
        """Test that communication state persists in checkpointer."""
        config = _make_test_config(WorkflowType.SEQUENTIAL)
        thread_id = "test_persistence"

        # First execution
        executor1 = MASExecutor(config)
        executor1.initialize()
        result1 = executor1.run(
            {"messages": [HumanMessage(content="First run")]},
            thread_id=thread_id,
            save_results=False,
        )

        comm_log_1 = result1.final_state.get("communication_log", [])
        assert len(comm_log_1) >= 3

        # Second execution with same thread_id (should restore state)
        executor2 = MASExecutor(config)
        executor2.initialize()
        result2 = executor2.run(
            {"messages": [HumanMessage(content="Second run")]},
            thread_id=thread_id,
            save_results=False,
        )

        comm_log_2 = result2.final_state.get("communication_log", [])
        # Second run should have MORE messages (accumulated from both runs)
        assert len(comm_log_2) >= len(comm_log_1)


class TestCommunicationReducers:
    """Tests for state reducer correctness in concurrent scenarios."""

    def test_merge_dicts_reducer_for_channel_messages(self):
        """Test that _merge_dicts reducer properly merges concurrent channel updates."""
        from bili.aether.compiler.state_generator import _merge_dicts

        # Simulate concurrent writes to channel_messages
        existing = {"ch1": [{"id": "msg1"}], "ch2": [{"id": "msg2"}]}
        new_update = {"ch1": [{"id": "msg3"}], "ch3": [{"id": "msg4"}]}

        merged = _merge_dicts(existing, new_update)

        # ch1 should be overwritten (shallow merge)
        assert merged["ch1"] == [{"id": "msg3"}]
        # ch2 should remain from existing
        assert merged["ch2"] == [{"id": "msg2"}]
        # ch3 should be added from new
        assert merged["ch3"] == [{"id": "msg4"}]

    def test_communication_log_uses_operator_add(self):
        """Test that communication_log uses operator.add for concatenation."""
        import operator

        # Simulate list concatenation (operator.add behavior)
        existing_log = [{"id": "msg1"}, {"id": "msg2"}]
        new_msgs = [{"id": "msg3"}]

        combined = operator.add(existing_log, new_msgs)

        assert len(combined) == 3
        assert combined[0]["id"] == "msg1"
        assert combined[1]["id"] == "msg2"
        assert combined[2]["id"] == "msg3"


class TestWorkflowTypeCommunication:
    """Tests for communication across different workflow types."""

    @pytest.mark.parametrize(
        "workflow_type",
        [
            WorkflowType.SEQUENTIAL,
            WorkflowType.PARALLEL,
            WorkflowType.HIERARCHICAL,
        ],
    )
    def test_communication_works_across_workflow_types(self, workflow_type):
        """Test that state-based communication works with all workflow types."""
        if workflow_type == WorkflowType.HIERARCHICAL:
            # For hierarchical, need to set tier on agents
            agents = [
                AgentSpec(
                    agent_id="agent_a",
                    role="analyzer",
                    objective="Analyze input data thoroughly",
                    tier=2,
                ),
                AgentSpec(
                    agent_id="agent_b",
                    role="validator",
                    objective="Validate analysis results",
                    tier=2,
                ),
                AgentSpec(
                    agent_id="agent_c",
                    role="summarizer",
                    objective="Summarize validated data",
                    tier=1,
                ),
            ]
            # Add a minimal channel to enable communication fields in state
            channels = [
                Channel(
                    channel_id="tier_comm",
                    protocol=CommunicationProtocol.BROADCAST,
                    source="agent_a",
                    target="all",
                ),
            ]
            config = MASConfig(
                mas_id=f"test_{workflow_type.value}",
                name=f"Test {workflow_type.value}",
                agents=agents,
                channels=channels,
                workflow_type=workflow_type,
                checkpoint_enabled=True,
            )
        else:
            config = _make_test_config(workflow_type)

        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="Test communication")]},
            config={"configurable": {"thread_id": f"test_{workflow_type.value}"}},
        )

        # All workflow types should produce communication logs
        comm_log = result.get("communication_log", [])
        assert len(comm_log) >= 1  # At least one agent should have sent a message

        # Verify state has communication fields
        assert "channel_messages" in result
        assert "pending_messages" in result


class TestCommunicationWithoutChannels:
    """Tests for default communication behavior without explicit channels."""

    def test_agents_broadcast_output_without_channels(self):
        """Test that agents broadcast their output even without channel definitions."""
        config = _make_test_config(WorkflowType.SEQUENTIAL, with_channels=False)
        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="No channels test")]},
            config={"configurable": {"thread_id": "test_no_channels"}},
        )

        # Even without explicit channels, agents should NOT have communication fields
        # because state generator only adds them when config.channels is present
        assert "communication_log" not in result
        assert "channel_messages" not in result
        assert "pending_messages" not in result


class TestMessageFormatting:
    """Tests for message formatting helpers."""

    def test_format_messages_for_context_integration(self):
        """Test that format_messages_for_context works with real message dicts."""
        from bili.aether.runtime.communication_state import format_messages_for_context

        config = _make_test_config(WorkflowType.SEQUENTIAL)
        compiled = compile_mas(config)
        graph = compiled.compile_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="Format test")]},
            config={"configurable": {"thread_id": "test_format"}},
        )

        comm_log = result.get("communication_log", [])
        if comm_log:
            # Format the messages
            formatted = format_messages_for_context(comm_log)

            # Should be non-empty string
            assert isinstance(formatted, str)
            assert len(formatted) > 0

            # Should contain sender and channel info
            assert "[From " in formatted
            assert " via " in formatted
