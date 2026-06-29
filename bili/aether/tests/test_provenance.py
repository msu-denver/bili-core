"""Tests for per-agent provenance capture in the AETHER checkpointed state.

Verifies that a completed multi-agent run durably records, in every
checkpointed superstep, the three provenance channels required for full
post-run observability:

1. ``current_agent`` — which agent acted at each superstep.
2. ``agent_outputs[agent_id]`` — the output attributed to that agent,
   and the corresponding ``AIMessage`` tagged with ``name=<agent_id>``.
3. ``communication_log`` — one broadcast entry per agent handoff, present
   even when no explicit inter-agent channels are declared.

Root-cause regression tests are also included to pin the two bugs that
were fixed:

- Bug A: ``communication_log`` was absent from the state schema (and hence
  never checkpointed) for sequential workflows that declared no explicit
  channels.  Fix: always include ``communication_log`` in the schema.

- Bug B: ``send_message_in_state`` returned the full accumulated log as the
  state-update value.  Because the reducer is ``operator.add`` (list
  concatenation), this caused exponential duplication — a 3-agent run
  produced 7 entries (1+3+7 pattern) instead of 3.  Fix: return only the
  delta (single-element list) for ``communication_log``.

All tests use stub agents (``model_name`` not set) and a MemorySaver
checkpointer so no LLM API calls or database servers are needed.
"""

# pylint: disable=missing-function-docstring

import operator

from langchain_core.messages import HumanMessage

from bili.aether.runtime.audit import audit_view
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import (
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    WorkflowType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    defaults = {"role": "test_role", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


def _seq_config(
    n_agents: int = 3,
    with_channels: bool = False,
    mas_id: str = "prov_test",
) -> MASConfig:
    """Build a sequential MASConfig with stub agents and a MemorySaver."""
    agents = [_agent(f"agent_{i}") for i in range(n_agents)]
    channels = []
    if with_channels:
        for i in range(n_agents - 1):
            channels.append(
                Channel(
                    channel_id=f"ch_{i}_to_{i+1}",
                    protocol=CommunicationProtocol.DIRECT,
                    source=f"agent_{i}",
                    target=f"agent_{i + 1}",
                )
            )
    return MASConfig(
        mas_id=mas_id,
        name="Provenance Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=agents,
        channels=channels,
        checkpoint_enabled=True,
        checkpoint_config={"type": "memory"},
    )


def _run(config: MASConfig, thread_id: str = "prov-001"):
    """Initialize, run, and return (executor, result)."""
    executor = MASExecutor(config)
    executor.initialize()
    result = executor.run(
        input_data={"messages": [HumanMessage(content="start")]},
        thread_id=thread_id,
        save_results=False,
    )
    return executor, result


def _checkpoints_chronological(executor, thread_id: str):
    """Return checkpoint dicts in chronological order (oldest first)."""
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    tuples = list(executor._checkpointer.list(cfg))  # pylint: disable=protected-access
    return list(reversed(tuples))


# ---------------------------------------------------------------------------
# current_agent provenance
# ---------------------------------------------------------------------------


class TestCurrentAgentProvenance:
    """current_agent is set to the active agent ID in every agent superstep."""

    def test_current_agent_set_per_superstep_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        tuples = _checkpoints_chronological(executor, "prov-001")

        # Collect agent-superstep checkpoints (skip initial empty ones)
        agent_steps = [
            t.checkpoint["channel_values"].get("current_agent")
            for t in tuples
            if t.checkpoint["channel_values"].get("current_agent")
        ]
        # Every agent ran exactly once and is represented in order
        assert agent_steps == ["agent_0", "agent_1", "agent_2"]

    def test_current_agent_set_per_superstep_with_channels(self):
        executor, result = _run(
            _seq_config(n_agents=3, with_channels=True),
            thread_id="prov-ch-001",
        )
        assert result.success
        tuples = _checkpoints_chronological(executor, "prov-ch-001")

        agent_steps = [
            t.checkpoint["channel_values"].get("current_agent")
            for t in tuples
            if t.checkpoint["channel_values"].get("current_agent")
        ]
        assert agent_steps == ["agent_0", "agent_1", "agent_2"]

    def test_current_agent_final_state(self):
        # The final accumulated state reflects the last agent that ran.
        _, result = _run(_seq_config(n_agents=2))
        assert result.final_state.get("current_agent") == "agent_1"


# ---------------------------------------------------------------------------
# agent_outputs provenance
# ---------------------------------------------------------------------------


class TestAgentOutputsProvenance:
    """agent_outputs accumulates each agent's output dict across supersteps."""

    def test_agent_outputs_accumulate_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        tuples = _checkpoints_chronological(executor, "prov-001")

        # After the last agent, all three should be present
        final_cv = tuples[-1].checkpoint["channel_values"]
        agent_outputs = final_cv.get("agent_outputs", {})
        assert set(agent_outputs.keys()) == {"agent_0", "agent_1", "agent_2"}
        for aid in ("agent_0", "agent_1", "agent_2"):
            assert agent_outputs[aid].get("status") == "stub"

    def test_agent_outputs_grow_per_superstep(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        tuples = _checkpoints_chronological(executor, "prov-001")

        # Filter to checkpoints where current_agent is set (agent supersteps)
        agent_cps = [
            t for t in tuples if t.checkpoint["channel_values"].get("current_agent")
        ]
        assert len(agent_cps) == 3

        # agent_outputs grows by one entry each superstep
        for idx, tup in enumerate(agent_cps):
            cv = tup.checkpoint["channel_values"]
            assert len(cv.get("agent_outputs", {})) == idx + 1

    def test_executor_result_agent_results_populated(self):
        _, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        assert len(result.agent_results) == 3
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"
            assert ar.agent_id.startswith("agent_")


# ---------------------------------------------------------------------------
# Message name attribution provenance
# ---------------------------------------------------------------------------


class TestMessageNameProvenance:
    """AIMessages emitted by agent nodes carry name=<agent_id>."""

    def test_message_names_set_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        final_cv = _checkpoints_chronological(executor, "prov-001")[-1].checkpoint[
            "channel_values"
        ]
        messages = final_cv.get("messages", [])
        ai_names = [
            getattr(m, "name", None)
            for m in messages
            if hasattr(m, "name") and m.__class__.__name__ == "AIMessage"
        ]
        # One AIMessage per agent, each tagged with the agent's ID
        assert set(ai_names) == {"agent_0", "agent_1", "agent_2"}

    def test_serialized_messages_carry_name(self):
        _, result = _run(_seq_config(n_agents=2, with_channels=False))
        assert result.success
        msgs = result.final_state.get("messages", [])
        ai_msgs = [m for m in msgs if m.get("type") == "AIMessage"]
        assert len(ai_msgs) == 2
        names = {m.get("name") for m in ai_msgs}
        assert names == {"agent_0", "agent_1"}


# ---------------------------------------------------------------------------
# communication_log provenance — no explicit channels (Bug A regression)
# ---------------------------------------------------------------------------


class TestCommunicationLogNoChannels:
    """communication_log is always present even without explicit channels.

    Regression for Bug A: the channel was absent from the state schema when
    no channels were declared, so per-agent handoffs were never logged.
    """

    def test_communication_log_in_schema_no_channels(self):
        from bili.aether.compiler.state_generator import (  # pylint: disable=import-outside-toplevel
            generate_state_schema,
        )

        config = _seq_config(n_agents=2, with_channels=False)
        schema = generate_state_schema(config)
        annotations = schema.__annotations__

        assert "communication_log" in annotations
        # Routing auxiliaries are NOT present without explicit channels
        assert "channel_messages" not in annotations
        assert "pending_messages" not in annotations

    def test_communication_log_in_initial_state_no_channels(self):
        executor = MASExecutor(_seq_config(n_agents=2, with_channels=False))
        executor.initialize()
        state = executor._build_initial_state({})  # pylint: disable=protected-access
        assert "communication_log" in state
        assert state["communication_log"] == []

    def test_communication_log_checkpointed_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        final_cv = tuples[-1].checkpoint["channel_values"]

        # communication_log is present in the final checkpoint
        assert "communication_log" in final_cv
        comm_log = final_cv["communication_log"]
        assert isinstance(comm_log, list)
        # One entry per agent
        assert len(comm_log) == 3

    def test_communication_log_entries_per_superstep_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        agent_cps = [
            t for t in tuples if t.checkpoint["channel_values"].get("current_agent")
        ]
        assert len(agent_cps) == 3

        # Accumulated log grows by exactly one entry per superstep
        for idx, tup in enumerate(agent_cps):
            cv = tup.checkpoint["channel_values"]
            comm_log = cv.get("communication_log", [])
            assert len(comm_log) == idx + 1
            # The entry at this step is from the current agent
            assert comm_log[idx]["sender"] == cv["current_agent"]

    def test_communication_log_senders_cover_all_agents_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        final_cv = tuples[-1].checkpoint["channel_values"]
        senders = {e["sender"] for e in final_cv.get("communication_log", [])}
        assert senders == {"agent_0", "agent_1", "agent_2"}

    def test_communication_log_channel_is_agent_output(self):
        executor, result = _run(_seq_config(n_agents=2, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        final_cv = tuples[-1].checkpoint["channel_values"]
        channels = {e["channel"] for e in final_cv.get("communication_log", [])}
        assert channels == {"__agent_output__"}


# ---------------------------------------------------------------------------
# communication_log deduplication (Bug B regression)
# ---------------------------------------------------------------------------


class TestCommunicationLogNoDeduplication:
    """communication_log entries are not duplicated across supersteps.

    Regression for Bug B: send_message_in_state returned the full accumulated
    log as the state-update value.  Because the reducer is operator.add, each
    superstep doubled all prior entries.  A 3-agent run produced 1+3+7=7
    entries instead of 3.
    """

    def test_no_duplicate_entries_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        final_cv = tuples[-1].checkpoint["channel_values"]
        comm_log = final_cv.get("communication_log", [])

        # Exactly one entry per agent, no duplicates
        assert len(comm_log) == 3
        ids = [e.get("message_id") for e in comm_log]
        assert len(set(ids)) == 3, f"Duplicate IDs found: {ids}"

    def test_no_duplicate_entries_with_channels(self):
        executor, result = _run(
            _seq_config(n_agents=3, with_channels=True),
            thread_id="prov-ch-dedup",
        )
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-ch-dedup")
        final_cv = tuples[-1].checkpoint["channel_values"]
        comm_log = final_cv.get("communication_log", [])

        assert len(comm_log) == 3
        ids = [e.get("message_id") for e in comm_log]
        assert len(set(ids)) == 3, f"Duplicate IDs found: {ids}"

    def test_communication_log_growth_is_linear(self):
        # Each superstep adds exactly one entry; growth is N, not 2^N - 1.
        executor, result = _run(_seq_config(n_agents=4, with_channels=False))
        assert result.success

        tuples = _checkpoints_chronological(executor, "prov-001")
        agent_cps = [
            t for t in tuples if t.checkpoint["channel_values"].get("current_agent")
        ]
        lengths = [
            len(t.checkpoint["channel_values"].get("communication_log", []))
            for t in agent_cps
        ]
        assert lengths == [1, 2, 3, 4], f"Non-linear growth: {lengths}"

    def test_send_message_returns_delta_only(self):
        """send_message_in_state returns [msg_dict], not the full accumulated list."""
        from bili.aether.runtime.communication_state import (  # pylint: disable=import-outside-toplevel
            send_message_in_state,
        )
        from bili.aether.runtime.messages import (  # pylint: disable=import-outside-toplevel
            MessageType,
        )

        existing_log = [{"sender": "agent_0", "content": "prior output"}]
        state = {
            "communication_log": existing_log,
            "channel_messages": {},
            "pending_messages": {},
        }
        update = send_message_in_state(
            state=state,
            channel_id="__agent_output__",
            sender="agent_1",
            content="new output",
            message_type=MessageType.BROADCAST,
        )
        # The update must contain only the NEW entry, not the full accumulated log.
        comm_delta = update["communication_log"]
        assert (
            len(comm_delta) == 1
        ), f"Expected delta of 1, got {len(comm_delta)}: {comm_delta}"
        assert comm_delta[0]["sender"] == "agent_1"

        # Verify operator.add would produce the correct 2-entry accumulated log.
        accumulated = operator.add(existing_log, comm_delta)
        assert len(accumulated) == 2


# ---------------------------------------------------------------------------
# audit_view provenance
# ---------------------------------------------------------------------------


class TestAuditViewProvenance:
    """audit_view() builds a clean per-agent timeline from checkpoints."""

    def test_audit_view_one_entry_per_agent_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        # Exactly one entry per agent (internal LangGraph seed checkpoints filtered)
        assert len(timeline) == 3

    def test_audit_view_agent_ids_in_order_no_channels(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        agent_ids = [e["agent_id"] for e in timeline]
        assert agent_ids == ["agent_0", "agent_1", "agent_2"]

    def test_audit_view_output_summary_populated(self):
        executor, result = _run(_seq_config(n_agents=2, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        for entry in timeline:
            assert entry["output_summary"] is not None
            assert len(entry["output_summary"]) > 0

    def test_audit_view_messages_sent_per_agent(self):
        # Each agent's timeline entry has exactly one messages_sent entry.
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        for entry in timeline:
            assert len(entry["messages_sent"]) == 1
            msg = entry["messages_sent"][0]
            assert msg["sender"] == entry["agent_id"]

    def test_audit_view_with_channels(self):
        executor, result = _run(
            _seq_config(n_agents=3, with_channels=True),
            thread_id="prov-ch-audit",
        )
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-ch-audit"
        )  # pylint: disable=protected-access
        assert len(timeline) == 3
        agent_ids = [e["agent_id"] for e in timeline]
        assert agent_ids == ["agent_0", "agent_1", "agent_2"]

    def test_audit_view_no_initial_noise(self):
        # The initial LangGraph seed checkpoints (current_agent='' or None)
        # must not appear in the timeline.
        executor, result = _run(_seq_config(n_agents=2, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        for entry in timeline:
            assert entry["agent_id"] not in (None, "", "''")

    def test_audit_view_raw_agent_outputs_populated(self):
        executor, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success

        timeline = audit_view(
            executor._checkpointer, "prov-001"
        )  # pylint: disable=protected-access
        for entry in timeline:
            assert entry["raw_agent_outputs"]
            assert entry["agent_id"] in entry["raw_agent_outputs"]


# ---------------------------------------------------------------------------
# MASExecutionResult provenance statistics
# ---------------------------------------------------------------------------


class TestExecutionResultProvenanceStats:
    """MASExecutionResult communication stats reflect per-agent provenance."""

    def test_total_messages_equals_agent_count_no_channels(self):
        # Without explicit channels, each agent emits exactly one broadcast.
        _, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        assert result.total_messages == 3

    def test_messages_by_channel_no_channels(self):
        _, result = _run(_seq_config(n_agents=3, with_channels=False))
        assert result.success
        assert result.messages_by_channel == {"__agent_output__": 3}

    def test_total_messages_with_channels(self):
        _, result = _run(
            _seq_config(n_agents=3, with_channels=True),
            thread_id="prov-ch-stats",
        )
        assert result.success
        # With or without channels, each agent emits exactly one broadcast.
        assert result.total_messages == 3
