"""Tests for per-agent provenance capture in the AETHER checkpointed state.

Verifies that a completed multi-agent run durably records, in every
checkpointed superstep, the three provenance channels required for full
post-run observability:

1. ``current_agent`` — which agent acted at each superstep.
2. ``agent_outputs[agent_id]`` — the output attributed to that agent,
   and the corresponding ``AIMessage`` tagged with ``name=<agent_id>``.
3. ``communication_log`` — one broadcast entry per agent handoff, present
   even when no explicit inter-agent channels are declared.

Root-cause regression tests are also included to pin the bugs that were fixed:

- Bug A: ``communication_log`` was absent from the state schema (and hence
  never checkpointed) for sequential workflows that declared no explicit
  channels.  Fix: always include ``communication_log`` in the schema.

- Bug B: ``send_message_in_state`` returned the full accumulated log as the
  state-update value.  Because the reducer is ``operator.add`` (list
  concatenation), this caused exponential duplication — a 3-agent run
  produced 7 entries (1+3+7 pattern) instead of 3.  Fix: return only the
  delta (single-element list) for ``communication_log``.

- Bug C: pipeline agents (``AgentSpec.pipeline`` set) compiled as inner
  sub-graphs did not emit a ``communication_log`` entry to the OUTER MAS
  state.  The inner sub-graph schema intentionally omits ``communication_log``
  (it is an outer MAS routing auxiliary), and the inner→outer output mapping
  in ``_wrap_pipeline_as_agent_node`` never called ``_build_communication_update``.
  Fix: call ``_build_pipeline_provenance`` at the outer-graph boundary after
  the sub-graph returns, just as plain agent nodes do.

All tests use stub agents (``model_name`` not set) and a MemorySaver
checkpointer so no LLM API calls or database servers are needed.
"""

# pylint: disable=missing-function-docstring

import operator

from langchain_core.messages import HumanMessage

from bili.aether.compiler import compile_mas
from bili.aether.runtime.audit import audit_view
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import (
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    WorkflowType,
)
from bili.aether.schema.pipeline_spec import (
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
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


# ---------------------------------------------------------------------------
# Pipeline-agent provenance via the compile_mas / GraphBuilder path
# ---------------------------------------------------------------------------


def _pipeline_spec() -> PipelineSpec:
    """A minimal two-node stub pipeline: step_a → step_b → END."""
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="step_a",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_a",
                    "role": "analyzer",
                    "objective": "First pipeline step for analysis work",
                },
            ),
            PipelineNodeSpec(
                node_id="step_b",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_b",
                    "role": "formatter",
                    "objective": "Format pipeline output for delivery",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="step_a", to_node="step_b"),
            PipelineEdgeSpec(from_node="step_b", to_node="END"),
        ],
    )


def _pipeline_config(mas_id: str = "pip_prov") -> MASConfig:
    """Sequential MAS with one plain agent and one pipeline agent."""
    return MASConfig(
        mas_id=mas_id,
        name="Pipeline Provenance Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="plain_agent",
                role="worker",
                objective="Run a plain agent task without a pipeline",
            ),
            AgentSpec(
                agent_id="pipe_agent",
                role="worker",
                objective="Run a pipeline agent task via inner sub-graph",
                pipeline=_pipeline_spec(),
            ),
        ],
        checkpoint_enabled=True,
        checkpoint_config={"type": "memory"},
    )


class TestPipelineAgentProvenance:
    """Per-agent provenance reaches the OUTER checkpointed state for pipeline agents.

    Regression for Bug C: ``_wrap_pipeline_as_agent_node`` compiled each agent as
    an inner sub-graph but never called ``_build_communication_update`` for the
    outer MAS state, so pipeline agents produced no ``communication_log`` entry
    in the OUTER (checkpointed) state even though plain agents did.

    These tests compile through the full ``compile_mas`` → ``GraphBuilder`` →
    ``_compile_pipeline_node`` → ``_wrap_pipeline_as_agent_node`` path to
    catch the exact boundary that the fix addresses.
    """

    def _run_pipeline_mas(self, thread_id: str = "pip-001"):
        """Run the mixed plain+pipeline MAS and return (checkpointer, result)."""
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
            MemorySaver,
        )

        config = _pipeline_config()
        compiled = compile_mas(config)
        checkpointer = MemorySaver()
        graph = compiled.compile_graph(checkpointer=checkpointer)

        result = graph.invoke(
            {"messages": [HumanMessage(content="start")]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return checkpointer, result

    def _outer_checkpoints(self, checkpointer, thread_id: str):
        """Return outer checkpoint channel_values dicts in chronological order."""
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        tuples = list(checkpointer.list(cfg))
        return [t.checkpoint["channel_values"] for t in reversed(tuples)]

    # ------------------------------------------------------------------
    # current_agent
    # ------------------------------------------------------------------

    def test_pipeline_agent_sets_current_agent_in_outer_checkpoint(self):
        """current_agent is set correctly in the outer checkpoint for a pipeline agent.

        This test exercises the compile_mas → GraphBuilder → _compile_pipeline_node
        path and asserts that provenance fields reach the OUTER (checkpointed) state,
        not just the inner sub-graph's transient state.
        """
        checkpointer, result = self._run_pipeline_mas()
        assert result.get("current_agent") == "pipe_agent"

        cvs = self._outer_checkpoints(checkpointer, "pip-001")
        agent_steps = [cv.get("current_agent") for cv in cvs if cv.get("current_agent")]
        assert "plain_agent" in agent_steps
        assert "pipe_agent" in agent_steps

    # ------------------------------------------------------------------
    # agent_outputs
    # ------------------------------------------------------------------

    def test_pipeline_agent_populates_agent_outputs_in_outer_checkpoint(self):
        checkpointer, result = self._run_pipeline_mas()
        agent_outputs = result.get("agent_outputs", {})

        # Both the plain agent and the pipeline agent must appear.
        assert "plain_agent" in agent_outputs
        assert "pipe_agent" in agent_outputs
        assert agent_outputs["pipe_agent"].get("status") == "completed"
        # pipeline_outputs records the inner sub-graph's agent_outputs dict.
        assert "pipeline_outputs" in agent_outputs["pipe_agent"]

    # ------------------------------------------------------------------
    # communication_log — Bug C regression
    # ------------------------------------------------------------------

    def test_pipeline_agent_emits_communication_log_entry(self):
        """Pipeline agent emits a communication_log entry to the OUTER state.

        Bug C regression: _wrap_pipeline_as_agent_node never called
        _build_communication_update, so pipeline agents produced no
        communication_log entry in the outer (checkpointed) state.
        """
        checkpointer, result = self._run_pipeline_mas()
        comm_log = result.get("communication_log", [])

        # One entry per outer agent (plain_agent + pipe_agent)
        assert len(comm_log) == 2
        senders = {e["sender"] for e in comm_log}
        assert senders == {"plain_agent", "pipe_agent"}

    def test_pipeline_agent_communication_log_entry_in_checkpoint(self):
        """The outer checkpoint carries pipe_agent's communication_log entry."""
        checkpointer, _ = self._run_pipeline_mas()
        cvs = self._outer_checkpoints(checkpointer, "pip-001")

        # Find the checkpoint where pipe_agent ran
        pipe_cvs = [cv for cv in cvs if cv.get("current_agent") == "pipe_agent"]
        assert pipe_cvs, "No checkpoint with current_agent='pipe_agent'"

        pipe_cv = pipe_cvs[-1]  # use the latest pipe_agent checkpoint
        log = pipe_cv.get("communication_log", [])
        senders = [e["sender"] for e in log]
        assert (
            "pipe_agent" in senders
        ), f"pipe_agent missing from communication_log. senders={senders}"

    def test_pipeline_agent_no_duplicate_log_entries(self):
        """Communication log entries are not duplicated for pipeline agents."""
        checkpointer, result = self._run_pipeline_mas()
        comm_log = result.get("communication_log", [])

        ids = [e.get("message_id") for e in comm_log]
        assert len(set(ids)) == len(ids), f"Duplicate IDs in log: {ids}"

    # ------------------------------------------------------------------
    # messages.name
    # ------------------------------------------------------------------

    def test_pipeline_agent_message_name_set(self):
        """AIMessage emitted by pipeline agent carries name=pipe_agent."""
        _, result = self._run_pipeline_mas()
        messages = result.get("messages", [])
        names = {getattr(m, "name", None) for m in messages if hasattr(m, "name")}
        assert "pipe_agent" in names, f"pipe_agent name missing from messages: {names}"

    # ------------------------------------------------------------------
    # audit_view end-to-end
    # ------------------------------------------------------------------

    def test_pipeline_agent_appears_in_audit_view(self):
        """audit_view returns one entry per agent, including the pipeline agent."""
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel
            MemorySaver,
        )

        config = _pipeline_config(mas_id="pip_audit")
        compiled = compile_mas(config)
        checkpointer = MemorySaver()
        graph = compiled.compile_graph(checkpointer=checkpointer)
        graph.invoke(
            {"messages": [HumanMessage(content="start")]},
            config={"configurable": {"thread_id": "pip-audit-001"}},
        )

        timeline = audit_view(checkpointer, "pip-audit-001")
        agent_ids = [e["agent_id"] for e in timeline]
        assert "plain_agent" in agent_ids
        assert "pipe_agent" in agent_ids
        # Both agents have messages_sent entries
        for entry in timeline:
            assert len(entry["messages_sent"]) >= 1
