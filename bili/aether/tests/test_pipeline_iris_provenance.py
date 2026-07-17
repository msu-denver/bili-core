"""Regression tests: pipeline-agent provenance when the inner graph mimics an IRIS
node that emits UNNAMED AIMessages and sets NEITHER current_agent NOR agent_outputs.

Real AETHER deployments commonly use agents whose ``pipeline`` field contains
an IRIS registry node (e.g. ``react_agent``, ``add_persona_and_summary``).
Those nodes have no knowledge of AETHER provenance — they emit raw
``AIMessage`` objects with ``name=None`` and do not write ``current_agent`` or
``agent_outputs`` to the inner state.

The bug path (fixed by #224 + #225):

- Bug A (#224): ``communication_log`` absent from the state schema for MAS
  configs without explicit channels, so pipeline-agent provenance was never
  checkpointed.
- Bug B (#224): ``send_message_in_state`` returned the full accumulated log,
  causing exponential duplication.
- Bug C (#225): ``_wrap_pipeline_as_agent_node`` never called
  ``_build_communication_update``, so pipeline agents emitted no
  ``communication_log`` entry to the outer MAS state.

The fix: ``_wrap_pipeline_as_agent_node`` now SYNTHESISES all provenance fields
(``current_agent``, ``agent_outputs``, named ``AIMessage``, ``communication_log``)
at the outer-graph boundary from the agent's closure variables, independent of
whether the inner pipeline set any of those fields.

These tests confirm the synthesized-at-boundary behaviour holds for an inner
graph that matches the IRIS react-node profile (unnamed messages, no provenance
state changes).  They are designed to fail on the pre-#224/#225 codebase and
pass on the fixed one.
"""

# pylint: disable=missing-function-docstring

from typing import Annotated, Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from typing_extensions import TypedDict

from bili.aether.compiler import compile_mas
from bili.aether.compiler.graph_builder import GraphBuilder
from bili.aether.compiler.state_generator import _merge_dicts, _replace_value
from bili.aether.runtime.audit import audit_view
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.pipeline_spec import (
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
)

# ---------------------------------------------------------------------------
# Inner graph factory — mimics an IRIS react node
# ---------------------------------------------------------------------------


class _IRISInnerState(TypedDict):
    """Minimal inner state, matching the AETHER pipeline inner schema."""

    messages: Annotated[list, add_messages]
    current_agent: Annotated[str, _replace_value]
    agent_outputs: Annotated[Dict[str, Any], _merge_dicts]


def _build_iris_like_inner_graph():
    """Build a compiled inner graph that behaves like an IRIS react node.

    The node:
    - Emits an ``AIMessage`` with **no name** (``name=None``).
    - Does **not** write ``current_agent`` or ``agent_outputs`` to the state.

    This is the exact profile of bili-core's IRIS react nodes, which have no
    knowledge of AETHER provenance.  The test confirms that the outer
    ``_wrap_pipeline_as_agent_node`` synthesises correct provenance regardless.
    """

    def _iris_like_node(state: dict) -> dict:
        # Emit an unnamed message only — no provenance fields.
        return {
            "messages": [
                AIMessage(content="IRIS inner response (no name, no provenance)")
            ]
        }

    g = StateGraph(_IRISInnerState)
    g.add_node("iris_like", _iris_like_node)
    g.add_edge(START, "iris_like")
    g.add_edge("iris_like", END)
    return g.compile(checkpointer=None)


# ---------------------------------------------------------------------------
# Helpers: build MAS config + inject the IRIS-like inner graph
# ---------------------------------------------------------------------------


def _pipeline_spec_placeholder() -> PipelineSpec:
    """A single-node pipeline spec used only as a marker (inner graph replaced)."""
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="step",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_placeholder",
                    "role": "inner_worker",
                    "objective": "Placeholder inner agent for injection",
                },
            )
        ],
        edges=[PipelineEdgeSpec(from_node="step", to_node="END")],
    )


def _make_config(mas_id: str = "iris_prov") -> MASConfig:
    """Sequential MAS with one plain agent + one pipeline agent."""
    return MASConfig(
        mas_id=mas_id,
        name="IRIS Pipeline Provenance Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="plain_agent",
                role="worker",
                objective="Run a plain agent task before the pipeline agent",
            ),
            AgentSpec(
                agent_id="pipe_agent",
                role="worker",
                objective="Run a pipeline agent backed by an IRIS-like inner graph",
                pipeline=_pipeline_spec_placeholder(),
            ),
        ],
        checkpoint_enabled=True,
        checkpoint_config={"type": "memory"},
    )


def _run_with_iris_inner(mas_id: str = "iris_prov", thread_id: str = "iris-001"):
    """Compile and run the MAS with the IRIS-like inner graph injected.

    Patches ``_compile_pipeline_node`` to substitute the placeholder pipeline
    with the IRIS-like compiled inner graph so that the inner graph truly
    emits unnamed messages with no provenance fields.
    """
    iris_inner = _build_iris_like_inner_graph()
    original_compile = GraphBuilder._compile_pipeline_node

    def patched_compile(self, agent):
        if agent.agent_id == "pipe_agent":
            return self._wrap_pipeline_as_agent_node(iris_inner, agent)
        return original_compile(self, agent)

    GraphBuilder._compile_pipeline_node = patched_compile
    try:
        config = _make_config(mas_id)
        compiled = compile_mas(config)
        checkpointer = MemorySaver()
        graph = compiled.compile_graph(checkpointer=checkpointer)
        result = graph.invoke(
            {"messages": [HumanMessage(content="start")]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return checkpointer, result
    finally:
        GraphBuilder._compile_pipeline_node = original_compile


def _outer_checkpoints(checkpointer, thread_id: str):
    """Return outer checkpoint channel_values dicts in chronological order."""
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    tuples = list(checkpointer.list(cfg))
    return [t.checkpoint["channel_values"] for t in reversed(tuples)]


# ---------------------------------------------------------------------------
# Confirm the inner graph is truly IRIS-like (no provenance)
# ---------------------------------------------------------------------------


class TestIRISLikeInnerGraphProperties:
    """Confirm the inner graph produces unnamed messages and no provenance."""

    def test_inner_messages_are_unnamed(self):
        iris_inner = _build_iris_like_inner_graph()
        inner_result = iris_inner.invoke(
            {
                "messages": [HumanMessage(content="start")],
                "current_agent": "",
                "agent_outputs": {},
            }
        )
        ai_msgs = [
            m for m in inner_result["messages"] if m.__class__.__name__ == "AIMessage"
        ]
        assert ai_msgs, "Inner graph must emit at least one AIMessage"
        assert all(
            m.name is None for m in ai_msgs
        ), f"Inner AIMessages must be unnamed. Got names: {[m.name for m in ai_msgs]}"

    def test_inner_graph_does_not_set_agent_outputs(self):
        iris_inner = _build_iris_like_inner_graph()
        inner_result = iris_inner.invoke(
            {
                "messages": [HumanMessage(content="start")],
                "current_agent": "",
                "agent_outputs": {},
            }
        )
        # agent_outputs must remain as-initialised ({}); the node does not write it.
        assert (
            inner_result.get("agent_outputs") == {}
        ), "Inner graph must not set agent_outputs"


# ---------------------------------------------------------------------------
# Provenance synthesis at the outer-graph boundary
# ---------------------------------------------------------------------------


class TestOuterProvenanceSynthesisFromIRISInner:
    """Outer MAS state carries correct provenance even when inner graph is IRIS-like.

    These tests are the canonical regression for the fix introduced in
    #224 + #225: the outer boundary synthesises provenance from the agent_id
    closure, not from what the inner graph wrote.
    """

    def test_current_agent_set_in_outer_final_state(self):
        _, result = _run_with_iris_inner()
        assert (
            result.get("current_agent") == "pipe_agent"
        ), f"Expected current_agent='pipe_agent', got {result.get('current_agent')!r}"

    def test_current_agent_set_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_iris_inner()
        cvs = _outer_checkpoints(checkpointer, "iris-001")
        pipe_cvs = [cv for cv in cvs if cv.get("current_agent") == "pipe_agent"]
        assert pipe_cvs, "No outer checkpoint with current_agent='pipe_agent'"

    def test_agent_outputs_keyed_by_pipe_agent_in_final_state(self):
        _, result = _run_with_iris_inner()
        agent_outputs = result.get("agent_outputs", {})
        assert (
            "pipe_agent" in agent_outputs
        ), f"pipe_agent missing from agent_outputs. keys={list(agent_outputs.keys())}"
        assert agent_outputs["pipe_agent"].get("status") == "completed"

    def test_agent_outputs_keyed_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_iris_inner()
        cvs = _outer_checkpoints(checkpointer, "iris-001")
        final_cv = cvs[-1]
        assert "pipe_agent" in final_cv.get(
            "agent_outputs", {}
        ), "pipe_agent missing from agent_outputs in outer checkpoint"

    def test_message_name_set_to_pipe_agent(self):
        """AIMessage in the outer state carries name=pipe_agent.

        Before the fix, the inner IRIS-like graph's unnamed message was the
        only message forwarded, producing name=None in the outer state.  The
        outer wrapper now emits a fresh AIMessage(name=agent_id) regardless of
        what the inner graph produced.
        """
        _, result = _run_with_iris_inner()
        ai_msgs = [
            m for m in result.get("messages", []) if m.__class__.__name__ == "AIMessage"
        ]
        pipe_msgs = [m for m in ai_msgs if m.name == "pipe_agent"]
        assert pipe_msgs, (
            f"No AIMessage with name='pipe_agent'. "
            f"AI msg names: {[m.name for m in ai_msgs]}"
        )

    def test_message_name_set_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_iris_inner()
        cvs = _outer_checkpoints(checkpointer, "iris-001")
        final_cv = cvs[-1]
        ai_msgs = [
            m
            for m in final_cv.get("messages", [])
            if m.__class__.__name__ == "AIMessage"
        ]
        named_pipe = [m for m in ai_msgs if m.name == "pipe_agent"]
        assert named_pipe, (
            f"No outer-checkpoint AIMessage with name='pipe_agent'. "
            f"names={[m.name for m in ai_msgs]}"
        )

    def test_communication_log_has_pipe_agent_entry(self):
        """communication_log carries a broadcast entry for the pipeline agent.

        Regression for Bug C (#225): _wrap_pipeline_as_agent_node never called
        _build_communication_update, so pipeline agents produced no
        communication_log entry.
        """
        _, result = _run_with_iris_inner()
        comm_log = result.get("communication_log", [])
        assert comm_log, "communication_log must not be empty"
        senders = {e["sender"] for e in comm_log}
        assert (
            "pipe_agent" in senders
        ), f"pipe_agent missing from communication_log senders. senders={senders}"

    def test_communication_log_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_iris_inner()
        cvs = _outer_checkpoints(checkpointer, "iris-001")
        pipe_cvs = [cv for cv in cvs if cv.get("current_agent") == "pipe_agent"]
        assert pipe_cvs, "No outer checkpoint for pipe_agent"
        pipe_cv = pipe_cvs[-1]
        log = pipe_cv.get("communication_log", [])
        assert any(e["sender"] == "pipe_agent" for e in log), (
            f"pipe_agent missing from communication_log in outer checkpoint. "
            f"senders={[e['sender'] for e in log]}"
        )

    def test_no_duplicate_communication_log_entries(self):
        _, result = _run_with_iris_inner()
        comm_log = result.get("communication_log", [])
        # 2 agents = exactly 2 entries
        assert (
            len(comm_log) == 2
        ), f"Expected 2 comm_log entries (one per agent), got {len(comm_log)}"
        ids = [e.get("message_id") for e in comm_log]
        assert len(set(ids)) == 2, f"Duplicate message IDs: {ids}"

    def test_both_agents_in_final_state(self):
        _, result = _run_with_iris_inner()
        agent_outputs = result.get("agent_outputs", {})
        assert "plain_agent" in agent_outputs
        assert "pipe_agent" in agent_outputs
        assert result.get("current_agent") == "pipe_agent"

    def test_audit_view_includes_pipe_agent(self):
        checkpointer, _ = _run_with_iris_inner(
            mas_id="iris_audit", thread_id="iris-audit-001"
        )
        timeline = audit_view(checkpointer, "iris-audit-001")
        agent_ids = [e["agent_id"] for e in timeline]
        assert (
            "pipe_agent" in agent_ids
        ), f"pipe_agent missing from audit timeline. agent_ids={agent_ids}"
        # Each agent has exactly one provenance entry
        pipe_entries = [e for e in timeline if e["agent_id"] == "pipe_agent"]
        assert len(pipe_entries) == 1
        assert len(pipe_entries[0]["messages_sent"]) == 1
        assert pipe_entries[0]["messages_sent"][0]["sender"] == "pipe_agent"

    def test_inner_content_reaches_outer_summary(self):
        """The outer agent_outputs message field contains the inner pipeline's content."""
        _, result = _run_with_iris_inner()
        pipe_output = result.get("agent_outputs", {}).get("pipe_agent", {})
        assert "IRIS inner response" in pipe_output.get("message", ""), (
            f"Expected inner content in outer agent_output.message. "
            f"Got: {pipe_output.get('message')!r}"
        )


# ---------------------------------------------------------------------------
# Direct wrapper unit test (no compile_mas, pinpoints the synthesis boundary)
# ---------------------------------------------------------------------------


class TestWrapperSynthesisDirectly:
    """Call _wrap_pipeline_as_agent_node directly to confirm synthesis happens
    at the wrapper boundary, not inside the inner graph."""

    def test_wrapper_synthesises_current_agent_from_closure(self):
        iris_inner = _build_iris_like_inner_graph()
        agent = AgentSpec(
            agent_id="direct_pipe",
            role="worker",
            objective="Direct wrapper test agent",
            pipeline=_pipeline_spec_placeholder(),
        )
        config = MASConfig(
            mas_id="direct_test",
            name="Direct Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        builder = GraphBuilder(config)
        wrapper = builder._wrap_pipeline_as_agent_node(iris_inner, agent)

        outer_state = {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {},
            "communication_log": [],
        }
        result = wrapper(outer_state)

        assert result["current_agent"] == "direct_pipe"
        assert "direct_pipe" in result["agent_outputs"]
        assert result["agent_outputs"]["direct_pipe"]["status"] == "completed"

    def test_wrapper_emits_named_ai_message(self):
        iris_inner = _build_iris_like_inner_graph()
        agent = AgentSpec(
            agent_id="direct_pipe",
            role="worker",
            objective="Direct wrapper test agent",
            pipeline=_pipeline_spec_placeholder(),
        )
        config = MASConfig(
            mas_id="direct_test2",
            name="Direct Test 2",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        builder = GraphBuilder(config)
        wrapper = builder._wrap_pipeline_as_agent_node(iris_inner, agent)

        outer_state = {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {},
            "communication_log": [],
        }
        result = wrapper(outer_state)

        ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
        assert ai_msgs, "Wrapper must emit at least one AIMessage"
        assert all(m.name == "direct_pipe" for m in ai_msgs), (
            f"All emitted AIMessages must have name='direct_pipe'. "
            f"Got: {[m.name for m in ai_msgs]}"
        )

    def test_wrapper_emits_communication_log_delta(self):
        iris_inner = _build_iris_like_inner_graph()
        agent = AgentSpec(
            agent_id="direct_pipe",
            role="worker",
            objective="Direct wrapper test agent",
            pipeline=_pipeline_spec_placeholder(),
        )
        config = MASConfig(
            mas_id="direct_test3",
            name="Direct Test 3",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        builder = GraphBuilder(config)
        wrapper = builder._wrap_pipeline_as_agent_node(iris_inner, agent)

        outer_state = {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {},
            "communication_log": [],
        }
        result = wrapper(outer_state)

        comm_log_delta = result.get("communication_log", [])
        assert (
            len(comm_log_delta) == 1
        ), f"Wrapper must return single-entry delta. Got {len(comm_log_delta)}"
        assert comm_log_delta[0]["sender"] == "direct_pipe"

    def test_wrapper_uses_inner_content_for_output_summary(self):
        """Outer agent_output.message captures the inner pipeline's final content."""
        iris_inner = _build_iris_like_inner_graph()
        agent = AgentSpec(
            agent_id="direct_pipe",
            role="worker",
            objective="Direct wrapper test agent",
            pipeline=_pipeline_spec_placeholder(),
        )
        config = MASConfig(
            mas_id="direct_test4",
            name="Direct Test 4",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        builder = GraphBuilder(config)
        wrapper = builder._wrap_pipeline_as_agent_node(iris_inner, agent)

        outer_state = {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {},
            "communication_log": [],
        }
        result = wrapper(outer_state)

        pipe_out = result["agent_outputs"]["direct_pipe"]
        # Content extracted from the inner graph's unnamed AIMessage
        assert "IRIS inner response" in pipe_out.get(
            "message", ""
        ), f"Expected inner content in summary. Got: {pipe_out.get('message')!r}"
