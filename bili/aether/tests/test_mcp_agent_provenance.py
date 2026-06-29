"""Regression tests: MCP/CLI agent provenance synthesis at the AETHER boundary.

Real MCP-path agents (``tool_strategy="mcp"``) delegate execution to an IRIS
MCP node built by ``bili.iris.mcp.server.build_mcp_node``.  That node is
intentionally generic: it returns only ``{"messages": [AIMessage(content=...)]}``
with no ``name``, no ``current_agent``, no ``agent_outputs``, and no
``communication_log``.

Before this fix, ``_generate_tool_agent_node`` returned ``build_mcp_node(...)``
raw, leaving outer MAS checkpoints with zero provenance: unnamed messages, empty
``current_agent`` / ``agent_outputs`` / ``communication_log``.  Audit views over
those checkpoints could not attribute any superstep to an agent.

The fix (``_wrap_mcp_node_with_provenance`` in
``bili/aether/compiler/agent_generator.py``): wrap the MCP node at the AETHER
boundary, synthesising the same provenance that the native ``_agent_node``
closure and ``_wrap_pipeline_as_agent_node`` produce:

- ``AIMessage(content=<inner content>, name=agent_id)``
- ``current_agent = agent_id``
- ``agent_outputs[agent_id] = <output dict>``
- ``communication_log`` delta with one broadcast entry

The MCP node in ``bili.iris.mcp.server`` remains unmodified — the synthesis
happens entirely at the AETHER boundary, in the wrapper returned by
``_generate_tool_agent_node``.

These tests confirm the synthesis behaviour and are designed to fail against the
pre-fix codebase (where ``build_mcp_node(...)`` was returned raw) and pass after.
"""

# pylint: disable=missing-function-docstring

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from bili.aether.compiler import compile_mas
from bili.aether.compiler.agent_generator import _wrap_mcp_node_with_provenance
from bili.aether.runtime.audit import audit_view
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType

# ---------------------------------------------------------------------------
# Fake MCP inner node factory — mimics what build_mcp_node returns
# ---------------------------------------------------------------------------

_MCP_INNER_CONTENT = "mcp cli result (no name, no provenance)"


def _build_fake_mcp_inner():
    """Return a callable that emits only an unnamed AIMessage.

    Matches the exact return signature of
    ``bili.iris.mcp.server.build_mcp_node._node``:
    ``{"messages": [AIMessage(content=content)]}`` with ``name=None``.
    """

    def _fake_mcp_node(state: dict) -> dict:  # pylint: disable=unused-argument
        return {"messages": [AIMessage(content=_MCP_INNER_CONTENT)]}

    return _fake_mcp_node


# ---------------------------------------------------------------------------
# MAS config + patched compile_mas runner
# ---------------------------------------------------------------------------


def _make_config(mas_id: str = "mcp_prov") -> MASConfig:
    """Sequential MAS: one plain (stub) agent followed by one MCP-strategy agent."""
    return MASConfig(
        mas_id=mas_id,
        name="MCP Provenance Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="plain_agent",
                role="worker",
                objective="Run a plain agent task before the MCP agent",
            ),
            AgentSpec(
                agent_id="mcp_agent",
                role="worker",
                objective="Run via MCP tool-calling strategy",
                model_name="cli_claude_code",
                tools=["mock_tool"],
            ),
        ],
        checkpoint_enabled=True,
        checkpoint_config={"type": "memory"},
    )


def _run_with_fake_mcp(mas_id: str = "mcp_prov", thread_id: str = "mcp-001"):
    """Compile and run with LLM/MCP internals patched to use fake responses.

    Patches five call sites so no real CLI, LLM, or tool resolution occurs:

    - ``create_llm``            — returns a MagicMock (fake CliLLM)
    - ``resolve_tools``         — returns a non-empty list of MagicMock tools
    - ``resolve_tool_strategy`` — returns ``"mcp"`` (forces the MCP branch)
    - ``resolve_mcp_injector``  — returns a MagicMock (non-None triggers MCP path)
    - ``build_mcp_node``        — returns the fake inner node callable

    After patching, ``_generate_tool_agent_node`` takes the MCP branch:

        mcp_node = build_mcp_node(...)   # returns _build_fake_mcp_inner()
        return _wrap_mcp_node_with_provenance(mcp_node, agent)

    The graph is then compiled and invoked so that the outer checkpoint state
    can be inspected for provenance fields.
    """
    fake_llm = MagicMock()
    fake_llm.command = ["claude"]
    fake_tool = MagicMock()
    fake_tool.name = "mock_tool"
    fake_injector = MagicMock()

    with (
        patch(
            "bili.aether.compiler.llm_resolver.create_llm",
            return_value=fake_llm,
        ),
        patch(
            "bili.aether.compiler.llm_resolver.resolve_tools",
            return_value=[fake_tool],
        ),
        patch(
            "bili.aether.compiler.llm_resolver.resolve_tool_strategy",
            return_value="mcp",
        ),
        patch(
            "bili.iris.mcp.server.resolve_mcp_injector",
            return_value=fake_injector,
        ),
        patch(
            "bili.iris.mcp.server.build_mcp_node",
            return_value=_build_fake_mcp_inner(),
        ),
    ):
        config = _make_config(mas_id)
        compiled = compile_mas(config)
        checkpointer = MemorySaver()
        graph = compiled.compile_graph(checkpointer=checkpointer)
        result = graph.invoke(
            {"messages": [HumanMessage(content="start")]},
            config={"configurable": {"thread_id": thread_id}},
        )

    return checkpointer, result


def _outer_checkpoints(checkpointer, thread_id: str):
    """Return outer checkpoint channel_values dicts in chronological order."""
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    tuples = list(checkpointer.list(cfg))
    return [t.checkpoint["channel_values"] for t in reversed(tuples)]


# ---------------------------------------------------------------------------
# Confirm the fake MCP inner node matches the real build_mcp_node profile
# ---------------------------------------------------------------------------


class TestFakeMcpInnerNodeProperties:
    """The fake MCP inner node is truly provenance-free, matching the real node."""

    def test_inner_node_emits_unnamed_message(self):
        node = _build_fake_mcp_inner()
        result = node({"messages": []})
        ai_msgs = [
            m for m in result.get("messages", []) if m.__class__.__name__ == "AIMessage"
        ]
        assert ai_msgs, "Fake MCP node must emit at least one AIMessage"
        assert all(m.name is None for m in ai_msgs), (
            f"Fake MCP node messages must be unnamed. "
            f"names={[m.name for m in ai_msgs]}"
        )

    def test_inner_node_returns_no_current_agent(self):
        node = _build_fake_mcp_inner()
        result = node({"messages": []})
        assert "current_agent" not in result

    def test_inner_node_returns_no_agent_outputs(self):
        node = _build_fake_mcp_inner()
        result = node({"messages": []})
        assert "agent_outputs" not in result

    def test_inner_node_returns_no_communication_log(self):
        node = _build_fake_mcp_inner()
        result = node({"messages": []})
        assert "communication_log" not in result


# ---------------------------------------------------------------------------
# Provenance synthesis at the outer MAS boundary (compile_mas integration)
# ---------------------------------------------------------------------------


class TestMcpAgentProvenanceThroughCompileMas:
    """Outer MAS checkpoints carry correct provenance for MCP-strategy agents.

    These are the canonical regression tests for the MCP provenance fix: they
    compile a real MAS through ``compile_mas``, run it with a MemorySaver
    checkpointer, and inspect the checkpointed state.

    They are designed to fail against the pre-fix codebase (where
    ``build_mcp_node(...)`` was returned raw without a provenance wrapper) and
    pass after the fix.
    """

    def test_current_agent_set_in_final_state(self):
        _, result = _run_with_fake_mcp()
        assert (
            result.get("current_agent") == "mcp_agent"
        ), f"Expected current_agent='mcp_agent', got {result.get('current_agent')!r}"

    def test_current_agent_set_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_fake_mcp(thread_id="mcp-ckpt-001")
        cvs = _outer_checkpoints(checkpointer, "mcp-ckpt-001")
        mcp_cvs = [cv for cv in cvs if cv.get("current_agent") == "mcp_agent"]
        assert mcp_cvs, "No outer checkpoint with current_agent='mcp_agent'"

    def test_agent_outputs_keyed_by_mcp_agent_in_final_state(self):
        _, result = _run_with_fake_mcp(thread_id="mcp-out-001")
        agent_outputs = result.get("agent_outputs", {})
        assert (
            "mcp_agent" in agent_outputs
        ), f"mcp_agent missing from agent_outputs. keys={list(agent_outputs.keys())}"
        assert agent_outputs["mcp_agent"].get("status") == "completed"

    def test_agent_outputs_keyed_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_fake_mcp(thread_id="mcp-out-ckpt-001")
        cvs = _outer_checkpoints(checkpointer, "mcp-out-ckpt-001")
        final_cv = cvs[-1]
        assert "mcp_agent" in final_cv.get(
            "agent_outputs", {}
        ), "mcp_agent missing from agent_outputs in final outer checkpoint"

    def test_message_name_set_to_mcp_agent(self):
        """AIMessage in the outer state carries name=mcp_agent.

        Before the fix, ``build_mcp_node`` was returned raw — its inner node
        emits ``AIMessage(name=None)``, which propagated unchanged to the outer
        MAS state, making it impossible to attribute the message to an agent.
        """
        _, result = _run_with_fake_mcp(thread_id="mcp-name-001")
        ai_msgs = [
            m for m in result.get("messages", []) if m.__class__.__name__ == "AIMessage"
        ]
        named = [m for m in ai_msgs if m.name == "mcp_agent"]
        assert named, (
            f"No AIMessage with name='mcp_agent'. "
            f"AI msg names: {[m.name for m in ai_msgs]}"
        )

    def test_message_name_set_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_fake_mcp(thread_id="mcp-name-ckpt-001")
        cvs = _outer_checkpoints(checkpointer, "mcp-name-ckpt-001")
        final_cv = cvs[-1]
        ai_msgs = [
            m
            for m in final_cv.get("messages", [])
            if m.__class__.__name__ == "AIMessage"
        ]
        named = [m for m in ai_msgs if m.name == "mcp_agent"]
        assert named, (
            f"No outer-checkpoint AIMessage with name='mcp_agent'. "
            f"names={[m.name for m in ai_msgs]}"
        )

    def test_communication_log_has_mcp_agent_entry(self):
        """communication_log carries a broadcast entry for the MCP agent.

        Before the fix, ``build_mcp_node`` was returned raw — its inner node
        produces no ``communication_log`` entry, so the outer MAS state had no
        provenance trace of the MCP agent's execution.
        """
        _, result = _run_with_fake_mcp(thread_id="mcp-log-001")
        comm_log = result.get("communication_log", [])
        assert comm_log, "communication_log must not be empty"
        senders = {e["sender"] for e in comm_log}
        assert (
            "mcp_agent" in senders
        ), f"mcp_agent missing from communication_log senders. senders={senders}"

    def test_communication_log_in_outer_checkpoint(self):
        checkpointer, _ = _run_with_fake_mcp(thread_id="mcp-log-ckpt-001")
        cvs = _outer_checkpoints(checkpointer, "mcp-log-ckpt-001")
        mcp_cvs = [cv for cv in cvs if cv.get("current_agent") == "mcp_agent"]
        assert mcp_cvs, "No outer checkpoint for mcp_agent"
        mcp_cv = mcp_cvs[-1]
        log = mcp_cv.get("communication_log", [])
        assert any(e["sender"] == "mcp_agent" for e in log), (
            f"mcp_agent missing from communication_log in outer checkpoint. "
            f"senders={[e['sender'] for e in log]}"
        )

    def test_no_duplicate_communication_log_entries(self):
        _, result = _run_with_fake_mcp(thread_id="mcp-dedup-001")
        comm_log = result.get("communication_log", [])
        # 2 agents = exactly 2 entries (plain stub + mcp)
        assert (
            len(comm_log) == 2
        ), f"Expected 2 comm_log entries (one per agent), got {len(comm_log)}"
        ids = [e.get("message_id") for e in comm_log]
        assert len(set(ids)) == 2, f"Duplicate message IDs in comm_log: {ids}"

    def test_both_agents_in_final_state(self):
        _, result = _run_with_fake_mcp(thread_id="mcp-both-001")
        agent_outputs = result.get("agent_outputs", {})
        assert "plain_agent" in agent_outputs
        assert "mcp_agent" in agent_outputs
        assert result.get("current_agent") == "mcp_agent"

    def test_inner_content_reaches_outer_summary(self):
        """The outer agent_outputs.message field contains the MCP node's content."""
        _, result = _run_with_fake_mcp(thread_id="mcp-content-001")
        mcp_output = result.get("agent_outputs", {}).get("mcp_agent", {})
        assert _MCP_INNER_CONTENT in mcp_output.get("message", ""), (
            f"Expected MCP content in outer agent_output.message. "
            f"Got: {mcp_output.get('message')!r}"
        )

    def test_audit_view_includes_mcp_agent(self):
        checkpointer, _ = _run_with_fake_mcp(
            mas_id="mcp_audit", thread_id="mcp-audit-001"
        )
        timeline = audit_view(checkpointer, "mcp-audit-001")
        agent_ids = [e["agent_id"] for e in timeline]
        assert (
            "mcp_agent" in agent_ids
        ), f"mcp_agent missing from audit timeline. agent_ids={agent_ids}"
        mcp_entries = [e for e in timeline if e["agent_id"] == "mcp_agent"]
        assert len(mcp_entries) == 1
        assert len(mcp_entries[0]["messages_sent"]) == 1
        assert mcp_entries[0]["messages_sent"][0]["sender"] == "mcp_agent"


# ---------------------------------------------------------------------------
# Direct wrapper unit tests (_wrap_mcp_node_with_provenance)
# ---------------------------------------------------------------------------


class TestWrapMcpNodeDirectly:
    """Call ``_wrap_mcp_node_with_provenance`` directly to pin synthesis at the boundary.

    These tests bypass ``compile_mas`` and ``_generate_tool_agent_node`` to
    verify the wrapper function in isolation.  They confirm the synthesis
    happens at the wrapper boundary, not inside the MCP inner node.
    """

    def _make_agent(self, agent_id: str = "direct_mcp") -> AgentSpec:
        return AgentSpec(
            agent_id=agent_id,
            role="worker",
            objective="Direct MCP wrapper test agent",
        )

    def _base_state(self) -> dict:
        return {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {},
            "communication_log": [],
        }

    def test_wrapper_synthesises_current_agent_from_closure(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        result = wrapper(self._base_state())
        assert result["current_agent"] == "direct_mcp"

    def test_wrapper_sets_agent_outputs_entry(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        result = wrapper(self._base_state())
        assert "direct_mcp" in result["agent_outputs"]
        assert result["agent_outputs"]["direct_mcp"]["status"] == "completed"

    def test_wrapper_emits_named_ai_message(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        result = wrapper(self._base_state())
        ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
        assert ai_msgs, "Wrapper must emit at least one AIMessage"
        assert all(m.name == "direct_mcp" for m in ai_msgs), (
            f"All AIMessages must have name='direct_mcp'. "
            f"Got: {[m.name for m in ai_msgs]}"
        )

    def test_wrapper_emits_communication_log_delta(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        result = wrapper(self._base_state())
        delta = result.get("communication_log", [])
        assert (
            len(delta) == 1
        ), f"Wrapper must return single-entry delta for operator.add. Got {len(delta)}"
        assert delta[0]["sender"] == "direct_mcp"

    def test_wrapper_uses_inner_content_for_output_summary(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        result = wrapper(self._base_state())
        output_msg = result["agent_outputs"]["direct_mcp"].get("message", "")
        assert (
            _MCP_INNER_CONTENT in output_msg
        ), f"Expected MCP inner content in summary. Got: {output_msg!r}"

    def test_wrapper_preserves_existing_agent_outputs(self):
        """Existing agent_outputs entries from prior agents are not clobbered."""
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        state = {
            "messages": [HumanMessage(content="start")],
            "agent_outputs": {
                "prior_agent": {"status": "completed", "message": "prior result"}
            },
            "communication_log": [],
        }
        result = wrapper(state)
        assert "prior_agent" in result["agent_outputs"]
        assert "direct_mcp" in result["agent_outputs"]

    def test_wrapper_handles_empty_inner_messages(self):
        """Wrapper degrades gracefully when inner node returns no messages."""

        def _empty_inner(state: dict) -> dict:  # pylint: disable=unused-argument
            return {"messages": []}

        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_empty_inner, agent)
        result = wrapper(self._base_state())
        # Provenance is still synthesised from the closure
        assert result["current_agent"] == "direct_mcp"
        assert "direct_mcp" in result["agent_outputs"]
        # Content is empty string (no message to extract)
        ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
        assert ai_msgs[0].name == "direct_mcp"
        assert ai_msgs[0].content == ""

    def test_wrapper_function_name_set_for_introspection(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        assert wrapper.__name__ == "agent_direct_mcp"
        assert wrapper.__qualname__ == "agent_direct_mcp"

    def test_wrapper_agent_spec_attribute_set(self):
        agent = self._make_agent()
        wrapper = _wrap_mcp_node_with_provenance(_build_fake_mcp_inner(), agent)
        assert hasattr(wrapper, "agent_spec")
        assert wrapper.agent_spec is agent
