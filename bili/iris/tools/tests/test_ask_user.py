"""Unit tests for bili.iris.tools.ask_user registration and dispatch.

End-to-end pause/resume behavior through a REAL compiled graph is covered by
the integration tests (test_ask_user_iris_integration.py,
test_ask_user_aether_integration.py) and the real MCP-server integration test
(bili/iris/mcp/tests/test_ask_user_mcp_integration.py) -- this module covers
the registration lifecycle and the dispatcher's own unit-level behavior:
double registration, unregister when absent, the tool's shape
(name/description/schema), and which of the two pause-path implementations
gets called for a given calling context.
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel
# Test-scoped imports (langgraph internals, IN_MCP_TOOL_CALL) are deferred to
# inside individual test functions so each test's dependency is visible right
# where it is used, and so importing this module never requires the [mcp]
# extra just to collect the tests that do not exercise the MCP path.

import logging

import pytest

from bili.iris.loaders.tools_loader import TOOL_REGISTRY
from bili.iris.tools.ask_user import (
    ASK_USER_TOOL_NAME,
    register_ask_user_tool,
    unregister_ask_user_tool,
)
from bili.iris.tools.hitl import NO_RESPONSE_PREFIX, ScriptedHitlResponder


class TestRegisterAskUserTool:
    """Tests for register_ask_user_tool()."""

    def teardown_method(self):
        unregister_ask_user_tool()

    def test_registers_into_tool_registry(self):
        register_ask_user_tool()
        assert ASK_USER_TOOL_NAME in TOOL_REGISTRY

    def test_registered_tool_has_expected_name_and_description(self):
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        assert tool.name == ASK_USER_TOOL_NAME
        assert "human" in tool.description.lower()

    def test_registered_tool_accepts_question_and_optional_options(self):
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        schema_fields = tool.args_schema.model_fields
        assert "question" in schema_fields
        assert "options" in schema_fields
        assert schema_fields["question"].is_required()
        assert not schema_fields["options"].is_required()

    def test_tool_func_reaches_interrupt_inside_a_graph(self):
        """Confirms the dispatcher routes to the native (interrupt()) impl
        when called from inside a LangGraph runnable context, by observing
        the specific failure interrupt() itself raises when there is no
        resume value available yet: RuntimeError from
        langgraph.config.get_config() is what fires OUTSIDE a runnable
        context (the MCP-path branch, covered separately below); INSIDE one
        with no prior resume, interrupt() raises GraphInterrupt. The full
        pause/resume-with-an-answer path is exercised for real inside a
        compiled graph by the integration tests
        (test_ask_user_iris_integration.py, test_ask_user_aether_integration.py).
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.errors import GraphBubbleUp
        from langgraph.graph import END, START, StateGraph

        from bili.utils.langgraph_utils import State

        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        raised = {}

        def node(state):  # pylint: disable=unused-argument
            try:
                tool.func(question="Which environment?")
            except GraphBubbleUp as exc:
                raised["exc"] = exc
            return {"messages": []}

        graph = StateGraph(State)
        graph.add_node("n", node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        compiled = graph.compile(checkpointer=MemorySaver())
        compiled.invoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        assert "exc" in raised, "expected interrupt() to raise inside a graph node"

    def test_tool_func_calls_responder_via_mcp_bridge(self):
        """Confirms the dispatcher routes to the MCP (responder-calling)
        impl when IN_MCP_TOOL_CALL is set -- the exact signal
        bili.iris.mcp.server._build_mcp_fn sets around its own
        tool.invoke(...) call. This is a real, distinguishing signal;
        merely being "outside a graph" is NOT (a bare tool.invoke() call,
        which the MCP bridge also makes, already populates LangChain's
        ambient RunnableConfig, so langgraph.config.get_config() succeeds
        in both contexts and cannot be used to tell them apart -- see the
        ask_user module docstring's "Dispatch signal" section).
        """
        from bili.iris.mcp.server import IN_MCP_TOOL_CALL

        responder = ScriptedHitlResponder(["staging"])
        register_ask_user_tool(responder)
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        token = IN_MCP_TOOL_CALL.set(True)
        try:
            answer = tool.func(question="Which environment?")
        finally:
            IN_MCP_TOOL_CALL.reset(token)

        assert answer == "staging"
        assert responder.calls == [{"question": "Which environment?", "options": None}]

    def test_tool_func_via_mcp_bridge_with_no_responder_returns_sentinel(self):
        """Via the MCP bridge with no responder registered, the dispatcher's
        MCP-path branch reaches NullHitlResponder's no-response sentinel
        rather than raising -- an unconfigured CLI-path ask_user degrades
        gracefully instead of crashing the calling CLI subprocess's turn.
        """
        from bili.iris.mcp.server import IN_MCP_TOOL_CALL

        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        token = IN_MCP_TOOL_CALL.set(True)
        try:
            answer = tool.func(question="Which environment?")
        finally:
            IN_MCP_TOOL_CALL.reset(token)

        assert answer.startswith(NO_RESPONSE_PREFIX)

    def test_tool_func_outside_a_graph_and_outside_mcp_bridge_raises(self):
        """Called neither via a graph nor via the MCP bridge (e.g. a stray
        direct call), the dispatcher falls through to the native path, which
        calls langgraph.types.interrupt() and raises RuntimeError there --
        proof that this fallback genuinely reaches interrupt() rather than
        silently returning a placeholder. This is the correct default: an
        untracked calling context is not silently assumed to be the MCP
        bridge.
        """
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        with pytest.raises(RuntimeError, match="runnable context"):
            tool.func(question="Which environment?")

    def test_double_registration_replaces_and_warns(self, caplog):
        register_ask_user_tool()
        first_tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        with caplog.at_level(logging.WARNING):
            register_ask_user_tool()

        assert any("already registered" in record.message for record in caplog.records)
        second_tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        assert first_tool is not second_tool

    def test_accepts_explicit_responder_without_error(self):
        responder = ScriptedHitlResponder(["staging"])
        register_ask_user_tool(responder)
        assert ASK_USER_TOOL_NAME in TOOL_REGISTRY


class TestUnregisterAskUserTool:
    """Tests for unregister_ask_user_tool()."""

    def test_removes_from_tool_registry(self):
        register_ask_user_tool()
        unregister_ask_user_tool()
        assert ASK_USER_TOOL_NAME not in TOOL_REGISTRY

    def test_noop_when_not_registered(self):
        unregister_ask_user_tool()  # already absent; must not raise
        unregister_ask_user_tool()  # calling twice must also not raise
        assert ASK_USER_TOOL_NAME not in TOOL_REGISTRY
