"""End-to-end integration test: ask_user through IRIS's native tool-calling path.

Builds a REAL compiled IRIS agent graph (``build_agent_graph``) with a fake
tool-calling chat model scripted to call ``ask_user``, drives it through
``langgraph.types.interrupt()`` pause and ``Command(resume=...)`` resume, and
asserts the second half of the turn genuinely receives the injected answer.

This is the IRIS half of the chunk-1 gating deliverable: prove the seam
end-to-end with one graph layer (outer StateGraph -> react_agent node ->
inner create_agent subgraph) before adding AETHER's extra provenance layer
on top (see test_ask_user_aether_integration.py).
"""

# pylint: disable=duplicate-code
# _ScriptedToolCallingModel is intentionally re-declared (not shared via
# import) in bili/aether/tests/test_ask_user_aether_integration.py: the two
# tests live in different packages (IRIS vs AETHER) with no existing
# cross-package test-helper module, and each file's fake model is a small,
# self-contained fixture scoped to that file's own test -- mirroring this
# codebase's existing precedent for pylint duplicate-code suppressions on
# intentionally-parallel test fixtures (e.g. bili/aegis/tests/conftest.py).

from typing import Any, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from bili.iris.graph_builder.classes.node import Node
from bili.iris.loaders.langchain_loader import build_agent_graph, react_agent_node
from bili.iris.tools.ask_user import (
    ASK_USER_TOOL_NAME,
    register_ask_user_tool,
    unregister_ask_user_tool,
)


def _single_node_graph_definition() -> List[Node]:
    """A minimal graph_definition with just react_agent as entry + exit.

    Isolates the interrupt/resume seam from the default pipeline's other
    nodes (persona, datetime, memory, normalization), which have their own
    required kwargs unrelated to what this test exercises.
    """
    node = react_agent_node()
    node.is_entry = True
    node.routes_to_end = True
    return [node]


class _ScriptedToolCallingModel(BaseChatModel):
    """Fake chat model that supports ``bind_tools`` and cycles scripted responses.

    LangChain's stock fakes (``FakeListChatModel``, ``FakeMessagesListChatModel``)
    do not override ``bind_tools``, so ``create_agent`` would reject them with
    ``NotImplementedError``. This minimal fake exists to unblock testing the
    native tool-calling path without a real LLM.
    """

    responses: List[BaseMessage] = []
    #: Index of the next scripted response to serve (cycles, does not advance
    #: past the last entry). Distinct from invocation_count below, which
    #: counts real _generate() calls -- the two would collapse into a single,
    #: ambiguous counter if conflated, exactly the kind of measurement
    #: confusion that must not leak into a test asserting exact call counts.
    next_response_index: int = 0
    invocation_count: int = 0

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "BaseChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.invocation_count += 1
        response = self.responses[self.next_response_index]
        if self.next_response_index < len(self.responses) - 1:
            self.next_response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"


def _scripted_ask_user_model() -> _ScriptedToolCallingModel:
    """Model scripted to call ask_user, then produce a final answer."""
    return _ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=ASK_USER_TOOL_NAME,
                        args={"question": "Which environment should I deploy to?"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content="Deploying to the environment you specified."),
        ]
    )


class TestAskUserIrisIntegration:
    """Proves the ask_user pause/resume seam through a real IRIS graph."""

    def setup_method(self):
        """Register ask_user before each test."""
        register_ask_user_tool()

    def teardown_method(self):
        """Unregister ask_user after each test."""
        unregister_ask_user_tool()

    def test_pause_and_resume_through_real_graph(self):
        """A real IRIS graph pauses at ask_user and resumes with the answer."""
        # pylint: disable=import-outside-toplevel
        from bili.iris.loaders.tools_loader import initialize_tools

        tools = initialize_tools(active_tools=[ASK_USER_TOOL_NAME], tool_prompts={})
        assert len(tools) == 1

        model = _scripted_ask_user_model()
        agent = build_agent_graph(
            graph_definition=_single_node_graph_definition(),
            node_kwargs={
                "llm_model": model,
                "tools": tools,
                "tool_strategy": "native",
            },
        )

        thread_id = "iris-ask-user-thread"
        config = {"configurable": {"thread_id": thread_id}}

        # --- First invoke: expect the graph to pause at the interrupt. ---
        result = agent.invoke(
            {"messages": [HumanMessage(content="Please deploy the app.")]},
            config=config,
        )

        assert "__interrupt__" in result, (
            "graph.invoke() must return early with a pending interrupt "
            "instead of completing the turn"
        )
        pending_interrupts = result["__interrupt__"]
        assert len(pending_interrupts) == 1
        payload = pending_interrupts[0].value
        assert payload["type"] == ASK_USER_TOOL_NAME
        assert payload["question"] == "Which environment should I deploy to?"

        # The model must have been invoked exactly once so far (the call
        # that produced the ask_user tool_call) -- not yet the post-answer
        # call.
        assert model.invocation_count == 1

        # --- Resume: supply the answer via Command(resume=...). ---
        final = agent.invoke(Command(resume="staging"), config=config)

        assert "__interrupt__" not in final, "the turn must complete after resume"
        final_message = final["messages"][-1]
        assert final_message.content == "Deploying to the environment you specified."

        # THE GATING ASSERTION: the model must be invoked exactly twice total
        # (once before the pause, once after resume with the answer folded
        # into a ToolMessage), not three times. A third call would mean the
        # pre-interrupt model call is being re-executed on resume -- the
        # double-fire risk this test exists to pin. See
        # test_ask_user_aether_integration.py for the equivalent assertion
        # with AETHER's extra provenance-wrapping layer on top.
        assert model.invocation_count == 2

        # The answer must have reached the model as a ToolMessage, not been
        # dropped or re-asked.
        tool_messages = [
            m for m in final["messages"] if type(m).__name__ == "ToolMessage"
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "staging"

    def test_null_responder_default_does_not_block_registration(self):
        """register_ask_user_tool() with no responder still registers a usable tool."""
        # pylint: disable=import-outside-toplevel
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        assert ASK_USER_TOOL_NAME in TOOL_REGISTRY
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        assert tool.name == ASK_USER_TOOL_NAME
