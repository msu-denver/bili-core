"""End-to-end integration test: ask_user through a real AETHER MAS.

Builds a REAL single-agent sequential MAS (real MASExecutor.initialize() ->
real compile_mas() -> real create_agent() subgraph), scripts a fake
tool-calling model to call ask_user, drives it through
langgraph.types.interrupt() pause and MASExecutor.resume_with_value()
resume, and pins the double-fire risk named in the chunk-1 design review:
does AETHER's outer-node bookkeeping (communication_log append, provenance/
agent_outputs, the wrap_node performance timer) fire once or twice across a
pause/resume cycle.

Only create_llm and resolve_tool_strategy are mocked (provider resolution
has nothing to do with the interrupt seam under test); resolve_tools,
compile_mas, and the full LangGraph execution path are all real. This is
deliberately NOT the same shape as the existing agent_generator tests in
test_compiler.py, which mock create_agent itself -- that would hide the
exact risk this test exists to pin.

See test_ask_user_iris_integration.py for the equivalent one-graph-layer
IRIS-only test this builds on.
"""

# pylint: disable=duplicate-code
# _ScriptedToolCallingModel is intentionally re-declared (not shared via
# import) from bili/iris/tools/tests/test_ask_user_iris_integration.py: see
# that file's own duplicate-code disable comment for why.

from typing import Any, List, Sequence
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult

from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.iris.tools.ask_user import (
    ASK_USER_TOOL_NAME,
    register_ask_user_tool,
    unregister_ask_user_tool,
)

_MOCK_CREATE_LLM = "bili.aether.compiler.llm_resolver.create_llm"
_MOCK_TOOL_STRATEGY = "bili.aether.compiler.llm_resolver.resolve_tool_strategy"


class _ScriptedToolCallingModel(BaseChatModel):
    """Fake chat model that supports bind_tools and cycles scripted responses.

    Instrumented with three independent counters so the test can assert
    exactly what re-executes on resume without conflating "which response is
    next" (an index that must NOT double-advance) with "how many times was
    _generate() really called" (a count that must be exactly 2, not 3, for
    the double-fire assertion to be meaningful).
    """

    responses: List[BaseMessage] = []
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


def _single_agent_config() -> MASConfig:
    """A single-agent sequential MAS with ask_user in its tool list."""
    agent = AgentSpec(
        agent_id="deployer",
        role="deployer",
        objective="Deploy the application, asking the human which environment to use.",
        model_name="gpt-4o",
        tools=[ASK_USER_TOOL_NAME],
    )
    return MASConfig(
        mas_id="ask_user_test_mas",
        name="ask_user Integration Test MAS",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
        checkpoint_enabled=True,
    )


class TestAskUserAetherIntegration:
    """Proves the ask_user pause/resume seam through a real AETHER MAS."""

    def setup_method(self):
        """Register ask_user before each test."""
        register_ask_user_tool()

    def teardown_method(self):
        """Unregister ask_user after each test."""
        unregister_ask_user_tool()

    def test_pause_and_resume_through_real_mas(  # pylint: disable=too-many-locals
        self,
    ):
        """A real AETHER MAS pauses at ask_user and resumes with the answer.

        THE GATING ASSERTIONS (chunk-1 design review deliverable): after
        resume, the communication log and agent_outputs (AETHER's per-agent
        provenance bookkeeping, appended only at the true end of the agent
        node) each have exactly ONE entry for the deployer agent -- not two.
        The interrupted tool-calling node genuinely re-executes from its own
        start on resume (langgraph.types.interrupt()'s documented behavior),
        but that does not mean the outer AETHER node's terminal bookkeeping
        fires twice; this test is the evidence, not an assumption.
        """
        model = _scripted_ask_user_model()
        config = _single_agent_config()

        with (
            patch(_MOCK_CREATE_LLM, return_value=model),
            patch(_MOCK_TOOL_STRATEGY, return_value="native"),
        ):
            executor = MASExecutor(config)
            executor.initialize()

            thread_id = "aether-ask-user-thread"
            events = list(
                executor.run_streaming(
                    {"messages": [HumanMessage(content="Please deploy the app.")]},
                    thread_id=thread_id,
                )
            )

            # --- Pause: expect exactly one __ask_user_pending__ sentinel. ---
            pending_events = [e for e in events if e[0] == "__ask_user_pending__"]
            assert len(pending_events) == 1
            pending_data = pending_events[0][1]
            assert pending_data["thread_id"] == thread_id
            assert len(pending_data["interrupts"]) == 1
            payload = pending_data["interrupts"][0]
            assert payload["type"] == ASK_USER_TOOL_NAME
            assert payload["question"] == "Which environment should I deploy to?"

            # No __human_interrupt__ sentinel: human_in_loop / is_human is a
            # distinct, unrelated mechanism and must not be touched by this
            # additive check (config.human_in_loop defaults to False here).
            assert not [e for e in events if e[0] == "__human_interrupt__"]

            # The deployer agent must not have completed yet -- its node is
            # still paused mid-execution, so no agent-level node_name event
            # for "deployer" should have reached the caller.
            assert not [e for e in events if e[0] == "deployer"]

            assert model.invocation_count == 1

            # --- Resume: supply the answer via resume_with_value(). ---
            resume_events = list(
                executor.resume_with_value("staging", thread_id=thread_id)
            )

        deployer_events = [e for e in resume_events if e[0] == "deployer"]
        assert len(deployer_events) == 1, (
            "the deployer agent node must complete exactly once after "
            "resume -- more than one 'deployer' event in the resumed "
            "stream would mean the outer AETHER node emitted its terminal "
            "state update twice"
        )
        final_state_update = deployer_events[0][1]
        final_messages = final_state_update["messages"]
        assert final_messages[-1].content == (
            "Deploying to the environment you specified."
        )

        # THE GATING ASSERTION: the model is invoked exactly twice total
        # (once before the pause, once after resume), not three times. A
        # third invocation would mean create_agent's already-completed
        # pre-interrupt model call is being replayed on resume.
        assert model.invocation_count == 2

        # THE GATING ASSERTION for AETHER's own bookkeeping: agent_outputs
        # and the communication log are written once at the true end of the
        # outer _agent_node closure (agent_generator.py), after
        # _invoke_executor() returns. If the outer node's full body were
        # replayed to completion twice, both would show 2 entries instead
        # of 1.
        agent_outputs = final_state_update.get("agent_outputs", {})
        assert "deployer" in agent_outputs
        assert agent_outputs["deployer"]["status"] == "completed"

        comm_log = final_state_update.get("communication_log", [])
        deployer_comm_entries = [
            entry for entry in comm_log if entry.get("sender") == "deployer"
        ]
        assert len(deployer_comm_entries) == 1, (
            f"expected exactly one communication_log entry from 'deployer' "
            f"after resume, got {len(deployer_comm_entries)}: {deployer_comm_entries}"
        )
