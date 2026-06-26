"""Tests for react_agent_node module.

Tests the ReAct agent node builder:
- Builder returns a callable
- Callable with tools creates a REACT agent via create_agent
- Callable without tools creates a fallback LLM-only node
- Fallback node filters non-text messages and invokes the LLM
- Fallback node forwards llm_config from state
- Middleware is forwarded to create_agent
- Prompted tool-calling path (supports_tools=False) exercises the hand-rolled
  ReAct loop: tool injection, Action/Final Answer parsing, tool execution,
  error handling, iteration cap, and consecutive-parse-failure escape hatch.
- Auto-selection between native, prompted, and fallback paths.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bili.iris.nodes.react_agent_node import (
    _build_prompted_react_loop,
    _parse_react_response,
    build_react_agent_node,
    react_agent_node,
)


class TestBuildReactAgentNode:
    """Tests for build_react_agent_node function."""

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_returns_callable_with_tools(self, mock_create_agent):
        """Build with tools should return whatever create_agent produces."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_llm = MagicMock()
        tools = [MagicMock()]

        result = build_react_agent_node(tools=tools, llm_model=mock_llm)

        assert result is mock_agent
        mock_create_agent.assert_called_once()

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_create_agent_receives_tools_and_model(self, mock_create_agent):
        """Verify create_agent is called with the right tools and model."""
        mock_llm = MagicMock()
        tools = [MagicMock(), MagicMock()]
        mock_create_agent.return_value = MagicMock()

        build_react_agent_node(tools=tools, llm_model=mock_llm)

        call_kwargs = mock_create_agent.call_args
        assert call_kwargs.kwargs["model"] is mock_llm
        assert call_kwargs.kwargs["tools"] is tools

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_middleware_forwarded_to_create_agent(self, mock_create_agent):
        """Middleware list should be passed through to create_agent."""
        mock_llm = MagicMock()
        middleware = [MagicMock(), MagicMock()]
        mock_create_agent.return_value = MagicMock()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=mock_llm,
            middleware=middleware,
        )

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["middleware"] is middleware

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_none_middleware_becomes_empty_tuple(self, mock_create_agent):
        """When middleware is None, an empty tuple should be passed."""
        mock_create_agent.return_value = MagicMock()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=MagicMock(),
            middleware=None,
        )

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["middleware"] == ()

    def test_returns_callable_without_tools(self):
        """Build without tools should return a callable fallback."""
        mock_llm = MagicMock()

        result = build_react_agent_node(tools=None, llm_model=mock_llm)

        assert callable(result)

    def test_fallback_invokes_llm_with_messages(self):
        """Fallback node should invoke the LLM with repackaged messages."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Hello back!")
        mock_llm.invoke.return_value = mock_response

        node_func = build_react_agent_node(tools=None, llm_model=mock_llm)
        state = {
            "messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Hi"),
            ]
        }

        result = node_func(state)

        assert "messages" in result
        assert result["messages"] == [mock_response]
        mock_llm.invoke.assert_called_once()

    def test_fallback_filters_tool_messages(self):
        """Fallback node should filter out ToolMessage objects."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="response")

        node_func = build_react_agent_node(tools=None, llm_model=mock_llm)
        state = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content="Let me check"),
                ToolMessage(
                    content="tool result",
                    tool_call_id="tc1",
                ),
                AIMessage(content="Here you go"),
            ]
        }

        node_func(state)

        sent_messages = mock_llm.invoke.call_args[0][0]
        for msg in sent_messages:
            assert isinstance(msg, HumanMessage)

    def test_fallback_repackages_as_human_messages(self):
        """Fallback node should convert all messages to HumanMessage."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        node_func = build_react_agent_node(tools=None, llm_model=mock_llm)
        state = {
            "messages": [
                SystemMessage(content="system"),
                HumanMessage(content="human"),
                AIMessage(content="ai"),
            ]
        }

        node_func(state)

        sent_messages = mock_llm.invoke.call_args[0][0]
        assert len(sent_messages) == 3
        for msg in sent_messages:
            assert isinstance(msg, HumanMessage)

    def test_fallback_passes_llm_config_from_state(self):
        """Fallback node should forward llm_config from state."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")
        config = {"thinking_config": {"budget": 500}}

        node_func = build_react_agent_node(tools=None, llm_model=mock_llm)
        state = {
            "messages": [HumanMessage(content="hi")],
            "llm_config": config,
        }

        node_func(state)

        call_kwargs = mock_llm.invoke.call_args.kwargs
        assert call_kwargs["config"] is config

    def test_fallback_uses_empty_dict_when_no_llm_config(self):
        """Fallback node should pass empty dict when no llm_config."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        node_func = build_react_agent_node(tools=None, llm_model=mock_llm)
        state = {"messages": [HumanMessage(content="hi")]}

        node_func(state)

        call_kwargs = mock_llm.invoke.call_args.kwargs
        assert call_kwargs["config"] == {}

    def test_accepts_extra_kwargs(self):
        """Builder should accept extra kwargs without error."""
        mock_llm = MagicMock()

        result = build_react_agent_node(
            tools=None,
            llm_model=mock_llm,
            extra_param="value",
        )

        assert callable(result)


class TestReactAgentNodePartial:
    """Tests for the react_agent_node partial."""

    def test_partial_creates_node_with_correct_name(self):
        """The partial should create a Node named 'react_agent'."""
        node = react_agent_node()
        assert node.name == "react_agent"

    def test_partial_creates_callable_node(self):
        """The Node created by the partial should be callable."""
        node = react_agent_node()
        assert callable(node)

    def test_partial_call_invokes_builder(self):
        """Calling the Node should invoke the builder function."""
        node = react_agent_node()
        mock_llm = MagicMock()
        result = node(tools=None, llm_model=mock_llm)
        assert callable(result)


# ---------------------------------------------------------------------------
# Helper: build a minimal mock tool
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str = "", return_value: str = "42"):
    """Return a MagicMock that quacks like a LangChain BaseTool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = None  # triggers the sig=(...)  fallback
    tool.invoke.return_value = return_value
    return tool


# ---------------------------------------------------------------------------
# _parse_react_response unit tests
# ---------------------------------------------------------------------------


class TestParseReactResponse:
    """Unit tests for the response-parser helper."""

    def test_final_answer(self):
        """'Final Answer:' in response returns ('final', answer_text, None)."""
        text = "Thought: I know the answer.\nFinal Answer: 42"
        kind, value, extra = _parse_react_response(text)
        assert kind == "final"
        assert value == "42"
        assert extra is None

    def test_action_with_valid_json(self):
        """Action + valid JSON Input returns ('action', tool_name, args_dict)."""
        text = 'Thought: I need to add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
        kind, value, extra = _parse_react_response(text)
        assert kind == "action"
        assert value == "add"
        assert extra == {"a": 1, "b": 2}

    def test_action_with_malformed_json(self):
        """Action with invalid JSON returns ('parse_error', raw_input, error_msg)."""
        text = "Action: add\nAction Input: {not valid json}"
        kind, value, extra = _parse_react_response(text)
        assert kind == "parse_error"
        assert value == "{not valid json}"
        assert extra  # error message string

    def test_unknown_response(self):
        """Response with neither marker returns ('unknown', None, None)."""
        kind, value, extra = _parse_react_response("I don't know what to do.")
        assert kind == "unknown"
        assert value is None
        assert extra is None

    def test_final_answer_wins_over_action(self):
        """When both markers appear, Final Answer takes precedence."""
        text = "Action: foo\nAction Input: {}\nFinal Answer: done"
        kind, value, _ = _parse_react_response(text)
        assert kind == "final"
        assert value == "done"

    def test_strips_markdown_code_fences(self):
        """Markdown code fences around JSON are stripped before parsing."""
        text = 'Action: add\nAction Input: ```json\n{"a": 1}\n```'
        kind, value, extra = _parse_react_response(text)
        assert kind == "action"
        assert value == "add"
        assert extra == {"a": 1}

    def test_case_insensitive_final_answer(self):
        """'final answer:' (lowercase) is accepted."""
        text = "final answer: result text"
        kind, value, _ = _parse_react_response(text)
        assert kind == "final"
        assert value == "result text"


# ---------------------------------------------------------------------------
# _build_prompted_react_loop integration tests
# ---------------------------------------------------------------------------


class TestPromptedReactLoop:
    """Tests for the prompted ReAct loop path."""

    # -- 1. Happy path: one tool call then final answer -----------------------

    def test_happy_path_one_tool_call(self):
        """Loop calls the tool once then returns the Final Answer."""
        add_tool = _make_tool("add", return_value="3")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content='Thought: I need to add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            ),
            AIMessage(content="Thought: I have the result.\nFinal Answer: 3"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [add_tool])
        state = {"messages": [HumanMessage(content="What is 1 + 2?")]}
        result = loop(state)

        assert result["messages"] == [AIMessage(content="3")]
        add_tool.invoke.assert_called_once_with({"a": 1, "b": 2})
        assert mock_llm.invoke.call_count == 2

    # -- 2. Multi-step tool calls ---------------------------------------------

    def test_multi_step_tool_calls(self):
        """Loop handles two sequential tool calls before the final answer."""
        search_tool = _make_tool("search", return_value="Paris")
        pop_tool = _make_tool("population", return_value="2 million")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content='Action: search\nAction Input: {"q": "capital of France"}'
            ),
            AIMessage(content='Action: population\nAction Input: {"city": "Paris"}'),
            AIMessage(content="Final Answer: Paris has about 2 million people."),
        ]

        loop = _build_prompted_react_loop(mock_llm, [search_tool, pop_tool])
        state = {
            "messages": [
                HumanMessage(content="What is the population of the French capital?")
            ]
        }
        result = loop(state)

        assert "Paris" in result["messages"][0].content
        assert search_tool.invoke.call_count == 1
        assert pop_tool.invoke.call_count == 1
        assert mock_llm.invoke.call_count == 3

    # -- 3. Final answer on first response (no tools needed) ------------------

    def test_final_answer_first_response(self):
        """When the model answers immediately no tool should be called."""
        tool = _make_tool("unused")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Final Answer: Paris")

        loop = _build_prompted_react_loop(mock_llm, [tool])
        result = loop({"messages": [HumanMessage(content="Capital of France?")]})

        assert result["messages"] == [AIMessage(content="Paris")]
        tool.invoke.assert_not_called()
        assert mock_llm.invoke.call_count == 1

    # -- 4. Unknown tool name -------------------------------------------------

    def test_unknown_tool_name_feeds_error_observation(self):
        """An unknown tool name produces an error observation and the loop continues."""
        real_tool = _make_tool("real_tool", return_value="ok")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Action: ghost_tool\nAction Input: {}"),
            AIMessage(content="Final Answer: recovered"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [real_tool])
        result = loop({"messages": [HumanMessage(content="test")]})

        # The error observation should have been fed back
        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        obs_texts = [
            m.content for m in second_call_messages if isinstance(m, HumanMessage)
        ]
        assert any("unknown tool" in t for t in obs_texts)
        assert result["messages"][0].content == "recovered"

    # -- 5. Malformed JSON in Action Input ------------------------------------

    def test_malformed_json_action_input(self):
        """Malformed JSON in Action Input feeds a parse-error observation."""
        tool = _make_tool("add")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Action: add\nAction Input: not json at all"),
            AIMessage(content="Final Answer: sorry, fixed"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [tool])
        result = loop({"messages": [HumanMessage(content="test")]})

        tool.invoke.assert_not_called()
        # Error observation should have been injected
        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        obs_texts = [
            m.content for m in second_call_messages if isinstance(m, HumanMessage)
        ]
        assert any("Error" in t for t in obs_texts)
        assert result["messages"][0].content == "sorry, fixed"

    # -- 6. Tool execution raises an exception --------------------------------

    def test_tool_execution_error(self):
        """A tool that raises feeds an error observation and the loop continues."""
        bad_tool = _make_tool("bad_tool")
        bad_tool.invoke.side_effect = RuntimeError("db timeout")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Action: bad_tool\nAction Input: {}"),
            AIMessage(content="Final Answer: could not retrieve data"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [bad_tool])
        result = loop({"messages": [HumanMessage(content="test")]})

        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        obs_texts = [
            m.content for m in second_call_messages if isinstance(m, HumanMessage)
        ]
        assert any("db timeout" in t for t in obs_texts)
        assert result["messages"][0].content == "could not retrieve data"

    # -- 7. Max iterations guard ----------------------------------------------

    def test_max_iterations_guard(self):
        """Loop terminates at max_react_iterations and returns last response."""
        tool = _make_tool("loop_tool", return_value="still looping")
        mock_llm = MagicMock()
        # Model always asks to call the tool; never produces Final Answer
        mock_llm.invoke.return_value = AIMessage(
            content="Action: loop_tool\nAction Input: {}"
        )

        loop = _build_prompted_react_loop(mock_llm, [tool], max_react_iterations=3)
        result = loop({"messages": [HumanMessage(content="go")]})

        # Should have stopped after 3 iterations
        assert mock_llm.invoke.call_count == 3
        assert "loop_tool" in result["messages"][0].content

    # -- 8. Consecutive parse failure escape hatch ----------------------------

    def test_consecutive_parse_failures_exit_loop(self):
        """Three consecutive unparseable responses exit the loop gracefully."""
        tool = _make_tool("some_tool")
        mock_llm = MagicMock()
        # Always returns garbage that can't be parsed
        mock_llm.invoke.return_value = AIMessage(
            content="I have no idea what to say here."
        )

        loop = _build_prompted_react_loop(mock_llm, [tool], max_react_iterations=10)
        result = loop({"messages": [HumanMessage(content="test")]})

        # Should have stopped after 3 consecutive failures (not all 10 iterations)
        assert mock_llm.invoke.call_count == 3
        assert result["messages"][0].content == "I have no idea what to say here."

    # -- 9. llm_config forwarded on each iteration ----------------------------

    def test_llm_config_forwarded_each_iteration(self):
        """llm_config from state is passed to every model.invoke() call."""
        tool = _make_tool("t", return_value="x")
        config = {"thinking_config": {"budget": 500}}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Action: t\nAction Input: {}"),
            AIMessage(content="Final Answer: done"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [tool])
        state = {
            "messages": [HumanMessage(content="test")],
            "llm_config": config,
        }
        loop(state)

        for c in mock_llm.invoke.call_args_list:
            assert c.kwargs["config"] is config

    # -- 10. Tool preamble injected into existing system message --------------

    def test_tool_preamble_appended_to_existing_system_message(self):
        """The tool preamble is appended to an existing SystemMessage."""
        tool = _make_tool("calc")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Final Answer: done")

        loop = _build_prompted_react_loop(mock_llm, [tool])
        state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="hi"),
            ]
        }
        loop(state)

        first_call_messages = mock_llm.invoke.call_args_list[0][0][0]
        sys_msgs = [m for m in first_call_messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "You are a helpful assistant." in sys_msgs[0].content
        assert "calc" in sys_msgs[0].content  # tool name injected

    # -- 11. Tool preamble added when no system message exists ----------------

    def test_tool_preamble_prepended_when_no_system_message(self):
        """When there is no system message, a new one is prepended with the preamble."""
        tool = _make_tool("lookup")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Final Answer: ok")

        loop = _build_prompted_react_loop(mock_llm, [tool])
        state = {"messages": [HumanMessage(content="question")]}
        loop(state)

        first_call_messages = mock_llm.invoke.call_args_list[0][0][0]
        assert isinstance(first_call_messages[0], SystemMessage)
        assert "lookup" in first_call_messages[0].content

    # -- 12. Observations injected as HumanMessage (strict-turn compat) -------

    def test_observation_injected_as_human_message(self):
        """Tool observations are fed back as HumanMessage, not ToolMessage."""
        tool = _make_tool("greet", return_value="hello world")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Action: greet\nAction Input: {}"),
            AIMessage(content="Final Answer: done"),
        ]

        loop = _build_prompted_react_loop(mock_llm, [tool])
        loop({"messages": [HumanMessage(content="greet me")]})

        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        obs_msgs = [
            m
            for m in second_call_messages
            if isinstance(m, HumanMessage) and "Observation:" in m.content
        ]
        assert len(obs_msgs) == 1
        assert "hello world" in obs_msgs[0].content


# ---------------------------------------------------------------------------
# Auto-selection tests
# ---------------------------------------------------------------------------


class TestAutoSelection:
    """Verify that build_react_agent_node selects the correct path."""

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_native_path_when_supports_tools_true(self, mock_create_agent):
        """tools + supports_tools=True (default) -> create_agent called."""
        mock_create_agent.return_value = MagicMock()
        tool = _make_tool("t")

        build_react_agent_node(tools=[tool], llm_model=MagicMock(), supports_tools=True)

        mock_create_agent.assert_called_once()

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_native_path_default_when_no_flag(self, mock_create_agent):
        """tools without supports_tools kwarg defaults to native path."""
        mock_create_agent.return_value = MagicMock()
        tool = _make_tool("t")

        build_react_agent_node(tools=[tool], llm_model=MagicMock())

        mock_create_agent.assert_called_once()

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_prompted_path_when_supports_tools_false(self, mock_create_agent):
        """tools + supports_tools=False -> create_agent NOT called; callable returned."""
        tool = _make_tool("t")

        result = build_react_agent_node(
            tools=[tool], llm_model=MagicMock(), supports_tools=False
        )

        mock_create_agent.assert_not_called()
        assert callable(result)

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_fallback_path_when_tools_none(self, mock_create_agent):
        """tools=None -> existing call_model fallback; create_agent not called."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="hi")

        result = build_react_agent_node(tools=None, llm_model=mock_llm)

        mock_create_agent.assert_not_called()
        assert callable(result)
        # Verify it's the call_model fallback, not the prompted loop
        state = {"messages": [HumanMessage(content="hi")]}
        out = result(state)
        assert "messages" in out

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_fallback_path_ignores_supports_tools_when_no_tools(
        self, mock_create_agent
    ):
        """tools=None with supports_tools=False still uses the tool-less fallback."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="hi")

        result = build_react_agent_node(
            tools=None, llm_model=mock_llm, supports_tools=False
        )

        mock_create_agent.assert_not_called()
        # Should be a plain callable (call_model), not the prompted loop
        assert callable(result)

    @patch("bili.iris.nodes.react_agent_node._build_prompted_react_loop")
    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_max_react_iterations_forwarded_to_prompted_loop(
        self, mock_create_agent, mock_prompted_factory
    ):
        """max_react_iterations kwarg is forwarded to _build_prompted_react_loop."""
        mock_prompted_factory.return_value = MagicMock()
        tool = _make_tool("t")

        build_react_agent_node(
            tools=[tool],
            llm_model=MagicMock(),
            supports_tools=False,
            max_react_iterations=5,
        )

        mock_create_agent.assert_not_called()
        mock_prompted_factory.assert_called_once()
        call_kwargs = mock_prompted_factory.call_args.kwargs
        assert call_kwargs["max_react_iterations"] == 5
