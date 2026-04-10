"""Tests for react_agent_node module.

Tests the ReAct agent node builder:
- Builder returns a callable
- Callable with tools creates a REACT agent via create_agent
- Callable without tools creates a fallback LLM-only node
- Fallback node filters non-text messages and invokes the LLM
- Fallback node forwards llm_config from state
- Middleware is forwarded to create_agent
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bili.iris.nodes.react_agent_node import build_react_agent_node, react_agent_node


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
