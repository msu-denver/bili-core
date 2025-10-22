"""
react_agent_node.py
-------------------

This module provides a utility for constructing a REACT agent node for conversational workflows.
It supports both tool-enabled and tool-less operation, adapting to the capabilities of the
provided language model (LLM).

Functions:
----------
- build_react_agent_node(tools: list[Tool] = None, state: StateSchema = State, llm_model=None, **kwargs):
    Constructs a REACT agent node using the specified tools and state schema. If tools are provided,
    a REACT agent is created with tool integration. If tools are not supported, a fallback node is
    created that directly invokes the LLM, filtering and repackaging messages for compatibility.

Dependencies:
-------------
- langchain_core.messages: Provides `AIMessage`, `HumanMessage`, and `SystemMessage`
classes for chat history.
- langchain_core.tools: Provides the `Tool` class for agent tool integration.
- langgraph.prebuilt.chat_agent_executor: Provides `StateSchema` and `create_react_agent`
for agent construction.
- bili.utils.langgraph_utils.State: Defines the state schema for conversation data.
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `build_react_agent_node` to create a REACT agent node for
conversational agent workflows.

Example:
--------
from bili.nodes.react_agent_node import build_react_agent_node

agent_node = build_react_agent_node(
    tools=my_tools,
    state=MyStateSchema,
    llm_model=my_llm
)
result = agent_node(state)
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.prebuilt.chat_agent_executor import StateSchema, create_react_agent

from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_react_agent_node(
    tools: list[Tool] = None, state: StateSchema = State, llm_model=None, **kwargs
):
    """
    Constructs a REACT agent node using the specified tools and state or a fallback
    node for models that do not support tools. The REACT agent integrates tools
    and language models to process the input state and perform tasks.

    If tools are provided, creates a REACT agent using the specified tools
    with the associated state schema. If tools are not supported, creates a
    fallback node that directly interacts with the language model (LLM) while
    adapting the state's message format to prevent compatibility issues.

    :param tools: List of tools to be integrated with the REACT agent. If None,
        a fallback mechanism is used. Tools enable additional functionality for
        task handling.
    :type tools: list[Tool]

    :param state: Represents the schema of the state that interacts with the
        agent or LLM. This includes maintaining messages and other context
        required for conversation and workflows.
    :type state: StateSchema

    :param llm_model: The language model to be used by the agent or fallback
        node. This model processes the input messages and produces responses.
        The exact behavior may vary depending on the model's capabilities.

    :param kwargs: Additional arguments and settings for constructing the REACT
        agent or fallback node. This may include configuration options that are
        passed to internal functions.

    :return: A REACT agent node if tools are provided, otherwise a fallback node
        that directly interacts with the language model. Both options are
        structured to handle the input state appropriately and return the
        processed state.
    :rtype: Callable or any
    """
    LOGGER.debug(
        "Using model: %s | Tools enabled: %s",
        getattr(llm_model, "__class__", None),
        tools is not None,
    )
    if tools is not None:
        # If tools are supported, create a REACT agent with the provided tools
        LOGGER.debug("Creating REACT agent with tools: %s", tools)
        compiled_agent = create_react_agent(
            model=llm_model,
            state_schema=state,
            tools=tools,
        )

        # Wrap the compiled agent in a callable function so it can be used as a node
        # (CompiledStateGraph can't be added directly as a node)
        def agent(state: State):
            return compiled_agent.invoke(state)
    else:
        LOGGER.debug(
            "TOOLS not supported for this LLM. Creating a simple graph node to invoke the LLM"
            "directly, and to convert the state to a format that the LLM can understand."
        )

        # If tools are not supported, create a simple graph node that calls the LLM directly
        def call_model(state: State):
            # Filter out any tool messages to prevent errors in the LLM during the call
            # This is to allow for the scenario where the state was established with a model
            # that used tools, and then later a model that does not support tools was selected.
            # This allows the conversation to continue without the history of tool messages being
            # passed to the model.
            messages = [
                message
                for message in state["messages"]
                if isinstance(message, (AIMessage, HumanMessage, SystemMessage))
            ]
            # Repackage every message as a HumanMessage with only the text content
            # There are some bugs out there right now with limitations of the kinds
            # of messages that can be sent to certain models, and this is a way to work around
            # that for now.
            messages_to_send = [
                HumanMessage(content=str(message.content)) for message in messages
            ]
            response = llm_model.invoke(messages_to_send)
            return {"messages": [response]}

        agent = call_model
    return agent
