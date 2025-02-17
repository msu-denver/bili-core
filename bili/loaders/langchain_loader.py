"""
Module: langchain_loader

This module provides functions to load and initialize various components for LangGraph,
replacing the old LangChain usage (initialize_agent). It includes functions to configure
and build a LangGraph REACT agent with customizable execution flow and tool support.

Functions:
    - load_langgraph_agent(llm_model, langgraph_system_prefix, active_tools=None,
      tool_prompts=None, tool_params=None, k=5, memory_strategy="trim",
      memory_limit_type="message_count", checkpoint_saver=None, custom_nodes=None,
      exclude_nodes=None):
      Initializes and configures a LangGraph REACT agent dynamically with customizable
      execution flow and tool support.

Dependencies:
    - logging: Provides logging functionality.
    - langchain: Provides core LangChain functionality.
    - langchain_core.messages: Imports message types for LangChain.
    - langgraph.checkpoint.memory: Imports MemorySaver for checkpointing.
    - langgraph.constants: Imports constants for LangGraph.
    - langgraph.graph: Imports StateGraph for building the graph.
    - langgraph.prebuilt: Imports create_react_agent for creating REACT agents.
    - bili.loaders.tools_loader: Imports initialize_tools for tool initialization.
    - bili.utils.langgraph_utils: Imports utility functions for LangGraph.
    - bili.utils.logging_utils: Imports get_logger for logging.

Usage:
    This module is intended to be used within applications that require dynamic
    initialization and configuration of LangGraph REACT agents. It provides a function
    to build and manage the execution order of nodes, add custom nodes, handle excluded
    nodes, and ensure dependency resolution.

Example:
    from bili.loaders.langchain_loader import load_langgraph_agent

    # Initialize a LangGraph REACT agent
    agent = load_langgraph_agent(
        llm_model=my_llm_model,
        langgraph_system_prefix="my_prefix",
        active_tools=["tool1", "tool2"],
        tool_prompts={"tool1": "Prompt for tool1", "tool2": "Prompt for tool2"},
        tool_params={"tool1": {"param1": "value1"}, "tool2": {"param2": "value2"}},
        k=5,
        memory_strategy="trim",
        memory_limit_type="message_count",
        checkpoint_saver=my_checkpoint_saver,
        custom_nodes={"custom_node": {"function": my_function, "after": ["node1"]}},
        exclude_nodes=["node_to_exclude"]
    )
"""

import logging

import langchain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from bili.loaders.tools_loader import initialize_tools
from bili.utils.langgraph_utils import (
    State,
    build_add_persona_and_summary_node,
    build_normalize_tool_state_node,
    build_trim_and_summarize_node,
)
from bili.utils.logging_utils import get_logger

# Get the logger for the module
LOGGER = get_logger(__name__)

# Get current logging log level
current_log_level = LOGGER.getEffectiveLevel()

# Set the langchain debug mode to True if log level is less than or equal to DEBUG
if current_log_level <= logging.DEBUG:
    langchain.debug = True


def load_langgraph_agent(
    llm_model,
    langgraph_system_prefix,
    active_tools=None,
    tool_prompts=None,
    tool_params=None,
    k=5,
    memory_strategy="trim",
    memory_limit_type="message_count",
    checkpoint_saver=None,
    custom_nodes=None,
    exclude_nodes=None,
):
    """
    Initializes and configures a LangGraph REACT agent dynamically with customizable execution flow
    and tool support. The function builds and manages the execution order of nodes,
    adds custom nodes, handles excluded nodes, and ensures dependency resolution.
    It supports runtime configuration of tools, memory strategies, and checkpointer mechanisms.

    :param llm_model: LLM model instance used by the agent.
    :param langgraph_system_prefix: String representing system prefix for LangGraph configuration.
    :param active_tools: List of active tool identifiers to enable, or None if no tools are used.
    :param tool_prompts: Optional prompts for each tool in active_tools.
    :param tool_params: Parameters for configuring behavior or settings of active_tools.
    :param k: Integer controlling a configuration parameter for summarization nodes.
    :param memory_strategy: String indicating strategy for managing memory, e.g., "trim".
    :param memory_limit_type: String indicating memory usage limit type, e.g., "message_count".
    :param checkpoint_saver: Checkpoint saving mechanism; defaults to an instance of MemorySaver.
    :param custom_nodes: Dictionary defining custom nodes to integrate dynamically. Each entry
        should specify a "function" (callable) and "after" (list of dependencies it depends on).
    :param exclude_nodes: List of node identifiers to exclude from the graph, if applicable.
    :return: Compiled StateGraph instance representing the LangGraph REACT agent.
    """
    LOGGER.debug("Initializing LangGraph agent with configurable execution flow...")

    # 1. Define default nodes and dependencies
    default_nodes = {
        "add_persona_and_summary": build_add_persona_and_summary_node(
            langgraph_system_prefix
        ),
        "trim_summarize": build_trim_and_summarize_node(
            memory_limit_type, memory_strategy, k, llm_model
        ),
        "normalize_state": build_normalize_tool_state_node(),
    }

    # 2. Build the REACT agent with enabled tools via create_react_agent if tools are supported
    if active_tools is not None:
        tools = initialize_tools(active_tools, tool_prompts, tool_params)
        default_nodes["react_agent"] = create_react_agent(
            model=llm_model,
            state_schema=State,
            tools=tools,
        )
        LOGGER.debug("Tools initialized: %s", tools)
    else:
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

        default_nodes["react_agent"] = call_model

    default_dependencies = {
        "add_persona_and_summary": [],
        "react_agent": ["add_persona_and_summary"],
        "trim_summarize": ["react_agent"],
        "normalize_state": ["trim_summarize"],
    }

    # 3. Remove any excluded nodes
    if exclude_nodes:
        for node in exclude_nodes:
            default_nodes.pop(node, None)
            default_dependencies.pop(node, None)

            # Also remove the excluded nodes from other node dependencies
            for deps in default_dependencies.values():
                if node in deps:
                    deps.remove(node)

    # 4. Add custom nodes dynamically
    if custom_nodes:
        for node_name, config in custom_nodes.items():
            function = config["function"]
            after_nodes = config["after"]

            if not isinstance(after_nodes, list):
                raise ValueError(
                    f"Expected list for 'after' in node {node_name}, got {type(after_nodes)}"
                )

            default_nodes[node_name] = function
            default_dependencies[node_name] = after_nodes

    # 5. Compute the execution order dynamically
    execution_order = []
    while default_dependencies:
        # Find nodes that have no unresolved dependencies
        ready_nodes = [
            node
            for node, deps in default_dependencies.items()
            if not any(dep in default_dependencies for dep in deps)
        ]

        if not ready_nodes:
            raise ValueError("Circular dependency detected in node definitions!")

        execution_order.extend(ready_nodes)

        # Remove processed nodes
        for node in ready_nodes:
            del default_dependencies[node]

    LOGGER.debug(f"Final execution order: {execution_order}")

    # 6. Build the LangGraph state graph
    parent_graph = StateGraph(State)

    # Add nodes dynamically
    for node_name in execution_order:
        parent_graph.add_node(node_name, default_nodes[node_name])

    # Add edges dynamically based on execution order
    for i in range(len(execution_order) - 1):
        LOGGER.debug(
            "Adding edge from %s to %s", execution_order[i], execution_order[i + 1]
        )
        parent_graph.add_edge(execution_order[i], execution_order[i + 1])

    # Connect the first and last nodes
    LOGGER.debug("Adding edge from START to %s", execution_order[0])
    parent_graph.add_edge(START, execution_order[0])

    LOGGER.debug("Adding edge from %s to END", execution_order[-1])
    parent_graph.add_edge(execution_order[-1], END)

    # 7. Use provided checkpoint saver, or default to MemorySaver
    if not checkpoint_saver:
        LOGGER.debug("No checkpoint_saver provided; using MemorySaver.")
        checkpoint_saver = MemorySaver()

    compiled_graph = parent_graph.compile(
        checkpointer=checkpoint_saver,
    )

    LOGGER.debug("LangGraph REACT agent created successfully.")
    return compiled_graph
