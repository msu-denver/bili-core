"""
langchain_loader.py
--------------------

This module provides functions to load and initialize various components for LangGraph,
replacing the old LangChain usage (initialize_agent). It includes functions to configure
and build a LangGraph REACT agent with customizable execution flow and tool support.

Functions:
----------
- build_agent_graph(checkpoint_saver=None, custom_node_registry=None,
  graph_definition=None, node_kwargs=None, state=None):
    Builds and compiles a state graph for an agent using the provided graph definition,
    node registry, and execution state. This function allows customization of the graph
    building process through adjustable parameters, enabling dynamic integration of
    custom nodes and configurations. The compiled graph can execute workflows for
    an agent by validating and linking the registered nodes.

Dependencies:
-------------
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
------
This module is intended to be used within applications that require dynamic
initialization and configuration of LangGraph REACT agents. It provides functions
to build and manage the execution order of nodes, add custom nodes, handle excluded
nodes, and ensure dependency resolution.

Example:
--------
from bili.loaders.langchain_loader import build_agent_graph

# Build and compile a LangGraph REACT agent
agent_graph = build_agent_graph(
    checkpoint_saver=my_checkpoint_saver,
    custom_node_registry={"custom_node": my_custom_node_function},
    graph_definition=["add_persona_and_summary", "custom_node", "react_agent"],
    node_kwargs={"llm_model": my_llm_model, "persona": "my_persona"},
    state=MyCustomState
)
"""

import difflib
import logging
import time
from typing import Callable

import langchain
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from bili.nodes.add_persona_and_summary import build_add_persona_and_summary_node
from bili.nodes.inject_current_datetime import build_inject_current_date_time
from bili.nodes.normalize_state import build_normalize_state_node
from bili.nodes.per_user_state import buld_per_user_state_node
from bili.nodes.react_agent_node import build_react_agent_node
from bili.nodes.trim_and_summarize import build_trim_and_summarize_node
from bili.nodes.update_timestamp import build_update_timestamp_node
from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Get the logger for the module
LOGGER = get_logger(__name__)

# Get current logging log level
current_log_level = LOGGER.getEffectiveLevel()

# Set the langchain debug mode to True if log level is less than or equal to DEBUG
if current_log_level <= logging.DEBUG:
    langchain.debug = True

# Define the graph node registry, which are nodes that are available to be used
# in the creation of a LangGraph REACT agent. Users can extend this registry with
# custom nodes.
GRAPH_NODE_REGISTRY = {
    "add_persona_and_summary": build_add_persona_and_summary_node,
    "normalize_state": build_normalize_state_node,
    "per_user_state": buld_per_user_state_node,
    "react_agent": build_react_agent_node,
    "trim_summarize": build_trim_and_summarize_node,
    "update_timestamp": build_update_timestamp_node,
    "inject_current_datetime": build_inject_current_date_time,
}

# Define the default graph definition, which is the default set of nodes that will
# be used in the creation of a LangGraph REACT agent. Users can extend this
# definition with custom nodes.
DEFAULT_GRAPH_DEFINITION = [
    "add_persona_and_summary",
    "inject_current_datetime",
    "react_agent",
    "update_timestamp",
    "trim_summarize",
    "normalize_state",
]


def wrap_node(node_func: Callable, node_name: str) -> Callable:
    """
    Wraps a node function to log its execution time.

    :param node_func: The node function to wrap.
    :param node_name: The name of the node for logging purposes.
    :return: A wrapped function that logs execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        # CompiledStateGraph objects need to use .invoke() method
        if isinstance(node_func, CompiledStateGraph):
            result = node_func.invoke(*args, **kwargs)
        else:
            result = node_func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000
        LOGGER.info(f"Node '{node_name}' executed in {execution_time:.2f} ms")
        return result

    return wrapper


def build_agent_graph(
    checkpoint_saver: BaseCheckpointSaver = None,
    custom_node_registry: dict[str, Callable] = None,
    graph_definition: list[str] = None,
    node_kwargs: dict = None,
    state: type = None,
) -> CompiledStateGraph:
    """
    Builds and compiles a state graph for a LangGraph-based agent using the specified
    parameters. The function initializes the graph with nodes defined by the provided
    or default graph definition, sets up edges between them, and compiles it using the
    provided or default checkpoint saver. The user can also supply custom node builders
    to extend or override the default node functionality.

    :param checkpoint_saver: A handler to save graph state updates during execution.
    :param custom_node_registry: Optional mapping of custom node names to their
        corresponding builder callables.
    :param graph_definition: A list of node names defining the sequence and the structure
        of the graph.
    :param node_kwargs: Additional keyword arguments passed to the node builder callables.
    :param state: Type representing the initial state of the graph to be used.
    :return: A compiled state graph ready for execution.
    :rtype: CompiledStateGraph
    """
    # Set default values for parameters
    # If no graph definition is provided, use the default graph definition
    if graph_definition is None:
        graph_definition = DEFAULT_GRAPH_DEFINITION

    # If no node kwargs are provided, use an empty dictionary
    if node_kwargs is None:
        node_kwargs = {}

    # If no checkpoint saver is provided, use a MemorySaver
    if checkpoint_saver is None:
        checkpoint_saver = MemorySaver()

    # If no state is provided, use the default State
    if state is None:
        state = State

    # If no custom node registry is provided, use the default graph node registry
    # This allows users to pass custom nodes to the graph builder without having to
    # modify the default graph node registry
    node_registry = GRAPH_NODE_REGISTRY.copy()
    if custom_node_registry:
        node_registry.update(custom_node_registry)

    # Validate the graph definition against the registered nodes and create the graph
    nodes = {}
    graph = StateGraph(state)
    LOGGER.debug("Building LangGraph agent with definition: %s", graph_definition)
    for node_name in graph_definition:
        if node_name not in node_registry:
            suggestions = difflib.get_close_matches(node_name, node_registry.keys())
            suggestion_msg = (
                f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            )
            raise ValueError(
                f"Node '{node_name}' is not defined in the graph registry.{suggestion_msg}"
            )
        builder = node_registry[node_name]
        nodes[node_name] = builder(**node_kwargs)
        nodes[node_name] = wrap_node(nodes[node_name], node_name)
        graph.add_node(node_name, nodes[node_name])

    # Add edges between nodes in the graph
    for i in range(len(graph_definition) - 1):
        graph.add_edge(graph_definition[i], graph_definition[i + 1])

    # Connect the start and end nodes
    graph.add_edge(START, graph_definition[0])
    graph.add_edge(graph_definition[-1], END)

    # Compile the graph with the provided checkpoint saver
    return graph.compile(checkpointer=checkpoint_saver)
