"""
langchain_loader.py
--------------------

This module provides functions to load and initialize LangGraph components for building
dynamic REACT agents with customizable execution flows. It handles node instantiation,
graph construction, and performance monitoring through node execution wrappers.

Key Components:
---------------
1. **Node Instantiation**: Converts functools.partial node factories into Node instances
2. **Default Graph Configuration**: Pre-configured execution pipeline for standard workflows
3. **Node Registry**: Maps node names to their factory functions for dynamic graph building
4. **Performance Monitoring**: Automatic execution time logging via node wrappers

Default Execution Pipeline:
---------------------------
START → add_persona_and_summary → inject_current_datetime → react_agent →
update_timestamp → trim_summarize → normalize_state → END

Available Nodes:
----------------
- add_persona_and_summary: Injects persona and conversation summary into system message
- inject_current_datetime: Adds current UTC timestamp to system message
- per_user_state: Injects user-specific profile information (available but not in default pipeline)
- react_agent: Core ReAct agent that processes queries with tools
- update_timestamp: Tracks message timing and delta between messages
- trim_summarize: Memory management via trimming or summarization
- normalize_state: Cleans up tool calls and removes invalid messages

Global Variables:
-----------------
- DEFAULT_GRAPH_DEFINITION: List of pre-configured Node instances for standard pipeline
- GRAPH_NODE_REGISTRY: Dict mapping node names to their factory functions (partials)

Functions:
----------
- wrap_node(node_func, node_name):
    Wraps a node function to log its execution time in milliseconds.

- build_agent_graph(checkpoint_saver=None, custom_node_registry=None,
  graph_definition=None, node_kwargs=None, state=None):
    Builds and compiles a StateGraph for a LangGraph REACT agent using the provided
    or default graph definition. Supports custom nodes, dynamic configuration,
    and automatic performance monitoring.

Dependencies:
-------------
- difflib: For suggesting similar node names when validation fails
- logging: Provides logging functionality
- time: For execution time measurement in wrap_node
- typing: Type hints for function signatures
- langchain: Core LangChain functionality with debug mode support
- langgraph.checkpoint: State persistence through BaseCheckpointSaver
- langgraph.graph: StateGraph and CompiledStateGraph for workflow building
- bili.nodes.*: Individual node implementations (imported as partials)
- bili.graph_builder.classes.node: Node class definition
- bili.utils.langgraph_utils: State type definition
- bili.utils.logging_utils: Logger initialization

Usage:
------
Basic usage with default configuration:

```python
from bili.loaders.langchain_loader import build_agent_graph

# Build with default graph and in-memory checkpointer
agent = build_agent_graph()

# Invoke the agent
result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "1"}}
)
```

Advanced usage with custom configuration:

```python
from bili.loaders.langchain_loader import (
    build_agent_graph,
    DEFAULT_GRAPH_DEFINITION,
    GRAPH_NODE_REGISTRY
)
import copy

# Deep copy to avoid mutating global state
custom_graph = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)

# Modify graph structure
custom_graph[custom_graph.index("inject_current_datetime")].edges = ["per_user_state"]
custom_graph[custom_graph.index("per_user_state")].edges.append("react_agent")

# Build with custom configuration
agent = build_agent_graph(
    checkpoint_saver=my_postgres_checkpointer,
    graph_definition=custom_graph,
    node_kwargs={
        "llm_model": my_llm,
        "tools": my_tools,
        "persona": "You are a helpful assistant",
        "current_user": {"uid": "user123"}
    },
    state=State
)
```

Important Notes:
----------------
- Always use copy.deepcopy() when modifying DEFAULT_GRAPH_DEFINITION to prevent
  mutations across requests
- GRAPH_NODE_REGISTRY stores functools.partial objects, not Node instances
- All nodes are automatically wrapped with execution time logging
- Node factories in the registry accept **kwargs for dynamic configuration
- Custom nodes can be added via custom_node_registry parameter
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

from bili.nodes.add_persona_and_summary import persona_and_summary_node
from bili.nodes.inject_current_datetime import inject_current_datetime_node
from bili.nodes.normalize_state import normalize_state_node
from bili.nodes.per_user_state import per_user_state_node
from bili.nodes.react_agent_node import react_agent_node
from bili.nodes.trim_and_summarize import trim_summarize_node
from bili.nodes.update_timestamp import update_timestamp_node
from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger
from bili.graph_builder.classes.node import Node

# Get the logger for the module
LOGGER = get_logger(__name__)

# Get current logging log level
current_log_level = LOGGER.getEffectiveLevel()

# Set the langchain debug mode to True if log level is less than or equal to DEBUG
if current_log_level <= logging.DEBUG:
    langchain.debug = True

# Instantiate the node objects from the partial functions
persona_and_summary_node_instance = persona_and_summary_node()
inject_current_datetime_node_instance = inject_current_datetime_node()
react_agent_node_instance = react_agent_node()
update_timestamp_node_instance = update_timestamp_node()
trim_summarize_node_instance = trim_summarize_node()
normalize_state_node_instance = normalize_state_node()
per_user_state_node_instance = per_user_state_node()

# Construct the default graph by setting node properties
persona_and_summary_node_instance.is_entry = True
persona_and_summary_node_instance.edges.append("inject_current_datetime")
inject_current_datetime_node_instance.edges.append("react_agent")
react_agent_node_instance.edges.append("update_timestamp")
update_timestamp_node_instance.edges.append("trim_summarize")
trim_summarize_node_instance.edges.append("normalize_state")
normalize_state_node_instance.routes_to_end = True

# Define the default graph definition, which is the default set of nodes that will
# be used in the creation of a LangGraph REACT agent. Users can extend this
# definition with custom nodes.
DEFAULT_GRAPH_DEFINITION = [
    persona_and_summary_node_instance,
    inject_current_datetime_node_instance,
    react_agent_node_instance,
    update_timestamp_node_instance,
    trim_summarize_node_instance,
    normalize_state_node_instance,
]

# Define the graph node registry, which are nodes that are available to be used
# in the creation of a LangGraph REACT agent. Users can extend this registry with
# custom nodes.
GRAPH_NODE_REGISTRY = {
    "add_persona_and_summary": persona_and_summary_node,
    "inject_current_datetime": inject_current_datetime_node,
    "react_agent": react_agent_node,
    "update_timestamp": update_timestamp_node,
    "trim_summarize": trim_summarize_node,
    "normalize_state": normalize_state_node,
    "per_user_state": per_user_state_node,
}

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
    custom_node_registry: dict[str, Node] = None,
    graph_definition: list[Node] = None,
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
    for node in graph_definition:
        if node.name not in node_registry:
            suggestions = difflib.get_close_matches(node.name, node_registry.keys())
            suggestion_msg = (
                f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            )
            raise ValueError(
                f"Node '{node.name}' is not defined in the graph registry.{suggestion_msg}"
            )
        builder = node_registry[node.name]
        # Check if builder is a partial function (node factory) or a Node instance
        if callable(builder) and not isinstance(builder, Node):
            # It's a partial function - call it to get a Node, then get its function
            node_instance = builder()
            node_func = node_instance.function(**node_kwargs)
        else:
            # It's already a Node instance
            node_func = builder.function(**node_kwargs)
        # Wrap the node function to log execution time
        wrapped_node_func = wrap_node(node_func, node.name)
        nodes[node.name] = wrapped_node_func
        graph.add_node(node.name, nodes[node.name])
        
    # Build the edges and conditional edges
    for node in graph_definition:

        # Edges
        for edge in node.edges:
            graph.add_edge(node.name, edge)

        # Conditional edges
        for cond_edge in node.conditional_edges:
            graph.add_conditional_edges(node.name, cond_edge.routing_function, cond_edge.path_map)

        # Construct the start
        if node.is_entry:
            graph.add_edge(START, node.name)
         
        # Construct the start - conditional entry
        if node.conditional_entry:
          graph.add_conditional_edges(
              START,
              node.conditional_entry.routing_function,
              node.conditional_entry.path_map
          )

        # Construct the end
        if node.routes_to_end:
            graph.add_edge(node.name, END)

    return graph.compile(checkpointer=checkpoint_saver)
