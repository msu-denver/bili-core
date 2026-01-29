# LangGraph Workflow System

This document describes how BiliCore uses LangGraph to build dynamic, customizable agent workflows.

## Overview

BiliCore uses [LangGraph](https://langchain-ai.github.io/langgraph/) to create stateful, graph-based agent workflows. The framework provides:

- **Node-based architecture**: Modular processing steps that can be composed and reordered
- **Registry pattern**: Dynamic node registration for extensibility
- **State management**: Automatic checkpointing and persistence
- **Performance monitoring**: Execution time logging for all nodes

## Default Execution Pipeline

```mermaid
graph LR
    START([START]) --> A[add_persona_and_summary]
    A --> B[inject_current_datetime]
    B --> C[react_agent]
    C --> D[update_timestamp]
    D --> E[trim_summarize]
    E --> F[normalize_state]
    F --> END([END])
```

### Node Descriptions

| Node | Purpose |
|------|---------|
| `add_persona_and_summary` | Injects persona and conversation summary into system message |
| `inject_current_datetime` | Adds current UTC timestamp to system message |
| `react_agent` | Core ReAct agent that processes queries with tools |
| `update_timestamp` | Tracks message timing and delta between messages |
| `trim_summarize` | Memory management via trimming or summarization |
| `normalize_state` | Cleans up tool calls and removes invalid messages |

### Optional Nodes

| Node | Purpose |
|------|---------|
| `per_user_state` | Injects user-specific profile information |
| `prepare_llm_config` | Prepares LLM configuration from state |

## Node Architecture

### The `functools.partial` Pattern

Nodes are defined using `functools.partial` to create node factories. This pattern separates node definition from instantiation:

```python
from functools import partial
from bili.graph_builder.classes.node import Node

def build_my_node(**kwargs):
    """Factory function that returns the actual node logic."""
    def _execute_node(state: dict) -> dict:
        # Node processing logic here
        messages = state["messages"]
        # ... modify state ...
        return {"messages": updated_messages}
    return _execute_node

# Create a node factory (partial) - NOT an instance yet
my_node = partial(Node, "my_node_name", build_my_node)

# Call the factory to create a Node instance
my_node_instance = my_node()
```

### Node Class Definition

```python
@dataclass(eq=False)
class Node:
    """Defines a graph node."""
    name: str
    function: Callable  # The actual node function (not a builder)

    # Edges
    edges: List[str] = field(default_factory=list)
    conditional_edges: List[ConditionalEdge] = field(default_factory=list)

    # Terminal/Entry properties
    is_entry: bool = False
    routes_to_end: bool = False
    conditional_entry: Optional[ConditionalEdge] = None

    # Optional features
    cache_policy: Optional[Dict[str, Any]] = None
    return_type_annotation: Optional[str] = None
```

### Node Equality

Nodes use custom equality that compares by name:

```python
def __eq__(self, name):
    if isinstance(name, str):
        return name == self.name
    if isinstance(name, type(self)):
        return self.name == name.name
```

This allows list operations like:
```python
graph_definition[graph_definition.index("inject_current_datetime")].edges = ["per_user_state"]
```

## Graph Building

### The `build_agent_graph` Function

Located in `bili/loaders/langchain_loader.py`:

```python
def build_agent_graph(
    checkpoint_saver: BaseCheckpointSaver = None,
    custom_node_registry: dict[str, Node] = None,
    graph_definition: list[Node] = None,
    node_kwargs: dict = None,
    state: type = None,
) -> CompiledStateGraph:
```

**Parameters:**
- `checkpoint_saver`: State persistence handler (default: MemorySaver)
- `custom_node_registry`: Additional nodes to register
- `graph_definition`: List of Node instances defining the pipeline
- `node_kwargs`: Arguments passed to all node builders (llm_model, tools, etc.)
- `state`: State type definition (default: State from langgraph_utils)

### Basic Usage

```python
from bili.loaders.langchain_loader import build_agent_graph

# Build with defaults
agent = build_agent_graph()

# Invoke the agent
result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "user123"}}
)
```

### Advanced Usage with Custom Configuration

```python
from bili.loaders.langchain_loader import (
    build_agent_graph,
    DEFAULT_GRAPH_DEFINITION,
    GRAPH_NODE_REGISTRY
)
import copy

# CRITICAL: Deep copy to avoid mutating global state
custom_graph = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)

# Add per_user_state node to the pipeline
dt_node = custom_graph[custom_graph.index("inject_current_datetime")]
dt_node.edges = ["per_user_state"]  # Redirect datetime → per_user_state

user_state_node = custom_graph[custom_graph.index("per_user_state")]
user_state_node.edges.append("react_agent")  # per_user_state → react_agent

# Build with custom configuration
agent = build_agent_graph(
    checkpoint_saver=my_postgres_checkpointer,
    graph_definition=custom_graph,
    node_kwargs={
        "llm_model": my_llm,
        "tools": my_tools,
        "persona": "You are a sustainability expert",
        "current_user": {"uid": "user123", "email": "user@example.com"}
    },
    state=State
)
```

## Graph Registries

### GRAPH_NODE_REGISTRY

Maps node names to their factory functions (partials):

```python
GRAPH_NODE_REGISTRY = {
    "add_persona_and_summary": persona_and_summary_node,
    "inject_current_datetime": inject_current_datetime_node,
    "prepare_llm_config": prepare_llm_config_node,
    "react_agent": react_agent_node,
    "update_timestamp": update_timestamp_node,
    "trim_summarize": trim_summarize_node,
    "normalize_state": normalize_state_node,
    "per_user_state": per_user_state_node,
}
```

### DEFAULT_GRAPH_DEFINITION

Pre-configured Node instances with edges set:

```python
DEFAULT_GRAPH_DEFINITION = [
    persona_and_summary_node_instance,    # is_entry=True, edges=["inject_current_datetime"]
    inject_current_datetime_node_instance, # edges=["react_agent"]
    react_agent_node_instance,             # edges=["update_timestamp"]
    update_timestamp_node_instance,        # edges=["trim_summarize"]
    trim_summarize_node_instance,          # edges=["normalize_state"]
    normalize_state_node_instance,         # routes_to_end=True
]
```

## State Management

### State Schema

The default state extends LangGraph's AgentState:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    """Extended state for BiliCore agents."""
    messages: list  # Conversation history
    summary: str    # Conversation summary
    template_dict: dict  # Template variables for persona
    llm_config: dict     # LLM configuration
    # ... additional fields
```

### Checkpointing

State is automatically persisted via checkpointers:

```python
from bili.checkpointers.checkpointer_functions import get_checkpointer

# Auto-selects based on environment (Postgres, Mongo, or Memory)
checkpointer = get_checkpointer()

agent = build_agent_graph(checkpoint_saver=checkpointer)

# Each thread_id maintains separate conversation state
result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "user@example.com_conv123"}}
)
```

## Performance Monitoring

### Node Wrapping

All nodes are automatically wrapped with execution time logging:

```python
def wrap_node(node_func: Callable, node_name: str) -> Callable:
    """Wraps a node function to log its execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = node_func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000
        LOGGER.info(f"Node '{node_name}' executed in {execution_time:.2f} ms")
        return result
    return wrapper
```

Example log output:
```
INFO - Node 'add_persona_and_summary' executed in 1.23 ms
INFO - Node 'inject_current_datetime' executed in 0.45 ms
INFO - Node 'react_agent' executed in 2341.67 ms
INFO - Node 'update_timestamp' executed in 0.89 ms
INFO - Node 'trim_summarize' executed in 12.34 ms
INFO - Node 'normalize_state' executed in 0.56 ms
```

## Creating Custom Nodes

### Step 1: Define the Node Builder

```python
# bili/nodes/my_custom_node.py
from functools import partial
from bili.graph_builder.classes.node import Node
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

def build_my_custom_node(**kwargs):
    """
    Build a custom node that processes state.

    kwargs may include:
    - llm_model: The language model
    - tools: Available tools
    - persona: System persona string
    - current_user: User information dict
    """
    custom_param = kwargs.get("custom_param", "default_value")

    def _execute_node(state: dict) -> dict:
        messages = state["messages"]

        # Your custom processing logic here
        LOGGER.debug(f"Processing with custom_param: {custom_param}")

        # Modify messages or other state fields
        # ...

        return {"messages": messages}

    return _execute_node

# Create the node factory
my_custom_node = partial(Node, "my_custom_node", build_my_custom_node)
```

### Step 2: Register the Node

```python
from bili.loaders.langchain_loader import build_agent_graph
from my_nodes.my_custom_node import my_custom_node

# Add to custom registry
custom_registry = {
    "my_custom_node": my_custom_node
}

# Build graph with custom node
agent = build_agent_graph(
    custom_node_registry=custom_registry,
    graph_definition=my_custom_graph_definition
)
```

### Step 3: Add to Graph Definition

```python
import copy
from bili.loaders.langchain_loader import DEFAULT_GRAPH_DEFINITION

# Deep copy and modify
graph_def = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)

# Create and configure custom node instance
custom_node_instance = my_custom_node()
custom_node_instance.edges.append("react_agent")

# Insert into pipeline
graph_def.insert(2, custom_node_instance)  # After datetime, before react_agent

# Update previous node's edges
graph_def[1].edges = ["my_custom_node"]  # datetime → my_custom_node
```

## Important Patterns and Warnings

### Always Use `copy.deepcopy()`

**CRITICAL**: When modifying graph definitions, always deep copy to prevent cross-request mutations:

```python
# ✅ CORRECT: Deep copy
graph_definition = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)
graph_definition[0].edges = ["new_node"]

# ❌ INCORRECT: Shallow copy causes global mutations
graph_definition = DEFAULT_GRAPH_DEFINITION.copy()
graph_definition[0].edges = ["new_node"]  # This mutates the global!

# ❌ INCORRECT: Direct mutation
DEFAULT_GRAPH_DEFINITION[0].edges = ["new_node"]  # Never do this!
```

### Node kwargs Flow

`node_kwargs` passed to `build_agent_graph` are forwarded to all node builders:

```python
node_kwargs = {
    "llm_model": my_llm,
    "tools": my_tools,
    "persona": "You are helpful",
    "middleware": my_middleware,
    "current_user": user_data,
}

# Each node builder receives all kwargs
def build_my_node(**kwargs):
    llm = kwargs.get("llm_model")
    tools = kwargs.get("tools")
    # Access what you need, ignore the rest
```

### Conditional Edges

For complex routing based on state:

```python
from bili.graph_builder.classes.conditional_edge import ConditionalEdge

def route_by_tool_count(state):
    """Route based on whether tools were called."""
    if state.get("tool_calls"):
        return "process_tools"
    return "direct_response"

node_instance.conditional_edges.append(
    ConditionalEdge(
        routing_function=route_by_tool_count,
        path_map={
            "process_tools": "tool_processor_node",
            "direct_response": "normalize_state"
        }
    )
)
```

## Debugging

### Enable LangChain Debug Mode

Debug mode is auto-enabled when log level is DEBUG or lower:

```python
import logging
from bili.utils.logging_utils import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# This automatically sets langchain.debug = True
```

### View Graph Structure

```python
# After building
agent = build_agent_graph(...)

# Print graph structure
print(agent.get_graph().draw_ascii())

# Or export as Mermaid
print(agent.get_graph().draw_mermaid())
```

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Overall system architecture
- [TOOLS.md](./TOOLS.md) - Tools framework documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
