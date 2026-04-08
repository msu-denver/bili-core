# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BiliCore is an open-source framework for benchmarking and building dynamic RAG (Retrieval-Augmented Generation) implementations. It enables rapid testing of LLMs across different cloud providers (AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI) and local environments.

## Development Commands

### Container-Based Development (Recommended)
```bash
# Start development environment
cd scripts/development
./start-container.sh  # or .bat on Windows

# Attach to container
./attach-container.sh

# Inside container
streamlit  # Start Streamlit UI
flask      # Start Flask API
```

### Code Quality Commands
```bash
# Run all formatters and linting (must pass before committing)
./run_python_formatters.sh

# Individual commands
black bili/                    # Format code
autoflake --recursive bili/    # Remove unused imports
isort --profile=black bili/    # Sort imports  
pylint bili/ --fail-under=9   # Lint (requires 9/10 score)
```

### Testing
```bash
# Run tests (inside container or with dependencies installed)
pytest tests/
```

### Build and Install
```bash
# Build package
cd scripts/build
./build-and-test.sh

# Install locally for development
pip install -e .
```

## High-Level Architecture

### Core Components

1. **Authentication System** (`bili/auth/`): Modular authentication with Firebase, SQLite, and in-memory providers. Each provider implements a common interface.

2. **Checkpointers** (`bili/iris/checkpointers/`): State persistence layer supporting MongoDB, PostgreSQL, and memory storage. All checkpointers implement a queryable interface for conversation management with both sync and async APIs.

3. **LLM Configuration** (`bili/iris/config/`): Model configurations for 60+ LLMs across AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, and local models. Uses factory pattern for model initialization.

4. **Tools Framework** (`bili/iris/tools/`): Extensible tool system including FAISS vector search, OpenSearch, weather APIs, and web search. Tools are dynamically loaded based on configuration.

5. **LangGraph Workflows** (`bili/iris/loaders/`, `bili/iris/nodes/`, `bili/iris/graph_builder/`): Node-based workflow system with a registry pattern. Default pipeline: persona → datetime → per_user_state → react agent → timestamp → memory management → normalization. Custom nodes can be registered dynamically.

6. **Middleware System** (`bili/iris/loaders/middleware_loader.py`, `bili/iris/config/middleware_config.py`): Extensible middleware framework for intercepting and modifying agent execution. Supports built-in middleware (summarization, model call limiting) and custom middleware creation.

### Key Patterns

- **Provider Pattern**: Consistent interfaces across auth, LLM, and tool providers
- **Registry Pattern**: Dynamic node and tool registration for workflow customization
- **Factory Pattern**: Model and checkpointer initialization based on configuration
- **Async/Sync Dual APIs**: Both synchronous and asynchronous interfaces throughout

### Dependencies and Configuration

- **Python 3.11+** required
- **LangChain/LangGraph** for LLM orchestration
- **Environment Variables**: `.env` file in project root (Docker Compose reads via `env_file: .env`). Copy `.env.example` to `.env` and fill in API keys. Shared variables are compatible with sustainability-hub-engine's `.env`.
  - AWS credentials in `env/bili_root/.aws/`
  - Google credentials in `env/bili_root/.google/`
- **Databases**: PostgreSQL with PostGIS and MongoDB for state management
- **Pre-commit hooks** enforce code quality standards

### Container Aliases

When you attach to the development container, `wait.sh` seeds `.bashrc` with these aliases:

- `streamlit` — Install deps, create PG database, start Streamlit UI (port 8501) with tee to `/tmp/streamlit.log`
- `flask` — Install deps, create PG database, start Flask API (port 5001)
- `deps` — Run `install-deps` script to install/update Python dependencies
- `cleandeps` — Clean reinstall of dependencies (runs install-deps with CLEAN=true)
- `seeds3` — Upload data files to LocalStack S3
- `createpgdb` — Create the LangGraph PostgreSQL database
- `awslocal` — AWS CLI pointing to LocalStack endpoint

The venv is automatically activated on attach.

### Authentication

- **Local development**: Uses SQLite provider. New accounts automatically get `researcher` role (no admin approval needed). Profile DB at `PROFILE_DB_PATH` (default: `/root/.bili/profile.db`).
- **Production (AWS)**: Uses Firebase provider. Accounts require admin approval.
- Configure in `bili/streamlit_app.py` via `initialize_auth_manager(auth_provider_name=...)`.

### Unified Streamlit App

The Streamlit app (`bili/streamlit_app.py`) is a multi-page app using `st.navigation()` on port 8501:
- **AETHER Multi-Agent** (`/aether`) — MAS visualizer and chat (default page)
- **Single-Agent RAG** (`/bili`) — LLM comparison chatbot

Both pages require authentication. AETHER page content is rendered from `bili/aether/ui/page.py`.

### AETHER Multi-Agent System

AETHER (Agent Ecosystems for Testing, Hardening, Evaluation and Research) is a declarative multi-agent orchestration framework (`bili/aether/`):

- **7 workflow types**: sequential, hierarchical, supervisor, consensus, deliberative, parallel, custom
- **YAML configuration**: Define agents, channels, and workflows declaratively
- **6 communication protocols**: direct, broadcast, request-response, pub-sub, competitive, consensus
- **Pipeline sub-graphs**: Rich multi-node pipelines within agents with custom state fields
- **Streaming**: Sync/async streaming with `StreamEvent` objects via `MASExecutor`
- **bili-core integration**: Agents can inherit LLM config, tools, middleware, and checkpointers

For comprehensive AETHER documentation, see `bili/aether/README.md` and `bili/aether/docs/`.

### Middleware Configuration

Middleware can be applied at two levels: agent-level and tool-level.

#### Agent-Level Middleware

Middleware can be configured via `node_kwargs` when building the agent graph:

```python
from bili.iris.loaders.middleware_loader import initialize_middleware

# Initialize middleware
middleware = initialize_middleware(
    active_middleware=["summarization", "model_call_limit"],
    middleware_params={
        "summarization": {"max_tokens_before_summary": 4000, "messages_to_keep": 20},
        "model_call_limit": {"run_limit": 10}
    }
)

# Pass to agent via node_kwargs
node_kwargs = {
    "llm_model": my_llm,
    "tools": my_tools,
    "middleware": middleware
}
```

#### Tool-Level Middleware

Middleware can also be applied to individual tools for fine-grained control:

```python
from bili.iris.loaders.tools_loader import initialize_tools
from bili.iris.loaders.middleware_loader import initialize_middleware

# Initialize middleware
middleware = initialize_middleware(
    active_middleware=["model_call_limit"],
    middleware_params={"model_call_limit": {"run_limit": 5}}
)

# Apply middleware to specific tools (dict approach)
tools = initialize_tools(
    active_tools=["weather_api_tool", "serp_api_tool"],
    tool_prompts={
        "weather_api_tool": "Get weather data",
        "serp_api_tool": "Search the web"
    },
    tool_params={
        "weather_api_tool": {"api_key": "your_key"}
    },
    tool_middleware={
        "weather_api_tool": middleware,  # Apply to specific tool
        "serp_api_tool": []  # No middleware
    }
)

# Or apply same middleware to all tools (list approach)
tools = initialize_tools(
    active_tools=["weather_api_tool", "serp_api_tool"],
    tool_prompts={
        "weather_api_tool": "Get weather data",
        "serp_api_tool": "Search the web"
    },
    tool_middleware=middleware  # Apply to all tools
)
```

### Graph Building and Node Architecture

#### Node Definition Pattern

Nodes are defined using `functools.partial` to create node factories:

```python
from functools import partial
from bili.iris.graph_builder.classes.node import Node

def build_my_node(**kwargs):
    def _execute_node(state: dict) -> dict:
        # Node logic here
        return {"messages": updated_messages}
    return _execute_node

# Create a node factory using partial
my_node = partial(Node, "my_node_name", build_my_node)
```

#### Graph Definition

The default graph is defined in `bili/iris/loaders/langchain_loader.py`:

1. **Node Instantiation**: Node factories (partials) are called to create Node instances
2. **Property Configuration**: Node properties (edges, is_entry, routes_to_end) are set on instances
3. **Registry**: `GRAPH_NODE_REGISTRY` stores node factories (partials) for dynamic graph building
4. **Default Definition**: `DEFAULT_GRAPH_DEFINITION` contains pre-configured Node instances

```python
# Node instances are created from partials
persona_node_instance = persona_and_summary_node()

# Properties are set on instances
persona_node_instance.is_entry = True
persona_node_instance.edges.append("inject_current_datetime")

# Registry stores partials for dynamic creation
GRAPH_NODE_REGISTRY = {
    "add_persona_and_summary": persona_and_summary_node,  # Partial function
    ...
}
```

#### Custom Graph Modification

When modifying graphs (e.g., in Streamlit UI), **always use `copy.deepcopy`** to avoid mutating global node objects:

```python
import copy
from bili.iris.loaders.langchain_loader import DEFAULT_GRAPH_DEFINITION

# CORRECT: Deep copy to avoid mutations
graph_definition = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)
graph_definition[graph_definition.index("inject_current_datetime")].edges = ["per_user_state"]

# INCORRECT: Shallow copy causes mutations across requests
graph_definition = DEFAULT_GRAPH_DEFINITION.copy()  # DON'T DO THIS
```

#### Node Wrapper for Performance Monitoring

The `wrap_node` function in `langchain_loader.py` wraps each node to log execution time:

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

### Streaming Support

Both normal bili and AETHER support streaming and non-streaming agent interactions with clean, symmetric APIs.

#### Normal Bili — Framework-Agnostic Streaming (`bili/iris/loaders/streaming_utils.py`)

```python
from bili.iris.loaders.langchain_loader import build_agent_graph
from bili.iris.loaders.streaming_utils import stream_agent, invoke_agent, astream_agent

agent = build_agent_graph(checkpoint_saver=saver, node_kwargs=kwargs, ...)

# Non-streaming — returns the full response string
response = invoke_agent(agent, "What is the weather?", thread_id="user1")

# Sync streaming — yields tokens as they arrive
for token in stream_agent(agent, "What is the weather?", thread_id="user1"):
    print(token, end="", flush=True)

# Async streaming — for async contexts
async for token in astream_agent(agent, "Hello", thread_id="user1"):
    print(token, end="", flush=True)
```

Flask and Streamlit integrations delegate to these utilities:
- **Flask**: `handle_agent_prompt_stream()` in `flask_utils.py` returns SSE `text/event-stream`
- **Streamlit**: `process_query_streaming()` in `streamlit_query_handler.py` returns a generator for `st.write_stream()`

#### AETHER — MASExecutor Streaming (`bili/aether/runtime/executor.py`)

```python
from bili.aether.runtime import MASExecutor, StreamEventType

executor = MASExecutor(config)
executor.initialize()

# Non-streaming
result = executor.run(input_data)

# Sync streaming — yields StreamEvent objects
for event in executor.stream(input_data):
    if event.event_type == StreamEventType.TOKEN:
        print(event.data["content"], end="", flush=True)

# Async streaming — token-level via astream_events(v2)
async for event in executor.astream(input_data):
    if event.event_type == StreamEventType.TOKEN:
        print(event.data["content"], end="", flush=True)
```

AETHER streaming uses structured `StreamEvent` objects with `StreamFilter` for declarative event filtering.

### Pre-Commit Linting (REQUIRED)

**ALWAYS run formatters before committing code.** This is enforced automatically via Claude Code hooks in `.claude/settings.json`, but you should also run them proactively while writing code to catch issues early.

- `./run_python_formatters.sh` - Run all formatters (Black, Autoflake, Isort)
- `pylint bili/ --fail-under=9` - Check code quality (must score 9+/10)

Do not commit code that has lint errors. The CI pipeline will reject it.

### Important Notes

- Always run formatters before committing code
- Pylint score must be ≥9/10
- Use type hints throughout the codebase
- Follow existing patterns when adding new providers or tools
- Container development environment includes all dependencies and services
- Middleware flows through `node_kwargs` to the react agent node automatically
- **Node Architecture**: Nodes use `functools.partial` pattern; call them to create Node instances
- **Graph Mutations**: Always use `copy.deepcopy()` when modifying graph definitions to prevent cross-request mutations
- **Performance**: All nodes are automatically wrapped with execution time logging