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

2. **Checkpointers** (`bili/checkpointers/`): State persistence layer supporting MongoDB, PostgreSQL, and memory storage. All checkpointers implement a queryable interface for conversation management with both sync and async APIs.

3. **LLM Configuration** (`bili/config/`): Model configurations for 60+ LLMs across AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, and local models. Uses factory pattern for model initialization.

4. **Tools Framework** (`bili/tools/`): Extensible tool system including FAISS vector search, OpenSearch, weather APIs, and web search. Tools are dynamically loaded based on configuration.

5. **LangGraph Workflows** (`bili/loaders/`, `bili/nodes/`): Node-based workflow system with a registry pattern. Default pipeline: persona → datetime → react agent → timestamp → memory management → normalization. Custom nodes can be registered dynamically.

6. **Middleware System** (`bili/loaders/middleware_loader.py`, `bili/config/middleware_config.py`): Extensible middleware framework for intercepting and modifying agent execution. Supports built-in middleware (summarization, model call limiting) and custom middleware creation.

### Key Patterns

- **Provider Pattern**: Consistent interfaces across auth, LLM, and tool providers
- **Registry Pattern**: Dynamic node and tool registration for workflow customization
- **Factory Pattern**: Model and checkpointer initialization based on configuration
- **Async/Sync Dual APIs**: Both synchronous and asynchronous interfaces throughout

### Dependencies and Configuration

- **Python 3.11+** required
- **LangChain/LangGraph** for LLM orchestration
- **Configuration Files**: 
  - Secrets in `scripts/development/secrets.template`
  - AWS credentials in `env/bili_root/.aws/`
  - Google credentials in `env/bili_root/.google/`
- **Databases**: PostgreSQL with PostGIS and MongoDB for state management
- **Pre-commit hooks** enforce code quality standards

### Middleware Configuration

Middleware can be applied at two levels: agent-level and tool-level.

#### Agent-Level Middleware

Middleware can be configured via `node_kwargs` when building the agent graph:

```python
from bili.loaders.middleware_loader import initialize_middleware

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
from bili.loaders.tools_loader import initialize_tools
from bili.loaders.middleware_loader import initialize_middleware

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

### Important Notes

- Always run formatters before committing code
- Pylint score must be ≥9/10
- Use type hints throughout the codebase
- Follow existing patterns when adding new providers or tools
- Container development environment includes all dependencies and services
- Middleware flows through `node_kwargs` to the react agent node automatically