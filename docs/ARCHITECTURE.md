# BiliCore Architecture

This document describes the architecture and organization of the BiliCore framework. It is written for new developers joining the project; if you are already familiar with the codebase, the [Table of Contents](#core-components) can help you jump to the section you need.

## Overview

BiliCore is an open-source framework for benchmarking and building dynamic RAG (Retrieval-Augmented Generation) implementations. It enables rapid testing of LLMs across different cloud providers (AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI) and local environments.

The codebase is split into **three major subsystems** plus a set of shared modules:

| Subsystem | Package | Responsibility |
|-----------|---------|----------------|
| **IRIS** | `bili/iris/` | Single-agent RAG orchestration -- LLM configs, LangGraph workflows, tools, checkpointers, and loaders |
| **AETHER** | `bili/aether/` | Multi-agent system (MAS) framework -- declarative YAML workflows, communication protocols, streaming execution |
| **AEGIS** | `bili/aegis/` | Security testing and evaluation -- adversarial attack runners, LLM evaluators, security scanners |

Shared modules (`bili/auth/`, `bili/utils/`, `bili/flask_api/`, `bili/streamlit_ui/`, `bili/prompts/`) are consumed by all three subsystems.

## Directory Structure

```
bili-core/
├── bili/                          # Main Python package
│   ├── iris/                      # IRIS: Single-agent RAG orchestration
│   │   ├── checkpointers/         #   State persistence layer
│   │   │   ├── migrations/        #     Schema migrations (Mongo, PostgreSQL)
│   │   │   ├── base_checkpointer.py
│   │   │   ├── mongo_checkpointer.py
│   │   │   ├── pg_checkpointer.py
│   │   │   └── memory_checkpointer.py
│   │   ├── config/                #   Configuration management
│   │   │   ├── llm_config.py      #     LLM model configurations (60+ models)
│   │   │   ├── tool_config.py     #     Tool configurations
│   │   │   └── middleware_config.py
│   │   ├── graph_builder/         #   LangGraph construction utilities
│   │   │   └── classes/           #     Node, ConditionalEdge classes
│   │   ├── loaders/               #   Component initialization
│   │   │   ├── langchain_loader.py  #   Graph builder & node registry
│   │   │   ├── tools_loader.py    #     Tool initialization & registry
│   │   │   ├── llm_loader.py      #     LLM initialization (factory pattern)
│   │   │   ├── embeddings_loader.py
│   │   │   └── middleware_loader.py
│   │   ├── nodes/                 #   LangGraph node implementations
│   │   │   ├── add_persona_and_summary.py
│   │   │   ├── inject_current_datetime.py
│   │   │   ├── per_user_state.py
│   │   │   ├── react_agent_node.py
│   │   │   ├── update_timestamp.py
│   │   │   ├── trim_and_summarize.py
│   │   │   └── normalize_state.py
│   │   └── tools/                 #   Tool implementations
│   │       ├── faiss_memory_indexing.py
│   │       ├── amazon_opensearch.py
│   │       ├── api_serp.py
│   │       ├── api_weather_gov.py
│   │       ├── api_open_weather.py
│   │       └── mock_tool.py
│   ├── aether/                    # AETHER: Multi-agent system framework
│   │   ├── runtime/               #   MASExecutor, streaming, events
│   │   ├── docs/                  #   Detailed AETHER documentation
│   │   └── ...                    #   Workflows, channels, agents, configs
│   ├── aegis/                     # AEGIS: Security testing & evaluation
│   │   ├── attacks/               #   Adversarial attack runners
│   │   ├── evaluator/             #   LLM output evaluators
│   │   ├── security/              #   Security scanning utilities
│   │   └── tests/                 #   AEGIS-specific tests
│   ├── auth/                      # Shared: Authentication system
│   │   └── providers/             #   Auth provider implementations
│   │       ├── auth/              #     Firebase, SQLite, In-memory
│   │       ├── role/              #     Role/permission providers
│   │       └── profile/           #     User profile providers
│   ├── flask_api/                 # Shared: Flask REST API
│   ├── streamlit_ui/              # Shared: Streamlit components
│   │   └── ui/                    #   UI modules
│   ├── prompts/                   # Shared: System prompts and templates
│   ├── utils/                     # Shared: Utility functions
│   ├── streamlit_app.py           # Streamlit entry point
│   └── flask_app.py               # Flask entry point
├── scripts/                       # Build and development scripts
│   ├── development/               #   Container scripts
│   └── build/                     #   Build scripts
├── env/                           # Environment configurations
├── data/                          # Data files (FAISS indexes, etc.)
├── models/                        # Local model files (symlink)
├── CLAUDE.md                      # AI assistant guidelines
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── docker-compose.yml             # Container orchestration
```

## Core Components

### 1. Authentication System (`bili/auth/`) -- Shared

The authentication system lives outside the three subsystems because it is shared by IRIS, AETHER, and AEGIS. It uses a provider-based architecture with pluggable implementations behind a common interface:

```
AuthManager
├── AuthProvider (Firebase, SQLite, In-memory)
├── ProfileProvider (user profile data)
└── RoleProvider (permissions/roles)
```

Each provider type has multiple implementations:
- **Firebase**: Production auth via Firebase Admin SDK (used in AWS deployments)
- **SQLite**: Local development with persistent storage (auto-grants `researcher` role)
- **In-memory**: Testing/ephemeral sessions (no persistence across restarts)

### 2. Checkpointers (`bili/iris/checkpointers/`) -- IRIS

Checkpointers are the state persistence layer for LangGraph agents. Every time a node in the graph executes, the current state (messages, summaries, metadata) is saved to a checkpoint so it can be resumed later. All checkpointers implement the `QueryableCheckpointerMixin` interface, which adds conversation-management queries on top of LangGraph's base checkpointer:

```python
class QueryableCheckpointerMixin(ABC):
    def get_user_threads(user_identifier, limit, offset) -> List[Dict]
    def get_thread_messages(thread_id, limit, offset, message_types) -> List[Dict]
    def delete_thread(thread_id) -> bool
    def get_user_stats(user_identifier) -> Dict
    def thread_exists(thread_id) -> bool
    def verify_thread_ownership(thread_id, user_identifier) -> bool
```

Available implementations:
- **PostgresSaver**: Production with PostGIS support
- **MongoDBSaver**: Document-based storage
- **MemorySaver**: In-memory for testing

#### Multi-Tenant Security

All checkpointers support multi-tenant isolation via the `user_id` parameter:

```python
from bili.iris.checkpointers.pg_checkpointer import AsyncPostgresSaver

# Initialize checkpointer with user_id for multi-tenant isolation
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string="postgresql://...",
    user_id="user@example.com"  # Enforces thread ownership validation
)
```

**Thread Ownership Validation:**
- Thread IDs must follow pattern: `{user_id}` or `{user_id}_{conversation_id}`
- Checkpointer validates ownership on all operations (get, put, delete)
- Raises `PermissionError` if thread doesn't belong to authenticated user
- Validation disabled when `user_id=None` (backward compatible)

**On-Demand Schema Migration:**
- Database schema changes occur only when `user_id` first provided
- PostgreSQL: Adds `user_id` column with index on first use
- MongoDB: Adds `user_id` field to documents on first use
- Zero downtime - migrations run automatically during checkpointer initialization

#### Multi-Conversation Support

Users can maintain multiple isolated conversation threads via `conversation_id`:

```python
# Default conversation (backward compatible)
config = {"configurable": {"thread_id": "user@example.com"}}

# Named conversations
config_work = {"configurable": {"thread_id": "user@example.com_work"}}
config_personal = {"configurable": {"thread_id": "user@example.com_personal"}}
```

**Thread ID Pattern:**
- Single conversation: `{user_id}` (e.g., `user@example.com`)
- Multi-conversation: `{user_id}_{conversation_id}` (e.g., `user@example.com_work`)
- Conversations are isolated - separate state, messages, and checkpoints

**Flask API Integration:**
```python
# Flask route with multi-conversation support
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    conversation_id = data.get("conversation_id")  # Optional
    return handle_agent_prompt(g.user, agent, data["prompt"], conversation_id)
```

#### Cloud-Ready State Management

Checkpointers provide cloud-native state persistence replacing file-based storage:

**Before (File-Based):**
- State stored in JSONL files on disk
- Lost when Kubernetes pods restart
- Not suitable for multi-instance deployments

**After (State-Based):**
- State persisted in PostgreSQL/MongoDB
- Survives pod restarts and scaling events
- Multi-instance safe with shared database backend
- Automatic state recovery on agent initialization

**Defense-in-Depth Security:**
- **Layer 1**: MASExecutor validates `user_id` and `conversation_id`
- **Layer 2**: Checkpointer validates thread ownership on every operation
- **Layer 3**: Database-level user isolation via indexed `user_id` column

**Backward Compatibility:**
- All security features are opt-in via `user_id` parameter
- Existing code without `user_id` continues to work unchanged
- No breaking changes to public APIs

### 3. LLM Configuration (`bili/iris/config/`) -- IRIS

The configuration module holds declarative metadata for every supported LLM model. Each entry describes the model's API identifier, which parameters it supports (temperature, top-p, seed, etc.), and provider-specific details. This metadata drives the Streamlit UI's dynamic parameter controls and the factory-pattern initialization in the loaders.

Configurations for 60+ LLMs across providers:

| Provider | Examples |
|----------|----------|
| AWS Bedrock | Claude 3/3.5, Llama, Mistral |
| Google Vertex AI | Gemini Pro/Flash |
| Azure OpenAI | GPT-4, GPT-4o |
| OpenAI | GPT-4, GPT-4o, o1 |
| Local | Ollama models |

Factory pattern initialization via `llm_loader.py`.

### 4. LangGraph Workflow (`bili/iris/loaders/`, `bili/iris/nodes/`) -- IRIS

The heart of single-agent RAG execution. The loaders module (`bili/iris/loaders/`) provides factory functions that wire together LLMs, tools, and checkpointers into a compiled LangGraph `StateGraph`. The nodes module (`bili/iris/nodes/`) contains the individual processing steps that make up the default pipeline. See [LANGGRAPH.md](./LANGGRAPH.md) for details.

**Default Pipeline:**
```
START → persona_summary → datetime → react_agent → timestamp → trim_summarize → normalize → END
```

### 5. Tools Framework (`bili/iris/tools/`) -- IRIS

Tools give agents the ability to call external services (weather APIs, search engines) or query internal data stores (FAISS, OpenSearch). Each tool is a LangChain `Tool` object created by a factory function in `bili/iris/loaders/tools_loader.py` and registered in the `TOOL_REGISTRY`. See [TOOLS.md](./TOOLS.md) for details.

**Available Tools:**
- FAISS vector search
- Amazon OpenSearch
- Weather APIs (OpenWeather, Weather.gov, Free Weather)
- SERP API (web search)
- Mock tool (testing)

### 6. Middleware System

Intercepts and modifies agent execution at two levels:
- **Agent-level**: Applied to entire conversation flow
- **Tool-level**: Applied to specific tool executions

Built-in middleware:
- `summarization`: Auto-summarize long conversations
- `model_call_limit`: Limit LLM invocations per turn

### 7. AETHER Multi-Agent System (`bili/aether/`)

AETHER (Agent Ecosystems for Testing, Hardening, Evaluation and Research) is a declarative multi-agent orchestration framework. It lets you define multiple cooperating agents, each with their own LLM, tools, and sub-graph, and wire them together using one of seven workflow types (sequential, hierarchical, supervisor, consensus, deliberative, parallel, custom). Configuration is done in YAML, and execution is handled by the `MASExecutor` class with sync and async streaming support.

For full documentation, see [`bili/aether/README.md`](../bili/aether/README.md) and [`bili/aether/docs/`](../bili/aether/docs/).

### 8. AEGIS Security Testing (`bili/aegis/`)

AEGIS provides adversarial testing and evaluation capabilities for LLM-based systems. It contains three sub-packages:

- **`bili/aegis/attacks/`**: Attack runners that generate adversarial prompts to test agent robustness (e.g., prompt injection, jailbreaking)
- **`bili/aegis/evaluator/`**: Evaluators that score LLM outputs for safety, accuracy, and compliance
- **`bili/aegis/security/`**: Security scanning utilities for detecting vulnerabilities in agent configurations

AEGIS was previously part of the AETHER package (as `bili.aether.attacks`, `bili.aether.evaluator`, `bili.aether.security`) and was extracted into its own top-level package to separate security concerns from multi-agent orchestration.

## Application Entry Points

### Streamlit Application (`streamlit_app.py`)

Interactive web UI for testing and configuration:

```python
def main():
    configure_streamlit()  # Page setup
    st.session_state.auth_manager = initialize_auth_manager(...)
    check_auth()  # Authentication
    checkpointer = get_checkpointer()
    run_app_page(checkpointer)  # Main UI
```

### Flask API (`flask_app.py`)

REST API for programmatic access and integration with other services.

## Design Patterns

### Provider Pattern
Consistent interfaces across auth, LLM, checkpointer, and tool providers enable swapping implementations without changing consuming code.

### Registry Pattern
Dynamic registration for nodes and tools:
```python
GRAPH_NODE_REGISTRY = {
    "add_persona_and_summary": persona_and_summary_node,
    "react_agent": react_agent_node,
    # ... extensible via custom_node_registry
}

TOOL_REGISTRY = {
    "faiss_retriever": lambda name, prompt, params: ...,
    "weather_api_tool": lambda name, prompt, params: ...,
    # ... extensible
}
```

### Factory Pattern
Model and checkpointer initialization based on configuration:
```python
llm = load_llm(provider="aws_bedrock", model_name="claude-3-5-sonnet")
checkpointer = get_checkpointer()  # Auto-selects based on environment
```

### Async/Sync Dual APIs
Both synchronous and asynchronous interfaces throughout for flexibility.

## Data Flow

```mermaid
graph TB
    User[User Input] --> Auth[Auth Manager]
    Auth --> |Authenticated| Streamlit[Streamlit UI]
    Auth --> |Authenticated| Flask[Flask API]

    Streamlit --> Loader[LangChain Loader]
    Flask --> Loader

    Loader --> Graph[StateGraph]
    Graph --> Nodes[Node Pipeline]

    Nodes --> |State| Checkpointer[Checkpointer]
    Nodes --> |Tool Calls| Tools[Tool Registry]
    Tools --> |External APIs| External[Weather, Search, etc.]
    Tools --> |Vector Search| FAISS[FAISS/OpenSearch]

    Checkpointer --> |Persist| Storage[(PostgreSQL/MongoDB)]
```

## Configuration

### Environment Variables
Key configuration via environment:
- `BILI_ENV`: Environment (local, development, production)
- `CHECKPOINTER_TYPE`: postgres, mongo, memory
- LLM provider credentials (AWS, Google, Azure, OpenAI)

### Configuration Files
- `.env.example`: Template for environment variables (copy to `.env`)
- `env/bili_root/.aws/`: AWS credentials
- `env/bili_root/.google/`: Google Cloud credentials

## Development Workflow

1. **Container Development** (recommended):
   ```bash
   cd scripts/development
   ./start-container.sh
   ./attach-container.sh
   streamlit  # Start Streamlit
   ```

2. **Code Quality**:
   ```bash
   ./run_python_formatters.sh  # Must pass before commit
   ```

3. **Testing**:
   ```bash
   pytest tests/
   ```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| langchain | LLM orchestration |
| langgraph | Workflow graphs |
| streamlit | Web UI |
| flask | REST API |
| psycopg | PostgreSQL |
| pymongo | MongoDB |
| faiss-cpu | Vector similarity |
| boto3 | AWS services |

## See Also

- [SECURITY.md](./SECURITY.md) - Multi-tenant security and cloud-ready features
- [LANGGRAPH.md](./LANGGRAPH.md) - LangGraph workflow documentation (IRIS)
- [TOOLS.md](./TOOLS.md) - Tools framework documentation (IRIS)
- [STREAMLIT.md](./STREAMLIT.md) - Streamlit UI documentation
- [bili/aether/README.md](../bili/aether/README.md) - AETHER multi-agent system
- [bili/aegis/](../bili/aegis/) - AEGIS security testing framework
- [../CLAUDE.md](../CLAUDE.md) - Development commands and patterns
