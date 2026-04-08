<p align="center">
  <img src="bili/images/logo.png" alt="BiliCore Logo" width="150"/>
</p>

# BiliCore: A Framework for Benchmarking and Building Dynamic RAG Implementations

## Overview

**BiliCore** is an open-source, reusable framework designed to benchmark Large Language Models (LLMs) across cloud providers and local environments. It enables rapid, reproducible testing without requiring researchers to run models locally. Developed as part of the [**Colorado Sustainability Hub**](https://sustainabilityhub.co/). initiative, BiliCore contributes to the AI/ML community by providing a **flexible and extensible core library** built with **LangChain** and **LangGraph**. The framework emphasizes modular authentication, customizable tool selection, support for multiple LLMs across different providers, a robust web interface in Streamlit that enables on-the-fly RAG customization during a user's session, and a Flask REST API for programmatic access and backend integration. This allows users to start a conversation with one LLM and switch to another mid-session without losing conversation history.

Via the Streamlit UI, users can also test different personas using unique prompts, configure tools, adjust LLM parameters such as top-p, top-k, and seed values, and manage state with complete flexibility, including using truncation or summarization to preserve context when a customizable context window limit is reached.

This project is funded by the **[National Science Foundation (NSF)](https://www.nsf.gov/)** and the **[NAIRR Pilot](https://nairrpilot.org/)**, aligning with their mission to advance AI accessibility and sustainability.

---

## Key Features

### 1. **Benchmarking and RAG Configuration**

BiliCore empowers users to benchmark and customize RAG implementations through an intuitive interface:

- Select and test various **Large Language Models (LLMs)**
- Define **custom prompts** for conversational agents
- Adjust RAG parameters, such as model size and context window
- **Integrate external tools** for enhanced conversational capabilities
- **Manage chat history** and state persistence with MongoDB or PostgreSQL
- Optimize memory management strategies for efficient token usage

### 2. **Modular Authentication System**

BiliCore provides a flexible authentication system with **prebuilt authentication options**, including:

- **SQLite Authentication** (local development): Used by default. New accounts are automatically assigned the `researcher` role, so you can sign up and immediately sign in. The profile database is stored at the path set by `PROFILE_DB_PATH` (default: `/root/.bili/profile.db` inside the container).
- **Firebase Authentication** (production): Used for deployed environments on AWS. Requires admin approval before new accounts can access the platform.
- **In-Memory Authentication**: A simple in-memory authentication suitable for testing.

To configure authentication, modify the `initialize_auth_manager` call in `bili/streamlit_app.py` to specify the desired provider (e.g., `"sqlite"`, `"firebase"`, or `"in_memory"`).

### 3. **Extensible Tools Framework**

BiliCore allows **customizable tool selection**, including:

- **[FAISS](https://github.com/facebookresearch/faiss)** for vectorized similarity search
- **[OpenSearch](https://opensearch.org/)** for vectorized search in AWS-hosted OpenSearch
- **[OpenWeather API](https://openweathermap.org/)** for weather-related queries
- **[Weather.gov API](https://www.weather.gov/documentation/services-web-api)** for real-time weather data
- **[SerpAPI](https://serpapi.com/)** for web search and scraping
- **MockTool** for testing new tool APIs before deployment

To use these tools:

1. Copy `.env.example` to `.env` in the project root: `cp .env.example .env`
2. Fill in API keys for tools, cloud providers, and LLMs in the `.env` file.
3. For AWS services, store credentials in `env/bili_root/.aws/`.
4. For Google Cloud services, store credentials in `env/bili_root/.google/`.

### 4. **LangGraph-Powered Conversational AI**

BiliCore integrates **[LangGraph](https://www.langchain.com/langgraph)** to enable advanced conversational AI workflows:

- **Prebuilt multi-LLM capable RAG graph** for modularity and extensibility
- **Custom nodes** to extend graph functionality with personalized features
- **Flexible conversation flow control**, allowing structured interactions between LLMs

Users can extend the prebuilt LangGraph workflows by adding **custom nodes** in the `bili` module by modifying the call to `load_langgraph_agent` in `bili/loaders/langchain_loader.py`.

### 5. **Chat History and Memory Management**

Efficiently manages conversation states in LangGraph by persisting chat history with:

- **[MongoDB Checkpointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/)**: Supports flexible document-based state storage with async support
- **[PostgreSQL Checkpointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)**: Provides relational database storage for robust checkpointing with async support
- **Memory Checkpointer**: In-memory storage for testing and development
- **Trim Strategy**: Retains only the last `k` messages for simple memory management
- **Summarization Strategy**: Aggregates older messages into a coherent summary to optimize token usage

All checkpointers implement a unified queryable interface for conversation management:

- **Query conversations by user**: List all conversations for a specific user
- **Retrieve conversation history**: Get all messages from a thread
- **Delete conversations**: Remove conversation data
- **Get usage statistics**: Track conversation counts and message volumes
- **Check thread existence**: Validate thread IDs before operations

Both synchronous and asynchronous checkpointer APIs are supported for flexibility across different application architectures.

### 6. **Streaming Responses**

BiliCore supports streaming (token-by-token) responses across the entire stack. Both normal bili agents and AETHER multi-agent systems have clean, symmetric APIs for streaming and non-streaming interactions:

**Framework-Agnostic (works anywhere — no Flask/Streamlit required):**

```python
from bili.iris.loaders.langchain_loader import build_agent_graph
from bili.iris.loaders.streaming_utils import stream_agent, invoke_agent

agent = build_agent_graph(checkpoint_saver=saver, node_kwargs=kwargs, ...)

# Non-streaming
response = invoke_agent(agent, "What is the weather?", thread_id="user1")

# Streaming — yields tokens as they arrive
for token in stream_agent(agent, "What is the weather?", thread_id="user1"):
    print(token, end="", flush=True)
```

**AETHER Multi-Agent Streaming:**

```python
from bili.aether.runtime import MASExecutor, StreamEventType

executor = MASExecutor(config)
executor.initialize()

for event in executor.stream(input_data):
    if event.event_type == StreamEventType.TOKEN:
        print(event.data["content"], end="", flush=True)
```

- **Flask**: `/nova_stream` SSE endpoint for real-time token streaming
- **Streamlit**: "Enable streaming responses" toggle for token-by-token display
- **Async**: `astream_agent()` and `executor.astream()` for async contexts

### 7. Flask REST API for Programmatic Access

BiliCore includes an optional Flask API for applications that require stateless, token-based authentication and API-based access to the benchmarking framework.
With the Flask integration, you can:

- Authenticate users via JWT-based authentication
- Secure API routes using role-based access control
- Access LLMs via RESTful API endpoints, including **SSE streaming** via `/nova_stream`
- Use the same authentication providers (Firebase, In-Memory) as the Streamlit UI

### 8. AETHER: Multi-Agent System Framework

BiliCore includes **AETHER** (Agent Ecosystems for Testing, Hardening, Evaluation, and Research), a domain-agnostic framework for building and testing multi-agent systems (MAS).

**Key Capabilities:**

- **Declarative Configuration**: Define multi-agent systems using YAML or Python
- **7 Workflow Types**: Sequential, hierarchical, supervisor, consensus, parallel, deliberative, and custom
- **Pipeline Sub-Graphs**: Rich multi-node pipelines within single agents for complex processing flows
- **Agent Communication**: Built-in channels for direct, broadcast, request-response, pub-sub, competitive, and consensus protocols
- **Custom Node Registry**: Register custom pipeline nodes via `register_node()` or per-compilation `custom_node_registry`
- **Custom Pipeline State**: Declare custom state fields with type-safe YAML (`state_fields`) — supports reducers, defaults, and promotion to outer MAS state
- **Runtime Dependency Injection**: `RuntimeContext` container injects services (Celery, HTTP clients, model singletons) into pipeline node builders
- **Streaming**: Sync `.stream()` and async `.astream()` on `MASExecutor` with structured `StreamEvent` objects and `StreamFilter`
- **bili-core Integration**: Leverage checkpointers, middleware, and role registry from bili-core
- **Execution & Testing**: Built-in executor with checkpoint persistence and cross-model transfer testing
- **Comprehensive Validation**: Catches structural issues (circular dependencies, orphaned agents) before execution
- **Cloud-Ready Architecture**: State-based persistence survives Kubernetes pod restarts and supports multi-instance deployments
- **Multi-Tenant Security**: Thread ownership validation and user isolation across all checkpointer backends
- **Multi-Conversation Support**: Users can maintain multiple isolated conversation threads simultaneously

**Example Use Cases:**

- Research analysis workflows (gather → analyze → synthesize → review)
- Content moderation pipelines (detect → classify → escalate → decide)
- Code review systems (parse → analyze → suggest → approve)
- Multi-agent debate and consensus building
- Hierarchical decision-making with escalation

**Quick Example:**

```python
from bili.aether import load_mas_from_yaml, compile_mas, execute_mas

# Load configuration
config = load_mas_from_yaml("examples/research_analysis.yaml")

# Compile to LangGraph and execute
result = execute_mas(config, {"messages": ["Analyze quantum computing trends"]})
print(result.get_summary())
```

**Multi-Tenant & Multi-Conversation Example:**

```python
from bili.aether.execution.mas_executor import MASExecutor
from bili.iris.checkpointers.pg_checkpointer import AsyncPostgresSaver

# Initialize checkpointer with user_id for multi-tenant isolation
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string="postgresql://user:pass@localhost/bili",
    user_id="user@example.com"  # Enforces thread ownership
)

# Create executor for user's work conversation
executor = MASExecutor(
    mas_config=config,
    checkpointer=checkpointer,
    user_id="user@example.com",
    conversation_id="work"  # Isolated work conversation
)

# Execute with automatic state persistence
result = await executor.execute_async({"messages": ["Summarize Q4 results"]})

# Create separate personal conversation (isolated state)
personal_executor = MASExecutor(
    mas_config=config,
    checkpointer=checkpointer,
    user_id="user@example.com",
    conversation_id="personal"  # Separate conversation thread
)
```

For complete documentation, examples, and API reference, see [`bili/aether/README.md`](bili/aether/README.md).

---

## Installation

### Prerequisites

1. **Docker**: Ensure Docker is installed and running. [Get Docker](https://docs.docker.com/get-docker/)
2. **Python**: Requires Python 3.11+. Python is provided as part of the Docker Compose setup in the `bili-core` container.
3. **MongoDB/PostgreSQL**: A MongoDB or PostgreSQL instance is required for state persistence. This is provided as part of the Docker Compose setup.
4. **Cloud Credentials**:
   - AWS keys must be placed in `env/bili_root/.aws/`
   - Google Cloud credentials must be placed in `env/bili_root/.google/`

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/msu-denver/bili-core.git
   cd bili-core
   ```
2. **Install Dependencies**
   - To install BiliCore locally with pre-commit hooks, run:
     ```bash
     python setup.py install
     ```
     This installs all dependencies, sets up pre-commit hooks, and installs any git-based dependencies.
   - If you want to use BiliCore as a dependency in another project, add the following line to your requirements.txt file:
     ```bash
     git+https://github.com/msu-denver/bili-core.git@main
     ```
     Then install it with:
     ```bash
     pip install -r requirements.txt
     ```
3. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include API keys for the tools, cloud providers, and LLMs you plan to use. Docker Compose reads this file automatically and injects the variables into the container at startup.

4. **Start Docker Containers**
   ```bash
   cd scripts/development
   ./start-container.sh    # Linux/macOS (.bat on Windows)
   ```
   This starts the bili-core container along with PostgreSQL, MongoDB, and LocalStack services.

5. **Attach to Development Container**
   ```bash
   cd scripts/development
   ./attach-container.sh   # Linux/macOS (.bat on Windows)
   ```
   When you attach, the container automatically activates the Python virtual environment and sets up shell aliases for common tasks.

6. **Run BiliCore** (inside the container)
   ```bash
   streamlit    # Start the Streamlit UI (port 8501)
   flask        # Start the Flask API (port 5001)
   ```
   These are shell aliases that handle dependency installation, database setup, and server startup. The Streamlit app provides a unified multi-page interface with both the **AETHER Multi-Agent** system and the **Single-Agent RAG** testing interface.

   Other useful container aliases:
   ```bash
   deps         # Install/update Python dependencies
   cleandeps    # Clean reinstall of dependencies
   seeds3       # Upload data files to LocalStack S3
   createpgdb   # Create the LangGraph PostgreSQL database
   ```

7. **Access the Application**
   - **Streamlit UI**: http://localhost:8501 (AETHER at `/aether`, Single-Agent RAG at `/bili`)
   - **Flask API**: http://localhost:5001

## Acknowledgments

This research is supported by the [National Science Foundation (NSF) (Grant No. 2318730)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2318730) and the [National Artificial Intelligence Research Resource (NAIRR) Pilot](https://nairrpilot.org/projects/awarded?_requestNumber=NAIRR240197). Their support has been instrumental in advancing AI accessibility and fostering innovation in sustainability-focused applications.

For more information, please visit the [Sustainability Hub Website](https://sustainabilityhub.co/).
