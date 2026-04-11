<p align="center">
  <img src="bili/images/logo.png" alt="BiliCore Logo" width="150"/>
</p>

# BiliCore: An Open-Source LLM Framework

<p align="center">
  <a href="https://github.com/msu-denver/bili-core/actions"><img src="https://github.com/msu-denver/bili-core/actions/workflows/ci-cd.yml/badge.svg?branch=develop" alt="CI/CD"></a>
  <a href="https://github.com/msu-denver/bili-core/releases/latest"><img src="https://img.shields.io/github/v/release/msu-denver/bili-core" alt="Latest Release"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-6800%2B%20passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-89%25-brightgreen.svg" alt="Coverage">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-framework-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-orchestration-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Streamlit-UI-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Flask-API-lightgrey.svg" alt="Flask">
  <img src="https://img.shields.io/badge/Docker-containerized-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/PostgreSQL-state-336791.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/MongoDB-state-47A248.svg" alt="MongoDB">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AWS_Bedrock-supported-FF9900.svg" alt="AWS Bedrock">
  <img src="https://img.shields.io/badge/Google_Vertex_AI-supported-4285F4.svg" alt="Google Vertex AI">
  <img src="https://img.shields.io/badge/Azure_OpenAI-supported-0078D4.svg" alt="Azure OpenAI">
  <img src="https://img.shields.io/badge/OpenAI-supported-412991.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/Ollama-supported-000000.svg" alt="Ollama">
</p>

**BiliCore** is an open-source, domain-agnostic framework for building and testing LLM-powered applications. It provides single-agent orchestration, multi-agent system creation, and adversarial security testing in one modular package.

Developed as part of the [**Colorado Sustainability Hub**](https://sustainabilityhub.co/) initiative, funded by the **[National Science Foundation (NSF)](https://www.nsf.gov/)** and the **[NAIRR Pilot](https://nairrpilot.org/)**.

---

## Three Components

BiliCore is organized into three named components, each solving a distinct problem:

### IRIS — Interactive Reasoning and Integration Services
**Single-agent orchestration.** 60+ models across 6 providers.

IRIS bridges users to LLMs, tools, and data sources. It provides a node-based workflow pipeline where each step (persona injection, tool execution, memory management, response normalization) is a composable node. Switch models mid-conversation, configure tools on the fly, and persist state across sessions.

- **Providers:** AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, Ollama, local models
- **Tools:** FAISS vector search, OpenSearch, weather APIs, web search, extensible tool registry
- **Middleware:** Summarization, model call limiting, custom middleware
- **Checkpointers:** MongoDB, PostgreSQL, in-memory — all with queryable conversation management
- **Streaming:** Token-by-token responses via sync and async APIs
- **Location:** `bili/iris/`

### AETHER — Agent Ecosystems for Testing, Hardening, Evaluation, and Research
**Multi-agent orchestration.** Declarative YAML configuration.

AETHER lets you define multi-agent systems (MAS) in YAML and compile them into executable LangGraph workflows. Each agent can have its own LLM, tools, persona, and multi-node processing pipeline. Agents communicate through typed channels with configurable protocols.

- **7 workflow types:** Sequential, hierarchical, supervisor, consensus, parallel, deliberative, custom
- **6 communication protocols:** Direct, broadcast, request-response, pub-sub, competitive, consensus
- **Pipeline sub-graphs:** Multi-node pipelines within individual agents
- **Custom state fields:** Type-safe YAML state declarations with reducers and defaults
- **Runtime injection:** `RuntimeContext` container for dependency injection into pipeline nodes
- **Streaming:** `MASExecutor` with structured `StreamEvent` objects and `StreamFilter`
- **Location:** `bili/aether/`

### AEGIS — Adversarial Evaluation and Guarding of Intelligent Systems
**Security testing for multi-agent systems.** Built on AETHER.

AEGIS provides a systematic framework for testing how adversarial payloads propagate through multi-agent systems. It injects attacks at different phases (pre-execution, mid-execution, checkpoint), tracks propagation across agents, and evaluates compliance using a 3-tier detection system.

- **7 test suites:** Prompt injection, jailbreak, memory poisoning, bias inheritance, agent impersonation, persistence, cross-model transferability
- **3-tier detection:** Structural (CI-safe), heuristic (propagation tracking), semantic (LLM-based scoring)
- **Baseline comparison:** Ground-truth runner for controlled before/after analysis
- **Results viewer:** Interactive Streamlit dashboards for attack results and baseline analysis
- **Attack GUI:** Run adversarial attacks interactively with graph visualization
- **Location:** `bili/aegis/`

---

## Quick Start

### Prerequisites

- **Docker**: [Get Docker](https://docs.docker.com/get-docker/) — all services run in containers
- **Git**: To clone the repository

### 1. Clone and configure

```bash
git clone https://github.com/msu-denver/bili-core.git
cd bili-core
cp .env.example .env
# Edit .env with your API keys (AWS, Google, OpenAI, etc.)
```

### 2. Start the development environment

```bash
cd scripts/development
./start-container.sh
./attach-container.sh
```

This starts the bili-core container along with PostgreSQL (with PostGIS), MongoDB, and LocalStack services. The container automatically activates a Python virtual environment and sets up shell aliases.

### 3. Run the application

Inside the container:

```bash
streamlit    # Start the Streamlit UI on port 8501
flask        # Start the Flask API on port 5001
```

### 4. Access the application

- **Streamlit UI:** http://localhost:8501
  - `/aether` — AETHER Multi-Agent system (visualizer, chat, attack suite)
  - `/bili` — Single-Agent RAG testing interface
  - `/attack-results` — AEGIS attack results viewer
  - `/results` — Baseline results viewer
- **Flask API:** http://localhost:5001

---

## Architecture Overview

```
bili-core/
├── bili/
│   ├── iris/                  # IRIS: Single-agent orchestration
│   │   ├── loaders/           #   Graph builder, streaming, tool/middleware/LLM loaders
│   │   ├── nodes/             #   Pipeline nodes (persona, datetime, react agent, etc.)
│   │   ├── graph_builder/     #   Node and edge class definitions
│   │   ├── config/            #   LLM, tool, and middleware configurations
│   │   ├── tools/             #   Tool implementations (FAISS, OpenSearch, weather, etc.)
│   │   └── checkpointers/     #   State persistence (MongoDB, PostgreSQL, memory)
│   │
│   ├── aether/                # AETHER: Multi-agent orchestration
│   │   ├── schema/            #   MASConfig, AgentSpec, WorkflowType, Channel definitions
│   │   ├── compiler/          #   YAML → LangGraph compilation (graph builder, LLM resolver)
│   │   ├── runtime/           #   MASExecutor, streaming, communication state
│   │   ├── config/examples/   #   Example YAML configurations
│   │   ├── integration/       #   Checkpointer factory for MAS
│   │   ├── validation/        #   Static MAS validation engine
│   │   └── ui/                #   Streamlit pages (chat, visualizer, attack, results)
│   │
│   ├── aegis/                 # AEGIS: Adversarial security testing
│   │   ├── attacks/           #   Attack injector, propagation tracker, strategies
│   │   ├── evaluator/         #   Semantic evaluator, scoring rubrics
│   │   ├── security/          #   Security event detector, logger
│   │   └── tests/             #   7 attack suites + baseline + analysis
│   │
│   ├── auth/                  # Shared: Authentication (Firebase, SQLite, in-memory)
│   ├── utils/                 # Shared: Logging, LangGraph utilities, file I/O
│   ├── prompts/               # Shared: Prompt templates
│   ├── streamlit_ui/          # Shared: Streamlit UI components
│   ├── flask_api/             # Shared: Flask API utilities
│   ├── streamlit_app.py       # Unified Streamlit entry point
│   └── flask_app.py           # Flask API entry point
│
├── docs/                      # Project-level documentation
├── scripts/                   # Development and build scripts
├── .env.example               # Environment variable template
├── docker-compose.yml         # Full development stack
└── requirements.txt           # Python dependencies
```

---

## Code Examples

### IRIS: Single-Agent Streaming

```python
from bili.iris.loaders.langchain_loader import build_agent_graph
from bili.iris.loaders.streaming_utils import stream_agent, invoke_agent

agent = build_agent_graph(checkpoint_saver=saver, node_kwargs=kwargs)

# Non-streaming
response = invoke_agent(agent, "What is the weather?", thread_id="user1")

# Streaming — yields tokens as they arrive
for token in stream_agent(agent, "What is the weather?", thread_id="user1"):
    print(token, end="", flush=True)
```

### AETHER: Multi-Agent System

```python
from bili.aether import load_mas_from_yaml, compile_mas, execute_mas

config = load_mas_from_yaml("bili/aether/config/examples/simple_chain.yaml")
result = execute_mas(config, {"messages": ["Analyze quantum computing trends"]})
print(result.get_summary())
```

### AETHER: Streaming Multi-Agent

```python
from bili.aether.runtime import MASExecutor, StreamEventType

executor = MASExecutor(config)
executor.initialize()

for event in executor.stream(input_data):
    if event.event_type == StreamEventType.TOKEN:
        print(event.data["content"], end="", flush=True)
```

### AEGIS: Run a Security Test Suite

```bash
# Stub mode (no LLM calls — validates framework execution)
python bili/aegis/suites/injection/run_injection_suite.py --stub

# Full run (requires API credentials)
python bili/aegis/suites/injection/run_injection_suite.py

# Generate statistics report
python bili/aegis/suites/analysis/generate_stats.py
```

---

## Authentication

BiliCore provides three authentication providers:

| Provider | Use Case | Auto-Approve? | Configuration |
|----------|----------|---------------|---------------|
| **SQLite** | Local development (default) | Yes — `researcher` role | `PROFILE_DB_PATH` env var |
| **Firebase** | Production (AWS) | No — admin approval | Firebase credentials in `.env` |
| **In-Memory** | Testing | Yes | No configuration needed |

Configure in `bili/streamlit_app.py` via `initialize_auth_manager(auth_provider_name=...)`.

---

## Development

### Container Aliases

Inside the development container:

| Alias | Description |
|-------|-------------|
| `streamlit` | Install deps, create PG database, start Streamlit UI (port 8501) |
| `flask` | Install deps, create PG database, start Flask API (port 5001) |
| `deps` | Install/update Python dependencies |
| `cleandeps` | Clean reinstall of dependencies |
| `seeds3` | Upload data files to LocalStack S3 |
| `createpgdb` | Create the LangGraph PostgreSQL database |

### Code Quality

All code must pass formatters and linting before committing (enforced via pre-commit hooks):

```bash
./run_python_formatters.sh       # Run all formatters (Black, Autoflake, Isort)
pylint bili/ --fail-under=9      # Lint check (must score 9+/10)
```

### Running Tests

```bash
# Inside the container
pytest bili/iris/                  # IRIS unit tests
pytest bili/aether/tests/          # AETHER unit tests
pytest bili/aegis/suites/test_*.py  # AEGIS unit tests
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys. Docker Compose reads this file automatically.

- **AWS credentials:** `env/bili_root/.aws/`
- **Google credentials:** `env/bili_root/.google/`
- **API keys:** Set in `.env` (OpenAI, SerpAPI, weather APIs, etc.)

---

## Migration from v4.x to v5.0

v5.0 reorganizes the codebase into the three-component architecture. Import paths have changed:

| Old Path | New Path |
|----------|----------|
| `bili.loaders.*` | `bili.iris.loaders.*` |
| `bili.nodes.*` | `bili.iris.nodes.*` |
| `bili.graph_builder.*` | `bili.iris.graph_builder.*` |
| `bili.config.*` | `bili.iris.config.*` |
| `bili.tools.*` | `bili.iris.tools.*` |
| `bili.checkpointers.*` | `bili.iris.checkpointers.*` |
| `bili.aether.attacks.*` | `bili.aegis.attacks.*` |
| `bili.aether.evaluator.*` | `bili.aegis.evaluator.*` |
| `bili.aether.security.*` | `bili.aegis.security.*` |
| `bili.aether.tests.injection.*` | `bili.aegis.suites.injection.*` |
| *(other attack suites)* | `bili.aegis.suites.<suite>.*` |

**Unchanged paths:** `bili.aether.*` (schema, compiler, runtime, config, UI), `bili.auth.*`, `bili.utils.*`, `bili.prompts.*`

All functionality is preserved — only the locations changed.

---

## Component Documentation

- **IRIS:** See `bili/iris/` source code and inline documentation
- **AETHER:** [`bili/aether/README.md`](bili/aether/README.md) — comprehensive MAS documentation
- **AEGIS:** [`bili/aegis/docs/security-testing-quickstart.md`](bili/aegis/docs/security-testing-quickstart.md) — security testing guide

---

## Acknowledgments

This research is supported by the [National Science Foundation (NSF) (Grant No. 2318730)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2318730) and the [National Artificial Intelligence Research Resource (NAIRR) Pilot](https://nairrpilot.org/projects/awarded?_requestNumber=NAIRR240197). Their support has been instrumental in advancing AI accessibility and fostering innovation in sustainability-focused applications.

For more information, visit the [Sustainability Hub Website](https://sustainabilityhub.co/).
