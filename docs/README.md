# BiliCore Documentation

Welcome to the BiliCore documentation. This guide provides comprehensive information for developers working with the BiliCore framework.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)
- [Project Structure](#project-structure)
- [Development Commands](#development-commands)
- [Additional Resources](#additional-resources)

---

## Overview

**BiliCore** is an open-source framework for benchmarking and building dynamic RAG (Retrieval-Augmented Generation) implementations. It enables rapid, reproducible testing of Large Language Models (LLMs) across different cloud providers and local environments without requiring researchers to run models locally.

Developed as part of the [Colorado Sustainability Hub](https://sustainabilityhub.co/) initiative, BiliCore provides a flexible and extensible core library built with **LangChain** and **LangGraph**. The framework supports research funded by the [National Science Foundation (NSF)](https://www.nsf.gov/) and the [NAIRR Pilot](https://nairrpilot.org/).

### What Makes BiliCore Different

- **Multi-Provider Support**: Test LLMs across AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, and local models using a unified interface
- **Dynamic Configuration**: Switch LLMs mid-conversation without losing chat history
- **Modular Architecture**: Extensible authentication, tools, checkpointing, and workflow systems
- **Research-Ready**: Built for benchmarking with reproducible configurations and comprehensive logging

---

## Key Features

| Feature | Description |
|---------|-------------|
| **60+ LLM Configurations** | Pre-configured models across major cloud providers |
| **LangGraph Workflows** | Node-based workflow system with customizable execution pipelines |
| **Extensible Tools** | FAISS, OpenSearch, weather APIs, web search, and custom tool support |
| **Flexible State Management** | MongoDB, PostgreSQL, and in-memory checkpointers with query APIs |
| **Modular Authentication** | Firebase, SQLite, and in-memory auth providers |
| **Dual Interfaces** | Streamlit UI for interactive testing, Flask API for programmatic access |
| **Memory Management** | Trimming and summarization strategies for context optimization |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (provided in Docker container)
- Cloud credentials for desired LLM providers

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/msu-denver/bili-core.git
cd bili-core

# Copy and configure secrets
cp scripts/development/secrets.template scripts/development/secrets
# Edit scripts/development/secrets with your API keys
```

### 2. Set Up Credentials

Place your cloud credentials in the appropriate directories:

```
env/bili_root/.aws/         # AWS credentials (config, credentials)
env/bili_root/.google/      # Google Cloud credentials (JSON key file)
```

### 3. Start Development Environment

```bash
# Start Docker containers
cd scripts/development
./start-container.sh    # Linux/macOS
# or
start-container.bat     # Windows

# Attach to the development container
./attach-container.sh   # Linux/macOS
# or
attach-container.bat    # Windows
```

### 4. Launch the Application

Inside the container:

```bash
# Start Streamlit UI (interactive benchmarking)
streamlit

# Or start Flask API (programmatic access)
flask
```

### 5. Access the Application

- **Streamlit UI**: http://localhost:8501
- **Flask API**: http://localhost:5000

---

## Documentation Index

Detailed documentation is organized by topic:

### [ARCHITECTURE.md](./ARCHITECTURE.md)

System architecture and code organization reference.

- Core component overview (auth, checkpointers, config, tools, loaders)
- Directory structure and module responsibilities
- Key design patterns (Provider, Registry, Factory)
- State management and data flow
- Configuration files and environment setup

### [LANGGRAPH.md](./LANGGRAPH.md)

LangGraph workflow system documentation.

- Default execution pipeline and node descriptions
- Node architecture using `functools.partial` pattern
- Graph definition and customization
- Custom node creation and registration
- Performance monitoring and execution logging
- Graph mutation safety (deep copy requirements)

### [TOOLS.md](./TOOLS.md)

Tools framework and extension guide.

- Available tools (FAISS, OpenSearch, weather APIs, SERP, mock tool)
- Tool registry and initialization patterns
- Creating custom tools
- Middleware integration for tools
- Tool configuration and prompts

### [STREAMLIT.md](./STREAMLIT.md)

Streamlit UI documentation.

- Application structure and components
- Authentication flow
- Chat interface and configuration panels
- Session state management
- Customizing the UI

---

## Project Structure

```
bili-core/
├── bili/                    # Main source package
│   ├── auth/               # Authentication providers
│   ├── checkpointers/      # State persistence (MongoDB, PostgreSQL, memory)
│   ├── config/             # LLM and tool configurations
│   ├── flask_api/          # Flask REST API components
│   ├── graph_builder/      # LangGraph building utilities
│   ├── loaders/            # Component initialization
│   │   ├── langchain_loader.py    # Graph building and node registration
│   │   ├── llm_loader.py          # LLM initialization
│   │   ├── tools_loader.py        # Tool initialization
│   │   └── middleware_loader.py   # Middleware configuration
│   ├── nodes/              # LangGraph node implementations
│   ├── prompts/            # System prompts and templates
│   ├── streamlit_ui/       # Streamlit UI components
│   ├── tools/              # Tool implementations
│   └── utils/              # Utility functions
├── docs/                   # Documentation (you are here)
├── env/                    # Environment configurations
├── scripts/                # Development and build scripts
├── tests/                  # Test suite
├── CLAUDE.md              # AI assistant development guide
├── README.MD              # Project overview
└── requirements.txt       # Python dependencies
```

---

## Development Commands

For a complete reference of development commands, see the [CLAUDE.md](../CLAUDE.md) in the project root.

### Essential Commands

```bash
# Code Quality (must pass before committing)
./run_python_formatters.sh          # Run all formatters and linting

# Individual Quality Commands
black bili/                          # Format code
isort --profile=black bili/          # Sort imports
pylint bili/ --fail-under=9          # Lint (requires 9/10 score)

# Testing
pytest tests/                        # Run test suite

# Building
cd scripts/build && ./build-and-test.sh   # Build package
pip install -e .                          # Install for development
```

### Code Quality Standards

- **Black** for code formatting
- **isort** for import sorting
- **Pylint** score must be 9/10 or higher
- **Type hints** throughout the codebase
- **Pre-commit hooks** enforce standards automatically

---

## Additional Resources

### External Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Project Links

- [GitHub Repository](https://github.com/msu-denver/bili-core)
- [Colorado Sustainability Hub](https://sustainabilityhub.co/)
- [NSF Award Information](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2318730)

### Getting Help

- Check existing documentation in this folder
- Review the [CLAUDE.md](../CLAUDE.md) for development patterns
- Open an issue on GitHub for bugs or feature requests

---

## Contributing

Contributions are welcome. Please ensure:

1. Code passes all formatters and linting (`./run_python_formatters.sh`)
2. Pylint score is 9/10 or higher
3. Tests pass (`pytest tests/`)
4. Type hints are included for new code
5. Documentation is updated for significant changes

See the [pull request template](../.github/pull_request_template.md) for submission guidelines.

---

*This project is funded by the National Science Foundation (NSF) and the National Artificial Intelligence Research Resource (NAIRR) Pilot.*
