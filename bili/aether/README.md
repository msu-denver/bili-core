# AETHER: Agentic Testing for Hardened Execution and Resilience

AETHER is a multi-agent system (MAS) security testing framework built as an extension to bili-core. It enables researchers to declaratively configure multi-agent systems and test security vulnerabilities specific to LLM-based agent architectures.

## Overview

AETHER extends bili-core's single-agent conversational AI capabilities to support multi-agent orchestration with a focus on security testing. It inherits all bili-core features (LLM providers, tools, authentication, state persistence) while adding capabilities for:

- Declarative multi-agent system configuration
- Agent-to-agent communication protocols
- Security attack vector testing and metrics
- Transparent agent communication logging
- Checkpoint persistence vulnerability analysis

## Architecture

```
bili/aether/
├── __init__.py           # Package initialization
├── README.md             # This file
├── schema/               # MAS configuration data models
├── validation/           # MAS configuration validation
├── compiler/             # AETHER → LangGraph translation
├── communication/        # Agent messaging protocol
├── execution/            # MAS runtime controller
└── attacks/              # Security testing framework
```

## Development Phases

### Phase 1: Hardcoded MAS Configuration (Tasks 1-8)

Build core AETHER functionality using programmatic (Python/YAML) MAS configuration without UI.

**Completed:**

- [x] Task 1: Set up AETHER module structure

**In Progress:**

- [ ] Task 2: Define Agent Specification Schema
- [ ] Task 3: Define Hardcoded MAS Configuration Format
- [ ] Task 4: Build Static MAS Validation Engine
- [ ] Task 5: Build AETHER-to-LangGraph Compiler
- [ ] Task 6: Implement Static Agent Communication Protocol
- [ ] Task 7: Integrate bili-core Features into AETHER Agents
- [ ] Task 8: Build MAS Execution Controller

### Phase 2: Visual Graph Builder & Attack Framework (Tasks 9-17)

Add ReactFlow UI for visual MAS configuration and comprehensive attack testing capabilities.

**Future:**

- [ ] Task 9: Build ReactFlow Graph Visualization
- [ ] Task 10-17: Attack vector framework and UI enhancements

## Integration with bili-core

AETHER agents are bili-core agents with additional capabilities:

### Inherited from bili-core:

- LLM model selection and configuration
- System prompt customization
- Tool access (FAISS, Weather APIs, SerpAPI, etc.)
- State persistence (MongoDB/PostgreSQL checkpointing)
- Conversation state management
- Authentication (Firebase, In-Memory)
- Memory management strategies

### Added by AETHER:

- Agent objectives within MAS context
- Output format specification (not just chat)
- Agent-to-agent communication channels
- Security attack injection points
- Communication transparency logging
- Multi-agent orchestration

## Example Usage (Post Task 5)

```python
from bili.aether.schema import AgentSpec, AgentRole
from bili.aether.compiler import compile_mas_to_langgraph

# Define a simple 2-agent MAS
agents = [
    AgentSpec(
        agent_id="reviewer",
        role=AgentRole.REVIEWER,
        objective="Review content for policy violations",
        llm_model="claude-sonnet-4",
        tools=["faiss_search"],
    ),
    AgentSpec(
        agent_id="policy_expert",
        role=AgentRole.POLICY,
        objective="Provide policy guidance to reviewer",
        llm_model="gpt-4",
        tools=[],
    ),
]

# Compile to executable LangGraph
graph = compile_mas_to_langgraph(agents, channels=[...])

# Execute MAS
result = graph.invoke({"input": "User query here"})
```

## Research Focus

AETHER is designed to support cybersecurity research on multi-agent LLM systems, specifically:

1. **Prompt Injection Attacks**: Single and multi-agent injection scenarios
2. **Memory Poisoning**: Persistent malicious state across agent interactions
3. **Bias Inheritance**: Propagation of biased outputs through agent chains
4. **Agent Impersonation**: Identity spoofing in multi-agent communications
5. **Checkpoint Persistence**: Attack payload survival through state checkpointing
6. **Cross-Model Transferability**: Attack effectiveness across different LLM providers

## Development Guidelines

### Code Style

- Follow bili-core's existing code style (Black formatting, pylint compliance)
- Use type hints for all function signatures
- Document all public APIs with docstrings
- Write unit tests for all new functionality

### Integration Points

When developing AETHER modules, leverage existing bili-core infrastructure:

```python
# Use bili-core's LLM loaders
from bili.loaders.langchain_loader import load_langgraph_agent

# Use bili-core's checkpointers
from bili.checkpointers.mongodb_checkpoint import MongoDBCheckpointer

# Use bili-core's tool configuration
from bili.tools import faiss_memory_indexing, api_serp
```

### Testing

- Place tests in `tests/aether/` (parallel to module structure)
- Use pytest for all testing
- Aim for >80% code coverage
- Test against multiple LLM providers when possible

## Contributing

AETHER is part of ongoing cybersecurity research at MSU Denver. For questions or contributions, please refer to the main bili-core repository guidelines.

## License

MIT License - See LICENSE file in the repository root.

---

**AETHER Version:** 0.0.1  
**Last Updated:** January 26, 2026  
**bili-core Version Required:** ≥3.0.0
