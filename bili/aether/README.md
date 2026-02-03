# AETHER: Agent Ecosystems for Testing, Hardening, Evaluation, and Research

AETHER is a domain-agnostic multi-agent system (MAS) framework built as an extension to bili-core. It enables declarative configuration of multi-agent systems for any domain - research, content moderation, code review, customer support, and more.

## Design Philosophy

AETHER uses a **domain-agnostic** design where agent roles and capabilities are **free-form strings**, not restrictive enums. This means:

- **Any role**: Define agents with any role you need (`researcher`, `security_engineer`, `content_reviewer`, etc.)
- **Any capabilities**: Specify any capabilities as strings (`web_search`, `code_analysis`, `policy_lookup`, etc.)
- **Presets for convenience**: Use built-in presets for common patterns without being restricted to them
- **Full flexibility**: The framework doesn't impose domain-specific constraints

## Quick Start

### Create an Agent (Domain-Agnostic)

```python
from bili.aether.schema import AgentSpec

# Any role and capabilities - not restricted to enums
agent = AgentSpec(
    agent_id="my_researcher",
    role="senior_researcher",  # Free-form string
    objective="Research and analyze AI developments",
    capabilities=["web_search", "document_analysis"],  # Free-form strings
    tools=["serp_api_tool"],
)
```

### Use Presets for Common Patterns

```python
from bili.aether.schema import create_agent_from_preset, list_presets

# See available presets
print(list_presets())
# ['content_reviewer', 'researcher', 'code_reviewer', 'supervisor', ...]

# Create from preset with optional overrides
agent = create_agent_from_preset(
    preset_name="researcher",
    agent_id="ai_researcher",
    objective="Research quantum computing advances",
    temperature=0.6,  # Override preset default
)
```

### Register Custom Presets

```python
from bili.aether.schema import register_preset, create_agent_from_preset

# Register your own preset
register_preset("my_custom_agent", {
    "role": "domain_expert",
    "capabilities": ["specialized_analysis", "reporting"],
    "temperature": 0.3,
})

# Use it
agent = create_agent_from_preset(
    preset_name="my_custom_agent",
    agent_id="expert_1",
    objective="Provide domain expertise",
)
```

## Example Workflows

AETHER includes example YAML configurations for multiple domains:

### Content Moderation
- `simple_chain.yaml` - Sequential review pipeline
- `hierarchical_voting.yaml` - Tiered debate and voting
- `supervisor_moderation.yaml` - Supervisor-routed specialists
- `consensus_network.yaml` - Peer deliberation
- `custom_escalation.yaml` - Human-in-the-loop

### Research (Domain-Agnostic)
- `research_analysis.yaml` - Research → Fact-check → Analyze → Synthesize

### Software Engineering (Domain-Agnostic)
- `code_review.yaml` - Lead engineer routes to security/performance/style reviewers

## Architecture

```
bili/aether/
├── __init__.py           # Package exports
├── README.md             # This file
├── schema/               # MAS configuration schemas
│   ├── agent_spec.py     # Domain-agnostic AgentSpec
│   ├── agent_presets.py  # Preset registry system
│   ├── mas_config.py     # MASConfig, Channel, WorkflowEdge
│   └── enums.py          # Structural enums only (WorkflowType, etc.)
├── config/               # YAML loading and examples
│   ├── loader.py         # load_mas_from_yaml, load_mas_from_dict
│   └── examples/         # Example YAML configurations
└── tests/                # Test suite
```

## Integration with bili-core

AETHER agents inherit all bili-core capabilities:

### Inherited from bili-core:
- LLM model selection and configuration
- System prompt customization
- Tool access (FAISS, Weather APIs, SerpAPI, etc.)
- State persistence (MongoDB/PostgreSQL checkpointing)
- Memory management strategies
- Authentication (Firebase, In-Memory)

### Added by AETHER:
- Multi-agent orchestration patterns
- Agent-to-agent communication channels
- Workflow type support (sequential, hierarchical, supervisor, consensus)
- Domain-agnostic agent configuration
- Preset system for common patterns

## Workflow Types

AETHER supports multiple workflow patterns:

| Type | Description | Use Case |
|------|-------------|----------|
| `sequential` | Chain of agents (A → B → C) | Step-by-step processing |
| `hierarchical` | Tiered voting with debate | Decision making with multiple perspectives |
| `supervisor` | Hub-and-spoke routing | Dynamic task delegation |
| `consensus` | Peer deliberation network | Collaborative decision making |
| `parallel` | Simultaneous execution | Independent parallel tasks |
| `custom` | User-defined edges | Complex custom workflows |

## Development

### Testing

```bash
# Run AETHER tests in isolation (avoids heavy bili-core dependencies)
python bili/aether/tests/run_tests.py -v
```

### Code Style

- Follow bili-core's existing style (Black formatting, pylint compliance)
- Use type hints for all function signatures
- Document all public APIs with docstrings

## Contributing

AETHER is part of ongoing research at MSU Denver. For questions or contributions, please refer to the main bili-core repository guidelines.

## License

MIT License - See LICENSE file in the repository root.

---

**AETHER Version:** 0.1.0
**Last Updated:** February 2, 2026
**bili-core Version Required:** ≥3.0.0
