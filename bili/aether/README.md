# AETHER - Agent Execution & Task Handling for Enhanced Reasoning

A **domain-agnostic** multi-agent system (MAS) framework for bili-core.

## Design Philosophy

AETHER is designed with three core principles:

### 1. Domain-Agnostic Configuration

Unlike the original design which tied agent types to content moderation roles (`CONTENT_REVIEWER`, `POLICY_EXPERT`, `JUDGE`), AETHER uses **free-form strings** for roles and capabilities:

```python
# Content moderation domain
agent = AgentSpec(
    agent_id="reviewer",
    role="content_reviewer",  # Free-form string
    objective="Review content for policy violations"
)

# Research domain - same framework, different domain
agent = AgentSpec(
    agent_id="researcher",
    role="research_analyst",  # Any role you need
    objective="Analyze scientific papers"
)

# Completely custom domain
agent = AgentSpec(
    agent_id="optimizer",
    role="quantum_circuit_optimizer",  # No restrictions!
    objective="Optimize quantum circuits for fidelity"
)
```

### 2. Registry-Based Preset System

Presets provide **convenience without restrictions**. They're optional starting points, not mandatory enums:

```python
from bili.aether import get_preset, register_preset

# Use a built-in preset
researcher = get_preset("researcher", agent_id="my_researcher")

# Override preset values
custom = get_preset(
    "researcher",
    agent_id="custom",
    temperature=0.8,
    tools=["custom_tool"]
)

# Register your own presets
register_preset("my_domain_expert", {
    "role": "domain_expert",
    "objective": "Provide domain expertise",
    "temperature": 0.3,
})
```

### 3. Structural Enums Only

The only enums in AETHER define **workflow structure**, not domain specifics:

- `WorkflowType`: SEQUENTIAL, PARALLEL, HIERARCHICAL, SUPERVISOR, CONSENSUS, etc.
- `OutputFormat`: TEXT, JSON, STRUCTURED
- `CommunicationProtocol`: DIRECT, BROADCAST, REQUEST_RESPONSE, etc.

## Quick Start

### Creating Agents

```python
from bili.aether import AgentSpec

# Direct creation
agent = AgentSpec(
    agent_id="my_agent",
    role="my_custom_role",
    objective="Do something specific and useful",
    temperature=0.5,
    capabilities=["skill_1", "skill_2"],
    tools=["serp_api_tool"],
)

# From preset with overrides
from bili.aether import get_preset

agent = get_preset(
    "researcher",
    agent_id="custom_researcher",
    temperature=0.8
)
```

### Creating Multi-Agent Systems

```python
from bili.aether import MASConfig, AgentSpec, WorkflowType

# Create agents
reviewer = AgentSpec(
    agent_id="reviewer",
    role="content_reviewer",
    objective="Review content for policy violations"
)
judge = AgentSpec(
    agent_id="judge",
    role="judge",
    objective="Make final moderation decision"
)

# Create MAS
mas = MASConfig(
    mas_id="moderation_pipeline",
    name="Content Moderation Pipeline",
    workflow_type=WorkflowType.SEQUENTIAL,
    agents=[reviewer, judge],
)

# Or use factory functions
from bili.aether.schema.mas_config import create_sequential_mas

mas = create_sequential_mas(
    mas_id="pipeline",
    name="My Pipeline",
    agents=[agent1, agent2, agent3],
)
```

### Using Factory Functions

```python
from bili.aether.schema.mas_config import (
    create_sequential_mas,
    create_supervisor_mas,
    create_consensus_mas,
)

# Sequential pipeline
pipeline = create_sequential_mas(
    mas_id="pipeline",
    name="Processing Pipeline",
    agents=[agent1, agent2, agent3],
)

# Supervisor pattern
support = create_supervisor_mas(
    mas_id="support",
    name="Support System",
    supervisor=triage_agent,
    specialists=[tech_agent, billing_agent],
)

# Consensus pattern
review = create_consensus_mas(
    mas_id="review",
    name="Peer Review",
    agents=[reviewer1, reviewer2, reviewer3],
    consensus_threshold=0.66,
)
```

## Available Presets

| Preset | Role | Description |
|--------|------|-------------|
| `researcher` | researcher | Research and analysis with web search |
| `analyst` | analyst | Data analysis and insights |
| `content_reviewer` | content_reviewer | Content policy evaluation |
| `moderator_judge` | judge | Final moderation decisions |
| `support_agent` | support_agent | Customer support |
| `escalation_handler` | escalation_handler | Handle escalated issues |
| `code_reviewer` | code_reviewer | Code quality review |
| `architect` | architect | System architecture |
| `supervisor` | supervisor | Task routing (is_supervisor=True) |
| `voter` | voter | Voting in consensus workflows |
| `advocate` | advocate | Argue a position |
| `summarizer` | summarizer | Summarize information |
| `validator` | validator | Validate against criteria |

## YAML Configuration

AETHER configurations can be serialized to/from YAML. See the `examples/` directory for complete examples:

- `content_moderation.yaml` - Content moderation pipeline
- `research_workflow.yaml` - Research analysis pipeline
- `customer_support.yaml` - Supervisor-based support system
- `consensus_review.yaml` - Peer code review with consensus

## Comparison with Original Design

| Aspect | Original | AETHER (Improved) |
|--------|----------|-------------------|
| Agent roles | `AgentRole` enum with 7 values | Free-form strings |
| Capabilities | `AgentCapability` enum | Free-form string list |
| Custom roles | Requires `role=CUSTOM` + `custom_role_name` | Just use any string |
| Adding new roles | Modify code (enum) | No code change needed |
| Domain support | Content moderation only | Any domain |
| Extensibility | Limited to enum values | Unlimited via strings/presets |

## Integration with bili-core

AETHER agents can inherit configuration from bili-core:

```python
agent = AgentSpec(
    agent_id="inherited_agent",
    role="assistant",
    objective="Use bili-core configuration",
    inherit_from_bili_core=True,
    inherit_system_prompt=False,  # Use own prompt
    inherit_tools=True,           # Use bili-core tools
    system_prompt="My custom prompt",
)
```

Agents can be converted to bili-core node kwargs:

```python
kwargs = agent.to_node_kwargs()
# Returns: {"agent_id": ..., "role": ..., "objective": ..., ...}
```

## Testing

```bash
# Run all AETHER tests
pytest bili/aether/tests/

# Run specific test file
pytest bili/aether/tests/test_agent_spec.py
pytest bili/aether/tests/test_presets.py
pytest bili/aether/tests/test_mas_config.py
```

## Architecture

```
bili/aether/
├── __init__.py           # Main exports
├── README.md             # This file
├── schema/
│   ├── __init__.py
│   ├── enums.py          # Structural enums only
│   ├── agent_spec.py     # AgentSpec with free-form roles
│   ├── agent_presets.py  # Registry-based presets
│   └── mas_config.py     # MASConfig and factories
├── tests/
│   ├── test_agent_spec.py
│   ├── test_presets.py
│   └── test_mas_config.py
└── examples/
    ├── content_moderation.yaml
    ├── research_workflow.yaml
    ├── customer_support.yaml
    └── consensus_review.yaml
```
