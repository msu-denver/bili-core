# AETHER: Agent Ecosystems for Testing, Hardening, Evaluation, and Research

AETHER is a domain-agnostic multi-agent system (MAS) framework built as an extension to bili-core. It enables declarative configuration of multi-agent systems for any domain - research, content moderation, code review, customer support, and more.

## Design Philosophy

AETHER uses a **domain-agnostic** design where agent roles and capabilities are **free-form strings**, not restrictive enums. This means:

- **Any role**: Define agents with any role you need (`researcher`, `security_engineer`, `content_reviewer`, etc.)
- **Any capabilities**: Specify any capabilities as strings (`web_search`, `code_analysis`, `policy_lookup`, etc.)
- **Presets for convenience**: Use built-in presets for common patterns without being restricted to them
- **Full flexibility**: The framework doesn't impose domain-specific constraints

## Quick Start

### Minimal Agent (Required Fields Only)

An agent needs only three fields — `agent_id`, `role`, and `objective`:

```yaml
agents:
  - agent_id: reviewer
    role: content_reviewer
    objective: >
      Review content to ensure it follows community standards and
      recommend whether it should be allowed or removed.
```

```python
from bili.aether.schema import AgentSpec

agent = AgentSpec(
    agent_id="reviewer",
    role="content_reviewer",  # Free-form string — any role you need
    objective="Review content for community standards violations",
)
```

Everything else has sensible defaults: `temperature` is `0.0`, `output_format` is `text`, no tools or capabilities, no inheritance.

### Fully-Configured Agent (All Options)

```yaml
agents:
  - agent_id: content_reviewer
    role: content_reviewer
    objective: >
      Review content to ensure it follows the criteria put forth in
      Meta's community standards.  If it doesn't, recommend that the
      post be flagged as inappropriate and removed from the platform.

    # --- LLM Configuration ---
    model_name: gpt-4                # LLM model to use for this agent
    temperature: 0.2                  # Sampling temperature (0.0-2.0)
    max_tokens: 2048                  # Maximum tokens in LLM response

    # --- System Prompt ---
    system_prompt: >
      You are a content moderation specialist with expertise in Meta's
      community standards.  Analyze the provided content carefully and
      produce a JSON verdict with the following fields: decision
      (allow/remove), confidence (0.0-1.0), and reasoning (string).
      Be objective and cite specific policy sections when possible.

    # --- Capabilities & Tools ---
    capabilities:                     # Free-form strings — define your own
      - rag_retrieval
      - policy_lookup
      - tool_calling
    tools:                            # Tool names from the tool registry
      - faiss_retriever
      - serp_api_tool

    # --- Output Configuration ---
    output_format: structured         # text | json | structured
    output_schema:                    # Required when output_format is "structured"
      type: object
      properties:
        decision:
          type: string
          enum: [allow, remove]
        confidence:
          type: number
        reasoning:
          type: string

    # --- bili-core Inheritance ---
    inherit_from_bili_core: true       # Master toggle for inheritance
    inherit_llm_config: false          # Override: use model_name above instead
    inherit_tools: false               # Override: use tools list above instead
    inherit_system_prompt: false       # Override: use system_prompt above instead
    inherit_memory: true               # Inherit memory management from bili-core
    inherit_checkpoint: true           # Inherit checkpoint/state persistence

    # --- Workflow-Specific ---
    tier: 2                            # Agent tier (hierarchical workflows)
    voting_weight: 1.5                 # Weight in voting/consensus (default 1.0)
    is_supervisor: false               # Can this agent route to specialists?
    consensus_vote_field: decision     # Field name containing vote (consensus workflows)

    # --- Metadata ---
    metadata:                          # Arbitrary key-value pairs for custom use
      department: trust_and_safety
      version: "2.1"
```

### Agent Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `agent_id` | Yes | — | Unique identifier (alphanumeric, underscores, hyphens) |
| `role` | Yes | — | Agent's role (free-form string) |
| `objective` | Yes | — | What the agent should accomplish (10-1000 chars) |
| `model_name` | No | `None` | LLM model (e.g., `gpt-4`, `claude-sonnet-3-5-20241022`) |
| `temperature` | No | `0.0` | LLM sampling temperature (0.0-2.0) |
| `max_tokens` | No | `None` | Maximum tokens in LLM response |
| `system_prompt` | No | `None` | Instructions for the LLM (up to 10,000 chars) |
| `capabilities` | No | `[]` | Agent capabilities (free-form strings) |
| `tools` | No | `[]` | Tool names from the tool registry |
| `output_format` | No | `text` | Output format: `text`, `json`, or `structured` |
| `output_schema` | No | `None` | JSON schema (required when `output_format: structured`) |
| `inherit_from_bili_core` | No | `false` | Master toggle: inherit config from bili-core |
| `inherit_llm_config` | No | `true` | Inherit LLM model/temperature (when master toggle is on) |
| `inherit_tools` | No | `true` | Inherit tool configuration (when master toggle is on) |
| `inherit_system_prompt` | No | `true` | Inherit system prompt (when master toggle is on) |
| `inherit_memory` | No | `true` | Inherit memory management (when master toggle is on) |
| `inherit_checkpoint` | No | `true` | Inherit checkpoint persistence (when master toggle is on) |
| `tier` | No | `None` | Tier in hierarchical workflows (1 = highest authority) |
| `voting_weight` | No | `1.0` | Weight in voting/consensus workflows |
| `is_supervisor` | No | `false` | Whether agent can dynamically route to specialists |
| `consensus_vote_field` | No | `None` | Field name in output containing vote |
| `metadata` | No | `{}` | Arbitrary key-value pairs for custom use |

### Objective vs. System Prompt

Each agent has two distinct text fields that serve different purposes:

| Field | Purpose | Used by |
|-------|---------|---------|
| `objective` | **What** the agent should accomplish — its goal within the MAS | The orchestrator, to describe the agent's purpose to other agents and in workflow routing |
| `system_prompt` | **How** the agent should behave — instructions for the LLM itself | The LLM, as the system message that shapes tone, format, constraints, and reasoning approach |

The `objective` is required and describes the agent's role in the system. The `system_prompt` is optional and controls the LLM's behavior when executing that agent. When `system_prompt` is omitted, the agent inherits its prompt from bili-core (if `inherit_from_bili_core: true` and `inherit_system_prompt: true`) or uses the LLM's default behavior.

Each agent can also specify its own LLM via `model_name` — different agents in the same MAS can use different models (e.g., a fast model for initial screening and a more capable model for final judgment).

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

## Configuring a Multi-Agent System

A `MASConfig` brings agents, channels, and workflow settings together into a complete system. Like agents, a MAS can be as simple or as detailed as you need.

### Minimal MAS (Sequential Chain)

A sequential MAS needs only an ID, name, workflow type, agents, and channels:

```yaml
mas_id: simple_review
name: Simple Review Pipeline
workflow_type: sequential

agents:
  - agent_id: reviewer
    role: content_reviewer
    objective: Review content for policy violations

  - agent_id: judge
    role: judge
    objective: Make final moderation decision

channels:
  - channel_id: reviewer_to_judge
    protocol: direct
    source: reviewer
    target: judge
```

```python
from bili.aether.config.loader import load_mas_from_yaml, load_mas_from_dict

# From YAML file
config = load_mas_from_yaml("path/to/config.yaml")

# From Python dict
config = load_mas_from_dict({
    "mas_id": "simple_review",
    "name": "Simple Review Pipeline",
    "workflow_type": "sequential",
    "agents": [
        {"agent_id": "reviewer", "role": "content_reviewer",
         "objective": "Review content for policy violations"},
        {"agent_id": "judge", "role": "judge",
         "objective": "Make final moderation decision"},
    ],
    "channels": [
        {"channel_id": "reviewer_to_judge", "protocol": "direct",
         "source": "reviewer", "target": "judge"},
    ],
})
```

### Fully-Configured MAS (All Options)

```yaml
# --- Identity ---
mas_id: custom_escalation
name: Custom Content Moderation
description: >
  A custom workflow where flagged content is reviewed in parallel by
  a content reviewer and policy expert, whose verdicts are passed to
  a judge.  The judge either rules or escalates to an appeals specialist.
version: "1.0.0"

# --- Workflow Configuration ---
workflow_type: custom                    # sequential | hierarchical | supervisor |
                                         # consensus | deliberative | parallel | custom
entry_point: community_manager           # Starting agent (defaults to first agent)

# --- Agents ---
agents:
  - agent_id: community_manager
    role: community_manager
    objective: Flag content for review based on key words and intent
    temperature: 0.2
    capabilities: [tool_calling, inter_agent_communication]
    output_format: json

  - agent_id: content_reviewer
    role: content_reviewer
    objective: Review content against Meta's community standards
    model_name: gpt-4                    # Per-agent LLM selection
    temperature: 0.2
    capabilities: [rag_retrieval, policy_lookup, tool_calling]
    output_format: json

  - agent_id: policy_expert
    role: policy_expert
    objective: Review content against OpenAI's Usage Policies
    model_name: claude-sonnet-3-5-20241022  # Different model per agent
    temperature: 0.1
    capabilities: [policy_lookup, rag_retrieval]
    output_format: json

  - agent_id: judge
    role: judge
    objective: Evaluate reviewer inputs and decide or escalate
    temperature: 0.0
    capabilities: [inter_agent_communication]
    output_format: json

  - agent_id: appeals_specialist
    role: appeals_specialist
    objective: Act as tie-breaker when the judge cannot decide
    temperature: 0.3
    capabilities: [memory_access, policy_lookup, rag_retrieval]
    output_format: json

# --- Communication Channels ---
#
# Channels define how agents communicate.  The protocol determines the
# interaction pattern; bidirectional enables two-way exchange.
#
# The choice of protocols implicitly expresses collaboration type:
#   - consensus protocol  → cooperative collaboration
#   - competitive protocol → competitive collaboration
#   - mix of both         → coopetition
#
# The channel topology implicitly expresses communication structure:
#   - hub-and-spoke (one source, many targets) → centralized
#   - peer-to-peer (bidirectional mesh)        → decentralized
#   - tiered (tier N → tier N-1)               → hierarchical

channels:
  - channel_id: reviewer_to_judge
    protocol: direct                     # Point-to-point delivery
    source: content_reviewer
    target: judge
    description: Content reviewer passes verdict to judge

  - channel_id: policy_to_judge
    protocol: direct
    source: policy_expert
    target: judge
    description: Policy expert passes verdict to judge

  - channel_id: judge_to_appeals
    protocol: direct
    source: judge
    target: appeals_specialist
    description: Judge escalates undecided cases to appeals specialist

# --- Workflow Edges (custom workflows only) ---
#
# Edges define the execution graph.  Conditions use Python expressions
# evaluated against the workflow state.  This implicitly expresses
# communication strategy:
#   - condition expressions → rule-based strategy
#   - role-based routing    → role-based strategy
#   - LLM-driven decisions  → model-based strategy

workflow_edges:
  # Parallel fan-out from community manager
  - from_agent: community_manager
    to_agent: content_reviewer
    label: flag-content

  - from_agent: community_manager
    to_agent: policy_expert
    label: flag-policy

  # Both reviewers feed into judge
  - from_agent: content_reviewer
    to_agent: judge
    label: verdict

  - from_agent: policy_expert
    to_agent: judge
    label: verdict

  # Judge decides or escalates (rule-based strategy)
  - from_agent: judge
    to_agent: END
    condition: "state.confidence >= 0.7"
    label: decide

  - from_agent: judge
    to_agent: appeals_specialist
    condition: "state.confidence < 0.7"
    label: escalate

  # Appeals specialist makes final decision
  - from_agent: appeals_specialist
    to_agent: END
    label: final-decision

# --- Hierarchical Workflow Settings ---
hierarchical_voting: false               # Enable hierarchical voting pattern
min_debate_rounds: 0                     # Minimum debate rounds (competitive agents)

# --- Consensus Workflow Settings ---
consensus_threshold: null                # Fraction needed for consensus (e.g., 0.66)
max_consensus_rounds: 10                 # Maximum deliberation rounds before timeout
consensus_detection: majority            # majority | similarity | explicit | any

# --- Human-in-the-Loop ---
human_in_loop: true                      # Whether MAS can escalate to human review
human_escalation_condition: >-           # Python expression for when to escalate
  state.tie_breaker_needed or state.confidence < 0.5

# --- Checkpoint Configuration ---
checkpoint_enabled: true                 # Enable state persistence
checkpoint_config:
  type: sqlite                           # sqlite | mongodb | postgresql
  path: checkpoints.db

# --- Metadata ---
tags:                                    # Searchable labels (not schema-enforced)
  - content-moderation
  - custom
  - cooperative
  - role-based
  - decentralized
  - escalation
  - human-in-loop

metadata:                                # Arbitrary key-value pairs
  author: trust_and_safety_team
  environment: staging
```

### MAS Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `mas_id` | Yes | — | Unique MAS identifier (alphanumeric, underscores, hyphens) |
| `name` | Yes | — | Human-readable MAS name (1-200 chars) |
| `description` | No | `""` | Detailed description (up to 2000 chars) |
| `version` | No | `"1.0.0"` | Configuration version string |
| `agents` | Yes | — | List of `AgentSpec` definitions (minimum 1) |
| `channels` | No | `[]` | Communication channels between agents |
| `workflow_type` | Yes | — | Execution pattern (see Workflow Types table) |
| `entry_point` | No | First agent | Starting agent ID |
| `workflow_edges` | No | `[]` | Custom workflow edges with optional conditions |
| `hierarchical_voting` | No | `false` | Enable hierarchical voting pattern |
| `min_debate_rounds` | No | `0` | Minimum debate rounds for competitive agents |
| `consensus_threshold` | No* | `None` | Fraction of agents needed for consensus (0.0-1.0) |
| `max_consensus_rounds` | No | `10` | Maximum deliberation rounds before timeout |
| `consensus_detection` | No | `"majority"` | Detection method: `majority`, `similarity`, `explicit`, `any` |
| `human_in_loop` | No | `false` | Whether MAS can escalate to human review |
| `human_escalation_condition` | No | `None` | Python expression for escalation trigger |
| `checkpoint_enabled` | No | `true` | Enable state persistence |
| `checkpoint_config` | No | `{"type": "sqlite", "path": "checkpoints.db"}` | Checkpoint backend configuration |
| `tags` | No | `[]` | Searchable labels for categorization |
| `metadata` | No | `{}` | Arbitrary key-value pairs |

*\* Required when `workflow_type` is `consensus`.*

### How MAS Fields Map to Communication Dimensions

The MAS configuration expresses the four communication dimensions through its concrete fields rather than abstract labels:

| Dimension | Expressed by | Example |
|-----------|-------------|---------|
| **Communication Channel** | `channels` list — `protocol`, `source`, `target`, `bidirectional` | `protocol: direct` for point-to-point; `protocol: consensus` for deliberation |
| **Collaboration Type** | Choice of `protocol` values across channels | All `consensus` = cooperative; all `competitive` = competitive; mixed = coopetition |
| **Communication Strategy** | `workflow_edges` conditions, `AgentSpec.role`, agent objectives | `condition: "state.confidence >= 0.7"` = rule-based; LLM-driven routing = model-based |
| **Communication Structure** | `workflow_type`, `is_supervisor`, `tier`, channel topology | `supervisor` + `entry_point` = centralized; `consensus` + bidirectional mesh = decentralized |

Use `tags` to label these dimensions for search and documentation purposes. See the [MAS Communication Framework](#mas-communication-framework) section for details and examples.

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

## MAS Communication Framework

Multi-agent systems are characterized by four communication dimensions. AETHER represents these through **concrete primitives** (channels, protocols, workflow edges, tiers) rather than abstract labels, because the primitives are what the compiler uses to build a runnable graph.

### Overview

| Dimension | What it describes | How AETHER represents it |
|-----------|-------------------|--------------------------|
| **Communication Channel** | Which agents talk, via what mechanism | `Channel` model with `CommunicationProtocol` enum |
| **Collaboration Type** | Whether agents cooperate, compete, or both | Implicit — expressed by protocol choice and `tags` |
| **Communication Strategy** | How interaction rules are determined | Implicit — expressed by workflow structure and `tags` |
| **Communication Structure** | How authority and control are distributed | Partially explicit via `workflow_type`, `tier`, `is_supervisor`; otherwise `tags` |

### Communication Channels

Channels are the core primitive. Each channel connects two agents with a specific protocol:

```yaml
channels:
  - channel_id: reviewer_to_judge
    protocol: direct            # Point-to-point message passing
    source: content_reviewer
    target: judge
    description: Reviewer passes verdict to judge

  - channel_id: judge_to_policy
    protocol: request_response  # Bidirectional exchange
    source: judge
    target: policy_expert
    bidirectional: true
```

For custom workflows, `workflow_edges` add conditional routing on top of channels:

```yaml
workflow_edges:
  - from_agent: judge
    to_agent: END
    condition: "state.confidence >= 0.7"
    label: decide

  - from_agent: judge
    to_agent: appeals_specialist
    condition: "state.confidence < 0.7"
    label: escalate
```

#### Protocol Reference

| Protocol | Pattern | Typical use |
|----------|---------|-------------|
| `direct` | A → B | Sequential handoff, submitting results upstream |
| `broadcast` | A → All | Announcements, distributing input to all agents |
| `request_response` | A ↔ B | Supervisor consulting a specialist |
| `pubsub` | Publish-subscribe | Event-driven architectures |
| `competitive` | A ↔ B (adversarial) | Debate between opposing advocates |
| `consensus` | A ↔ B (deliberative) | Peer agents working toward agreement |

### Collaboration Types

Collaboration type describes *how agents relate to each other's goals*. It is expressed through the choice of protocols and channel topology, not a dedicated schema field.

**Cooperation** — agents align individual objectives with a shared collective goal:
```yaml
# Peer deliberation with consensus protocol
channels:
  - channel_id: manager_reviewer
    protocol: consensus
    source: community_manager
    target: content_reviewer
    bidirectional: true
tags: [cooperative]
```

**Competition** — agents prioritize their own objectives, which may conflict:
```yaml
# Adversarial debate between opposing advocates
channels:
  - channel_id: content_debate
    protocol: competitive
    source: allow_content_agent
    target: block_content_agent
    bidirectional: true
tags: [competitive]
```

**Coopetition** — agents collaborate on shared tasks while competing on others:
```yaml
# Mix of cooperative and competitive channels in one MAS
channels:
  - channel_id: debate
    protocol: competitive
    source: advocate_a
    target: advocate_b
    bidirectional: true
  - channel_id: to_aggregator
    protocol: direct
    source: advocate_a
    target: aggregator
tags: [coopetition]
```

### Communication Strategies

Communication strategy describes *how interaction rules are determined*. This is expressed through the workflow structure itself.

| Strategy | How it's expressed in AETHER |
|----------|------------------------------|
| **Rule-based** | `WorkflowEdge.condition` with predefined expressions (e.g., `"state.confidence >= 0.7"`) |
| **Role-based** | `AgentSpec.role` determines what each agent does; channels route based on roles |
| **Model-based** | Agents use probabilistic reasoning via LLM calls; expressed through `temperature` and agent objectives |

Use `tags` to label which strategy a MAS employs (e.g., `tags: [rule-based]` or `tags: [model-based]`).

### Communication Structures

Communication structure describes *how authority and control are distributed*.

| Structure | AETHER primitives |
|-----------|-------------------|
| **Centralized** | `workflow_type: supervisor` + `is_supervisor: true` + `entry_point` — one agent controls routing |
| **Decentralized** | `workflow_type: consensus` + bidirectional peer channels — no single point of control |
| **Hierarchical** | `workflow_type: hierarchical` + `tier` values on agents — layered authority |

Use `tags` to label the structure (e.g., `tags: [centralized, hub-and-spoke]`).

### Design Rationale

These higher-level dimensions (collaboration type, strategy, structure) are intentionally **not** explicit schema fields. The reasoning:

1. **The compiler operates on primitives.** The AETHER-to-LangGraph compiler builds graphs from channels, edges, protocols, and tiers. It doesn't need an abstract `collaboration_type` label to construct the graph — the concrete wiring *is* the collaboration type.

2. **Explicit labels create redundancy.** If a config declares `communication_structure: centralized` but wires decentralized peer channels, which is truth? The topology is always the source of truth. Labels would require cross-validation against the actual structure.

3. **Tags are the right layer for taxonomy.** The `tags` field provides searchable, filterable metadata without being load-bearing. Tags describe a finished config; primitives build it.

4. **Templates over taxonomy.** When building user-facing tooling, present these dimensions as template selectors ("I want a cooperative, decentralized system") that output concrete channel and edge configurations. The preset system (`agent_presets.py`) follows this same pattern at the agent level.

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
