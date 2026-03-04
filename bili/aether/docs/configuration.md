# AETHER Configuration Reference

Agent and MAS field references, YAML examples, preset system, and validation rules.

## Table of Contents

- [Agent Configuration (AgentSpec)](#agent-configuration-agentspec)
  - [Minimal Agent](#minimal-agent)
  - [Fully-Configured Agent](#fully-configured-agent)
  - [Agent Field Reference](#agent-field-reference)
  - [Objective vs. System Prompt](#objective-vs-system-prompt)
  - [Preset System](#preset-system)
- [MAS Configuration (MASConfig)](#mas-configuration-masconfig)
  - [Minimal MAS](#minimal-mas)
  - [Fully-Configured MAS](#fully-configured-mas)
  - [MAS Field Reference](#mas-field-reference)
  - [Channel Field Reference](#channel-field-reference)
  - [WorkflowEdge Field Reference](#workflowedge-field-reference)
  - [Communication Protocols](#communication-protocols)
- [Validation](#validation)

---

## Agent Configuration (AgentSpec)

### Minimal Agent

An agent requires only three fields — `agent_id`, `role`, and `objective`. Everything else defaults sensibly.

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

Defaults: `temperature=0.0`, `output_format=text`, no tools, no capabilities, no inheritance.

### Fully-Configured Agent

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

    # --- Per-Agent Middleware ---
    middleware:
      - summarization
    middleware_params:
      summarization:
        max_tokens_before_summary: 4000
        messages_to_keep: 20

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
| `middleware` | No | `[]` | Middleware names for agent execution |
| `middleware_params` | No | `{}` | Parameters for middleware, keyed by name |
| `inherit_from_bili_core` | No | `false` | Master toggle: inherit config from bili-core |
| `inherit_llm_config` | No | `true` | Inherit LLM model/temperature (requires master toggle) |
| `inherit_tools` | No | `true` | Inherit tool configuration (requires master toggle) |
| `inherit_system_prompt` | No | `true` | Inherit system prompt (requires master toggle) |
| `inherit_memory` | No | `true` | Inherit memory management (requires master toggle) |
| `inherit_checkpoint` | No | `true` | Inherit checkpoint persistence (requires master toggle) |
| `tier` | No | `None` | Tier in hierarchical workflows (1 = highest authority) |
| `voting_weight` | No | `1.0` | Weight in voting/consensus workflows |
| `is_supervisor` | No | `false` | Whether agent can dynamically route to specialists |
| `consensus_vote_field` | No | `None` | Field name in output containing vote |
| `metadata` | No | `{}` | Arbitrary key-value pairs for custom use |

### Objective vs. System Prompt

| Field | Purpose | Used by |
|-------|---------|---------|
| `objective` | **What** the agent should accomplish — its goal within the MAS | The orchestrator; describes the agent's purpose to other agents and in workflow routing |
| `system_prompt` | **How** the agent should behave — instructions for the LLM itself | The LLM; shapes tone, format, constraints, and reasoning approach |

`objective` is required. `system_prompt` is optional — when omitted, the agent inherits from bili-core (if `inherit_from_bili_core: true` and `inherit_system_prompt: true`) or uses the LLM's default behavior.

Each agent can specify its own LLM via `model_name`. Different agents in the same MAS can use different models.

### Preset System

Presets provide sensible defaults for common roles, letting you skip manual configuration.

```python
from bili.aether.schema import create_agent_from_preset, list_presets, register_preset

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

# Register a custom preset for reuse
register_preset("my_analyst", {
    "role": "analyst",
    "capabilities": ["data_analysis", "reporting"],
    "temperature": 0.3,
})

agent = create_agent_from_preset(
    preset_name="my_analyst",
    agent_id="analyst_1",
    objective="Analyze quarterly trends",
)
```

---

## MAS Configuration (MASConfig)

### Minimal MAS

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

### Fully-Configured MAS

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
    model_name: gpt-4
    temperature: 0.2
    capabilities: [rag_retrieval, policy_lookup, tool_calling]
    output_format: json

  - agent_id: policy_expert
    role: policy_expert
    objective: Review content against OpenAI's Usage Policies
    model_name: claude-sonnet-3-5-20241022
    temperature: 0.1
    capabilities: [policy_lookup, rag_retrieval]
    output_format: json

  - agent_id: judge
    role: judge
    objective: Evaluate reviewer inputs and decide or escalate
    temperature: 0.0
    output_format: json

  - agent_id: appeals_specialist
    role: appeals_specialist
    objective: Act as tie-breaker when the judge cannot decide
    temperature: 0.3
    output_format: json

channels:
  - channel_id: reviewer_to_judge
    protocol: direct
    source: content_reviewer
    target: judge

  - channel_id: policy_to_judge
    protocol: direct
    source: policy_expert
    target: judge

  - channel_id: judge_to_appeals
    protocol: direct
    source: judge
    target: appeals_specialist

workflow_edges:
  - from_agent: community_manager
    to_agent: content_reviewer
    label: flag-content

  - from_agent: community_manager
    to_agent: policy_expert
    label: flag-policy

  - from_agent: content_reviewer
    to_agent: judge
    label: verdict

  - from_agent: policy_expert
    to_agent: judge
    label: verdict

  - from_agent: judge
    to_agent: END
    condition: "state.confidence >= 0.7"
    label: decide

  - from_agent: judge
    to_agent: appeals_specialist
    condition: "state.confidence < 0.7"
    label: escalate

  - from_agent: appeals_specialist
    to_agent: END
    label: final-decision

# --- Consensus Settings ---
consensus_threshold: null
max_consensus_rounds: 10
consensus_detection: majority            # majority | similarity | explicit | any

# --- Human-in-the-Loop ---
human_in_loop: true
human_escalation_condition: >-
  state.tie_breaker_needed or state.confidence < 0.5

# --- Checkpoint Configuration ---
checkpoint_enabled: true
checkpoint_config:
  type: auto                             # memory | postgres | mongo | auto
  keep_last_n: 10

# --- Metadata ---
tags:
  - content-moderation
  - custom
  - cooperative
  - escalation
  - human-in-loop

metadata:
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
| `workflow_type` | Yes | — | Execution pattern (see [compiler.md](compiler.md)) |
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
| `checkpoint_config` | No | `{"type": "memory"}` | Checkpoint backend config |
| `tags` | No | `[]` | Searchable labels for categorization |
| `metadata` | No | `{}` | Arbitrary key-value pairs |

*\* Required when `workflow_type` is `consensus`.*

### Channel Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `channel_id` | Yes | — | Unique channel identifier |
| `protocol` | Yes | — | Communication protocol (see table below) |
| `source` | Yes | — | Sending agent ID |
| `target` | Yes | — | Receiving agent ID |
| `description` | No | `""` | Human-readable description |
| `bidirectional` | No | `false` | Enable two-way exchange (primarily for `request_response`) |

### WorkflowEdge Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `from_agent` | Yes | — | Source agent ID (or `START`) |
| `to_agent` | Yes | — | Destination agent ID (or `END`) |
| `condition` | No | `None` | Python expression evaluated against state for conditional routing |
| `label` | No | `None` | Human-readable edge label |

### Communication Protocols

| Protocol | Pattern | Typical Use |
|----------|---------|-------------|
| `direct` | A → B | Sequential handoff, submitting results upstream |
| `broadcast` | A → All | Announcements, distributing input to all agents |
| `request_response` | A ↔ B | Supervisor consulting a specialist |
| `pubsub` | Publish-subscribe | Event-driven architectures |
| `competitive` | A ↔ B (adversarial) | Debate between opposing advocates |
| `consensus` | A ↔ B (deliberative) | Peer agents working toward agreement |

> **Note:** `pubsub`, `competitive`, and `consensus` channel protocols currently fall back to `DirectChannel` — full implementations are planned.

---

## Validation

Run `validate_mas()` after building a config to catch structural issues before execution:

```python
from bili.aether import validate_mas, load_mas_from_yaml

config = load_mas_from_yaml("path/to/config.yaml")
result = validate_mas(config)

if not result:
    print(result)  # Prints numbered errors and warnings
```

Validation checks for **errors** (fatal — block execution) and **warnings** (non-fatal — should review):

| Type | Check |
|------|-------|
| Error | Duplicate channels (same source + target + protocol) |
| Error | Circular dependencies in sequential workflows |
| Error | Hierarchical workflow missing tier 1 agent |
| Warning | Agent with no channel connections (orphaned) |
| Warning | Supervisor agent missing `inter_agent_communication` capability |
| Warning | Bidirectional channel with redundant reverse channel |
| Warning | Sequential workflow agent with multiple outgoing edges |
| Warning | Unreachable agent from entry point in custom workflow |
| Warning | Custom workflow with no path to END |
| Warning | Consensus agent missing `consensus_vote_field` |
| Warning | Hierarchical tier gap (e.g., tier 1 and 3 but no 2) |
| Warning | Supervisor entry point not marked `is_supervisor` |
| Warning | `human_in_loop` enabled without `human_escalation_condition` |

Validation runs on top of Pydantic field-level validation — it catches cross-field and structural issues that individual field constraints can't express.

`compile_mas()` runs the validator automatically. Errors raise `ValueError`; warnings do not block compilation.
