# AETHER: Agent Ecosystems for Testing, Hardening, Evaluation, and Research

AETHER is a domain-agnostic multi-agent system (MAS) framework built as an extension to bili-core. It enables declarative configuration of multi-agent systems for any domain - research, content moderation, code review, customer support, and more.

## Table of Contents

- [How to Use](#how-to-use)
  - [Using Presets](#using-presets)
- [Design Philosophy](#design-philosophy)
- [Quick Start](#quick-start)
  - [Minimal Agent (Required Fields Only)](#minimal-agent-required-fields-only)
  - [Fully-Configured Agent (All Options)](#fully-configured-agent-all-options)
  - [Agent Field Reference](#agent-field-reference)
  - [Objective vs. System Prompt](#objective-vs-system-prompt)
  - [Use Presets for Common Patterns](#use-presets-for-common-patterns)
  - [Register Custom Presets](#register-custom-presets)
- [Configuring a Multi-Agent System](#configuring-a-multi-agent-system)
  - [Minimal MAS (Sequential Chain)](#minimal-mas-sequential-chain)
  - [Fully-Configured MAS (All Options)](#fully-configured-mas-all-options)
  - [MAS Field Reference](#mas-field-reference)
  - [How MAS Fields Map to Communication Dimensions](#how-mas-fields-map-to-communication-dimensions)
- [Validating a Configuration](#validating-a-configuration)
- [Compiling to LangGraph](#compiling-to-langgraph)
  - [What the Compiler Produces](#what-the-compiler-produces)
  - [How Workflow Types Map to Graph Topologies](#how-workflow-types-map-to-graph-topologies)
  - [Generated State Schema](#generated-state-schema)
  - [Agent Nodes](#agent-nodes)
  - [Agent Communication Protocol (Runtime)](#agent-communication-protocol-runtime)
  - [bili-core Inheritance](#bili-core-inheritance)
  - [CLI Tool (Compiler)](#cli-tool-compiler)
- [Executing a MAS](#executing-a-mas)
  - [Basic Execution](#basic-execution)
  - [MASExecutor Methods](#masexecutor-methods)
  - [MASExecutionResult](#masexecutionresult)
  - [Checkpoint Persistence Testing](#checkpoint-persistence-testing)
  - [Cross-Model Transfer Testing](#cross-model-transfer-testing)
  - [CLI Tool (Runtime)](#cli-tool-runtime)
- [Example Workflows](#example-workflows)
- [Architecture](#architecture)
- [Integration with bili-core](#integration-with-bili-core)
- [Workflow Types](#workflow-types)
- [MAS Communication Framework](#mas-communication-framework)
  - [Communication Channels](#communication-channels)
  - [Collaboration Types](#collaboration-types)
  - [Communication Strategies](#communication-strategies)
  - [Communication Structures](#communication-structures)
  - [Design Rationale](#design-rationale)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## How to Use

AETHER follows a five-step workflow: **Define → Configure → Validate → Compile → Execute**.

**1. Define your agents** — Create `AgentSpec` objects with a role, objective, and optional LLM/tool configuration. Start with the [Minimal Agent](#minimal-agent-required-fields-only) example or use a [preset](#use-presets-for-common-patterns) for common patterns.

**2. Configure a MAS** — Assemble agents into a `MASConfig` with a [workflow type](#workflow-types) (`sequential`, `hierarchical`, `supervisor`, `consensus`, `parallel`, or `custom`) and [communication channels](#communication-channels). See [Configuring a Multi-Agent System](#configuring-a-multi-agent-system) for YAML and Python examples.

**3. Validate** — Run `validate_mas(config)` to catch structural issues (duplicate channels, circular dependencies, orphaned agents) before execution. See [Validating a Configuration](#validating-a-configuration).

**4. Compile** — Call `compile_mas(config)` to transform the declarative config into an executable LangGraph `StateGraph`. See [Compiling to LangGraph](#compiling-to-langgraph).

**5. Execute** — Use `MASExecutor` or the `execute_mas()` convenience function to run the graph and collect structured results. See [Executing a MAS](#executing-a-mas).

```python
from bili.aether import load_mas_from_yaml, compile_mas, execute_mas

config = load_mas_from_yaml("path/to/config.yaml")   # Steps 1-2: Load agents + MAS
compiled = compile_mas(config)                         # Steps 3-4: Validate + compile
result = execute_mas(config, {"messages": [...]})      # Step 5: Execute
print(result.get_summary())
```

### Using Presets

Presets let you skip manual agent configuration by providing sensible defaults for common roles. You can list available presets, create agents from them with optional overrides, and register your own:

```python
from bili.aether.schema import create_agent_from_preset, list_presets, register_preset

# See what's available
print(list_presets())
# ['content_reviewer', 'researcher', 'code_reviewer', 'supervisor', ...]

# Create an agent from a preset (override any defaults as needed)
agent = create_agent_from_preset(
    preset_name="researcher",
    agent_id="ai_researcher",
    objective="Research quantum computing advances",
    temperature=0.6,
)

# Register a custom preset for reuse across projects
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

For the full preset API, see [Use Presets for Common Patterns](#use-presets-for-common-patterns) and [Register Custom Presets](#register-custom-presets).

For CLI usage, see the [Compiler CLI](#cli-tool-compiler) and [Runtime CLI](#cli-tool-runtime) sections.

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
| `middleware` | No | `[]` | Middleware names for agent execution (see Per-Agent Middleware) |
| `middleware_params` | No | `{}` | Parameters for middleware, keyed by name |
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
  type: auto                             # memory | postgres | mongo | auto
  keep_last_n: 10                        # Checkpoints to retain per thread

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
| `checkpoint_config` | No | `{"type": "memory"}` | Checkpoint backend config (see Checkpointer Configuration) |
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

## Validating a Configuration

After building a `MASConfig`, run it through the static validator to catch structural issues before execution:

```python
from bili.aether import validate_mas, load_mas_from_yaml

config = load_mas_from_yaml("path/to/config.yaml")
result = validate_mas(config)

if not result:
    print(result)  # Prints numbered errors and warnings
```

The validator checks for **errors** (fatal — block execution) and **warnings** (non-fatal — should review):

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

Validation runs on top of Pydantic's field-level validation — it catches cross-field and structural issues that individual field constraints can't express.

## Compiling to LangGraph

After validation, compile a `MASConfig` into an executable [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`:

```python
from bili.aether import compile_mas, load_mas_from_yaml

config = load_mas_from_yaml("path/to/config.yaml")
compiled = compile_mas(config)       # Validates, then builds the graph
graph = compiled.compile_graph()     # Returns a CompiledStateGraph

# graph is ready for .invoke() or .stream()
```

`compile_mas()` runs the validator first — if validation produces errors, it raises `ValueError` with the full report. Warnings do not block compilation.

### What the Compiler Produces

`compile_mas()` returns a `CompiledMAS` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `MASConfig` | The original configuration used to build the graph |
| `graph` | `StateGraph` | The uncompiled LangGraph graph (can be inspected or modified before compilation) |
| `state_schema` | `TypedDict` | The generated state class with workflow-specific fields |
| `agent_nodes` | `Dict[str, Callable]` | Mapping of `agent_id` to its node callable |
| `checkpoint_config` | `Dict` | Checkpoint backend configuration |
| `channel_manager` | `ChannelManager` (optional) | Manages inter-agent channels; `None` if no channels configured |

**Methods:**
- `compile_graph(checkpointer=None)` — Compile to an executable `CompiledStateGraph`. If `checkpointer` is `None` and `checkpoint_enabled` is `True`, creates a checkpointer from `checkpoint_config` via the bili-core factory (falls back to in-memory if unavailable).
- `get_agent_node(agent_id)` — Look up a specific agent's callable by ID.

### How Workflow Types Map to Graph Topologies

The compiler builds a different graph topology for each `workflow_type`:

| Workflow Type | Graph Topology |
|---------------|----------------|
| `sequential` | Linear chain: `START → A → B → C → END` |
| `hierarchical` | Tier-based fan-out: `START → [leaf tier] → [mid tiers] → [tier 1] → END`. Uses channel definitions for specific inter-tier routing. |
| `supervisor` | Hub-and-spoke: `START → supervisor ⇄ [workers] → END`. Supervisor conditionally routes to workers via `state["next_agent"]`. |
| `consensus` | Round-based deliberation: `START → [all agents] → consensus check → (repeat or END)`. Loops until consensus or `max_consensus_rounds`. |
| `parallel` | Fan-out/fan-in: `START → [all agents] → END`. All agents execute simultaneously. |
| `deliberative` | Delegates to `custom` if `workflow_edges` are defined, otherwise falls back to `sequential`. |
| `custom` | Explicit edges from `workflow_edges`. Supports conditional routing via `condition` expressions evaluated against state. |

### Generated State Schema

Each compiled graph gets a dynamic `TypedDict` state schema. Base fields are always present; workflow-specific fields are added automatically:

**Base fields (all workflows):**
- `messages` — LangGraph message list with `add_messages` reducer
- `current_agent` — ID of the currently executing agent
- `agent_outputs` — `Dict[str, Any]` mapping each agent's ID to its latest output
- `mas_id` — The MAS identifier

**Workflow-specific fields:**

| Workflow Type | Additional State Fields |
|---------------|------------------------|
| `consensus` | `current_round`, `votes`, `consensus_reached`, `max_rounds` |
| `hierarchical` | `current_tier`, `tier_results` |
| `supervisor` | `next_agent`, `pending_tasks`, `completed_tasks` |
| `custom` (with `human_in_loop`) | `needs_human_review` |
| Any workflow with `channels` | `channel_messages`, `pending_messages`, `communication_log` |

### Agent Nodes

Agent nodes are generated in one of three modes based on the `AgentSpec` configuration:

1. **Tool-enabled LLM node** — When `model_name` is set and `tools` are configured, the node uses `create_agent()` from `langchain.agents` with resolved tool instances. This mirrors the pattern used by `bili/nodes/react_agent_node.py`.

2. **Direct LLM node** — When `model_name` is set but no tools are configured, the node calls `llm.invoke(messages)` directly. Messages are filtered to compatible types (`AIMessage`, `HumanMessage`, `SystemMessage`).

3. **Stub node** — When `model_name` is `None`, the node emits a placeholder `AIMessage` without making LLM calls. This allows compilation and graph execution without API keys (all example YAMLs use this mode).

Model resolution is handled by `llm_resolver.py`, which maps `AgentSpec.model_name` to a `(provider_type, model_id)` pair. It searches `bili.config.llm_config.LLM_MODELS` first (by `model_id`, then by display name), and falls back to heuristic prefix-based detection. LLM instances are created via `bili.loaders.llm_loader.load_model()`. Tool names are resolved via `bili.loaders.tools_loader.initialize_tools()`.

Each agent node callable has an `.agent_spec` attribute attached for introspection:

```python
compiled = compile_mas(config)
node = compiled.get_agent_node("content_reviewer")
print(node.agent_spec.role)       # "content_reviewer"
print(node.agent_spec.model_name) # "gpt-4o" (or None for stub mode)
```

### Agent Communication Protocol (Runtime)

When a MAS config includes `channels`, the compiler activates the **runtime communication layer**. This enables agents to send and receive structured messages through declared channels, with every message logged to a JSONL audit file.

#### How It Works

1. **At compile time** — `GraphBuilder` creates a `ChannelManager` from the config's `channels` list. Communication state fields (`channel_messages`, `pending_messages`, `communication_log`) are added to the LangGraph state schema.

2. **At execution time** — Before each agent node runs, pending messages addressed to that agent are retrieved from state and injected into its system prompt:
   ```
   --- Messages from other agents ---
   [From content_reviewer via reviewer_to_judge]: Content violates policy section 3.2.
   [From policy_expert via policy_to_judge]: No policy violations found.
   ```

3. **After execution** — The agent's output is recorded in `communication_log` and its pending messages are cleared.

#### Implemented Channel Types

| Protocol | Class | Behaviour |
|----------|-------|-----------|
| `direct` | `DirectChannel` | Point-to-point (A → B). Supports `bidirectional: true` for two-way. |
| `broadcast` | `BroadcastChannel` | One-to-many (A → all agents except sender). |
| `request_response` | `RequestResponseChannel` | Bidirectional with request/response correlation tracking. |
| `pubsub`, `competitive`, `consensus` | Falls back to `DirectChannel` | Not yet implemented — planned for future tasks. |

#### JSONL Communication Log

Every message is appended to a JSONL file (one JSON object per line):

```json
{"message_id": "a1b2c3...", "timestamp": "2026-02-04T12:00:00+00:00", "sender": "reviewer", "receiver": "judge", "channel": "reviewer_to_judge", "content": "Content violates policy.", "message_type": "direct", "metadata": {}, "in_reply_to": null}
```

Log files are created per execution in the working directory: `communication_{mas_id}_{hash}.jsonl`.

#### Programmatic Usage

The `ChannelManager` can also be used directly outside the compiled graph:

```python
from bili.aether.compiler import compile_mas
from bili.aether.config.loader import load_mas_from_yaml

config = load_mas_from_yaml("path/to/config.yaml")
compiled = compile_mas(config)

# Access the channel manager
mgr = compiled.channel_manager
if mgr:
    mgr.send_message("reviewer_to_judge", "reviewer", "Looks good.")
    pending = mgr.get_messages_for_agent("judge")
    mgr.close()  # Flush and close the JSONL log
```

### bili-core Inheritance

When an agent sets `inherit_from_bili_core: true`, the compiler automatically enriches its `AgentSpec` with role-based defaults before generating nodes. This activates the dormant inheritance flags on `AgentSpec` and connects agents to bili-core's system prompts, tools, and LLM configuration.

#### How It Works

1. **At compile time** — `GraphBuilder` calls `apply_inheritance()` for each agent with `inherit_from_bili_core=True`. The function looks up the agent's `role` in `ROLE_DEFAULTS` and merges defaults per the enabled sub-flags.

2. **Priority rule** — User-specified values **always** take priority over inherited defaults. A custom `system_prompt`, explicit `temperature`, or manually listed `tools` will never be overridden.

3. **Granular control** — Five sub-flags allow selective inheritance:

| Flag | What It Controls | Inheritance Rule |
|------|-----------------|------------------|
| `inherit_system_prompt` | System prompt | Set from registry if agent's is `None` |
| `inherit_llm_config` | Temperature, model_name | Temperature inherited if `0.0` (default); model_name if `None` |
| `inherit_tools` | Tools and capabilities | Additive merge (registry tools + agent tools, deduplicated) |
| `inherit_memory` | Memory management | Extension point (no-op) |
| `inherit_checkpoint` | Checkpoint config | Extension point (no-op, MAS-level) |

#### Role Registry

The registry maps role strings to `RoleDefaults` (system prompt, tools, temperature, capabilities):

| Role | Default Tools | Temperature |
|------|--------------|-------------|
| `content_reviewer` | — | 0.3 |
| `policy_expert` | `faiss_retriever` | 0.2 |
| `judge` | — | 0.1 |
| `appeals_specialist` | `faiss_retriever` | 0.3 |
| `researcher` | `serp_api_tool` | 0.5 |
| `fact_checker` | `serp_api_tool` | 0.2 |
| `support_agent` | `faiss_retriever` | 0.4 |
| `supervisor` | — | 0.2 |

Custom roles can be registered at runtime via `register_role_defaults()`.

#### YAML Example

```yaml
agents:
  - agent_id: reviewer
    role: content_reviewer
    objective: "Review content for policy violations"
    model_name: gpt-4o
    inherit_from_bili_core: true
    # Gets: system_prompt, temperature=0.3, capabilities

  - agent_id: policy
    role: policy_expert
    objective: "Provide policy guidance"
    model_name: gpt-4o
    inherit_from_bili_core: true
    # Gets: system_prompt, tools=[faiss_retriever], temperature=0.2

  - agent_id: analyst
    role: analyst
    objective: "Analyse findings"
    model_name: gpt-4o
    inherit_from_bili_core: true
    inherit_tools: false  # Opt out of tool inheritance
    system_prompt: "Custom prompt overrides registry default"
```

### Checkpointer Configuration

The `checkpoint_config` dict in `MASConfig` controls which checkpointer backend is used when compiling the graph. The factory maps `config["type"]` to a bili-core checkpointer:

| Type | Backend | Notes |
|------|---------|-------|
| `memory` (default) | `QueryableMemorySaver` | In-memory, no persistence across restarts |
| `postgres` / `pg` | `PruningPostgresSaver` | Requires `POSTGRES_CONNECTION_STRING` env var |
| `mongo` / `mongodb` | `PruningMongoDBSaver` | Requires `MONGO_CONNECTION_STRING` env var |
| `auto` | Auto-detected | Tries postgres, then mongo, then memory (based on env vars) |

Additional keys are forwarded to the checkpointer constructor:

```yaml
checkpoint_config:
  type: postgres
  keep_last_n: 10      # Prune to last 10 checkpoints per thread
```

**Fallback behaviour:** If the requested backend is unavailable (missing dependency or env var), the factory falls back to `MemorySaver` with a warning. The graph always compiles successfully.

**Explicit override:** Pass a checkpointer directly to `compile_graph()` to bypass the factory:

```python
from bili.checkpointers.pg_checkpointer import get_pg_checkpointer

compiled = compile_mas(config)
graph = compiled.compile_graph(checkpointer=get_pg_checkpointer(keep_last_n=5))
```

### Per-Agent Middleware

Agents can configure middleware to intercept and modify their execution. Middleware is resolved via bili-core's `initialize_middleware()` and passed to `create_agent()` for tool-enabled agents.

```yaml
agents:
  - agent_id: researcher
    role: researcher
    objective: Research and compile findings
    tools:
      - serp_api_tool
    middleware:
      - summarization
    middleware_params:
      summarization:
        max_tokens_before_summary: 4000
        messages_to_keep: 20

  - agent_id: analyst
    role: analyst
    objective: Analyse findings with rate limiting
    tools:
      - serp_api_tool
    middleware:
      - model_call_limit
    middleware_params:
      model_call_limit:
        run_limit: 10
```

**Available middleware:**

| Name | Description | Key Parameters |
|------|-------------|---------------|
| `summarization` | Auto-summarizes conversation when tokens exceed threshold | `max_tokens_before_summary` (default: 4000), `messages_to_keep` (default: 20) |
| `model_call_limit` | Circuit-breaker for runaway agent loops | `run_limit` (default: 10), `exit_behavior` (`"end"` or `"error"`) |

**Limitations:**
- Middleware only applies to **tool-enabled agents** (agents with `tools` configured). If middleware is set on an agent without tools, a warning is logged and middleware is skipped.
- Middleware requires `langchain.agents.middleware` — if unavailable, agents run without middleware.

### CLI Tool (Compiler)

A CLI tool is included for quick compilation testing:

```bash
# Compile all example configs
python bili/aether/compiler/cli.py

# Compile a specific YAML
python bili/aether/compiler/cli.py path/to/config.yaml
```

## Executing a MAS

After compilation, the **MAS Execution Controller** runs the graph end-to-end and collects structured results.

### Basic Execution

```python
from bili.aether import load_mas_from_yaml, MASExecutor, execute_mas

# Option 1: MASExecutor (full control)
config = load_mas_from_yaml("simple_chain.yaml")
executor = MASExecutor(config, log_dir="logs")
executor.initialize()
result = executor.run({"messages": [HumanMessage(content="Test content")]})

print(result.get_summary())           # Concise text summary
print(result.get_formatted_output())  # Asterisk-bordered output

# Option 2: execute_mas() convenience function
result = execute_mas(config, {"messages": [HumanMessage(content="Test")]})
```

### MASExecutor Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Compile config to executable LangGraph |
| `run(input_data, thread_id, save_results)` | `MASExecutionResult` | Execute graph, collect results |
| `run_with_checkpoint_persistence(input_data, thread_id)` | `(original, restored)` | Run twice with checkpoint save/restore |
| `run_cross_model_test(input_data, source_model, target_model, thread_id)` | `(source, target)` | Run with two different model configurations |

### MASExecutionResult

| Field | Type | Description |
|-------|------|-------------|
| `mas_id` | `str` | MAS identifier |
| `execution_id` | `str` | Unique execution run ID |
| `start_time` / `end_time` | `str` | ISO-8601 UTC timestamps |
| `total_execution_time_ms` | `float` | Total wall-clock time |
| `agent_results` | `List[AgentExecutionResult]` | Per-agent outputs and statistics |
| `total_messages` | `int` | Total inter-agent messages |
| `messages_by_channel` | `Dict[str, int]` | Message counts by channel |
| `checkpoint_saved` | `bool` | Whether checkpoint was persisted |
| `success` | `bool` | Whether execution succeeded |
| `error` | `str` (optional) | Error message on failure |

**Methods:** `to_dict()`, `save_to_file(path)`, `get_summary()`, `get_formatted_output()`

### Checkpoint Persistence Testing

```python
executor = MASExecutor(config)
executor.initialize()

# Run twice — first saves checkpoint, second restores it
original, restored = executor.run_with_checkpoint_persistence(
    input_data={"messages": [HumanMessage(content="Test")]},
    thread_id="test_001"
)
print(f"Original: {original.success}, Restored: {restored.success}")
```

### Cross-Model Transfer Testing

```python
source, target = executor.run_cross_model_test(
    input_data={"messages": [HumanMessage(content="Test")]},
    source_model="gpt-4",
    target_model="claude-sonnet-3-5-20241022",
    thread_id="transfer_001"
)
```

### CLI Tool (Runtime)

```bash
# Basic execution
python bili/aether/runtime/cli.py configs/simple_chain.yaml \
    --input "Test content to review"

# With input file
python bili/aether/runtime/cli.py configs/production.yaml \
    --input-file input.txt --log-dir my_logs

# Test checkpoint persistence
python bili/aether/runtime/cli.py configs/production.yaml \
    --input "Test payload" --test-checkpoint --thread-id test_001

# Test cross-model transfer
python bili/aether/runtime/cli.py configs/production.yaml \
    --input "Test payload" --test-cross-model \
    --source-model gpt-4 --target-model claude-sonnet-3-5-20241022
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
- `inherited_research.yaml` - Same pipeline using `inherit_from_bili_core: true`

### Integration Features
- `middleware_checkpointer.yaml` - Per-agent middleware + auto-detect checkpoint config

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
│   └── enums.py          # Structural enums (WorkflowType, etc.)
├── config/               # YAML loading and examples
│   ├── loader.py         # load_mas_from_yaml, load_mas_from_dict
│   └── examples/         # Example YAML configurations
├── compiler/             # AETHER-to-LangGraph compiler
│   ├── __init__.py       # compile_mas() entry point
│   ├── graph_builder.py  # StateGraph construction
│   ├── agent_generator.py # LLM-backed and stub agent nodes
│   ├── llm_resolver.py   # Model resolution and LLM instantiation
│   ├── state_generator.py # TypedDict state schema generation
│   ├── compiled_mas.py   # CompiledMAS container
│   └── cli.py            # CLI compilation tool
├── runtime/              # Agent communication and execution
│   ├── messages.py       # Message, MessageType, MessageHistory
│   ├── logger.py         # CommunicationLogger (JSONL)
│   ├── channels.py       # DirectChannel, BroadcastChannel, RequestResponseChannel
│   ├── channel_manager.py # ChannelManager
│   ├── communication_state.py # LangGraph state integration helpers
│   ├── execution_result.py # AgentExecutionResult, MASExecutionResult
│   ├── executor.py       # MASExecutor, execute_mas()
│   └── cli.py            # Runtime CLI tool
├── integration/          # bili-core integration layer
│   ├── __init__.py       # Package exports
│   ├── role_registry.py  # RoleDefaults registry (16 roles)
│   ├── inheritance.py    # apply_inheritance() resolution logic
│   ├── checkpointer_factory.py # Checkpoint config → bili-core checkpointer
│   └── state_integration.py # State schema extension hook
├── validation/           # Static MAS validation engine
│   ├── __init__.py       # validate_mas() entry point
│   └── engine.py         # 13 structural validation checks
├── examples/             # Complete workflow examples
│   └── complete_aether_workflow.py
└── tests/                # Test suite (200+ tests)
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
**Last Updated:** February 5, 2026
**bili-core Version Required:** ≥3.0.0
