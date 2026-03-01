# AETHER Compiler & Execution Reference

Compiler internals, workflow types, state schema, bili-core inheritance, communication framework, and the execution API.

## Table of Contents

- [Compilation](#compilation)
  - [What the Compiler Produces](#what-the-compiler-produces)
  - [Workflow Types & Graph Topologies](#workflow-types--graph-topologies)
  - [Generated State Schema](#generated-state-schema)
  - [Agent Nodes](#agent-nodes)
  - [CLI Tool (Compiler)](#cli-tool-compiler)
- [Communication Framework](#communication-framework)
  - [Runtime Channel Layer](#runtime-channel-layer)
  - [Collaboration Types](#collaboration-types)
  - [Communication Strategies](#communication-strategies)
  - [Communication Structures](#communication-structures)
  - [Design Rationale](#design-rationale)
- [bili-core Inheritance](#bili-core-inheritance)
  - [How It Works](#how-it-works)
  - [Role Registry](#role-registry)
  - [Checkpointer Configuration](#checkpointer-configuration)
  - [Per-Agent Middleware](#per-agent-middleware)
- [Execution](#execution)
  - [MASExecutor](#masexecutor)
  - [MASExecutionResult](#masexecutionresult)
  - [Checkpoint Persistence Testing](#checkpoint-persistence-testing)
  - [Cross-Model Transfer Testing](#cross-model-transfer-testing)
  - [CLI Tool (Runtime)](#cli-tool-runtime)
- [Architecture](#architecture)

---

## Compilation

```python
from bili.aether import compile_mas, load_mas_from_yaml

config = load_mas_from_yaml("path/to/config.yaml")
compiled = compile_mas(config)       # Validates, then builds the graph
graph = compiled.compile_graph()     # Returns a CompiledStateGraph ready for .invoke()/.stream()
```

`compile_mas()` runs the validator first. Validation errors raise `ValueError` with the full report; warnings do not block compilation.

### What the Compiler Produces

`compile_mas()` returns a `CompiledMAS` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `MASConfig` | The original configuration |
| `graph` | `StateGraph` | The uncompiled LangGraph graph |
| `state_schema` | `TypedDict` | Generated state class with workflow-specific fields |
| `agent_nodes` | `Dict[str, Callable]` | Mapping of `agent_id` to its node callable |
| `checkpoint_config` | `Dict` | Checkpoint backend configuration |

**Methods:**

- `compile_graph(checkpointer=None)` — Compile to an executable `CompiledStateGraph`. If `checkpointer` is `None` and `checkpoint_enabled` is `True`, creates one from `checkpoint_config` via the bili-core factory.
- `get_agent_node(agent_id)` — Look up a specific agent's callable by ID.
- `compile_graph(interrupt_before=[...], interrupt_after=[...])` — Compile with interrupt points for mid-execution inspection or attack injection.

### Workflow Types & Graph Topologies

The compiler builds a different graph topology for each `workflow_type`:

| Workflow Type | Graph Topology |
|---------------|----------------|
| `sequential` | Linear chain: `START → A → B → C → END` |
| `hierarchical` | Tier-based fan-out: `START → [leaf tier] → [mid tiers] → [tier 1] → END`. Channel definitions control specific inter-tier routing. |
| `supervisor` | Hub-and-spoke: `START → supervisor ⇄ [workers] → END`. Supervisor routes to workers via `state["next_agent"]`. |
| `consensus` | Round-based deliberation: `START → [all agents] → consensus check → (repeat or END)`. Loops until consensus or `max_consensus_rounds`. |
| `parallel` | Fan-out/fan-in: `START → [all agents] → END`. All agents execute simultaneously. |
| `deliberative` | Delegates to `custom` if `workflow_edges` are defined; otherwise falls back to `sequential`. |
| `custom` | Explicit edges from `workflow_edges`. Supports conditional routing via `condition` expressions evaluated against state. |

**ASCII topology diagrams:**

```
sequential:    A ──► B ──► C ──► END

hierarchical:  tier-3      tier-2      tier-1
               A ──► C ──────────► F ──► END
               B ──► D ──► E ──────────►

supervisor:    START ──► judge ◄──► reviewer
                              ◄──► policy_expert
                              ◄──► appeals
                         ──► END

consensus:     START ──► [A, B, C simultaneously] ──► check ──► repeat/END

parallel:      START ──► [A, B, C simultaneously] ──► END

custom:        User-defined edges with optional conditional branching
```

### Generated State Schema

Each compiled graph gets a dynamic `TypedDict` state schema. Base fields are always present:

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

Agent nodes are generated in one of three modes based on `AgentSpec` configuration:

1. **Tool-enabled LLM node** — `model_name` set and `tools` configured. Uses `create_agent()` from `langchain.agents` with resolved tool instances. Middleware applies here.

2. **Direct LLM node** — `model_name` set but no tools. Calls `llm.invoke(messages)` directly. Messages are filtered to compatible types (`AIMessage`, `HumanMessage`, `SystemMessage`).

3. **Stub node** — `model_name` is `None`. Emits a placeholder `AIMessage` without LLM calls. Allows full compilation and graph execution without API keys. All example YAMLs use this mode.

Model resolution is handled by `llm_resolver.py`, which maps `model_name` to a `(provider_type, model_id)` pair by searching `bili.config.llm_config.LLM_MODELS` and falling back to heuristic prefix-based detection. LLMs are created via `bili.loaders.llm_loader.load_model()`. Tools are resolved via `bili.loaders.tools_loader.initialize_tools()`.

Each agent node callable has an `.agent_spec` attribute for introspection:

```python
compiled = compile_mas(config)
node = compiled.get_agent_node("content_reviewer")
print(node.agent_spec.role)       # "content_reviewer"
print(node.agent_spec.model_name) # "gpt-4" (or None for stub)
```

### CLI Tool (Compiler)

```bash
# Compile all example configs
python bili/aether/compiler/cli.py

# Compile a specific YAML
python bili/aether/compiler/cli.py path/to/config.yaml
```

---

## Communication Framework

Multi-agent systems are characterized by four communication dimensions. AETHER represents these through **concrete primitives** (channels, protocols, workflow edges, tiers) rather than abstract labels.

| Dimension | What it describes | How AETHER represents it |
|-----------|-------------------|--------------------------|
| **Communication Channel** | Which agents talk, via what mechanism | `Channel` model with `CommunicationProtocol` enum |
| **Collaboration Type** | Whether agents cooperate, compete, or both | Implicit — expressed by protocol choice and `tags` |
| **Communication Strategy** | How interaction rules are determined | Implicit — expressed by workflow structure and `tags` |
| **Communication Structure** | How authority and control are distributed | Partially explicit via `workflow_type`, `tier`, `is_supervisor`; otherwise `tags` |

### Runtime Channel Layer

When a MAS config includes `channels`, the compiler activates the runtime communication layer:

1. **At compile time** — `GraphBuilder` reads the `channels` list and adds communication state fields (`channel_messages`, `pending_messages`, `communication_log`) to the state schema. Channel routing is handled internally through state — there is no separate manager object on `CompiledMAS`.

2. **At execution time** — Before each agent node runs, pending messages addressed to it are retrieved from state and injected into its system prompt:
   ```
   --- Messages from other agents ---
   [From content_reviewer via reviewer_to_judge]: Content violates policy section 3.2.
   [From policy_expert via policy_to_judge]: No policy violations found.
   ```

3. **After execution** — The agent's output is recorded in `communication_log` and its pending messages are cleared.

**Implemented channel types:**

| Protocol | Class | Behaviour |
|----------|-------|-----------|
| `direct` | `DirectChannel` | Point-to-point (A → B). Supports `bidirectional: true`. |
| `broadcast` | `BroadcastChannel` | One-to-many (A → all agents except sender). |
| `request_response` | `RequestResponseChannel` | Bidirectional with request/response correlation tracking. |
| `pubsub`, `competitive`, `consensus` | Falls back to `DirectChannel` | Not yet fully implemented. |

**JSONL Communication Log:**

Every message is appended to a JSONL file (`communication_{mas_id}_{hash}.jsonl`):

```json
{"message_id": "a1b2...", "timestamp": "2026-02-04T12:00:00+00:00", "sender": "reviewer", "receiver": "judge", "channel": "reviewer_to_judge", "content": "Content violates policy.", "message_type": "direct", "metadata": {}, "in_reply_to": null}
```

**Channel state at runtime:**

Channel messages are accessible through graph state after execution. Inspect `communication_log` in the final state returned by `graph.invoke()`:

```python
graph = compiled.compile_graph()
final_state = graph.invoke({"messages": [...]})
for entry in final_state.get("communication_log", []):
    print(entry)  # {"sender": "reviewer", "receiver": "judge", "content": "..."}
```

### Collaboration Types

Collaboration type describes how agents relate to each other's goals. It is expressed through protocol choice and channel topology, not a dedicated schema field.

**Cooperation** — agents align toward a shared collective goal:
```yaml
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

Communication strategy describes how interaction rules are determined:

| Strategy | How it's expressed in AETHER |
|----------|------------------------------|
| **Rule-based** | `WorkflowEdge.condition` with predefined expressions (e.g., `"state.confidence >= 0.7"`) |
| **Role-based** | `AgentSpec.role` determines what each agent does; channels route based on roles |
| **Model-based** | Agents use probabilistic LLM reasoning; expressed through `temperature` and agent objectives |

Use `tags` to label which strategy a MAS employs (e.g., `tags: [rule-based]`).

### Communication Structures

Communication structure describes how authority and control are distributed:

| Structure | AETHER primitives |
|-----------|-------------------|
| **Centralized** | `workflow_type: supervisor` + `is_supervisor: true` + `entry_point` |
| **Decentralized** | `workflow_type: consensus` + bidirectional peer channels |
| **Hierarchical** | `workflow_type: hierarchical` + `tier` values on agents |

Use `tags` to label the structure (e.g., `tags: [centralized, hub-and-spoke]`).

### Design Rationale

Higher-level dimensions (collaboration type, strategy, structure) are intentionally **not** explicit schema fields:

1. **The compiler operates on primitives.** It builds graphs from channels, edges, protocols, and tiers — not abstract labels.
2. **Explicit labels create redundancy.** The topology is always the source of truth.
3. **Tags are the right layer for taxonomy.** They provide searchable metadata without being load-bearing.
4. **Templates over taxonomy.** Present these dimensions as template selectors that output concrete channel/edge configurations — the preset system follows this same pattern at the agent level.

---

## bili-core Inheritance

When an agent sets `inherit_from_bili_core: true`, the compiler enriches its `AgentSpec` with role-based defaults before generating nodes.

### How It Works

1. **At compile time** — `GraphBuilder` calls `apply_inheritance()` for each agent with the flag enabled. The function looks up the agent's `role` in `ROLE_DEFAULTS` and merges per the enabled sub-flags.

2. **Priority rule** — User-specified values **always** take priority over inherited defaults. A custom `system_prompt`, explicit `temperature`, or manually listed `tools` will never be overridden.

3. **Granular control via sub-flags:**

| Flag | What It Controls | Inheritance Rule |
|------|-----------------|------------------|
| `inherit_system_prompt` | System prompt | Set from registry if agent's is `None` |
| `inherit_llm_config` | Temperature, model_name | Temperature inherited if `0.0` (default); model_name if `None` |
| `inherit_tools` | Tools and capabilities | Additive merge (registry tools + agent tools, deduplicated) |
| `inherit_memory` | Memory management | Extension point (no-op currently) |
| `inherit_checkpoint` | Checkpoint config | Extension point (no-op, MAS-level) |

```yaml
agents:
  - agent_id: reviewer
    role: content_reviewer
    objective: "Review content for policy violations"
    model_name: gpt-4o
    inherit_from_bili_core: true
    # Gets: system_prompt from registry, temperature=0.3

  - agent_id: analyst
    role: analyst
    objective: "Analyse findings"
    model_name: gpt-4o
    inherit_from_bili_core: true
    inherit_tools: false          # Opt out of tool inheritance
    system_prompt: "Custom prompt overrides registry default"
```

### Role Registry

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

### Checkpointer Configuration

The `checkpoint_config` dict in `MASConfig` controls which checkpointer backend is used:

| Type | Backend | Notes |
|------|---------|-------|
| `memory` (default) | `QueryableMemorySaver` | In-memory, no persistence across restarts |
| `postgres` / `pg` | `PruningPostgresSaver` | Requires `POSTGRES_CONNECTION_STRING` env var |
| `mongo` / `mongodb` | `PruningMongoDBSaver` | Requires `MONGO_CONNECTION_STRING` env var |
| `auto` | Auto-detected | Tries postgres → mongo → memory based on env vars |

```yaml
checkpoint_config:
  type: postgres
  keep_last_n: 10      # Prune to last 10 checkpoints per thread
```

**Fallback behaviour:** If the requested backend is unavailable, the factory falls back to `MemorySaver` with a warning. The graph always compiles.

**Explicit override:**

```python
from bili.checkpointers.pg_checkpointer import get_pg_checkpointer

compiled = compile_mas(config)
graph = compiled.compile_graph(checkpointer=get_pg_checkpointer(keep_last_n=5))
```

### Per-Agent Middleware

Agents can configure middleware to intercept and modify their execution. Applies to tool-enabled agents only.

```yaml
agents:
  - agent_id: researcher
    role: researcher
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

> Middleware only applies to **tool-enabled agents**. If set on an agent without tools, a warning is logged and middleware is skipped.

---

## Execution

### MASExecutor

```python
from bili.aether import load_mas_from_yaml, MASExecutor, execute_mas
from langchain_core.messages import HumanMessage

# Full control
config = load_mas_from_yaml("simple_chain.yaml")
executor = MASExecutor(config, log_dir="logs")
executor.initialize()
result = executor.run({"messages": [HumanMessage(content="Test content")]})

print(result.get_summary())           # Concise text summary
print(result.get_formatted_output())  # Asterisk-bordered output

# Convenience function
result = execute_mas(config, {"messages": [HumanMessage(content="Test")]})
```

**MASExecutor Methods:**

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
| `run_id` | `str` | Alias for `execution_id` used in cross-log correlation |
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

# Skip result saving
python bili/aether/runtime/cli.py configs/simple_chain.yaml \
    --input "Test" --no-save
```

---

## Architecture

```
bili/aether/
├── schema/               # MAS configuration schemas
│   ├── agent_spec.py     # Domain-agnostic AgentSpec
│   ├── agent_presets.py  # Preset registry system
│   ├── mas_config.py     # MASConfig, Channel, WorkflowEdge
│   └── enums.py          # WorkflowType, CommunicationProtocol, OutputFormat
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
│   ├── execution_result.py # AgentExecutionResult, MASExecutionResult
│   ├── executor.py       # MASExecutor, execute_mas()
│   └── cli.py            # Runtime CLI tool
├── integration/          # bili-core integration layer
│   ├── role_registry.py  # RoleDefaults registry
│   ├── inheritance.py    # apply_inheritance() resolution logic
│   ├── checkpointer_factory.py # Checkpoint config → bili-core checkpointer
│   └── state_integration.py # State schema extension hook
└── validation/           # Static MAS validation engine
    └── engine.py         # 13 structural validation checks
```
