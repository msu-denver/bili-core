# AETHER Example Workflows

Feature coverage matrix and descriptions for all 12 included example YAML configurations.

Example YAML files live in [`bili/aether/config/examples/`](../config/examples/).

---

## Feature Coverage Matrix

| Example | Domain | Workflow | Channels | Output | Special Features |
|---------|--------|----------|----------|--------|------------------|
| `simple_chain.yaml` | Content Mod | **Sequential** | Direct | Text | Basic pipeline |
| `hierarchical_voting.yaml` | Content Mod | **Hierarchical** | Direct, Competitive | Text | Tier-based voting |
| `supervisor_moderation.yaml` | Content Mod | **Supervisor** | Direct, Req-Resp | Text | Hub-and-spoke routing |
| `consensus_network.yaml` | Content Mod | **Consensus** | Consensus | Text | Peer deliberation |
| `custom_escalation.yaml` | Content Mod | **Custom** | Direct | Text | Conditional edges |
| `research_analysis.yaml` | Research | **Sequential** | Direct | Text | Multi-step analysis |
| `inherited_research.yaml` | Research | **Sequential** | Direct | Text | **bili-core Inheritance** |
| `code_review.yaml` | Software Eng | **Supervisor** | Direct (6) | Text | Specialist routing |
| `middleware_checkpointer.yaml` | Technical | **Sequential** | Direct | Text | **Middleware**, **Checkpointer** |
| `parallel_processing.yaml` | Data Analysis | **Parallel** | **Broadcast**, **PubSub** | Text | Concurrent execution |
| `deliberative_escalation.yaml` | Customer Support | **Deliberative** | Direct, Req-Resp | **JSON** | **Human-in-Loop** |
| `structured_consensus.yaml` | Peer Review | **Consensus** | Consensus (4) | **JSON**, **Structured** | **Vote Fields**, Schema validation |

### Legend

**Workflow Types:**
- **Sequential** — Linear chain (A → B → C)
- **Hierarchical** — Tier-based with aggregation
- **Supervisor** — Central router dispatches to specialists
- **Consensus** — Peer deliberation until agreement
- **Parallel** — All agents execute concurrently
- **Deliberative** — Custom workflow with escalation
- **Custom** — User-defined edges and conditions

**Channel Protocols:**
- **Direct** — Point-to-point (A → B)
- **Broadcast** — One-to-many (A → All)
- **PubSub** — Publish-subscribe pattern
- **Req-Resp** — Request-response (bidirectional)
- **Competitive** — Adversarial debate
- **Consensus** — Peer deliberation

**Output Formats:**
- **Text** — Plain text responses (default)
- **JSON** — Structured JSON objects
- **Structured** — Schema-validated JSON

**Special Features:**
- **Middleware** — Per-agent middleware (summarization, call limits)
- **Checkpointer** — State persistence configuration
- **Inheritance** — Inherit defaults from bili-core role registry
- **Human-in-Loop** — Pause execution for human approval
- **Vote Fields** — Extract consensus votes from structured output

---

## Examples by Domain

### Content Moderation (5 examples)

**`simple_chain.yaml`** — Sequential Review Pipeline

The foundational example. Four agents in a linear chain: `community_manager` flags content, `content_reviewer` applies community standards, `policy_expert` checks usage policies, and `judge` delivers the final verdict. Three `direct` channels connect each agent to the next. A good starting point for any sequential pipeline.

**`hierarchical_voting.yaml`** — Tiered Debate and Voting

Demonstrates the `hierarchical` workflow type. Agents are assigned `tier` values (1 = highest authority). Lower-tier agents debate via `competitive` channels; verdicts propagate upward through `direct` channels to the tier-1 judge. Shows how `hierarchical_voting: true` and `min_debate_rounds` enable structured adversarial deliberation.

**`supervisor_moderation.yaml`** — Supervisor-Routed Specialists

Demonstrates the `supervisor` workflow type. A `judge` agent (marked `is_supervisor: true`) acts as the entry point and dynamically routes to `content_reviewer`, `policy_expert`, and `appeals_specialist` via `request_response` channels. The supervisor queries specialists as needed rather than always running all agents.

**`consensus_network.yaml`** — Peer Deliberation

Demonstrates the `consensus` workflow type with a fully-connected mesh of three agents. All channels use the `consensus` protocol and are `bidirectional: true`. `consensus_threshold: 0.66` and `max_consensus_rounds: 5` control when the loop terminates. Each agent sets `consensus_vote_field: decision` so the framework can extract votes from structured output.

**`custom_escalation.yaml`** — Conditional Escalation

Demonstrates the `custom` workflow type with `workflow_edges`. A `community_manager` fans out in parallel to `content_reviewer` and `policy_expert` (two simultaneous edges). Both feed into a `judge` that either decides (`state.confidence >= 0.7`) or escalates (`state.confidence < 0.7`) to an `appeals_specialist`. Also demonstrates `human_in_loop: true` with an escalation condition.

---

### Research (2 examples)

**`research_analysis.yaml`** — Multi-Step Research Pipeline

A sequential four-agent research pipeline: `researcher` gathers information, `fact_checker` validates it, `analyst` interprets the findings, and `synthesizer` produces a final report. Shows how to compose a deep-analysis pipeline with clear handoffs.

**`inherited_research.yaml`** — bili-core Inheritance

The same research pipeline as above, but agents use `inherit_from_bili_core: true` to pull system prompts, tools, and temperature defaults from the bili-core role registry. Demonstrates the `inherit_*` flags and how agent definitions can be kept minimal while leveraging existing bili-core configuration.

---

### Software Engineering (1 example)

**`code_review.yaml`** — Lead Engineer + Specialists

A `supervisor` workflow with a `lead_engineer` as the central router. Three specialist reviewers (`security_reviewer`, `performance_reviewer`, `style_reviewer`) are connected via six `direct` channels (three routing down, three feedback up). Shows how to model a real-world code review process with role-based specialist dispatch.

---

### Data Analysis (1 example)

**`parallel_processing.yaml`** — Concurrent Statistical Analysis

A `parallel` workflow with a `coordinator`, three concurrent processors (`statistical_processor`, `trend_processor`, `quality_processor`), and an `aggregator`. Uses a `broadcast` channel from coordinator to all processors and three `pubsub` channels from processors to aggregator. All three processors run simultaneously, making this the example to study for concurrent execution.

---

### Customer Support (1 example)

**`deliberative_escalation.yaml`** — Ticket Classification with Human Escalation

A `deliberative` workflow where a `ticket_classifier` and `support_specialist` handle customer queries. Uses `request_response` channels for back-and-forth consultation. `human_in_loop: true` with a condition expression demonstrates how AETHER pauses execution for human review when confidence is low. Agents produce `json` output to enable structured escalation decisions.

---

### Peer Review (1 example)

**`structured_consensus.yaml`** — Research Paper Review

A `consensus` workflow with four reviewer agents connected by a fully-connected mesh of four `consensus` channels. Each agent uses `output_format: structured` with an `output_schema` that enforces `decision`, `confidence`, and `reasoning` fields. The `consensus_vote_field: decision` setting enables the framework to extract votes automatically. Shows schema validation and vote extraction working together.

---

### Technical Demos (1 example)

**`middleware_checkpointer.yaml`** — Middleware + Persistence

A sequential pipeline explicitly configured to demonstrate middleware and checkpoint features. The `researcher` agent uses `summarization` middleware; the `analyst` uses `model_call_limit` middleware. `checkpoint_config.type: auto` enables auto-detection of available backends (postgres → mongo → memory). Use this example when testing middleware parameters or checkpoint configurations.

---

## Running an Example

```bash
# CLI — run any example
python bili/aether/runtime/cli.py \
    bili/aether/config/examples/simple_chain.yaml \
    --input "Review this content for policy violations"

# Python — run and inspect results
from bili.aether import load_mas_from_yaml, execute_mas
from langchain_core.messages import HumanMessage

config = load_mas_from_yaml("bili/aether/config/examples/supervisor_moderation.yaml")
result = execute_mas(config, {"messages": [HumanMessage(content="Analyze this post")]})
print(result.get_summary())
```

See [compiler.md](compiler.md) for `MASExecutor` options and [configuration.md](configuration.md) for field references.
