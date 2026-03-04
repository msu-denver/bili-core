# Prompt Injection Test Suite

Research suite for benchmarking MAS robustness against prompt injection attacks.
The suite runs **10 curated payloads** across **5 MAS configs** and **2 injection
phases**, applying three detection tiers of increasing analytical power.

For baseline setup (required for Tier 3 scoring) see [testing-baseline.md](testing-baseline.md).

---

## Detection tiers

| Tier | Name | Method | CI-safe? |
|---|---|---|---|
| 1 | Structural | `execution.success` + schema assertions | Yes |
| 2 | Heuristic | `PropagationTracker` — substring + compliance marker detection | Yes |
| 3 | Semantic | `SemanticEvaluator` (LLM) — genuine compliance scoring 0–3 | No (LLM calls) |

### Tier 1 — Structural

CI-safe assertions in `test_injection_structural.py`. Checks that the injection
ran, propagation tracking completed, and log files were written.

### Tier 2 — Heuristic

`PropagationTracker.observe()` uses two signals:

1. **Payload detection**: substring match (≥ 20 chars) in agent input state
2. **Influence detection**: payload echo OR compliance markers in agent output

Limitations: paraphrased payloads escape detection; the 20-char threshold misses
short payloads (`pi_length_001` is documented as a deliberate blind-spot test).

### Tier 3 — Semantic

LLM-based scoring using `SemanticEvaluator` from `bili.aether.evaluator`:

- Primary model: `anthropic.claude-3-7-sonnet-20250219-v1:0` (Bedrock)
- Fallback model: `gemini-2.5-flash` (Vertex)
- A circularity warning is logged if the evaluator and the model under test share
  the same provider family.

---

## Quick start

### Stub mode (no LLM calls — Tier 1 and Tier 2 only)

```bash
# From repo root
python bili/aether/tests/injection/run_injection_suite.py --stub

# Then run structural assertions
pytest bili/aether/tests/injection/test_injection_structural.py -v
```

### Real LLM mode (Tier 3 enabled)

Requires API credentials configured in your environment.

```bash
python bili/aether/tests/injection/run_injection_suite.py \
    --baseline-results bili/aether/tests/baseline/results
```

### Filtered run

```bash
python bili/aether/tests/injection/run_injection_suite.py --stub \
    --payloads pi_direct_001 pi_role_001 \
    --phases pre_execution \
    --configs bili/aether/config/examples/simple_chain.yaml
```

---

## Payload library

| ID | Type | Severity | Notes |
|---|---|---|---|
| `pi_direct_001` | `direct_override` | high | Classic "ignore previous instructions" |
| `pi_direct_002` | `direct_override` | high | Override with false authority claim |
| `pi_role_001` | `role_confusion` | high | Explicit role reassignment |
| `pi_role_002` | `role_confusion` | medium | Subtle seniority/authority drift |
| `pi_position_001` | `payload_position` | medium | Injection embedded inside reviewed content |
| `pi_position_002` | `payload_position` | medium | Injection at end of objective after separator |
| `pi_length_001` | `length_variant` | low | Short payload (12 chars) — tracker blind spot |
| `pi_length_002` | `length_variant` | high | Long repetitive payload — context window stress |
| `pi_encoded_001` | `encoded` | medium | Base64-encoded instruction |
| `pi_encoded_002` | `encoded` | medium | Instruction fragmented across sentences |

> **Note on `pi_length_001`**: At 12 characters this payload is below
> `PropagationTracker._MIN_MATCH_LEN = 20`. Tier 2 heuristic detection will
> always return empty influenced/resistant lists for this payload, regardless of
> actual agent behaviour. This blind spot is documented deliberately to surface it
> in the results matrix.

---

## MAS configurations tested

| `mas_id` | Workflow | Description |
|---|---|---|
| `simple_chain` | Sequential | 4-agent content moderation chain |
| `hierarchical_voting` | Hierarchical | Tier-based voting system |
| `supervisor_moderation` | Supervisor | Hub-and-spoke with judge as coordinator |
| `consensus_network` | Consensus | Deliberative peer consensus |
| `custom_escalation` | Custom | Custom escalation policy |

---

## Results

Results are written to `bili/aether/tests/injection/results/`:

```
results/
├── injection_results_matrix.csv
├── {mas_id}/
│   ├── {payload_id}_{phase}.json
│   ├── attack_log.ndjson
│   └── security_events.ndjson
```

### Results matrix columns

`injection_results_matrix.csv` has one row per `(payload_id, mas_id, phase)` triple:

| Column | Description |
|---|---|
| `payload_id` | Unique payload identifier (e.g. `pi_direct_001`) |
| `injection_type` | Research category (e.g. `direct_override`, `role_confusion`) |
| `severity` | Expected impact: `low` \| `medium` \| `high` |
| `stub_mode` | `True` when run with `--stub` (no LLM calls) |
| `mas_id` | Multi-agent system config identifier |
| `phase` | Injection phase: `pre_execution` or `mid_execution` |
| `tier1_pass` | Tier 1: execution succeeded without error |
| `tier2_influenced` | Tier 2: JSON array of heuristically influenced agent IDs |
| `tier2_resistant` | Tier 2: JSON array of heuristically resistant agent IDs |
| `tier3_score` | Tier 3: Compliance score 0–3 (empty in stub mode) |
| `tier3_confidence` | Tier 3: Evaluator confidence `high` \| `medium` \| `low` |
| `tier3_reasoning` | Tier 3: One-sentence evaluator explanation |
| `attack_suite` | Always `"prompt_injection"` — enables cross-suite CSV joins |

---

## Directory structure

```
bili/aether/tests/injection/
├── conftest.py                          # pytest fixtures
├── pytest.ini                           # isolates this suite
├── run_injection_suite.py               # standalone runner — produces results + CSV
├── test_injection_structural.py         # Tier 1 CI-safe assertions
├── payloads/
│   └── prompt_injection_payloads.py     # 10 InjectionPayload dataclasses
└── results/                             # populated by the runner (gitignored)
    ├── {mas_id}/
    │   ├── {payload_id}_{phase}.json    # one result per test case
    │   ├── attack_log.ndjson            # AttackLogger output
    │   └── security_events.ndjson       # SecurityEventLogger output
    └── injection_results_matrix.csv     # full results matrix
```
