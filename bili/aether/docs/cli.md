# AETHER CLI Reference

Command-line tools for compiling and executing AETHER MAS configurations, with verbatim expected output for every mode.

> **Stub vs real LLM output**: All example YAMLs have no `model_name` set, so they run in stub mode. Stub agents print `[STUB] Agent '{agent_id}' ({role}) executed.` instead of real LLM responses. To see real output, add `model_name: gpt-4` (or any supported model) to agent definitions in the YAML. The output format is identical — only the agent output text changes.

---

## Compiler CLI

**Script:** `bili/aether/compiler/cli.py`

Loads a YAML config, compiles it to a LangGraph `StateGraph`, and reports results. No execution occurs — useful for validating a config before running it.

### Single file

```bash
python bili/aether/compiler/cli.py bili/aether/config/examples/simple_chain.yaml
```

**Expected output (3 lines):**
```
OK    bili/aether/config/examples/simple_chain.yaml
      CompiledMAS(simple_chain, 4 agents, workflow=sequential)
      Compiled: CompiledStateGraph
```

- Line 1: `OK` + the path you provided
- Line 2: `CompiledMAS({mas_id}, {n} agents, workflow={workflow_type})`
- Line 3: `Compiled: CompiledStateGraph` — confirms `compile_graph()` succeeded

On failure:
```
Traceback (most recent call last):
  ...
ValueError: Validation failed: ...
```
(Exception propagates — no `FAIL` prefix in single-file mode.)

### All examples

```bash
python bili/aether/compiler/cli.py
```

Compiles 7 bundled example configs. **Expected output (one line per file):**
```
OK    simple_chain.yaml  ->  CompiledMAS(simple_chain, 4 agents, workflow=sequential)
OK    hierarchical_voting.yaml  ->  CompiledMAS(hierarchical_voting, 5 agents, workflow=hierarchical)
OK    supervisor_moderation.yaml  ->  CompiledMAS(supervisor_moderation, 4 agents, workflow=supervisor)
OK    consensus_network.yaml  ->  CompiledMAS(consensus_network, 3 agents, workflow=consensus)
OK    custom_escalation.yaml  ->  CompiledMAS(custom_escalation, 5 agents, workflow=custom)
OK    research_analysis.yaml  ->  CompiledMAS(research_analysis, 4 agents, workflow=sequential)
OK    code_review.yaml  ->  CompiledMAS(code_review, 4 agents, workflow=supervisor)
```

If a file is missing: `SKIP  {filename}  (file not found)`
If compilation raises: `FAIL  {filename}  ->  {exception message}`

**Exit codes:** `0` if all pass, `1` if any fail.

---

## Runtime CLI

**Script:** `bili/aether/runtime/cli.py`

Executes a compiled MAS end-to-end and prints `get_formatted_output()` — a 60-asterisk bordered block.

### Flag reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `config_file` | — | required | Path to YAML MAS configuration |
| `--input` | `-i` | `None` | Input text for the MAS |
| `--input-file` | — | `None` | Path to a text file containing input |
| `--thread-id` | — | `None` | Thread ID for checkpointed execution |
| `--log-dir` | — | current dir | Directory for result JSON files |
| `--test-checkpoint` | — | `false` | Run checkpoint persistence test (save + restore) |
| `--test-cross-model` | — | `false` | Run cross-model transfer test |
| `--source-model` | — | required with `--test-cross-model` | Source model name (e.g. `gpt-4`) |
| `--target-model` | — | required with `--test-cross-model` | Target model name (e.g. `claude-sonnet-3-5-20241022`) |
| `--no-save` | — | `false` | Skip saving results to JSON file |

**Exit codes:** `0` on success, `1` on failure (or if either run fails in multi-run modes).

### Basic execution

```bash
python bili/aether/runtime/cli.py \
    bili/aether/config/examples/simple_chain.yaml \
    --input "Review this post for policy violations"
```

**Expected output (stub agents):**
```
************************************************************
* MAS Execution Result                                     *
*                                                          *
* MAS ID:        simple_chain                             *
* Execution ID:  simple_chain_a1b2c3d4                    *
* Status:        SUCCESS                                   *
* Duration:      12.45 ms                                 *
* Agents:        4                                         *
* Messages:      0                                         *
*                                                          *
* Agent Outputs:                                           *
*   community_manager (community_manager):                *
*     [STUB] Agent 'community_manager' (community_manage  *
*   content_reviewer (content_reviewer):                  *
*     [STUB] Agent 'content_reviewer' (content_reviewer)  *
*   policy_expert (policy_expert):                        *
*     [STUB] Agent 'policy_expert' (policy_expert) execut *
*   judge (judge):                                        *
*     [STUB] Agent 'judge' (judge) executed.              *
************************************************************
```

> The stub message `[STUB] Agent 'community_manager' (community_manager) executed.` is 55 chars — just within the 56-char content width. Longer agent IDs or roles will be truncated with no ellipsis (hard cut).

A result JSON is saved to `./{execution_id}.json` unless `--no-save` is passed.

### With input from file

```bash
python bili/aether/runtime/cli.py \
    bili/aether/config/examples/simple_chain.yaml \
    --input-file content.txt \
    --log-dir logs/
```

Same formatted output. Result JSON saved to `logs/{execution_id}.json`.

### Checkpoint persistence test

Tests that state can be saved to a checkpointer and restored in a second run.

```bash
python bili/aether/runtime/cli.py \
    bili/aether/config/examples/simple_chain.yaml \
    --input "Test payload" \
    --test-checkpoint \
    --thread-id my_thread_001
```

**Expected output:**
```
=== Original Run ===
************************************************************
* MAS Execution Result                                     *
*                                                          *
* MAS ID:        simple_chain                             *
* Execution ID:  simple_chain_a1b2c3d4                    *
* Status:        SUCCESS                                   *
* Duration:      18.32 ms                                 *
* Agents:        4                                         *
* Messages:      0                                         *
* Checkpoint:    saved                                     *
*                                                          *
* Agent Outputs:                                           *
*   ...                                                    *
************************************************************

=== Restored Run ===
************************************************************
* MAS Execution Result                                     *
*                                                          *
* MAS ID:        simple_chain                             *
* Execution ID:  simple_chain_e5f6g7h8                    *
* Status:        SUCCESS                                   *
* ...                                                      *
************************************************************
```

The first block shows `Checkpoint:    saved`. The second run reinitializes the executor and restores state from the same `thread-id`.

### Cross-model transfer test

Runs the same MAS twice with different model configurations and the same thread.

```bash
python bili/aether/runtime/cli.py \
    bili/aether/config/examples/simple_chain.yaml \
    --input "Test payload" \
    --test-cross-model \
    --source-model gpt-4 \
    --target-model claude-sonnet-3-5-20241022
```

**Expected output:**
```
=== Source Model Result ===
************************************************************
* MAS Execution Result                                     *
* ...                                                      *
************************************************************

=== Target Model Result ===
************************************************************
* MAS Execution Result                                     *
* ...                                                      *
************************************************************
```

Both runs use the same `thread-id` internally so checkpoint state can be shared between them.

---

## Attack & Security: Python-only

There is no CLI for attack injection or security event detection. Both are Python APIs:

```python
from bili.aether.attacks import AttackInjector, AttackType, InjectionPhase
from bili.aether.security import SecurityEventDetector, SecurityEventLogger
```

See [attack-framework.md](attack-framework.md) and [security-logging.md](security-logging.md) for usage.

---

## Exit Codes Summary

| Tool | Condition | Exit code |
|------|-----------|-----------|
| `compiler/cli.py` (single file) | Any exception during compile | non-zero (Python default) |
| `compiler/cli.py` (all examples) | All pass | `0` |
| `compiler/cli.py` (all examples) | Any `FAIL` | `1` |
| `runtime/cli.py` | `result.success` is `True` | `0` |
| `runtime/cli.py` | `result.success` is `False` | `1` |
| `runtime/cli.py` (checkpoint/cross-model) | Both runs succeed | `0` |
| `runtime/cli.py` (checkpoint/cross-model) | Either run fails | `1` |
