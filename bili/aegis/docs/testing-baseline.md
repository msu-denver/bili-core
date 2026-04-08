# Baseline Test Suite

The baseline suite records ground-truth outputs for each MAS configuration under
normal (non-injected) conditions. These results serve as the comparison anchor for
the prompt injection and jailbreak suites — the `SemanticEvaluator` diffs injected
outputs against baselines to produce per-agent compliance scores.

---

## Reproducibility requirements

Results are only comparable across runs when the following are identical:

- `model_name` — must match exactly per agent (e.g. `gpt-4o`, not `gpt-4`)
- `temperature` — must match per agent (check each agent block in the YAML)
- YAML config — verified via `config_fingerprint.yaml_hash` in result files
- Provider credentials — same provider must be used (Azure OpenAI and OpenAI
  produce different outputs for nominally the same model ID)

The `config_fingerprint` field in every result file records the YAML hash,
model name(s), and per-agent temperatures. If these differ between your
baseline run and a later injection test run, the `SemanticEvaluator` will log
a warning that comparisons may not be valid.

---

## Configuring a reproducible run

Before running `run_baseline.py`, set `model_name` and `temperature`
consistently in each MAS YAML:

```yaml
agents:
  - agent_id: community_manager
    model_name: gpt-4o      # must match across all baseline and injection runs
    temperature: 0.2        # must match across all baseline and injection runs
```

Do not change these values between your baseline run and your injection test runs.

---

## Running in stub mode

Stub mode uses the built-in stub agent (no LLM calls) so the structural suite
can be verified without API credentials. Stub results **must not** be used as
baselines for injection test comparison — they carry no semantic content.

```bash
# From repo root
python bili/aegis/tests/baseline/run_baseline.py --stub
```

Results are stored with `"stub_mode": true` and `"semantic_tier": "skipped"`.

---

## Running with real LLMs

```bash
# Ensure API credentials are set in .env (loaded by Docker Compose)
# or in your shell environment, then:
python bili/aegis/tests/baseline/run_baseline.py
```

This runs 20 prompts × 5 MAS configs = 100 executions and writes
`results/{mas_id}/{prompt_id}.json` for each.

To restrict to specific configs or prompts:

```bash
python bili/aegis/tests/baseline/run_baseline.py \
    --configs bili/aether/config/examples/simple_chain.yaml \
    --prompts benign_001 edge_003
```

---

## Running the structural pytest suite

```bash
# From repo root — runs against result files already in results/
python -m pytest bili/aegis/tests/baseline/ -v

# Or from the baseline directory (uses local pytest.ini):
cd bili/aegis/tests/baseline
python -m pytest -v
```

The structural suite is CI-safe: it asserts schema validity, non-empty outputs,
and metadata presence only. It is parametrized over result files — when
`results/` is empty, all tests are skipped with no errors.

---

## Result file format

Each `results/{mas_id}/{prompt_id}.json` file contains:

```json
{
  "prompt_id": "benign_001",
  "prompt_category": "benign",
  "prompt_text": "...",
  "mas_id": "simple_chain",
  "config_fingerprint": {
    "yaml_hash": "abc123def456",
    "model_name": "gpt-4o",
    "temperature": {
      "community_manager": 0.2,
      "content_reviewer": 0.2,
      "policy_expert": 0.1,
      "judge": 0.0
    }
  },
  "execution": {
    "success": true,
    "duration_ms": 4823.1,
    "agent_count": 4,
    "message_count": 3
  },
  "agent_outputs": {
    "community_manager": { "raw": "...", "parsed": null },
    "content_reviewer":  { "raw": "...", "parsed": null },
    "policy_expert":     { "raw": "...", "parsed": null },
    "judge":             { "raw": "...", "parsed": null }
  },
  "run_metadata": {
    "timestamp": "2026-03-01T12:00:00+00:00",
    "stub_mode": false,
    "semantic_tier": "skipped"
  }
}
```

`semantic_tier` is `"skipped"` for baseline runs. It is populated by the
`SemanticEvaluator` in the injection and jailbreak test suites.
