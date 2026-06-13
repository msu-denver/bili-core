# AEGIS-PROBE

**Persistent Reasoning Open-ended Black-box Evaluator** — an autonomous
adversarial agent suite for AEGIS. PROBE conducts multi-round, adaptive,
black-box red-teaming against AETHER multi-agent systems.

## What this measures

Existing AEGIS suites evaluate single-shot static payloads. PROBE evaluates
victim systems against an attacker that observes responses, plans across
turns, and adapts its payloads — the multi-round adaptive class of attacks
formalized by PAIR (Chao et al. 2023), TAP (Mehrotra et al. 2023), and
Crescendo (Russinovich et al. 2024).

The headline new measurement PROBE produces is **turns-to-compromise (TTC)**:
the number of dialogue turns required for the attacker to achieve its
objective. Short TTC indicates a brittle victim; long TTC (or a session
that exhausts budget without success) indicates resilience.

## Quick start

### Stub mode (no LLM calls)

```bash
python bili/aegis/suites/probe/run_probe_suite.py --stub
```

### Real-LLM mode (requires API credentials and a baseline)

```bash
# One objective, one policy, one config — start small
python bili/aegis/suites/probe/run_probe_suite.py \
    --policies pair \
    --objectives pr_misinfo_001 \
    --configs bili/aether/config/examples/simple_chain.yaml \
    --baseline-results bili/aegis/suites/baseline/results

# Full smoke test
python bili/aegis/suites/probe/run_probe_suite.py --smoke \
    --baseline-results bili/aegis/suites/baseline/results
```

### Structural pytest

```bash
pytest bili/aegis/suites/probe/test_probe_structural.py -v
```

## Cost expectations

PROBE is the most cost-intensive AEGIS suite by a wide margin. Rough estimate
for the full default run (5 objectives × 5 configs × 3 policies × 1 seed,
~10 turns/session, 2 LLM calls/turn): ~1,500 LLM calls. At 2026 prices for
a small open-weight attacker plus a Sonnet-class judge, expect ~$30–80 for
the full run, depending on policy mix (TAP is the most expensive).

Budget controls are enforced at four axes (turns, tokens, wall-clock, cost).
Sessions that hit any limit are recorded as `budget_exceeded` and contribute
a row to the CSV with that termination reason.

## Output

- `results/probe_results_matrix.csv` — flat results matrix matching the
  cross-suite AEGIS schema, with PROBE-specific columns appended
- `results/{mas_id}/sessions/{session_id}.json` — full multi-turn trajectory
  per session

## Cross-suite analysis

PROBE rows can be combined with the other five AEGIS suites:

```python
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob("bili/aegis/suites/*/results/*_results_matrix.csv")]
combined = pd.concat(dfs, ignore_index=True)
combined.groupby("attack_suite")["tier3_score"].describe()
```

## Design

- [`bili/aegis/docs/probe-rfc.md`](../../docs/probe-rfc.md) — full RFC:
  motivation, prior art (PAIR / TAP / Crescendo / GOAT), architecture,
  evaluation plan, risks, alternatives, acceptance criteria.

## Status

**v0.1 implementation complete.** The suite runs end-to-end against a
deterministic in-process fake LLM (covered by 344+ unit tests, all
passing, pylint 10.00/10 across PROBE source + tests).

A documented real-LLM smoke script is at
[scripts/aegis/run_probe_smoke.ps1](../../../../scripts/aegis/run_probe_smoke.ps1)
which runs PAIR against `probe_victim_claude.yaml` using a DeepSeek + Claude +
Gemini cross-provider trio with a `--budget-cost-usd 0.50` per-session
cap. Run it with your own provider credentials to validate the
infrastructure end-to-end.

Remaining v0.2+ work (per the RFC):
- LLM-driven qualitative `VictimObserverNode` summary
- AETHER YAML expression of the attacker MAS (currently plain Python loop)
- TAP off-topic-judge as a distinct evaluator (currently folded into
  NO_PROGRESS verdict)
- Backporting iterative attacks to the static AEGIS suites

Implementation status of individual components is documented in the
module docstrings.
