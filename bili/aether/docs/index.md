# AETHER Documentation

AETHER is a research platform for declaratively configuring, testing, and hardening
multi-agent LLM systems. See the [main README](../README.md) for an overview and
end-to-end usage examples.

---

## Getting Started

| Doc | Contents |
|---|---|
| [quickstart-stub.md](quickstart-stub.md) | Full workflow with stub agents — no API keys or model files required. **Start here.** |
| [quickstart.md](quickstart.md) | Swap in real LLM calls (OpenAI, AWS Bedrock, Google Vertex) |
| [quickstart-local.md](quickstart-local.md) | Use local models (LlamaCPP GGUF, HuggingFace) — no API keys |
| [security-testing-quickstart.md](security-testing-quickstart.md) | End-to-end guide to all five adversarial test suites — suite selection, three-tier detection, customisation, cross-suite analysis |

## Reference

| Doc | Contents |
|---|---|
| [configuration.md](configuration.md) | Agent & MAS field reference, YAML examples, preset system, validation rules |
| [compiler.md](compiler.md) | Workflow types & graph topologies, state schema, bili-core inheritance, communication framework, execution API |
| [examples.md](examples.md) | Feature coverage matrix, descriptions of all 12 example configurations |
| [cli.md](cli.md) | CLI flag reference, verbatim expected output for every mode |

## Security & Attack Research

| Doc | Contents |
|---|---|
| [attack-framework.md](attack-framework.md) | Attack injection framework — attack types, injection phases, propagation tracking |
| [security-logging.md](security-logging.md) | Security event detection & logging, cross-log correlation |

## Testing

| Doc | Contents |
|---|---|
| [testing-baseline.md](testing-baseline.md) | Baseline test suite — reproducibility requirements, stub and real-LLM modes, result file format |
| [testing-injection.md](testing-injection.md) | Prompt injection suite — 15 payloads × 10 injection types × 5 configs × 2 phases, three-tier detection, customisation |
| [testing-jailbreak.md](testing-jailbreak.md) | Jailbreak suite — 15 payloads × 10 injection types, role-abandonment Tier 3 rubric, customisation |
| [testing-memory-poisoning.md](testing-memory-poisoning.md) | Memory poisoning suite — 15 payloads × 10 injection types, context-acceptance Tier 3 rubric, customisation |
| [testing-bias-inheritance.md](testing-bias-inheritance.md) | Bias inheritance suite — 15 payloads × 10 injection types, directional drift Tier 3 rubric, customisation |
| [testing-agent-impersonation.md](testing-agent-impersonation.md) | Agent impersonation suite — 15 payloads × 7 injection types, identity-adoption Tier 3 rubric, customisation |

## Tools

| Doc | Contents |
|---|---|
| [ui.md](ui.md) | Streamlit UI — MAS graph visualizer and multi-turn chat interface; installation, navigation, features, troubleshooting |
