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
| [testing-injection.md](testing-injection.md) | Prompt injection test suite — 10 payloads × 5 configs × 2 phases, three-tier detection, results matrix |
| [testing-jailbreak.md](testing-jailbreak.md) | Jailbreak test suite — payload taxonomy, role-abandonment scoring rubric, cross-suite CSV joins |

## Tools

| Doc | Contents |
|---|---|
| [ui.md](ui.md) | Streamlit UI — MAS graph visualizer and multi-turn chat interface; installation, navigation, features, troubleshooting |
