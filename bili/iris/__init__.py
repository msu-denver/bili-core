"""IRIS — Interactive Reasoning and Integration Services.

Single-agent orchestration framework. Provides LLM configuration for 97
model configurations across 17 provider types (11 remote API providers:
AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, Anthropic, Mistral AI,
Cohere, Google Generative AI, DeepSeek, xAI, Groq; 3 CLI presets: Claude
Code, Codex CLI, Gemini CLI; generic CLI subprocess; and 2 local providers:
llama.cpp, HuggingFace), a node-based workflow pipeline with native and
prompted ReAct tool-calling paths, extensible tool system, middleware
framework, and state persistence via checkpointers.

Provider abstraction
--------------------
``bili.iris.providers`` exposes the ``LLMProvider`` abstract base class and
``ProviderRegistry`` that unify all provider shapes behind a single
``.invoke()`` contract.  Third-party providers can be registered at
application startup via :func:`bili.iris.providers.register_provider`.
"""
