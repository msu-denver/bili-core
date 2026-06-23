"""IRIS — Interactive Reasoning and Integration Services.

Single-agent orchestration framework. Provides LLM configuration for 60+
models across 6 providers (AWS Bedrock, Google Vertex AI, Azure OpenAI,
OpenAI, Ollama, local), a node-based workflow pipeline, extensible tool
system, middleware framework, and state persistence via checkpointers.

Provider abstraction
--------------------
``bili.iris.providers`` exposes the ``LLMProvider`` abstract base class and
``ProviderRegistry`` that unify all provider shapes behind a single
``.invoke()`` contract.  Third-party providers can be registered at
application startup via :func:`bili.iris.providers.register_provider`.
"""
