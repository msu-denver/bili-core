"""LLM provider abstraction layer for bili-core IRIS.

This subpackage defines the ``LLMProvider`` abstract base class and the
``ProviderRegistry`` that maps provider-type strings to their concrete
implementations.  It establishes the seam through which all provider
shapes — LangChain-native API chat models, subprocess/CLI providers, and
MCP-backed providers — are reached from a single ``load_model()`` call.

Public API
----------
- :class:`LLMProvider`  — abstract base all providers implement
- :class:`ProviderRegistry` — maps type strings to ``LLMProvider`` classes
- :data:`PROVIDER_REGISTRY` — the module-level singleton registry instance
- :func:`register_provider` — convenience for runtime provider registration
- :func:`get_provider` — look up a registered provider class
- :class:`FallbackLLM` — transparent proxy that falls through an ordered
  provider chain on retryable errors
- :class:`FallbackPolicy` — classifies exceptions as retryable vs fatal
- :class:`ProviderChain` — ordered ``(provider_type, kwargs)`` sequence
- :func:`build_fallback_llm` — wrap an existing LLM with a fallback chain
- :data:`DEFAULT_POLICY` — default name-based exception classification policy

Design contract
---------------
Every concrete provider MUST override ``load(**kwargs)`` and return any
object with a ``.invoke(messages)`` method.  Providers need not be
LangChain ``BaseChatModel`` subclasses; the only required interface is the
``.invoke()`` method (plus optional ``.stream()`` / ``.astream()``).

Lazy imports
------------
Heavy dependencies (cloud SDKs, torch, etc.) MUST be imported inside the
``load()`` method body, never at module scope.  This lets the providers
subpackage import without installing optional dependencies.
"""

# Import builtin after the public API is defined so the side-effect
# (populating PROVIDER_REGISTRY with the six built-in providers) runs after
# the registry singleton exists.  Any code that does
# `from bili.iris.providers import ...` — including the lazy import in
# llm_loader.py's else branch — triggers this registration automatically, so
# the registry is always populated on the production path without a separate
# explicit call.  The isort: skip comment keeps isort from hoisting this above
# the functional imports.
from . import (  # noqa: F401  pylint: disable=wrong-import-position  # isort: skip
    builtin as _builtin,
)
from .base import LLMProvider
from .fallback import (
    DEFAULT_POLICY,
    FallbackLLM,
    FallbackPolicy,
    ProviderChain,
    build_fallback_llm,
)
from .registry import (
    PROVIDER_REGISTRY,
    ProviderRegistry,
    get_provider,
    register_provider,
)

__all__ = [
    "LLMProvider",
    "ProviderRegistry",
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_provider",
    "FallbackLLM",
    "FallbackPolicy",
    "ProviderChain",
    "build_fallback_llm",
    "DEFAULT_POLICY",
]
