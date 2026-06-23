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

from .base import LLMProvider
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
]
