"""Provider registry — maps provider-type strings to ``LLMProvider`` classes.

The registry is the authoritative routing table for :func:`load_model`.
Registration of built-in providers happens in
:mod:`bili.iris.providers.builtin` at module import time; external code can
register additional providers at application startup via
:func:`register_provider`.

Thread-safety
-------------
The registry is not thread-safe for concurrent mutation.  All registrations
must happen at application startup before the application begins serving
requests.  This mirrors the same constraint documented for
``GRAPH_NODE_REGISTRY`` in ``bili/iris/loaders/langchain_loader.py``.
"""

import logging
from typing import Dict, Optional, Type

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


class ProviderRegistry:
    """Maps provider-type strings to :class:`~bili.iris.providers.base.LLMProvider` classes.

    The registry stores the *class*, not an instance.  Callers instantiate
    and invoke :meth:`~bili.iris.providers.base.LLMProvider.load` themselves
    (or via the convenience :func:`load_model` function in
    :mod:`bili.iris.loaders.llm_loader`).

    Example::

        registry = ProviderRegistry()
        registry.register("remote_openai", OpenAIProvider)
        ProviderClass = registry.get("remote_openai")
        llm = ProviderClass().load(model_name="gpt-4o", max_tokens=1024)
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Type[LLMProvider]] = {}

    def register(self, provider_type: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider class for a given provider-type string.

        :param provider_type: The provider-type key used in ``LLM_MODELS``
            and passed to :func:`~bili.iris.loaders.llm_loader.load_model`
            (e.g. ``"remote_aws_bedrock"``).
        :param provider_class: A concrete subclass of :class:`LLMProvider`.
        :raises TypeError: If ``provider_class`` is not a subclass of
            :class:`LLMProvider`.
        :raises ValueError: If ``provider_type`` is already registered.
            Call :meth:`unregister` first to replace a built-in.

        Note:
            Registration should happen at application startup, before the
            application begins serving requests.  The registry is not
            thread-safe for concurrent mutation.
        """
        if not (
            isinstance(provider_class, type) and issubclass(provider_class, LLMProvider)
        ):
            raise TypeError(
                f"provider_class must be a subclass of LLMProvider, got {provider_class!r}"
            )
        if provider_type in self._providers:
            raise ValueError(
                f"Provider type '{provider_type}' is already registered. "
                "Call unregister() first to replace it."
            )
        self._providers[provider_type] = provider_class
        LOGGER.debug(
            "Registered provider '%s' → %s", provider_type, provider_class.__name__
        )

    def unregister(self, provider_type: str) -> None:
        """Remove a provider registration.

        Primarily useful for test teardown or intentional replacement of a
        built-in provider.

        :param provider_type: The provider-type key to remove.
        :raises KeyError: If ``provider_type`` is not registered.
        """
        if provider_type not in self._providers:
            raise KeyError(
                f"Provider type '{provider_type}' is not registered. "
                f"Available: {sorted(self._providers)}"
            )
        del self._providers[provider_type]
        LOGGER.debug("Unregistered provider '%s'", provider_type)

    def get(self, provider_type: str) -> Optional[Type[LLMProvider]]:
        """Return the provider class for a type string, or ``None``.

        :param provider_type: The provider-type key to look up.
        :returns: The registered :class:`LLMProvider` subclass, or ``None``
            if no provider is registered for that type.
        """
        return self._providers.get(provider_type)

    def get_or_raise(self, provider_type: str) -> Type[LLMProvider]:
        """Return the provider class, raising ``ValueError`` if not found.

        :param provider_type: The provider-type key to look up.
        :returns: The registered :class:`LLMProvider` subclass.
        :raises ValueError: If ``provider_type`` is not registered.
        """
        provider = self._providers.get(provider_type)
        if provider is None:
            available = sorted(self._providers)
            raise ValueError(
                f"Invalid model type: '{provider_type}'. "
                f"Registered types: {available}. "
                "To add a new provider, call register_provider() at application startup."
            )
        return provider

    def list_types(self) -> list:
        """Return a sorted list of registered provider-type strings."""
        return sorted(self._providers)

    def __contains__(self, provider_type: str) -> bool:
        return provider_type in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        return f"ProviderRegistry({self.list_types()})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: The global provider registry instance.  Built-in providers are registered
#: by importing :mod:`bili.iris.providers.builtin`.  External providers can
#: be added at startup via :func:`register_provider`.
PROVIDER_REGISTRY = ProviderRegistry()


def register_provider(provider_type: str, provider_class: Type[LLMProvider]) -> None:
    """Register a provider class in the global registry.

    Convenience wrapper around :meth:`ProviderRegistry.register` targeting
    the module-level :data:`PROVIDER_REGISTRY` singleton.

    :param provider_type: Provider-type key (e.g. ``"remote_my_api"``).
    :param provider_class: A concrete :class:`LLMProvider` subclass.

    Example::

        from bili.iris.providers import register_provider
        from mypackage.my_provider import MyProvider

        register_provider("remote_my_api", MyProvider)
    """
    PROVIDER_REGISTRY.register(provider_type, provider_class)


def get_provider(provider_type: str) -> Optional[Type[LLMProvider]]:
    """Look up a provider class in the global registry.

    Returns ``None`` if the provider type is not registered.  Prefer
    :func:`~bili.iris.loaders.llm_loader.load_model` for typical usage.

    :param provider_type: Provider-type key to look up.
    :returns: The :class:`LLMProvider` subclass or ``None``.
    """
    return PROVIDER_REGISTRY.get(provider_type)
