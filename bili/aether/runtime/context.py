"""RuntimeContext — dependency injection container for AETHER pipelines.

Holds named services (Celery workers, HTTP clients, model singletons,
etc.) that are injected into pipeline node builders via ``**kwargs``
during compilation.

Usage::

    from bili.aether.runtime.context import RuntimeContext

    ctx = RuntimeContext()
    ctx.register("celery_app", celery_app_instance)
    ctx.register("sentiment_model", loaded_model)

    executor = MASExecutor(config, runtime_context=ctx)
    executor.initialize()

Node builders receive services via ``**kwargs``::

    def build_sentiment_node(**kwargs):
        model = kwargs["sentiment_model"]

        def _execute(state: dict) -> dict:
            score = model.predict(state.get("utterance", ""))
            return {**state, "sentiment_score": score}

        return _execute
"""

import logging
from typing import Any, Dict, Iterator

LOGGER = logging.getLogger(__name__)


class RuntimeContext:
    """Dependency injection container for AETHER agent pipelines.

    Services are registered at setup time and injected into node kwargs
    during pipeline compilation.  Priority order when merged with other
    kwargs sources:

        ``node_spec.config``  >  parent agent config  >  ``RuntimeContext``

    This means node-specific YAML config overrides parent agent config
    which overrides runtime context values.
    """

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, service: Any) -> "RuntimeContext":
        """Register a named service.

        Args:
            name: Service name (used as the kwarg key in node builders).
            service: The service instance.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._services:
            raise ValueError(
                f"Service '{name}' is already registered. "
                "Call unregister() first to replace it."
            )
        self._services[name] = service
        LOGGER.debug("RuntimeContext: registered service '%s'", name)
        return self

    def unregister(self, name: str) -> "RuntimeContext":
        """Remove a named service.

        Args:
            name: Service name to remove.

        Returns:
            ``self`` for method chaining.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._services:
            raise KeyError(
                f"Service '{name}' is not registered. "
                f"Available: {sorted(self._services.keys())}"
            )
        del self._services[name]
        return self

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str, default: Any = None) -> Any:
        """Get a registered service by name, with optional default."""
        return self._services.get(name, default)

    def require(self, name: str) -> Any:
        """Get a registered service, raising if not found.

        Args:
            name: Service name.

        Returns:
            The registered service.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._services:
            raise KeyError(
                f"Required service '{name}' not registered in RuntimeContext. "
                f"Available: {sorted(self._services.keys())}"
            )
        return self._services[name]

    def as_dict(self) -> Dict[str, Any]:
        """Return all services as a dict for merging into node kwargs."""
        return dict(self._services)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._services

    def __len__(self) -> int:
        return len(self._services)

    def __iter__(self) -> Iterator[str]:
        return iter(self._services)

    def __repr__(self) -> str:
        names = sorted(self._services.keys())
        return f"RuntimeContext({names})"
