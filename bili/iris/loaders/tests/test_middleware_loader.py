"""Tests for bili.iris.loaders.middleware_loader public API."""

from unittest.mock import patch

import pytest

from bili.iris.loaders.middleware_loader import (
    MIDDLEWARE_REGISTRY,
    initialize_middleware,
)

# ---------------------------------------------------------------------------
# MIDDLEWARE_REGISTRY
# ---------------------------------------------------------------------------


class TestMiddlewareRegistry:
    """Verify the middleware registry structure."""

    def test_registry_is_a_dict(self):
        """Verify MIDDLEWARE_REGISTRY is a dictionary."""
        assert isinstance(MIDDLEWARE_REGISTRY, dict)

    def test_registry_has_expected_keys(self):
        """Verify expected middleware keys are present."""
        assert "summarization" in MIDDLEWARE_REGISTRY
        assert "model_call_limit" in MIDDLEWARE_REGISTRY

    def test_registry_values_are_callable(self):
        """Verify all registry values are callable factories."""
        for name, factory in MIDDLEWARE_REGISTRY.items():
            assert callable(factory), f"Middleware '{name}' is not callable"


# ---------------------------------------------------------------------------
# initialize_middleware
# ---------------------------------------------------------------------------


class TestInitializeMiddleware:
    """Test the initialize_middleware public function."""

    def test_returns_empty_list_for_none(self):
        """Verify None active_middleware returns an empty list."""
        assert not initialize_middleware(active_middleware=None)

    def test_returns_empty_list_for_empty_list(self):
        """Verify empty active_middleware returns an empty list."""
        assert not initialize_middleware(active_middleware=[])

    def test_raises_for_unknown_middleware(self):
        """Verify unknown middleware name raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            initialize_middleware(active_middleware=["nonexistent_middleware_xyz"])

    def test_returns_list_type(self):
        """Verify the return type is a list."""
        result = initialize_middleware(active_middleware=[])
        assert isinstance(result, list)

    def test_initializes_known_middleware(self):
        """Verify known middleware is initialized and returned."""
        result = initialize_middleware(
            active_middleware=["model_call_limit"],
            middleware_params={"model_call_limit": {"run_limit": 5}},
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_returns_empty_when_middleware_unavailable(self):
        """When middleware is unavailable, should return empty list."""
        with patch(
            "bili.iris.loaders.middleware_loader" ".LANGCHAIN_MIDDLEWARE_AVAILABLE",
            False,
        ):
            result = initialize_middleware(active_middleware=["summarization"])
            assert not result
