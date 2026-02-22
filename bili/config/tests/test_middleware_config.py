"""Tests for middleware configuration module.

Tests the middleware configuration helper functions:
- get_middleware_info: Retrieve middleware configuration
- get_enabled_middleware: Get list of enabled middleware
- get_middleware_defaults: Get default parameter values
"""

# pylint: disable=missing-function-docstring


from bili.config.middleware_config import (
    MIDDLEWARE,
    get_enabled_middleware,
    get_middleware_defaults,
    get_middleware_info,
)


class TestGetMiddlewareInfo:
    """Tests for get_middleware_info function."""

    def test_get_summarization_middleware(self):
        """Test retrieving summarization middleware configuration."""
        info = get_middleware_info("summarization")

        assert info is not None
        assert "description" in info
        assert "enabled" in info
        assert "params" in info
        assert "Automatically summarizes" in info["description"]

    def test_get_model_call_limit_middleware(self):
        """Test retrieving model_call_limit middleware configuration."""
        info = get_middleware_info("model_call_limit")

        assert info is not None
        assert "description" in info
        assert "enabled" in info
        assert "params" in info
        assert "Limits the maximum number" in info["description"]

    def test_get_nonexistent_middleware(self):
        """Test that nonexistent middleware returns None."""
        info = get_middleware_info("nonexistent_middleware")
        assert info is None

    def test_middleware_has_params(self):
        """Test that middleware configurations include parameter definitions."""
        for middleware_name in MIDDLEWARE:
            info = get_middleware_info(middleware_name)
            assert "params" in info
            assert isinstance(info["params"], dict)


class TestGetEnabledMiddleware:
    """Tests for get_enabled_middleware function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        enabled = get_enabled_middleware()
        assert isinstance(enabled, list)

    def test_no_middleware_enabled_by_default(self):
        """Test that no middleware is enabled by default in current config."""
        enabled = get_enabled_middleware()
        # Based on current MIDDLEWARE config, both are enabled=False
        assert len(enabled) == 0

    def test_only_enabled_middleware_returned(self):
        """Test that only middleware with enabled=True are returned."""
        # Temporarily modify MIDDLEWARE to test
        original_enabled = MIDDLEWARE["summarization"]["enabled"]
        MIDDLEWARE["summarization"]["enabled"] = True

        enabled = get_enabled_middleware()
        assert "summarization" in enabled

        # Restore original value
        MIDDLEWARE["summarization"]["enabled"] = original_enabled


class TestGetMiddlewareDefaults:
    """Tests for get_middleware_defaults function."""

    def test_get_summarization_defaults(self):
        """Test retrieving default parameters for summarization middleware."""
        defaults = get_middleware_defaults("summarization")

        assert isinstance(defaults, dict)
        assert "max_tokens_before_summary" in defaults
        assert "messages_to_keep" in defaults
        assert defaults["max_tokens_before_summary"] == 4000
        assert defaults["messages_to_keep"] == 20

    def test_get_model_call_limit_defaults(self):
        """Test retrieving default parameters for model_call_limit middleware."""
        defaults = get_middleware_defaults("model_call_limit")

        assert isinstance(defaults, dict)
        assert "thread_limit" in defaults
        assert "run_limit" in defaults
        assert "exit_behavior" in defaults
        assert defaults["thread_limit"] is None
        assert defaults["run_limit"] == 10
        assert defaults["exit_behavior"] == "end"

    def test_get_nonexistent_middleware_defaults(self):
        """Test that nonexistent middleware returns empty dict."""
        defaults = get_middleware_defaults("nonexistent_middleware")
        assert defaults == {}

    def test_all_params_have_defaults(self):
        """Test that all middleware parameters have default values."""
        for middleware_name in MIDDLEWARE:
            defaults = get_middleware_defaults(middleware_name)
            params = MIDDLEWARE[middleware_name]["params"]

            # All params should be in defaults
            assert len(defaults) == len(params)

            # All defaults should be defined (even if None)
            for param_name in params:
                assert param_name in defaults


class TestMiddlewareConfig:
    """Tests for MIDDLEWARE configuration structure."""

    def test_middleware_structure(self):
        """Test that MIDDLEWARE dictionary has expected structure."""
        assert "summarization" in MIDDLEWARE
        assert "model_call_limit" in MIDDLEWARE

    def test_each_middleware_has_required_fields(self):
        """Test that each middleware has description, enabled, and params."""
        for middleware_name, config in MIDDLEWARE.items():
            assert "description" in config, f"{middleware_name} missing description"
            assert "enabled" in config, f"{middleware_name} missing enabled flag"
            assert "params" in config, f"{middleware_name} missing params"
            assert isinstance(
                config["description"], str
            ), f"{middleware_name} description not string"
            assert isinstance(
                config["enabled"], bool
            ), f"{middleware_name} enabled not bool"
            assert isinstance(
                config["params"], dict
            ), f"{middleware_name} params not dict"

    def test_each_param_has_required_fields(self):
        """Test that each parameter has description, default, and type."""
        for middleware_name, config in MIDDLEWARE.items():
            for param_name, param_config in config["params"].items():
                assert (
                    "description" in param_config
                ), f"{middleware_name}.{param_name} missing description"
                assert (
                    "default" in param_config
                ), f"{middleware_name}.{param_name} missing default"
                assert (
                    "type" in param_config
                ), f"{middleware_name}.{param_name} missing type"

    def test_summarization_params(self):
        """Test specific parameters for summarization middleware."""
        params = MIDDLEWARE["summarization"]["params"]

        assert "max_tokens_before_summary" in params
        assert "messages_to_keep" in params

        assert params["max_tokens_before_summary"]["type"] == "int"
        assert params["messages_to_keep"]["type"] == "int"

    def test_model_call_limit_params(self):
        """Test specific parameters for model_call_limit middleware."""
        params = MIDDLEWARE["model_call_limit"]["params"]

        assert "thread_limit" in params
        assert "run_limit" in params
        assert "exit_behavior" in params

        assert params["thread_limit"]["type"] == "int"
        assert params["run_limit"]["type"] == "int"
        assert params["exit_behavior"]["type"] == "str"
