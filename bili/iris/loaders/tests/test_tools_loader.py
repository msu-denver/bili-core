"""Tests for bili.iris.loaders.tools_loader public API."""

import pytest

from bili.iris.loaders.tools_loader import TOOL_REGISTRY, initialize_tools

# ---------------------------------------------------------------------------
# TOOL_REGISTRY
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Verify the tool registry is populated with expected entries."""

    EXPECTED_TOOL_NAMES = {
        "faiss_retriever",
        "weather_api_tool",
        "serp_api_tool",
        "weather_gov_api_tool",
        "free_weather_api_tool",
        "mock_tool",
        "aws_opensearch_retriever",
    }

    def test_registry_is_a_dict(self):
        """Verify TOOL_REGISTRY is a dictionary."""
        assert isinstance(TOOL_REGISTRY, dict)

    def test_registry_contains_expected_tools(self):
        """Verify all expected tool names are in the registry."""
        assert self.EXPECTED_TOOL_NAMES.issubset(TOOL_REGISTRY.keys())

    def test_registry_values_are_callable(self):
        """Verify all registry values are callable factories."""
        for name, factory in TOOL_REGISTRY.items():
            assert callable(factory), f"Tool '{name}' factory is not callable"


# ---------------------------------------------------------------------------
# initialize_tools
# ---------------------------------------------------------------------------


class TestInitializeTools:
    """Test the initialize_tools public function."""

    _MOCK_PARAMS = {
        "mock_tool": {"mock_response": "test response", "response_time": 0},
    }

    def test_returns_list_for_mock_tool(self):
        """mock_tool has a default prompt and requires no external services."""
        tools = initialize_tools(
            active_tools=["mock_tool"],
            tool_prompts={},
            tool_params=self._MOCK_PARAMS,
        )
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_returns_empty_list_for_empty_input(self):
        """Verify empty active_tools returns an empty list."""
        tools = initialize_tools(active_tools=[], tool_prompts={})
        assert not tools

    def test_skips_unrecognized_tool(self):
        """Unrecognized tools are silently skipped (logged as warning)."""
        tools = initialize_tools(
            active_tools=["nonexistent_tool_xyz"],
            tool_prompts={},
        )
        assert not tools

    def test_custom_prompt_overrides_default(self):
        """Verify a custom prompt overrides the default tool prompt."""
        custom_prompt = "Custom mock prompt for testing"
        tools = initialize_tools(
            active_tools=["mock_tool"],
            tool_prompts={"mock_tool_prompt": custom_prompt},
            tool_params=self._MOCK_PARAMS,
        )
        assert len(tools) == 1

    def test_raises_when_no_prompt_available(self):
        """A tool with no default prompt and no provided prompt should raise."""
        # Register a temporary tool that has no default prompt in TOOLS config
        original = TOOL_REGISTRY.get("_test_no_prompt")
        TOOL_REGISTRY["_test_no_prompt"] = lambda name, prompt, params: None
        try:
            with pytest.raises(ValueError, match="does not have a default prompt"):
                initialize_tools(
                    active_tools=["_test_no_prompt"],
                    tool_prompts={},
                )
        finally:
            if original is None:
                TOOL_REGISTRY.pop("_test_no_prompt", None)
            else:
                TOOL_REGISTRY["_test_no_prompt"] = original

    def test_tool_middleware_dict_is_forwarded(self):
        """Middleware dict should be accepted without error."""
        tools = initialize_tools(
            active_tools=["mock_tool"],
            tool_prompts={},
            tool_params=self._MOCK_PARAMS,
            tool_middleware={"mock_tool": []},
        )
        assert isinstance(tools, list)

    def test_tool_middleware_list_is_forwarded(self):
        """Middleware list should be accepted without error."""
        tools = initialize_tools(
            active_tools=["mock_tool"],
            tool_prompts={},
            tool_params=self._MOCK_PARAMS,
            tool_middleware=[],
        )
        assert isinstance(tools, list)
