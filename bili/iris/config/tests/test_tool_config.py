"""Tests for bili.iris.config.tool_config structure and defaults."""

import pytest

from bili.iris.config.tool_config import TOOLS


class TestToolsConfig:
    """Verify the TOOLS configuration dictionary."""

    EXPECTED_TOOL_NAMES = {
        "local_faiss_retriever",
        "aws_opensearch_retriever",
        "weather_api_tool",
        "weather_gov_api_tool",
        "free_weather_api_tool",
        "serp_api_tool",
        "mock_tool",
    }

    def test_tools_is_non_empty_dict(self):
        """Verify TOOLS is a non-empty dictionary."""
        assert isinstance(TOOLS, dict)
        assert len(TOOLS) > 0

    def test_contains_expected_tools(self):
        """Verify all expected tool names are present."""
        assert self.EXPECTED_TOOL_NAMES.issubset(TOOLS.keys())

    @pytest.mark.parametrize("tool_name", list(TOOLS.keys()))
    def test_tool_has_description(self, tool_name):
        """Verify each tool has a string description field."""
        assert "description" in TOOLS[tool_name]
        assert isinstance(TOOLS[tool_name]["description"], str)

    @pytest.mark.parametrize("tool_name", list(TOOLS.keys()))
    def test_tool_has_enabled_flag(self, tool_name):
        """Verify each tool has a boolean enabled field."""
        assert "enabled" in TOOLS[tool_name]
        assert isinstance(TOOLS[tool_name]["enabled"], bool)

    @pytest.mark.parametrize("tool_name", list(TOOLS.keys()))
    def test_tool_has_default_prompt(self, tool_name):
        """Verify each tool has a non-empty default_prompt string."""
        assert "default_prompt" in TOOLS[tool_name]
        assert isinstance(TOOLS[tool_name]["default_prompt"], str)
        assert len(TOOLS[tool_name]["default_prompt"]) > 0
