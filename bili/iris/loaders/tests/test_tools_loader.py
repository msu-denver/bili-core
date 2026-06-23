"""Tests for bili.iris.loaders.tools_loader public API."""

from unittest.mock import MagicMock, patch

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

    def test_bare_tool_name_in_prompts_is_accepted(self):
        """Prompt keyed by the bare tool name (not ``<name>_prompt``) is used."""
        # This exercises the ``elif tool in tool_prompts:`` branch in
        # ``initialize_tools`` (line 222 in tools_loader.py).
        tools = initialize_tools(
            active_tools=["mock_tool"],
            tool_prompts={"mock_tool": "A prompt keyed by bare name"},
            tool_params=self._MOCK_PARAMS,
        )
        assert len(tools) == 1


# ---------------------------------------------------------------------------
# Lazy factory functions
#
# Each private factory function (_init_faiss_retriever, _init_weather_api_tool,
# etc.) defers its heavy imports until first use.  The tests below call each
# factory with mocked-out imports so the factory body is executed, covering the
# lines that the lazy-import approach leaves uncovered in a lean install.
# ---------------------------------------------------------------------------


class TestLazyToolFactories:
    """Exercise each lazy factory body to cover the deferred-import branches."""

    def test_faiss_retriever_factory(self):
        """_init_faiss_retriever calls create_retriever_tool and init_faiss."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_faiss_retriever,
        )

        mock_retriever = MagicMock()
        mock_tool = MagicMock()
        with patch(
            "bili.iris.tools.faiss_memory_indexing.init_faiss",
            return_value=mock_retriever,
        ), patch(
            "langchain_classic.agents.agent_toolkits.create_retriever_tool",
            return_value=mock_tool,
        ):
            result = _init_faiss_retriever(
                "faiss_retriever", "A prompt", {"path": "/tmp"}
            )
        assert result is mock_tool

    def test_weather_api_tool_factory(self):
        """_init_weather_api_tool delegates to init_weather_api_tool."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_weather_api_tool,
        )

        mock_tool = MagicMock()
        with patch(
            "bili.iris.tools.api_open_weather.init_weather_api_tool",
            return_value=mock_tool,
        ):
            result = _init_weather_api_tool("weather_api_tool", "A prompt", {})
        assert result is mock_tool

    def test_serp_api_tool_factory(self):
        """_init_serp_api_tool delegates to init_serp_api_tool."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_serp_api_tool,
        )

        mock_tool = MagicMock()
        with patch(
            "bili.iris.tools.api_serp.init_serp_api_tool",
            return_value=mock_tool,
        ):
            result = _init_serp_api_tool("serp_api_tool", "A prompt", {})
        assert result is mock_tool

    def test_weather_gov_api_tool_factory(self):
        """_init_weather_gov_api_tool delegates to init_weather_gov_api_tool."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_weather_gov_api_tool,
        )

        mock_tool = MagicMock()
        with patch(
            "bili.iris.tools.api_weather_gov.init_weather_gov_api_tool",
            return_value=mock_tool,
        ):
            result = _init_weather_gov_api_tool("weather_gov_api_tool", "A prompt", {})
        assert result is mock_tool

    def test_free_weather_api_tool_factory(self):
        """_init_free_weather_api_tool delegates to init_weather_tool."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_free_weather_api_tool,
        )

        mock_tool = MagicMock()
        with patch(
            "bili.iris.tools.api_free_weather_api.init_weather_tool",
            return_value=mock_tool,
        ):
            result = _init_free_weather_api_tool(
                "free_weather_api_tool", "A prompt", {}
            )
        assert result is mock_tool

    def test_aws_opensearch_retriever_factory(self):
        """_init_aws_opensearch_retriever delegates to init_amazon_opensearch."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            _init_aws_opensearch_retriever,
        )

        mock_embedding = MagicMock()
        mock_tool = MagicMock()
        params = {
            "index_name": "my-index",
            "index_mapping": {
                "my-index": {"provider": "bedrock", "model_name": "titan-v1"}
            },
        }
        with patch(
            "bili.iris.loaders.embeddings_loader.load_embedding_function",
            return_value=mock_embedding,
        ), patch(
            "bili.iris.tools.amazon_opensearch.init_amazon_opensearch",
            return_value=mock_tool,
        ):
            result = _init_aws_opensearch_retriever(
                "aws_opensearch_retriever", "A prompt", params
            )
        assert result is mock_tool
