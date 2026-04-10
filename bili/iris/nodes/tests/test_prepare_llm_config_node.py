"""Tests for prepare_llm_config_node module.

Tests the LLM config preparation functionality:
- Builder returns a callable
- Config resolution delegates to prepare_runtime_config
- Default behavior when no thinking_config or model_type
- Returns dict with llm_config key
"""

from unittest.mock import patch

from bili.iris.nodes.prepare_llm_config_node import (
    build_prepare_llm_config_node,
    prepare_llm_config_node,
)


class TestBuildPrepareLlmConfigNode:
    """Tests for build_prepare_llm_config_node function."""

    def test_returns_callable(self):
        """Builder should return a callable function."""
        node_func = build_prepare_llm_config_node()
        assert callable(node_func)

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_delegates_to_prepare_runtime_config(self, mock_prep):
        """Node should call prepare_runtime_config with kwargs."""
        mock_prep.return_value = {"some": "config"}

        node_func = build_prepare_llm_config_node(
            thinking_config={"budget": 1000},
            model_type="remote_google_vertex",
        )
        state = {"messages": []}

        result = node_func(state)

        mock_prep.assert_called_once_with(
            model_type="remote_google_vertex",
            thinking_config={"budget": 1000},
        )
        assert result == {"llm_config": {"some": "config"}}

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_default_none_for_missing_kwargs(self, mock_prep):
        """Missing kwargs should default to None."""
        mock_prep.return_value = {}

        node_func = build_prepare_llm_config_node()
        node_func({"messages": []})

        mock_prep.assert_called_once_with(
            model_type=None,
            thinking_config=None,
        )

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_returns_llm_config_key(self, mock_prep):
        """Result should contain only the llm_config key."""
        mock_prep.return_value = {"key": "value"}

        node_func = build_prepare_llm_config_node(model_type="remote_openai")
        result = node_func({"messages": []})

        assert list(result.keys()) == ["llm_config"]
        assert result["llm_config"] == {"key": "value"}

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_empty_config_returned(self, mock_prep):
        """Empty config from prepare_runtime_config is valid."""
        mock_prep.return_value = {}

        node_func = build_prepare_llm_config_node(
            model_type="remote_openai",
            thinking_config={"budget": 0},
        )
        result = node_func({"messages": []})

        assert result == {"llm_config": {}}

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_state_is_not_modified(self, mock_prep):
        """The input state should not be mutated."""
        mock_prep.return_value = {"cfg": True}

        node_func = build_prepare_llm_config_node()
        state = {"messages": [], "other": "data"}

        node_func(state)

        assert "llm_config" not in state

    def test_accepts_extra_kwargs(self):
        """Builder should accept extra kwargs without error."""
        node_func = build_prepare_llm_config_node(extra_param="ignored")
        assert callable(node_func)


class TestPrepareLlmConfigNodePartial:
    """Tests for the prepare_llm_config_node partial."""

    def test_partial_creates_node_with_correct_name(self):
        """Partial should produce a Node named 'prepare_llm_config'."""
        node = prepare_llm_config_node()
        assert node.name == "prepare_llm_config"

    def test_partial_creates_callable_node(self):
        """The Node created by the partial should be callable."""
        node = prepare_llm_config_node()
        assert callable(node)

    @patch("bili.iris.nodes.prepare_llm_config_node" ".prepare_runtime_config")
    def test_partial_call_invokes_builder(self, mock_prep):
        """Calling the Node should invoke the builder function."""
        mock_prep.return_value = {}
        node = prepare_llm_config_node()
        result = node()
        assert callable(result)
