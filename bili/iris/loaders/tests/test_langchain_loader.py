"""Tests for bili.iris.loaders.langchain_loader public API."""

from unittest.mock import MagicMock

import pytest

from bili.iris.graph_builder.classes.node import Node
from bili.iris.loaders.langchain_loader import (
    DEFAULT_GRAPH_DEFINITION,
    GRAPH_NODE_REGISTRY,
    register_node,
    unregister_node,
    wrap_node,
)

# ---------------------------------------------------------------------------
# GRAPH_NODE_REGISTRY
# ---------------------------------------------------------------------------


class TestGraphNodeRegistry:
    """Verify the global node registry is populated correctly."""

    EXPECTED_NODE_NAMES = {
        "add_persona_and_summary",
        "inject_current_datetime",
        "prepare_llm_config",
        "react_agent",
        "update_timestamp",
        "trim_summarize",
        "normalize_state",
        "per_user_state",
    }

    def test_registry_contains_expected_nodes(self):
        """Verify the registry contains all expected node names."""
        assert self.EXPECTED_NODE_NAMES.issubset(GRAPH_NODE_REGISTRY.keys())

    def test_registry_values_are_callable(self):
        """Verify all registry values are callable factories."""
        for name, factory in GRAPH_NODE_REGISTRY.items():
            assert callable(factory), f"Registry entry '{name}' is not callable"


# ---------------------------------------------------------------------------
# DEFAULT_GRAPH_DEFINITION
# ---------------------------------------------------------------------------


class TestDefaultGraphDefinition:
    """Verify the default graph definition structure."""

    EXPECTED_DEFAULT_NODES = [
        "add_persona_and_summary",
        "inject_current_datetime",
        "react_agent",
        "update_timestamp",
        "trim_summarize",
        "normalize_state",
    ]

    def test_definition_is_a_list_of_nodes(self):
        """Verify the definition is a list of Node instances."""
        assert isinstance(DEFAULT_GRAPH_DEFINITION, list)
        for item in DEFAULT_GRAPH_DEFINITION:
            assert isinstance(item, Node)

    def test_definition_contains_expected_nodes_in_order(self):
        """Verify node names match the expected order."""
        names = [node.name for node in DEFAULT_GRAPH_DEFINITION]
        assert names == self.EXPECTED_DEFAULT_NODES

    def test_first_node_is_entry(self):
        """Verify the first node is marked as entry."""
        assert DEFAULT_GRAPH_DEFINITION[0].is_entry is True

    def test_last_node_routes_to_end(self):
        """Verify the last node routes to end."""
        assert DEFAULT_GRAPH_DEFINITION[-1].routes_to_end is True

    def test_edges_form_a_connected_pipeline(self):
        """Each node (except the last) should have an edge to the next node."""
        for i in range(len(DEFAULT_GRAPH_DEFINITION) - 1):
            current = DEFAULT_GRAPH_DEFINITION[i]
            next_name = DEFAULT_GRAPH_DEFINITION[i + 1].name
            assert (
                next_name in current.edges
            ), f"Node '{current.name}' does not have an edge to '{next_name}'"


# ---------------------------------------------------------------------------
# register_node / unregister_node
# ---------------------------------------------------------------------------


class TestRegisterNode:
    """Test dynamic node registration and unregistration."""

    CUSTOM_NAME = "_test_custom_node"

    def _dummy_factory(self):
        return MagicMock(spec=Node)

    def test_register_adds_to_registry(self):
        """Verify register_node adds the factory to the registry."""
        try:
            register_node(self.CUSTOM_NAME, self._dummy_factory)
            assert self.CUSTOM_NAME in GRAPH_NODE_REGISTRY
        finally:
            GRAPH_NODE_REGISTRY.pop(self.CUSTOM_NAME, None)

    def test_register_duplicate_raises_value_error(self):
        """Verify registering a duplicate name raises ValueError."""
        try:
            register_node(self.CUSTOM_NAME, self._dummy_factory)
            with pytest.raises(ValueError, match="already registered"):
                register_node(self.CUSTOM_NAME, self._dummy_factory)
        finally:
            GRAPH_NODE_REGISTRY.pop(self.CUSTOM_NAME, None)

    def test_unregister_removes_from_registry(self):
        """Verify unregister_node removes the entry from the registry."""
        try:
            register_node(self.CUSTOM_NAME, self._dummy_factory)
            unregister_node(self.CUSTOM_NAME)
            assert self.CUSTOM_NAME not in GRAPH_NODE_REGISTRY
        finally:
            GRAPH_NODE_REGISTRY.pop(self.CUSTOM_NAME, None)

    def test_unregister_missing_raises_key_error(self):
        """Verify unregistering a missing node raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            unregister_node("_nonexistent_node_xyz")


# ---------------------------------------------------------------------------
# wrap_node
# ---------------------------------------------------------------------------


class TestWrapNode:
    """Test the node-wrapping performance logger."""

    def test_wrap_node_returns_callable(self):
        """Verify wrap_node returns a callable wrapper."""
        original = MagicMock(return_value={"messages": []})
        wrapped = wrap_node(original, "test_node")
        assert callable(wrapped)

    def test_wrapped_function_calls_original(self):
        """Verify the wrapper delegates to the original function."""
        original = MagicMock(return_value={"messages": []})
        wrapped = wrap_node(original, "test_node")
        state = {"messages": []}
        result = wrapped(state)
        original.assert_called_once_with(state)
        assert result == {"messages": []}

    def test_wrapped_function_returns_original_result(self):
        """Verify the wrapper returns the original function's result."""
        expected = {"messages": ["hello"]}
        original = MagicMock(return_value=expected)
        wrapped = wrap_node(original, "test_node")
        assert wrapped({}) == expected
