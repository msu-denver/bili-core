"""Tests for bili.iris.loaders.langchain_loader public API."""

import copy
from functools import partial
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from bili.iris.graph_builder.classes.node import Node
from bili.iris.loaders.langchain_loader import (
    DEFAULT_GRAPH_DEFINITION,
    GRAPH_NODE_REGISTRY,
    build_agent_graph,
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

    def test_wrap_compiled_state_graph_uses_invoke(self):
        """Verify CompiledStateGraph nodes use .invoke() method."""
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.invoke.return_value = {"messages": []}
        wrapped = wrap_node(mock_graph, "subgraph")
        state = {"messages": []}
        result = wrapped(state)
        mock_graph.invoke.assert_called_once_with(state)
        assert result == {"messages": []}


# ---------------------------------------------------------------------------
# build_agent_graph
# ---------------------------------------------------------------------------


class TestBuildAgentGraph:
    """Test build_agent_graph with mocked checkpointer and node_kwargs."""

    @staticmethod
    def _default_node_kwargs():
        """Build minimal node_kwargs that satisfy all node builders."""
        return {
            "persona": "You are a test assistant",
            "llm_model": MagicMock(),
            "tools": [],
        }

    def test_returns_compiled_state_graph(self):
        """Verify build_agent_graph returns a CompiledStateGraph."""
        agent = build_agent_graph(node_kwargs=self._default_node_kwargs())
        assert isinstance(agent, CompiledStateGraph)

    def test_uses_memory_saver_by_default(self):
        """Verify default checkpointer is MemorySaver."""
        agent = build_agent_graph(node_kwargs=self._default_node_kwargs())
        assert agent.checkpointer is not None

    def test_custom_checkpointer_is_used(self):
        """Verify custom checkpointer is passed through."""
        custom_saver = MemorySaver()
        agent = build_agent_graph(
            checkpoint_saver=custom_saver,
            node_kwargs=self._default_node_kwargs(),
        )
        assert agent.checkpointer is custom_saver

    def test_invalid_node_in_definition_raises_value_error(self):
        """Verify ValueError for unregistered node names."""
        bad_definition = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)
        bad_definition[0].name = "totally_bogus_node"
        with pytest.raises(ValueError, match="not defined"):
            build_agent_graph(
                graph_definition=bad_definition,
                node_kwargs=self._default_node_kwargs(),
            )

    def test_custom_node_registry_extends_default(self):
        """Verify custom_node_registry merges with default."""

        def build_custom(**_kwargs):
            """Build a passthrough node function."""

            def _execute(state):
                return state

            return _execute

        custom_node = partial(Node, "custom_passthrough", build_custom)

        custom_def = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)
        custom_node_instance = custom_node()
        custom_node_instance.is_entry = False
        custom_node_instance.routes_to_end = False

        last_node = custom_def[-1]
        second_last = custom_def[-2]
        second_last.edges = ["custom_passthrough"]
        custom_node_instance.edges.append(last_node.name)
        custom_def.insert(-1, custom_node_instance)

        agent = build_agent_graph(
            graph_definition=custom_def,
            custom_node_registry={
                "custom_passthrough": custom_node,
            },
            node_kwargs=self._default_node_kwargs(),
        )
        assert isinstance(agent, CompiledStateGraph)

    def test_node_kwargs_are_forwarded(self):
        """Verify node_kwargs are passed to node builders."""
        agent = build_agent_graph(node_kwargs=self._default_node_kwargs())
        assert isinstance(agent, CompiledStateGraph)

    def test_mixed_entry_and_conditional_raises(self):
        """Verify mixing is_entry and conditional_entry raises."""
        bad_def = copy.deepcopy(DEFAULT_GRAPH_DEFINITION)
        mock_cond = MagicMock()
        mock_cond.routing_function = lambda x: "some_node"
        mock_cond.path_map = {"some_node": "some_node"}
        bad_def[1].conditional_entry = mock_cond

        with pytest.raises(ValueError, match="Cannot mix"):
            build_agent_graph(
                graph_definition=bad_def,
                node_kwargs=self._default_node_kwargs(),
            )
