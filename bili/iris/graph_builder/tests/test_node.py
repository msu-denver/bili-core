"""Tests for bili.iris.graph_builder.classes.node.Node."""

from bili.iris.graph_builder.classes.node import Node


def _noop_builder(**kwargs):  # pylint: disable=unused-argument
    """A trivial builder function for testing."""

    def _execute(state):
        return state

    return _execute


class TestNodeInstantiation:
    """Verify Node dataclass creation and default values."""

    def test_basic_creation(self):
        """Verify name and function are assigned correctly."""
        node = Node(name="test", function=_noop_builder)
        assert node.name == "test"
        assert node.function is _noop_builder

    def test_default_edges_empty(self):
        """Verify edges default to an empty list."""
        node = Node(name="test", function=_noop_builder)
        assert not node.edges

    def test_default_conditional_edges_empty(self):
        """Verify conditional_edges default to an empty list."""
        node = Node(name="test", function=_noop_builder)
        assert not node.conditional_edges

    def test_default_is_entry_false(self):
        """Verify is_entry defaults to False."""
        node = Node(name="test", function=_noop_builder)
        assert node.is_entry is False

    def test_default_routes_to_end_false(self):
        """Verify routes_to_end defaults to False."""
        node = Node(name="test", function=_noop_builder)
        assert node.routes_to_end is False

    def test_default_conditional_entry_none(self):
        """Verify conditional_entry defaults to None."""
        node = Node(name="test", function=_noop_builder)
        assert node.conditional_entry is None

    def test_default_cache_policy_none(self):
        """Verify cache_policy defaults to None."""
        node = Node(name="test", function=_noop_builder)
        assert node.cache_policy is None


class TestNodeEquality:
    """Verify the custom __eq__ behavior."""

    def test_equals_matching_string(self):
        """Verify node equals a string with the same name."""
        node = Node(name="my_node", function=_noop_builder)
        assert node == "my_node"

    def test_not_equals_different_string(self):
        """Verify node does not equal a string with a different name."""
        node = Node(name="my_node", function=_noop_builder)
        assert node != "other_node"

    def test_equals_another_node_same_name(self):
        """Verify two nodes with the same name are equal."""
        node_a = Node(name="same", function=_noop_builder)
        node_b = Node(name="same", function=_noop_builder)
        assert node_a == node_b

    def test_not_equals_another_node_different_name(self):
        """Verify two nodes with different names are not equal."""
        node_a = Node(name="alpha", function=_noop_builder)
        node_b = Node(name="beta", function=_noop_builder)
        assert node_a != node_b

    def test_equals_unsupported_type_returns_not_implemented(self):
        """Verify __eq__ returns NotImplemented for unsupported types."""
        node = Node(name="test", function=_noop_builder)
        # pylint: disable=unnecessary-dunder-call
        assert node.__eq__(42) is NotImplemented


class TestNodeCallable:
    """Verify that calling a Node delegates to its function."""

    def test_call_invokes_function(self):
        """Verify calling a Node delegates to its builder function."""
        result = Node(name="test", function=_noop_builder)(some_kwarg="value")
        # _noop_builder returns a callable; calling Node returns the inner function
        assert callable(result)

    def test_edges_are_independent_between_instances(self):
        """Default list factory should not be shared across instances."""
        node_a = Node(name="a", function=_noop_builder)
        node_b = Node(name="b", function=_noop_builder)
        node_a.edges.append("x")
        assert "x" not in node_b.edges
