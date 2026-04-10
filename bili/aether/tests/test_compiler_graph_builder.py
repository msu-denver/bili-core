"""Tests for the AETHER graph builder internals.

Covers SafeConditionEvaluator, _StateProxy, supervisor routing instructions,
GraphBuilder workflow dispatch, MAS objective injection, and inheritance.
"""

from unittest.mock import patch

import pytest

from bili.aether.compiler import compile_mas
from bili.aether.compiler.graph_builder import (
    GraphBuilder,
    SafeConditionEvaluator,
    _build_supervisor_routing_instructions,
    _StateProxy,
    safe_eval_condition,
)
from bili.aether.schema import MASConfig, WorkflowType
from bili.aether.tests.test_compiler import _agent

# =========================================================================
# SafeConditionEvaluator TESTS
# =========================================================================


class TestSafeConditionEvaluator:
    """Tests for the AST-based safe condition evaluator."""

    def test_simple_equality(self):
        """Evaluates simple equality comparison."""
        result = safe_eval_condition("x == 1", {"x": 1})
        assert result is True

    def test_inequality(self):
        """Evaluates inequality comparison."""
        result = safe_eval_condition("x != 2", {"x": 1})
        assert result is True

    def test_boolean_and(self):
        """Evaluates boolean AND operation."""
        result = safe_eval_condition("x == 1 and y == 2", {"x": 1, "y": 2})
        assert result is True

    def test_boolean_or(self):
        """Evaluates boolean OR operation."""
        result = safe_eval_condition("x == 1 or y == 99", {"x": 1, "y": 2})
        assert result is True

    def test_not_operator(self):
        """Evaluates unary NOT operation."""
        result = safe_eval_condition("not x", {"x": False})
        assert result is True

    def test_arithmetic(self):
        """Evaluates basic arithmetic in condition."""
        result = safe_eval_condition("x + y == 5", {"x": 2, "y": 3})
        assert result is True

    def test_attribute_access(self):
        """Evaluates attribute access on objects."""
        obj = type("Obj", (), {"val": 42})()
        result = safe_eval_condition("obj.val == 42", {"obj": obj})
        assert result is True

    def test_rejects_function_calls(self):
        """Blocks function call expressions."""
        with pytest.raises(ValueError, match="Invalid condition"):
            safe_eval_condition("print('hi')", {})

    def test_rejects_imports(self):
        """Blocks import expressions."""
        evaluator = SafeConditionEvaluator({})
        with pytest.raises(ValueError):
            evaluator.eval("__import__('os')")

    def test_undefined_variable(self):
        """Raises ValueError for undefined variables."""
        with pytest.raises(ValueError, match="Invalid condition"):
            safe_eval_condition("missing == 1", {})

    def test_in_operator(self):
        """Evaluates 'in' membership test."""
        result = safe_eval_condition("x in items", {"x": 2, "items": [1, 2, 3]})
        assert result is True

    def test_not_in_operator(self):
        """Evaluates 'not in' membership test."""
        result = safe_eval_condition("x not in items", {"x": 5, "items": [1, 2, 3]})
        assert result is True

    def test_constant_true(self):
        """Evaluates constant True expression."""
        assert safe_eval_condition("True", {}) is True

    def test_constant_false(self):
        """Evaluates constant False expression."""
        assert safe_eval_condition("False", {}) is False

    def test_comparison_chain(self):
        """Evaluates chained comparisons like 1 < x < 10."""
        result = safe_eval_condition("1 < x < 10", {"x": 5})
        assert result is True


# =========================================================================
# _StateProxy TESTS
# =========================================================================


class TestStateProxy:
    """Tests for the _StateProxy helper class."""

    def test_attribute_access(self):
        """Accesses state fields as attributes."""
        proxy = _StateProxy({"x": 42, "name": "test"})
        assert proxy.x == 42
        assert proxy.name == "test"

    def test_missing_field_raises_attribute_error(self):
        """Raises AttributeError for missing state fields."""
        proxy = _StateProxy({"x": 1})
        with pytest.raises(AttributeError, match="State field 'missing'"):
            _ = proxy.missing

    def test_error_message_lists_available_fields(self):
        """Error message includes available field names."""
        proxy = _StateProxy({"alpha": 1, "beta": 2})
        with pytest.raises(AttributeError, match="alpha"):
            _ = proxy.gamma


# =========================================================================
# _build_supervisor_routing_instructions TESTS
# =========================================================================


class TestBuildSupervisorRoutingInstructions:
    """Tests for supervisor prompt injection."""

    def test_includes_worker_ids(self):
        """Routing instructions include worker agent IDs."""
        workers = [
            _agent("w1", objective="Writer agent for content"),
            _agent("w2", objective="Reviewer agent for quality"),
        ]
        result = _build_supervisor_routing_instructions("supervisor", workers)
        assert "w1" in result
        assert "w2" in result

    def test_includes_routing_format(self):
        """Routing instructions mention the expected format."""
        workers = [_agent("w1")]
        result = _build_supervisor_routing_instructions("sup", workers)
        assert "next_agent" in result
        assert "ROUTE_TO" in result

    def test_includes_end_instruction(self):
        """Routing instructions mention END routing."""
        workers = [_agent("w1")]
        result = _build_supervisor_routing_instructions("sup", workers)
        assert "END" in result


# =========================================================================
# GraphBuilder._get_workflow_builder TESTS
# =========================================================================


def test_unsupported_workflow_type_raises():
    """Raises ValueError for unsupported workflow type."""
    config = MASConfig(
        mas_id="bad_wf",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
    )
    builder = GraphBuilder(config)
    # Monkey-patch the config to have an invalid workflow type
    builder._config.workflow_type = "invalid_type"  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="Unsupported workflow type"):
        builder._get_workflow_builder()  # pylint: disable=protected-access


# =========================================================================
# GraphBuilder with MAS objective TESTS
# =========================================================================


class TestMasObjective:
    """Tests for MAS-level objective injection."""

    def test_objective_adds_entry_node(self):
        """Config with objective gets __mas_objective__ node."""
        config = MASConfig(
            mas_id="obj_test",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
            objective="Answer all questions helpfully.",
        )

        result = compile_mas(config)
        assert "__mas_objective__" in result.graph.nodes

    def test_no_objective_no_entry_node(self):
        """Config without objective has no __mas_objective__ node."""
        config = MASConfig(
            mas_id="no_obj",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )

        result = compile_mas(config)
        assert "__mas_objective__" not in result.graph.nodes


# =========================================================================
# GraphBuilder._apply_inheritance TESTS
# =========================================================================


class TestApplyInheritance:
    """Tests for bili-core inheritance application."""

    def test_no_inheritance_is_noop(self):
        """No agents with inherit_from_bili_core skips import."""
        config = MASConfig(
            mas_id="noinherit",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        graph_builder = GraphBuilder(config)
        # Should not raise even without the integration module
        graph_builder._apply_inheritance()  # pylint: disable=protected-access

    @patch("bili.aether.compiler.graph_builder.GraphBuilder._apply_inheritance")
    def test_inheritance_called_during_build(self, mock_inherit):
        """_apply_inheritance is called during build()."""
        config = MASConfig(
            mas_id="inherit_build",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        compile_mas(config)
        mock_inherit.assert_called_once()
