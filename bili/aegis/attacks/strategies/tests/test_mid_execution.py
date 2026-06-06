"""Tests for the mid-execution injection strategy.

Covers the branches of run_with_mid_execution_injection that are hard to
reach through the higher-level runners: the get_state() fallback on the
early-return path, the NodeInterrupt path (missing node attribute and node
mismatch), and the _get_interrupt_state helper.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langgraph.errors import NodeInterrupt

from bili.aegis.attacks.strategies.mid_execution import (
    _get_interrupt_state,
    run_with_mid_execution_injection,
)


def _agent(agent_id, role="worker"):
    """Build a minimal agent spec stand-in with id and role attributes."""
    return SimpleNamespace(agent_id=agent_id, role=role)


def _compiled_mas(graph, agent_ids=("target",)):
    """Build a compiled-MAS stand-in whose compile_graph returns *graph*."""
    config = SimpleNamespace(agents=[_agent(a) for a in agent_ids])
    mas = MagicMock()
    mas.config = config
    mas.compile_graph.return_value = graph
    return mas


class TestEarlyReturnPath:
    """LangGraph >= 1.x returns early from invoke() rather than raising."""

    def test_falls_back_to_invoke_result_when_get_state_fails(self):
        """A get_state() failure falls back to the invoke() return dict."""
        graph = MagicMock()
        graph.invoke.return_value = {"messages": []}
        # get_state raising a generic error triggers the fallback branch.
        graph.get_state.side_effect = ValueError("no checkpointer")
        graph.stream.return_value = []

        result = run_with_mid_execution_injection(
            compiled_mas=_compiled_mas(graph),
            input_data={"messages": []},
            target_agent_id="target",
            payload="INJECT",
            tracker=MagicMock(),
        )

        # The injected payload is present in the accumulated state messages.
        assert any(getattr(m, "content", None) == "INJECT" for m in result["messages"])


class TestNodeInterruptPath:
    """Older LangGraph raises NodeInterrupt from invoke()."""

    def test_missing_node_attribute_skips_validation(self):
        """A NodeInterrupt without a .node attribute proceeds without raising."""
        graph = MagicMock()
        graph.invoke.side_effect = NodeInterrupt("paused")
        graph.get_state.return_value = SimpleNamespace(values={"messages": []})
        graph.stream.return_value = []

        result = run_with_mid_execution_injection(
            compiled_mas=_compiled_mas(graph),
            input_data={"messages": []},
            target_agent_id="target",
            payload="INJECT",
            tracker=MagicMock(),
        )

        assert any(getattr(m, "content", None) == "INJECT" for m in result["messages"])

    def test_node_mismatch_raises_runtime_error(self):
        """A NodeInterrupt at a different node raises a descriptive RuntimeError."""
        exc = NodeInterrupt("paused")
        exc.node = "some_other_node"
        graph = MagicMock()
        graph.invoke.side_effect = exc

        with pytest.raises(RuntimeError, match="Expected NodeInterrupt at 'target'"):
            run_with_mid_execution_injection(
                compiled_mas=_compiled_mas(graph),
                input_data={"messages": []},
                target_agent_id="target",
                payload="INJECT",
                tracker=MagicMock(),
            )


class TestGetInterruptState:
    """_get_interrupt_state reads the snapshot or degrades to an empty dict."""

    def test_returns_snapshot_values(self):
        """A successful get_state returns the snapshot values as a dict."""
        graph = MagicMock()
        graph.get_state.return_value = SimpleNamespace(values={"k": "v"})
        assert _get_interrupt_state(graph, {}) == {"k": "v"}

    def test_returns_empty_on_error(self):
        """A get_state failure degrades to an empty dict, not an exception."""
        graph = MagicMock()
        graph.get_state.side_effect = RuntimeError("unavailable")
        assert _get_interrupt_state(graph, {}) == {}
