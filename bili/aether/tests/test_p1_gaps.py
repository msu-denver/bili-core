"""Tests for P1 functional testing gaps in the AETHER framework.

Covers:
    - P1.1: End-to-end execution of non-sequential workflows
            (supervisor, consensus, hierarchical, parallel, custom)
    - P1.2: Checkpoint resume state continuity verification
    - P1.3: Compiler CLI tests
    - P1.4: needs_human_review state field in execution
    - P1.5: Inter-agent message injection into LLM context

All tests use stub agents (no ``model_name``) unless noted, so no LLM
API calls are needed.  Runs under the isolated test runner
(``run_tests.py``).
"""

# pylint: disable=missing-function-docstring

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage  # pylint: disable=import-error

from bili.aether.compiler.agent_generator import generate_agent_node
from bili.aether.runtime.execution_result import MASExecutionResult
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowEdge, WorkflowType

# =========================================================================
# Helpers
# =========================================================================

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "examples",
)


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    """Shortcut to build an AgentSpec with sensible defaults."""
    defaults = {
        "role": "test_role",
        "objective": f"Test objective for {agent_id}",
    }
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


# =========================================================================
# P1.1 — End-to-end execution of non-sequential workflows
# =========================================================================


class TestSupervisorExecution:
    """E2E execution of supervisor workflow with stub agents."""

    def test_supervisor_executes_without_error(self):
        config = MASConfig(
            mas_id="sup_e2e",
            name="Supervisor E2E",
            workflow_type=WorkflowType.SUPERVISOR,
            entry_point="boss",
            agents=[
                _agent("boss", is_supervisor=True),
                _agent("worker1"),
                _agent("worker2"),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert isinstance(result, MASExecutionResult)
        # Boss runs; workers may not be reached since stubs don't set next_agent
        assert len(result.agent_results) >= 1

    def test_supervisor_boss_appears_in_results(self):
        config = MASConfig(
            mas_id="sup_boss",
            name="Supervisor Boss",
            workflow_type=WorkflowType.SUPERVISOR,
            entry_point="boss",
            agents=[
                _agent("boss", is_supervisor=True),
                _agent("helper"),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        boss_results = [r for r in result.agent_results if r.agent_id == "boss"]
        assert len(boss_results) == 1
        assert boss_results[0].output.get("status") == "stub"


class TestConsensusExecution:
    """E2E execution of consensus workflow with stub agents."""

    def test_consensus_single_agent_executes(self):
        config = MASConfig(
            mas_id="cons_single",
            name="Consensus Single",
            workflow_type=WorkflowType.CONSENSUS,
            consensus_threshold=0.5,
            max_consensus_rounds=2,
            agents=[_agent("solo_voter")],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert len(result.agent_results) == 1

    def test_consensus_multi_agent_executes(self):
        config = MASConfig(
            mas_id="cons_e2e",
            name="Consensus E2E",
            workflow_type=WorkflowType.CONSENSUS,
            consensus_threshold=0.5,
            max_consensus_rounds=2,
            agents=[_agent("voter_a"), _agent("voter_b")],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert isinstance(result, MASExecutionResult)
        assert result.success
        assert len(result.agent_results) >= 2
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"


class TestHierarchicalExecution:
    """E2E execution of hierarchical workflow with stub agents."""

    def test_hierarchical_single_per_tier_executes(self):
        config = MASConfig(
            mas_id="hier_linear",
            name="Hierarchical Linear",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=[
                _agent("leaf", tier=2),
                _agent("root", tier=1),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert len(result.agent_results) == 2
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"

    def test_hierarchical_multi_leaf_executes(self):
        config = MASConfig(
            mas_id="hier_e2e",
            name="Hierarchical E2E",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=[
                _agent("leaf1", tier=2),
                _agent("leaf2", tier=2),
                _agent("root", tier=1),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert isinstance(result, MASExecutionResult)
        assert result.success
        assert len(result.agent_results) == 3
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"

    def test_hierarchical_multi_tier_executes(self):
        config = MASConfig(
            mas_id="hier_all",
            name="Hierarchical All",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=[
                _agent("t3a", tier=3),
                _agent("t3b", tier=3),
                _agent("t2", tier=2),
                _agent("t1", tier=1),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert isinstance(result, MASExecutionResult)
        assert result.success
        assert len(result.agent_results) == 4
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"


class TestParallelExecution:
    """E2E execution of parallel workflow with stub agents."""

    def test_parallel_single_agent_executes(self):
        config = MASConfig(
            mas_id="par_single",
            name="Parallel Single",
            workflow_type=WorkflowType.PARALLEL,
            agents=[_agent("solo")],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert len(result.agent_results) == 1
        assert result.agent_results[0].output.get("status") == "stub"

    def test_parallel_multi_agent_executes(self):
        config = MASConfig(
            mas_id="par_e2e",
            name="Parallel E2E",
            workflow_type=WorkflowType.PARALLEL,
            agents=[_agent("a"), _agent("b"), _agent("c")],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert isinstance(result, MASExecutionResult)
        assert result.success
        assert len(result.agent_results) == 3
        for ar in result.agent_results:
            assert ar.output.get("status") == "stub"


class TestCustomExecution:
    """E2E execution of custom workflow with explicit edges."""

    def test_custom_linear_edges(self):
        config = MASConfig(
            mas_id="cust_e2e",
            name="Custom E2E",
            workflow_type=WorkflowType.CUSTOM,
            entry_point="a",
            agents=[_agent("a"), _agent("b"), _agent("c")],
            workflow_edges=[
                WorkflowEdge(from_agent="a", to_agent="b"),
                WorkflowEdge(from_agent="b", to_agent="c"),
                WorkflowEdge(from_agent="c", to_agent="END"),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert len(result.agent_results) == 3
        executed_ids = [ar.agent_id for ar in result.agent_results]
        assert "a" in executed_ids
        assert "b" in executed_ids
        assert "c" in executed_ids


# =========================================================================
# P1.2 — Checkpoint resume state continuity
# =========================================================================


class TestCheckpointStateContinuity:
    """Verify that checkpoint resume actually restores prior state."""

    def test_second_run_accumulates_messages(self):
        config = MASConfig(
            mas_id="cp_continuity",
            name="Checkpoint Continuity",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a"), _agent("b")],
            checkpoint_enabled=True,
        )
        thread_id = "continuity-test-thread"

        # --- First run ---
        executor = MASExecutor(config)
        executor.initialize()
        result1 = executor.run(
            input_data={"messages": [HumanMessage(content="first run input")]},
            thread_id=thread_id,
            save_results=False,
        )
        assert result1.success
        first_msg_count = len(result1.final_state.get("messages", []))
        assert first_msg_count > 0

        # --- Second run (same thread_id, same executor — shares MemorySaver) ---
        result2 = executor.run(
            input_data={"messages": [HumanMessage(content="second run input")]},
            thread_id=thread_id,
            save_results=False,
        )
        assert result2.success
        second_msg_count = len(result2.final_state.get("messages", []))

        # Checkpoint should carry forward: second run has more messages
        assert second_msg_count > first_msg_count


# =========================================================================
# P1.3 — Compiler CLI tests
# =========================================================================


class TestCompilerCLI:
    """Tests for the AETHER compiler CLI."""

    def test_compile_single_yaml(self, capsys):
        from bili.aether.compiler.cli import (  # pylint: disable=import-outside-toplevel
            main,
        )

        simple_chain = os.path.join(_EXAMPLES_DIR, "simple_chain.yaml")
        if not os.path.exists(simple_chain):
            pytest.skip("simple_chain.yaml not found")

        with patch.object(sys, "argv", ["cli.py", simple_chain]):
            main()

        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_compile_all_examples(self, capsys):
        from bili.aether.compiler.cli import (  # pylint: disable=import-outside-toplevel
            main,
        )

        with patch.object(sys, "argv", ["cli.py"]):
            # main() calls sys.exit(0) on success — catch SystemExit
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "FAIL" not in captured.out

    def test_compile_nonexistent_file(self):
        from bili.aether.compiler.cli import (  # pylint: disable=import-outside-toplevel
            main,
        )

        with patch.object(sys, "argv", ["cli.py", "/nonexistent/path.yaml"]):
            with pytest.raises(FileNotFoundError):
                main()


# =========================================================================
# P1.4 — needs_human_review state field in execution
# =========================================================================


class TestHumanInLoopStateField:
    """Verify that needs_human_review state field flows through execution."""

    def test_custom_human_in_loop_has_state_field(self):
        config = MASConfig(
            mas_id="hil_e2e",
            name="Human In Loop E2E",
            workflow_type=WorkflowType.CUSTOM,
            human_in_loop=True,
            human_escalation_condition="True",
            entry_point="a",
            agents=[_agent("a"), _agent("b")],
            workflow_edges=[
                WorkflowEdge(from_agent="a", to_agent="b"),
                WorkflowEdge(from_agent="b", to_agent="END"),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert "needs_human_review" in result.final_state


# =========================================================================
# P1.5 — Inter-agent message injection into LLM context
# =========================================================================


_MOCK_CREATE = "bili.aether.compiler.llm_resolver.create_llm"
_MOCK_TOOLS = "bili.aether.compiler.llm_resolver.resolve_tools"


class TestCommunicationContextInjection:
    """Verify that pending messages are injected into LLM context."""

    def test_pending_messages_appear_in_llm_context(self):
        agent = _agent("receiver", model_name="gpt-4o")

        with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="acknowledged")
            mock_create.return_value = mock_llm

            node_fn = generate_agent_node(agent)

            # State with pending inter-agent messages
            state = {
                "messages": [],
                "agent_outputs": {},
                "pending_messages": {
                    "receiver": [
                        {
                            "sender": "sender_agent",
                            "channel": "direct_ch",
                            "content": "Please review this document",
                        }
                    ]
                },
                "communication_log": [],
                "channel_messages": {},
            }

            node_fn(state)

            # Verify LLM was called with messages containing the inter-agent context
            call_args = mock_llm.invoke.call_args[0][0]
            system_msg = call_args[0]
            assert "Messages from other agents" in system_msg.content
            assert "Please review this document" in system_msg.content
