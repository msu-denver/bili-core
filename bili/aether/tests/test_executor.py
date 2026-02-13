"""Tests for the AETHER MAS Execution Controller (Task 8).

Covers:
    - MASExecutor initialization and lifecycle
    - Sequential execution with stub agents
    - Communication log collection
    - Result serialization (to_dict, save_to_file, get_summary, get_formatted_output)
    - Checkpoint persistence workflow
    - Cross-model transfer workflow
    - execute_mas() convenience function
    - Error handling (graceful failure)
    - CLI argument parsing

All tests use stub agents (no ``model_name``), so no LLM API calls are
needed.  Runs under the isolated test runner (``run_tests.py``).
"""

# pylint: disable=missing-function-docstring

import json
import os
import tempfile

import pytest
from langchain_core.messages import HumanMessage  # pylint: disable=import-error

from bili.aether.runtime.execution_result import (
    AgentExecutionResult,
    MASExecutionResult,
)
from bili.aether.runtime.executor import MASExecutor, execute_mas
from bili.aether.schema import (
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    WorkflowType,
)

# =========================================================================
# Helper
# =========================================================================


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    """Shortcut to build an AgentSpec with sensible defaults."""
    defaults = {
        "role": "test_role",
        "objective": f"Test objective for {agent_id}",
    }
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


def _seq_config(mas_id: str = "test_seq", n_agents: int = 3, **kwargs) -> MASConfig:
    """Build a simple sequential MASConfig with *n_agents* stub agents."""
    agents = [_agent(f"agent_{i}") for i in range(n_agents)]
    defaults = {
        "mas_id": mas_id,
        "name": "Test Sequential MAS",
        "workflow_type": WorkflowType.SEQUENTIAL,
        "agents": agents,
        "checkpoint_enabled": False,
    }
    defaults.update(kwargs)
    return MASConfig(**defaults)


# =========================================================================
# Executor Initialization
# =========================================================================


class TestMASExecutorInit:
    """Tests for MASExecutor creation and initialization."""

    def test_executor_creation(self):
        config = _seq_config()
        executor = MASExecutor(config)
        assert executor.config is config

    def test_executor_initialize_compiles_graph(self):
        config = _seq_config()
        executor = MASExecutor(config)
        executor.initialize()
        assert executor._compiled_mas is not None  # pylint: disable=protected-access
        assert executor._compiled_graph is not None  # pylint: disable=protected-access

    def test_run_before_initialize_raises(self):
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            executor.run()


# =========================================================================
# Sequential Execution
# =========================================================================


class TestSequentialExecution:
    """Tests for sequential MAS execution with stub agents."""

    def test_run_sequential_stub_agents(self):
        config = _seq_config(n_agents=3)
        with tempfile.TemporaryDirectory() as tmp:
            executor = MASExecutor(config, log_dir=tmp)
            executor.initialize()
            result = executor.run(save_results=False)

            assert result.success
            assert result.mas_id == "test_seq"
            assert len(result.agent_results) == 3

    def test_agent_outputs_collected(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        for agent_result in result.agent_results:
            assert agent_result.output.get("status") == "stub"
            assert agent_result.agent_id in ("agent_0", "agent_1")

    def test_final_state_contains_messages(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        messages = result.final_state.get("messages", [])
        # Each stub agent emits an AIMessage
        ai_messages = [m for m in messages if m.get("type") == "AIMessage"]
        assert len(ai_messages) >= 2

    def test_timing_fields_populated(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.total_execution_time_ms > 0
        assert result.start_time != ""
        assert result.end_time != ""
        assert result.execution_id.startswith("test_seq_")

    def test_run_with_input_data(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(
            input_data={"messages": [HumanMessage(content="Hello world")]},
            save_results=False,
        )
        assert result.success


# =========================================================================
# Communication Logs
# =========================================================================


class TestCommunicationLogs:
    """Tests for communication statistics collection."""

    def test_no_channels_zero_messages(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.total_messages == 0
        assert result.messages_by_channel == {}

    def test_channels_collect_communication_log(self):
        config = MASConfig(
            mas_id="comm_test",
            name="Comm Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("sender"), _agent("receiver")],
            channels=[
                Channel(
                    channel_id="main",
                    protocol=CommunicationProtocol.DIRECT,
                    source="sender",
                    target="receiver",
                ),
            ],
            checkpoint_enabled=False,
        )
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        # Stub agents record output in communication_log via
        # _build_communication_update; at minimum the log field exists
        assert isinstance(result.final_state.get("communication_log"), list)


# =========================================================================
# Result Serialization
# =========================================================================


class TestResultSerialization:
    """Tests for AgentExecutionResult and MASExecutionResult serialization."""

    def test_agent_result_to_dict(self):
        ar = AgentExecutionResult(
            agent_id="test",
            role="tester",
            output={"status": "ok"},
            execution_time_ms=42.5,
            tools_used=["tool_a"],
            messages_sent=2,
            messages_received=1,
        )
        d = ar.to_dict()
        assert d["agent_id"] == "test"
        assert d["execution_time_ms"] == 42.5
        assert d["tools_used"] == ["tool_a"]
        # Verify JSON-serializable
        json.dumps(d)

    def test_mas_result_to_dict(self):
        mr = MASExecutionResult(
            mas_id="test",
            execution_id="test_abc123",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T00:00:01Z",
            total_execution_time_ms=1000.0,
            agent_results=[
                AgentExecutionResult(agent_id="a", role="r"),
            ],
        )
        d = mr.to_dict()
        assert d["mas_id"] == "test"
        assert len(d["agent_results"]) == 1
        assert d["agent_results"][0]["agent_id"] == "a"
        json.dumps(d)

    def test_save_to_file(self):
        mr = MASExecutionResult(
            mas_id="test",
            execution_id="test_save",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.json")
            mr.save_to_file(path)

            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert data["mas_id"] == "test"
            assert data["success"] is True

    def test_get_summary(self):
        mr = MASExecutionResult(
            mas_id="test",
            execution_id="test_sum",
            total_execution_time_ms=42.0,
            agent_results=[
                AgentExecutionResult(
                    agent_id="a",
                    role="tester",
                    output={"message": "hello"},
                ),
            ],
        )
        summary = mr.get_summary()
        assert "test" in summary
        assert "SUCCESS" in summary
        assert "42.00" in summary

    def test_get_formatted_output_has_borders(self):
        mr = MASExecutionResult(
            mas_id="test",
            execution_id="test_fmt",
            agent_results=[
                AgentExecutionResult(
                    agent_id="a",
                    role="tester",
                    output={"message": "formatted output"},
                ),
            ],
        )
        output = mr.get_formatted_output()
        assert output.startswith("*" * 60)
        assert output.endswith("*" * 60)
        assert "test" in output

    def test_save_results_creates_file(self):
        config = _seq_config(n_agents=1)
        with tempfile.TemporaryDirectory() as tmp:
            executor = MASExecutor(config, log_dir=tmp)
            executor.initialize()
            result = executor.run(save_results=True)

            result_path = os.path.join(tmp, f"{result.execution_id}.json")
            assert os.path.exists(result_path)


# =========================================================================
# Checkpoint Persistence
# =========================================================================


class TestCheckpointPersistence:
    """Tests for run_with_checkpoint_persistence()."""

    def test_returns_tuple_of_two_results(self):
        config = MASConfig(
            mas_id="cp_test",
            name="Checkpoint Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a"), _agent("b")],
            checkpoint_enabled=True,
        )
        executor = MASExecutor(config)
        executor.initialize()

        original, restored = executor.run_with_checkpoint_persistence(
            thread_id="test-thread"
        )

        assert isinstance(original, MASExecutionResult)
        assert isinstance(restored, MASExecutionResult)

    def test_both_runs_succeed(self):
        config = MASConfig(
            mas_id="cp_success",
            name="CP Success",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
            checkpoint_enabled=True,
        )
        executor = MASExecutor(config)
        executor.initialize()

        original, restored = executor.run_with_checkpoint_persistence()

        assert original.success
        assert restored.success
        assert original.checkpoint_saved
        assert original.metadata.get("checkpoint_test") == "original"
        assert restored.metadata.get("checkpoint_test") == "restored"


# =========================================================================
# Cross-Model Transfer
# =========================================================================


class TestCrossModelTransfer:
    """Tests for run_cross_model_test()."""

    def test_returns_tuple_of_two_results(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()

        source, target = executor.run_cross_model_test(
            source_model=None,
            target_model=None,
        )

        assert isinstance(source, MASExecutionResult)
        assert isinstance(target, MASExecutionResult)

    def test_metadata_records_model_info(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        source, target = executor.run_cross_model_test(
            source_model=None,
            target_model=None,
        )

        assert source.metadata.get("cross_model_test") == "source"
        assert target.metadata.get("cross_model_test") == "target"
        assert source.success
        assert target.success


# =========================================================================
# Convenience Function
# =========================================================================


class TestExecuteMasConvenience:
    """Tests for the execute_mas() convenience function."""

    def test_execute_mas_returns_result(self):
        config = _seq_config(n_agents=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_mas(config, log_dir=tmp)
            assert isinstance(result, MASExecutionResult)
            assert result.success

    def test_execute_mas_with_input(self):
        config = _seq_config(n_agents=1)
        result = execute_mas(
            config,
            input_data={"messages": [HumanMessage(content="Test input")]},
        )
        assert result.success
        assert len(result.agent_results) == 1


# =========================================================================
# Error Handling
# =========================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_failed_execution_returns_result_not_exception(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        # Force an error by invalidating the compiled graph
        executor._compiled_graph = _BrokenGraph()  # pylint: disable=protected-access

        result = executor.run(save_results=False)
        assert not result.success
        assert result.error is not None
        assert isinstance(result, MASExecutionResult)

    def test_error_result_has_timing(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        executor._compiled_graph = _BrokenGraph()  # pylint: disable=protected-access

        result = executor.run(save_results=False)
        assert result.total_execution_time_ms > 0
        assert result.start_time != ""
        assert result.end_time != ""


class _BrokenGraph:  # pylint: disable=too-few-public-methods
    """Mock graph that always raises on invoke."""

    def invoke(self, *args, **kwargs):
        raise RuntimeError("Simulated graph failure")


# =========================================================================
# CLI Argument Parsing
# =========================================================================


class TestCLIArgParsing:
    """Tests for the CLI argument parser."""

    def test_basic_args(self):
        from bili.aether.runtime.cli import (  # pylint: disable=import-outside-toplevel
            _build_parser,
        )

        args = _build_parser().parse_args(["config.yaml", "--input", "hello"])
        assert args.config_file == "config.yaml"
        assert args.input_text == "hello"

    def test_checkpoint_flag(self):
        from bili.aether.runtime.cli import (  # pylint: disable=import-outside-toplevel
            _build_parser,
        )

        args = _build_parser().parse_args(
            ["config.yaml", "--test-checkpoint", "--thread-id", "t1"]
        )
        assert args.test_checkpoint is True
        assert args.thread_id == "t1"

    def test_cross_model_flags(self):
        from bili.aether.runtime.cli import (  # pylint: disable=import-outside-toplevel
            _build_parser,
        )

        args = _build_parser().parse_args(
            [
                "config.yaml",
                "--test-cross-model",
                "--source-model",
                "gpt-4",
                "--target-model",
                "claude-3-sonnet",
            ]
        )
        assert args.test_cross_model is True
        assert args.source_model == "gpt-4"
        assert args.target_model == "claude-3-sonnet"

    def test_no_save_flag(self):
        from bili.aether.runtime.cli import (  # pylint: disable=import-outside-toplevel
            _build_parser,
        )

        args = _build_parser().parse_args(["config.yaml", "--no-save"])
        assert args.no_save is True

    def test_log_dir(self):
        from bili.aether.runtime.cli import (  # pylint: disable=import-outside-toplevel
            _build_parser,
        )

        args = _build_parser().parse_args(["config.yaml", "--log-dir", "/tmp/logs"])
        assert args.log_dir == "/tmp/logs"


# =========================================================================
# Multi-Tenant and Multi-Conversation Tests (Tasks #23-25)
# =========================================================================


class TestMASExecutorUserIdentity:
    """Tests for user_id and conversation_id parameters."""

    def test_executor_with_user_id(self):
        """Test MASExecutor initialization with user_id parameter."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        assert (
            executor._user_id == "user@example.com"
        )  # pylint: disable=protected-access
        assert executor._conversation_id is None  # pylint: disable=protected-access

    def test_executor_with_user_id_and_conversation_id(self):
        """Test MASExecutor initialization with both user_id and conversation_id."""
        config = _seq_config()
        executor = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_123"
        )

        assert (
            executor._user_id == "user@example.com"
        )  # pylint: disable=protected-access
        assert (
            executor._conversation_id == "conv_123"
        )  # pylint: disable=protected-access

    def test_thread_id_construction_with_user_id_and_conversation_id(self):
        """Test that thread_id is constructed as {user_id}_{conversation_id}."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_123"
        )
        executor.initialize()

        # Extract thread_id from run
        result = executor.run(
            {"messages": [HumanMessage(content="Test")]}, save_results=False
        )

        # Thread ID should be user@example.com_conv_123
        thread_id = result.metadata.get("thread_id")
        assert thread_id == "user@example.com_conv_123"

    def test_thread_id_construction_with_user_id_only(self):
        """Test thread_id generation when only user_id is set."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]}, save_results=False
        )

        thread_id = result.metadata.get("thread_id")
        # Should start with user_id and have auto-generated suffix
        assert thread_id.startswith("user@example.com_")

    def test_thread_id_with_explicit_thread_id_parameter(self):
        """Test that explicit thread_id parameter is used as conversation_id."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]},
            thread_id="my_conversation",
            save_results=False,
        )

        thread_id = result.metadata.get("thread_id")
        # Should prepend user_id to provided thread_id
        assert thread_id == "user@example.com_my_conversation"

    def test_backward_compat_no_user_id(self):
        """Test backward compatibility when user_id is not set."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(config)  # No user_id
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]},
            thread_id="my_thread",
            save_results=False,
        )

        thread_id = result.metadata.get("thread_id")
        # Should use thread_id as-is (no user_id prefix)
        assert thread_id == "my_thread"


class TestThreadOwnershipValidation:
    """Tests for thread ownership validation in multi-tenant mode."""

    def test_validate_thread_ownership_valid_exact_match(self):
        """Test validation passes when thread_id exactly matches user_id."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        # Should not raise
        executor._validate_thread_ownership(
            "user@example.com"
        )  # pylint: disable=protected-access

    def test_validate_thread_ownership_valid_prefix(self):
        """Test validation passes when thread_id starts with user_id_."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        # Should not raise
        executor._validate_thread_ownership(
            "user@example.com_conv_123"
        )  # pylint: disable=protected-access

    def test_validate_thread_ownership_invalid_user(self):
        """Test validation fails when thread_id belongs to different user."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(PermissionError, match="Access denied"):
            executor._validate_thread_ownership(
                "other@example.com_conv_123"
            )  # pylint: disable=protected-access

    def test_validate_thread_ownership_invalid_no_prefix(self):
        """Test validation fails when thread_id doesn't have user_id prefix."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(PermissionError, match="does not belong to"):
            executor._validate_thread_ownership(
                "conv_123"
            )  # pylint: disable=protected-access

    def test_validate_thread_ownership_no_op_without_user_id(self):
        """Test validation is no-op when user_id not set."""
        config = _seq_config()
        executor = MASExecutor(config)  # No user_id

        # Should not raise (validation disabled)
        executor._validate_thread_ownership(
            "any_thread_id"
        )  # pylint: disable=protected-access

    def test_validation_error_message_includes_pattern(self):
        """Test that validation error includes expected pattern."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(
            PermissionError, match="'user@example.com' or 'user@example.com_\\*'"
        ):
            executor._validate_thread_ownership(
                "wrong_thread"
            )  # pylint: disable=protected-access


class TestMultiConversationScenarios:
    """Tests for multi-conversation use cases."""

    def test_multiple_conversations_same_user(self):
        """Test that same user can have multiple concurrent conversations."""
        config = _seq_config()
        config.checkpoint_enabled = True

        # First conversation
        executor1 = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_1"
        )
        executor1.initialize()
        result1 = executor1.run(
            {"messages": [HumanMessage(content="Conversation 1")]}, save_results=False
        )

        # Second conversation
        executor2 = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_2"
        )
        executor2.initialize()
        result2 = executor2.run(
            {"messages": [HumanMessage(content="Conversation 2")]}, save_results=False
        )

        # Thread IDs should be different
        thread1 = result1.metadata.get("thread_id")
        thread2 = result2.metadata.get("thread_id")
        assert thread1 == "user@example.com_conv_1"
        assert thread2 == "user@example.com_conv_2"
        assert thread1 != thread2

    def test_conversation_reuse_same_thread_id(self):
        """Test that providing same thread_id reuses conversation."""
        config = _seq_config()
        config.checkpoint_enabled = True
        user_id = "user@example.com"
        conv_id = "reusable_conv"

        # First execution
        executor1 = MASExecutor(config, user_id=user_id, conversation_id=conv_id)
        executor1.initialize()
        executor1.run(
            {"messages": [HumanMessage(content="First message")]}, save_results=False
        )

        # Second execution with same conversation_id (should reuse thread)
        executor2 = MASExecutor(config, user_id=user_id, conversation_id=conv_id)
        executor2.initialize()
        result2 = executor2.run(
            {"messages": [HumanMessage(content="Second message")]},
            save_results=False,
        )

        # Should use the same thread_id
        thread_id = result2.metadata.get("thread_id")
        assert thread_id == f"{user_id}_{conv_id}"

    def test_thread_id_with_prefix_reuse(self):
        """Test reusing thread_id that already has user_id prefix."""
        config = _seq_config()
        config.checkpoint_enabled = True

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        # Provide thread_id that already has user_id prefix
        result = executor.run(
            {"messages": [HumanMessage(content="Test")]},
            thread_id="user@example.com_existing_conv",
            save_results=False,
        )

        thread_id = result.metadata.get("thread_id")
        # Should use as-is (not double-prefix)
        assert thread_id == "user@example.com_existing_conv"
        assert thread_id.count("user@example.com") == 1


class TestCheckpointerIntegration:
    """Tests for checkpointer integration with user_id."""

    def test_checkpointer_created_with_user_id(self):
        """Test that checkpointer is created with user_id when provided."""
        config = _seq_config()
        config.checkpoint_enabled = True

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        # Graph should have been compiled with a checkpointer
        assert executor._compiled_graph is not None  # pylint: disable=protected-access

    def test_no_checkpointer_when_checkpoint_disabled(self):
        """Test that user_id doesn't create checkpointer if checkpointing disabled."""
        config = _seq_config()
        config.checkpoint_enabled = False

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        # Should still initialize successfully (checkpointer not created)
        assert executor._compiled_graph is not None  # pylint: disable=protected-access
