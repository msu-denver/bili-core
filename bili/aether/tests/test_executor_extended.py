"""Extended tests for the AETHER MAS Execution Controller.

Covers:
    - run_streaming (low-level streaming)
    - CLI argument parsing
    - Multi-tenant / multi-conversation scenarios
    - Thread ownership validation
    - Checkpointer integration
    - stream() / astream() high-level streaming
    - run_streaming_tokens
    - _map_langgraph_event helper
    - _serialize_message helper
    - resume_streaming

Split from test_executor.py to stay within the pylint line limit.
"""

# pylint: disable=missing-function-docstring

import asyncio
from unittest.mock import MagicMock as Mock
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage  # pylint: disable=import-error

from bili.aether.runtime.executor import MASExecutor, _serialize_message
from bili.aether.runtime.streaming import StreamEvent, StreamEventType, StreamFilter
from bili.aether.tests.test_executor import _seq_config


class _BrokenStreamingGraph:  # pylint: disable=too-few-public-methods
    """Mock graph that always raises on stream."""

    def stream(self, *args, **kwargs):
        """Raise RuntimeError on stream."""
        raise RuntimeError("Simulated streaming failure")


# =========================================================================
# run_streaming
# =========================================================================


class TestRunStreaming:
    """Tests for MASExecutor.run_streaming()."""

    def test_run_streaming_requires_initialize(self):
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.run_streaming())

    def test_run_streaming_yields_tuples(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        results = list(executor.run_streaming())
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_run_streaming_tuple_types(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        for node_name, state_update in executor.run_streaming():
            assert isinstance(node_name, str)
            assert isinstance(state_update, dict)

    def test_run_streaming_with_input_data(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        results = list(
            executor.run_streaming(
                input_data={"messages": [HumanMessage(content="Hello")]}
            )
        )
        assert len(results) > 0

    def test_run_streaming_agent_node_names(self):
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        node_names = [name for name, _ in executor.run_streaming()]
        assert any("agent_0" in name or "agent_1" in name for name in node_names)

    def test_run_streaming_with_thread_id(self):
        config = _seq_config(n_agents=1, checkpoint_enabled=True)
        executor = MASExecutor(config)
        executor.initialize()
        results = list(executor.run_streaming(thread_id="test-thread"))
        assert len(results) > 0

    def test_run_streaming_raises_on_graph_error(self):
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        executor._compiled_graph = (  # pylint: disable=protected-access
            _BrokenStreamingGraph()
        )
        with pytest.raises(RuntimeError, match="Simulated streaming failure"):
            list(executor.run_streaming())


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


class TestMASExecutorUserIdentity:  # pylint: disable=protected-access
    """Tests for user_id and conversation_id parameters."""

    def test_executor_with_user_id(self):
        """Test MASExecutor initialization with user_id parameter."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        assert executor._user_id == "user@example.com"
        assert executor._conversation_id is None

    def test_executor_with_user_id_and_conversation_id(self):
        """Test MASExecutor initialization with both user_id and conversation_id."""
        config = _seq_config()
        executor = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_123"
        )

        assert executor._user_id == "user@example.com"
        assert executor._conversation_id == "conv_123"

    def test_thread_id_construction_with_user_id_and_conversation_id(self):
        """Test that thread_id is constructed as {user_id}_{conversation_id}."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_123"
        )
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]}, save_results=False
        )

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
        assert thread_id == "user@example.com_my_conversation"

    def test_backward_compat_no_user_id(self):
        """Test backward compatibility when user_id is not set."""
        config = _seq_config()
        config.checkpoint_enabled = True
        executor = MASExecutor(config)
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]},
            thread_id="my_thread",
            save_results=False,
        )

        thread_id = result.metadata.get("thread_id")
        assert thread_id == "my_thread"


class TestThreadOwnershipValidation:  # pylint: disable=protected-access
    """Tests for thread ownership validation in multi-tenant mode."""

    def test_validate_thread_ownership_valid_exact_match(self):
        """Test validation passes when thread_id exactly matches user_id."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        executor._validate_thread_ownership("user@example.com")

    def test_validate_thread_ownership_valid_prefix(self):
        """Test validation passes when thread_id starts with user_id_."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        executor._validate_thread_ownership("user@example.com_conv_123")

    def test_validate_thread_ownership_invalid_user(self):
        """Test validation fails when thread_id belongs to different user."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(PermissionError, match="Access denied"):
            executor._validate_thread_ownership("other@example.com_conv_123")

    def test_validate_thread_ownership_invalid_no_prefix(self):
        """Test validation fails when thread_id doesn't have user_id prefix."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(PermissionError, match="does not belong to"):
            executor._validate_thread_ownership("conv_123")

    def test_validate_thread_ownership_no_op_without_user_id(self):
        """Test validation is no-op when user_id not set."""
        config = _seq_config()
        executor = MASExecutor(config)

        executor._validate_thread_ownership("any_thread_id")

    def test_validation_error_message_includes_pattern(self):
        """Test that validation error includes expected pattern."""
        config = _seq_config()
        executor = MASExecutor(config, user_id="user@example.com")

        with pytest.raises(
            PermissionError, match="'user@example.com' or 'user@example.com_\\*'"
        ):
            executor._validate_thread_ownership("wrong_thread")


class TestMultiConversationScenarios:
    """Tests for multi-conversation use cases."""

    def test_multiple_conversations_same_user(self):
        """Test that same user can have multiple concurrent conversations."""
        config = _seq_config()
        config.checkpoint_enabled = True

        executor1 = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_1"
        )
        executor1.initialize()
        result1 = executor1.run(
            {"messages": [HumanMessage(content="Conversation 1")]}, save_results=False
        )

        executor2 = MASExecutor(
            config, user_id="user@example.com", conversation_id="conv_2"
        )
        executor2.initialize()
        result2 = executor2.run(
            {"messages": [HumanMessage(content="Conversation 2")]}, save_results=False
        )

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

        executor1 = MASExecutor(config, user_id=user_id, conversation_id=conv_id)
        executor1.initialize()
        executor1.run(
            {"messages": [HumanMessage(content="First message")]}, save_results=False
        )

        executor2 = MASExecutor(config, user_id=user_id, conversation_id=conv_id)
        executor2.initialize()
        result2 = executor2.run(
            {"messages": [HumanMessage(content="Second message")]},
            save_results=False,
        )

        thread_id = result2.metadata.get("thread_id")
        assert thread_id == f"{user_id}_{conv_id}"

    def test_thread_id_with_prefix_reuse(self):
        """Test reusing thread_id that already has user_id prefix."""
        config = _seq_config()
        config.checkpoint_enabled = True

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        result = executor.run(
            {"messages": [HumanMessage(content="Test")]},
            thread_id="user@example.com_existing_conv",
            save_results=False,
        )

        thread_id = result.metadata.get("thread_id")
        assert thread_id == "user@example.com_existing_conv"
        assert thread_id.count("user@example.com") == 1


class TestCheckpointerIntegration:  # pylint: disable=protected-access
    """Tests for checkpointer integration with user_id."""

    def test_checkpointer_created_with_user_id(self):
        """Test that checkpointer is created with user_id when provided."""
        config = _seq_config()
        config.checkpoint_enabled = True

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        assert executor._compiled_graph is not None

    def test_no_checkpointer_when_checkpoint_disabled(self):
        """Test that user_id doesn't create checkpointer if checkpointing disabled."""
        config = _seq_config()
        config.checkpoint_enabled = False

        executor = MASExecutor(config, user_id="user@example.com")
        executor.initialize()

        assert executor._compiled_graph is not None


# =========================================================================
# stream() method
# =========================================================================


class TestStreamMethod:
    """Tests for MASExecutor.stream() synchronous streaming."""

    def test_stream_requires_initialize(self):
        """stream() raises RuntimeError if not initialized."""
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.stream())

    def test_stream_yields_stream_events(self):
        """stream() yields StreamEvent objects."""
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        events = list(executor.stream())
        assert len(events) > 0
        for event in events:
            assert isinstance(event, StreamEvent)

    def test_stream_emits_run_start_and_end(self):
        """stream() emits RUN_START and RUN_END events."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        events = list(executor.stream())
        event_types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in event_types
        assert StreamEventType.RUN_END in event_types

    def test_stream_with_filter(self):
        """stream() respects StreamFilter to select event types."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        filt = StreamFilter(include_types={StreamEventType.NODE_END})
        events = list(executor.stream(stream_filter=filt))
        for event in events:
            assert event.event_type == StreamEventType.NODE_END

    def test_stream_error_yields_error_event(self):
        """stream() yields an ERROR event on graph failure."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        executor._compiled_graph = (  # pylint: disable=protected-access
            _BrokenStreamingGraph()
        )
        events = list(executor.stream())
        error_events = [e for e in events if e.event_type == StreamEventType.ERROR]
        assert len(error_events) == 1
        assert "Simulated" in error_events[0].data["error"]


# =========================================================================
# run_streaming_tokens
# =========================================================================


class TestRunStreamingTokens:
    """Tests for MASExecutor.run_streaming_tokens()."""

    def test_requires_initialize(self):
        """run_streaming_tokens() raises if not initialized."""
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.run_streaming_tokens())

    def test_yields_node_complete_events(self):
        """run_streaming_tokens yields __node_complete__ events."""
        config = _seq_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        events = list(executor.run_streaming_tokens())
        node_complete = [e for e in events if e[0] == "__node_complete__"]
        assert len(node_complete) > 0
        for _, data in node_complete:
            assert "node" in data
            assert "state_update" in data

    def test_raises_on_graph_error(self):
        """run_streaming_tokens re-raises on graph failure."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()
        executor._compiled_graph = (  # pylint: disable=protected-access
            _BrokenStreamingGraph()
        )
        with pytest.raises(RuntimeError, match="Simulated"):
            list(executor.run_streaming_tokens())


# =========================================================================
# astream() -- async streaming
# =========================================================================


class TestAstreamMethod:
    """Tests for MASExecutor.astream() async streaming."""

    def test_astream_requires_initialize(self):
        """astream() raises RuntimeError if not initialized."""
        config = _seq_config()
        executor = MASExecutor(config)

        async def _run():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(_run())

    def test_astream_yields_events(self):
        """astream() yields StreamEvent objects for stub agents."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        async def _run():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        events = asyncio.run(_run())
        assert len(events) > 0
        for event in events:
            assert isinstance(event, StreamEvent)


# =========================================================================
# _map_langgraph_event
# =========================================================================


class TestMapLanggraphEvent:  # pylint: disable=protected-access
    """Tests for the internal _map_langgraph_event helper."""

    def test_maps_chat_model_stream_to_token(self):
        """on_chat_model_stream events map to TOKEN type."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        chunk = Mock()
        chunk.content = "hello"

        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
            "name": "test_node",
        }
        result = executor._map_langgraph_event(event, "exec_1")
        assert result is not None
        assert result.event_type == StreamEventType.TOKEN
        assert result.data["content"] == "hello"

    def test_maps_chain_start_to_node_start(self):
        """on_chain_start events map to NODE_START type."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        event = {
            "event": "on_chain_start",
            "data": {},
            "name": "my_node",
        }
        result = executor._map_langgraph_event(event, "exec_1")
        assert result is not None
        assert result.event_type == StreamEventType.NODE_START

    def test_skips_langgraph_root_node(self):
        """on_chain_start with name='LangGraph' is skipped."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        event = {
            "event": "on_chain_start",
            "data": {},
            "name": "LangGraph",
        }
        result = executor._map_langgraph_event(event, "exec_1")
        assert result is None

    def test_unknown_event_returns_none(self):
        """Unknown event types return None."""
        config = _seq_config(n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        event = {
            "event": "on_something_else",
            "data": {},
            "name": "x",
        }
        result = executor._map_langgraph_event(event, "exec_1")
        assert result is None


# =========================================================================
# _serialize_message helper
# =========================================================================


class TestSerializeMessage:
    """Tests for the module-level _serialize_message helper."""

    def test_serializes_message_with_content(self):
        """Messages with content attr are serialized to dict."""
        msg = HumanMessage(content="hello")
        result = _serialize_message(msg)
        assert result["type"] == "HumanMessage"
        assert result["content"] == "hello"

    def test_serializes_unknown_object(self):
        """Objects without content attr get str representation."""
        result = _serialize_message(42)
        assert result["type"] == "unknown"
        assert result["content"] == "42"

    def test_serializes_message_with_name(self):
        """Messages with name attr include it in result."""
        msg = Mock()
        msg.content = "hi"
        msg.name = "tool_call"
        type(msg).__name__ = "ToolMessage"
        result = _serialize_message(msg)
        assert result["name"] == "tool_call"


# =========================================================================
# resume_streaming
# =========================================================================


class TestResumeStreaming:  # pylint: disable=too-few-public-methods
    """Tests for MASExecutor.resume_streaming()."""

    def test_resume_requires_initialize(self):
        """resume_streaming() raises if not initialized."""
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.resume_streaming("input", "thread-1"))


# =========================================================================
# resume_with_value (the ask_user Command(resume=...) resume path)
# =========================================================================


class TestResumeWithValue:
    """Tests for MASExecutor.resume_with_value()."""

    def test_resume_with_value_requires_initialize(self):
        config = _seq_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.resume_with_value("staging", "thread-1"))

    def test_resume_with_value_propagates_graph_errors(self):
        """A stream() failure during resume is logged and re-raised, not
        swallowed -- mirrors resume_streaming's own error-propagation
        contract so callers of either resume path get the same guarantee.
        """
        config = _seq_config()
        executor = MASExecutor(config)
        executor.initialize()
        executor._compiled_graph = (  # pylint: disable=protected-access
            _BrokenStreamingGraph()
        )
        with pytest.raises(RuntimeError, match="Simulated streaming failure"):
            list(executor.resume_with_value("staging", "thread-1"))


# =========================================================================
# __ask_user_pending__ sentinel detection in run_streaming
# =========================================================================


class TestAskUserPendingSentinel:
    """Tests for the __ask_user_pending__ sentinel run_streaming() yields."""

    def test_get_state_failure_is_logged_not_raised(self, caplog):
        """A get_state() failure during the ask_user interrupt check must not
        break run_streaming() for callers who never use ask_user -- mirrors
        the existing __human_interrupt__ check's own tolerance of a get_state
        failure (executor.py already logs-and-continues there; this is the
        same contract for the additive check).
        """
        config = _seq_config(checkpoint_enabled=True)
        executor = MASExecutor(config)
        executor.initialize()

        class _FailingGetState:
            """Wraps the real compiled graph but breaks get_state()."""

            def __init__(self, inner):
                self._inner = inner

            def stream(self, *args, **kwargs):
                return self._inner.stream(*args, **kwargs)

            def get_state(self, *args, **kwargs):
                raise RuntimeError("Simulated get_state failure")

        executor._compiled_graph = _FailingGetState(  # pylint: disable=protected-access
            executor._compiled_graph
        )  # pylint: disable=protected-access

        with caplog.at_level("WARNING"):
            events = list(
                executor.run_streaming(
                    {"messages": [HumanMessage(content="hi")]}, thread_id="t-fail"
                )
            )

        assert not [e for e in events if e[0] == "__ask_user_pending__"]
        assert any(
            "ask_user interrupt check" in record.message for record in caplog.records
        )

    def test_no_pending_interrupt_yields_no_sentinel(self):
        """A normal run (no interrupt() call anywhere) yields no
        __ask_user_pending__ sentinel -- the additive check is a no-op for
        every MAS that does not use ask_user.
        """
        config = _seq_config(checkpoint_enabled=True, n_agents=1)
        executor = MASExecutor(config)
        executor.initialize()

        events = list(
            executor.run_streaming(
                {"messages": [HumanMessage(content="hi")]}, thread_id="t-normal"
            )
        )

        assert not [e for e in events if e[0] == "__ask_user_pending__"]


# =========================================================================
# Helper-method coverage
# =========================================================================


class TestExecutorHelpers:
    """Direct tests for small executor helper methods and properties."""

    def test_log_dir_property_returns_configured_dir(self, tmp_path):
        executor = MASExecutor(_seq_config(), log_dir=str(tmp_path))
        assert executor.log_dir == str(tmp_path)

    def test_resolve_agent_for_node_exact_prefix_and_miss(self):
        executor = MASExecutor(_seq_config(n_agents=2))
        executor.initialize()
        # Exact match returns the node name itself.
        assert executor._resolve_agent_for_node("agent_0") == "agent_0"
        # A pipeline sub-node prefixed by an agent id resolves to that agent.
        assert executor._resolve_agent_for_node("agent_0__sub") == "agent_0"
        # An unrelated node name resolves to None.
        assert executor._resolve_agent_for_node("nonexistent") is None

    def test_resolve_agent_for_node_returns_none_before_init(self):
        executor = MASExecutor(_seq_config())
        assert executor._resolve_agent_for_node("agent_0") is None

    def test_get_communication_log_path_is_none(self):
        executor = MASExecutor(_seq_config())
        assert executor._get_communication_log_path() is None

    def test_serialize_state_stringifies_unserializable_values(self):
        # A dict with non-string keys cannot be JSON-encoded even with default=str.
        state = {
            "messages": [HumanMessage(content="hi")],
            "weird": {(1, 2): "tuple-key"},
        }
        result = MASExecutor._serialize_state(state)
        assert result["messages"][0]["content"] == "hi"
        assert isinstance(result["weird"], str)

    def test_create_checkpointer_with_user_id_uses_factory(self):
        executor = MASExecutor(
            _seq_config(
                checkpoint_enabled=True, checkpoint_config={"type": "postgres"}
            ),
            user_id="u@example.com",
        )
        sentinel = Mock()
        with patch(
            "bili.aether.integration.checkpointer_factory.create_checkpointer_from_config",
            return_value=sentinel,
        ) as mock_factory:
            result = executor._create_checkpointer_with_user_id()
        assert result is sentinel
        assert mock_factory.call_args.kwargs["user_id"] == "u@example.com"

    def test_create_checkpointer_with_user_id_falls_back_on_import_error(self):
        from langgraph.checkpoint.memory import MemorySaver

        executor = MASExecutor(_seq_config(), user_id="u@example.com")
        # Making the factory module unimportable forces the MemorySaver fallback.
        with patch.dict(
            "sys.modules",
            {"bili.aether.integration.checkpointer_factory": None},
        ):
            result = executor._create_checkpointer_with_user_id()
        assert isinstance(result, MemorySaver)

    def test_initialize_with_user_id_creates_tenant_checkpointer(self):
        executor = MASExecutor(
            _seq_config(
                checkpoint_enabled=True, checkpoint_config={"type": "postgres"}
            ),
            user_id="u@example.com",
        )
        with patch.object(
            executor, "_create_checkpointer_with_user_id", return_value=Mock()
        ) as mock_create:
            executor.initialize()
        mock_create.assert_called_once()


# =========================================================================
# run_streaming_tokens branch coverage
# =========================================================================


class TestRunStreamingTokensBranches:
    """Drive run_streaming_tokens with a stubbed graph to cover edge branches."""

    def test_token_and_node_complete_branches(self):
        executor = MASExecutor(_seq_config())
        executor.initialize()

        chunk_text = Mock()
        chunk_text.content = "hello"
        chunk_empty = Mock()
        chunk_empty.content = ""
        chunk_list = Mock()
        chunk_list.content = [{"text": "abc"}, {"no_text": "skip"}]

        fake_stream = [
            # Empty content is skipped.
            ("messages", (chunk_empty, {"langgraph_node": "agent_0"})),
            # Missing node name is skipped.
            ("messages", (chunk_text, {"langgraph_node": ""})),
            # Structured (list) content yields only the text blocks.
            ("messages", (chunk_list, {"langgraph_node": "agent_0"})),
            # Plain string content yields a token.
            ("messages", (chunk_text, {"langgraph_node": "agent_0"})),
            # Updates mode yields a node-complete event.
            ("updates", {"agent_0": {"messages": []}}),
        ]
        executor._compiled_graph = Mock()
        executor._compiled_graph.stream.return_value = iter(fake_stream)

        events = list(executor.run_streaming_tokens({"messages": []}))
        tokens = [d["token"] for k, d in events if k == "__token__"]
        assert "abc" in tokens
        assert "hello" in tokens
        assert any(k == "__node_complete__" for k, _ in events)

    def test_emits_human_interrupt_when_graph_pauses(self):
        executor = MASExecutor(_seq_config(human_in_loop=True))
        executor.initialize()

        executor._compiled_graph = Mock()
        executor._compiled_graph.stream.return_value = iter([])
        graph_state = Mock()
        graph_state.next = ["agent_1"]
        executor._compiled_graph.get_state.return_value = graph_state

        events = list(executor.run_streaming_tokens({"messages": []}, thread_id="t1"))
        interrupts = [d for k, d in events if k == "__human_interrupt__"]
        assert len(interrupts) == 1
        assert interrupts[0]["next"] == ["agent_1"]
