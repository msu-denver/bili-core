# pylint: disable=missing-function-docstring,import-outside-toplevel
"""Tests for AETHER streaming execution.

Covers:
- StreamEvent model and serialization
- StreamFilter accept/reject logic
- MASExecutor.stream() synchronous streaming
- MASExecutor.astream() async streaming
- Event lifecycle (run_start → node events → run_end)
- Error handling during streaming
- StreamFilter convenience constructors
"""

import asyncio
from functools import partial

import pytest

from bili.aether.runtime.streaming import StreamEvent, StreamEventType, StreamFilter

# =========================================================================
# HELPERS
# =========================================================================


def _make_node_factory(name, builder):
    from bili.graph_builder.classes.node import Node

    return partial(Node, name, builder)


def _stub_builder(**_kwargs):
    """Builder that returns a simple node function."""

    def _execute(state: dict) -> dict:
        return {
            "messages": state.get("messages", []),
            "current_agent": "test",
            "agent_outputs": {"test": {"status": "ok"}},
        }

    return _execute


def _error_builder(**_kwargs):
    """Builder that returns a node function that raises."""

    def _execute(state: dict) -> dict:
        raise ValueError("Simulated node failure")

    return _execute


def _pipeline_config(node_type="stub_node"):
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    return PipelineSpec(
        nodes=[PipelineNodeSpec(node_id="step1", node_type=node_type)],
        edges=[PipelineEdgeSpec(from_node="step1", to_node="END")],
    )


def _mas_config(node_type="stub_node"):
    from bili.aether.schema import AgentSpec, MASConfig, WorkflowType

    return MASConfig(
        mas_id="test_stream",
        name="Test Streaming",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="streamer",
                role="tester",
                objective="Test streaming execution",
                pipeline=_pipeline_config(node_type),
            )
        ],
        checkpoint_enabled=False,
    )


def _make_executor(node_type="stub_node", builder=None):
    from bili.aether.runtime.executor import MASExecutor

    actual_builder = builder or _stub_builder
    custom_reg = {
        node_type: _make_node_factory(node_type, actual_builder),
    }
    config = _mas_config(node_type)
    executor = MASExecutor(config, custom_node_registry=custom_reg)
    executor.initialize()
    return executor


# =========================================================================
# TEST: StreamEvent model
# =========================================================================


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_create_event(self):
        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"content": "hello"},
        )
        assert event.event_type == "token"
        assert event.data == {"content": "hello"}
        assert event.timestamp  # non-empty

    def test_to_dict(self):
        event = StreamEvent(
            event_type=StreamEventType.NODE_START,
            data={"name": "my_node"},
            node_name="my_node",
            agent_id="agent_1",
            run_id="run_123",
        )
        d = event.to_dict()
        assert d["event_type"] == "node_start"
        assert d["node_name"] == "my_node"
        assert d["agent_id"] == "agent_1"
        assert d["run_id"] == "run_123"

    def test_to_sse(self):
        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"content": "world"},
        )
        sse = event.to_sse()
        assert sse.startswith("event: token\n")
        assert '"content": "world"' in sse
        assert sse.endswith("\n\n")

    def test_event_types_are_strings(self):
        assert StreamEventType.TOKEN == "token"
        assert StreamEventType.NODE_START == "node_start"
        assert StreamEventType.RUN_END == "run_end"
        assert StreamEventType.ERROR == "error"


# =========================================================================
# TEST: StreamFilter
# =========================================================================


class TestStreamFilter:
    """Tests for StreamFilter declarative filtering."""

    def test_default_accepts_all(self):
        f = StreamFilter()
        event = StreamEvent(event_type=StreamEventType.TOKEN)
        assert f.accepts(event) is True

    def test_include_types(self):
        f = StreamFilter(include_types={StreamEventType.TOKEN})
        assert f.accepts(StreamEvent(event_type=StreamEventType.TOKEN)) is True
        assert f.accepts(StreamEvent(event_type=StreamEventType.NODE_START)) is False

    def test_exclude_types(self):
        f = StreamFilter(exclude_types={StreamEventType.NODE_START})
        assert f.accepts(StreamEvent(event_type=StreamEventType.TOKEN)) is True
        assert f.accepts(StreamEvent(event_type=StreamEventType.NODE_START)) is False

    def test_include_takes_precedence_over_exclude(self):
        f = StreamFilter(
            include_types={StreamEventType.TOKEN},
            exclude_types={StreamEventType.TOKEN},
        )
        # include_types takes precedence; TOKEN is in include_types so it passes
        assert f.accepts(StreamEvent(event_type=StreamEventType.TOKEN)) is True

    def test_include_agents(self):
        f = StreamFilter(include_agents={"agent_a"})
        assert (
            f.accepts(StreamEvent(event_type=StreamEventType.TOKEN, agent_id="agent_a"))
            is True
        )
        assert (
            f.accepts(StreamEvent(event_type=StreamEventType.TOKEN, agent_id="agent_b"))
            is False
        )
        # No agent_id set — passes (no filtering on None)
        assert (
            f.accepts(StreamEvent(event_type=StreamEventType.TOKEN, agent_id=None))
            is True
        )

    def test_include_nodes(self):
        f = StreamFilter(include_nodes={"node_x"})
        assert (
            f.accepts(
                StreamEvent(event_type=StreamEventType.NODE_END, node_name="node_x")
            )
            is True
        )
        assert (
            f.accepts(
                StreamEvent(event_type=StreamEventType.NODE_END, node_name="node_y")
            )
            is False
        )

    def test_tokens_only_convenience(self):
        f = StreamFilter.tokens_only()
        assert f.accepts(StreamEvent(event_type=StreamEventType.TOKEN)) is True
        assert f.accepts(StreamEvent(event_type=StreamEventType.NODE_START)) is False
        assert f.accepts(StreamEvent(event_type=StreamEventType.RUN_END)) is False

    def test_lifecycle_only_convenience(self):
        f = StreamFilter.lifecycle_only()
        assert f.accepts(StreamEvent(event_type=StreamEventType.TOKEN)) is False
        assert f.accepts(StreamEvent(event_type=StreamEventType.NODE_START)) is True
        assert f.accepts(StreamEvent(event_type=StreamEventType.RUN_END)) is True


# =========================================================================
# TEST: MASExecutor.stream() — synchronous streaming
# =========================================================================


class TestSyncStreaming:
    """Tests for MASExecutor.stream() synchronous streaming."""

    def test_stream_yields_events(self):
        executor = _make_executor()
        events = list(executor.stream())
        assert len(events) >= 2  # at least run_start + run_end

    def test_stream_lifecycle_events(self):
        executor = _make_executor()
        events = list(executor.stream())
        event_types = [e.event_type for e in events]
        assert event_types[0] == StreamEventType.RUN_START
        assert event_types[-1] == StreamEventType.RUN_END

    def test_stream_contains_node_events(self):
        executor = _make_executor()
        events = list(executor.stream())
        node_events = [e for e in events if e.event_type == StreamEventType.NODE_END]
        assert len(node_events) >= 1

    def test_stream_run_id_consistent(self):
        executor = _make_executor()
        events = list(executor.stream())
        run_ids = {e.run_id for e in events}
        assert len(run_ids) == 1  # all events share one run_id

    def test_stream_with_input_data(self):
        executor = _make_executor()
        events = list(executor.stream(input_data={"messages": []}))
        assert any(e.event_type == StreamEventType.RUN_END for e in events)

    def test_stream_filter_applied(self):
        executor = _make_executor()
        lifecycle_filter = StreamFilter.lifecycle_only()
        events = list(executor.stream(stream_filter=lifecycle_filter))
        # Should only have lifecycle events
        for event in events:
            assert event.event_type in {
                StreamEventType.NODE_START,
                StreamEventType.NODE_END,
                StreamEventType.AGENT_START,
                StreamEventType.AGENT_END,
                StreamEventType.RUN_START,
                StreamEventType.RUN_END,
            }

    def test_stream_pipeline_error_completes_gracefully(self):
        """Pipeline node errors are caught by the pipeline wrapper, not streamed as ERROR events."""
        executor = _make_executor(node_type="error_node", builder=_error_builder)
        events = list(executor.stream())
        # Pipeline wrapper catches errors internally, so execution completes
        event_types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in event_types
        assert StreamEventType.RUN_END in event_types

    def test_stream_not_initialized_raises(self):
        from bili.aether.runtime.executor import MASExecutor

        config = _mas_config()
        executor = MASExecutor(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.stream())

    def test_stream_run_start_data(self):
        executor = _make_executor()
        events = list(executor.stream())
        start = events[0]
        assert start.event_type == StreamEventType.RUN_START
        assert "execution_id" in start.data
        assert start.data["mas_id"] == "test_stream"


# =========================================================================
# TEST: MASExecutor.astream() — async streaming
# =========================================================================


class TestAsyncStreaming:
    """Tests for MASExecutor.astream() async streaming."""

    def test_astream_yields_events(self):
        executor = _make_executor()

        async def _collect():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        events = asyncio.run(_collect())
        assert len(events) >= 2  # at least run_start + run_end

    def test_astream_lifecycle_events(self):
        executor = _make_executor()

        async def _collect():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        events = asyncio.run(_collect())
        event_types = [e.event_type for e in events]
        assert event_types[0] == StreamEventType.RUN_START
        assert event_types[-1] == StreamEventType.RUN_END

    def test_astream_run_id_consistent(self):
        executor = _make_executor()

        async def _collect():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        events = asyncio.run(_collect())
        run_ids = {e.run_id for e in events}
        assert len(run_ids) == 1

    def test_astream_with_filter(self):
        executor = _make_executor()
        only_lifecycle = StreamFilter(exclude_types={StreamEventType.TOKEN})

        async def _collect():
            events = []
            async for event in executor.astream(stream_filter=only_lifecycle):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        assert len(token_events) == 0

    def test_astream_pipeline_error_completes_gracefully(self):
        """Pipeline node errors are caught by the pipeline wrapper, not streamed as ERROR events."""
        executor = _make_executor(node_type="error_node", builder=_error_builder)

        async def _collect():
            events = []
            async for event in executor.astream():
                events.append(event)
            return events

        events = asyncio.run(_collect())
        event_types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in event_types
        assert StreamEventType.RUN_END in event_types

    def test_astream_not_initialized_raises(self):
        from bili.aether.runtime.executor import MASExecutor

        config = _mas_config()
        executor = MASExecutor(config)

        async def _run():
            events = []
            async for event in executor.astream():
                events.append(event)

        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(_run())

    def test_astream_with_input_data(self):
        executor = _make_executor()

        async def _collect():
            events = []
            async for event in executor.astream(input_data={"messages": []}):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        assert any(e.event_type == StreamEventType.RUN_END for e in events)
