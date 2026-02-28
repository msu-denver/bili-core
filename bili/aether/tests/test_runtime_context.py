# pylint: disable=missing-function-docstring,import-outside-toplevel
"""Tests for RuntimeContext dependency injection container.

Covers:
- RuntimeContext register/unregister/get/require API
- Services flow through to pipeline node kwargs
- Priority ordering: node config > parent agent > runtime context
- Threading through compile_mas() and MASExecutor
"""

from functools import partial

import pytest

from bili.aether.compiler.graph_builder import GraphBuilder
from bili.aether.runtime.context import RuntimeContext
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.pipeline_spec import (
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
    PipelineStateField,
)

# =========================================================================
# HELPERS
# =========================================================================


def _make_node_factory(name, builder):
    from bili.graph_builder.classes.node import Node

    return partial(Node, name, builder)


def _pipeline_with_service_node():
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(node_id="service_node", node_type="service_consumer"),
        ],
        edges=[
            PipelineEdgeSpec(from_node="service_node", to_node="END"),
        ],
        state_fields=[
            PipelineStateField(
                name="service_result", type="str", default="", reducer="replace"
            ),
        ],
    )


def _mas_config():
    return MASConfig(
        mas_id="test_ctx",
        name="Test RuntimeContext",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="test_agent",
                role="tester",
                objective="Test runtime context injection",
                pipeline=_pipeline_with_service_node(),
            )
        ],
        checkpoint_enabled=False,
    )


# =========================================================================
# TEST: RuntimeContext API
# =========================================================================


class TestRuntimeContextAPI:
    """Tests for RuntimeContext register/get/require methods."""

    def test_register_and_get(self):
        ctx = RuntimeContext()
        ctx.register("my_service", "hello")
        assert ctx.get("my_service") == "hello"

    def test_get_missing_returns_default(self):
        ctx = RuntimeContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "fallback") == "fallback"

    def test_require_present(self):
        ctx = RuntimeContext()
        ctx.register("svc", 42)
        assert ctx.require("svc") == 42

    def test_require_missing_raises(self):
        ctx = RuntimeContext()
        with pytest.raises(KeyError, match="not registered"):
            ctx.require("missing")

    def test_register_duplicate_raises(self):
        ctx = RuntimeContext()
        ctx.register("svc", 1)
        with pytest.raises(ValueError, match="already registered"):
            ctx.register("svc", 2)

    def test_unregister(self):
        ctx = RuntimeContext()
        ctx.register("svc", 1)
        ctx.unregister("svc")
        assert ctx.get("svc") is None

    def test_unregister_missing_raises(self):
        ctx = RuntimeContext()
        with pytest.raises(KeyError, match="not registered"):
            ctx.unregister("missing")

    def test_chaining(self):
        ctx = RuntimeContext()
        result = ctx.register("a", 1).register("b", 2).register("c", 3)
        assert result is ctx
        assert len(ctx) == 3

    def test_contains(self):
        ctx = RuntimeContext()
        ctx.register("svc", 1)
        assert "svc" in ctx
        assert "missing" not in ctx

    def test_len(self):
        ctx = RuntimeContext()
        assert len(ctx) == 0
        ctx.register("a", 1)
        assert len(ctx) == 1

    def test_iter(self):
        ctx = RuntimeContext()
        ctx.register("a", 1)
        ctx.register("b", 2)
        assert sorted(ctx) == ["a", "b"]

    def test_as_dict(self):
        ctx = RuntimeContext()
        ctx.register("a", 1)
        ctx.register("b", 2)
        d = ctx.as_dict()
        assert d == {"a": 1, "b": 2}
        # Should be a copy
        d["c"] = 3
        assert "c" not in ctx

    def test_repr(self):
        ctx = RuntimeContext()
        ctx.register("beta", 1)
        ctx.register("alpha", 2)
        assert repr(ctx) == "RuntimeContext(['alpha', 'beta'])"


# =========================================================================
# TEST: RuntimeContext through GraphBuilder
# =========================================================================


class TestRuntimeContextInGraphBuilder:
    """Tests for runtime context injection into pipeline nodes."""

    def test_service_injected_into_node_kwargs(self):
        """Services from RuntimeContext appear in node builder kwargs."""
        received_kwargs = {}

        def _capturing_builder(**kwargs):
            received_kwargs.update(kwargs)

            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "consumer",
                    "agent_outputs": {},
                    "service_result": str(kwargs.get("my_model", "none")),
                }

            return _execute

        ctx = RuntimeContext()
        ctx.register("my_model", "FakeModel()")

        custom_reg = {
            "service_consumer": _make_node_factory(
                "service_consumer", _capturing_builder
            ),
        }
        config = _mas_config()
        builder = GraphBuilder(
            config, custom_node_registry=custom_reg, runtime_context=ctx
        )
        builder.build()

        assert "my_model" in received_kwargs
        assert received_kwargs["my_model"] == "FakeModel()"

    def test_node_config_overrides_runtime_context(self):
        """Node-specific config in YAML overrides RuntimeContext values."""
        received_kwargs = {}

        def _capturing_builder(**kwargs):
            received_kwargs.update(kwargs)

            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "consumer",
                    "agent_outputs": {},
                    "service_result": "",
                }

            return _execute

        ctx = RuntimeContext()
        ctx.register("temperature", 0.5)

        # Node config sets temperature=0.9 which should override context
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="service_node",
                    node_type="service_consumer",
                    config={"temperature": 0.9},
                ),
            ],
            edges=[PipelineEdgeSpec(from_node="service_node", to_node="END")],
        )
        agent = AgentSpec(
            agent_id="test_agent",
            role="tester",
            objective="Test priority ordering in kwargs",
            pipeline=pipeline,
        )
        config = MASConfig(
            mas_id="test_priority",
            name="Test Priority",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
            checkpoint_enabled=False,
        )

        custom_reg = {
            "service_consumer": _make_node_factory(
                "service_consumer", _capturing_builder
            ),
        }
        builder = GraphBuilder(
            config, custom_node_registry=custom_reg, runtime_context=ctx
        )
        builder.build()

        # Node config should win
        assert received_kwargs["temperature"] == 0.9

    def test_multiple_services_injected(self):
        """Multiple services from RuntimeContext all appear in kwargs."""
        received_kwargs = {}

        def _capturing_builder(**kwargs):
            received_kwargs.update(kwargs)

            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "consumer",
                    "agent_outputs": {},
                    "service_result": "",
                }

            return _execute

        ctx = RuntimeContext()
        ctx.register("celery_app", "FakeCelery")
        ctx.register("http_client", "FakeHTTP")
        ctx.register("sentiment_model", "FakeModel")

        custom_reg = {
            "service_consumer": _make_node_factory(
                "service_consumer", _capturing_builder
            ),
        }
        config = _mas_config()
        builder = GraphBuilder(
            config, custom_node_registry=custom_reg, runtime_context=ctx
        )
        builder.build()

        assert received_kwargs.get("celery_app") == "FakeCelery"
        assert received_kwargs.get("http_client") == "FakeHTTP"
        assert received_kwargs.get("sentiment_model") == "FakeModel"


# =========================================================================
# TEST: RuntimeContext through compile_mas and MASExecutor
# =========================================================================


class TestRuntimeContextThreading:
    """Tests for runtime context threading through the full stack."""

    def test_compile_mas_accepts_runtime_context(self):
        from bili.aether.compiler import compile_mas

        def _stub_builder(**_kwargs):
            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "stub",
                    "agent_outputs": {},
                    "service_result": "",
                }

            return _execute

        ctx = RuntimeContext()
        ctx.register("my_service", "value")

        custom_reg = {
            "service_consumer": _make_node_factory("service_consumer", _stub_builder),
        }
        config = _mas_config()
        compiled = compile_mas(
            config, custom_node_registry=custom_reg, runtime_context=ctx
        )
        assert "test_agent" in compiled.agent_nodes

    def test_executor_e2e_with_runtime_context(self):
        """End-to-end: RuntimeContext services available during execution."""
        from bili.aether.runtime.executor import MASExecutor

        def _service_builder(**kwargs):
            service_value = kwargs.get("my_service", "missing")

            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "consumer",
                    "agent_outputs": {"consumer": {"used_service": service_value}},
                    "service_result": service_value,
                }

            return _execute

        ctx = RuntimeContext()
        ctx.register("my_service", "injected_value")

        custom_reg = {
            "service_consumer": _make_node_factory(
                "service_consumer", _service_builder
            ),
        }
        config = _mas_config()
        executor = MASExecutor(
            config, custom_node_registry=custom_reg, runtime_context=ctx
        )
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert result.final_state.get("service_result") == "injected_value"
