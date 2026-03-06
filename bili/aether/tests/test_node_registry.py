"""Tests for the node registration API and custom registry threading.

Covers:
- register_node() / unregister_node() in langchain_loader
- custom_node_registry parameter through GraphBuilder → compile_mas → MASExecutor
- Custom registry nodes resolve before global GRAPH_NODE_REGISTRY
- Error messages include both custom and global node names
"""

from functools import partial

import pytest

from bili.aether.compiler.graph_builder import GraphBuilder
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.pipeline_spec import (
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
)

# =========================================================================
# HELPERS
# =========================================================================


def _build_echo_node(**kwargs):
    """A trivial node builder that returns messages unchanged."""

    def _execute(state: dict) -> dict:
        return {
            "messages": state.get("messages", []),
            "current_agent": "echo",
            "agent_outputs": {"echo": {"ran": True}},
        }

    return _execute


def _make_node_factory(name, builder):
    """Create a partial(Node, name, builder) — same pattern as GRAPH_NODE_REGISTRY."""
    from bili.graph_builder.classes.node import Node

    return partial(Node, name, builder)


def _pipeline_with_custom_node(node_type="custom_echo"):
    """Pipeline that references a custom node_type."""
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(node_id="echo", node_type=node_type),
            PipelineNodeSpec(
                node_id="finish",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_finish",
                    "role": "finisher",
                    "objective": "Wrap up the pipeline processing",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="echo", to_node="finish"),
            PipelineEdgeSpec(from_node="finish", to_node="END"),
        ],
    )


def _agent_with_custom_pipeline(node_type="custom_echo"):
    return AgentSpec(
        agent_id="test_agent",
        role="tester",
        objective="Test custom node registration",
        pipeline=_pipeline_with_custom_node(node_type),
    )


def _mas_config(agent=None):
    return MASConfig(
        mas_id="test_registry",
        name="Test Custom Registry",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent or _agent_with_custom_pipeline()],
        checkpoint_enabled=False,
    )


# =========================================================================
# TEST: register_node / unregister_node
# =========================================================================


class TestRegisterNode:
    """Tests for the global register_node() API."""

    def test_register_and_resolve(self):
        """Registered node appears in GRAPH_NODE_REGISTRY."""
        from bili.loaders.langchain_loader import (
            GRAPH_NODE_REGISTRY,
            register_node,
            unregister_node,
        )

        factory = _make_node_factory("test_custom", _build_echo_node)
        try:
            register_node("test_custom", factory)
            assert "test_custom" in GRAPH_NODE_REGISTRY
            assert GRAPH_NODE_REGISTRY["test_custom"] is factory
        finally:
            unregister_node("test_custom")

    def test_register_duplicate_raises(self):
        """register_node() raises ValueError for duplicate names."""
        from bili.loaders.langchain_loader import register_node, unregister_node

        factory = _make_node_factory("dup_test", _build_echo_node)
        try:
            register_node("dup_test", factory)
            with pytest.raises(ValueError, match="already registered"):
                register_node("dup_test", factory)
        finally:
            unregister_node("dup_test")

    def test_unregister_missing_raises(self):
        """unregister_node() raises KeyError for unknown names."""
        from bili.loaders.langchain_loader import unregister_node

        with pytest.raises(KeyError, match="not registered"):
            unregister_node("nonexistent_node_xyz")

    def test_register_unregister_roundtrip(self):
        """Node can be registered and then cleanly unregistered."""
        from bili.loaders.langchain_loader import (
            GRAPH_NODE_REGISTRY,
            register_node,
            unregister_node,
        )

        factory = _make_node_factory("roundtrip_test", _build_echo_node)
        register_node("roundtrip_test", factory)
        assert "roundtrip_test" in GRAPH_NODE_REGISTRY
        unregister_node("roundtrip_test")
        assert "roundtrip_test" not in GRAPH_NODE_REGISTRY


# =========================================================================
# TEST: custom_node_registry through GraphBuilder
# =========================================================================


class TestCustomNodeRegistry:
    """Tests for per-compilation custom_node_registry in GraphBuilder."""

    def test_custom_registry_resolves_node(self):
        """Custom registry node is used when building a pipeline."""
        factory = _make_node_factory("custom_echo", _build_echo_node)
        custom_reg = {"custom_echo": factory}

        config = _mas_config()
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        # Should compile without error and have the agent node
        assert "test_agent" in compiled.agent_nodes

    def test_custom_registry_takes_precedence(self):
        """Custom registry is checked before global GRAPH_NODE_REGISTRY."""
        call_tracker = {"custom_called": False}

        def _tracking_builder(**kwargs):
            call_tracker["custom_called"] = True
            return _build_echo_node(**kwargs)

        factory = _make_node_factory("add_persona_and_summary", _tracking_builder)
        custom_reg = {"add_persona_and_summary": factory}

        # Use a pipeline that references "add_persona_and_summary" — exists
        # in both custom and global registries
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="persona", node_type="add_persona_and_summary"
                ),
                PipelineNodeSpec(
                    node_id="finish",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "finisher",
                        "objective": "Finish the pipeline processing",
                    },
                ),
            ],
            edges=[
                PipelineEdgeSpec(from_node="persona", to_node="finish"),
                PipelineEdgeSpec(from_node="finish", to_node="END"),
            ],
        )
        agent = AgentSpec(
            agent_id="test_agent",
            role="tester",
            objective="Test precedence",
            pipeline=pipeline,
        )
        config = _mas_config(agent)
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        builder.build()

        assert call_tracker[
            "custom_called"
        ], "Custom registry should be checked before global registry"

    def test_unknown_node_type_error_includes_both_registries(self):
        """Error message for unknown node_type lists custom + global nodes."""
        custom_reg = {
            "my_special_node": _make_node_factory("my_special_node", _build_echo_node)
        }

        agent = _agent_with_custom_pipeline(node_type="nonexistent_type")
        config = _mas_config(agent)
        builder = GraphBuilder(config, custom_node_registry=custom_reg)

        with pytest.raises(ValueError, match="nonexistent_type") as exc_info:
            builder.build()

        error_msg = str(exc_info.value)
        assert (
            "my_special_node" in error_msg
        ), "Custom node should appear in available list"
        assert "react_agent" in error_msg, "Global node should appear in available list"

    def test_empty_custom_registry_falls_back_to_global(self):
        """With empty custom registry, global GRAPH_NODE_REGISTRY is used."""
        # Use a pipeline with a built-in registry node
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="persona",
                    node_type="add_persona_and_summary",
                ),
                PipelineNodeSpec(
                    node_id="finish",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "finisher",
                        "objective": "Finish the pipeline processing",
                    },
                ),
            ],
            edges=[
                PipelineEdgeSpec(from_node="persona", to_node="finish"),
                PipelineEdgeSpec(from_node="finish", to_node="END"),
            ],
        )
        agent = AgentSpec(
            agent_id="test_agent",
            role="tester",
            objective="Test fallback",
            pipeline=pipeline,
        )
        config = _mas_config(agent)
        builder = GraphBuilder(config, custom_node_registry={})
        compiled = builder.build()

        assert "test_agent" in compiled.agent_nodes


# =========================================================================
# TEST: custom_node_registry through compile_mas
# =========================================================================


class TestCompileMasRegistry:
    """Tests for custom_node_registry threading through compile_mas()."""

    def test_compile_mas_accepts_custom_registry(self):
        """compile_mas() passes custom_node_registry to GraphBuilder."""
        from bili.aether.compiler import compile_mas

        factory = _make_node_factory("custom_echo", _build_echo_node)
        custom_reg = {"custom_echo": factory}

        config = _mas_config()
        compiled = compile_mas(config, custom_node_registry=custom_reg)

        assert "test_agent" in compiled.agent_nodes

    def test_compile_mas_without_registry_works(self):
        """compile_mas() works without custom_node_registry (backwards compat)."""
        from bili.aether.compiler import compile_mas

        # Use simple agent without pipeline (no registry needed)
        agent = AgentSpec(
            agent_id="simple",
            role="simple",
            objective="No pipeline",
        )
        config = MASConfig(
            mas_id="test",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
            checkpoint_enabled=False,
        )
        compiled = compile_mas(config)
        assert "simple" in compiled.agent_nodes


# =========================================================================
# TEST: custom_node_registry through MASExecutor
# =========================================================================


class TestMASExecutorRegistry:
    """Tests for custom_node_registry threading through MASExecutor."""

    def test_executor_accepts_custom_registry(self):
        """MASExecutor passes custom_node_registry through to compilation."""
        from bili.aether.runtime.executor import MASExecutor

        factory = _make_node_factory("custom_echo", _build_echo_node)
        custom_reg = {"custom_echo": factory}

        config = _mas_config()
        executor = MASExecutor(config, custom_node_registry=custom_reg)
        executor.initialize()

        assert executor._compiled_graph is not None

    def test_executor_e2e_with_custom_node(self):
        """End-to-end: custom node runs in pipeline via MASExecutor."""
        from bili.aether.runtime.executor import MASExecutor

        factory = _make_node_factory("custom_echo", _build_echo_node)
        custom_reg = {"custom_echo": factory}

        config = _mas_config()
        executor = MASExecutor(config, custom_node_registry=custom_reg)
        executor.initialize()

        # Run without checkpointer
        result = executor.run(save_results=False)
        assert result.success
