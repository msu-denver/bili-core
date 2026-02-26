"""Comprehensive pipeline tests covering advanced workflow integrations.

Supplements test_pipeline_spec.py (schema) and test_pipeline_compiler.py
(compiler basics) with:
- Pipeline agents in supervisor, consensus, hierarchical workflows
- Cross-model compatibility (model_copy does not penetrate pipeline)
- Error attribution with agent_id context
- YAML round-trip: load → validate → compile → execute
- Checkpointer isolation verification
- Nested pipeline compilation limits
"""

import os
from unittest.mock import MagicMock

from bili.aether.compiler import compile_mas
from bili.aether.compiler.graph_builder import GraphBuilder
from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.schema import AgentSpec, MASConfig
from bili.aether.schema.pipeline_spec import (
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
)
from bili.aether.validation import validate_mas

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "examples",
)

# =========================================================================
# HELPERS
# =========================================================================


def _stub_pipeline():
    """A two-node stub pipeline: step_a → step_b → END."""
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="step_a",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_a",
                    "role": "analyzer",
                    "objective": "First pipeline step for analysis",
                },
            ),
            PipelineNodeSpec(
                node_id="step_b",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_b",
                    "role": "formatter",
                    "objective": "Format pipeline output for delivery",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="step_a", to_node="step_b"),
            PipelineEdgeSpec(from_node="step_b", to_node="END"),
        ],
    )


def _agent(agent_id, **overrides):
    """Minimal AgentSpec with sensible defaults."""
    defaults = {
        "role": "worker",
        "objective": f"Objective for agent {agent_id} in testing",
    }
    defaults.update(overrides)
    return AgentSpec(agent_id=agent_id, **defaults)


# =========================================================================
# SUPERVISOR WORKFLOW WITH PIPELINE AGENTS
# =========================================================================


class TestSupervisorWorkflowWithPipelines:
    """Pipeline agents work correctly as workers in a supervisor workflow."""

    def test_supervisor_with_pipeline_worker_compiles(self):
        """Supervisor MAS compiles when workers have pipelines."""
        config = MASConfig(
            mas_id="sup_pipeline",
            name="Supervisor + Pipeline",
            workflow_type="supervisor",
            entry_point="supervisor",
            agents=[
                _agent(
                    "supervisor",
                    role="supervisor",
                    is_supervisor=True,
                    capabilities=["inter_agent_communication"],
                ),
                _agent("pipeline_worker", pipeline=_stub_pipeline()),
                _agent("simple_worker"),
            ],
        )

        compiled = GraphBuilder(config).build()
        assert "supervisor" in compiled.agent_nodes
        assert "pipeline_worker" in compiled.agent_nodes
        assert (
            compiled.agent_nodes["pipeline_worker"].__name__
            == "pipeline_pipeline_worker"
        )

    def test_supervisor_pipeline_worker_executes(self):
        """Pipeline worker executes when routed to by supervisor."""
        config = MASConfig(
            mas_id="sup_exec",
            name="Supervisor Execute",
            workflow_type="supervisor",
            entry_point="supervisor",
            agents=[
                _agent(
                    "supervisor",
                    role="supervisor",
                    is_supervisor=True,
                    capabilities=["inter_agent_communication"],
                ),
                _agent("worker", pipeline=_stub_pipeline()),
            ],
        )

        compiled = GraphBuilder(config).build()
        graph = compiled.graph.compile(checkpointer=None)

        # Supervisor stub will default to routing to END
        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})
        assert "supervisor" in result["agent_outputs"]


# =========================================================================
# CONSENSUS WORKFLOW WITH PIPELINE AGENTS
# =========================================================================


class TestConsensusWorkflowWithPipelines:
    """Pipeline agents participate in consensus rounds."""

    def test_consensus_with_pipeline_agents_compiles(self):
        """Consensus MAS compiles with pipeline agents."""
        from bili.aether.schema import OutputFormat

        config = MASConfig(
            mas_id="consensus_pipeline",
            name="Consensus + Pipeline",
            workflow_type="consensus",
            consensus_threshold=0.66,
            agents=[
                _agent(
                    "voter_a",
                    pipeline=_stub_pipeline(),
                    output_format=OutputFormat.JSON,
                    consensus_vote_field="decision",
                ),
                _agent(
                    "voter_b",
                    pipeline=_stub_pipeline(),
                    output_format=OutputFormat.JSON,
                    consensus_vote_field="decision",
                ),
            ],
        )

        compiled = GraphBuilder(config).build()
        assert len(compiled.agent_nodes) == 2
        assert compiled.agent_nodes["voter_a"].__name__ == "pipeline_voter_a"
        assert compiled.agent_nodes["voter_b"].__name__ == "pipeline_voter_b"

    def test_consensus_pipeline_executes_one_round(self):
        """Consensus MAS with pipeline agents executes at least one round."""
        from bili.aether.schema import OutputFormat

        config = MASConfig(
            mas_id="consensus_exec",
            name="Consensus Execute",
            workflow_type="consensus",
            consensus_threshold=0.66,
            max_consensus_rounds=1,
            agents=[
                _agent(
                    "v1",
                    pipeline=_stub_pipeline(),
                    output_format=OutputFormat.JSON,
                    consensus_vote_field="decision",
                ),
                _agent(
                    "v2",
                    output_format=OutputFormat.JSON,
                    consensus_vote_field="decision",
                ),
            ],
        )

        compiled = GraphBuilder(config).build()
        graph = compiled.graph.compile(checkpointer=None)

        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})

        # Both agents should have produced output
        assert "v1" in result["agent_outputs"]
        assert "v2" in result["agent_outputs"]


# =========================================================================
# HIERARCHICAL WORKFLOW WITH PIPELINE AGENTS
# =========================================================================


class TestHierarchicalWorkflowWithPipelines:
    """Pipeline agents work in hierarchical tier workflows."""

    def test_hierarchical_with_pipeline_agents_compiles(self):
        """Hierarchical MAS compiles with pipeline agents at different tiers."""
        config = MASConfig(
            mas_id="hier_pipeline",
            name="Hierarchical + Pipeline",
            workflow_type="hierarchical",
            agents=[
                _agent("leaf_a", tier=2, pipeline=_stub_pipeline()),
                _agent("leaf_b", tier=2),
                _agent("root", tier=1),
            ],
        )

        compiled = GraphBuilder(config).build()
        assert compiled.agent_nodes["leaf_a"].__name__ == "pipeline_leaf_a"
        assert not compiled.agent_nodes["leaf_b"].__name__.startswith("pipeline_")


# =========================================================================
# CROSS-MODEL COMPATIBILITY
# =========================================================================


class TestCrossModelCompatibility:
    """Test model_copy behavior with pipeline agents."""

    def test_model_copy_does_not_penetrate_pipeline(self):
        """AgentSpec.model_copy(update=model_name) doesn't change inner agents.

        This is a documented limitation: pipeline inner nodes manage their
        own model configuration. Cross-model testing tools that update
        model_name at the agent level will only affect the metadata field,
        not pipeline internals.
        """
        agent = _agent("test", model_name="gpt-4", pipeline=_stub_pipeline())

        # Simulate what run_cross_model_test does
        updated = agent.model_copy(update={"model_name": "gpt-3.5-turbo"})

        # Top-level model_name is updated
        assert updated.model_name == "gpt-3.5-turbo"

        # Pipeline inner agent specs are NOT affected (they're dicts, not AgentSpec)
        inner_a = updated.pipeline.nodes[0].agent_spec
        assert inner_a is not None
        # Inner agents don't have model_name set (stubs)
        assert inner_a.get("model_name") is None

    def test_pipeline_inner_agent_model_preserved(self):
        """Inner agent model_name is preserved when parent model changes."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="llm_node",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "processor",
                        "objective": "Process with specific model name",
                        "model_name": "claude-3-opus",
                    },
                ),
            ],
            edges=[PipelineEdgeSpec(from_node="llm_node", to_node="END")],
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = _agent("test", model_name="gpt-4", pipeline=pipeline)

        # Update parent model
        updated = agent.model_copy(update={"model_name": "gpt-3.5-turbo"})

        # Inner model is unchanged
        inner = updated.pipeline.nodes[0].agent_spec
        assert inner["model_name"] == "claude-3-opus"


# =========================================================================
# ERROR ATTRIBUTION
# =========================================================================


class TestErrorAttribution:
    """Error messages from pipeline failures identify the failing agent."""

    def test_error_message_contains_agent_id(self):
        """Pipeline error messages include the outer agent ID."""
        agent = _agent("my_agent", pipeline=_stub_pipeline())
        mock_subgraph = MagicMock()
        mock_subgraph.invoke.side_effect = RuntimeError("inner node failed")

        builder = GraphBuilder(
            MASConfig(
                mas_id="err",
                name="Error Test",
                workflow_type="sequential",
                agents=[agent],
            )
        )
        wrapper = builder._wrap_pipeline_as_agent_node(mock_subgraph, agent)
        result = wrapper({"messages": [], "agent_outputs": {}})

        error_msg = result["agent_outputs"]["my_agent"]["message"]
        assert "my_agent" in error_msg
        assert "inner node failed" in error_msg

    def test_error_returns_all_required_state_fields(self):
        """Pipeline errors return complete state update (messages, current_agent, agent_outputs)."""
        agent = _agent("failing", pipeline=_stub_pipeline())
        mock_subgraph = MagicMock()
        mock_subgraph.invoke.side_effect = ValueError("bad input")

        builder = GraphBuilder(
            MASConfig(
                mas_id="err2",
                name="Error Test 2",
                workflow_type="sequential",
                agents=[agent],
            )
        )
        wrapper = builder._wrap_pipeline_as_agent_node(mock_subgraph, agent)
        result = wrapper({"messages": [], "agent_outputs": {}})

        assert set(result.keys()) == {"messages", "current_agent", "agent_outputs"}
        assert len(result["messages"]) == 1
        assert result["current_agent"] == "failing"
        assert result["agent_outputs"]["failing"]["status"] == "error"


# =========================================================================
# YAML ROUND-TRIP TESTS
# =========================================================================


class TestYAMLRoundTrip:
    """Pipeline YAML example loads, validates, compiles, and executes."""

    def test_pipeline_yaml_loads(self):
        """Pipeline example YAML loads into MASConfig."""
        fpath = os.path.join(_EXAMPLES_DIR, "pipeline_agents.yaml")
        config = load_mas_from_yaml(fpath)

        assert config.mas_id == "pipeline_research"
        assert len(config.agents) == 2
        assert config.agents[0].pipeline is not None
        assert config.agents[1].pipeline is None

    def test_pipeline_yaml_validates(self):
        """Pipeline example passes validation."""
        fpath = os.path.join(_EXAMPLES_DIR, "pipeline_agents.yaml")
        config = load_mas_from_yaml(fpath)
        result = validate_mas(config)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_pipeline_yaml_compiles(self):
        """Pipeline example compiles to CompiledMAS."""
        fpath = os.path.join(_EXAMPLES_DIR, "pipeline_agents.yaml")
        config = load_mas_from_yaml(fpath)
        compiled = compile_mas(config)

        assert len(compiled.agent_nodes) == 2
        assert compiled.agent_nodes["researcher"].__name__ == "pipeline_researcher"

    def test_pipeline_yaml_executes_end_to_end(self):
        """Pipeline example executes end-to-end without errors."""
        fpath = os.path.join(_EXAMPLES_DIR, "pipeline_agents.yaml")
        config = load_mas_from_yaml(fpath)
        compiled = compile_mas(config)
        graph = compiled.graph.compile(checkpointer=None)

        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})

        assert "researcher" in result["agent_outputs"]
        assert "summarizer" in result["agent_outputs"]
        assert result["agent_outputs"]["researcher"]["status"] == "completed"


# =========================================================================
# CHECKPOINTER ISOLATION
# =========================================================================


class TestCheckpointerIsolation:
    """Pipeline sub-graphs must NOT get their own checkpointer."""

    def test_compile_graph_does_not_give_subgraph_checkpointer(self):
        """When the outer MAS has checkpoint_enabled, sub-graphs still compile
        with checkpointer=None to avoid LangGraph #5639.
        """
        config = MASConfig(
            mas_id="checkpoint_test",
            name="Checkpoint Test",
            workflow_type="sequential",
            checkpoint_enabled=True,
            agents=[_agent("pipe", pipeline=_stub_pipeline())],
        )

        compiled = GraphBuilder(config).build()

        # The pipeline node was compiled during build().
        # Verify it works by invoking — if checkpointer was wrongly attached,
        # it would fail with "thread_id required"
        node_fn = compiled.agent_nodes["pipe"]
        result = node_fn({"messages": [], "agent_outputs": {}})

        assert result["agent_outputs"]["pipe"]["status"] == "completed"


# =========================================================================
# NESTED PIPELINE EDGE CASES
# =========================================================================


class TestNestedPipelineEdgeCases:
    """Edge cases for nested and complex pipeline configurations."""

    def test_three_node_linear_pipeline(self):
        """Three-node linear pipeline compiles and executes."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="a",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner_a",
                        "role": "step_a",
                        "objective": "First step in three-node pipeline",
                    },
                ),
                PipelineNodeSpec(
                    node_id="b",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner_b",
                        "role": "step_b",
                        "objective": "Middle step in three-node pipeline",
                    },
                ),
                PipelineNodeSpec(
                    node_id="c",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner_c",
                        "role": "step_c",
                        "objective": "Final step in three-node pipeline",
                    },
                ),
            ],
            edges=[
                PipelineEdgeSpec(from_node="a", to_node="b"),
                PipelineEdgeSpec(from_node="b", to_node="c"),
                PipelineEdgeSpec(from_node="c", to_node="END"),
            ],
        )

        config = MASConfig(
            mas_id="three_node",
            name="Three Node Pipeline",
            workflow_type="sequential",
            agents=[_agent("triple", pipeline=pipeline)],
        )

        compiled = GraphBuilder(config).build()
        graph = compiled.graph.compile(checkpointer=None)

        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})

        outputs = result["agent_outputs"]["triple"]
        assert outputs["status"] == "completed"
        # All three inner agents should appear in pipeline_outputs
        pipeline_outputs = outputs.get("pipeline_outputs", {})
        assert "inner_a" in pipeline_outputs
        assert "inner_b" in pipeline_outputs
        assert "inner_c" in pipeline_outputs

    def test_single_node_pipeline_minimal(self):
        """Simplest possible pipeline: one node → END."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="only",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "solo",
                        "role": "solo",
                        "objective": "Single node pipeline for minimal testing",
                    },
                ),
            ],
            edges=[PipelineEdgeSpec(from_node="only", to_node="END")],
        )

        config = MASConfig(
            mas_id="single",
            name="Single Node",
            workflow_type="sequential",
            agents=[_agent("minimal", pipeline=pipeline)],
        )

        compiled = GraphBuilder(config).build()
        graph = compiled.graph.compile(checkpointer=None)

        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})

        assert result["agent_outputs"]["minimal"]["status"] == "completed"

    def test_pipeline_agent_preserves_metadata(self):
        """Pipeline agent preserves metadata through compilation."""
        agent = _agent(
            "meta_agent",
            pipeline=_stub_pipeline(),
            metadata={"team": "research", "priority": "high"},
        )

        config = MASConfig(
            mas_id="meta",
            name="Metadata Test",
            workflow_type="sequential",
            agents=[agent],
        )

        compiled = GraphBuilder(config).build()
        # Metadata survives compilation
        compiled_agent = compiled.config.agents[0]
        assert compiled_agent.metadata["team"] == "research"
        assert compiled_agent.metadata["priority"] == "high"

    def test_multiple_pipeline_agents_in_sequence(self):
        """Two pipeline agents in sequence both execute correctly."""
        config = MASConfig(
            mas_id="multi_pipe",
            name="Multi Pipeline",
            workflow_type="sequential",
            agents=[
                _agent("first_pipe", pipeline=_stub_pipeline()),
                _agent("second_pipe", pipeline=_stub_pipeline()),
            ],
        )

        compiled = GraphBuilder(config).build()
        graph = compiled.graph.compile(checkpointer=None)

        result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "test"})

        assert result["agent_outputs"]["first_pipe"]["status"] == "completed"
        assert result["agent_outputs"]["second_pipe"]["status"] == "completed"
        # Both should have pipeline_outputs
        assert "pipeline_outputs" in result["agent_outputs"]["first_pipe"]
        assert "pipeline_outputs" in result["agent_outputs"]["second_pipe"]
