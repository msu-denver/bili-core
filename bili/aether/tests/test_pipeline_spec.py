"""Tests for PipelineSpec schema and AgentSpec pipeline integration.

Covers:
- PipelineNodeSpec validation (node_type, agent_spec requirements)
- PipelineEdgeSpec validation (references, END terminal)
- PipelineSpec validation (node uniqueness, edge validity, path to END, entry point)
- AgentSpec pipeline field (backwards compat, model_name coexistence, depth limits)
- YAML round-trip serialization
- Complex pipeline configurations
"""

import warnings

import pytest
from pydantic import ValidationError

from bili.aether.schema import AgentSpec, PipelineSpec
from bili.aether.schema.pipeline_spec import (
    MAX_PIPELINE_DEPTH,
    PipelineEdgeSpec,
    PipelineNodeSpec,
)

# =========================================================================
# HELPERS
# =========================================================================


def _simple_pipeline():
    """Create a minimal valid pipeline for reuse in tests."""
    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(node_id="persona", node_type="add_persona_and_summary"),
            PipelineNodeSpec(node_id="react", node_type="react_agent"),
        ],
        edges=[
            PipelineEdgeSpec(from_node="persona", to_node="react"),
            PipelineEdgeSpec(from_node="react", to_node="END"),
        ],
    )


def _simple_agent(**overrides):
    """Create a minimal valid AgentSpec for reuse in tests."""
    defaults = {
        "agent_id": "test_agent",
        "role": "tester",
        "objective": "Test pipeline functionality in agent spec",
    }
    defaults.update(overrides)
    return AgentSpec(**defaults)


# =========================================================================
# PipelineNodeSpec TESTS
# =========================================================================


class TestPipelineNodeSpec:
    """Tests for individual pipeline node specifications."""

    def test_minimal_registry_node(self):
        """Registry node needs only node_id and node_type."""
        node = PipelineNodeSpec(node_id="persona", node_type="add_persona_and_summary")
        assert node.node_id == "persona"
        assert node.node_type == "add_persona_and_summary"
        assert node.agent_spec is None
        assert node.config == {}

    def test_registry_node_with_config(self):
        """Registry node can carry configuration kwargs."""
        node = PipelineNodeSpec(
            node_id="react",
            node_type="react_agent",
            config={"temperature": 0.7, "max_tokens": 1024},
        )
        assert node.config["temperature"] == 0.7
        assert node.config["max_tokens"] == 1024

    def test_agent_node_with_spec(self):
        """Agent node type requires an inline agent_spec dict."""
        node = PipelineNodeSpec(
            node_id="inner_agent",
            node_type="agent",
            agent_spec={
                "agent_id": "inner",
                "role": "analyzer",
                "objective": "Analyze data from the pipeline context",
                "model_name": "gpt-4",
            },
        )
        assert node.agent_spec is not None
        assert node.agent_spec["model_name"] == "gpt-4"

    def test_agent_node_without_spec_fails(self):
        """Agent node type without agent_spec should fail validation."""
        with pytest.raises(ValidationError, match="agent_spec"):
            PipelineNodeSpec(node_id="bad_agent", node_type="agent")

    def test_non_agent_with_spec_warns(self):
        """Non-agent node with agent_spec should issue a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PipelineNodeSpec(
                node_id="react_with_spec",
                node_type="react_agent",
                agent_spec={"agent_id": "x", "role": "y", "objective": "z" * 10},
            )
            spec_warnings = [
                w for w in caught if "agent_spec" in str(w.message).lower()
            ]
            assert (
                len(spec_warnings) == 1
            ), "Expected warning for agent_spec on non-agent node"

    def test_invalid_node_id_pattern(self):
        """Node IDs must match alphanumeric + underscore/dash pattern."""
        with pytest.raises(ValidationError):
            PipelineNodeSpec(node_id="invalid id!", node_type="react_agent")

    def test_empty_node_id_fails(self):
        """Empty node_id should fail validation."""
        with pytest.raises(ValidationError):
            PipelineNodeSpec(node_id="", node_type="react_agent")

    def test_empty_node_type_fails(self):
        """Empty node_type should fail validation."""
        with pytest.raises(ValidationError):
            PipelineNodeSpec(node_id="test", node_type="")


# =========================================================================
# PipelineEdgeSpec TESTS
# =========================================================================


class TestPipelineEdgeSpec:
    """Tests for pipeline edge specifications."""

    def test_simple_edge(self):
        """Basic edge between two nodes."""
        edge = PipelineEdgeSpec(from_node="a", to_node="b")
        assert edge.from_node == "a"
        assert edge.to_node == "b"
        assert edge.condition is None
        assert edge.label == ""

    def test_terminal_edge(self):
        """Edge to END is valid."""
        edge = PipelineEdgeSpec(from_node="final", to_node="END")
        assert edge.to_node == "END"

    def test_conditional_edge(self):
        """Edge with condition expression."""
        edge = PipelineEdgeSpec(
            from_node="decision",
            to_node="escalation",
            condition="state.score < 0.5",
            label="low confidence",
        )
        assert edge.condition == "state.score < 0.5"
        assert edge.label == "low confidence"

    def test_invalid_condition_syntax_fails(self):
        """Condition with invalid Python syntax should fail validation."""
        with pytest.raises(ValidationError, match="Invalid condition"):
            PipelineEdgeSpec(
                from_node="a",
                to_node="b",
                condition="import os",
            )

    def test_valid_condition_expressions_pass(self):
        """Various valid condition expressions pass validation."""
        valid_conditions = [
            "state.score > 0.5",
            "state.count == 3",
            "state.flag == True",
            "state.value >= 10 and state.ready == True",
        ]
        for cond in valid_conditions:
            edge = PipelineEdgeSpec(from_node="a", to_node="b", condition=cond)
            assert edge.condition == cond


# =========================================================================
# PipelineSpec TESTS
# =========================================================================


class TestPipelineSpec:
    """Tests for complete pipeline specifications."""

    def test_minimal_pipeline(self):
        """Simplest valid pipeline: one node with edge to END."""
        pipeline = PipelineSpec(
            nodes=[PipelineNodeSpec(node_id="react", node_type="react_agent")],
            edges=[PipelineEdgeSpec(from_node="react", to_node="END")],
        )
        assert len(pipeline.nodes) == 1
        assert pipeline.get_entry_node() == "react"

    def test_multi_node_pipeline(self):
        """Pipeline with multiple nodes in sequence."""
        pipeline = _simple_pipeline()
        assert len(pipeline.nodes) == 2
        assert len(pipeline.edges) == 2
        assert pipeline.get_entry_node() == "persona"

    def test_complex_pipeline(self):
        """Pipeline with branching and conditional edges."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="persona", node_type="add_persona_and_summary"
                ),
                PipelineNodeSpec(
                    node_id="datetime", node_type="inject_current_datetime"
                ),
                PipelineNodeSpec(node_id="react", node_type="react_agent"),
                PipelineNodeSpec(node_id="sentiment", node_type="sentiment_analysis"),
                PipelineNodeSpec(node_id="escalation", node_type="react_agent"),
            ],
            edges=[
                PipelineEdgeSpec(from_node="persona", to_node="datetime"),
                PipelineEdgeSpec(from_node="datetime", to_node="react"),
                PipelineEdgeSpec(from_node="react", to_node="sentiment"),
                PipelineEdgeSpec(
                    from_node="sentiment",
                    to_node="escalation",
                    condition="state.sentiment_score < -0.5",
                ),
                PipelineEdgeSpec(from_node="sentiment", to_node="END"),
                PipelineEdgeSpec(from_node="escalation", to_node="END"),
            ],
            entry_point="persona",
        )
        assert len(pipeline.nodes) == 5
        assert pipeline.get_entry_node() == "persona"

    def test_explicit_entry_point(self):
        """Pipeline with explicit entry_point (not first node)."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(node_id="a", node_type="react_agent"),
                PipelineNodeSpec(node_id="b", node_type="react_agent"),
            ],
            edges=[
                PipelineEdgeSpec(from_node="b", to_node="a"),
                PipelineEdgeSpec(from_node="a", to_node="END"),
            ],
            entry_point="b",
        )
        assert pipeline.get_entry_node() == "b"

    def test_default_entry_point_is_first_node(self):
        """Entry point defaults to first node when not specified."""
        pipeline = _simple_pipeline()
        assert pipeline.entry_point is None
        assert pipeline.get_entry_node() == "persona"

    # --- Validation failures ---

    def test_duplicate_node_ids_fail(self):
        """Duplicate node IDs within pipeline should fail."""
        with pytest.raises(ValidationError, match="Duplicate node IDs"):
            PipelineSpec(
                nodes=[
                    PipelineNodeSpec(node_id="dup", node_type="react_agent"),
                    PipelineNodeSpec(node_id="dup", node_type="react_agent"),
                ],
                edges=[PipelineEdgeSpec(from_node="dup", to_node="END")],
            )

    def test_invalid_from_node_reference_fails(self):
        """Edge referencing nonexistent from_node should fail."""
        with pytest.raises(ValidationError, match="from_node.*not found"):
            PipelineSpec(
                nodes=[PipelineNodeSpec(node_id="a", node_type="react_agent")],
                edges=[PipelineEdgeSpec(from_node="nonexistent", to_node="END")],
            )

    def test_invalid_to_node_reference_fails(self):
        """Edge referencing nonexistent to_node should fail."""
        with pytest.raises(ValidationError, match="to_node.*not found"):
            PipelineSpec(
                nodes=[PipelineNodeSpec(node_id="a", node_type="react_agent")],
                edges=[PipelineEdgeSpec(from_node="a", to_node="nonexistent")],
            )

    def test_no_path_to_end_fails(self):
        """Pipeline without any edge to END should fail."""
        with pytest.raises(ValidationError, match="END"):
            PipelineSpec(
                nodes=[
                    PipelineNodeSpec(node_id="a", node_type="react_agent"),
                    PipelineNodeSpec(node_id="b", node_type="react_agent"),
                ],
                edges=[PipelineEdgeSpec(from_node="a", to_node="b")],
            )

    def test_invalid_entry_point_fails(self):
        """Entry point referencing nonexistent node should fail."""
        with pytest.raises(ValidationError, match="entry_point.*not found"):
            PipelineSpec(
                nodes=[PipelineNodeSpec(node_id="a", node_type="react_agent")],
                edges=[PipelineEdgeSpec(from_node="a", to_node="END")],
                entry_point="nonexistent",
            )

    def test_empty_nodes_fail(self):
        """Pipeline with no nodes should fail."""
        with pytest.raises(ValidationError):
            PipelineSpec(
                nodes=[],
                edges=[PipelineEdgeSpec(from_node="a", to_node="END")],
            )

    def test_empty_edges_fail(self):
        """Pipeline with no edges should fail."""
        with pytest.raises(ValidationError):
            PipelineSpec(
                nodes=[PipelineNodeSpec(node_id="a", node_type="react_agent")],
                edges=[],
            )


# =========================================================================
# AgentSpec PIPELINE INTEGRATION TESTS
# =========================================================================


class TestAgentSpecPipelineIntegration:
    """Tests for pipeline field on AgentSpec."""

    def test_agent_without_pipeline_backwards_compat(self):
        """AgentSpec without pipeline works exactly as before."""
        agent = _simple_agent()
        assert agent.pipeline is None

    def test_agent_with_pipeline(self):
        """AgentSpec with pipeline field set."""
        agent = _simple_agent(pipeline=_simple_pipeline())
        assert agent.pipeline is not None
        assert len(agent.pipeline.nodes) == 2

    def test_model_name_pipeline_coexistence_warns(self):
        """Setting both model_name and pipeline should issue a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simple_agent(model_name="gpt-4", pipeline=_simple_pipeline())
            relevant = [
                w
                for w in caught
                if "pipeline" in str(w.message) and "model_name" in str(w.message)
            ]
            assert (
                len(relevant) == 1
            ), "Expected warning for model_name+pipeline coexistence"

    def test_pipeline_without_model_name_no_warning(self):
        """Pipeline without model_name should not warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simple_agent(pipeline=_simple_pipeline())
            relevant = [
                w
                for w in caught
                if "pipeline" in str(w.message) and "model_name" in str(w.message)
            ]
            assert len(relevant) == 0, "No warning expected when model_name is not set"

    def test_pipeline_depth_limit(self):
        """Nested pipelines beyond MAX_PIPELINE_DEPTH should fail."""
        # Build a nested pipeline that exceeds the depth limit
        # depth 3 (inner-inner-inner) should fail when MAX_PIPELINE_DEPTH = 3
        innermost_pipeline = {
            "nodes": [{"node_id": "r", "node_type": "react_agent"}],
            "edges": [{"from_node": "r", "to_node": "END"}],
        }

        def _nest(pipeline_dict, remaining_depth):
            """Wrap a pipeline in another pipeline layer."""
            return {
                "nodes": [
                    {
                        "node_id": f"agent_d{remaining_depth}",
                        "node_type": "agent",
                        "agent_spec": {
                            "agent_id": f"nested_{remaining_depth}",
                            "role": "nested",
                            "objective": "Test nested pipeline depth limits",
                            "pipeline": pipeline_dict,
                        },
                    }
                ],
                "edges": [{"from_node": f"agent_d{remaining_depth}", "to_node": "END"}],
            }

        # Build nesting that exceeds the limit
        pipeline_dict = innermost_pipeline
        for depth in range(MAX_PIPELINE_DEPTH + 1, 0, -1):
            pipeline_dict = _nest(pipeline_dict, depth)

        with pytest.raises(ValidationError, match="nesting depth"):
            _simple_agent(pipeline=PipelineSpec(**pipeline_dict))

    def test_pipeline_depth_within_limit(self):
        """Nested pipeline within MAX_PIPELINE_DEPTH should succeed."""
        inner_pipeline = {
            "nodes": [{"node_id": "r", "node_type": "react_agent"}],
            "edges": [{"from_node": "r", "to_node": "END"}],
        }
        outer_pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="inner",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "nested_1",
                        "role": "nested",
                        "objective": "Test single level nesting works",
                        "pipeline": inner_pipeline,
                    },
                )
            ],
            edges=[PipelineEdgeSpec(from_node="inner", to_node="END")],
        )
        # Depth = 2 (outer -> inner), should be within limit
        agent = _simple_agent(pipeline=outer_pipeline)
        assert agent.pipeline is not None

    def test_pipeline_preserves_other_agent_fields(self):
        """Pipeline field doesn't interfere with other AgentSpec fields."""
        agent = _simple_agent(
            pipeline=_simple_pipeline(),
            capabilities=["analysis", "search"],
            tools=["tavily_search"],
            temperature=0.5,
            tier=2,
            voting_weight=1.5,
            metadata={"team": "research"},
        )
        assert agent.pipeline is not None
        assert agent.capabilities == ["analysis", "search"]
        assert agent.tools == ["tavily_search"]
        assert agent.temperature == 0.5
        assert agent.tier == 2
        assert agent.voting_weight == 1.5
        assert agent.metadata["team"] == "research"


# =========================================================================
# SERIALIZATION TESTS
# =========================================================================


class TestPipelineSerialization:
    """Tests for pipeline dict/JSON round-trip serialization."""

    def test_pipeline_to_dict(self):
        """Pipeline can be serialized to dict."""
        pipeline = _simple_pipeline()
        data = pipeline.model_dump()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 2
        assert data["entry_point"] is None

    def test_pipeline_from_dict(self):
        """Pipeline can be deserialized from dict."""
        data = {
            "nodes": [
                {"node_id": "a", "node_type": "react_agent"},
                {"node_id": "b", "node_type": "react_agent"},
            ],
            "edges": [
                {"from_node": "a", "to_node": "b"},
                {"from_node": "b", "to_node": "END"},
            ],
        }
        pipeline = PipelineSpec(**data)
        assert len(pipeline.nodes) == 2
        assert pipeline.get_entry_node() == "a"

    def test_agent_with_pipeline_round_trip(self):
        """AgentSpec with pipeline survives dict round-trip."""
        agent = _simple_agent(pipeline=_simple_pipeline())
        data = agent.model_dump()
        restored = AgentSpec(**data)
        assert restored.pipeline is not None
        assert len(restored.pipeline.nodes) == len(agent.pipeline.nodes)
        assert restored.pipeline.get_entry_node() == agent.pipeline.get_entry_node()

    def test_agent_without_pipeline_round_trip(self):
        """AgentSpec without pipeline survives dict round-trip."""
        agent = _simple_agent()
        data = agent.model_dump()
        restored = AgentSpec(**data)
        assert restored.pipeline is None
        assert restored.agent_id == agent.agent_id


# =========================================================================
# MASConfig INTEGRATION TESTS
# =========================================================================


class TestPipelineInMASConfig:
    """Tests for pipeline agents within MASConfig."""

    def test_mas_with_pipeline_agents(self):
        """MASConfig can contain agents with pipelines."""
        from bili.aether.schema import MASConfig

        config = MASConfig(
            mas_id="pipeline_test",
            name="Pipeline Test MAS",
            workflow_type="sequential",
            agents=[
                AgentSpec(
                    agent_id="rich_agent",
                    role="researcher",
                    objective="Research with full pipeline depth",
                    pipeline=_simple_pipeline(),
                ),
                AgentSpec(
                    agent_id="simple_agent",
                    role="summarizer",
                    objective="Summarize the research output",
                    model_name="gpt-4",
                ),
            ],
        )
        assert config.agents[0].pipeline is not None
        assert config.agents[1].pipeline is None

    def test_mas_mixed_agents_validate(self):
        """MASConfig validation works with mixed pipeline/non-pipeline agents."""
        from bili.aether.schema import MASConfig, WorkflowEdge

        config = MASConfig(
            mas_id="mixed_test",
            name="Mixed Pipeline MAS",
            workflow_type="custom",
            agents=[
                AgentSpec(
                    agent_id="pipeline_agent",
                    role="processor",
                    objective="Process data with internal pipeline",
                    pipeline=_simple_pipeline(),
                ),
                AgentSpec(
                    agent_id="reviewer",
                    role="reviewer",
                    objective="Review the processed output data",
                    model_name="gpt-4",
                ),
            ],
            workflow_edges=[
                WorkflowEdge(from_agent="pipeline_agent", to_agent="reviewer"),
                WorkflowEdge(from_agent="reviewer", to_agent="END"),
            ],
        )
        assert len(config.agents) == 2
        assert config.agents[0].pipeline is not None
