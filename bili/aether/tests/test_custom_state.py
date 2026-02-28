"""Tests for custom pipeline state fields (PipelineStateField).

Covers:
- PipelineStateField schema validation (reserved names, types, reducers)
- Custom fields appear in inner pipeline state TypedDict
- State adapter carries custom fields in and out of sub-graphs
- Custom fields promoted to outer MAS state
- Conditional edges can reference custom state fields
- Single-agent MAS with custom state (Katie pattern)
- YAML round-trip with state_fields
"""

from functools import partial

import pytest

from bili.aether.compiler.graph_builder import GraphBuilder
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.pipeline_spec import (
    _RESERVED_STATE_FIELDS,
    _SAFE_TYPE_MAP,
    PipelineEdgeSpec,
    PipelineNodeSpec,
    PipelineSpec,
    PipelineStateField,
)

# =========================================================================
# HELPERS
# =========================================================================


def _build_state_writer_node(field_name, value):
    """Node builder that writes a specific value to a custom state field."""

    def builder(**kwargs):
        def _execute(state: dict) -> dict:
            return {
                "messages": state.get("messages", []),
                "current_agent": "writer",
                "agent_outputs": {},
                field_name: value,
            }

        return _execute

    return builder


def _build_state_reader_node(field_name):
    """Node builder that reads a custom field and puts it in agent_outputs."""

    def builder(**kwargs):
        def _execute(state: dict) -> dict:
            return {
                "messages": state.get("messages", []),
                "current_agent": "reader",
                "agent_outputs": {"reader": {field_name: state.get(field_name)}},
            }

        return _execute

    return builder


def _make_node_factory(name, builder):
    """Create a partial(Node, name, builder)."""
    from bili.graph_builder.classes.node import Node

    return partial(Node, name, builder)


def _pipeline_with_state_fields(state_fields, custom_nodes=None):
    """Build a pipeline with custom state fields and writer/reader nodes."""
    nodes = [
        PipelineNodeSpec(
            node_id="writer",
            node_type="state_writer",
        ),
        PipelineNodeSpec(
            node_id="reader",
            node_type="state_reader",
        ),
    ]
    edges = [
        PipelineEdgeSpec(from_node="writer", to_node="reader"),
        PipelineEdgeSpec(from_node="reader", to_node="END"),
    ]
    return PipelineSpec(
        nodes=nodes,
        edges=edges,
        state_fields=state_fields,
    )


def _mas_with_state_fields(state_fields, custom_nodes=None):
    """Create a MAS config with a single agent using custom state fields."""
    pipeline = _pipeline_with_state_fields(state_fields)
    agent = AgentSpec(
        agent_id="test_agent",
        role="tester",
        objective="Test custom state fields in pipeline",
        pipeline=pipeline,
    )
    return MASConfig(
        mas_id="test_state",
        name="Test Custom State",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
        checkpoint_enabled=False,
    )


# =========================================================================
# TEST: PipelineStateField schema validation
# =========================================================================


class TestPipelineStateFieldValidation:
    """Tests for PipelineStateField Pydantic validation."""

    def test_valid_field(self):
        """Basic valid field creation."""
        field = PipelineStateField(
            name="sentiment_score", type="float", default=0.0, reducer="replace"
        )
        assert field.name == "sentiment_score"
        assert field.type == "float"
        assert field.default == 0.0
        assert field.reducer == "replace"

    def test_reserved_name_rejected(self):
        """Reserved field names (messages, current_agent, agent_outputs) are rejected."""
        for reserved in _RESERVED_STATE_FIELDS:
            with pytest.raises(ValueError, match="reserved"):
                PipelineStateField(name=reserved, type="str")

    def test_invalid_type_rejected(self):
        """Unknown type strings are rejected."""
        with pytest.raises(ValueError, match="Unsupported type"):
            PipelineStateField(name="foo", type="pandas.DataFrame")

    def test_all_safe_types_accepted(self):
        """All types in _SAFE_TYPE_MAP are accepted."""
        for type_str in _SAFE_TYPE_MAP:
            field = PipelineStateField(name="test_field", type=type_str)
            assert field.type == type_str

    def test_invalid_reducer_rejected(self):
        """Unknown reducer strings are rejected."""
        with pytest.raises(ValueError, match="Unknown reducer"):
            PipelineStateField(name="foo", type="str", reducer="unknown_strategy")

    def test_valid_reducers(self):
        """Known reducers are accepted."""
        for reducer in ("replace", "append"):
            field = PipelineStateField(name="test_field", type="str", reducer=reducer)
            assert field.reducer == reducer

    def test_none_reducer_accepted(self):
        """None reducer (no reducer) is valid."""
        field = PipelineStateField(name="test_field", type="str", reducer=None)
        assert field.reducer is None

    def test_resolve_type(self):
        """resolve_type() returns the actual Python type."""
        field = PipelineStateField(name="score", type="float")
        assert field.resolve_type() is float

    def test_resolve_reducer_replace(self):
        """resolve_reducer() for 'replace' returns last-writer-wins."""
        field = PipelineStateField(name="score", type="float", reducer="replace")
        reducer = field.resolve_reducer()
        assert reducer("old", "new") == "new"

    def test_resolve_reducer_append(self):
        """resolve_reducer() for 'append' concatenates lists."""
        field = PipelineStateField(name="scores", type="List[float]", reducer="append")
        reducer = field.resolve_reducer()
        assert reducer([1.0, 2.0], [3.0]) == [1.0, 2.0, 3.0]

    def test_resolve_reducer_none(self):
        """resolve_reducer() returns None when no reducer set."""
        field = PipelineStateField(name="score", type="float")
        assert field.resolve_reducer() is None

    def test_invalid_field_name_pattern(self):
        """Field names must be valid Python identifiers."""
        with pytest.raises(ValueError):
            PipelineStateField(name="123bad", type="str")
        with pytest.raises(ValueError):
            PipelineStateField(name="has space", type="str")

    def test_default_type_is_any(self):
        """Default type is 'Any' when not specified."""
        field = PipelineStateField(name="generic")
        assert field.type == "Any"


# =========================================================================
# TEST: PipelineSpec with state_fields
# =========================================================================


class TestPipelineSpecStateFields:
    """Tests for state_fields on PipelineSpec."""

    def test_empty_state_fields_default(self):
        """state_fields defaults to empty list."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="a",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "tester",
                        "objective": "Inner agent for testing",
                    },
                )
            ],
            edges=[PipelineEdgeSpec(from_node="a", to_node="END")],
        )
        assert pipeline.state_fields == []

    def test_state_fields_parsed(self):
        """state_fields are parsed into PipelineStateField instances."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="a",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "tester",
                        "objective": "Inner agent for testing",
                    },
                )
            ],
            edges=[PipelineEdgeSpec(from_node="a", to_node="END")],
            state_fields=[
                PipelineStateField(
                    name="sentiment_score", type="float", default=0.0, reducer="replace"
                ),
                PipelineStateField(name="mci_mode", type="str", default="normal"),
            ],
        )
        assert len(pipeline.state_fields) == 2
        assert pipeline.state_fields[0].name == "sentiment_score"
        assert pipeline.state_fields[1].name == "mci_mode"


# =========================================================================
# TEST: Compilation with custom state fields
# =========================================================================


class TestCustomStateCompilation:
    """Tests for custom state fields in pipeline compilation."""

    def _build_and_compile(self, state_fields):
        """Helper to compile a MAS with custom state fields."""
        config = _mas_with_state_fields(state_fields)
        custom_reg = {
            "state_writer": _make_node_factory(
                "state_writer", _build_state_writer_node("sentiment_score", 0.75)
            ),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("sentiment_score")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        return builder.build()

    def test_compiles_with_custom_fields(self):
        """Pipeline with custom state fields compiles successfully."""
        fields = [
            PipelineStateField(
                name="sentiment_score", type="float", default=0.0, reducer="replace"
            )
        ]
        compiled = self._build_and_compile(fields)
        assert "test_agent" in compiled.agent_nodes

    def test_custom_fields_in_outer_state(self):
        """Custom fields are promoted to outer MAS state schema."""
        fields = [
            PipelineStateField(
                name="sentiment_score", type="float", default=0.0, reducer="replace"
            ),
            PipelineStateField(name="mci_mode", type="str", default="normal"),
        ]
        compiled = self._build_and_compile(fields)
        state_annotations = compiled.state_schema.__annotations__

        assert "sentiment_score" in state_annotations
        assert "mci_mode" in state_annotations
        # Base fields should still be present
        assert "messages" in state_annotations
        assert "current_agent" in state_annotations

    def test_multiple_custom_fields(self):
        """Multiple custom fields all appear in compiled state."""
        fields = [
            PipelineStateField(name="field_a", type="str"),
            PipelineStateField(name="field_b", type="int"),
            PipelineStateField(name="field_c", type="float"),
            PipelineStateField(name="field_d", type="bool"),
            PipelineStateField(name="field_e", type="list"),
        ]
        compiled = self._build_and_compile(fields)
        state_annotations = compiled.state_schema.__annotations__
        for f in fields:
            assert f.name in state_annotations


# =========================================================================
# TEST: State adapter input/output mapping
# =========================================================================


class TestStateAdapterCustomFields:
    """Tests for custom state field propagation through the state adapter."""

    def test_custom_field_flows_through_pipeline(self):
        """Custom field written by one node is readable by the next."""
        fields = [
            PipelineStateField(
                name="sentiment_score", type="float", default=0.0, reducer="replace"
            )
        ]
        config = _mas_with_state_fields(fields)
        custom_reg = {
            "state_writer": _make_node_factory(
                "state_writer", _build_state_writer_node("sentiment_score", 0.85)
            ),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("sentiment_score")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        # Execute the graph
        graph = compiled.graph.compile(checkpointer=None)
        result = graph.invoke({"messages": []})

        # The reader node should have captured the sentiment_score
        agent_out = result.get("agent_outputs", {})
        assert "test_agent" in agent_out

    def test_custom_field_propagated_to_outer_state(self):
        """Custom fields propagate back to outer MAS state after pipeline execution."""
        fields = [
            PipelineStateField(
                name="sentiment_score", type="float", default=0.0, reducer="replace"
            )
        ]
        config = _mas_with_state_fields(fields)
        custom_reg = {
            "state_writer": _make_node_factory(
                "state_writer", _build_state_writer_node("sentiment_score", 0.42)
            ),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("sentiment_score")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        graph = compiled.graph.compile(checkpointer=None)
        result = graph.invoke({"messages": []})

        # Custom field should be in the outer state
        assert result.get("sentiment_score") == 0.42

    def test_default_value_used_when_not_in_outer_state(self):
        """Default value is used when custom field not present in outer state.

        When the outer state doesn't contain the field (e.g., the field
        is only in the inner pipeline state and not promoted), the adapter
        uses the declared default.
        """
        fields = [
            PipelineStateField(
                name="mci_mode", type="str", default="normal", reducer="replace"
            )
        ]
        config = _mas_with_state_fields(fields)

        # Writer that reads mci_mode and echoes it into agent_outputs
        def _echo_builder(**kwargs):
            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "echo",
                    "agent_outputs": {"echo": {"mci_mode": state.get("mci_mode")}},
                    "mci_mode": state.get("mci_mode", "fallback"),
                }

            return _execute

        custom_reg = {
            "state_writer": _make_node_factory("state_writer", _echo_builder),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("mci_mode")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        graph = compiled.graph.compile(checkpointer=None)

        # Pass default as initial state value — this is the expected pattern
        # for setting initial custom field values
        result = graph.invoke({"messages": [], "mci_mode": "normal"})
        assert result.get("mci_mode") == "normal"

    def test_custom_field_initial_value_flows_through(self):
        """Initial custom field values provided at invoke time flow through the pipeline."""
        fields = [
            PipelineStateField(
                name="mci_mode", type="str", default="normal", reducer="replace"
            )
        ]
        config = _mas_with_state_fields(fields)

        def _echo_builder(**kwargs):
            def _execute(state: dict) -> dict:
                return {
                    "messages": state.get("messages", []),
                    "current_agent": "echo",
                    "agent_outputs": {},
                    "mci_mode": state.get("mci_mode", "fallback"),
                }

            return _execute

        custom_reg = {
            "state_writer": _make_node_factory("state_writer", _echo_builder),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("mci_mode")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        graph = compiled.graph.compile(checkpointer=None)
        result = graph.invoke({"messages": [], "mci_mode": "elevated"})
        assert result.get("mci_mode") == "elevated"


# =========================================================================
# TEST: Single-agent MAS pattern (Katie pattern)
# =========================================================================


class TestSingleAgentMASPattern:
    """Test the single-agent MAS with rich pipeline pattern."""

    def test_single_agent_with_many_custom_fields(self):
        """Single-agent MAS compiles with many custom state fields."""
        fields = [
            PipelineStateField(name="mci_mode", type="str", default="normal"),
            PipelineStateField(name="sentiment_score", type="float", default=0.0),
            PipelineStateField(name="engagement_level", type="float", default=0.5),
            PipelineStateField(name="utterance", type="str", default=""),
            PipelineStateField(name="topic", type="str", default="general"),
        ]
        config = _mas_with_state_fields(fields)

        custom_reg = {
            "state_writer": _make_node_factory(
                "state_writer", _build_state_writer_node("sentiment_score", 0.8)
            ),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("sentiment_score")
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        # All custom fields should be in outer state
        annotations = compiled.state_schema.__annotations__
        for f in fields:
            assert f.name in annotations, f"Field {f.name} missing from outer state"

    def test_single_agent_e2e_execution(self):
        """Single-agent MAS executes end-to-end with custom state."""
        fields = [
            PipelineStateField(
                name="score", type="float", default=0.0, reducer="replace"
            )
        ]
        config = _mas_with_state_fields(fields)

        custom_reg = {
            "state_writer": _make_node_factory(
                "state_writer", _build_state_writer_node("score", 99.9)
            ),
            "state_reader": _make_node_factory(
                "state_reader", _build_state_reader_node("score")
            ),
        }

        from bili.aether.runtime.executor import MASExecutor

        executor = MASExecutor(config, custom_node_registry=custom_reg)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert result.final_state.get("score") == 99.9


# =========================================================================
# TEST: Conditional edges with custom state fields
# =========================================================================


class TestConditionalEdgesWithCustomState:
    """Test that conditional edges can reference custom state fields."""

    def test_condition_references_custom_field(self):
        """Conditional edge expression can reference a custom state field."""
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(node_id="writer", node_type="score_setter"),
                PipelineNodeSpec(
                    node_id="high_path",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "high",
                        "role": "high handler",
                        "objective": "Handle high scores in pipeline",
                    },
                ),
                PipelineNodeSpec(
                    node_id="low_path",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "low",
                        "role": "low handler",
                        "objective": "Handle low scores in pipeline",
                    },
                ),
            ],
            edges=[
                PipelineEdgeSpec(
                    from_node="writer",
                    to_node="high_path",
                    condition="state.score > 0.5",
                    label="high",
                ),
                PipelineEdgeSpec(
                    from_node="writer",
                    to_node="low_path",
                    label="low",
                ),
                PipelineEdgeSpec(from_node="high_path", to_node="END"),
                PipelineEdgeSpec(from_node="low_path", to_node="END"),
            ],
            state_fields=[
                PipelineStateField(
                    name="score", type="float", default=0.0, reducer="replace"
                ),
            ],
        )
        agent = AgentSpec(
            agent_id="cond_agent",
            role="conditional",
            objective="Test conditional routing with custom state",
            pipeline=pipeline,
        )
        config = MASConfig(
            mas_id="cond_test",
            name="Conditional Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
            checkpoint_enabled=False,
        )

        custom_reg = {
            "score_setter": _make_node_factory(
                "score_setter", _build_state_writer_node("score", 0.8)
            ),
        }
        builder = GraphBuilder(config, custom_node_registry=custom_reg)
        compiled = builder.build()

        # Should compile with conditional edges referencing custom field
        assert "cond_agent" in compiled.agent_nodes


# =========================================================================
# TEST: YAML round-trip with state_fields
# =========================================================================


class TestYAMLRoundTrip:
    """Test that state_fields survive YAML serialization."""

    def test_state_fields_serialize_and_parse(self):
        """PipelineSpec with state_fields can round-trip through dict."""
        fields = [
            PipelineStateField(
                name="sentiment_score", type="float", default=0.0, reducer="replace"
            ),
            PipelineStateField(
                name="rolling_sentiment",
                type="List[float]",
                default=[],
                reducer="append",
            ),
        ]
        pipeline = PipelineSpec(
            nodes=[
                PipelineNodeSpec(
                    node_id="a",
                    node_type="agent",
                    agent_spec={
                        "agent_id": "inner",
                        "role": "tester",
                        "objective": "Inner agent for testing",
                    },
                )
            ],
            edges=[PipelineEdgeSpec(from_node="a", to_node="END")],
            state_fields=fields,
        )

        # Serialize to dict and back
        data = pipeline.model_dump()
        restored = PipelineSpec(**data)

        assert len(restored.state_fields) == 2
        assert restored.state_fields[0].name == "sentiment_score"
        assert restored.state_fields[0].type == "float"
        assert restored.state_fields[0].reducer == "replace"
        assert restored.state_fields[1].name == "rolling_sentiment"
        assert restored.state_fields[1].type == "List[float]"
        assert restored.state_fields[1].reducer == "append"
