"""Tests for the static MAS validation engine."""

import os

from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.schema import (
    AgentSpec,
    Channel,
    CommunicationProtocol,
    MASConfig,
    OutputFormat,
    WorkflowEdge,
    WorkflowType,
)
from bili.aether.validation import MASValidator, ValidationResult, validate_mas

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "examples",
)


# =========================================================================
# Helper — build a minimal valid MASConfig
# =========================================================================


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    """Shortcut to build an AgentSpec with sensible defaults."""
    defaults = {
        "role": "judge",
        "objective": f"Test objective for {agent_id}",
    }
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


# =========================================================================
# VALIDATION RESULT TESTS
# =========================================================================


def test_validation_result_defaults():
    """Fresh ValidationResult is valid with no errors or warnings."""
    result = ValidationResult()

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert bool(result) is True


def test_validation_result_add_error():
    """Adding an error marks the result invalid."""
    result = ValidationResult()
    result.add_error("something broke")

    assert result.valid is False
    assert len(result.errors) == 1
    assert result.errors[0] == "something broke"
    assert bool(result) is False


def test_validation_result_merge():
    """Merging combines errors and warnings from both results."""
    r1 = ValidationResult()
    r1.add_error("err1")
    r1.add_warning("warn1")

    r2 = ValidationResult()
    r2.add_warning("warn2")

    r1.merge(r2)

    assert r1.valid is False
    assert len(r1.errors) == 1
    assert len(r1.warnings) == 2


def test_validation_result_str_formatting():
    """__str__ includes errors and warnings with numbering."""
    result = ValidationResult()
    result.add_error("Error one")
    result.add_warning("Warning one")

    text = str(result)

    assert "FAILED" in text
    assert "1. Error one" in text
    assert "1. Warning one" in text

    # Clean result
    clean = ValidationResult()
    assert "passed" in str(clean).lower()


# =========================================================================
# AGENT VALIDATION TESTS
# =========================================================================


def test_orphaned_agent_warning():
    """W1: Agent with no channel connections triggers a warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("connected"), _agent("orphaned")],
        channels=[
            Channel(
                channel_id="ch1",
                protocol=CommunicationProtocol.DIRECT,
                source="connected",
                target="connected",
            )
        ],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("orphaned" in w for w in result.warnings)


def test_supervisor_missing_capability_warning():
    """W2: Supervisor without inter_agent_communication triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        agents=[_agent("sup", is_supervisor=True, capabilities=[])],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("inter_agent_communication" in w for w in result.warnings)


def test_supervisor_with_capability_no_warning():
    """W2 inverse: Supervisor with the capability has no such warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        agents=[
            _agent(
                "sup",
                is_supervisor=True,
                capabilities=["inter_agent_communication"],
            )
        ],
    )

    result = validate_mas(config)

    assert not any("inter_agent_communication" in w for w in result.warnings)


# =========================================================================
# CHANNEL VALIDATION TESTS
# =========================================================================


def test_duplicate_channels_error():
    """E1: Duplicate channels (same source+target+protocol) are an error."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
        channels=[
            Channel(
                channel_id="ch1",
                protocol=CommunicationProtocol.DIRECT,
                source="a",
                target="b",
            ),
            Channel(
                channel_id="ch2",
                protocol=CommunicationProtocol.DIRECT,
                source="a",
                target="b",
            ),
        ],
    )

    result = validate_mas(config)

    assert result.valid is False
    assert any("Duplicate channel" in e for e in result.errors)


def test_bidirectional_with_reverse_warning():
    """W3: Bidirectional channel + separate reverse channel is warned."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
        channels=[
            Channel(
                channel_id="forward",
                protocol=CommunicationProtocol.DIRECT,
                source="a",
                target="b",
                bidirectional=True,
            ),
            Channel(
                channel_id="reverse",
                protocol=CommunicationProtocol.DIRECT,
                source="b",
                target="a",
            ),
        ],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("Bidirectional" in w and "reverse" in w for w in result.warnings)


# =========================================================================
# WORKFLOW GRAPH TESTS
# =========================================================================


def test_sequential_circular_dependency_error():
    """E2: Cycle in SEQUENTIAL workflow triggers an error."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="a"),
        ],
    )

    result = validate_mas(config)

    assert result.valid is False
    assert any("Circular" in e or "circular" in e for e in result.errors)


def test_sequential_non_linear_warning():
    """W4: SEQUENTIAL agent with >1 outgoing edge triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="a", to_agent="c"),
        ],
    )

    result = validate_mas(config)

    assert any("outgoing edges" in w for w in result.warnings)


def test_custom_unreachable_agent_warning():
    """W5: Agent unreachable from entry point in CUSTOM workflow."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        entry_point="a",
        agents=[_agent("a"), _agent("b"), _agent("island")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="END"),
        ],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("island" in w and "unreachable" in w for w in result.warnings)


def test_custom_no_path_to_end_warning():
    """W6: CUSTOM workflow with no edge to END triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
        ],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("END" in w for w in result.warnings)


# =========================================================================
# WORKFLOW-SPECIFIC TESTS
# =========================================================================


def test_consensus_missing_vote_field_warning():
    """W7: CONSENSUS agent without consensus_vote_field triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
        agents=[_agent("voter", output_format=OutputFormat.JSON)],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("consensus_vote_field" in w for w in result.warnings)


def test_hierarchical_tier_gap_warning():
    """W8: Gap in hierarchical tiers triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.HIERARCHICAL,
        agents=[_agent("t1", tier=1), _agent("t3", tier=3)],
    )

    result = validate_mas(config)

    assert any(
        "tier gap" in w.lower() or "missing tier" in w.lower() for w in result.warnings
    )


def test_hierarchical_no_tier_1_error():
    """E3: HIERARCHICAL workflow with no tier 1 agent is an error."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.HIERARCHICAL,
        agents=[_agent("t2", tier=2), _agent("t3", tier=3)],
    )

    result = validate_mas(config)

    assert result.valid is False
    assert any("tier 1" in e for e in result.errors)


def test_supervisor_entry_not_supervisor_warning():
    """W9: SUPERVISOR entry point not marked is_supervisor triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="leader",
        agents=[_agent("leader", is_supervisor=False), _agent("worker")],
    )

    result = validate_mas(config)

    assert any("is_supervisor" in w for w in result.warnings)


def test_human_in_loop_missing_condition_warning():
    """W10: human_in_loop=true without escalation condition triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a")],
        human_in_loop=True,
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("human_escalation_condition" in w for w in result.warnings)


# =========================================================================
# INTEGRATION TESTS
# =========================================================================


def test_validate_mas_convenience_function():
    """validate_mas() produces same result as MASValidator().validate()."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
    )

    r1 = MASValidator().validate(config)
    r2 = validate_mas(config)

    assert r1.valid == r2.valid
    assert len(r1.errors) == len(r2.errors)
    assert len(r1.warnings) == len(r2.warnings)


def test_all_example_yamls_pass_validation():
    """All 6 example YAML configs pass validation with no errors."""
    example_files = [
        "simple_chain.yaml",
        "hierarchical_voting.yaml",
        "supervisor_moderation.yaml",
        "consensus_network.yaml",
        "custom_escalation.yaml",
        "pipeline_agents.yaml",
    ]

    for fname in example_files:
        fpath = os.path.join(_EXAMPLES_DIR, fname)
        config = load_mas_from_yaml(fpath)
        result = validate_mas(config)

        assert result.valid is True, f"{fname} failed validation: {result.errors}"


# =========================================================================
# PIPELINE VALIDATION TESTS
# =========================================================================


def _simple_pipeline():
    """Create a minimal valid pipeline for testing."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    return PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="step_a",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_a",
                    "role": "step_a",
                    "objective": "First step in pipeline processing",
                },
            ),
            PipelineNodeSpec(
                node_id="step_b",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_b",
                    "role": "step_b",
                    "objective": "Second step in pipeline processing",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="step_a", to_node="step_b"),
            PipelineEdgeSpec(from_node="step_b", to_node="END"),
        ],
    )


def test_pipeline_valid_no_errors():
    """A well-formed pipeline agent produces no pipeline errors."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a", pipeline=_simple_pipeline())],
    )

    result = validate_mas(config)

    assert result.valid is True
    # Should not have pipeline-specific errors
    assert not any("pipeline" in e.lower() for e in result.errors)


def test_pipeline_cycle_detection_error():
    """E4: Cycle within pipeline edges triggers an error."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="a",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_a",
                    "role": "looper",
                    "objective": "First node in a cycle for testing",
                },
            ),
            PipelineNodeSpec(
                node_id="b",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_b",
                    "role": "looper",
                    "objective": "Second node in a cycle for testing",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="a", to_node="b"),
            PipelineEdgeSpec(from_node="b", to_node="a"),
            # Still has END edge to pass Pydantic validation
            PipelineEdgeSpec(from_node="b", to_node="END"),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("cyclic_agent", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert result.valid is False
    assert any(
        "circular" in e.lower() and "pipeline" in e.lower() for e in result.errors
    )


def test_pipeline_no_cycle_passes():
    """E4 inverse: Linear pipeline has no cycle error."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("linear_agent", pipeline=_simple_pipeline())],
    )

    result = validate_mas(config)

    assert not any("circular" in e.lower() for e in result.errors)


def test_pipeline_unreachable_node_warning():
    """W11: Unreachable pipeline node triggers a warning."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="main",
                node_type="agent",
                agent_spec={
                    "agent_id": "main_inner",
                    "role": "main",
                    "objective": "Main pipeline node that goes to END",
                },
            ),
            PipelineNodeSpec(
                node_id="island",
                node_type="agent",
                agent_spec={
                    "agent_id": "island_inner",
                    "role": "island",
                    "objective": "Unreachable pipeline node for testing",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="main", to_node="END"),
            # island → END exists but island is never reached from entry
            PipelineEdgeSpec(from_node="island", to_node="END"),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("agent_with_island", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("island" in w and "unreachable" in w.lower() for w in result.warnings)


def test_pipeline_all_nodes_reachable_no_warning():
    """W11 inverse: All-reachable pipeline has no unreachable warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("reachable_agent", pipeline=_simple_pipeline())],
    )

    result = validate_mas(config)

    assert not any(
        "unreachable" in w.lower() and "pipeline" in w.lower() for w in result.warnings
    )


def test_pipeline_all_stubs_warning():
    """W12: Pipeline with only stub agents (no model_name) triggers warning."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("stubs_only", pipeline=_simple_pipeline())],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any("stub" in w.lower() for w in result.warnings)


def test_pipeline_with_model_no_stubs_warning():
    """W12 inverse: Pipeline agent with model_name on inner node has no stub warning."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="llm_node",
                node_type="agent",
                agent_spec={
                    "agent_id": "inner_llm",
                    "role": "processor",
                    "objective": "Process data with an actual LLM call",
                    "model_name": "gpt-4",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="llm_node", to_node="END"),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("has_model", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert not any("stub" in w.lower() for w in result.warnings)


def test_pipeline_registry_nodes_no_stub_warning():
    """W12: Pipeline with only registry nodes (no agent type) skips stub check."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="persona",
                node_type="add_persona_and_summary",
            ),
            PipelineNodeSpec(
                node_id="react",
                node_type="react_agent",
            ),
        ],
        edges=[
            PipelineEdgeSpec(from_node="persona", to_node="react"),
            PipelineEdgeSpec(from_node="react", to_node="END"),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("registry_only", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert not any("stub" in w.lower() for w in result.warnings)


def test_pipeline_conditional_without_fallback_warning():
    """W13: Conditional edges without unconditional fallback triggers warning."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="decision",
                node_type="agent",
                agent_spec={
                    "agent_id": "decider",
                    "role": "decider",
                    "objective": "Make a routing decision based on context",
                },
            ),
            PipelineNodeSpec(
                node_id="path_a",
                node_type="agent",
                agent_spec={
                    "agent_id": "handler_a",
                    "role": "handler",
                    "objective": "Handle the positive case for testing",
                },
            ),
        ],
        edges=[
            # Only conditional, no unconditional fallback from 'decision'
            PipelineEdgeSpec(
                from_node="decision",
                to_node="path_a",
                condition="state.score > 0.5",
            ),
            PipelineEdgeSpec(from_node="path_a", to_node="END"),
            # decision also needs an END path to pass Pydantic
            PipelineEdgeSpec(
                from_node="decision",
                to_node="END",
                condition="state.score <= 0.5",
            ),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("conditional_agent", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert result.valid is True
    assert any(
        "conditional" in w.lower() and "fallback" in w.lower() for w in result.warnings
    )


def test_pipeline_conditional_with_fallback_no_warning():
    """W13 inverse: Conditional edges with unconditional fallback has no warning."""
    from bili.aether.schema.pipeline_spec import (
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="decision",
                node_type="agent",
                agent_spec={
                    "agent_id": "decider",
                    "role": "decider",
                    "objective": "Make a routing decision for testing",
                },
            ),
            PipelineNodeSpec(
                node_id="path_a",
                node_type="agent",
                agent_spec={
                    "agent_id": "handler_a",
                    "role": "handler",
                    "objective": "Handle the positive path in pipeline",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(
                from_node="decision",
                to_node="path_a",
                condition="state.score > 0.5",
            ),
            # Unconditional fallback
            PipelineEdgeSpec(from_node="decision", to_node="END"),
            PipelineEdgeSpec(from_node="path_a", to_node="END"),
        ],
    )

    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("safe_agent", pipeline=pipeline)],
    )

    result = validate_mas(config)

    assert not any(
        "conditional" in w.lower() and "fallback" in w.lower() for w in result.warnings
    )


def test_pipeline_agent_without_pipeline_no_pipeline_checks():
    """Agents without pipelines should not trigger any pipeline warnings."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("normal_agent")],
    )

    result = validate_mas(config)

    assert not any("pipeline" in w.lower() for w in result.warnings)
    assert not any("pipeline" in e.lower() for e in result.errors)


def test_mixed_pipeline_and_non_pipeline_validation():
    """MAS with both pipeline and non-pipeline agents validates correctly."""
    config = MASConfig(
        mas_id="test",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            _agent("pipeline_agent", pipeline=_simple_pipeline()),
            _agent("normal_agent"),
        ],
    )

    result = validate_mas(config)

    assert result.valid is True
