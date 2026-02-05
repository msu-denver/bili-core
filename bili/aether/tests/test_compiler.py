"""Tests for the AETHER-to-LangGraph compiler."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (  # pylint: disable=import-error
    AIMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph  # pylint: disable=import-error
from langgraph.graph.state import CompiledStateGraph  # pylint: disable=import-error

from bili.aether.compiler import CompiledMAS, compile_mas
from bili.aether.compiler.agent_generator import generate_agent_node
from bili.aether.compiler.state_generator import generate_state_schema
from bili.aether.config.loader import load_mas_from_yaml
from bili.aether.schema import (
    AgentSpec,
    MASConfig,
    OutputFormat,
    WorkflowEdge,
    WorkflowType,
)

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "examples",
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


# =========================================================================
# COMPILED MAS TESTS
# =========================================================================


def test_compile_mas_returns_compiled_mas():
    """compile_mas() returns a CompiledMAS with correct agent count."""
    config = MASConfig(
        mas_id="test_seq",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
    )

    result = compile_mas(config)

    assert isinstance(result, CompiledMAS)
    assert isinstance(result.graph, StateGraph)
    assert len(result.agent_nodes) == 2
    assert "a" in result.agent_nodes
    assert "b" in result.agent_nodes


def test_compile_mas_rejects_invalid_config():
    """compile_mas() raises ValueError when validation has errors."""
    config = MASConfig(
        mas_id="bad",
        name="Bad",
        workflow_type=WorkflowType.HIERARCHICAL,
        agents=[_agent("t2", tier=2), _agent("t3", tier=3)],
    )

    with pytest.raises(ValueError, match="validation failed"):
        compile_mas(config)


def test_compile_mas_allows_warnings():
    """compile_mas() succeeds when validation has only warnings."""
    config = MASConfig(
        mas_id="warn",
        name="Warn",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("orphan")],
    )

    result = compile_mas(config)
    assert isinstance(result, CompiledMAS)


def test_compiled_mas_str():
    """__str__ includes mas_id, agent count, and workflow type."""
    config = MASConfig(
        mas_id="test_str",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
    )

    result = compile_mas(config)
    text = str(result)

    assert "test_str" in text
    assert "2 agents" in text
    assert "sequential" in text


def test_get_agent_node():
    """get_agent_node() returns correct callable by ID."""
    config = MASConfig(
        mas_id="test_get",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("x"), _agent("y")],
    )

    result = compile_mas(config)

    assert result.get_agent_node("x") is not None
    assert result.get_agent_node("y") is not None
    assert result.get_agent_node("nonexistent") is None


# =========================================================================
# STATE SCHEMA TESTS
# =========================================================================


def test_state_schema_base_fields():
    """Base state schema has messages, current_agent, agent_outputs."""
    config = MASConfig(
        mas_id="test_state",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
    )

    schema = generate_state_schema(config)
    annotations = schema.__annotations__

    assert "messages" in annotations
    assert "current_agent" in annotations
    assert "agent_outputs" in annotations
    assert "mas_id" in annotations


def test_state_schema_consensus_fields():
    """Consensus state has round tracking and vote fields."""
    config = MASConfig(
        mas_id="test_consensus",
        name="Test",
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
        agents=[_agent("a"), _agent("b")],
    )

    schema = generate_state_schema(config)
    annotations = schema.__annotations__

    assert "current_round" in annotations
    assert "votes" in annotations
    assert "consensus_reached" in annotations
    assert "max_rounds" in annotations


def test_state_schema_hierarchical_fields():
    """Hierarchical state has tier tracking fields."""
    config = MASConfig(
        mas_id="test_hier",
        name="Test",
        workflow_type=WorkflowType.HIERARCHICAL,
        agents=[_agent("a", tier=1)],
    )

    schema = generate_state_schema(config)
    annotations = schema.__annotations__

    assert "current_tier" in annotations
    assert "tier_results" in annotations


def test_state_schema_supervisor_fields():
    """Supervisor state has next_agent and task tracking fields."""
    config = MASConfig(
        mas_id="test_sup",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        agents=[_agent("sup", is_supervisor=True)],
    )

    schema = generate_state_schema(config)
    annotations = schema.__annotations__

    assert "next_agent" in annotations
    assert "pending_tasks" in annotations
    assert "completed_tasks" in annotations


def test_state_schema_sanitizes_mas_id():
    """Hyphens in mas_id are converted to underscores for the class name."""
    config = MASConfig(
        mas_id="my-mas-id",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
    )

    schema = generate_state_schema(config)
    assert "my_mas_id" in schema.__name__


# =========================================================================
# AGENT NODE TESTS
# =========================================================================


def test_agent_node_callable():
    """Agent node callable returns dict with required keys."""
    agent = _agent("test_agent")
    node_fn = generate_agent_node(agent)

    state = {"messages": [], "agent_outputs": {}}
    result = node_fn(state)

    assert isinstance(result, dict)
    assert "messages" in result
    assert "current_agent" in result
    assert "agent_outputs" in result
    assert result["current_agent"] == "test_agent"


def test_agent_node_returns_ai_message():
    """Agent node emits an AIMessage with name set to agent_id."""
    agent = _agent("msg_agent")
    node_fn = generate_agent_node(agent)

    state = {"messages": [], "agent_outputs": {}}
    result = node_fn(state)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.name == "msg_agent"


def test_agent_node_has_spec_attribute():
    """.agent_spec is accessible on the callable."""
    agent = _agent("spec_agent")
    node_fn = generate_agent_node(agent)

    assert hasattr(node_fn, "agent_spec")
    assert node_fn.agent_spec.agent_id == "spec_agent"


def test_agent_node_accumulates_outputs():
    """Agent node merges into existing agent_outputs."""
    agent = _agent("accumulator")
    node_fn = generate_agent_node(agent)

    state = {
        "messages": [],
        "agent_outputs": {"other_agent": {"status": "done"}},
    }
    result = node_fn(state)

    assert "other_agent" in result["agent_outputs"]
    assert "accumulator" in result["agent_outputs"]


# =========================================================================
# GRAPH COMPILATION TESTS
# =========================================================================


def test_sequential_compiles():
    """3-agent sequential graph compiles to a CompiledStateGraph."""
    config = MASConfig(
        mas_id="seq3",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b"), _agent("c")],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_supervisor_compiles():
    """Supervisor graph with workers compiles."""
    config = MASConfig(
        mas_id="sup",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="boss",
        agents=[
            _agent("boss", is_supervisor=True),
            _agent("worker1"),
            _agent("worker2"),
        ],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert "boss" in compiled.agent_nodes
    assert "worker1" in compiled.agent_nodes


def test_parallel_compiles():
    """Parallel fan-out graph compiles."""
    config = MASConfig(
        mas_id="par",
        name="Test",
        workflow_type=WorkflowType.PARALLEL,
        agents=[_agent("a"), _agent("b"), _agent("c")],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_consensus_compiles():
    """Consensus graph compiles and has checker node."""
    config = MASConfig(
        mas_id="cons",
        name="Test",
        workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.5,
        agents=[_agent("a"), _agent("b")],
    )

    compiled = compile_mas(config)
    assert "__consensus_checker__" in compiled.graph.nodes
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_hierarchical_compiles():
    """Hierarchical graph with tiers compiles."""
    config = MASConfig(
        mas_id="hier",
        name="Test",
        workflow_type=WorkflowType.HIERARCHICAL,
        agents=[
            _agent("leaf1", tier=2),
            _agent("leaf2", tier=2),
            _agent("root", tier=1),
        ],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_custom_with_conditions_compiles():
    """Custom graph with conditional edges compiles."""
    config = MASConfig(
        mas_id="cust",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(
                from_agent="a", to_agent="b", condition="state.x == 1", label="go_b"
            ),
            WorkflowEdge(
                from_agent="a", to_agent="c", condition="state.x == 2", label="go_c"
            ),
            WorkflowEdge(from_agent="b", to_agent="END", label="done_b"),
            WorkflowEdge(from_agent="c", to_agent="END", label="done_c"),
        ],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_compile_graph_returns_compiled_type():
    """compile_graph() return type is CompiledStateGraph."""
    config = MASConfig(
        mas_id="type_check",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_checkpoint_disabled():
    """checkpoint_enabled=False still compiles without error."""
    config = MASConfig(
        mas_id="no_cp",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a")],
        checkpoint_enabled=False,
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_deliberative_with_edges_compiles():
    """Deliberative workflow with edges delegates to custom builder."""
    config = MASConfig(
        mas_id="delib",
        name="Test",
        workflow_type=WorkflowType.DELIBERATIVE,
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="END"),
        ],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_deliberative_without_edges_compiles():
    """Deliberative workflow without edges falls back to sequential."""
    config = MASConfig(
        mas_id="delib_seq",
        name="Test",
        workflow_type=WorkflowType.DELIBERATIVE,
        agents=[_agent("a"), _agent("b")],
    )

    compiled = compile_mas(config)
    graph = compiled.compile_graph()
    assert isinstance(graph, CompiledStateGraph)


# =========================================================================
# INTEGRATION TESTS — all example YAMLs
# =========================================================================


@pytest.mark.parametrize(
    "fname",
    [
        "simple_chain.yaml",
        "hierarchical_voting.yaml",
        "supervisor_moderation.yaml",
        "consensus_network.yaml",
        "custom_escalation.yaml",
        "research_analysis.yaml",
        "code_review.yaml",
        "inherited_research.yaml",
    ],
)
def test_example_yaml_compiles(fname):
    """Each example YAML must compile without errors."""
    fpath = os.path.join(_EXAMPLES_DIR, fname)
    if not os.path.exists(fpath):
        pytest.skip(f"{fname} not found")

    config = load_mas_from_yaml(fpath)
    result = compile_mas(config)

    assert isinstance(result, CompiledMAS)
    assert len(result.agent_nodes) == len(config.agents)

    compiled = result.compile_graph()
    assert isinstance(compiled, CompiledStateGraph)


# =========================================================================
# LLM AGENT NODE TESTS (mocked — no API keys required)
# =========================================================================

# Shared mock targets
_MOCK_CREATE = "bili.aether.compiler.llm_resolver.create_llm"
_MOCK_TOOLS = "bili.aether.compiler.llm_resolver.resolve_tools"


def test_llm_agent_node_invokes_model():
    """Agent node with model_name calls LLM invoke."""
    agent = _agent("llm_agent", model_name="gpt-4o")

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="LLM response")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        state = {"messages": [], "agent_outputs": {}}
        result = node_fn(state)

        mock_llm.invoke.assert_called_once()
        assert result["current_agent"] == "llm_agent"
        assert result["messages"][0].content == "LLM response"


def test_agent_without_model_uses_stub():
    """Agent without model_name falls back to stub."""
    agent = _agent("stub_agent")  # no model_name
    node_fn = generate_agent_node(agent)

    state = {"messages": [], "agent_outputs": {}}
    result = node_fn(state)
    assert "[STUB]" in result["messages"][0].content


def test_llm_agent_uses_system_prompt():
    """Agent node passes system_prompt to LLM."""
    agent = _agent(
        "prompt_agent",
        model_name="gpt-4o",
        system_prompt="You are a helper.",
    )

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="response")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        node_fn({"messages": [], "agent_outputs": {}})

        call_args = mock_llm.invoke.call_args[0][0]
        # First message should be the SystemMessage
        assert call_args[0].content == "You are a helper."


def test_llm_agent_falls_back_to_objective():
    """Agent without system_prompt uses objective as system message."""
    agent = _agent("obj_agent", model_name="gpt-4o")

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="response")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        node_fn({"messages": [], "agent_outputs": {}})

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0].content == "Test objective for obj_agent"


def test_llm_agent_json_output_parsing():
    """Agent with output_format=JSON parses response as JSON."""
    agent = _agent(
        "json_agent",
        model_name="gpt-4o",
        output_format=OutputFormat.JSON,
    )

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"key": "value"}')
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        result = node_fn({"messages": [], "agent_outputs": {}})

        assert result["agent_outputs"]["json_agent"]["parsed"] == {"key": "value"}
        assert result["agent_outputs"]["json_agent"]["status"] == "completed"


def test_llm_agent_json_parse_failure():
    """Agent with output_format=JSON handles non-JSON responses gracefully."""
    agent = _agent(
        "bad_json",
        model_name="gpt-4o",
        output_format=OutputFormat.JSON,
    )

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        result = node_fn({"messages": [], "agent_outputs": {}})

        output = result["agent_outputs"]["bad_json"]
        assert "parsed" not in output
        assert output["raw"] == "not json"


def test_llm_agent_has_spec_attribute():
    """.agent_spec is accessible on LLM-backed node."""
    agent = _agent("spec_llm", model_name="gpt-4o")

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_create.return_value = MagicMock()
        node_fn = generate_agent_node(agent)

        assert hasattr(node_fn, "agent_spec")
        assert node_fn.agent_spec.agent_id == "spec_llm"


def test_llm_agent_forwards_state_messages():
    """Agent node forwards existing state messages to LLM."""
    agent = _agent("fwd_agent", model_name="gpt-4o")

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="reply")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        existing_msg = HumanMessage(content="Hello agent")
        node_fn({"messages": [existing_msg], "agent_outputs": {}})

        call_args = mock_llm.invoke.call_args[0][0]
        # SystemMessage + the existing HumanMessage
        assert len(call_args) == 2
        assert call_args[1].content == "Hello agent"


# =========================================================================
# MODEL RESOLUTION TESTS
# =========================================================================


def test_resolve_model_by_model_id():
    """resolve_model finds provider by model_id match."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        resolve_model,
    )

    # Mock LLM_MODELS with a known entry
    mock_models = {
        "remote_openai": {
            "models": [
                {"model_name": "GPT-4o", "model_id": "gpt-4o"},
            ]
        }
    }
    with patch(
        "bili.aether.compiler.llm_resolver.LLM_MODELS",
        mock_models,
        create=True,
    ):
        # Patch the import inside _lookup_in_llm_models
        with patch("bili.config.llm_config.LLM_MODELS", mock_models):
            provider, model_id = resolve_model("gpt-4o")
            assert provider == "remote_openai"
            assert model_id == "gpt-4o"


def test_resolve_model_by_display_name():
    """resolve_model maps display name to model_id."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        resolve_model,
    )

    mock_models = {
        "remote_openai": {
            "models": [
                {"model_name": "GPT-4o", "model_id": "gpt-4o"},
            ]
        }
    }
    with patch("bili.config.llm_config.LLM_MODELS", mock_models):
        provider, model_id = resolve_model("GPT-4o")
        assert provider == "remote_openai"
        assert model_id == "gpt-4o"


def test_resolve_model_heuristic_fallback():
    """resolve_model uses heuristic when LLM_MODELS has no match."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        resolve_model,
    )

    # Empty LLM_MODELS so lookup fails
    with patch("bili.config.llm_config.LLM_MODELS", {}):
        provider, model_id = resolve_model("gpt-4o-mini")
        assert provider == "remote_openai"
        # Heuristic keeps original name as model_id
        assert model_id == "gpt-4o-mini"


def test_resolve_model_bedrock_claude():
    """resolve_model detects Bedrock-hosted Claude by model_id prefix."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        resolve_model,
    )

    with patch("bili.config.llm_config.LLM_MODELS", {}):
        provider, _model_id = resolve_model("anthropic.claude-3-sonnet-20240229-v1:0")
        assert provider == "remote_aws_bedrock"


def test_resolve_model_unknown_raises():
    """resolve_model raises ValueError for unknown model names."""
    from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
        resolve_model,
    )

    with patch("bili.config.llm_config.LLM_MODELS", {}):
        with pytest.raises(ValueError, match="Cannot resolve model"):
            resolve_model("totally-unknown-model-xyz")


# =========================================================================
# TOOL-ENABLED AGENT TESTS
# =========================================================================


def test_tool_agent_uses_create_agent():
    """Agent with tools uses create_agent() for tool-enabled execution."""
    import sys  # pylint: disable=import-outside-toplevel
    import types  # pylint: disable=import-outside-toplevel

    agent = _agent("tool_agent", model_name="gpt-4o", tools=["mock_tool"])

    mock_tool = MagicMock()
    mock_react_agent = MagicMock()
    mock_react_agent.invoke.return_value = {
        "messages": [AIMessage(content="tool result", name="tool_agent")]
    }

    # Stub langchain.agents if not installed
    mock_create_agent_fn = MagicMock(return_value=mock_react_agent)
    langchain_stub = types.ModuleType("langchain")
    agents_stub = types.ModuleType("langchain.agents")
    agents_stub.create_agent = mock_create_agent_fn
    langchain_stub.agents = agents_stub

    with (
        patch(_MOCK_CREATE) as mock_create,
        patch(_MOCK_TOOLS, return_value=[mock_tool]),
        patch.dict(
            sys.modules,
            {
                "langchain": langchain_stub,
                "langchain.agents": agents_stub,
            },
        ),
    ):
        mock_create.return_value = MagicMock()

        node_fn = generate_agent_node(agent)

        # Verify create_agent was called with the LLM and tools
        mock_create_agent_fn.assert_called_once()
        call_kwargs = mock_create_agent_fn.call_args
        assert call_kwargs.kwargs["tools"] == [mock_tool]

        # Invoke and check output
        result = node_fn({"messages": [], "agent_outputs": {}})
        assert result["current_agent"] == "tool_agent"
        assert result["messages"][0].content == "tool result"


def test_agent_with_empty_tools_uses_direct_llm():
    """Agent whose tools resolve to empty list uses direct LLM invoke."""
    agent = _agent("no_tools", model_name="gpt-4o", tools=["nonexistent"])

    with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="direct response")
        mock_create.return_value = mock_llm

        node_fn = generate_agent_node(agent)
        result = node_fn({"messages": [], "agent_outputs": {}})

        # Should use direct LLM, not create_agent
        mock_llm.invoke.assert_called_once()
        assert result["messages"][0].content == "direct response"
