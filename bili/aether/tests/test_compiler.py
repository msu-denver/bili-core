"""Tests for the AETHER-to-LangGraph compiler."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (  # pylint: disable=import-error
    AIMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph  # pylint: disable=import-error
from langgraph.graph.state import CompiledStateGraph  # pylint: disable=import-error

from bili.aether.compiler import CompiledMAS, compile_mas
from bili.aether.compiler.agent_generator import _ensure_human_last, generate_agent_node
from bili.aether.compiler.llm_resolver import resolve_model
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
        "middleware_checkpointer.yaml",
        "pipeline_agents.yaml",
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
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            provider, model_id = resolve_model("gpt-4o")
            assert provider == "remote_openai"
            assert model_id == "gpt-4o"


def test_resolve_model_by_display_name():
    """resolve_model maps display name to model_id."""
    mock_models = {
        "remote_openai": {
            "models": [
                {"model_name": "GPT-4o", "model_id": "gpt-4o"},
            ]
        }
    }
    with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
        provider, model_id = resolve_model("GPT-4o")
        assert provider == "remote_openai"
        assert model_id == "gpt-4o"


def test_resolve_model_heuristic_fallback():
    """resolve_model uses heuristic when LLM_MODELS has no match."""
    # Empty LLM_MODELS so lookup fails
    with patch("bili.iris.config.llm_config.LLM_MODELS", {}):
        provider, model_id = resolve_model("gpt-4o-mini")
        assert provider == "remote_openai"
        # Heuristic keeps original name as model_id
        assert model_id == "gpt-4o-mini"


def test_resolve_model_bedrock_claude():
    """resolve_model detects Bedrock-hosted Claude by model_id prefix."""
    with patch("bili.iris.config.llm_config.LLM_MODELS", {}):
        provider, _model_id = resolve_model("anthropic.claude-3-sonnet-20240229-v1:0")
        assert provider == "remote_aws_bedrock"


def test_resolve_model_unknown_raises():
    """resolve_model raises ValueError for unknown model names."""
    with patch("bili.iris.config.llm_config.LLM_MODELS", {}):
        with pytest.raises(ValueError, match="Cannot resolve model"):
            resolve_model("totally-unknown-model-xyz")


# =========================================================================
# TOOL-ENABLED AGENT TESTS
# =========================================================================


def test_tool_agent_uses_create_agent():
    """Agent with tools uses create_agent() for tool-enabled execution."""
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


# =========================================================================
# _ensure_human_last TESTS
# =========================================================================


class TestEnsureHumanLast:
    """Tests for _ensure_human_last helper."""

    def test_appends_human_when_last_is_ai(self):
        """Appends HumanMessage when the last message is an AIMessage."""
        agent = _agent("test", objective="Do the thing")
        messages = [HumanMessage(content="hello"), AIMessage(content="reply")]
        _ensure_human_last(messages, agent)
        assert len(messages) == 3
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "Do the thing"

    def test_noop_when_last_is_human(self):
        """Does not append when the last message is already a HumanMessage."""
        agent = _agent("test", objective="Do the thing")
        messages = [HumanMessage(content="hello")]
        _ensure_human_last(messages, agent)
        assert len(messages) == 1

    def test_noop_when_empty(self):
        """Does not append when messages list is empty."""
        agent = _agent("test")
        messages = []
        _ensure_human_last(messages, agent)
        assert len(messages) == 0

    def test_uses_fallback_when_no_objective(self):
        """Uses fallback text when agent has no objective."""
        agent = _agent("test")
        # Override objective to None via direct attribute set (bypasses pydantic)
        object.__setattr__(agent, "objective", None)
        messages = [AIMessage(content="prior output")]
        _ensure_human_last(messages, agent)
        assert len(messages) == 2
        assert isinstance(messages[-1], HumanMessage)
        assert "complete your task" in messages[-1].content

    def test_mutates_in_place(self):
        """Mutates the original list rather than creating a new one."""
        agent = _agent("test", objective="Complete the assigned analysis goal")
        messages = [AIMessage(content="response")]
        original_id = id(messages)
        _ensure_human_last(messages, agent)
        assert id(messages) == original_id
        assert len(messages) == 2


# =========================================================================
# PROMPTED TOOL-CALLING TESTS (supports_tools=False path, #304)
# =========================================================================

# Shared mock targets for resolve_tool_strategy.
# resolve_tool_strategy is imported lazily from llm_resolver inside
# _generate_llm_agent_node, so we patch it at the llm_resolver source.
_MOCK_TOOL_STRATEGY = "bili.aether.compiler.llm_resolver.resolve_tool_strategy"

# Backward-compat alias: tests that still call the wrapper function patch here.
_MOCK_SUPPORTS_TOOLS = "bili.aether.compiler.llm_resolver.resolve_supports_tools"


class TestPromptedToolCalling:
    """Tests for the prompted ReAct path in AETHER agent_generator.

    Verifies that an AETHER agent on a non-tool-calling model uses the shared
    prompted ReAct loop from bili.iris.nodes.react_agent_node, not
    create_agent / bind_tools, and that the native and no-tools paths are
    unchanged.
    """

    def _make_mock_tool(self, name: str, return_value: str) -> MagicMock:
        """Build a minimal LangChain-style tool mock."""
        tool = MagicMock()
        tool.name = name
        tool.description = f"Mock tool: {name}"
        tool.args_schema = None
        tool.invoke.return_value = return_value
        return tool

    # ------------------------------------------------------------------
    # Prompted path: non-tool-calling model with tools
    # ------------------------------------------------------------------

    def test_prompted_path_does_not_call_create_agent(self):
        """A facilitated-strategy model with tools must NOT invoke create_agent."""
        agent = _agent("cli_agent", model_name="cli:claude", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("weather", "sunny")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="facilitated"),
            patch(
                "bili.iris.nodes.react_agent_node.create_agent"
            ) as patched_create_agent,
        ):
            mock_llm = MagicMock()
            # Model returns a Final Answer on the first call
            mock_llm.invoke.return_value = MagicMock(
                content="Thought: done\nFinal Answer: The weather is sunny."
            )
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            node_fn({"messages": [], "agent_outputs": {}})
            patched_create_agent.assert_not_called()

    def test_prompted_path_tool_is_invoked(self):
        """Prompted path must invoke the resolved tool when model requests it."""
        agent = _agent("cli_agent", model_name="cli:claude", tools=["weather"])
        weather_tool = self._make_mock_tool("weather", "clear skies")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[weather_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="facilitated"),
        ):
            mock_llm = MagicMock()
            # First call: model requests the weather tool
            # Second call: model produces the final answer
            mock_llm.invoke.side_effect = [
                MagicMock(
                    content=(
                        "Thought: I need the weather.\n"
                        "Action: weather\n"
                        'Action Input: {"location": "Denver"}'
                    )
                ),
                MagicMock(
                    content="Thought: Got it.\nFinal Answer: The weather is clear skies."
                ),
            ]
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            weather_tool.invoke.assert_called_once_with({"location": "Denver"})
            assert result["current_agent"] == "cli_agent"
            assert "clear skies" in result["messages"][0].content

    def test_prompted_path_final_answer_returned(self):
        """Prompted path returns the Final Answer content in the AETHER state update."""
        agent = _agent("cli_agent", model_name="cli:claude", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("mock_tool", "some output")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="facilitated"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="Thought: Done.\nFinal Answer: Forty-two."
            )
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            assert result["current_agent"] == "cli_agent"
            assert result["messages"][0].content == "Forty-two."
            output = result["agent_outputs"]["cli_agent"]
            assert output["status"] == "completed"
            assert output["message"] == "Forty-two."

    def test_prompted_path_respects_max_react_iterations_from_metadata(self):
        """max_react_iterations from agent.metadata caps the prompted loop."""
        agent = _agent(
            "cli_agent",
            model_name="cli:claude",
            tools=["mock_tool"],
            metadata={"max_react_iterations": 2},
        )
        mock_tool = self._make_mock_tool("mock_tool", "result")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="facilitated"),
        ):
            mock_llm = MagicMock()
            # Always return an unparseable response — loop should cap at 2
            mock_llm.invoke.return_value = MagicMock(content="I am confused.")
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            # With max_react_iterations=2 and _MAX_CONSECUTIVE_PARSE_FAILURES=3,
            # the iteration cap fires first; the last model response is returned.
            assert result["current_agent"] == "cli_agent"
            assert mock_llm.invoke.call_count == 2

    # ------------------------------------------------------------------
    # Native path: tool-calling model with tools — unchanged
    # ------------------------------------------------------------------

    def test_native_path_still_uses_create_agent(self):
        """A tool-capable model must use create_agent (native path unchanged)."""
        agent = _agent("api_agent", model_name="gpt-4o", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("mock_tool", "output")
        mock_react_agent = MagicMock()
        mock_react_agent.invoke.return_value = {
            "messages": [AIMessage(content="native result")]
        }

        langchain_stub = types.ModuleType("langchain")
        agents_stub = types.ModuleType("langchain.agents")
        mock_create_agent_fn = MagicMock(return_value=mock_react_agent)
        agents_stub.create_agent = mock_create_agent_fn
        langchain_stub.agents = agents_stub

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="native"),
            patch.dict(
                sys.modules,
                {"langchain": langchain_stub, "langchain.agents": agents_stub},
            ),
        ):
            mock_create_llm.return_value = MagicMock()

            node_fn = generate_agent_node(agent)

            mock_create_agent_fn.assert_called_once()
            result = node_fn({"messages": [], "agent_outputs": {}})
            assert result["current_agent"] == "api_agent"
            assert result["messages"][0].content == "native result"

    # ------------------------------------------------------------------
    # No-tools path — unchanged
    # ------------------------------------------------------------------

    def test_no_tools_path_unchanged(self):
        """Agent without tools calls LLM directly regardless of tool_strategy."""
        agent = _agent("direct_agent", model_name="gpt-4o")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[]),
            patch(_MOCK_TOOL_STRATEGY, return_value="facilitated"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="direct answer")
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            mock_llm.invoke.assert_called_once()
            assert result["messages"][0].content == "direct answer"

    # ------------------------------------------------------------------
    # resolve_supports_tools
    # ------------------------------------------------------------------

    def test_resolve_supports_tools_returns_false_for_flagged_model(self):
        """resolve_supports_tools returns False for a model with supports_tools=False."""
        from bili.aether.compiler.llm_resolver import resolve_supports_tools

        mock_models = {
            "remote_aws_bedrock": {
                "models": [
                    {
                        "model_name": "Amazon Titan Text G1 - Premier",
                        "model_id": "amazon.titan-text-premier-v1:0",
                        "supports_tools": False,
                    },
                ]
            }
        }
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            assert resolve_supports_tools("amazon.titan-text-premier-v1:0") is False
            assert resolve_supports_tools("Amazon Titan Text G1 - Premier") is False

    def test_resolve_supports_tools_defaults_to_true(self):
        """resolve_supports_tools returns True when flag is absent or model unknown."""
        from bili.aether.compiler.llm_resolver import resolve_supports_tools

        mock_models = {
            "remote_openai": {
                "models": [
                    {"model_name": "GPT-4o", "model_id": "gpt-4o"},
                ]
            }
        }
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            # Entry present, no supports_tools key → default True
            assert resolve_supports_tools("gpt-4o") is True
            # Entry not in catalog → default True
            assert resolve_supports_tools("unknown-model-xyz") is True

    def test_resolve_supports_tools_returns_true_on_import_error(self):
        """resolve_supports_tools returns True when bili.iris.config.llm_config is absent.

        Setting sys.modules["bili.iris.config.llm_config"] = None forces Python's
        import machinery to raise ImportError on the lazy ``from bili.iris.config...
        import LLM_MODELS`` inside resolve_tool_strategy, exercising the real
        except-ImportError branch.  The previous patch of the module attribute did
        not trigger that branch (the import succeeded from cache).
        """
        from bili.aether.compiler.llm_resolver import resolve_supports_tools

        with patch.dict(sys.modules, {"bili.iris.config.llm_config": None}):
            # Should not raise; ImportError path defaults to True (via "native").
            result = resolve_supports_tools("any-model")
            assert result is True

    # ------------------------------------------------------------------
    # resolve_tool_strategy
    # ------------------------------------------------------------------

    def test_resolve_tool_strategy_returns_field_when_present(self):
        """resolve_tool_strategy reads the tool_strategy field directly."""
        from bili.aether.compiler.llm_resolver import resolve_tool_strategy

        mock_models = {
            "remote_openai": {
                "models": [
                    {
                        "model_name": "OpenAI GPT-4o Omni",
                        "model_id": "gpt-4o",
                        "tool_strategy": "native",
                        "supports_tools": True,
                    },
                ]
            },
            "remote_deepseek": {
                "models": [
                    {
                        "model_name": "DeepSeek Reasoner",
                        "model_id": "deepseek-reasoner",
                        "tool_strategy": "none",
                        "supports_tools": False,
                    },
                ]
            },
        }
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            assert resolve_tool_strategy("gpt-4o") == "native"
            assert resolve_tool_strategy("deepseek-reasoner") == "none"
            # Lookup by display name also works.
            assert resolve_tool_strategy("OpenAI GPT-4o Omni") == "native"

    def test_resolve_tool_strategy_infers_from_supports_tools_when_field_absent(self):
        """resolve_tool_strategy infers from supports_tools when tool_strategy absent."""
        from bili.aether.compiler.llm_resolver import resolve_tool_strategy

        mock_models = {
            "remote_aws_bedrock": {
                "models": [
                    {
                        "model_name": "Legacy True",
                        "model_id": "legacy-true",
                        "supports_tools": True,
                    },
                    {
                        "model_name": "Legacy False",
                        "model_id": "legacy-false",
                        "supports_tools": False,
                    },
                ]
            }
        }
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            assert resolve_tool_strategy("legacy-true") == "native"
            assert resolve_tool_strategy("legacy-false") == "facilitated"

    def test_resolve_tool_strategy_defaults_to_native_for_unknown_model(self):
        """resolve_tool_strategy defaults to 'native' when model is not in catalog."""
        from bili.aether.compiler.llm_resolver import resolve_tool_strategy

        mock_models = {"remote_openai": {"models": []}}
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            assert resolve_tool_strategy("unknown-model-xyz") == "native"

    def test_resolve_tool_strategy_returns_native_on_import_error(self):
        """resolve_tool_strategy defaults to 'native' when LLM_MODELS cannot be imported."""
        from bili.aether.compiler.llm_resolver import resolve_tool_strategy

        with patch(
            "bili.iris.config.llm_config.LLM_MODELS",
            side_effect=ImportError("no module"),
            create=True,
        ):
            result = resolve_tool_strategy("any-model")
            assert result == "native"

    def test_resolve_tool_strategy_mcp_and_none_values(self):
        """resolve_tool_strategy returns 'mcp' and 'none' for the new strategy values."""
        from bili.aether.compiler.llm_resolver import resolve_tool_strategy

        mock_models = {
            "cli_claude_code": {
                "models": [
                    {
                        "model_name": "Claude Code CLI",
                        "model_id": "cli:claude_code",
                        "tool_strategy": "mcp",
                        "supports_tools": False,
                    },
                ]
            },
            "remote_openai": {
                "models": [
                    {
                        "model_name": "OpenAI o1-mini",
                        "model_id": "o1-mini",
                        "tool_strategy": "none",
                        "supports_tools": False,
                    },
                ]
            },
        }
        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            assert resolve_tool_strategy("cli:claude_code") == "mcp"
            assert resolve_tool_strategy("o1-mini") == "none"

    # ------------------------------------------------------------------
    # mcp and none routing in _generate_tool_agent_node
    # ------------------------------------------------------------------

    def test_mcp_strategy_with_known_cli_calls_build_mcp_node(self):
        """An 'mcp' strategy with a known CLI -> build_mcp_node is invoked."""
        agent = _agent("cli_agent", model_name="cli:claude_code", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("mock_tool", "some output")

        mock_node = MagicMock(
            return_value={
                "messages": [],
                "agent_outputs": {},
                "current_agent": "cli_agent",
            }
        )

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="mcp"),
            patch(
                "bili.iris.mcp.server.build_mcp_node", return_value=mock_node
            ) as mock_build,
            patch("bili.iris.mcp.server.resolve_mcp_injector") as mock_resolve,
        ):
            from bili.iris.mcp.cli_injectors import ClaudeCodeInjector

            mock_llm = MagicMock()
            mock_llm.command = ["claude", "-p"]
            mock_create_llm.return_value = mock_llm
            mock_resolve.return_value = ClaudeCodeInjector()

            generate_agent_node(agent)

            mock_build.assert_called_once()

    def test_mcp_strategy_unknown_cli_falls_back_to_direct_llm(self):
        """An 'mcp' strategy with no injector falls back to the direct-LLM node."""
        agent = _agent("cli_agent", model_name="cli:custom", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("mock_tool", "some output")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="mcp"),
            patch("bili.iris.mcp.server.resolve_mcp_injector", return_value=None),
        ):
            mock_llm = MagicMock()
            mock_llm.command = ["unknown-cli"]
            mock_llm.invoke.return_value = MagicMock(content="fallback answer")
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            mock_llm.invoke.assert_called_once()
            assert result["current_agent"] == "cli_agent"

    def test_none_strategy_drops_tools_and_uses_direct_llm(self):
        """A 'none' strategy drops tools and routes to the direct-LLM node."""
        agent = _agent("reasoner", model_name="deepseek-reasoner", tools=["mock_tool"])
        mock_tool = self._make_mock_tool("mock_tool", "some output")

        with (
            patch(_MOCK_CREATE) as mock_create_llm,
            patch(_MOCK_TOOLS, return_value=[mock_tool]),
            patch(_MOCK_TOOL_STRATEGY, return_value="none"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="reasoner answer")
            mock_create_llm.return_value = mock_llm

            node_fn = generate_agent_node(agent)
            result = node_fn({"messages": [], "agent_outputs": {}})

            mock_llm.invoke.assert_called_once()
            assert result["current_agent"] == "reasoner"
