"""Targeted coverage tests for AETHER compiler, integration, schema, and package init.

These tests exercise the error branches, fallback paths, and helper functions
that the broader functional tests do not reach:

- ``bili.aether.__init__`` lazy-import machinery (PEP 562 ``__getattr__``/``__dir__``)
- ``compiler.cli`` main entry points (single-file and all-examples modes)
- ``compiler.llm_resolver`` model creation, tool resolution, fallbacks
- ``compiler.agent_generator`` content normalisation, middleware, routing, supervisor paths
- ``compiler.compiled_mas`` checkpointer ImportError fallback
- ``compiler.graph_builder`` evaluator error branches, consensus voting,
  custom edges, hierarchical fallback, registry resolution
- ``integration.checkpointer_factory`` all dispatch branches and fallbacks
- ``schema.mas_config`` validation error branches and helper methods
"""

import ast
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (  # pylint: disable=import-error
    AIMessage,
    HumanMessage,
)

from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.aether.schema.enums import CommunicationProtocol
from bili.aether.schema.mas_config import Channel, WorkflowEdge


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    """Build an AgentSpec with sensible defaults."""
    defaults = {"role": "test_role", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


# =========================================================================
# bili.aether.__init__ — lazy import machinery
# =========================================================================


class TestPackageLazyImports:
    """Tests for PEP 562 __getattr__ / __dir__ on bili.aether."""

    def test_lazy_attribute_resolves_to_real_object(self):
        """Accessing a mapped name imports and returns the real object."""
        import bili.aether as aether  # pylint: disable=import-outside-toplevel

        # MASConfig maps to the schema submodule
        resolved = aether.MASConfig
        assert resolved is MASConfig
        # After access, the name is cached in module globals
        assert "MASConfig" in vars(aether)

    def test_lazy_submodule_import(self):
        """The __getattr__ submodule branch imports and returns the module.

        __getattr__ only fires when normal attribute lookup fails, so we drop
        any cached value from the module globals first to force the lazy path.
        """
        import bili.aether as aether  # pylint: disable=import-outside-toplevel

        vars(aether).pop("schema", None)
        schema_mod = aether.__getattr__("schema")
        assert isinstance(schema_mod, types.ModuleType)
        assert schema_mod.__name__ == "bili.aether.schema"
        # The resolved module is now cached back into globals.
        assert vars(aether)["schema"] is schema_mod

    def test_unknown_attribute_raises_attribute_error(self):
        """Accessing an unmapped name raises AttributeError."""
        import bili.aether as aether  # pylint: disable=import-outside-toplevel

        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            _ = aether.does_not_exist

    def test_dir_lists_lazy_names_and_dunders(self):
        """__dir__ includes lazy imports, submodules, and version dunders."""
        import bili.aether as aether  # pylint: disable=import-outside-toplevel

        names = dir(aether)
        assert "MASConfig" in names
        assert "schema" in names
        assert "__version__" in names
        assert "__author__" in names

    def test_compile_mas_lazy_import(self):
        """compile_mas is reachable via the lazy mapping to the compiler."""
        import bili.aether as aether  # pylint: disable=import-outside-toplevel
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        assert aether.compile_mas is compile_mas


# =========================================================================
# compiler.cli — main()
# =========================================================================


_EXAMPLES_PATH = "bili/aether/config/examples"


class TestCompilerCli:
    """Tests for the compiler CLI entry point."""

    def test_main_single_file(self, capsys):
        """main() with a file argument compiles that one file and prints OK."""
        from bili.aether.compiler import cli  # pylint: disable=import-outside-toplevel

        fpath = f"{_EXAMPLES_PATH}/simple_chain.yaml"
        with patch.object(sys, "argv", ["cli.py", fpath]):
            cli.main()

        out = capsys.readouterr().out
        assert f"OK    {fpath}" in out
        assert "Compiled:" in out

    def test_main_all_examples_success(self, capsys):
        """main() with no args compiles all known examples and exits 0."""
        from bili.aether.compiler import cli  # pylint: disable=import-outside-toplevel

        with patch.object(sys, "argv", ["cli.py"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "OK    simple_chain.yaml" in out

    def test_main_skips_missing_file(self, capsys):
        """main() prints SKIP for a missing example and still exits."""
        from bili.aether.compiler import cli  # pylint: disable=import-outside-toplevel

        with patch.object(sys, "argv", ["cli.py"]):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "SKIP" in out

    def test_main_reports_failure(self, capsys):
        """main() prints FAIL and exits 1 when a compile raises."""
        from bili.aether.compiler import cli  # pylint: disable=import-outside-toplevel

        # Force load_mas_from_yaml to raise for every example
        with patch.object(sys, "argv", ["cli.py"]):
            with patch(
                "bili.aether.config.loader.load_mas_from_yaml",
                side_effect=RuntimeError("boom"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_ensure_bili_stub_inserts_module_and_path(self):
        """_ensure_bili_stub installs a bili stub and inserts project_root on sys.path."""
        # Compute the project_root the same way the function does.
        import os  # pylint: disable=import-outside-toplevel

        from bili.aether.compiler import cli  # pylint: disable=import-outside-toplevel

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(cli.__file__)))
            )
        )

        saved_mod = sys.modules.get("bili")
        had_path = project_root in sys.path
        try:
            sys.modules.pop("bili", None)
            if had_path:
                sys.path.remove(project_root)

            cli._ensure_bili_stub()  # pylint: disable=protected-access

            assert project_root in sys.path
            assert "bili" in sys.modules
            assert hasattr(sys.modules["bili"], "__path__")
        finally:
            if saved_mod is not None:
                sys.modules["bili"] = saved_mod
            if not had_path and project_root in sys.path:
                sys.path.remove(project_root)


# =========================================================================
# compiler.llm_resolver — create_llm / resolve_tools
# =========================================================================


class TestCreateLlm:
    """Tests for create_llm()."""

    def test_create_llm_no_model_name_raises(self):
        """create_llm raises ValueError when the agent has no model_name."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            create_llm,
        )

        agent = _agent("no_model")
        with pytest.raises(ValueError, match="has no model_name"):
            create_llm(agent)

    def test_create_llm_passes_resolved_kwargs(self):
        """create_llm resolves model_id and forwards temperature/max_tokens."""
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            llm_resolver,
        )

        agent = _agent("m", model_name="gpt-4o", temperature=0.3, max_tokens=512)

        fake_load_model = MagicMock(return_value="LLM_INSTANCE")
        fake_loader = types.ModuleType("bili.iris.loaders.llm_loader")
        fake_loader.load_model = fake_load_model

        with patch("bili.iris.config.llm_config.LLM_MODELS", {}):
            with patch.dict(sys.modules, {"bili.iris.loaders.llm_loader": fake_loader}):
                result = llm_resolver.create_llm(agent)

        assert result == "LLM_INSTANCE"
        provider_arg = fake_load_model.call_args[0][0]
        kwargs = fake_load_model.call_args[1]
        assert provider_arg == "remote_openai"
        assert kwargs["model_name"] == "gpt-4o"
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 512

    def test_create_llm_extra_kwargs_from_llm_models(self):
        """Provider-specific kwargs from LLM_MODELS are forwarded, model_id wins."""
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            llm_resolver,
        )

        mock_models = {
            "remote_azure_openai": {
                "models": [
                    {
                        "model_name": "Azure GPT",
                        "model_id": "azure-gpt-4o",
                        "kwargs": {"api_version": "2024-02-01"},
                    }
                ]
            }
        }
        agent = _agent("azure", model_name="Azure GPT")

        fake_load_model = MagicMock(return_value="AZURE_LLM")
        fake_loader = types.ModuleType("bili.iris.loaders.llm_loader")
        fake_loader.load_model = fake_load_model

        with patch("bili.iris.config.llm_config.LLM_MODELS", mock_models):
            with patch.dict(sys.modules, {"bili.iris.loaders.llm_loader": fake_loader}):
                llm_resolver.create_llm(agent)

        kwargs = fake_load_model.call_args[1]
        assert kwargs["api_version"] == "2024-02-01"
        assert kwargs["model_name"] == "azure-gpt-4o"

    def test_create_llm_model_type_override_uses_provider_verbatim(self):
        """model_type set → provider used as-is, model_name forwarded as model_id."""
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            llm_resolver,
        )

        agent = _agent(
            "victim",
            model_name="claude-sonnet-4-6",
            model_type="remote_anthropic",
            temperature=0.0,
        )

        fake_load_model = MagicMock(return_value="ANTHROPIC_LLM")
        fake_loader = types.ModuleType("bili.iris.loaders.llm_loader")
        fake_loader.load_model = fake_load_model

        # LLM_MODELS deliberately holds a colliding entry that would resolve
        # claude-sonnet-4-6 to Bedrock; model_type must win and skip the lookup.
        colliding_models = {
            "remote_aws_bedrock": {
                "models": [
                    {
                        "model_name": "claude-sonnet-4-6",
                        "model_id": "us.anthropic.claude-sonnet-4-6",
                    }
                ]
            }
        }

        with patch("bili.iris.config.llm_config.LLM_MODELS", colliding_models):
            with patch.dict(sys.modules, {"bili.iris.loaders.llm_loader": fake_loader}):
                result = llm_resolver.create_llm(agent)

        assert result == "ANTHROPIC_LLM"
        provider_arg = fake_load_model.call_args[0][0]
        kwargs = fake_load_model.call_args[1]
        assert provider_arg == "remote_anthropic"
        # model_name passed through verbatim, NOT rewritten to the Bedrock id.
        assert kwargs["model_name"] == "claude-sonnet-4-6"
        assert kwargs["temperature"] == 0.0

    def test_create_llm_model_type_override_skips_resolution(self):
        """model_type bypasses _resolve_model_full entirely (never called)."""
        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            llm_resolver,
        )

        # A model_name the resolver could never resolve on its own — proves the
        # override path does not fall through to the registry/heuristic.
        agent = _agent(
            "victim",
            model_name="totally-unknown-direct-model",
            model_type="remote_deepseek",
        )

        fake_load_model = MagicMock(return_value="DEEPSEEK_LLM")
        fake_loader = types.ModuleType("bili.iris.loaders.llm_loader")
        fake_loader.load_model = fake_load_model

        with patch.object(
            llm_resolver, "_resolve_model_full", side_effect=AssertionError("called")
        ):
            with patch.dict(sys.modules, {"bili.iris.loaders.llm_loader": fake_loader}):
                result = llm_resolver.create_llm(agent)

        assert result == "DEEPSEEK_LLM"
        assert fake_load_model.call_args[0][0] == "remote_deepseek"
        assert (
            fake_load_model.call_args[1]["model_name"] == "totally-unknown-direct-model"
        )


class TestResolveProvider:
    """Tests for resolve_provider() and the LLM_MODELS ImportError path."""

    def test_resolve_provider_returns_provider_only(self):
        """resolve_provider returns just the provider type string."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_provider,
        )

        with patch("bili.iris.config.llm_config.LLM_MODELS", {}):
            assert resolve_provider("gpt-4o") == "remote_openai"

    def test_lookup_import_error_returns_none(self):
        """_lookup_in_llm_models returns None when llm_config is unimportable."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            _lookup_in_llm_models,
        )

        with patch.dict(sys.modules, {"bili.iris.config.llm_config": None}):
            assert _lookup_in_llm_models("anything") is None


class TestResolveTools:
    """Tests for resolve_tools()."""

    def test_resolve_tools_empty_returns_empty(self):
        """Agent with no tools returns an empty list without importing loaders."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_tools,
        )

        assert resolve_tools(_agent("none")) == []

    def test_resolve_tools_import_error_returns_empty(self):
        """ImportError on the tools loader yields an empty list."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_tools,
        )

        agent = _agent("t", tools=["weather_api_tool"])
        with patch.dict(
            sys.modules,
            {
                "bili.iris.config.tool_config": None,
                "bili.iris.loaders.tools_loader": None,
            },
        ):
            assert resolve_tools(agent) == []

    def test_resolve_tools_builds_prompts_and_calls_initialize(self):
        """resolve_tools forwards active tools and default prompts to initialize_tools."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_tools,
        )

        agent = _agent("t", tools=["weather_api_tool"])

        fake_tool_config = types.ModuleType("bili.iris.config.tool_config")
        fake_tool_config.TOOLS = {
            "weather_api_tool": {"default_prompt": "Get the weather"}
        }
        fake_init = MagicMock(return_value=["TOOL_OBJ"])
        fake_loader = types.ModuleType("bili.iris.loaders.tools_loader")
        fake_loader.initialize_tools = fake_init

        with patch.dict(
            sys.modules,
            {
                "bili.iris.config.tool_config": fake_tool_config,
                "bili.iris.loaders.tools_loader": fake_loader,
            },
        ):
            result = resolve_tools(agent)

        assert result == ["TOOL_OBJ"]
        call_kwargs = fake_init.call_args[1]
        assert call_kwargs["active_tools"] == ["weather_api_tool"]
        assert call_kwargs["tool_prompts"] == {"weather_api_tool": "Get the weather"}

    def test_resolve_tools_initialize_failure_returns_empty(self):
        """A failure inside initialize_tools is swallowed and yields an empty list."""
        from bili.aether.compiler.llm_resolver import (  # pylint: disable=import-outside-toplevel
            resolve_tools,
        )

        agent = _agent("t", tools=["weather_api_tool"])

        fake_tool_config = types.ModuleType("bili.iris.config.tool_config")
        fake_tool_config.TOOLS = {}
        fake_loader = types.ModuleType("bili.iris.loaders.tools_loader")
        fake_loader.initialize_tools = MagicMock(side_effect=RuntimeError("nope"))

        with patch.dict(
            sys.modules,
            {
                "bili.iris.config.tool_config": fake_tool_config,
                "bili.iris.loaders.tools_loader": fake_loader,
            },
        ):
            assert resolve_tools(agent) == []


# =========================================================================
# compiler.agent_generator — normalisation, middleware, routing, supervisor
# =========================================================================


class TestContentNormalisation:
    """Tests for _normalise_content_value and _normalise_message_content."""

    def test_normalise_list_of_part_dicts(self):
        """List of part dicts is joined on the 'text' key."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _normalise_content_value,
        )

        value = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert _normalise_content_value(value) == "hello world"

    def test_normalise_list_with_non_dict_part(self):
        """Non-dict list parts fall back to str()."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _normalise_content_value,
        )

        assert _normalise_content_value(["a", 7]) == "a 7"

    def test_normalise_message_with_list_content(self):
        """A message whose content is a list is rewritten to a string copy."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _normalise_message_content,
        )

        msg = AIMessage(content=[{"type": "text", "text": "joined"}])
        normalised = _normalise_message_content(msg)
        assert normalised.content == "joined"

    def test_normalise_message_with_str_content_unchanged(self):
        """A message with string content is returned unchanged."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _normalise_message_content,
        )

        msg = AIMessage(content="plain")
        assert _normalise_message_content(msg) is msg


class TestResolveMiddleware:
    """Tests for _resolve_middleware()."""

    def test_no_middleware_returns_empty(self):
        """Agent with no middleware returns an empty list."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _resolve_middleware,
        )

        assert _resolve_middleware(_agent("a")) == []

    def test_import_error_returns_empty(self):
        """ImportError on the middleware loader returns an empty list."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _resolve_middleware,
        )

        agent = _agent("a", middleware=["summarization"])
        with patch.dict(sys.modules, {"bili.iris.loaders.middleware_loader": None}):
            assert _resolve_middleware(agent) == []

    def test_resolves_instances(self):
        """Middleware names resolve to instances via initialize_middleware."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _resolve_middleware,
        )

        agent = _agent(
            "a",
            middleware=["model_call_limit"],
            middleware_params={"model_call_limit": {"run_limit": 3}},
        )
        fake_init = MagicMock(return_value=["MW1"])
        fake_loader = types.ModuleType("bili.iris.loaders.middleware_loader")
        fake_loader.initialize_middleware = fake_init

        with patch.dict(
            sys.modules, {"bili.iris.loaders.middleware_loader": fake_loader}
        ):
            result = _resolve_middleware(agent)

        assert result == ["MW1"]
        call_kwargs = fake_init.call_args[1]
        assert call_kwargs["active_middleware"] == ["model_call_limit"]
        assert call_kwargs["middleware_params"] == {
            "model_call_limit": {"run_limit": 3}
        }

    def test_initialize_failure_returns_empty(self):
        """A failure inside initialize_middleware yields an empty list."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _resolve_middleware,
        )

        agent = _agent("a", middleware=["summarization"])
        fake_loader = types.ModuleType("bili.iris.loaders.middleware_loader")
        fake_loader.initialize_middleware = MagicMock(side_effect=RuntimeError("x"))

        with patch.dict(
            sys.modules, {"bili.iris.loaders.middleware_loader": fake_loader}
        ):
            assert _resolve_middleware(agent) == []


class TestExtractNextAgent:
    """Tests for _extract_next_agent routing extraction."""

    def test_json_next_agent(self):
        """JSON object with next_agent field is parsed."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _extract_next_agent,
        )

        result = _extract_next_agent('{"next_agent": "worker_2"}', _agent("sup"))
        assert result == "worker_2"

    def test_text_route_pattern(self):
        """ROUTE_TO directive is matched when JSON parsing fails."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _extract_next_agent,
        )

        result = _extract_next_agent("Decision made. ROUTE_TO: worker_1", _agent("sup"))
        assert result == "worker_1"

    def test_defaults_to_end(self):
        """No routing decision defaults to END."""
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            _extract_next_agent,
        )

        assert _extract_next_agent("just some text", _agent("sup")) == "END"


class TestDirectLlmNodeBranches:
    """Tests for the direct (no-tools) LLM node's prompt-assembly branches."""

    def _make_node(self, agent):
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            generate_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="resp")
        with patch(
            "bili.aether.compiler.llm_resolver.create_llm", return_value=mock_llm
        ), patch("bili.aether.compiler.llm_resolver.resolve_tools", return_value=[]):
            node = generate_agent_node(agent)
        return node, mock_llm

    def test_inserts_human_cue_before_leading_ai_message(self):
        """A leading AIMessage gets a synthetic HumanMessage inserted before it."""
        agent = _agent("d", model_name="gpt-4o")
        node, mock_llm = self._make_node(agent)
        node({"messages": [AIMessage(content="prior")], "agent_outputs": {}})

        sent = mock_llm.invoke.call_args[0][0]
        # SystemMessage, then synthetic HumanMessage, then the prior AIMessage
        assert isinstance(sent[1], HumanMessage)
        assert "complete your task" in sent[1].content

    def test_supervisor_sets_next_agent(self):
        """Supervisor agent populates next_agent from the LLM response."""
        agent = _agent("boss", model_name="gpt-4o", is_supervisor=True)
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            generate_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ROUTE_TO: helper")
        with patch(
            "bili.aether.compiler.llm_resolver.create_llm", return_value=mock_llm
        ), patch("bili.aether.compiler.llm_resolver.resolve_tools", return_value=[]):
            node = generate_agent_node(agent)
        result = node({"messages": [], "agent_outputs": {}})
        assert result["next_agent"] == "helper"

    def test_middleware_without_tools_is_ignored(self):
        """Direct LLM path warns and ignores middleware when no tools are present."""
        agent = _agent("mw", model_name="gpt-4o", middleware=["summarization"])
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            generate_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="resp")
        with patch(
            "bili.aether.compiler.llm_resolver.create_llm", return_value=mock_llm
        ), patch(
            "bili.aether.compiler.llm_resolver.resolve_tools", return_value=[]
        ), patch(
            "bili.aether.compiler.agent_generator._resolve_middleware",
            return_value=["MW"],
        ):
            node = generate_agent_node(agent)
        # Direct node still invokes the LLM directly
        result = node({"messages": [], "agent_outputs": {}})
        assert result["current_agent"] == "mw"
        mock_llm.invoke.assert_called_once()


class TestToolAgentNodeBranches:
    """Tests for the tool-enabled node's comm-context and supervisor branches."""

    def _build_tool_node(self, agent, react_return):
        from bili.aether.compiler.agent_generator import (  # pylint: disable=import-outside-toplevel
            generate_agent_node,
        )

        mock_react = MagicMock()
        mock_react.invoke.return_value = react_return
        create_agent_fn = MagicMock(return_value=mock_react)

        langchain_stub = types.ModuleType("langchain")
        agents_stub = types.ModuleType("langchain.agents")
        agents_stub.create_agent = create_agent_fn
        langchain_stub.agents = agents_stub

        with patch(
            "bili.aether.compiler.llm_resolver.create_llm", return_value=MagicMock()
        ), patch(
            "bili.aether.compiler.llm_resolver.resolve_tools",
            return_value=[MagicMock()],
        ), patch.dict(
            sys.modules,
            {"langchain": langchain_stub, "langchain.agents": agents_stub},
        ):
            node = generate_agent_node(agent)
        return node, mock_react

    def test_tool_supervisor_sets_next_agent(self):
        """Tool-enabled supervisor extracts next_agent from its output."""
        agent = _agent("sup", model_name="gpt-4o", tools=["x"], is_supervisor=True)
        react_return = {
            "messages": [AIMessage(content='{"next_agent": "w1"}', name="sup")]
        }
        node, _ = self._build_tool_node(agent, react_return)
        result = node({"messages": [], "agent_outputs": {}})
        assert result["next_agent"] == "w1"

    def test_tool_node_appends_communication_context(self):
        """Pending inter-agent messages are appended to the tool node's system prompt."""
        agent = _agent("t", model_name="gpt-4o", tools=["x"])
        react_return = {"messages": [AIMessage(content="done", name="t")]}
        node, mock_react = self._build_tool_node(agent, react_return)

        state = {
            "messages": [],
            "agent_outputs": {},
            "communication_log": [],
            "pending_messages": {
                "__all__": [
                    {
                        "sender": "other",
                        "channel_id": "__agent_output__",
                        "content": "Prior analysis result.",
                    }
                ]
            },
        }
        node(state)

        sent = mock_react.invoke.call_args[0][0]["messages"]
        system_msg = sent[0]
        assert "Messages from other agents" in system_msg.content
        assert "Prior analysis result." in system_msg.content

    def test_tool_node_injects_human_before_leading_ai(self):
        """Tool node inserts a synthetic HumanMessage before a leading AIMessage."""
        agent = _agent("t", model_name="gpt-4o", tools=["x"])
        react_return = {"messages": [AIMessage(content="done", name="t")]}
        node, mock_react = self._build_tool_node(agent, react_return)
        node({"messages": [AIMessage(content="prior")], "agent_outputs": {}})

        sent = mock_react.invoke.call_args[0][0]["messages"]
        # The first non-system message should be the injected HumanMessage
        [m for m in sent if not isinstance(m, type(sent[0]))]
        assert any(
            isinstance(m, HumanMessage) and "complete your task" in m.content
            for m in sent
        )


# =========================================================================
# compiler.compiled_mas — checkpointer ImportError fallback
# =========================================================================


class TestCompiledMasCheckpointer:
    """Tests for CompiledMAS._create_checkpointer fallback."""

    def test_create_checkpointer_import_error_falls_back_to_memory(self):
        """ImportError on the factory falls back to a langgraph MemorySaver."""
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel,import-error
            MemorySaver,
        )

        from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
            compile_mas,
        )

        config = MASConfig(
            mas_id="cp",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
            checkpoint_enabled=True,
        )
        compiled = compile_mas(config)

        with patch.dict(
            sys.modules,
            {"bili.aether.integration.checkpointer_factory": None},
        ):
            saver = compiled._create_checkpointer()  # pylint: disable=protected-access

        assert isinstance(saver, MemorySaver)


# =========================================================================
# integration.checkpointer_factory
# =========================================================================


class TestCheckpointerFactory:
    """Tests for create_checkpointer_from_config dispatch and fallbacks."""

    def test_unknown_type_falls_back_to_memory(self):
        """Unknown checkpoint type falls back to a memory checkpointer."""
        from bili.aether.integration.checkpointer_factory import (  # pylint: disable=import-outside-toplevel
            create_checkpointer_from_config,
        )

        saver = create_checkpointer_from_config({"type": "redis"})
        assert saver is not None

    def test_memory_uses_queryable_saver(self):
        """memory type returns a QueryableMemorySaver when available."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        fake_saver = MagicMock(name="QueryableMemorySaver")
        fake_mod = types.ModuleType("bili.iris.checkpointers.memory_checkpointer")
        fake_mod.QueryableMemorySaver = MagicMock(return_value=fake_saver)

        with patch.dict(
            sys.modules,
            {"bili.iris.checkpointers.memory_checkpointer": fake_mod},
        ):
            saver = checkpointer_factory.create_checkpointer_from_config(
                {"type": "memory"}, user_id="u1"
            )

        assert saver is fake_saver
        fake_mod.QueryableMemorySaver.assert_called_once_with(user_id="u1")

    def test_memory_import_error_falls_back_to_langgraph(self):
        """When QueryableMemorySaver is unavailable, a plain MemorySaver is returned."""
        from langgraph.checkpoint.memory import (  # pylint: disable=import-outside-toplevel,import-error
            MemorySaver,
        )

        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        with patch.dict(
            sys.modules,
            {"bili.iris.checkpointers.memory_checkpointer": None},
        ):
            saver = (
                checkpointer_factory._create_memory_checkpointer()
            )  # pylint: disable=protected-access

        assert isinstance(saver, MemorySaver)

    def test_postgres_returns_checkpointer(self):
        """postgres type forwards keep_last_n and returns the checkpointer."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        fake_cp = MagicMock(name="pg_cp")
        fake_mod = types.ModuleType("bili.iris.checkpointers.pg_checkpointer")
        fake_mod.get_pg_checkpointer = MagicMock(return_value=fake_cp)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.pg_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory.create_checkpointer_from_config(
                {"type": "postgres", "keep_last_n": 9}, user_id="u2"
            )

        assert saver is fake_cp
        fake_mod.get_pg_checkpointer.assert_called_once_with(
            keep_last_n=9, user_id="u2"
        )

    def test_postgres_none_falls_back_to_memory(self):
        """postgres returning None falls back to a memory checkpointer."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        fake_mod = types.ModuleType("bili.iris.checkpointers.pg_checkpointer")
        fake_mod.get_pg_checkpointer = MagicMock(return_value=None)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.pg_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory._create_postgres_checkpointer(  # pylint: disable=protected-access
                {"type": "postgres"}
            )

        # Falls back to memory (some saver instance)
        assert saver is not None

    def test_postgres_import_error_falls_back_to_memory(self):
        """ImportError on pg_checkpointer falls back to memory."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        with patch.dict(sys.modules, {"bili.iris.checkpointers.pg_checkpointer": None}):
            saver = checkpointer_factory._create_postgres_checkpointer(  # pylint: disable=protected-access
                {"type": "postgres"}
            )
        assert saver is not None

    def test_mongo_returns_checkpointer(self):
        """mongo type forwards keep_last_n and returns the checkpointer."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        fake_cp = MagicMock(name="mongo_cp")
        fake_mod = types.ModuleType("bili.iris.checkpointers.mongo_checkpointer")
        fake_mod.get_mongo_checkpointer = MagicMock(return_value=fake_cp)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.mongo_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory.create_checkpointer_from_config(
                {"type": "mongodb", "keep_last_n": 4}
            )

        assert saver is fake_cp
        fake_mod.get_mongo_checkpointer.assert_called_once_with(
            keep_last_n=4, user_id=None
        )

    def test_mongo_none_falls_back_to_memory(self):
        """mongo returning None falls back to memory."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        fake_mod = types.ModuleType("bili.iris.checkpointers.mongo_checkpointer")
        fake_mod.get_mongo_checkpointer = MagicMock(return_value=None)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.mongo_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory._create_mongo_checkpointer(  # pylint: disable=protected-access
                {"type": "mongo"}
            )
        assert saver is not None

    def test_mongo_import_error_falls_back_to_memory(self):
        """ImportError on mongo_checkpointer falls back to memory."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.mongo_checkpointer": None}
        ):
            saver = checkpointer_factory._create_mongo_checkpointer(  # pylint: disable=protected-access
                {"type": "mongo"}
            )
        assert saver is not None

    def test_auto_detects_postgres(self, monkeypatch):
        """auto type uses Postgres when POSTGRES_CONNECTION_STRING is set."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://x")
        monkeypatch.delenv("MONGO_CONNECTION_STRING", raising=False)
        fake_cp = MagicMock(name="auto_pg")
        fake_mod = types.ModuleType("bili.iris.checkpointers.pg_checkpointer")
        fake_mod.get_pg_checkpointer = MagicMock(return_value=fake_cp)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.pg_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory.create_checkpointer_from_config(
                {"type": "auto"}
            )
        assert saver is fake_cp

    def test_auto_detects_mongo(self, monkeypatch):
        """auto type uses Mongo when only MONGO_CONNECTION_STRING is set."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
        monkeypatch.setenv("MONGO_CONNECTION_STRING", "mongodb://x")
        fake_cp = MagicMock(name="auto_mongo")
        fake_mod = types.ModuleType("bili.iris.checkpointers.mongo_checkpointer")
        fake_mod.get_mongo_checkpointer = MagicMock(return_value=fake_cp)

        with patch.dict(
            sys.modules, {"bili.iris.checkpointers.mongo_checkpointer": fake_mod}
        ):
            saver = checkpointer_factory._create_auto_checkpointer(  # pylint: disable=protected-access
                {"type": "auto"}
            )
        assert saver is fake_cp

    def test_auto_falls_back_to_memory(self, monkeypatch):
        """auto type falls back to QueryableMemorySaver with no DB env vars."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
        monkeypatch.delenv("MONGO_CONNECTION_STRING", raising=False)
        fake_saver = MagicMock(name="mem")
        fake_mod = types.ModuleType("bili.iris.checkpointers.memory_checkpointer")
        fake_mod.QueryableMemorySaver = MagicMock(return_value=fake_saver)

        with patch.dict(
            sys.modules,
            {"bili.iris.checkpointers.memory_checkpointer": fake_mod},
        ):
            saver = checkpointer_factory._create_auto_checkpointer(  # pylint: disable=protected-access
                {"type": "auto"}, user_id="u3"
            )
        assert saver is fake_saver

    def test_auto_import_error_falls_back_to_memory(self, monkeypatch):
        """An ImportError during auto-detection falls back to memory."""
        from bili.aether.integration import (  # pylint: disable=import-outside-toplevel
            checkpointer_factory,
        )

        monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://x")
        with patch.dict(sys.modules, {"bili.iris.checkpointers.pg_checkpointer": None}):
            saver = checkpointer_factory._create_auto_checkpointer(  # pylint: disable=protected-access
                {"type": "auto"}
            )
        assert saver is not None


# =========================================================================
# schema.mas_config — validation error branches and helpers
# =========================================================================


class TestMasConfigValidation:
    """Tests for MASConfig validators and helper methods."""

    def test_duplicate_agent_ids_raise(self):
        """Duplicate agent IDs raise a ValueError naming the duplicate."""
        with pytest.raises(ValueError, match="Duplicate agent IDs"):
            MASConfig(
                mas_id="dup",
                name="Test",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[_agent("a"), _agent("a")],
            )

    def test_entry_point_not_found_raises(self):
        """An entry_point that does not match any agent raises ValueError."""
        with pytest.raises(ValueError, match="entry_point 'ghost' not found"):
            MASConfig(
                mas_id="ep",
                name="Test",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[_agent("a")],
                entry_point="ghost",
            )

    def test_channel_source_not_found_raises(self):
        """A channel source that is not an agent (and not 'any') raises."""
        with pytest.raises(ValueError, match="source 'missing' not found"):
            MASConfig(
                mas_id="ch",
                name="Test",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[_agent("a")],
                channels=[
                    Channel(
                        channel_id="c1",
                        protocol=CommunicationProtocol.DIRECT,
                        source="missing",
                        target="a",
                    )
                ],
            )

    def test_channel_target_not_found_raises(self):
        """A channel target that is not an agent (and not 'all') raises."""
        with pytest.raises(ValueError, match="target 'missing' not found"):
            MASConfig(
                mas_id="ch2",
                name="Test",
                workflow_type=WorkflowType.SEQUENTIAL,
                agents=[_agent("a")],
                channels=[
                    Channel(
                        channel_id="c2",
                        protocol=CommunicationProtocol.DIRECT,
                        source="a",
                        target="missing",
                    )
                ],
            )

    def test_workflow_edge_from_agent_not_found_raises(self):
        """A workflow edge from_agent that is unknown raises."""
        with pytest.raises(ValueError, match="from_agent 'ghost' not found"):
            MASConfig(
                mas_id="we",
                name="Test",
                workflow_type=WorkflowType.CUSTOM,
                agents=[_agent("a")],
                workflow_edges=[WorkflowEdge(from_agent="ghost", to_agent="a")],
            )

    def test_workflow_edge_to_agent_not_found_raises(self):
        """A workflow edge to_agent that is unknown raises."""
        with pytest.raises(ValueError, match="to_agent 'ghost' not found"):
            MASConfig(
                mas_id="we2",
                name="Test",
                workflow_type=WorkflowType.CUSTOM,
                agents=[_agent("a")],
                workflow_edges=[WorkflowEdge(from_agent="a", to_agent="ghost")],
            )

    def test_consensus_requires_threshold(self):
        """Consensus workflow without a threshold raises."""
        with pytest.raises(ValueError, match="requires consensus_threshold"):
            MASConfig(
                mas_id="cn",
                name="Test",
                workflow_type=WorkflowType.CONSENSUS,
                consensus_threshold=None,
                agents=[_agent("a"), _agent("b")],
            )

    def test_invalid_consensus_detection_raises(self):
        """An unsupported consensus_detection value raises."""
        with pytest.raises(ValueError, match="Invalid consensus_detection"):
            MASConfig(
                mas_id="cd",
                name="Test",
                workflow_type=WorkflowType.CONSENSUS,
                consensus_threshold=0.5,
                consensus_detection="telepathy",
                agents=[_agent("a"), _agent("b")],
            )

    def test_hierarchical_without_tiers_raises(self):
        """Hierarchical workflow with no tier values raises."""
        with pytest.raises(ValueError, match="requires agents to have tier"):
            MASConfig(
                mas_id="hr",
                name="Test",
                workflow_type=WorkflowType.HIERARCHICAL,
                agents=[_agent("a"), _agent("b")],
            )

    def test_get_agent_returns_none_for_missing(self):
        """get_agent returns None when the ID is not present."""
        config = MASConfig(
            mas_id="ga",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        assert config.get_agent("nope") is None

    def test_get_entry_agent_missing_raises(self):
        """get_entry_agent raises when the resolved entry ID is absent."""
        config = MASConfig(
            mas_id="ea",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        # Force an entry_point that bypasses construction-time validation
        object.__setattr__(config, "entry_point", "ghost")
        with pytest.raises(ValueError, match="Entry agent 'ghost' not found"):
            config.get_entry_agent()

    def test_str_includes_id_count_and_workflow(self):
        """__str__ reports mas_id, agent count, and workflow type."""
        config = MASConfig(
            mas_id="strcfg",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a"), _agent("b")],
        )
        text = str(config)
        assert "strcfg" in text
        assert "2 agents" in text
        assert "sequential" in text


# =========================================================================
# compiler.graph_builder — evaluator error branches
# =========================================================================


class TestSafeConditionEvaluatorErrors:
    """Tests for unsupported-operator error branches in SafeConditionEvaluator."""

    def test_unsupported_comparison_operator(self):
        """An 'is' comparison (not in the supported map) raises ValueError."""
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            SafeConditionEvaluator,
        )

        evaluator = SafeConditionEvaluator({"x": None})
        with pytest.raises(ValueError, match="Invalid condition"):
            evaluator.eval("x is None")

    def test_unsupported_comparison_visit_message(self):
        """visit_Compare raises a descriptive error for an unmapped operator."""
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            SafeConditionEvaluator,
        )

        node = ast.parse("x is None", mode="eval").body
        evaluator = SafeConditionEvaluator({"x": None})
        with pytest.raises(ValueError, match="Unsupported comparison operator"):
            evaluator.visit_Compare(node)

    def test_unsupported_bool_operator(self):
        """visit_BoolOp raises for an operator missing from the bool map."""
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            SafeConditionEvaluator,
        )

        node = ast.parse("a and b", mode="eval").body
        # Replace the op with one not present in _BOOL_OPS.
        node.op = ast.BitAnd()
        evaluator = SafeConditionEvaluator({"a": True, "b": True})
        with pytest.raises(ValueError, match="Unsupported boolean operator"):
            evaluator.visit_BoolOp(node)

    def test_unsupported_unary_operator(self):
        """A bitwise-not unary op (not mapped) raises ValueError."""
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            SafeConditionEvaluator,
        )

        evaluator = SafeConditionEvaluator({"x": 1})
        with pytest.raises(ValueError, match="Invalid condition"):
            evaluator.eval("~x")

    def test_unsupported_binary_operator(self):
        """A matrix-multiply binary op (not mapped) raises ValueError."""
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            SafeConditionEvaluator,
        )

        evaluator = SafeConditionEvaluator({"x": 1, "y": 2})
        with pytest.raises(ValueError, match="Invalid condition"):
            evaluator.eval("x @ y")


# =========================================================================
# compiler.graph_builder — MAS objective node body
# =========================================================================


def _build(config, **kwargs):
    """Build a CompiledMAS via GraphBuilder."""
    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    return GraphBuilder(config, **kwargs).build()


class TestMasObjectiveNodeBody:
    """Tests for the __mas_objective__ injection node behaviour."""

    def _objective_node(self, config):
        from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
            StateGraph,
        )

        captured = {}

        original_add_node = StateGraph.add_node

        def capture_add_node(self_graph, name, fn, *a, **k):
            if name == "__mas_objective__":
                captured["fn"] = fn
            return original_add_node(self_graph, name, fn, *a, **k)

        with patch.object(StateGraph, "add_node", capture_add_node):
            _build(config)
        return captured["fn"]

    def test_objective_node_injects_system_message(self):
        """The objective node prepends a SystemMessage on a fresh state."""
        from langchain_core.messages import (  # pylint: disable=import-outside-toplevel,import-error
            SystemMessage,
        )

        config = MASConfig(
            mas_id="obj",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
            objective="Be helpful and concise.",
        )
        node_fn = self._objective_node(config)
        result = node_fn({"messages": []})
        assert isinstance(result["messages"][0], SystemMessage)
        assert result["messages"][0].content == "Be helpful and concise."

    def test_objective_node_is_idempotent(self):
        """The objective node is a no-op when its SystemMessage is already first."""
        from langchain_core.messages import (  # pylint: disable=import-outside-toplevel,import-error
            SystemMessage,
        )

        config = MASConfig(
            mas_id="obj2",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
            objective="Stay on task.",
        )
        node_fn = self._objective_node(config)
        existing = {"messages": [SystemMessage(content="Stay on task.")]}
        assert node_fn(existing) == {}


# =========================================================================
# compiler.graph_builder — inheritance ImportError branch
# =========================================================================


def test_apply_inheritance_import_error_is_graceful():
    """Inheritance with the integration package unavailable is a no-op."""
    config = MASConfig(
        mas_id="inh",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a", inherit_from_bili_core=True)],
    )
    with patch.dict(sys.modules, {"bili.aether.integration": None}):
        compiled = _build(config)
    # Build still succeeds; the agent is present.
    assert "a" in compiled.agent_nodes


# =========================================================================
# compiler.graph_builder — registry node resolution branches
# =========================================================================


class TestRegistryNodeBranches:
    """Tests for _resolve_registry_node import and build error branches."""

    def _pipeline_with_registry_node(self, node_type="custom_reg"):
        from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
            PipelineEdgeSpec,
            PipelineNodeSpec,
            PipelineSpec,
        )

        return PipelineSpec(
            nodes=[PipelineNodeSpec(node_id="n", node_type=node_type)],
            edges=[PipelineEdgeSpec(from_node="n", to_node="END")],
        )

    def test_registry_import_error_raises_value_error(self):
        """Unknown registry type with langchain_loader unavailable raises ValueError."""
        agent = _agent("p", pipeline=self._pipeline_with_registry_node())
        config = MASConfig(
            mas_id="reg",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        with patch.dict(sys.modules, {"bili.iris.loaders.langchain_loader": None}):
            with pytest.raises(ValueError, match="is not available"):
                _build(config)

    def test_unknown_type_error_message_handles_second_import_failure(self):
        """The unknown-type error message tolerates the registry import failing.

        The first registry import yields a registry that lacks the requested
        type, then the error-message builder re-imports the registry to list
        available types. This exercises the ImportError guard on that second
        import (graph_builder lines 534-535) where the module disappears.
        """
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            GraphBuilder,
        )

        config = MASConfig(
            mas_id="reg_msg",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        builder = GraphBuilder(config, custom_node_registry={"other": lambda: None})

        node_spec = MagicMock()
        node_spec.node_id = "n"
        node_spec.node_type = "ghost_type"

        # A module that supplies an empty registry on the first import and then
        # raises ImportError on the second access (the error-message path).
        class _FlakyLoader(types.ModuleType):
            def __init__(self):
                super().__init__("bili.iris.loaders.langchain_loader")
                self._calls = 0

            def __getattr__(self, name):
                if name == "GRAPH_NODE_REGISTRY":
                    self._calls += 1
                    if self._calls == 1:
                        return {}
                    raise ImportError("registry vanished")
                raise AttributeError(name)

        flaky = _FlakyLoader()
        with patch.dict(sys.modules, {"bili.iris.loaders.langchain_loader": flaky}):
            with pytest.raises(ValueError, match="unknown registry type 'ghost_type'"):
                builder._resolve_registry_node(  # pylint: disable=protected-access
                    node_spec, _agent("a")
                )

    def test_custom_registry_node_factory_is_node_instance(self):
        """A registry factory that is already a Node instance is used directly."""
        from bili.iris.graph_builder.classes.node import (  # pylint: disable=import-outside-toplevel
            Node,
        )

        def builder(**_kwargs):
            return lambda state: {
                "messages": [],
                "current_agent": "n",
                "agent_outputs": {},
            }

        node_instance = Node("custom_reg", builder)

        agent = _agent("p", pipeline=self._pipeline_with_registry_node())
        config = MASConfig(
            mas_id="reg2",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        compiled = _build(config, custom_node_registry={"custom_reg": node_instance})
        assert "p" in compiled.agent_nodes

    def test_registry_node_build_failure_raises_value_error(self):
        """A builder that raises is wrapped in a ValueError naming the node."""
        from bili.iris.graph_builder.classes.node import (  # pylint: disable=import-outside-toplevel
            Node,
        )

        def failing_builder(**_kwargs):
            raise RuntimeError("builder blew up")

        factory = lambda: Node("custom_reg", failing_builder)

        agent = _agent("p", pipeline=self._pipeline_with_registry_node())
        config = MASConfig(
            mas_id="reg3",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[agent],
        )
        with pytest.raises(ValueError, match="Failed to build pipeline node"):
            _build(config, custom_node_registry={"custom_reg": factory})


def test_build_registry_node_kwargs_resolves_llm_and_tools():
    """_build_registry_node_kwargs resolves the parent LLM and tools."""
    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )
    from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
        PipelineNodeSpec,
    )

    agent = _agent("p", model_name="gpt-4o", tools=["weather_api_tool"])
    config = MASConfig(
        mas_id="kw",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
    )
    builder = GraphBuilder(config)
    node_spec = PipelineNodeSpec(node_id="n", node_type="react_agent")

    with patch(
        "bili.aether.compiler.llm_resolver.create_llm", return_value="LLM"
    ), patch("bili.aether.compiler.llm_resolver.resolve_tools", return_value=["TOOL"]):
        kwargs = (
            builder._build_registry_node_kwargs(  # pylint: disable=protected-access
                agent, node_spec
            )
        )

    assert kwargs["llm_model"] == "LLM"
    assert kwargs["tools"] == ["TOOL"]


def test_build_registry_node_kwargs_llm_failure_is_logged():
    """A failure resolving the parent LLM does not propagate out of kwargs build."""
    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )
    from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
        PipelineNodeSpec,
    )

    agent = _agent("p", model_name="gpt-4o")
    config = MASConfig(
        mas_id="kw2",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
    )
    builder = GraphBuilder(config)
    node_spec = PipelineNodeSpec(node_id="n", node_type="react_agent")

    with patch(
        "bili.aether.compiler.llm_resolver.create_llm",
        side_effect=RuntimeError("resolve failed"),
    ):
        kwargs = (
            builder._build_registry_node_kwargs(  # pylint: disable=protected-access
                agent, node_spec
            )
        )

    # No llm_model key because resolution failed, but build did not raise.
    assert "llm_model" not in kwargs


# =========================================================================
# compiler.graph_builder — pipeline conditional router execution
# =========================================================================


def test_pipeline_conditional_router_executes_branches():
    """The pipeline conditional router picks branches by condition at runtime."""
    from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
        PipelineStateField,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="check",
                node_type="agent",
                agent_spec={
                    "agent_id": "checker",
                    "role": "checker",
                    "objective": "Decide which branch to take next",
                },
            ),
            PipelineNodeSpec(
                node_id="high",
                node_type="agent",
                agent_spec={
                    "agent_id": "high_h",
                    "role": "high",
                    "objective": "Handle the high-score branch path",
                },
            ),
            PipelineNodeSpec(
                node_id="low",
                node_type="agent",
                agent_spec={
                    "agent_id": "low_h",
                    "role": "low",
                    "objective": "Handle the low-score branch path",
                },
            ),
        ],
        edges=[
            PipelineEdgeSpec(
                from_node="check",
                to_node="high",
                condition="state.score > 0.5",
                label="hi",
            ),
            PipelineEdgeSpec(from_node="check", to_node="low", label="lo"),
            PipelineEdgeSpec(from_node="high", to_node="END"),
            PipelineEdgeSpec(from_node="low", to_node="END"),
        ],
        state_fields=[
            PipelineStateField(
                name="score", type="float", default=0.0, reducer="replace"
            )
        ],
    )
    agent = _agent("p", pipeline=pipeline)
    config = MASConfig(
        mas_id="cond",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
    )
    compiled = _build(config)
    graph = compiled.graph.compile(checkpointer=None)

    # Custom state field carries the routing value into the inner pipeline.
    result = graph.invoke(
        {"messages": [], "agent_outputs": {}, "mas_id": "cond", "score": 0.9}
    )
    inner = result["agent_outputs"]["p"]["pipeline_outputs"]
    # High branch should have run, low branch should not.
    assert "high_h" in inner
    assert "low_h" not in inner


def _capture_pipeline_router(builder, from_node, conditional, unconditional, end="END"):
    """Build pipeline conditional edges and capture the router callable."""
    captured = {}

    class _FakeGraph:
        def add_conditional_edges(self, src, router, path_map):
            captured["router"] = router
            captured["path_map"] = path_map

    builder._add_pipeline_conditional_edges(  # pylint: disable=protected-access
        _FakeGraph(), from_node, conditional, unconditional, end
    )
    return captured["router"]


class TestPipelineConditionalRouterRuntime:
    """Tests for the runtime router built by _add_pipeline_conditional_edges."""

    def _builder(self):
        from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
            GraphBuilder,
        )

        config = MASConfig(
            mas_id="pcr",
            name="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        return GraphBuilder(config)

    def test_condition_failure_falls_back_to_unconditional(self):
        """A failing condition logs a warning and falls back to the unconditional edge."""
        from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
            PipelineEdgeSpec,
        )

        cond = PipelineEdgeSpec(
            from_node="x", to_node="a", condition="state.missing == 1", label="ca"
        )
        uncond = PipelineEdgeSpec(from_node="x", to_node="b", label="cb")
        router = _capture_pipeline_router(self._builder(), "x", [cond], [uncond])
        # Missing field raises ValueError inside the router; it falls back to "cb".
        assert router({"present": 1}) == "cb"

    def test_all_conditional_uses_last_edge_fallback(self):
        """When every edge is conditional and none match, the last edge is used."""
        from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
            PipelineEdgeSpec,
        )

        c1 = PipelineEdgeSpec(
            from_node="x", to_node="a", condition="state.flag == 1", label="ca"
        )
        c2 = PipelineEdgeSpec(
            from_node="x", to_node="b", condition="state.flag == 2", label="cb"
        )
        router = _capture_pipeline_router(self._builder(), "x", [c1, c2], [])
        # Neither condition matches, so the last edge's label is returned.
        assert router({"flag": 9}) == "cb"


def test_pipeline_carries_custom_state_default():
    """Custom state fields absent from outer state fall back to their default."""
    from bili.aether.schema.pipeline_spec import (  # pylint: disable=import-outside-toplevel
        PipelineEdgeSpec,
        PipelineNodeSpec,
        PipelineSpec,
        PipelineStateField,
    )

    pipeline = PipelineSpec(
        nodes=[
            PipelineNodeSpec(
                node_id="only",
                node_type="agent",
                agent_spec={
                    "agent_id": "solo",
                    "role": "proc",
                    "objective": "Single processing step in the pipeline",
                },
            ),
        ],
        edges=[PipelineEdgeSpec(from_node="only", to_node="END")],
        state_fields=[
            PipelineStateField(name="counter", type="int", default=7, reducer="replace")
        ],
    )
    agent = _agent("p", pipeline=pipeline)
    config = MASConfig(
        mas_id="def",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[agent],
    )
    compiled = _build(config)
    node_fn = compiled.agent_nodes["p"]

    captured = {}
    real_invoke = None

    # Wrap the inner subgraph invoke to capture the inner state it receives.
    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    builder = GraphBuilder(config)
    # Rebuild and patch the wrapper directly for inspection.
    mock_subgraph = MagicMock()
    mock_subgraph.invoke.return_value = {
        "messages": [],
        "current_agent": "solo",
        "agent_outputs": {},
    }
    wrapper = builder._wrap_pipeline_as_agent_node(  # pylint: disable=protected-access
        mock_subgraph, agent
    )
    # Outer state does NOT include "counter"; default should be injected.
    wrapper({"messages": [], "agent_outputs": {}})
    inner_state = mock_subgraph.invoke.call_args[0][0]
    assert inner_state["counter"] == 7
    assert node_fn is not None
    assert real_invoke is None
    assert captured == {}


# =========================================================================
# compiler.graph_builder — workflow edge / hierarchical / supervisor branches
# =========================================================================


def test_sequential_uses_explicit_edges_when_present():
    """A sequential workflow with workflow_edges routes through them."""
    config = MASConfig(
        mas_id="seq_edges",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[WorkflowEdge(from_agent="a", to_agent="b")],
    )
    compiled = _build(config)
    # _build_from_explicit_edges appends an END edge since none was declared.
    graph = compiled.graph.compile(checkpointer=None)
    result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "x"})
    assert "a" in result["agent_outputs"]
    assert "b" in result["agent_outputs"]


def test_explicit_edges_with_end_skip_synthetic_end():
    """When an explicit END edge exists, no synthetic END edge is appended."""
    config = MASConfig(
        mas_id="seq_edges_end",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="END"),
        ],
    )
    compiled = _build(config)
    graph = compiled.graph.compile(checkpointer=None)
    result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "x"})
    assert "b" in result["agent_outputs"]


def test_hierarchical_without_tiers_falls_back_to_sequential():
    """A hierarchical config whose agents lack tiers falls back to sequential."""
    config = MASConfig(
        mas_id="hier_fb",
        name="Test",
        workflow_type=WorkflowType.HIERARCHICAL,
        hierarchical_voting=True,
        agents=[_agent("a", tier=1)],
    )
    # Drop the tier so _build_hierarchical sees no tiers and uses sequential.
    object.__setattr__(config.agents[0], "tier", None)
    compiled = _build(config)
    graph = compiled.graph.compile(checkpointer=None)
    result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "x"})
    assert "a" in result["agent_outputs"]


def test_supervisor_router_unknown_next_agent_routes_to_end():
    """The supervisor router returns END for an unrecognised next_agent."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    config = MASConfig(
        mas_id="sup_router",
        name="Test",
        workflow_type=WorkflowType.SUPERVISOR,
        entry_point="boss",
        agents=[_agent("boss", is_supervisor=True), _agent("w1")],
    )

    captured = {}
    original = StateGraph.add_conditional_edges

    def capture(self_graph, source, router, path_map, *a, **k):
        if source == "boss":
            captured["router"] = router
        return original(self_graph, source, router, path_map, *a, **k)

    with patch.object(StateGraph, "add_conditional_edges", capture):
        GraphBuilder(config).build()

    router = captured["router"]
    assert router({"next_agent": "ghost"}) == "END"
    assert router({"next_agent": "w1"}) == "w1"


# =========================================================================
# compiler.graph_builder — custom edges: START skip, conditional router, fan-out
# =========================================================================


def test_custom_skips_explicit_start_edge():
    """An explicit from_agent='START' edge is skipped to avoid the sentinel mismatch."""
    config = MASConfig(
        mas_id="custom_start",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        entry_point="a",
        agents=[_agent("a"), _agent("b")],
        workflow_edges=[
            WorkflowEdge(from_agent="START", to_agent="a"),
            WorkflowEdge(from_agent="a", to_agent="b"),
            WorkflowEdge(from_agent="b", to_agent="END"),
        ],
    )
    compiled = _build(config)
    graph = compiled.graph.compile(checkpointer=None)
    result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "x"})
    assert "a" in result["agent_outputs"]
    assert "b" in result["agent_outputs"]


def test_custom_conditional_router_runtime_branches():
    """The custom conditional router selects branches and falls back correctly."""
    config = MASConfig(
        mas_id="custom_cond",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(
                from_agent="a",
                to_agent="b",
                condition="state.current_agent == 'a'",
                label="to_b",
            ),
            WorkflowEdge(from_agent="a", to_agent="c", label="to_c"),
            WorkflowEdge(from_agent="b", to_agent="END"),
            WorkflowEdge(from_agent="c", to_agent="END"),
        ],
    )
    compiled = _build(config)
    graph = compiled.graph.compile(checkpointer=None)
    result = graph.invoke({"messages": [], "agent_outputs": {}, "mas_id": "x"})
    # Condition is true after 'a' runs, so 'b' executes (not 'c').
    assert "b" in result["agent_outputs"]
    assert "c" not in result["agent_outputs"]


def test_custom_conditional_router_handles_eval_failure():
    """A condition that references a missing field logs a warning and falls back."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    config = MASConfig(
        mas_id="custom_fail",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(
                from_agent="a",
                to_agent="b",
                condition="state.nonexistent == 1",
                label="cond_b",
            ),
            WorkflowEdge(from_agent="a", to_agent="c", label="fallback_c"),
            WorkflowEdge(from_agent="b", to_agent="END"),
            WorkflowEdge(from_agent="c", to_agent="END"),
        ],
    )

    captured = {}
    original = StateGraph.add_conditional_edges

    def capture(self_graph, source, router, path_map, *a, **k):
        if source == "a":
            captured["router"] = router
        return original(self_graph, source, router, path_map, *a, **k)

    with patch.object(StateGraph, "add_conditional_edges", capture):
        GraphBuilder(config).build()

    router = captured["router"]
    # Condition eval fails (missing field) so it falls back to the unconditional edge.
    assert router({"current_agent": "a"}) == "fallback_c"


def test_custom_conditional_router_all_conditional_last_edge_fallback():
    """When all custom edges are conditional and none match, the last edge wins."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    config = MASConfig(
        mas_id="custom_all_cond",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(
                from_agent="a",
                to_agent="b",
                condition="state.current_agent == 'never'",
                label="to_b",
            ),
            WorkflowEdge(
                from_agent="a",
                to_agent="c",
                condition="state.current_agent == 'nope'",
                label="to_c",
            ),
            WorkflowEdge(from_agent="b", to_agent="END"),
            WorkflowEdge(from_agent="c", to_agent="END"),
        ],
    )

    captured = {}
    original = StateGraph.add_conditional_edges

    def capture(self_graph, source, router, path_map, *a, **k):
        if source == "a":
            captured["router"] = router
        return original(self_graph, source, router, path_map, *a, **k)

    with patch.object(StateGraph, "add_conditional_edges", capture):
        GraphBuilder(config).build()

    router = captured["router"]
    # No condition matches and there is no unconditional edge, so the last
    # edge's label is returned as a final fallback.
    assert router({"current_agent": "a"}) == "to_c"


def test_sequential_entry_not_in_agent_ids_uses_natural_order():
    """When the entry agent is not in the list, sequential keeps natural order."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    config = MASConfig(
        mas_id="seq_foreign_entry",
        name="Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[_agent("a"), _agent("b")],
    )
    builder = GraphBuilder(config)

    # Force get_entry_agent to return an agent whose ID is not in the list,
    # exercising the natural-order fallback branch.
    foreign = _agent("foreign")
    edges = []
    original_add_edge = StateGraph.add_edge

    def capture_add_edge(self_graph, src, dst, *a, **k):
        edges.append((src, dst))
        return original_add_edge(self_graph, src, dst, *a, **k)

    with patch.object(
        type(builder._config),  # pylint: disable=protected-access
        "get_entry_agent",
        lambda _self: foreign,
    ), patch.object(StateGraph, "add_edge", capture_add_edge):
        builder.build()

    # The entry edge should target 'a' (natural list order preserved because
    # the foreign entry ID is not present in the agent list).
    start_targets = [dst for src, dst in edges if "start" in str(src).lower()]
    assert "a" in start_targets


def test_custom_fan_out_router_returns_all_targets():
    """Multiple unconditional edges from one source fan out to all targets."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    from bili.aether.compiler.graph_builder import (  # pylint: disable=import-outside-toplevel
        GraphBuilder,
    )

    config = MASConfig(
        mas_id="custom_fan",
        name="Test",
        workflow_type=WorkflowType.CUSTOM,
        agents=[_agent("a"), _agent("b"), _agent("c")],
        workflow_edges=[
            WorkflowEdge(from_agent="a", to_agent="b", label="fb"),
            WorkflowEdge(from_agent="a", to_agent="c", label="fc"),
            WorkflowEdge(from_agent="b", to_agent="END"),
            WorkflowEdge(from_agent="c", to_agent="END"),
        ],
    )

    captured = {}
    original = StateGraph.add_conditional_edges

    def capture(self_graph, source, router, path_map, *a, **k):
        if source == "a":
            captured["router"] = router
        return original(self_graph, source, router, path_map, *a, **k)

    with patch.object(StateGraph, "add_conditional_edges", capture):
        GraphBuilder(config).build()

    router = captured["router"]
    targets = router({})
    assert set(targets) == {"fb", "fc"}


# =========================================================================
# compiler.graph_builder — consensus checker / router voting logic
# =========================================================================


def _consensus_checker(config):
    """Capture the __consensus_checker__ node function from a consensus build."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    captured = {}
    original_add_node = StateGraph.add_node

    def capture_add_node(self_graph, name, fn, *a, **k):
        if name == "__consensus_checker__":
            captured["fn"] = fn
        return original_add_node(self_graph, name, fn, *a, **k)

    with patch.object(StateGraph, "add_node", capture_add_node):
        _build(config)
    return captured["fn"]


class TestConsensusVoting:
    """Tests for the consensus checker's vote-extraction and threshold logic."""

    def _config(self, **kwargs):
        defaults = dict(
            mas_id="cons_vote",
            name="Test",
            workflow_type=WorkflowType.CONSENSUS,
            consensus_threshold=0.5,
            max_consensus_rounds=3,
        )
        defaults.update(kwargs)
        return MASConfig(**defaults)

    def test_consensus_reached_from_message_votes(self):
        """Votes parsed from 'vote:' message text drive a reached consensus."""
        config = self._config(
            agents=[_agent("a"), _agent("b")],
        )
        checker = _consensus_checker(config)
        state = {
            "current_round": 0,
            "agent_outputs": {
                "a": {"message": "My vote: yes, I agree"},
                "b": {"message": "Decision: yes overall"},
            },
        }
        result = checker(state)
        assert result["votes"] == {"a": "yes", "b": "yes"}
        assert result["consensus_reached"] is True
        assert result["current_round"] == 1

    def test_consensus_uses_parsed_vote_field(self):
        """A configured consensus_vote_field is read from parsed JSON output."""
        config = self._config(
            agents=[
                _agent(
                    "a",
                    output_format="json",
                    consensus_vote_field="decision",
                ),
                _agent(
                    "b",
                    output_format="json",
                    consensus_vote_field="decision",
                ),
            ],
        )
        checker = _consensus_checker(config)
        state = {
            "current_round": 0,
            "agent_outputs": {
                "a": {"parsed": {"decision": "approve"}, "message": ""},
                "b": {"parsed": {"decision": "approve"}, "message": ""},
            },
        }
        result = checker(state)
        assert result["votes"] == {"a": "approve", "b": "approve"}
        assert result["consensus_reached"] is True

    def test_consensus_round_limit_forces_completion(self):
        """With no votes, consensus is forced once the round limit is reached."""
        config = self._config(
            agents=[_agent("a"), _agent("b")],
            max_consensus_rounds=1,
        )
        checker = _consensus_checker(config)
        # No vote signals in the outputs.
        state = {
            "current_round": 0,
            "agent_outputs": {"a": {"message": "thinking"}, "b": {"message": "hmm"}},
        }
        result = checker(state)
        assert result["votes"] == {}
        assert result["consensus_reached"] is True
        assert result["current_round"] == 1


def _consensus_router(config):
    """Capture the consensus router from add_conditional_edges."""
    from langgraph.graph import (  # pylint: disable=import-outside-toplevel,import-error
        StateGraph,
    )

    captured = {}
    original = StateGraph.add_conditional_edges

    def capture(self_graph, source, router, path_map, *a, **k):
        if source == "__consensus_checker__":
            captured["router"] = router
        return original(self_graph, source, router, path_map, *a, **k)

    with patch.object(StateGraph, "add_conditional_edges", capture):
        _build(config)
    return captured["router"]


class TestConsensusRouter:
    """Tests for the consensus router continue/end decision."""

    def _config(self):
        return MASConfig(
            mas_id="cons_router",
            name="Test",
            workflow_type=WorkflowType.CONSENSUS,
            consensus_threshold=0.5,
            max_consensus_rounds=2,
            agents=[_agent("a"), _agent("b")],
        )

    def test_router_ends_when_consensus_reached(self):
        """The router ends when consensus_reached is True."""
        router = _consensus_router(self._config())
        assert router({"consensus_reached": True, "current_round": 0}) == "end"

    def test_router_ends_at_round_limit(self):
        """The router ends when the current round meets max_consensus_rounds."""
        router = _consensus_router(self._config())
        assert router({"consensus_reached": False, "current_round": 2}) == "end"

    def test_router_continues_below_limit(self):
        """The router continues when no consensus and below the round limit."""
        router = _consensus_router(self._config())
        assert router({"consensus_reached": False, "current_round": 1}) == "continue"
