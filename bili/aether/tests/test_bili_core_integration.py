"""Tests for bili-core integration — Tasks 7 & 7b.

Covers:
    - Role defaults registry
    - Inheritance resolution logic (per-flag, priority, selectivity)
    - apply_inheritance_to_all batch helper
    - State integration extension point
    - Checkpointer factory (type dispatch, fallbacks, aliases)
    - Per-agent middleware fields on AgentSpec

All tests run under the isolated test runner (``run_tests.py``) which
stubs the top-level ``bili`` package.  No actual bili-core dependencies
(firebase, torch, LLM providers) are needed.
"""

# pylint: disable=missing-function-docstring

from bili.aether.integration.checkpointer_factory import (
    _TYPE_ALIASES,
    create_checkpointer_from_config,
)
from bili.aether.integration.inheritance import (
    apply_inheritance,
    apply_inheritance_to_all,
)
from bili.aether.integration.role_registry import (
    ROLE_DEFAULTS,
    RoleDefaults,
    get_role_defaults,
    register_role_defaults,
)
from bili.aether.integration.state_integration import get_inheritance_state_fields
from bili.aether.schema import AgentSpec

# ======================================================================
# Helpers
# ======================================================================


def _make_agent(**kwargs) -> AgentSpec:
    """Create a minimal AgentSpec for testing."""
    defaults = {
        "agent_id": "test_agent",
        "role": "researcher",
        "objective": "Test objective for this agent",
        "inherit_from_bili_core": True,
    }
    defaults.update(kwargs)
    return AgentSpec(**defaults)


# ======================================================================
# Role Registry tests
# ======================================================================


class TestRoleRegistry:
    """Tests for the role defaults registry."""

    def test_known_roles_have_defaults(self):
        expected_roles = [
            "content_reviewer",
            "policy_expert",
            "judge",
            "appeals_specialist",
            "community_manager",
            "researcher",
            "analyst",
            "fact_checker",
            "code_reviewer",
            "security_auditor",
            "documentation_writer",
            "support_agent",
            "escalation_specialist",
            "supervisor",
            "voter",
            "advocate",
        ]
        for role in expected_roles:
            assert (
                get_role_defaults(role) is not None
            ), f"Role '{role}' missing from ROLE_DEFAULTS"

    def test_unknown_role_returns_none(self):
        assert get_role_defaults("nonexistent_role_xyz") is None

    def test_register_custom_role(self):
        register_role_defaults(
            "test_custom_role_7",
            RoleDefaults(
                system_prompt="Custom prompt for testing",
                tools=["mock_tool"],
            ),
        )
        defaults = get_role_defaults("test_custom_role_7")
        assert defaults is not None
        assert defaults.system_prompt == "Custom prompt for testing"
        assert "mock_tool" in defaults.tools

    def test_all_roles_have_system_prompts(self):
        for role, defaults in ROLE_DEFAULTS.items():
            assert (
                defaults.system_prompt is not None
            ), f"Role '{role}' missing system_prompt"
            assert (
                len(defaults.system_prompt) > 20
            ), f"Role '{role}' system_prompt too short"

    def test_role_defaults_dataclass_defaults(self):
        rd = RoleDefaults()
        assert rd.system_prompt is None
        assert rd.model_name is None
        assert rd.temperature is None
        assert rd.tools == []
        assert rd.capabilities == []


# ======================================================================
# Inheritance Logic tests
# ======================================================================


class TestInheritance:
    """Tests for apply_inheritance()."""

    def test_no_inheritance_returns_same_object(self):
        agent = _make_agent(inherit_from_bili_core=False)
        result = apply_inheritance(agent)
        assert result is agent

    def test_system_prompt_inherited_when_none(self):
        agent = _make_agent(system_prompt=None)
        result = apply_inheritance(agent)
        assert result.system_prompt is not None
        assert "research" in result.system_prompt.lower()

    def test_system_prompt_not_overridden(self):
        agent = _make_agent(system_prompt="My custom prompt")
        result = apply_inheritance(agent)
        assert result.system_prompt == "My custom prompt"

    def test_tools_merged_additively(self):
        agent = _make_agent(tools=["mock_tool"])
        result = apply_inheritance(agent)
        assert "serp_api_tool" in result.tools
        assert "mock_tool" in result.tools

    def test_tools_deduplicated(self):
        agent = _make_agent(tools=["serp_api_tool"])
        result = apply_inheritance(agent)
        assert result.tools.count("serp_api_tool") == 1

    def test_tools_registry_first(self):
        agent = _make_agent(tools=["mock_tool"])
        result = apply_inheritance(agent)
        assert result.tools.index("serp_api_tool") < result.tools.index("mock_tool")

    def test_temperature_inherited_when_default(self):
        agent = _make_agent(temperature=0.0)
        result = apply_inheritance(agent)
        assert result.temperature == 0.5  # researcher default

    def test_temperature_not_overridden_when_set(self):
        agent = _make_agent(temperature=0.8)
        result = apply_inheritance(agent)
        assert result.temperature == 0.8

    def test_model_name_inherited_when_none(self):
        register_role_defaults(
            "test_model_role",
            RoleDefaults(
                system_prompt="Test prompt for model role",
                model_name="gpt-4",
            ),
        )
        agent = _make_agent(role="test_model_role", model_name=None)
        result = apply_inheritance(agent)
        assert result.model_name == "gpt-4"

    def test_model_name_not_overridden(self):
        register_role_defaults(
            "test_model_role_2",
            RoleDefaults(
                system_prompt="Test prompt",
                model_name="gpt-4",
            ),
        )
        agent = _make_agent(
            role="test_model_role_2",
            model_name="claude-sonnet-3-5-20241022",
        )
        result = apply_inheritance(agent)
        assert result.model_name == "claude-sonnet-3-5-20241022"

    def test_capabilities_merged(self):
        agent = _make_agent(capabilities=["custom_cap"])
        result = apply_inheritance(agent)
        assert "custom_cap" in result.capabilities
        assert "web_search" in result.capabilities

    def test_unknown_role_returns_agent_unchanged(self):
        agent = _make_agent(role="completely_unknown_role")
        result = apply_inheritance(agent)
        assert result is agent

    def test_selective_inherit_system_prompt_only(self):
        agent = _make_agent(
            inherit_tools=False,
            inherit_llm_config=False,
            inherit_memory=False,
            inherit_checkpoint=False,
        )
        result = apply_inheritance(agent)
        assert result.system_prompt is not None
        assert result.tools == []
        assert result.temperature == 0.0

    def test_selective_inherit_tools_only(self):
        agent = _make_agent(
            inherit_system_prompt=False,
            inherit_llm_config=False,
        )
        result = apply_inheritance(agent)
        assert result.system_prompt is None
        assert "serp_api_tool" in result.tools

    def test_original_agent_not_mutated(self):
        agent = _make_agent(tools=["mock_tool"])
        original_tools = list(agent.tools)
        _ = apply_inheritance(agent)
        assert agent.tools == original_tools


# ======================================================================
# Batch helper tests
# ======================================================================


class TestApplyInheritanceToAll:
    """Tests for apply_inheritance_to_all()."""

    def test_mixed_agents(self):
        agents = [
            AgentSpec(
                agent_id="a1",
                role="researcher",
                objective="Research something useful",
                inherit_from_bili_core=True,
            ),
            AgentSpec(
                agent_id="a2",
                role="analyst",
                objective="Analyse something useful",
                inherit_from_bili_core=False,
            ),
        ]
        results = apply_inheritance_to_all(agents)
        assert len(results) == 2
        assert results[0].system_prompt is not None
        assert results[1].system_prompt is None

    def test_empty_list(self):
        assert apply_inheritance_to_all([]) == []


# ======================================================================
# State integration tests
# ======================================================================


class TestStateIntegration:  # pylint: disable=too-few-public-methods
    """Tests for state integration extension point."""

    def test_returns_empty_when_no_inheritance(self):
        agents = [
            AgentSpec(
                agent_id="a1",
                role="analyst",
                objective="Analyse something useful",
                inherit_from_bili_core=False,
            ),
        ]
        assert not get_inheritance_state_fields(agents)

    def test_returns_dict_when_inheritance_active(self):
        agents = [
            AgentSpec(
                agent_id="a1",
                role="researcher",
                objective="Research something useful",
                inherit_from_bili_core=True,
            ),
        ]
        result = get_inheritance_state_fields(agents)
        assert isinstance(result, dict)


# ======================================================================
# Checkpointer Factory tests
# ======================================================================


class TestCheckpointerFactory:
    """Tests for create_checkpointer_from_config."""

    def test_memory_type_returns_checkpointer(self):
        result = create_checkpointer_from_config({"type": "memory"})
        assert result is not None
        assert "MemorySaver" in type(result).__name__

    def test_unknown_type_falls_back_to_memory(self):
        result = create_checkpointer_from_config({"type": "sqlite"})
        assert result is not None
        assert "MemorySaver" in type(result).__name__

    def test_empty_config_defaults_to_memory(self):
        result = create_checkpointer_from_config({})
        assert result is not None

    def test_postgres_falls_back_when_unavailable(self):
        # Without POSTGRES_CONNECTION_STRING env var, should fall back
        result = create_checkpointer_from_config({"type": "postgres"})
        assert result is not None

    def test_mongo_falls_back_when_unavailable(self):
        # Without MONGO_CONNECTION_STRING env var, should fall back
        result = create_checkpointer_from_config({"type": "mongo"})
        assert result is not None

    def test_auto_type_returns_checkpointer(self):
        # auto without DB env vars should fall back to memory
        result = create_checkpointer_from_config({"type": "auto"})
        assert result is not None

    def test_pg_alias_maps_to_postgres(self):
        assert _TYPE_ALIASES["pg"] == "postgres"
        assert _TYPE_ALIASES["postgres"] == "postgres"

    def test_mongodb_alias_maps_to_mongo(self):
        assert _TYPE_ALIASES["mongodb"] == "mongo"
        assert _TYPE_ALIASES["mongo"] == "mongo"

    def test_keep_last_n_accepted(self):
        result = create_checkpointer_from_config({"type": "memory", "keep_last_n": 10})
        assert result is not None

    def test_case_insensitive_type(self):
        result = create_checkpointer_from_config({"type": "MEMORY"})
        assert result is not None
        assert "MemorySaver" in type(result).__name__


# ======================================================================
# AgentSpec Middleware field tests
# ======================================================================


class TestAgentSpecMiddleware:
    """Tests for middleware fields on AgentSpec."""

    def test_middleware_defaults_to_empty(self):
        agent = _make_agent()
        assert agent.middleware == []
        assert agent.middleware_params == {}

    def test_middleware_accepts_valid_names(self):
        agent = _make_agent(
            middleware=["summarization", "model_call_limit"],
            middleware_params={
                "summarization": {"max_tokens_before_summary": 4000},
                "model_call_limit": {"run_limit": 10},
            },
        )
        assert len(agent.middleware) == 2
        assert "summarization" in agent.middleware
        assert agent.middleware_params["model_call_limit"]["run_limit"] == 10

    def test_middleware_preserved_in_model_copy(self):
        agent = _make_agent(middleware=["summarization"])
        copy = agent.model_copy(update={"temperature": 0.5})
        assert copy.middleware == ["summarization"]
        assert copy.temperature == 0.5

    def test_original_not_mutated_when_middleware_set(self):
        agent = _make_agent(middleware=["summarization"])
        original_mw = list(agent.middleware)
        _ = agent.model_copy(update={"middleware": ["model_call_limit"]})
        assert agent.middleware == original_mw

    def test_middleware_in_yaml_round_trip(self):
        agent = _make_agent(
            middleware=["model_call_limit"],
            middleware_params={"model_call_limit": {"run_limit": 5}},
        )
        data = agent.model_dump()
        restored = AgentSpec(**data)
        assert restored.middleware == ["model_call_limit"]
        assert restored.middleware_params["model_call_limit"]["run_limit"] == 5
