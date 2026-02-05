"""Tests for bili-core integration (inheritance resolution) — Task 7.

Covers:
    - Role defaults registry
    - Inheritance resolution logic (per-flag, priority, selectivity)
    - apply_inheritance_to_all batch helper
    - State integration extension point

All tests run under the isolated test runner (``run_tests.py``) which
stubs the top-level ``bili`` package.  No actual bili-core dependencies
(firebase, torch, LLM providers) are needed.
"""

# pylint: disable=missing-function-docstring

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
