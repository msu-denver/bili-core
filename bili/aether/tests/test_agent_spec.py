"""
Tests for AgentSpec schema.

These tests verify the domain-agnostic design where roles and capabilities
are free-form strings, not restrictive enums.
"""

import pytest
from pydantic import ValidationError

from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.enums import OutputFormat


class TestAgentSpecBasic:
    """Basic AgentSpec creation tests."""

    def test_create_simple_agent(self):
        """Test creating a simple agent with minimal fields."""
        agent = AgentSpec(
            agent_id="test_agent",
            role="test_role",
            objective="Test the agent creation process",
        )
        assert agent.agent_id == "test_agent"
        assert agent.role == "test_role"
        assert agent.temperature == 0.7  # default
        assert agent.capabilities == []
        assert agent.tools == []

    def test_create_agent_with_custom_role(self):
        """Test that any role string is accepted (domain-agnostic)."""
        # Content moderation domain
        agent1 = AgentSpec(
            agent_id="mod1",
            role="content_reviewer",
            objective="Review content for violations",
        )
        assert agent1.role == "content_reviewer"

        # Research domain
        agent2 = AgentSpec(
            agent_id="res1",
            role="research_analyst",
            objective="Analyze research papers",
        )
        assert agent2.role == "research_analyst"

        # Completely custom domain
        agent3 = AgentSpec(
            agent_id="custom1",
            role="quantum_optimizer",
            objective="Optimize quantum circuits",
        )
        assert agent3.role == "quantum_optimizer"

    def test_create_agent_with_capabilities(self):
        """Test that any capability strings are accepted."""
        agent = AgentSpec(
            agent_id="capable_agent",
            role="analyst",
            objective="Analyze data with multiple capabilities",
            capabilities=["custom_skill_1", "web_search", "data_analysis"],
        )
        assert len(agent.capabilities) == 3
        assert "custom_skill_1" in agent.capabilities
        assert agent.has_capability("web_search")
        assert not agent.has_capability("nonexistent")

    def test_create_agent_with_all_fields(self):
        """Test creating an agent with all optional fields."""
        agent = AgentSpec(
            agent_id="full_agent",
            role="specialist",
            objective="Handle specialized tasks with full configuration",
            system_prompt="You are a specialized agent.",
            temperature=0.5,
            model_name="gpt-4",
            max_tokens=2000,
            capabilities=["analysis", "synthesis"],
            tools=["tool1", "tool2"],
            output_format=OutputFormat.JSON,
            tier=2,
            voting_weight=1.5,
            is_supervisor=False,
            tags=["test", "full"],
            metadata={"custom_field": "value"},
        )
        assert agent.temperature == 0.5
        assert agent.model_name == "gpt-4"
        assert agent.tier == 2
        assert agent.voting_weight == 1.5
        assert "test" in agent.tags


class TestAgentSpecValidation:
    """Validation tests for AgentSpec."""

    def test_reject_empty_agent_id(self):
        """Test that empty agent_id is rejected."""
        with pytest.raises(ValidationError):
            AgentSpec(
                agent_id="",
                role="test",
                objective="Test objective here",
            )

    def test_reject_invalid_agent_id_characters(self):
        """Test that invalid characters in agent_id are rejected."""
        with pytest.raises(ValidationError):
            AgentSpec(
                agent_id="invalid agent!",
                role="test",
                objective="Test objective here",
            )

    def test_reject_short_objective(self):
        """Test that objectives must be at least 10 characters."""
        with pytest.raises(ValidationError):
            AgentSpec(
                agent_id="test",
                role="test",
                objective="Too short",
            )

    def test_reject_invalid_temperature(self):
        """Test that temperature must be between 0 and 2."""
        with pytest.raises(ValidationError):
            AgentSpec(
                agent_id="test",
                role="test",
                objective="Test objective here",
                temperature=3.0,
            )

    def test_structured_output_requires_schema(self):
        """Test that structured output format requires output_schema."""
        with pytest.raises(ValidationError) as exc_info:
            AgentSpec(
                agent_id="test",
                role="test",
                objective="Test objective here",
                output_format=OutputFormat.STRUCTURED,
            )
        assert "output_schema" in str(exc_info.value)

    def test_structured_output_with_schema(self):
        """Test that structured output works with schema."""
        agent = AgentSpec(
            agent_id="test",
            role="test",
            objective="Test objective here",
            output_format=OutputFormat.STRUCTURED,
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )
        assert agent.output_schema is not None


class TestAgentSpecInheritance:
    """Tests for bili-core inheritance flags."""

    def test_inheritance_defaults(self):
        """Test default inheritance configuration."""
        agent = AgentSpec(
            agent_id="test",
            role="test",
            objective="Test inheritance defaults",
        )
        assert agent.inherit_from_bili_core is False
        assert agent.inherit_llm_config is True
        assert agent.inherit_tools is True
        assert agent.inherit_system_prompt is True

    def test_inheritance_normalization(self):
        """Test that sub-flags are normalized when master is off."""
        agent = AgentSpec(
            agent_id="test",
            role="test",
            objective="Test inheritance normalization",
            inherit_from_bili_core=False,
            inherit_llm_config=False,  # This should be normalized to True
        )
        # When master is off, sub-flags are normalized to True
        assert agent.inherit_llm_config is True

    def test_selective_inheritance(self):
        """Test selective inheritance (inherit some, not others)."""
        agent = AgentSpec(
            agent_id="test",
            role="test",
            objective="Test selective inheritance",
            inherit_from_bili_core=True,
            inherit_system_prompt=False,
            system_prompt="My custom prompt",
        )
        assert agent.inherit_from_bili_core is True
        assert agent.inherit_system_prompt is False
        assert agent.inherit_tools is True  # Still True


class TestAgentSpecHelpers:
    """Tests for helper methods."""

    def test_get_display_name(self):
        """Test display name generation."""
        agent = AgentSpec(
            agent_id="test",
            role="content_reviewer",
            objective="Test display name generation",
        )
        assert agent.get_display_name() == "Content Reviewer"

    def test_to_node_kwargs(self):
        """Test conversion to bili-core node kwargs."""
        agent = AgentSpec(
            agent_id="test",
            role="test_role",
            objective="Test node kwargs conversion",
            temperature=0.5,
            tools=["tool1"],
            system_prompt="Custom prompt",
        )
        kwargs = agent.to_node_kwargs()
        assert kwargs["agent_id"] == "test"
        assert kwargs["role"] == "test_role"
        assert kwargs["temperature"] == 0.5
        assert kwargs["system_prompt"] == "Custom prompt"

    def test_string_representation(self):
        """Test string representations."""
        agent = AgentSpec(
            agent_id="test",
            role="my_role",
            objective="Test string representation",
        )
        assert "test" in str(agent)
        assert "my_role" in str(agent)


class TestAgentSpecDomainAgnostic:
    """Tests demonstrating domain-agnostic design."""

    def test_content_moderation_domain(self):
        """Test agents for content moderation domain."""
        reviewer = AgentSpec(
            agent_id="reviewer",
            role="content_reviewer",
            objective="Review user-generated content for policy violations",
            capabilities=["text_analysis", "policy_evaluation"],
        )
        assert reviewer.role == "content_reviewer"

    def test_research_domain(self):
        """Test agents for research domain."""
        researcher = AgentSpec(
            agent_id="researcher",
            role="research_analyst",
            objective="Analyze scientific literature and extract findings",
            capabilities=["literature_review", "citation_analysis"],
        )
        assert researcher.role == "research_analyst"

    def test_customer_support_domain(self):
        """Test agents for customer support domain."""
        support = AgentSpec(
            agent_id="support_agent",
            role="tier1_support",
            objective="Handle first-level customer inquiries",
            capabilities=["faq_lookup", "ticket_routing"],
        )
        assert support.role == "tier1_support"

    def test_code_review_domain(self):
        """Test agents for code review domain."""
        reviewer = AgentSpec(
            agent_id="security_reviewer",
            role="security_analyst",
            objective="Review code changes for security vulnerabilities",
            capabilities=["vulnerability_detection", "owasp_compliance"],
        )
        assert reviewer.role == "security_analyst"

    def test_completely_custom_domain(self):
        """Test agents for a completely custom domain."""
        agent = AgentSpec(
            agent_id="quantum_agent",
            role="quantum_circuit_optimizer",
            objective="Optimize quantum circuit parameters for maximum gate fidelity",
            capabilities=["quantum_simulation", "parameter_tuning", "noise_mitigation"],
        )
        assert agent.role == "quantum_circuit_optimizer"
        assert "quantum_simulation" in agent.capabilities
