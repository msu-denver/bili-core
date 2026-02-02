"""
Tests for agent presets registry.

These tests verify the registry-based preset system that provides
convenience without restrictions.
"""

import pytest

from bili.aether.schema.agent_presets import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    get_preset_config,
    list_presets,
    register_preset,
    update_preset,
)


class TestPresetRegistry:
    """Tests for preset registry functions."""

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "researcher" in presets
        assert "content_reviewer" in presets

    def test_get_preset_config(self):
        """Test getting raw preset configuration."""
        config = get_preset_config("researcher")
        assert config is not None
        assert "role" in config
        assert "objective" in config
        assert config["role"] == "researcher"

    def test_get_preset_config_not_found(self):
        """Test getting config for nonexistent preset."""
        config = get_preset_config("nonexistent_preset")
        assert config is None


class TestPresetCreation:
    """Tests for creating agents from presets."""

    def test_get_preset_basic(self):
        """Test creating agent from preset with just agent_id."""
        agent = get_preset("researcher", agent_id="my_researcher")
        assert agent.agent_id == "my_researcher"
        assert agent.role == "researcher"
        assert agent.temperature == 0.5  # From preset

    def test_get_preset_with_overrides(self):
        """Test creating agent from preset with overrides."""
        agent = get_preset(
            "researcher",
            agent_id="custom_researcher",
            temperature=0.8,
            objective="Custom research objective for this task",
        )
        assert agent.agent_id == "custom_researcher"
        assert agent.temperature == 0.8  # Override
        assert "Custom research" in agent.objective  # Override

    def test_get_preset_not_found(self):
        """Test that nonexistent preset raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("nonexistent_preset", agent_id="test")
        assert "not found" in str(exc_info.value)
        assert "Available presets" in str(exc_info.value)

    def test_create_agent_from_preset_alias(self):
        """Test that create_agent_from_preset is an alias for get_preset."""
        agent = create_agent_from_preset("researcher", agent_id="test")
        assert agent.role == "researcher"


class TestPresetRegistration:
    """Tests for registering custom presets."""

    def test_register_new_preset(self):
        """Test registering a new preset."""
        preset_name = "test_custom_preset"

        # Clean up if exists from previous test run
        if preset_name in AGENT_PRESETS:
            del AGENT_PRESETS[preset_name]

        register_preset(
            preset_name,
            {
                "role": "custom_role",
                "objective": "Custom objective for testing",
                "temperature": 0.4,
            },
        )

        assert preset_name in list_presets()
        agent = get_preset(preset_name, agent_id="test")
        assert agent.role == "custom_role"

        # Clean up
        del AGENT_PRESETS[preset_name]

    def test_register_duplicate_preset_fails(self):
        """Test that registering duplicate preset name fails."""
        with pytest.raises(ValueError) as exc_info:
            register_preset("researcher", {"role": "test", "objective": "Test objective"})
        assert "already exists" in str(exc_info.value)

    def test_register_preset_missing_required_fields(self):
        """Test that preset must have required fields."""
        with pytest.raises(ValueError) as exc_info:
            register_preset("incomplete", {"role": "test"})  # Missing objective
        assert "missing required fields" in str(exc_info.value)

    def test_update_preset(self):
        """Test updating an existing preset."""
        preset_name = "test_update_preset"

        # Register initial
        if preset_name in AGENT_PRESETS:
            del AGENT_PRESETS[preset_name]

        register_preset(
            preset_name,
            {
                "role": "original_role",
                "objective": "Original objective here",
            },
        )

        # Update
        update_preset(
            preset_name,
            {
                "role": "updated_role",
                "objective": "Updated objective here",
            },
        )

        config = get_preset_config(preset_name)
        assert config["role"] == "updated_role"

        # Clean up
        del AGENT_PRESETS[preset_name]


class TestBuiltInPresets:
    """Tests for built-in presets."""

    def test_researcher_preset(self):
        """Test researcher preset configuration."""
        agent = get_preset("researcher", agent_id="test")
        assert agent.role == "researcher"
        assert agent.temperature == 0.5
        assert "serp_api_tool" in agent.tools

    def test_content_reviewer_preset(self):
        """Test content_reviewer preset configuration."""
        agent = get_preset("content_reviewer", agent_id="test")
        assert agent.role == "content_reviewer"
        assert agent.temperature == 0.0  # Deterministic

    def test_supervisor_preset(self):
        """Test supervisor preset has is_supervisor flag."""
        agent = get_preset("supervisor", agent_id="test")
        assert agent.is_supervisor is True

    def test_voter_preset(self):
        """Test voter preset has voting_weight."""
        agent = get_preset("voter", agent_id="test")
        assert agent.voting_weight == 1.0

    def test_all_presets_create_valid_agents(self):
        """Test that all built-in presets create valid agents."""
        for preset_name in list_presets():
            agent = get_preset(preset_name, agent_id=f"test_{preset_name}")
            assert agent.agent_id == f"test_{preset_name}"
            assert agent.role
            assert len(agent.objective) >= 10
