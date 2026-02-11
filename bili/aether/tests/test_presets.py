"""Tests for the agent preset system."""

import pytest

from bili.aether.schema import (
    AGENT_PRESETS,
    create_agent_from_preset,
    get_preset,
    list_presets,
    register_preset,
)


def test_list_presets():
    """Test listing available presets."""
    presets = list_presets()
    assert isinstance(presets, list)
    assert len(presets) > 0
    assert "researcher" in presets
    assert "content_reviewer" in presets
    assert "supervisor" in presets


def test_get_preset():
    """Test getting a preset by name."""
    preset = get_preset("researcher")
    assert preset is not None
    assert preset["role"] == "researcher"
    assert "web_search" in preset["capabilities"]


def test_get_nonexistent_preset():
    """Test getting a preset that doesn't exist."""
    preset = get_preset("nonexistent_preset")
    assert preset is None


def test_create_agent_from_preset():
    """Test creating an agent from a preset."""
    agent = create_agent_from_preset(
        preset_name="researcher",
        agent_id="my_researcher",
        objective="Research AI developments",
    )

    assert agent.agent_id == "my_researcher"
    assert agent.role == "researcher"
    assert agent.objective == "Research AI developments"
    assert "web_search" in agent.capabilities


def test_create_agent_with_overrides():
    """Test creating agent from preset with overrides."""
    agent = create_agent_from_preset(
        preset_name="code_reviewer",
        agent_id="security_expert",
        objective="Review Python code for security",
        temperature=0.1,
        capabilities=["python_security", "owasp_detection"],
    )

    assert agent.agent_id == "security_expert"
    assert agent.role == "code_reviewer"
    assert agent.temperature == 0.1
    # Overrides replace preset capabilities
    assert "python_security" in agent.capabilities
    assert "owasp_detection" in agent.capabilities


def test_create_agent_unknown_preset():
    """Test that unknown preset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset"):
        create_agent_from_preset(
            preset_name="unknown_preset",
            agent_id="test",
            objective="This should fail",
        )


def test_register_preset():
    """Test registering a custom preset."""
    register_preset("my_custom_preset", {
        "role": "custom_role",
        "capabilities": ["custom_cap"],
        "temperature": 0.4,
    })

    assert "my_custom_preset" in list_presets()
    preset = get_preset("my_custom_preset")
    assert preset["role"] == "custom_role"


def test_create_agent_from_custom_preset():
    """Test creating agent from a registered custom preset."""
    # Register first
    register_preset("test_custom", {
        "role": "test_role",
        "capabilities": ["test_cap"],
        "temperature": 0.5,
    })

    # Create agent
    agent = create_agent_from_preset(
        preset_name="test_custom",
        agent_id="test_agent",
        objective="Testing custom preset functionality",
    )

    assert agent.role == "test_role"
    assert agent.temperature == 0.5
    assert "test_cap" in agent.capabilities


def test_preset_registry_not_empty():
    """Test that AGENT_PRESETS registry has entries."""
    assert len(AGENT_PRESETS) > 0
    assert "researcher" in AGENT_PRESETS
    assert "supervisor" in AGENT_PRESETS


def test_all_presets_have_role():
    """Test that all presets have a role defined."""
    for name, preset in AGENT_PRESETS.items():
        assert "role" in preset, f"Preset '{name}' missing 'role'"


def test_supervisor_preset_has_flag():
    """Test supervisor preset has is_supervisor flag."""
    preset = get_preset("supervisor")
    assert preset is not None
    assert preset.get("is_supervisor") is True
