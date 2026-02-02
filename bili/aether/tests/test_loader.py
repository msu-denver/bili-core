"""Tests for YAML and dict MAS configuration loaders."""

import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError

from bili.aether.config.loader import load_mas_from_dict, load_mas_from_yaml
from bili.aether.schema import MASConfig, WorkflowType

# Path to example YAML configs
_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "examples",
)


# =========================================================================
# YAML LOADING TESTS — one per example config
# =========================================================================


def test_load_simple_chain_yaml():
    """Load simple_chain.yaml and verify sequential workflow with 4 agents."""
    config = load_mas_from_yaml(os.path.join(_EXAMPLES_DIR, "simple_chain.yaml"))

    assert isinstance(config, MASConfig)
    assert config.mas_id == "simple_chain"
    assert config.workflow_type == WorkflowType.SEQUENTIAL
    assert len(config.agents) == 4
    assert [a.agent_id for a in config.agents] == [
        "community_manager",
        "content_reviewer",
        "policy_expert",
        "judge",
    ]
    assert len(config.channels) == 3


def test_load_hierarchical_yaml():
    """Load hierarchical_voting.yaml and verify 7 agents across 3 tiers."""
    config = load_mas_from_yaml(os.path.join(_EXAMPLES_DIR, "hierarchical_voting.yaml"))

    assert config.mas_id == "hierarchical_voting"
    assert config.workflow_type == WorkflowType.HIERARCHICAL
    assert config.hierarchical_voting is True
    assert config.min_debate_rounds == 2
    assert len(config.agents) == 7

    # Check tiers: 4 at tier 3, 2 at tier 2, 1 at tier 1
    tier_3 = config.get_agents_by_tier(3)
    assert len(tier_3) == 4
    tier_2 = config.get_agents_by_tier(2)
    assert len(tier_2) == 2
    tier_1 = config.get_agents_by_tier(1)
    assert len(tier_1) == 1
    assert tier_1[0].agent_id == "vote_agent"
    assert tier_1[0].voting_weight == 2.0

    # Tier 2 agents are aggregators
    for agent in tier_2:
        assert "aggregator" in agent.role


def test_load_supervisor_yaml():
    """Load supervisor_moderation.yaml and verify judge as supervisor."""
    config = load_mas_from_yaml(
        os.path.join(_EXAMPLES_DIR, "supervisor_moderation.yaml")
    )

    assert config.mas_id == "supervisor_moderation"
    assert config.workflow_type == WorkflowType.SUPERVISOR
    assert config.entry_point == "judge"
    assert len(config.agents) == 4

    judge = config.get_agent("judge")
    assert judge is not None
    assert judge.is_supervisor is True

    # Verify appeals_specialist is present
    appeals = config.get_agent("appeals_specialist")
    assert appeals is not None
    assert appeals.role == "appeals_specialist"


def test_load_consensus_yaml():
    """Load consensus_network.yaml and verify role-based consensus agents."""
    config = load_mas_from_yaml(os.path.join(_EXAMPLES_DIR, "consensus_network.yaml"))

    assert config.mas_id == "consensus_network"
    assert config.workflow_type == WorkflowType.CONSENSUS
    assert config.consensus_threshold == 0.66
    assert config.max_consensus_rounds == 5
    assert config.consensus_detection == "majority"
    assert len(config.agents) == 3

    # Verify specific role-based agents (not generic judges)
    agent_ids = [a.agent_id for a in config.agents]
    assert "community_manager" in agent_ids
    assert "content_reviewer" in agent_ids
    assert "policy_expert" in agent_ids

    # All agents should have consensus_vote_field
    for agent in config.agents:
        assert agent.consensus_vote_field == "decision"

    # All channels should be bidirectional consensus
    for chan in config.channels:
        assert chan.bidirectional is True


def test_load_custom_escalation_yaml():
    """Load custom_escalation.yaml and verify custom workflow with 5 agents."""
    config = load_mas_from_yaml(os.path.join(_EXAMPLES_DIR, "custom_escalation.yaml"))

    assert config.mas_id == "custom_escalation"
    assert config.workflow_type == WorkflowType.CUSTOM
    assert config.human_in_loop is True
    assert config.human_escalation_condition is not None
    assert len(config.agents) == 5
    assert len(config.workflow_edges) == 7

    # Verify community_manager is present
    assert config.get_agent("community_manager") is not None

    # Verify conditional edges exist (judge decides or escalates)
    conditional = [e for e in config.workflow_edges if e.condition is not None]
    assert len(conditional) == 2

    # Verify END terminal edges
    terminal = [e for e in config.workflow_edges if e.to_agent == "END"]
    assert len(terminal) == 2


# =========================================================================
# DICT LOADING TESTS
# =========================================================================


def test_load_from_dict():
    """Load a MASConfig from a plain Python dict."""
    data = {
        "mas_id": "dict_test",
        "name": "Dict Test MAS",
        "workflow_type": "sequential",
        "agents": [
            {
                "agent_id": "reviewer",
                "role": "content_reviewer",
                "objective": "Review content for violations",
            },
            {
                "agent_id": "judge",
                "role": "judge",
                "objective": "Make final moderation decision",
            },
        ],
        "channels": [
            {
                "channel_id": "reviewer_to_judge",
                "protocol": "direct",
                "source": "reviewer",
                "target": "judge",
            }
        ],
    }

    config = load_mas_from_dict(data)
    assert isinstance(config, MASConfig)
    assert config.mas_id == "dict_test"
    assert len(config.agents) == 2
    assert len(config.channels) == 1


def test_dict_matches_yaml():
    """Verify that loading from dict produces the same result as YAML."""
    yaml_path = os.path.join(_EXAMPLES_DIR, "simple_chain.yaml")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    from_yaml = load_mas_from_yaml(yaml_path)
    from_dict = load_mas_from_dict(data)

    assert from_yaml.mas_id == from_dict.mas_id
    assert from_yaml.workflow_type == from_dict.workflow_type
    assert len(from_yaml.agents) == len(from_dict.agents)
    for a_yaml, a_dict in zip(from_yaml.agents, from_dict.agents):
        assert a_yaml.agent_id == a_dict.agent_id
        assert a_yaml.role == a_dict.role


# =========================================================================
# ERROR HANDLING TESTS
# =========================================================================


def test_yaml_file_not_found():
    """FileNotFoundError for missing YAML file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_mas_from_yaml("/nonexistent/path/config.yaml")


def test_yaml_invalid_syntax():
    """yaml.YAMLError for malformed YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write("mas_id: test\n  bad_indent: [unclosed\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(yaml.YAMLError):
            load_mas_from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_yaml_not_a_mapping():
    """ValueError when YAML top-level is not a mapping."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write("- item1\n- item2\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="mapping"):
            load_mas_from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_yaml_validation_error():
    """ValidationError for schema-invalid data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        # Missing required fields (agents, name, workflow_type)
        tmp.write("mas_id: incomplete\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValidationError):
            load_mas_from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)


# =========================================================================
# ROUNDTRIP TEST
# =========================================================================


def test_all_examples_roundtrip():
    """Load each YAML, export to dict, reload, and compare."""
    example_files = [
        "simple_chain.yaml",
        "hierarchical_voting.yaml",
        "supervisor_moderation.yaml",
        "consensus_network.yaml",
        "custom_escalation.yaml",
    ]

    for fname in example_files:
        fpath = os.path.join(_EXAMPLES_DIR, fname)
        original = load_mas_from_yaml(fpath)

        # Roundtrip: MASConfig → dict → MASConfig
        as_dict = original.model_dump()
        reloaded = load_mas_from_dict(as_dict)

        assert original.mas_id == reloaded.mas_id, f"Failed on {fname}"
        assert original.workflow_type == reloaded.workflow_type, f"Failed on {fname}"
        assert len(original.agents) == len(reloaded.agents), f"Failed on {fname}"
