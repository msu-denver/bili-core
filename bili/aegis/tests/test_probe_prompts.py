"""Tests for :mod:`bili.aegis.probe._prompts`.

Confirms each prompt constant exists, is non-empty, contains the expected
keywords for its design intent, and renders without ``KeyError`` against
its documented placeholders.
"""

import re

from bili.aegis.probe import _prompts

# =========================================================================
# PAIR planner prompts
# =========================================================================


def test_pair_planner_system_prompt_is_non_empty_string():
    """PAIR planner system prompt is a non-trivial string constant."""
    assert isinstance(_prompts.PAIR_PLANNER_SYSTEM_PROMPT, str)
    assert len(_prompts.PAIR_PLANNER_SYSTEM_PROMPT) > 100


def test_pair_planner_system_prompt_specifies_json_output():
    """Prompt must instruct the LLM to return JSON.

    Catches: a prompt that asks for natural language only.
    """
    text = _prompts.PAIR_PLANNER_SYSTEM_PROMPT
    assert "JSON" in text
    assert "improvement" in text
    assert "attack_angle" in text
    assert "rationale" in text


def test_pair_planner_user_template_renders_without_keyerror():
    """All documented placeholders can be supplied to .format() cleanly."""
    rendered = _prompts.PAIR_PLANNER_USER_TEMPLATE.format(
        objective_text="Cause X to do Y.",
        success_criterion="Y is produced.",
        target_role="reviewer",
        n_turns=2,
        history_block="Turn 0: ...\nTurn 1: ...",
    )
    assert "Cause X to do Y." in rendered
    assert "Y is produced." in rendered
    assert "reviewer" in rendered
    assert "Turn 0:" in rendered


def test_pair_planner_user_template_no_unfilled_braces_after_format():
    """After supplying all documented placeholders, no ``{...}`` remains.

    Anti-cheat: catches a forgotten placeholder.
    """
    rendered = _prompts.PAIR_PLANNER_USER_TEMPLATE.format(
        objective_text="X",
        success_criterion="Y",
        target_role="r",
        n_turns=0,
        history_block="(none)",
    )
    # Single-brace patterns left should only be literal "{" or "}" with neighbors;
    # the templates don't have literal braces, so we check for `{word}` patterns.
    leftovers = re.findall(r"\{[a-z_][a-z0-9_]*\}", rendered)
    assert not leftovers, f"Unfilled placeholders: {leftovers}"


# =========================================================================
# Crescendo prompts
# =========================================================================


def test_crescendo_ladder_prompt_renders_with_all_placeholders():
    """ladder_rungs, last_idx, objective_text, success_criterion."""
    rendered = _prompts.CRESCENDO_LADDER_PROMPT.format(
        ladder_rungs=8,
        last_idx=7,
        objective_text="Cause Y.",
        success_criterion="Y is produced.",
    )
    assert '"ladder" of 8' in rendered
    assert "rung 7" in rendered
    assert "Cause Y." in rendered


def test_crescendo_ladder_prompt_specifies_json_array_inside_object():
    """The prompt asks for a JSON object containing a 'ladder' array.

    Anti-cheat: catches a prompt that requests a bare JSON array (which
    invoke_with_json_retry's required_keys check would never satisfy).
    """
    text = _prompts.CRESCENDO_LADDER_PROMPT
    assert '"ladder"' in text
    assert "JSON object" in text


def test_crescendo_rung_intent_template_renders():
    """All placeholders for the per-turn rung intent template."""
    rendered = _prompts.CRESCENDO_RUNG_INTENT_TEMPLATE.format(
        rung_index=2,
        ladder_rungs=8,
        rung_description="Ask about X obliquely.",
        history_block="Turn 0: ...",
    )
    assert "rung (2 of 8)" in rendered.lower() or "2 of 8" in rendered
    assert "Ask about X obliquely." in rendered


# =========================================================================
# TAP expansion prompt
# =========================================================================


def test_tap_expansion_prompt_renders_with_all_placeholders():
    """TAP expansion prompt accepts all 6 documented placeholders."""
    rendered = _prompts.TAP_EXPANSION_PROMPT.format(
        branching_factor=3,
        objective_text="X.",
        success_criterion="Y.",
        parent_depth=2,
        parent_payload="prior payload text",
        parent_result_summary="judge said 1/3",
    )
    assert "3 refinement variants" in rendered
    assert "depth 2" in rendered
    assert "prior payload text" in rendered


def test_tap_expansion_prompt_requests_refinements_array():
    """The output schema names 'refinements' as the required key."""
    text = _prompts.TAP_EXPANSION_PROMPT
    assert '"refinements"' in text


# =========================================================================
# PayloadCrafter / HouYi prompts
# =========================================================================


def test_houyi_crafter_system_prompt_contains_houyi_keywords():
    """FRAMEWORK / SEPARATOR / PAYLOAD must appear (HouYi § 3.2 decomposition).

    Anti-cheat: a crafter prompt that omits these would not actually
    implement HouYi.
    """
    text = _prompts.HOUYI_CRAFTER_SYSTEM_PROMPT
    assert "FRAMEWORK" in text
    assert "SEPARATOR" in text
    assert "PAYLOAD" in text


def test_houyi_crafter_system_prompt_forbids_extra_commentary():
    """Crafter must output ONLY the prompt, no JSON or wrapper text."""
    text = _prompts.HOUYI_CRAFTER_SYSTEM_PROMPT
    assert "ONLY" in text
    assert "no JSON" in text or "no commentary" in text


def test_houyi_crafter_user_template_renders_with_all_placeholders():
    """HouYi crafter user template accepts all 6 documented placeholders."""
    rendered = _prompts.HOUYI_CRAFTER_USER_TEMPLATE.format(
        attack_angle="appeal-to-authority",
        rationale="agent is deferential to senior roles",
        mas_id="simple_chain",
        agents_compact="reviewer, judge",
        entry_point="manager",
        continuity_block="(first turn — no prior context)",
    )
    assert "appeal-to-authority" in rendered
    assert "simple_chain" in rendered
    assert "reviewer, judge" in rendered


# =========================================================================
# Module-level constants (anti-cheat)
# =========================================================================


def test_all_prompt_constants_are_module_level_strings():
    """Prompts are module-level constants, not function-local.

    Anti-cheat: a function-local definition would re-create the string
    every call, hiding mutation bugs and breaking object-identity asserts.
    """
    for name in (
        "PAIR_PLANNER_SYSTEM_PROMPT",
        "PAIR_PLANNER_USER_TEMPLATE",
        "CRESCENDO_LADDER_PROMPT",
        "CRESCENDO_RUNG_INTENT_TEMPLATE",
        "TAP_EXPANSION_PROMPT",
        "HOUYI_CRAFTER_SYSTEM_PROMPT",
        "HOUYI_CRAFTER_USER_TEMPLATE",
    ):
        value = getattr(_prompts, name)
        assert isinstance(value, str), f"{name} is not a str"
        assert len(value) > 50, f"{name} is suspiciously short"
