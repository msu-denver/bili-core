"""Tests for the PROBE-specific constants added to evaluator_config.py."""

from bili.aegis.evaluator.evaluator_config import (
    PROBE_JUDGE_PROMPT,
    PROBE_SCORE_DESCRIPTIONS,
    PROVIDER_FAMILY_PREFIXES,
)

# =========================================================================
# PROBE_SCORE_DESCRIPTIONS
# =========================================================================


def test_probe_score_descriptions_has_exactly_keys_0_1_2_3():
    """Completeness: rubric covers every score in the documented range."""
    assert set(PROBE_SCORE_DESCRIPTIONS.keys()) == {0, 1, 2, 3}


def test_probe_score_descriptions_all_non_empty_substantive_strings():
    """Anti-cheat: catches placeholder text or empty strings."""
    for score, desc in PROBE_SCORE_DESCRIPTIONS.items():
        assert isinstance(desc, str), f"score {score} description is not a str"
        assert len(desc) > 50, f"score {score} description is suspiciously short"


# =========================================================================
# PROBE_JUDGE_PROMPT
# =========================================================================


def test_probe_judge_prompt_contains_all_required_placeholders():
    """Five placeholders must be present for `.format()` substitution."""
    for placeholder in (
        "{objective_text}",
        "{success_criterion}",
        "{baseline_section}",
        "{victim_output_text}",
        "{score_guide}",
    ):
        assert placeholder in PROBE_JUDGE_PROMPT, f"missing {placeholder}"


def test_probe_judge_prompt_specifies_json_output_format():
    """Prompt must instruct LLM to return JSON with score/reasoning/confidence."""
    assert "Return ONLY a JSON object" in PROBE_JUDGE_PROMPT
    assert '"score"' in PROBE_JUDGE_PROMPT
    assert '"reasoning"' in PROBE_JUDGE_PROMPT
    assert '"confidence"' in PROBE_JUDGE_PROMPT


def test_probe_judge_prompt_format_renders_end_to_end():
    """All placeholders accept actual values without KeyError."""
    rendered = PROBE_JUDGE_PROMPT.format(
        objective_text="Cause X to do Y.",
        success_criterion="Y is produced verbatim.",
        baseline_section="Baseline output: NOT AVAILABLE",
        victim_output_text="The agent said Y.",
        score_guide="  0: ...\n  1: ...\n  2: ...\n  3: ...",
    )
    assert "Cause X to do Y." in rendered
    assert "Y is produced verbatim." in rendered
    assert "The agent said Y." in rendered


# =========================================================================
# PROVIDER_FAMILY_PREFIXES — DeepSeek addition for PROBE smoke trio
# =========================================================================


def test_provider_family_prefixes_contains_deepseek():
    """PROBE's smoke trio uses DeepSeek as attacker; prefix must be registered."""
    prefixes_lookup = dict(PROVIDER_FAMILY_PREFIXES)
    assert "deepseek-" in prefixes_lookup
    assert prefixes_lookup["deepseek-"] == "deepseek"
