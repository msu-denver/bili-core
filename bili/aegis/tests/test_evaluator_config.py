"""Tests for evaluator_config constants and structures.

Validates that score description dicts, judge prompts, and
provider family prefixes are well-formed.
"""

from bili.aegis.evaluator.evaluator_config import (
    AGENT_IMPERSONATION_JUDGE_PROMPT,
    AGENT_IMPERSONATION_SCORE_DESCRIPTIONS,
    BIAS_INHERITANCE_JUDGE_PROMPT,
    BIAS_INHERITANCE_SCORE_DESCRIPTIONS,
    DEFAULT_JUDGE_PROMPT,
    EVALUATOR_TEMPERATURE,
    FALLBACK_EVALUATOR_MODEL,
    JAILBREAK_JUDGE_PROMPT,
    JAILBREAK_SCORE_DESCRIPTIONS,
    MEMORY_POISONING_JUDGE_PROMPT,
    MEMORY_POISONING_SCORE_DESCRIPTIONS,
    PERSISTENCE_JUDGE_PROMPT,
    PERSISTENCE_SCORE_DESCRIPTIONS,
    PRIMARY_EVALUATOR_MODEL,
    PROVIDER_FAMILY_PREFIXES,
    SAME_MODEL_WARNING,
    SAME_PROVIDER_WARNING,
    VERDICT_CONFIDENCE_VALUES,
    VERDICT_SCORE_DESCRIPTIONS,
)


class TestScoreDescriptions:
    """Tests for all score description dictionaries."""

    _EXPECTED_KEYS = {0, 1, 2, 3}

    def test_verdict_score_descriptions_keys(self):
        """VERDICT_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(VERDICT_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_jailbreak_score_descriptions_keys(self):
        """JAILBREAK_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(JAILBREAK_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_memory_poisoning_score_descriptions_keys(self):
        """MEMORY_POISONING_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(MEMORY_POISONING_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_bias_inheritance_score_descriptions_keys(self):
        """BIAS_INHERITANCE_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(BIAS_INHERITANCE_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_agent_impersonation_score_descriptions_keys(self):
        """AGENT_IMPERSONATION_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(AGENT_IMPERSONATION_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_persistence_score_descriptions_keys(self):
        """PERSISTENCE_SCORE_DESCRIPTIONS has keys 0-3."""
        assert set(PERSISTENCE_SCORE_DESCRIPTIONS.keys()) == self._EXPECTED_KEYS

    def test_all_descriptions_are_nonempty_strings(self):
        """All description values are non-empty strings."""
        all_dicts = [
            VERDICT_SCORE_DESCRIPTIONS,
            JAILBREAK_SCORE_DESCRIPTIONS,
            MEMORY_POISONING_SCORE_DESCRIPTIONS,
            BIAS_INHERITANCE_SCORE_DESCRIPTIONS,
            AGENT_IMPERSONATION_SCORE_DESCRIPTIONS,
            PERSISTENCE_SCORE_DESCRIPTIONS,
        ]
        for desc_dict in all_dicts:
            for key, val in desc_dict.items():
                assert isinstance(val, str), f"Key {key} not str"
                assert len(val) > 0, f"Key {key} is empty"


class TestJudgePrompts:
    """Tests for judge prompt templates."""

    _ALL_PROMPTS = [
        DEFAULT_JUDGE_PROMPT,
        JAILBREAK_JUDGE_PROMPT,
        MEMORY_POISONING_JUDGE_PROMPT,
        BIAS_INHERITANCE_JUDGE_PROMPT,
        AGENT_IMPERSONATION_JUDGE_PROMPT,
        PERSISTENCE_JUDGE_PROMPT,
    ]

    def test_prompts_are_nonempty_strings(self):
        """All judge prompts are non-empty strings."""
        for prompt in self._ALL_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_prompts_have_required_placeholders(self):
        """All prompts contain the required format placeholders."""
        required = [
            "{agent_id}",
            "{payload}",
            "{baseline_section}",
            "{test_text}",
            "{score_guide}",
        ]
        for prompt in self._ALL_PROMPTS:
            for placeholder in required:
                assert placeholder in prompt, f"Missing {placeholder}"


class TestProviderFamilyPrefixes:
    """Tests for PROVIDER_FAMILY_PREFIXES structure."""

    def test_is_list_of_tuples(self):
        """PROVIDER_FAMILY_PREFIXES is a list of 2-tuples."""
        assert isinstance(PROVIDER_FAMILY_PREFIXES, list)
        assert len(PROVIDER_FAMILY_PREFIXES) > 0
        for item in PROVIDER_FAMILY_PREFIXES:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_all_entries_are_strings(self):
        """Both prefix and family name are strings."""
        for prefix, family in PROVIDER_FAMILY_PREFIXES:
            assert isinstance(prefix, str)
            assert isinstance(family, str)
            assert len(prefix) > 0
            assert len(family) > 0

    def test_known_families_present(self):
        """Key provider families are present."""
        families = {f for _, f in PROVIDER_FAMILY_PREFIXES}
        assert "anthropic_bedrock" in families
        assert "google_vertex" in families
        assert "openai" in families


class TestModelConstants:
    """Tests for evaluator model constants."""

    def test_primary_model_is_string(self):
        """PRIMARY_EVALUATOR_MODEL is a non-empty string."""
        assert isinstance(PRIMARY_EVALUATOR_MODEL, str)
        assert len(PRIMARY_EVALUATOR_MODEL) > 0

    def test_fallback_model_is_string(self):
        """FALLBACK_EVALUATOR_MODEL is a non-empty string."""
        assert isinstance(FALLBACK_EVALUATOR_MODEL, str)
        assert len(FALLBACK_EVALUATOR_MODEL) > 0

    def test_temperature_is_zero(self):
        """Evaluator temperature is 0.0 for reproducibility."""
        assert EVALUATOR_TEMPERATURE == 0.0


class TestConfidenceValues:
    """Tests for VERDICT_CONFIDENCE_VALUES."""

    def test_contains_expected_values(self):
        """Confidence list has high, medium, low."""
        assert "high" in VERDICT_CONFIDENCE_VALUES
        assert "medium" in VERDICT_CONFIDENCE_VALUES
        assert "low" in VERDICT_CONFIDENCE_VALUES

    def test_is_list(self):
        """VERDICT_CONFIDENCE_VALUES is a list."""
        assert isinstance(VERDICT_CONFIDENCE_VALUES, list)


class TestWarningTemplates:
    """Tests for warning message templates."""

    def test_same_model_warning_has_placeholders(self):
        """SAME_MODEL_WARNING has model and fallback placeholders."""
        assert "{model}" in SAME_MODEL_WARNING
        assert "{fallback}" in SAME_MODEL_WARNING

    def test_same_provider_warning_has_placeholders(self):
        """SAME_PROVIDER_WARNING has model, family, fallback."""
        assert "{model}" in SAME_PROVIDER_WARNING
        assert "{family}" in SAME_PROVIDER_WARNING
        assert "{fallback}" in SAME_PROVIDER_WARNING

    def test_warnings_can_be_formatted(self):
        """Warning templates format without error."""
        result = SAME_MODEL_WARNING.format(
            model="test-model", fallback="fallback-model"
        )
        assert "test-model" in result

        result = SAME_PROVIDER_WARNING.format(
            model="test-model",
            family="openai",
            fallback="fallback-model",
        )
        assert "openai" in result
