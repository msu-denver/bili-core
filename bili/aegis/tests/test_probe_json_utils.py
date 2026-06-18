"""Tests for :mod:`bili.aegis.probe._json_utils`.

Aggressively covers the parse / retry / fallback / token-accumulation
semantics that PROBE policies and the SuccessEvaluator depend on.
"""

import logging

import pytest

from bili.aegis.probe._json_utils import _extract_json_object, invoke_with_json_retry
from bili.aegis.probe._llm import _FakeLLM

# =========================================================================
# _extract_json_object — happy path
# =========================================================================


def test_extract_plain_json_object():
    """Bare JSON object parses directly."""
    parsed = _extract_json_object('{"a": 1, "b": "hi"}')
    assert parsed == {"a": 1, "b": "hi"}


def test_extract_json_strips_markdown_fence():
    """Standard ```json ... ``` fence is stripped before parsing."""
    raw = '```json\n{"a": 1}\n```'
    assert _extract_json_object(raw) == {"a": 1}


def test_extract_json_strips_bare_fence():
    """``` ... ``` without language tag also works."""
    raw = '```\n{"a": 1}\n```'
    assert _extract_json_object(raw) == {"a": 1}


def test_extract_json_recovers_from_surrounding_prose():
    """When the LLM wraps JSON in commentary, the {...} slice is parsed."""
    raw = 'Here is the JSON you asked for:\n\n{"score": 2, "reasoning": "ok"}\n\nDone.'
    parsed = _extract_json_object(raw)
    assert parsed == {"score": 2, "reasoning": "ok"}


# =========================================================================
# _extract_json_object — error paths
# =========================================================================


def test_extract_json_raises_when_no_object_present():
    """No braces in the response → ValueError."""
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("plain text response with no braces")


def test_extract_json_raises_when_malformed_inside_braces():
    """Braces present but content is not valid JSON."""
    with pytest.raises(ValueError, match="Could not parse"):
        _extract_json_object('{"a": 1, "b": invalid}')


def test_extract_json_rejects_top_level_array():
    """A JSON array is not an object; this MUST raise.

    Anti-cheat: a lenient parser that accepts lists would break
    invoke_with_json_retry's required-keys check.
    """
    with pytest.raises(ValueError, match="Expected JSON object"):
        _extract_json_object("[1, 2, 3]")


def test_extract_json_rejects_top_level_scalar():
    """Top-level scalar values are not objects."""
    with pytest.raises(ValueError, match="Expected JSON object"):
        _extract_json_object("42")


# =========================================================================
# invoke_with_json_retry — happy path (no retry)
# =========================================================================


def test_invoke_returns_three_tuple():
    """Return shape: (dict, int, int)."""
    llm = _FakeLLM(
        responder=lambda p: ('{"score": 1}', 10, 5),
    )
    result = invoke_with_json_retry(
        llm,
        prompt="test",
        required_keys={"score"},
        fallback_factory=lambda: {"score": 0},
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    parsed, t_in, t_out = result
    assert isinstance(parsed, dict)
    assert t_in == 10
    assert t_out == 5


def test_invoke_first_call_valid_returns_immediately():
    """Valid JSON on first call → no retry, no fallback."""
    invocations = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        invocations.append(prompt)
        return '{"a": 1, "b": 2}', 10, 5

    llm = _FakeLLM(responder=_resp)
    parsed, t_in, t_out = invoke_with_json_retry(
        llm,
        prompt="p",
        required_keys={"a", "b"},
        fallback_factory=lambda: {"a": -1, "b": -1},
    )
    assert parsed == {"a": 1, "b": 2}
    assert (t_in, t_out) == (10, 5)
    assert len(invocations) == 1


def test_invoke_with_empty_required_keys_skips_key_check():
    """Pass required_keys=set() and parse-only-success suffices.

    Catches: an implementation that always retries when required_keys is empty.
    """
    invocations = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        invocations.append(prompt)
        return "{}", 1, 1

    llm = _FakeLLM(responder=_resp)
    parsed, _, _ = invoke_with_json_retry(
        llm,
        prompt="p",
        required_keys=set(),
        fallback_factory=lambda: {"fallback": True},
    )
    assert parsed == {}
    assert len(invocations) == 1


# =========================================================================
# invoke_with_json_retry — key-missing triggers retry
# =========================================================================


def test_invoke_missing_required_key_triggers_retry():
    """First response parses but misses a required key → retry.

    Anti-cheat: catches a parse-then-return-on-success-only flow that
    ignores required_keys.
    """
    fake = _FakeLLM(
        script={
            "default": [
                '{"score": 1}',  # missing 'reasoning'
                '{"score": 2, "reasoning": "took retry"}',
            ]
        },
        tokens_per_call=(7, 3),
    )
    parsed, t_in, t_out = invoke_with_json_retry(
        fake,
        prompt="p",
        required_keys={"score", "reasoning"},
        fallback_factory=lambda: {"score": 0, "reasoning": "FB"},
    )
    assert parsed == {"score": 2, "reasoning": "took retry"}
    # Both attempts charged
    assert t_in == 14
    assert t_out == 6


# =========================================================================
# invoke_with_json_retry — malformed JSON triggers retry
# =========================================================================


def test_invoke_malformed_json_first_then_valid_retries():
    """First response is garbage → retry once → second parses and returns."""
    fake = _FakeLLM(
        script={
            "default": [
                "this is not json at all",
                '{"a": 99}',
            ]
        },
        tokens_per_call=(5, 5),
    )
    parsed, t_in, t_out = invoke_with_json_retry(
        fake,
        prompt="p",
        required_keys={"a"},
        fallback_factory=lambda: {"a": 0},
    )
    assert parsed == {"a": 99}
    assert (t_in, t_out) == (10, 10)


def test_invoke_retry_prompt_signals_error_to_llm():
    """Second invocation's prompt has an error preamble.

    Anti-cheat: catches a retry that sends the identical prompt
    (giving the LLM no signal that anything went wrong).
    """
    invocations: list[str] = []

    counter = {"n": 0}

    def _resp(prompt: str) -> tuple[str, int, int]:
        invocations.append(prompt)
        counter["n"] += 1
        if counter["n"] == 1:
            return "garbage", 0, 0
        return '{"a": 1}', 0, 0

    llm = _FakeLLM(responder=_resp)
    invoke_with_json_retry(
        llm,
        prompt="ORIGINAL PROMPT",
        required_keys={"a"},
        fallback_factory=lambda: {"a": 0},
    )
    assert len(invocations) == 2
    # The retry prompt is distinct from the original and references the failure
    assert invocations[0] != invocations[1]
    assert "not valid JSON" in invocations[1] or "Retry" in invocations[1]
    # The retry prompt still contains the original prompt content
    assert "ORIGINAL PROMPT" in invocations[1]


# =========================================================================
# invoke_with_json_retry — fallback after double failure
# =========================================================================


def test_invoke_falls_back_after_two_parse_failures():
    """Both attempts unparseable → fallback_factory is called and its
    dict is returned.
    """
    factory_calls = {"n": 0}

    def _factory():
        factory_calls["n"] += 1
        return {"score": -1, "reason": "fallback"}

    fake = _FakeLLM(
        script={"default": ["garbage 1", "garbage 2"]},
        tokens_per_call=(3, 2),
    )
    parsed, t_in, t_out = invoke_with_json_retry(
        fake,
        prompt="p",
        required_keys={"score"},
        fallback_factory=_factory,
    )
    assert parsed == {"score": -1, "reason": "fallback"}
    assert factory_calls["n"] == 1
    # Token cost from both attempts still counted
    assert (t_in, t_out) == (6, 4)


def test_invoke_falls_back_after_two_key_misses():
    """Both attempts parse but miss required keys → fallback fires."""
    factory_calls = {"n": 0}

    def _factory():
        factory_calls["n"] += 1
        return {"score": 0, "reasoning": "fb", "confidence": "low"}

    fake = _FakeLLM(
        script={
            "default": [
                '{"score": 1}',  # missing reasoning + confidence
                '{"reasoning": "second", "confidence": "high"}',  # missing score
            ]
        },
    )
    parsed, _, _ = invoke_with_json_retry(
        fake,
        prompt="p",
        required_keys={"score", "reasoning", "confidence"},
        fallback_factory=_factory,
    )
    assert parsed == {"score": 0, "reasoning": "fb", "confidence": "low"}
    assert factory_calls["n"] == 1


def test_invoke_fallback_factory_called_at_most_once():
    """Anti-cheat: a retry loop that re-calls factory would mask real bugs."""
    factory_calls = {"n": 0}

    def _factory():
        factory_calls["n"] += 1
        return {"x": "fb"}

    fake = _FakeLLM(script={"default": ["garbage", "still garbage"]})
    invoke_with_json_retry(
        fake,
        prompt="p",
        required_keys={"x"},
        fallback_factory=_factory,
    )
    assert factory_calls["n"] == 1


# =========================================================================
# invoke_with_json_retry — logging
# =========================================================================


def test_invoke_logs_warning_on_first_parse_failure(caplog):
    """A failed parse emits a warning naming the label.

    Catches: silent failures that leave operators no trace.
    """
    fake = _FakeLLM(
        script={"default": ["garbage 1", '{"a": 1}']},
    )
    with caplog.at_level(logging.WARNING, logger="bili.aegis.probe._json_utils"):
        invoke_with_json_retry(
            fake,
            prompt="p",
            required_keys={"a"},
            fallback_factory=lambda: {"a": 0},
            label="planner_unit_test",
        )
    assert any("planner_unit_test" in rec.message for rec in caplog.records)
    assert any("first" in rec.message.lower() for rec in caplog.records)


def test_invoke_logs_warning_when_falling_back(caplog):
    """The fallback path emits a distinct final warning."""
    fake = _FakeLLM(script={"default": ["x", "y"]})
    with caplog.at_level(logging.WARNING, logger="bili.aegis.probe._json_utils"):
        invoke_with_json_retry(
            fake,
            prompt="p",
            required_keys={"a"},
            fallback_factory=lambda: {"a": 0},
            label="my_test_label",
        )
    assert any(
        "falling back" in rec.message.lower() and "my_test_label" in rec.message
        for rec in caplog.records
    )


def test_invoke_no_warnings_on_clean_first_attempt(caplog):
    """Happy path: NO warning records emitted by _json_utils.

    Anti-cheat: unconditional warning logging would pollute production logs.
    """
    fake = _FakeLLM(responder=lambda p: ('{"a": 1}', 0, 0))
    with caplog.at_level(logging.WARNING, logger="bili.aegis.probe._json_utils"):
        invoke_with_json_retry(
            fake,
            prompt="p",
            required_keys={"a"},
            fallback_factory=lambda: {"a": -1},
            label="clean",
        )
    own_warnings = [
        r
        for r in caplog.records
        if r.name == "bili.aegis.probe._json_utils" and r.levelno >= logging.WARNING
    ]
    assert own_warnings == []
