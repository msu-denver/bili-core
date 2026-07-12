"""Tests for bili.iris.tools.hitl.

Covers:
- NullHitlResponder returns the no-response sentinel without blocking.
- ScriptedHitlResponder cycles scripted answers and records calls.
- ScriptedHitlResponder raises when the script is exhausted (a test-authoring
  bug, not a runtime condition to degrade from).
- HitlResponder is runtime_checkable and isinstance-matches any object with
  a compatible ask() method (structural typing, not a required base class).
"""

# pylint: disable=missing-function-docstring

import pytest

from bili.iris.tools.hitl import (
    NO_RESPONSE_PREFIX,
    HitlResponder,
    NullHitlResponder,
    ScriptedHitlResponder,
)


class TestNullHitlResponder:
    """Tests for NullHitlResponder."""

    def test_returns_no_response_sentinel(self):
        responder = NullHitlResponder()
        answer = responder.ask("Which environment?")
        assert answer.startswith(NO_RESPONSE_PREFIX)

    def test_ignores_options(self):
        responder = NullHitlResponder()
        answer = responder.ask("Pick one", options=["a", "b"])
        assert answer.startswith(NO_RESPONSE_PREFIX)

    def test_satisfies_hitl_responder_protocol(self):
        assert isinstance(NullHitlResponder(), HitlResponder)


class TestScriptedHitlResponder:
    """Tests for ScriptedHitlResponder."""

    def test_returns_answers_in_order(self):
        responder = ScriptedHitlResponder(["staging", "yes"])
        assert responder.ask("Which environment?") == "staging"
        assert responder.ask("Proceed?") == "yes"

    def test_records_calls(self):
        responder = ScriptedHitlResponder(["staging"])
        responder.ask("Which environment?", options=["staging", "prod"])
        assert responder.calls == [
            {"question": "Which environment?", "options": ["staging", "prod"]}
        ]

    def test_calls_property_returns_a_copy(self):
        responder = ScriptedHitlResponder(["staging"])
        responder.ask("q")
        calls = responder.calls
        calls.append({"question": "tampered", "options": None})
        assert len(responder.calls) == 1

    def test_raises_when_script_exhausted(self):
        responder = ScriptedHitlResponder(["only-one"])
        responder.ask("first question")
        with pytest.raises(IndexError):
            responder.ask("second question, no script left")

    def test_satisfies_hitl_responder_protocol(self):
        assert isinstance(ScriptedHitlResponder([]), HitlResponder)
