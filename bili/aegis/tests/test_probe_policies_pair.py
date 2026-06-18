"""Tests for :class:`bili.aegis.probe.policies.pair.PAIRPolicy`.

Covers:
  * name() literal
  * plan_next_intent: tuple-return contract, prompt content, history rendering,
    truncation, retry + fallback semantics
  * should_continue: self-abandon trigger, streak reset semantics, boundary
    conditions on session length
  * mutation isolation: a single policy instance handles fresh sessions
    without state leak
"""

from typing import Optional

from bili.aegis.probe._llm import _FakeLLM
from bili.aegis.probe.policies.pair import PAIRPolicy
from bili.aegis.probe.schema import AttackIntent, ProbeSession, ProbeTurn, TurnVerdict
from bili.aegis.tests.conftest import (
    make_probe_intent,
    make_probe_objective,
    make_probe_session,
    make_probe_turn,
)


def _session(
    turns: Optional[list[ProbeTurn]] = None,
    target_role: Optional[str] = None,
) -> ProbeSession:
    """Local helper: PAIR session whose objective optionally pins ``target_role``."""
    return make_probe_session(
        objective=make_probe_objective(target_agent_role=target_role),
        turns=turns or [],
    )


def _turn(
    idx: int,
    payload: str = "prior payload",
    observation: str = "prior observation",
    **kwargs,
) -> ProbeTurn:
    """Local helper: positional ``idx`` + ``payload`` / ``observation`` aliases.

    ``payload`` maps to ``payload_text``; ``observation`` to
    ``observation_summary``. Other kwargs forward to :func:`make_probe_turn`.
    The default intent uses PAIR-specific labels ("prior_angle"/"r"/"r") so
    history-rendering assertions pick up exactly those strings.
    """
    kwargs.setdefault("tier3_reasoning", "prior reasoning")
    return make_probe_turn(
        turn_index=idx,
        intent=make_probe_intent(
            target_agent_role="r", attack_angle="prior_angle", rationale="r"
        ),
        payload_text=payload,
        observation_summary=observation,
        **kwargs,
    )


_VALID_PLANNER_JSON = (
    '{"improvement": "try roleplay since direct ask failed", '
    '"attack_angle": "roleplay_authority", '
    '"rationale": "the reviewer defers to characters with seniority"}'
)


def _policy(llm: _FakeLLM, max_turns: int = 8) -> PAIRPolicy:
    return PAIRPolicy(llm=llm, max_turns=max_turns)


# =========================================================================
# name()
# =========================================================================


def test_name_returns_pair_literal():
    """CSV column value is the literal `"pair"` string."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.name() == "pair"


# =========================================================================
# plan_next_intent — return shape
# =========================================================================


def test_plan_next_intent_returns_three_tuple():
    """Universal contract: (AttackIntent, int, int)."""
    fake = _FakeLLM(responder=lambda p: (_VALID_PLANNER_JSON, 50, 25))
    result = _policy(fake).plan_next_intent(_session())
    assert isinstance(result, tuple)
    assert len(result) == 3
    intent, t_in, t_out = result
    assert isinstance(intent, AttackIntent)
    assert (t_in, t_out) == (50, 25)


def test_plan_next_intent_packs_attack_angle_and_rationale_into_intent():
    """The LLM's JSON fields propagate verbatim into the returned AttackIntent."""
    fake = _FakeLLM(responder=lambda p: (_VALID_PLANNER_JSON, 0, 0))
    intent, _, _ = _policy(fake).plan_next_intent(_session())
    assert intent.attack_angle == "roleplay_authority"
    assert "defers to characters" in intent.rationale


def test_plan_next_intent_uses_objective_target_role_when_set():
    """When objective.target_agent_role is set, intent.target_agent_role matches."""
    fake = _FakeLLM(responder=lambda p: (_VALID_PLANNER_JSON, 0, 0))
    sess = _session(target_role="reviewer_alpha")
    intent, _, _ = _policy(fake).plan_next_intent(sess)
    assert intent.target_agent_role == "reviewer_alpha"


def test_plan_next_intent_uses_unspecified_placeholder_when_target_role_none():
    """No target_agent_role on objective → placeholder used in intent."""
    fake = _FakeLLM(responder=lambda p: (_VALID_PLANNER_JSON, 0, 0))
    intent, _, _ = _policy(fake).plan_next_intent(_session(target_role=None))
    assert intent.target_agent_role == "<unspecified>"


# =========================================================================
# plan_next_intent — prompt content (responder-mode assertions)
# =========================================================================


def _capture_prompt():
    seen: list[str] = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        seen.append(prompt)
        return (_VALID_PLANNER_JSON, 0, 0)

    return seen, _resp


def test_first_turn_prompt_has_no_prior_turns_placeholder():
    """Empty session.turns renders the (no prior turns) sentinel."""
    seen, resp = _capture_prompt()
    _policy(_FakeLLM(responder=resp)).plan_next_intent(_session(turns=[]))
    assert "no prior turns" in seen[0]


def test_subsequent_turn_prompt_contains_prior_payload():
    """Prior turn's payload appears in the planner prompt."""
    seen, resp = _capture_prompt()
    sess = _session(turns=[_turn(0, payload="UNIQUE_PRIOR_PAYLOAD_xx")])
    _policy(_FakeLLM(responder=resp)).plan_next_intent(sess)
    assert "UNIQUE_PRIOR_PAYLOAD_xx" in seen[0]


def test_subsequent_turn_prompt_contains_prior_verdict_value():
    """Prior turn's verdict (enum .value) appears in the prompt."""
    seen, resp = _capture_prompt()
    sess = _session(turns=[_turn(0, verdict=TurnVerdict.PARTIAL_PROGRESS)])
    _policy(_FakeLLM(responder=resp)).plan_next_intent(sess)
    assert "partial_progress" in seen[0]


def test_subsequent_turn_prompt_contains_judge_reasoning():
    """Prior judge reasoning is shown to the planner."""
    seen, resp = _capture_prompt()
    sess = _session(turns=[_turn(0, tier3_reasoning="REASONING_MARKER_42")])
    _policy(_FakeLLM(responder=resp)).plan_next_intent(sess)
    assert "REASONING_MARKER_42" in seen[0]


def test_subsequent_turn_prompt_contains_objective_and_success_criterion():
    """Objective text and success criterion appear in every prompt."""
    seen, resp = _capture_prompt()
    sess = _session()
    _policy(_FakeLLM(responder=resp)).plan_next_intent(sess)
    assert sess.objective.objective_text in seen[0]
    assert sess.objective.success_criterion in seen[0]


def test_prior_payload_truncated_at_500_chars():
    """Boundary: 500-char limit on payload rendering keeps prompts bounded.

    Anti-cheat: catches an "unbounded" history render that would blow up
    after several long turns.
    """
    seen, resp = _capture_prompt()
    huge_payload = "X" * 5_000
    sess = _session(turns=[_turn(0, payload=huge_payload)])
    _policy(_FakeLLM(responder=resp)).plan_next_intent(sess)
    # The full 5000-X string MUST NOT all appear
    assert "X" * 5_000 not in seen[0]
    # But a 500-char prefix DOES appear (truncation, not omission)
    assert "X" * 500 in seen[0]


# =========================================================================
# plan_next_intent — JSON retry + fallback
# =========================================================================


def test_plan_next_intent_retries_on_malformed_json():
    """First response unparseable → retry → returns valid intent."""
    fake = _FakeLLM(
        script={
            "default": [
                "garbage not json",
                _VALID_PLANNER_JSON,
            ]
        },
        tokens_per_call=(10, 5),
    )
    intent, t_in, t_out = _policy(fake).plan_next_intent(_session())
    assert intent.attack_angle == "roleplay_authority"
    # Both attempts charged
    assert (t_in, t_out) == (20, 10)


def test_plan_next_intent_falls_back_after_two_parse_failures():
    """Both responses unparseable → fallback intent with attack_angle='fallback'."""
    fake = _FakeLLM(
        script={"default": ["garbage 1", "garbage 2"]},
    )
    intent, _, _ = _policy(fake).plan_next_intent(_session())
    assert intent.attack_angle == "fallback"


def test_plan_next_intent_falls_back_when_missing_required_key():
    """Response parses but omits required key → fallback fires."""
    fake = _FakeLLM(
        script={
            "default": [
                '{"attack_angle": "x", "rationale": "y"}',  # missing improvement
                '{"attack_angle": "z", "rationale": "w"}',  # still missing
            ]
        },
    )
    intent, _, _ = _policy(fake).plan_next_intent(_session())
    assert intent.attack_angle == "fallback"


# =========================================================================
# should_continue — self-abandon semantics
# =========================================================================


def test_should_continue_true_with_no_history():
    """Fresh session (zero turns) MUST continue.

    Anti-cheat: catches `len(turns) > 0` check that would abandon on turn 0.
    """
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.should_continue(_session(turns=[])) is True


def test_should_continue_true_with_one_no_progress():
    """One NO_PROGRESS is below the threshold of 3."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(turns=[_turn(0, verdict=TurnVerdict.NO_PROGRESS)])
    assert policy.should_continue(sess) is True


def test_should_continue_true_with_two_consecutive_no_progress():
    """Two consecutive NO_PROGRESS — boundary, still under threshold."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is True


def test_should_continue_false_with_three_consecutive_no_progress():
    """Three consecutive NO_PROGRESS triggers self-abandon."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.NO_PROGRESS),
            _turn(2, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is False


def test_should_continue_true_with_three_no_progress_interleaved():
    """NO_PROGRESS, PARTIAL_PROGRESS, NO_PROGRESS, NO_PROGRESS → continue.

    Anti-cheat: catches `count(NO_PROGRESS) >= 3` instead of
    consecutive-streak logic. Total NO_PROGRESS count is 3 but they are
    NOT consecutive in the trailing window.
    """
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.PARTIAL_PROGRESS),
            _turn(2, verdict=TurnVerdict.NO_PROGRESS),
            _turn(3, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    # Trailing 3 are [PARTIAL_PROGRESS, NO_PROGRESS, NO_PROGRESS] — NOT all NO_PROGRESS
    assert policy.should_continue(sess) is True


def test_should_continue_streak_resets_on_partial_progress():
    """NP, NP, NP, PARTIAL, NP, NP → continue (last 3 are PARTIAL, NP, NP)."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.NO_PROGRESS),
            _turn(2, verdict=TurnVerdict.NO_PROGRESS),
            _turn(3, verdict=TurnVerdict.PARTIAL_PROGRESS),
            _turn(4, verdict=TurnVerdict.NO_PROGRESS),
            _turn(5, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is True


def test_should_continue_streak_resets_on_success_verdict():
    """A SUCCESS verdict mid-history resets the streak — last 3 must all be NP."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.NO_PROGRESS),
            _turn(2, verdict=TurnVerdict.SUCCESS),
            _turn(3, verdict=TurnVerdict.NO_PROGRESS),
            _turn(4, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    # Trailing 3: SUCCESS, NO_PROGRESS, NO_PROGRESS → continue
    assert policy.should_continue(sess) is True


def test_should_continue_handles_trailing_three_exactly_no_progress():
    """Exactly 3 turns (all NO_PROGRESS) → False at boundary.

    Anti-cheat: catches `>` vs `>=` confusion on the streak length.
    """
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    sess = _session(
        turns=[
            _turn(0, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, verdict=TurnVerdict.NO_PROGRESS),
            _turn(2, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is False


# =========================================================================
# Mutation isolation (anti-cheat: no shared state between sessions)
# =========================================================================


def test_policy_handles_back_to_back_sessions_independently():
    """One policy instance, two sessions in sequence — no state leak.

    Anti-cheat: catches accidental class-level history or memoization.
    """
    fake = _FakeLLM(responder=lambda p: (_VALID_PLANNER_JSON, 0, 0))
    policy = _policy(fake)
    sess1 = _session(turns=[_turn(0)])
    sess2 = _session(turns=[])  # fresh
    policy.plan_next_intent(sess1)
    # The second session has no turns; should_continue must return True
    # (not be influenced by sess1's NO_PROGRESS history)
    assert policy.should_continue(sess2) is True


# =========================================================================
# Configuration
# =========================================================================


def test_max_turns_default_is_eight():
    """Default budget is 8 turns per RFC § 2.3."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.max_turns == 8


def test_max_turns_overridable_via_constructor():
    """A non-default max_turns is stored verbatim."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)), max_turns=4)
    assert policy.max_turns == 4


def test_class_constant_self_abandon_after_n_no_progress_is_three():
    """The self-abandon threshold is documented as 3 per RFC § 2.3."""
    assert PAIRPolicy.SELF_ABANDON_AFTER_N_NO_PROGRESS == 3
