"""Tests for :class:`bili.aegis.probe.policies.crescendo.CrescendoPolicy`.

Coverage:
  * name() literal
  * Lazy ladder generation on turn 0; cached on subsequent turns; per-session
    isolation (anti-cheat for shared mutable state).
  * Rung selection state machine: SUCCESS → advance, PARTIAL → reattempt,
    NO_PROGRESS → step back, clamped to [0, last_rung].
  * AttackIntent fields: attack_angle="crescendo_rung_{idx}", rationale =
    ladder[idx], rung_index = idx.
  * should_continue: top-rung-success stop, stuck-rung stop, threshold
    edge cases.
  * Ladder normalization: too few entries padded; too many truncated;
    non-list response falls back.
"""

import json
from typing import Optional

from bili.aegis.probe._llm import _FakeLLM
from bili.aegis.probe.policies.crescendo import CrescendoPolicy
from bili.aegis.probe.schema import ProbeSession, ProbeTurn, TurnVerdict
from bili.aegis.tests.conftest import (
    make_probe_intent,
    make_probe_objective,
    make_probe_session,
    make_probe_turn,
)


def _turn(
    idx: int, rung: int, verdict: TurnVerdict = TurnVerdict.NO_PROGRESS
) -> ProbeTurn:
    """Local helper: Crescendo turn carrying ``rung_index`` and rung-keyed labels.

    The intent uses ``crescendo_rung_{rung}`` as attack_angle and
    ``rung {rung} content`` as rationale, matching what CrescendoPolicy
    itself emits.
    """
    return make_probe_turn(
        turn_index=idx,
        intent=make_probe_intent(
            target_agent_role="r",
            attack_angle=f"crescendo_rung_{rung}",
            rationale=f"rung {rung} content",
            rung_index=rung,
        ),
        payload_text=f"payload for turn {idx}",
        observation_summary="",
        verdict=verdict,
        tier3_reasoning="",
        tier3_confidence="low",
    )


def _session(
    session_id: str = "sess-1",
    turns: Optional[list[ProbeTurn]] = None,
    target_role: Optional[str] = None,
) -> ProbeSession:
    """Local helper: Crescendo session keyed by ``session_id`` and optional ``target_role``."""
    return make_probe_session(
        session_id=session_id,
        objective=make_probe_objective(target_agent_role=target_role),
        policy_name="crescendo",
        turns=turns or [],
    )


def _ladder_json(n: int) -> str:
    """Build a valid `{"ladder": [...]}` JSON string with `n` entries."""
    return json.dumps({"ladder": [f"rung {i} description" for i in range(n)]})


def _policy(
    llm: _FakeLLM,
    ladder_rungs: int = 8,
    max_turns: int = 10,
) -> CrescendoPolicy:
    return CrescendoPolicy(llm=llm, ladder_rungs=ladder_rungs, max_turns=max_turns)


# =========================================================================
# name() + defaults
# =========================================================================


def test_name_returns_crescendo_literal():
    """CSV column value is the literal `"crescendo"` string."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.name() == "crescendo"


def test_default_ladder_rungs_is_eight():
    """Per RFC § 2.3, default ladder is 8 rungs."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.ladder_rungs == 8


def test_default_max_turns_is_ten():
    """Per RFC § 2.3, default budget is 10 turns (2 turns of headroom)."""
    policy = _policy(_FakeLLM(responder=lambda p: ("", 0, 0)))
    assert policy.max_turns == 10


# =========================================================================
# Lazy ladder generation
# =========================================================================


def test_ladder_generated_on_turn_zero():
    """First plan_next_intent call invokes the LLM (ladder generation)."""
    calls = {"n": 0}

    def _resp(*_args) -> tuple[str, int, int]:
        calls["n"] += 1
        return (_ladder_json(8), 50, 25)

    policy = _policy(_FakeLLM(responder=_resp))
    policy.plan_next_intent(_session())
    assert calls["n"] == 1


def test_ladder_generation_tokens_charged_only_on_turn_zero():
    """Subsequent turns reuse the cached ladder; no further LLM calls."""
    calls = {"n": 0}

    def _resp(*_args) -> tuple[str, int, int]:
        calls["n"] += 1
        return (_ladder_json(8), 50, 25)

    policy = _policy(_FakeLLM(responder=_resp))
    sess = _session()
    # Turn 0
    _, t_in_0, t_out_0 = policy.plan_next_intent(sess)
    assert (t_in_0, t_out_0) == (50, 25)
    # Append a turn so plan_next_intent sees turn 1
    sess.turns.append(_turn(0, rung=0, verdict=TurnVerdict.PARTIAL_PROGRESS))
    # Turn 1
    _, t_in_1, t_out_1 = policy.plan_next_intent(sess)
    assert (t_in_1, t_out_1) == (0, 0)
    assert calls["n"] == 1


def test_ladder_isolated_per_session():
    """Two distinct session_ids get separately-generated ladders.

    Anti-cheat: catches shared-state-across-sessions.
    """
    calls = {"n": 0}

    def _resp(*_args) -> tuple[str, int, int]:
        calls["n"] += 1
        return (_ladder_json(8), 0, 0)

    policy = _policy(_FakeLLM(responder=_resp))
    policy.plan_next_intent(_session(session_id="sess-A"))
    policy.plan_next_intent(_session(session_id="sess-B"))
    assert calls["n"] == 2


def test_ladder_generation_prompt_contains_objective_and_success_criterion():
    """The ladder prompt includes objective text and success criterion."""
    seen: list[str] = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        seen.append(prompt)
        return (_ladder_json(8), 0, 0)

    policy = _policy(_FakeLLM(responder=_resp))
    policy.plan_next_intent(_session())
    obj = make_probe_objective()
    assert obj.objective_text in seen[0]
    assert obj.success_criterion in seen[0]


def test_ladder_generation_prompt_includes_rung_count():
    """The ladder prompt specifies the requested ladder size."""
    seen: list[str] = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        seen.append(prompt)
        return (_ladder_json(5), 0, 0)

    policy = _policy(_FakeLLM(responder=_resp), ladder_rungs=5)
    policy.plan_next_intent(_session())
    assert '"ladder" of 5' in seen[0]


# =========================================================================
# Rung selection state machine
# =========================================================================


def test_turn_zero_intent_has_rung_index_zero():
    """Brand-new session → first intent targets rung 0."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    intent, _, _ = policy.plan_next_intent(_session())
    assert intent.rung_index == 0


def test_advances_rung_on_success_verdict():
    """Last turn SUCCESS at rung 3 → next intent rung 4."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=3, verdict=TurnVerdict.SUCCESS)])
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.rung_index == 4


def test_reattempts_rung_on_partial_progress():
    """Last turn PARTIAL at rung 3 → next intent stays at rung 3."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=3, verdict=TurnVerdict.PARTIAL_PROGRESS)])
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.rung_index == 3


def test_steps_back_rung_on_no_progress():
    """Last turn NO_PROGRESS at rung 3 → next intent rung 2."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=3, verdict=TurnVerdict.NO_PROGRESS)])
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.rung_index == 2


def test_rung_clamped_to_zero_on_no_progress_at_rung_zero():
    """Step-back from rung 0 stays at 0 (cannot go negative).

    Anti-cheat: catches an off-by-one that would produce rung -1.
    """
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=0, verdict=TurnVerdict.NO_PROGRESS)])
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.rung_index == 0


def test_rung_clamped_to_last_on_success_at_top():
    """Advance from the last rung stays at the last rung.

    Anti-cheat: catches an off-by-one that would produce rung == ladder_rungs.
    """
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=7, verdict=TurnVerdict.SUCCESS)])
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.rung_index == 7


# =========================================================================
# AttackIntent field plumbing
# =========================================================================


def test_intent_attack_angle_includes_rung_index():
    """The intent's attack_angle encodes the rung for traceability."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    intent, _, _ = policy.plan_next_intent(_session())
    assert intent.attack_angle == "crescendo_rung_0"


def test_intent_rationale_is_ladder_description():
    """Each turn's rationale is the ladder entry for the current rung."""
    fake = _FakeLLM(
        responder=lambda p: (
            json.dumps(
                {
                    "ladder": [
                        "UNIQUE_RUNG_ZERO_xxx",
                        "UNIQUE_RUNG_ONE_yyy",
                    ]
                }
            ),
            0,
            0,
        )
    )
    policy = _policy(fake, ladder_rungs=2)
    intent, _, _ = policy.plan_next_intent(_session())
    assert intent.rationale == "UNIQUE_RUNG_ZERO_xxx"


def test_intent_target_role_propagates_from_objective():
    """objective.target_agent_role flows into the AttackIntent."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(target_role="reviewer_alpha")
    intent, _, _ = policy.plan_next_intent(sess)
    assert intent.target_agent_role == "reviewer_alpha"


# =========================================================================
# should_continue — top-rung-success stop
# =========================================================================


def test_should_continue_false_when_last_rung_succeeded():
    """Top-rung SUCCESS → stop."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=7, verdict=TurnVerdict.SUCCESS)])
    assert policy.should_continue(sess) is False


def test_should_continue_true_when_lower_rung_succeeded():
    """SUCCESS at a non-top rung does NOT stop the session."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=5, verdict=TurnVerdict.SUCCESS)])
    assert policy.should_continue(sess) is True


def test_should_continue_true_when_top_rung_partial():
    """Top rung reached but only PARTIAL — keep climbing."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=7, verdict=TurnVerdict.PARTIAL_PROGRESS)])
    assert policy.should_continue(sess) is True


# =========================================================================
# should_continue — stuck-rung stop
# =========================================================================


def test_should_continue_false_after_two_consecutive_no_progress_same_rung():
    """Stuck-rung: 2 NO_PROGRESS in a row on the SAME rung → stop."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(
        turns=[
            _turn(0, rung=3, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, rung=3, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is False


def test_should_continue_true_after_two_no_progress_different_rungs():
    """2 NO_PROGRESS but on different rungs → continue.

    Anti-cheat: stuck-rung requires SAME rung. Different rungs means the
    policy is exploring; it's not stuck.
    """
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(
        turns=[
            _turn(0, rung=3, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, rung=2, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is True


def test_should_continue_true_with_one_no_progress():
    """One NO_PROGRESS is under the threshold of 2."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(turns=[_turn(0, rung=3, verdict=TurnVerdict.NO_PROGRESS)])
    assert policy.should_continue(sess) is True


def test_should_continue_true_when_streak_interrupted_by_partial():
    """NO_PROGRESS, PARTIAL, NO_PROGRESS → not stuck."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess = _session(
        turns=[
            _turn(0, rung=3, verdict=TurnVerdict.NO_PROGRESS),
            _turn(1, rung=3, verdict=TurnVerdict.PARTIAL_PROGRESS),
            _turn(2, rung=3, verdict=TurnVerdict.NO_PROGRESS),
        ]
    )
    assert policy.should_continue(sess) is True


def test_should_continue_true_with_no_turns():
    """Fresh session before any turn → continue."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    assert policy.should_continue(_session()) is True


# =========================================================================
# Ladder normalization
# =========================================================================


def test_ladder_too_few_entries_is_padded():
    """LLM returned 3 rungs but policy needs 8 → padded with generics."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(3), 0, 0)))
    sess = _session()
    policy.plan_next_intent(sess)
    # After turn 0, the cached ladder must have 8 entries
    stored = policy.peek_ladder(sess.session_id)
    assert stored is not None
    assert len(stored) == 8


def test_ladder_too_many_entries_is_truncated():
    """LLM returned 20 rungs but policy needs 8 → truncated to 8."""
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(20), 0, 0)))
    sess = _session()
    policy.plan_next_intent(sess)
    stored = policy.peek_ladder(sess.session_id)
    assert stored is not None
    assert len(stored) == 8


def test_ladder_non_list_response_falls_back():
    """LLM returns `{"ladder": "not a list"}` → fallback ladder."""
    fake = _FakeLLM(responder=lambda p: ('{"ladder": "garbage"}', 0, 0))
    policy = _policy(fake)
    sess = _session()
    policy.plan_next_intent(sess)
    stored = policy.peek_ladder(sess.session_id)
    assert stored is not None
    assert isinstance(stored, list)
    assert len(stored) == 8


def test_ladder_parse_failure_falls_back_to_stub():
    """LLM returns garbage twice → fallback factory produces a stub ladder."""
    fake = _FakeLLM(
        script={"default": ["garbage 1", "garbage 2"]},
    )
    policy = _policy(fake)
    sess = _session()
    policy.plan_next_intent(sess)
    stored = policy.peek_ladder(sess.session_id)
    assert stored is not None
    assert len(stored) == 8


# =========================================================================
# Mutation isolation (anti-cheat)
# =========================================================================


def test_policy_state_does_not_leak_across_two_sessions():
    """Running sess-A then sess-B → sess-B starts with rung 0, not sess-A's rung.

    Anti-cheat: catches shared `self._current_rung_index` or similar
    class-level state.
    """
    policy = _policy(_FakeLLM(responder=lambda p: (_ladder_json(8), 0, 0)))
    sess_a = _session(
        session_id="A",
        turns=[
            _turn(0, rung=0, verdict=TurnVerdict.SUCCESS),
            _turn(1, rung=1, verdict=TurnVerdict.SUCCESS),
            _turn(2, rung=2, verdict=TurnVerdict.SUCCESS),
        ],
    )
    intent_a, _, _ = policy.plan_next_intent(sess_a)
    assert intent_a.rung_index == 3  # advanced from rung 2 success

    # Now fresh session B — must start at rung 0
    sess_b = _session(session_id="B", turns=[])
    intent_b, _, _ = policy.plan_next_intent(sess_b)
    assert intent_b.rung_index == 0
