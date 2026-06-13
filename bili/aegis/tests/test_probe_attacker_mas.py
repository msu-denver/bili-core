"""Tests for :class:`bili.aegis.probe.attacker_mas.AttackerMAS`.

Integration tests for the per-turn loop. Each test injects fake nodes
(policy, crafter, observer, evaluator) via post-initialize replacement so
the loop logic can be exercised without LLM calls.
"""

from typing import Any, Optional

import pytest

from bili.aegis.probe._llm import _FakeLLM, _StubVictimExecutor
from bili.aegis.probe.attacker_mas import AttackerMAS, AttackerModelConfigs
from bili.aegis.probe.budget import BudgetState
from bili.aegis.probe.exceptions import JudgeUnavailableError
from bili.aegis.probe.policies.base import AttackPolicy
from bili.aegis.probe.schema import (
    AttackIntent,
    ProbeOutcomeReason,
    ProbeSession,
    TurnVerdict,
)
from bili.aegis.tests.conftest import make_probe_objective, make_probe_session


def _session(session_id: str = "sess-1") -> ProbeSession:
    """Local helper: session pinned to the smoke trio's attacker/judge configs.

    The objective targets the ``reviewer`` role explicitly so victim-side
    routing tests pick up a non-None target_agent_role.
    """
    return make_probe_session(
        session_id=session_id,
        objective=make_probe_objective(target_agent_role="reviewer"),
        victim_mas_id="m",
        victim_mas_path="p",
        policy_name="stub",
        attacker_model_config={"model_name": "deepseek-chat"},
        judge_model_config={"model_name": "gemini-2.5-flash"},
    )


def _budget(max_turns: int = 5, max_tokens: int = 100_000) -> BudgetState:
    return BudgetState(
        max_turns=max_turns,
        max_tokens_total=max_tokens,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )


class _StubPolicy(AttackPolicy):
    """Deterministic policy. plan_next_intent always returns a fresh intent."""

    def __init__(
        self,
        should_continue_value: bool = True,
        raise_on_plan: Optional[Exception] = None,
    ) -> None:
        self._should_continue = should_continue_value
        self._raise = raise_on_plan

    def name(self) -> str:
        return "stub"

    def plan_next_intent(self, session: ProbeSession) -> tuple[AttackIntent, int, int]:
        if self._raise is not None:
            raise self._raise
        return (
            AttackIntent(
                target_agent_role="reviewer",
                attack_angle="stub_angle",
                rationale="stub rationale",
            ),
            10,
            5,
        )

    def should_continue(self, session: ProbeSession) -> bool:
        return self._should_continue


def _stub_crafter(payload: str = "stub payload", tokens: tuple[int, int] = (20, 10)):
    """Build a PayloadCrafterNode stub: a callable returning a fixed payload + tokens.

    Matches the crafter interface ``(intent, session) -> (text, tokens_in,
    tokens_out)``; both positional args are ignored, so the closure absorbs them.
    """

    def _call(*_args: Any) -> tuple[str, int, int]:
        return payload, tokens[0], tokens[1]

    return _call


def _stub_evaluator(
    score: int = 1,
    verdict: TurnVerdict = TurnVerdict.PARTIAL_PROGRESS,
    tokens: tuple[int, int] = (15, 8),
):
    """Build a SuccessEvaluatorNode stub: a callable returning a fixed verdict + tokens."""

    def _call(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], int, int]:
        return (
            {
                "verdict": verdict,
                "tier3_score": score,
                "tier3_reasoning": "stub reasoning",
                "tier3_confidence": "medium",
            },
            tokens[0],
            tokens[1],
        )

    return _call


_DEFAULT_TRIO = AttackerModelConfigs(
    attacker={"model_name": "deepseek-chat"},
    judge={"model_name": "gemini-2.5-flash"},
    victim={"model_name": "us.anthropic.claude-sonnet-4-6"},
)


def _build_attacker(
    policy: Optional[AttackPolicy] = None,
    crafter: Any = None,
    observer: Any = None,
    evaluator: Any = None,
) -> AttackerMAS:
    """Build an initialized AttackerMAS with optional stubs replacing nodes."""
    attacker = AttackerMAS(
        policy=policy or _StubPolicy(),
        model_configs=_DEFAULT_TRIO,
        victim_mas_shape={"mas_id": "m", "agents": [], "entry_point": "x"},
        crafter_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
        evaluator_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
    )
    attacker.initialize()
    # Replace nodes with stubs AFTER init so JudgeUnavailableError still
    # has a chance to fire during init.
    if crafter is not None:
        attacker.nodes.payload_crafter = crafter
    if observer is not None:
        attacker.nodes.observer = observer
    if evaluator is not None:
        attacker.nodes.evaluator = evaluator
    return attacker


# =========================================================================
# initialize() wiring
# =========================================================================


def test_initialize_wires_planner_with_policy():
    """The planner's policy attribute is the same object passed in."""
    policy = _StubPolicy()
    attacker = AttackerMAS(
        policy=policy,
        model_configs=_DEFAULT_TRIO,
        crafter_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
        evaluator_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
    )
    attacker.initialize()
    assert attacker.nodes.planner is not None
    assert attacker.nodes.planner.policy is policy


def test_initialize_creates_all_four_nodes():
    """All four node attributes are populated after initialize."""
    attacker = _build_attacker()
    assert attacker.nodes.planner is not None
    assert attacker.nodes.payload_crafter is not None
    assert attacker.nodes.observer is not None
    assert attacker.nodes.evaluator is not None


def test_initialize_propagates_judge_unavailable_error():
    """A circular judge/victim/attacker config raises during initialize.

    Anti-cheat: the cross-provider check must fire BEFORE the loop starts,
    not during the first turn.
    """
    # judge family == victim family (both anthropic)
    attacker = AttackerMAS(
        policy=_StubPolicy(),
        model_configs=AttackerModelConfigs(
            attacker={"model_name": "deepseek-chat"},
            judge={"model_name": "us.anthropic.claude-sonnet-4-6"},
            victim={"model_name": "us.anthropic.claude-opus-4-7"},
        ),
        crafter_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
        evaluator_llm_override=_FakeLLM(responder=lambda p: ("", 0, 0)),
    )
    with pytest.raises(JudgeUnavailableError, match="victim family"):
        attacker.initialize()


# =========================================================================
# run_session — termination conditions
# =========================================================================


def test_run_session_requires_initialize_first():
    """Calling run_session without initialize raises RuntimeError."""
    attacker = AttackerMAS(
        policy=_StubPolicy(),
        model_configs=_DEFAULT_TRIO,
    )
    with pytest.raises(RuntimeError, match="initialize"):
        attacker.run_session(_session(), _StubVictimExecutor(), _budget())


def test_run_session_terminates_on_success_verdict():
    """Score == 3 on any turn → SUCCESS termination."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=3, verdict=TurnVerdict.SUCCESS)
    )
    sess = _session()
    sess = attacker.run_session(sess, _StubVictimExecutor(), _budget())
    assert sess.final_outcome is not None
    assert sess.final_outcome.reason == ProbeOutcomeReason.SUCCESS
    # Exactly one turn since success on turn 0
    assert len(sess.turns) == 1


def test_run_session_terminates_on_budget_exceeded():
    """When can_continue returns False before SUCCESS, BUDGET_EXCEEDED fires."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS)
    )
    sess = _session()
    budget = BudgetState(
        max_turns=2,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    sess = attacker.run_session(sess, _StubVictimExecutor(), budget)
    assert sess.final_outcome.reason == ProbeOutcomeReason.BUDGET_EXCEEDED
    assert len(sess.turns) == 2


def test_run_session_terminates_on_policy_self_abandon():
    """policy.should_continue=False after turn → ATTACKER_SELF_ABANDONED."""
    attacker = _build_attacker(
        policy=_StubPolicy(should_continue_value=False),
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS),
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget())
    assert sess.final_outcome.reason == ProbeOutcomeReason.ATTACKER_SELF_ABANDONED
    assert len(sess.turns) == 1


def test_run_session_terminates_on_victim_crash():
    """When victim_executor.run raises, VICTIM_CRASHED is recorded."""
    attacker = _build_attacker()
    sess = attacker.run_session(
        _session(),
        _StubVictimExecutor(raises=RuntimeError("victim exploded")),
        _budget(),
    )
    assert sess.final_outcome.reason == ProbeOutcomeReason.VICTIM_CRASHED
    # No turn was completed (victim crashed mid-turn)
    assert len(sess.turns) == 0


def test_run_session_terminates_on_attacker_crash():
    """Planner raises → ATTACKER_CRASHED is recorded."""
    attacker = _build_attacker(
        policy=_StubPolicy(raise_on_plan=RuntimeError("planner exploded"))
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget())
    assert sess.final_outcome.reason == ProbeOutcomeReason.ATTACKER_CRASHED


def test_run_session_refuses_to_double_finalize():
    """Re-running a finalized session raises RuntimeError."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=3, verdict=TurnVerdict.SUCCESS)
    )
    sess = _session()
    attacker.run_session(sess, _StubVictimExecutor(), _budget())
    # Second call must raise
    with pytest.raises(RuntimeError, match="already finalized"):
        attacker.run_session(sess, _StubVictimExecutor(), _budget())


# =========================================================================
# Per-turn data integrity
# =========================================================================


def test_run_session_appends_turns_with_monotonic_indices():
    """turn_index is 0, 1, 2, ... in order."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS)
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget(max_turns=4))
    indices = [t.turn_index for t in sess.turns]
    assert indices == [0, 1, 2, 3]


def test_run_session_invokes_victim_executor_once_per_turn():
    """Each turn calls victim_executor.run exactly once."""
    victim = _StubVictimExecutor()
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS)
    )
    attacker.run_session(_session(), victim, _budget(max_turns=3))
    assert victim.calls == 3


def test_run_session_passes_crafted_payload_to_victim():
    """The crafter's output reaches the victim as a HumanMessage."""
    victim = _StubVictimExecutor()
    crafter = _stub_crafter(payload="UNIQUE_PAYLOAD_MARKER")
    attacker = _build_attacker(
        crafter=crafter,
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS),
    )
    attacker.run_session(_session(), victim, _budget(max_turns=1))
    assert victim.received[0] == "UNIQUE_PAYLOAD_MARKER"


def test_run_session_per_turn_token_breakdown_matches_node_returns():
    """tokens_attacker = planner_tokens + crafter_tokens; tokens_judge = evaluator_tokens."""
    attacker = _build_attacker(
        crafter=_stub_crafter(tokens=(20, 10)),  # 30 total
        evaluator=_stub_evaluator(
            score=1, verdict=TurnVerdict.PARTIAL_PROGRESS, tokens=(15, 8)
        ),  # 23 total
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget(max_turns=1))
    turn = sess.turns[0]
    # Planner stub returns (10, 5), so planner contributes 15.
    # Crafter contributes 30. Total attacker = 45.
    assert turn.tokens_attacker == 15 + 30
    assert turn.tokens_judge == 15 + 8


def test_run_session_records_into_budget_each_turn():
    """budget.turns_used == len(session.turns); tokens accumulate."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS)
    )
    budget = _budget(max_turns=3)
    sess = attacker.run_session(_session(), _StubVictimExecutor(), budget)
    assert budget.turns_used == 3
    assert budget.turns_used == len(sess.turns)


# =========================================================================
# final_outcome correctness
# =========================================================================


def test_final_outcome_tier3_score_is_max_across_turns():
    """final_tier3_score = max of per-turn tier3_score."""

    # Custom evaluator that returns score 2 on first call then 0 then 1
    scores = [2, 0, 1]
    state = {"i": 0}

    def _vary_evaluator(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], int, int]:
        score = scores[state["i"]]
        state["i"] += 1
        return (
            {
                "verdict": TurnVerdict.PARTIAL_PROGRESS,
                "tier3_score": score,
                "tier3_reasoning": "",
                "tier3_confidence": "low",
            },
            0,
            0,
        )

    attacker = _build_attacker(evaluator=_vary_evaluator)
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget(max_turns=3))
    assert sess.final_outcome.final_tier3_score == 2


def test_final_outcome_turns_to_compromise_only_set_on_success():
    """SUCCESS → TTC = turn_index of first score==3 turn. Non-success → None."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=3, verdict=TurnVerdict.SUCCESS)
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget())
    assert sess.final_outcome.turns_to_compromise == 0


def test_final_outcome_ttc_none_on_budget_exceeded():
    """When session ends without SUCCESS, TTC is None."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS)
    )
    sess = attacker.run_session(
        _session(),
        _StubVictimExecutor(),
        BudgetState(
            max_turns=2,
            max_tokens_total=None,
            max_wall_clock_seconds=None,
            max_cost_usd=None,
        ),
    )
    assert sess.final_outcome.turns_to_compromise is None


def test_final_outcome_aggregates_per_turn_tokens():
    """total_tokens_* is the sum across turns."""
    attacker = _build_attacker(
        crafter=_stub_crafter(tokens=(20, 10)),
        evaluator=_stub_evaluator(
            score=1, verdict=TurnVerdict.PARTIAL_PROGRESS, tokens=(15, 8)
        ),
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget(max_turns=3))
    # Per turn: planner (15) + crafter (30) = 45 attacker; judge 23. × 3 turns.
    assert sess.final_outcome.total_tokens_attacker == 45 * 3
    assert sess.final_outcome.total_tokens_judge == 23 * 3


def test_final_outcome_total_duration_ms_is_positive():
    """The session-level duration is non-zero."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=3, verdict=TurnVerdict.SUCCESS)
    )
    sess = attacker.run_session(_session(), _StubVictimExecutor(), _budget())
    assert sess.final_outcome.total_duration_ms > 0


# =========================================================================
# State threading across turns (anti-cheat)
# =========================================================================


def test_planner_sees_growing_session_turns_across_turns():
    """At each call, planner.plan_next_intent sees the updated session.turns."""
    captured_turn_counts: list[int] = []

    class _CountingPolicy(AttackPolicy):
        def name(self) -> str:
            return "counting"

        def plan_next_intent(
            self, session: ProbeSession
        ) -> tuple[AttackIntent, int, int]:
            captured_turn_counts.append(len(session.turns))
            return (
                AttackIntent(
                    target_agent_role="r",
                    attack_angle="x",
                    rationale="y",
                ),
                0,
                0,
            )

        def should_continue(self, session: ProbeSession) -> bool:
            return True

    attacker = _build_attacker(
        policy=_CountingPolicy(),
        evaluator=_stub_evaluator(score=1, verdict=TurnVerdict.PARTIAL_PROGRESS),
    )
    attacker.run_session(_session(), _StubVictimExecutor(), _budget(max_turns=3))
    # Turn 0: planner sees 0 prior turns. Turn 1: sees 1. Turn 2: sees 2.
    assert captured_turn_counts == [0, 1, 2]


def test_session_id_unchanged_after_run():
    """Anti-cheat: the loop must not mutate session.session_id."""
    attacker = _build_attacker(
        evaluator=_stub_evaluator(score=3, verdict=TurnVerdict.SUCCESS)
    )
    sess = _session(session_id="original-id-xyz")
    attacker.run_session(sess, _StubVictimExecutor(), _budget())
    assert sess.session_id == "original-id-xyz"
