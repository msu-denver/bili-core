"""Tests for :class:`bili.aegis.probe.budget.BudgetState`.

Each test names the specific failure mode it catches (anti-cheat philosophy
from the plan: every test must fail under at least one trivial-wrong
implementation, not just under the bug it's named after).
"""

import math

import pytest

from bili.aegis.probe.budget import BudgetState

# =========================================================================
# __post_init__ validation
# =========================================================================


def test_post_init_rejects_all_none_axes():
    """All-None constructor raises ValueError.

    Catches: missing the "at least one bounded axis" check.
    """
    with pytest.raises(ValueError, match="at least one bounded axis"):
        BudgetState(
            max_turns=None,
            max_tokens_total=None,
            max_wall_clock_seconds=None,
            max_cost_usd=None,
        )


def test_post_init_rejects_negative_max_turns():
    """Negative max_turns raises ValueError.

    Catches: missing per-axis non-negative check.
    """
    with pytest.raises(ValueError, match="max_turns"):
        BudgetState(max_turns=-1)


def test_post_init_rejects_negative_max_tokens():
    """Negative max_tokens_total raises ValueError."""
    with pytest.raises(ValueError, match="max_tokens_total"):
        BudgetState(max_tokens_total=-1)


def test_post_init_rejects_negative_max_wall_clock():
    """Negative max_wall_clock_seconds raises ValueError."""
    with pytest.raises(ValueError, match="max_wall_clock_seconds"):
        BudgetState(max_wall_clock_seconds=-0.5)


def test_post_init_rejects_negative_max_cost():
    """Negative max_cost_usd raises ValueError."""
    with pytest.raises(ValueError, match="max_cost_usd"):
        BudgetState(max_cost_usd=-0.01)


@pytest.mark.parametrize(
    "axis_kwargs",
    [
        {
            "max_turns": 1,
            "max_tokens_total": None,
            "max_wall_clock_seconds": None,
            "max_cost_usd": None,
        },
        {
            "max_turns": None,
            "max_tokens_total": 1,
            "max_wall_clock_seconds": None,
            "max_cost_usd": None,
        },
        {
            "max_turns": None,
            "max_tokens_total": None,
            "max_wall_clock_seconds": 1.0,
            "max_cost_usd": None,
        },
        {
            "max_turns": None,
            "max_tokens_total": None,
            "max_wall_clock_seconds": None,
            "max_cost_usd": 0.01,
        },
    ],
)
def test_post_init_accepts_single_bounded_axis(axis_kwargs):
    """Each axis alone is sufficient to satisfy the "at least one bounded" rule.

    Catches: a check that requires ALL axes to be bounded.
    """
    BudgetState(**axis_kwargs)  # must not raise


def test_post_init_accepts_zero_limit():
    """A limit of 0 is bounded (degenerate but valid).

    Catches: a truthy check (``if not value``) that would reject zero.
    """
    state = BudgetState(
        max_turns=0,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    assert state.can_continue() is False  # zero-turn budget is exhausted


# =========================================================================
# remaining_turns
# =========================================================================


def test_remaining_turns_unbounded_returns_inf():
    """When max_turns is None, remaining_turns is math.inf.

    Catches: ``return self.max_turns - self.turns_used`` without None check
    (would raise TypeError).
    """
    state = BudgetState(max_turns=None, max_cost_usd=1.0)
    assert state.remaining_turns() == math.inf


def test_remaining_turns_decrements_with_record_turn():
    """After record_turn, remaining_turns decreases by 1.

    Catches: missing turn-counter increment.
    """
    state = BudgetState(max_turns=5)
    assert state.remaining_turns() == 5
    state.record_turn(turn_tokens=10, turn_seconds=0.1, turn_cost_usd=0.001)
    assert state.remaining_turns() == 4
    state.record_turn(turn_tokens=10, turn_seconds=0.1, turn_cost_usd=0.001)
    assert state.remaining_turns() == 3


def test_remaining_turns_floors_at_zero():
    """remaining_turns never goes negative even after overuse.

    Catches: negative subtraction result (would confuse downstream callers).
    """
    state = BudgetState(max_turns=2)
    for _ in range(5):
        state.record_turn(turn_tokens=1, turn_seconds=0.01, turn_cost_usd=0.0)
    assert state.remaining_turns() == 0


# =========================================================================
# can_continue
# =========================================================================


def test_can_continue_true_when_no_axis_exhausted():
    """Fresh state with non-zero limits → can_continue True."""
    state = BudgetState()
    assert state.can_continue() is True


def test_can_continue_false_when_turns_exhausted():
    """Turn axis at limit → can_continue False.

    Catches: ``can_continue`` that always returns True.
    """
    state = BudgetState(
        max_turns=2,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    state.record_turn(0, 0.0, 0.0)
    state.record_turn(0, 0.0, 0.0)
    assert state.can_continue() is False


def test_can_continue_false_when_tokens_exhausted():
    """Token axis at limit → can_continue False.

    Catches: only-checking-turns implementation.
    """
    state = BudgetState(
        max_turns=None,
        max_tokens_total=100,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    state.record_turn(100, 0.0, 0.0)
    assert state.can_continue() is False


def test_can_continue_false_when_wall_clock_exhausted():
    """Wall-clock axis at limit → can_continue False."""
    state = BudgetState(
        max_turns=None,
        max_tokens_total=None,
        max_wall_clock_seconds=2.0,
        max_cost_usd=None,
    )
    state.record_turn(0, 2.0, 0.0)
    assert state.can_continue() is False


def test_can_continue_false_when_cost_exhausted():
    """Cost axis at limit → can_continue False."""
    state = BudgetState(
        max_turns=None,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=0.50,
    )
    state.record_turn(0, 0.0, 0.50)
    assert state.can_continue() is False


def test_can_continue_uses_and_not_or():
    """One axis exhausted ⇒ False even when others have slack.

    Catches: ``return any(axis_exceeded for ...)`` instead of all-clear logic.
    """
    state = BudgetState(
        max_turns=10,
        max_tokens_total=10,
        max_wall_clock_seconds=10.0,
        max_cost_usd=10.0,
    )
    # Exhaust only tokens; turns / wall-clock / cost still have slack
    state.record_turn(10, 0.1, 0.01)
    assert state.can_continue() is False


def test_can_continue_at_exact_limit_returns_false():
    """``turns_used == max_turns`` is exhausted.

    Catches: ``<`` vs ``<=`` confusion. With 2 turns recorded against
    max_turns=2, the budget is fully consumed.
    """
    state = BudgetState(
        max_turns=2,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    state.record_turn(0, 0.0, 0.0)
    state.record_turn(0, 0.0, 0.0)
    assert state.turns_used == 2
    assert state.can_continue() is False


def test_can_continue_one_below_limit_returns_true():
    """``turns_used == max_turns - 1`` ⇒ can_continue True."""
    state = BudgetState(
        max_turns=3,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    state.record_turn(0, 0.0, 0.0)
    state.record_turn(0, 0.0, 0.0)
    assert state.can_continue() is True


def test_can_continue_unbounded_axis_does_not_gate():
    """A None axis is never the cause of can_continue=False.

    Catches: a None-axis erroneously treated as zero limit.
    """
    state = BudgetState(
        max_turns=10,
        max_tokens_total=None,
        max_wall_clock_seconds=None,
        max_cost_usd=None,
    )
    # Even after a "huge" token charge, can_continue remains True
    state.record_turn(turn_tokens=10**9, turn_seconds=10**6, turn_cost_usd=1.0)
    assert state.can_continue() is True


# =========================================================================
# record_turn
# =========================================================================


def test_record_turn_accumulates_all_four_axes():
    """A single record_turn updates turns, tokens, wall-clock, and cost.

    Catches: forgetting one of the four axes.
    """
    state = BudgetState()
    state.record_turn(turn_tokens=100, turn_seconds=1.5, turn_cost_usd=0.02)
    assert state.turns_used == 1
    assert state.tokens_used == 100
    assert state.wall_clock_seconds_used == 1.5
    assert state.estimated_cost_usd == 0.02


def test_record_turn_multiple_calls_compound():
    """Three calls produce three-fold accumulation.

    Catches: state-replacement (assignment) instead of accumulation.
    """
    state = BudgetState()
    for _ in range(3):
        state.record_turn(turn_tokens=10, turn_seconds=1.0, turn_cost_usd=0.01)
    assert state.turns_used == 3
    assert state.tokens_used == 30
    assert state.wall_clock_seconds_used == 3.0
    assert state.estimated_cost_usd == pytest.approx(0.03)


def test_record_turn_with_zero_values_still_increments_turn_counter():
    """Zero tokens/seconds/cost still counts as a turn.

    Catches: a turn-counter increment that's gated on non-zero token usage.
    """
    state = BudgetState()
    state.record_turn(0, 0.0, 0.0)
    assert state.turns_used == 1
