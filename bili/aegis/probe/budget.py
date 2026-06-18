"""
Budget enforcement for PROBE sessions.

Multi-turn adaptive attackers are the most cost-intensive thing AEGIS contains.
Budget enforcement is non-optional: a session that would exceed any limit is
force-terminated with `ProbeOutcomeReason.BUDGET_EXCEEDED`.

See RFC § 9.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BudgetState:  # pylint: disable=too-many-instance-attributes  # 4 limits + 4 used + price_table (reserved for v0.2) = 9 fields
    """
    Tracks all four budget axes for a session.

    Limits with value None are unbounded for that axis. At least one axis
    must be bounded; constructor enforces this.

    Cost is supplied by the caller in `record_turn()`. PROBE v0.1 lets the
    runner compute cost per call (since it knows the model_name); a future
    `price_table` field is reserved for a pluggable model→price lookup.

    Note: `can_continue()` uses ``>=`` against limits, so an axis at-limit
    is treated as exhausted. A budget of `max_turns=8` permits exactly
    8 turns (turns_used 0..7) and stops at the 9th check.
    """

    max_turns: int | None = 12
    max_tokens_total: int | None = 200_000
    max_wall_clock_seconds: float | None = 300.0
    max_cost_usd: float | None = 5.00

    # Running totals (updated via record_turn)
    turns_used: int = 0
    tokens_used: int = 0
    wall_clock_seconds_used: float = 0.0
    estimated_cost_usd: float = 0.0

    # Token-to-cost lookup; reserved for v0.2
    price_table: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce at least one bounded axis and reject negative limits."""
        all_unbounded = all(
            axis is None
            for axis in (
                self.max_turns,
                self.max_tokens_total,
                self.max_wall_clock_seconds,
                self.max_cost_usd,
            )
        )
        if all_unbounded:
            raise ValueError(
                "BudgetState requires at least one bounded axis "
                "(max_turns, max_tokens_total, max_wall_clock_seconds, "
                "or max_cost_usd)."
            )
        for name, value in (
            ("max_turns", self.max_turns),
            ("max_tokens_total", self.max_tokens_total),
            ("max_wall_clock_seconds", self.max_wall_clock_seconds),
            ("max_cost_usd", self.max_cost_usd),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative; got {value!r}")

    def remaining_turns(self) -> float:
        """Turns left before turn-axis exhaustion, or ``math.inf`` if unbounded."""
        if self.max_turns is None:
            return math.inf
        return float(max(0, self.max_turns - self.turns_used))

    def can_continue(self) -> bool:
        """True iff every bounded axis still has headroom.

        Returns False as soon as ANY bounded axis is at or above its limit.
        Unbounded axes (None) never gate. Logically: AND across bounded axes.
        """
        if self.max_turns is not None and self.turns_used >= self.max_turns:
            return False
        if (
            self.max_tokens_total is not None
            and self.tokens_used >= self.max_tokens_total
        ):
            return False
        if (
            self.max_wall_clock_seconds is not None
            and self.wall_clock_seconds_used >= self.max_wall_clock_seconds
        ):
            return False
        if (
            self.max_cost_usd is not None
            and self.estimated_cost_usd >= self.max_cost_usd
        ):
            return False
        return True

    def record_turn(
        self,
        turn_tokens: int,
        turn_seconds: float,
        turn_cost_usd: float,
    ) -> None:
        """Accumulate one turn's resource usage into the running totals.

        Caller supplies pre-computed cost; PROBE v0.1 does no model→price
        lookup here. Negative arguments are not validated (caller's job).
        """
        self.turns_used += 1
        self.tokens_used += turn_tokens
        self.wall_clock_seconds_used += turn_seconds
        self.estimated_cost_usd += turn_cost_usd
