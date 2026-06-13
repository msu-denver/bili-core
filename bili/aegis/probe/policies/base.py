"""
AttackPolicy abstract base class.

Policies are the swappable component of PROBE. Each policy controls:
- How the planner reasons about strategy across turns (`plan_next_intent`)
- When the session should stop early (`should_continue`)

The ABC is intentionally narrow — see RFC § 6.4. Future policies (Bayesian
optimization, RL attackers, ensemble) only need to implement these two methods
plus a `name()` for the CSV column.

Note: budget enforcement is *not* the policy's responsibility. The runner
checks `BudgetState.can_continue()` before every turn. Policies can stop
*early* via `should_continue` (e.g. self-abandon when no progress detected
across N turns), but cannot extend a session past the budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bili.aegis.probe.schema import AttackIntent, ProbeSession


class AttackPolicy(ABC):
    """Strategy interface for PROBE attackers."""

    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in the CSV `policy` column."""

    @abstractmethod
    def plan_next_intent(self, session: ProbeSession) -> tuple[AttackIntent, int, int]:
        """Return the next AttackIntent and its (tokens_in, tokens_out) cost.

        The tuple-return shape honors the universal PROBE node contract:
        every component that may invoke an LLM reports its token cost so
        ``AttackerMAS.run_session`` can accumulate it into the
        ``BudgetState`` without inspecting per-component attributes.

        Pure deterministic policies should return ``(intent, 0, 0)``.
        """

    @abstractmethod
    def should_continue(self, session: ProbeSession) -> bool:
        """
        Policy-level early-stop check, run after each turn.

        Returns False to stop the session (will be recorded as
        `attacker_self_abandoned`). Returning True does not bypass budget
        enforcement.
        """
