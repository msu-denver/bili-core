"""
Planner node — strategy decisions only, no payload realization.

The planner reads the current :class:`ProbeSession` (objective, full turn
history) and the active :class:`AttackPolicy`, and produces an
:class:`AttackIntent` (target_agent_role, attack_angle, rationale, optional
rung_index).

The intent is intentionally abstract: "exploit the supervisor's tendency
to defer to subordinates" rather than "send the string 'You are now DAN'."
The payload_crafter realizes intents into concrete prompts.

This split is what lets us swap policies without rewriting prompt
engineering. See RFC § 6.

The planner is a thin delegating wrapper around
``policy.plan_next_intent``; LLM invocation and prompt construction live
inside each policy because they are policy-specific (PAIR's full-history
reflection differs from Crescendo's per-rung intent from TAP's tree
expansion). The wrapper exists so the runner has a uniform
``self.planner(session)`` shape and a single hook point for cross-cutting
concerns (telemetry, future caching).
"""

from __future__ import annotations

from typing import Any

from bili.aegis.probe.policies.base import AttackPolicy
from bili.aegis.probe.schema import AttackIntent, ProbeSession


class PlannerNode:  # pylint: disable=too-few-public-methods  # single-__call__ delegation node by design
    """Delegating wrapper over :meth:`AttackPolicy.plan_next_intent`.

    Stateless: all session state lives in :class:`ProbeSession` and all
    policy state lives in the :class:`AttackPolicy` instance. The
    ``model_config`` attribute is held for forward compatibility with
    planner-level telemetry; it is NOT used to construct a separate LLM
    (policies own their own LLM via their own ``__init__``).
    """

    def __init__(self, policy: AttackPolicy, model_config: dict[str, Any]) -> None:
        self.policy = policy
        self.model_config = model_config

    def __call__(self, session: ProbeSession) -> tuple[AttackIntent, int, int]:
        """Produce the next ``(AttackIntent, tokens_in, tokens_out)`` triple.

        Token counts come from the policy (which owns the planner-LLM call);
        deterministic policies will return ``(intent, 0, 0)``.
        """
        return self.policy.plan_next_intent(session)
