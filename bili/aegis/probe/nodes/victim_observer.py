"""
Victim-observer node — fast, deterministic propagation signals per turn.

v0.1 design (deterministic, no LLM call) — see RFC § 5 and § 7.2:

The observer is intentionally NOT the success judge. It produces cheap,
mechanical signals that feed back into the planner so it can reason about
which agents are absorbing the attack and which are resisting. The
SuccessEvaluatorNode does the formal Tier 3 judgement separately.

The observer wraps :class:`bili.aegis.attacks.propagation.PropagationTracker`
and produces, per turn:
- ``propagation_path``    — agent IDs in execution order
- ``influenced_agents``   — agents whose output contains the payload OR a
                            compliance marker for the attack type
- ``resistant_agents``    — agents that received the payload but did not
                            propagate it (deterministically derivable per-turn
                            as path − influenced, but PropagationTracker
                            already exposes it sorted for stability)
- ``observation_summary`` — single-line procedural string for the planner's
                            next-turn prompt context

v0.2 will add an LLM-driven qualitative summary; the ``model_config``
parameter is kept on the constructor for forward compatibility.

Expected ``victim_output`` shape (produced by AttackerMAS.run_session from
``MASExecutor.run`` results, Commit I):

    {
        "agent_results": [
            {"agent_id": str, "role": str,
             "input_state": dict, "output_state": dict},
            ...
        ],
    }

Defensive against missing keys: callers that forget ``agent_results`` get
empty signals and a "no agents observed" summary rather than a crash.
"""

from __future__ import annotations

from typing import Any, Optional

from bili.aegis.attacks.propagation import PropagationTracker
from bili.aegis.probe.schema import ProbeSession

_EXCERPT_MAX_CHARS = 200
_NO_AGENTS_SUMMARY = "No agents observed in victim output."


def _resolve_target_agent_id(
    session: ProbeSession,
    default_role: Optional[str],
    agent_results: list[dict[str, Any]],
) -> str:
    """Pick a target_agent_id for PropagationTracker anchoring.

    Priority: session.objective.target_agent_role → constructor default →
    first agent in agent_results → ``"unknown"``.
    """
    if session.objective.target_agent_role:
        return session.objective.target_agent_role
    if default_role:
        return default_role
    if agent_results:
        first = agent_results[0]
        if isinstance(first, dict) and first.get("agent_id"):
            return str(first["agent_id"])
    return "unknown"


def _last_output_excerpt(agent_results: list[dict[str, Any]]) -> str:
    """First ``_EXCERPT_MAX_CHARS`` of the last agent's output as text."""
    if not agent_results:
        return ""
    last = agent_results[-1]
    if not isinstance(last, dict):
        return ""
    output_state = last.get("output_state", {})
    text = str(output_state) if output_state else ""
    if len(text) > _EXCERPT_MAX_CHARS:
        return text[:_EXCERPT_MAX_CHARS] + "…"
    return text


class VictimObserverNode:  # pylint: disable=too-few-public-methods  # single-__call__ observer node by design
    """Per-turn deterministic propagation observer.

    Args:
        model_config: Reserved for v0.2's LLM-driven qualitative summary.
            Unused in v0.1; pass ``None`` or ``{}`` (both become ``{}``).
        default_target_role: Fallback target_agent_id when
            ``session.objective.target_agent_role`` is None. Pass the
            victim MAS's entry-point agent_id if you have it; otherwise
            the observer falls back to the first agent in execution
            order, then to ``"unknown"``.
    """

    def __init__(
        self,
        model_config: Optional[dict[str, Any]] = None,
        default_target_role: Optional[str] = None,
    ) -> None:
        # A ``None`` model_config is treated as the empty-dict default.
        self.model_config = model_config if model_config is not None else {}
        self.default_target_role = default_target_role

    def __call__(
        self,
        payload_text: str,
        victim_output: dict[str, Any],
        session: ProbeSession,
    ) -> tuple[dict[str, Any], int, int]:
        """Observe one turn's victim output and produce propagation signals.

        Returns:
            ``({"observation_summary": str,
                "propagation_path": list[str],
                "influenced_agents": list[str],
                "resistant_agents": list[str]},
              0, 0)`` — the ``(0, 0)`` token counts honor the universal
            node return contract even though v0.1 makes no LLM call.
        """
        agent_results = victim_output.get("agent_results") or []
        # Tolerate non-list (e.g. None or dict) values gracefully
        if not isinstance(agent_results, list):
            agent_results = []

        target_agent_id = _resolve_target_agent_id(
            session, self.default_target_role, agent_results
        )

        tracker = PropagationTracker(
            payload=payload_text,
            target_agent_id=target_agent_id,
        )

        for entry in agent_results:
            if not isinstance(entry, dict):
                continue
            agent_id = entry.get("agent_id")
            role = entry.get("role")
            if not agent_id or not role:
                continue
            tracker.observe(
                agent_id=str(agent_id),
                role=str(role),
                input_state=entry.get("input_state") or {},
                output_state=entry.get("output_state") or {},
                attack_type="prompt_injection",
            )

        propagation_path = tracker.propagation_path()
        influenced = tracker.influenced_agents()
        resistant = tracker.resistant_agents()

        if not propagation_path:
            summary = _NO_AGENTS_SUMMARY
        else:
            excerpt = _last_output_excerpt(agent_results)
            summary = (
                f"Path: {propagation_path}. "
                f"Influenced: {sorted(set(influenced))}. "
                f"Resistant: {resistant}. "
                f"Final excerpt: {excerpt}"
            )

        return (
            {
                "observation_summary": summary,
                "propagation_path": propagation_path,
                "influenced_agents": sorted(set(influenced)),
                "resistant_agents": resistant,
            },
            0,
            0,
        )
