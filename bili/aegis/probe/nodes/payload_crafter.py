"""
Payload-crafter node — realizes ``AttackIntent`` into a concrete
victim-facing prompt.

Conditioned on:

- ``AttackIntent``                  (from the planner)
- Victim MAS shape                  (agent roles, entry_point, mas_id —
                                     known statically from the YAML config)
- Last two turns of the session     (for cross-turn conversational
                                     coherence; especially important for
                                     Crescendo-style escalation)

This node is intentionally a different LLM call from the planner, with a
different prompt template. Conceptually: planner = strategist, crafter =
ghostwriter. The crafter prompt follows the HouYi
framework/separator/payload decomposition (Liu et al. 2023, arXiv:2306.05499
§ 3.2).

Returns ``(payload_text, tokens_in, tokens_out)`` per the universal node
contract.
"""

from __future__ import annotations

from typing import Any, Optional

from bili.aegis.probe._llm import ProbeLLM, resolve_real_llm
from bili.aegis.probe._prompts import (
    HOUYI_CRAFTER_SYSTEM_PROMPT,
    HOUYI_CRAFTER_USER_TEMPLATE,
)
from bili.aegis.probe.schema import AttackIntent, ProbeSession

# How many trailing turns to summarize as continuity context for the crafter.
_CONTINUITY_TURN_WINDOW: int = 2

_FIRST_TURN_CONTINUITY_PLACEHOLDER = "(first turn — no prior context)"


def _agents_compact(victim_mas_shape: dict[str, Any]) -> str:
    """Render the victim agent list as a compact "role:id, role:id, ..." string."""
    agents = victim_mas_shape.get("agents") or []
    if not isinstance(agents, list):
        return ""
    parts = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        role = agent.get("role", "")
        agent_id = agent.get("agent_id", "")
        if role and agent_id:
            parts.append(f"{role}:{agent_id}")
        elif agent_id:
            parts.append(str(agent_id))
    return ", ".join(parts)


def _continuity_block(session: ProbeSession) -> str:
    """Render the last ``_CONTINUITY_TURN_WINDOW`` turns for crafter context."""
    if not session.turns:
        return _FIRST_TURN_CONTINUITY_PLACEHOLDER
    recent = session.turns[-_CONTINUITY_TURN_WINDOW:]
    lines = []
    for turn in recent:
        lines.append(
            f"Turn {turn.turn_index} "
            f"(angle: {turn.intent.attack_angle}, "
            f"verdict: {turn.verdict.value}, "
            f"score: {turn.tier3_score}/3): "
            f"{turn.observation_summary}"
        )
    return "\n".join(lines)


class PayloadCrafterNode:  # pylint: disable=too-few-public-methods  # single-__call__ crafter node by design
    """LLM-driven crafter producing one victim-facing prompt per call.

    Args:
        model_config: kwargs passed to
            :func:`bili.aegis.probe._llm.resolve_real_llm`. Must contain
            ``model_type`` and ``model_name`` (plus any
            provider-specific kwargs) when ``llm_override`` is None.
        victim_mas_shape: a dict describing the victim MAS topology.
            The crafter reads ``mas_id`` (str), ``agents`` (list of
            ``{agent_id, role}`` dicts), and ``entry_point`` (str).
            Defensive against missing keys.
        llm_override: test hook; when supplied, ``resolve_real_llm`` is
            NOT called. It is consumed at construction and never stored
            as an attribute, so mutating the node post-init cannot make
            ``_llm`` re-resolve.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        victim_mas_shape: dict[str, Any],
        llm_override: Optional[ProbeLLM] = None,
    ) -> None:
        self.model_config = model_config
        self.victim_mas_shape = victim_mas_shape
        self._llm: ProbeLLM = (
            llm_override if llm_override is not None else resolve_real_llm(model_config)
        )

    def __call__(
        self, intent: AttackIntent, session: ProbeSession
    ) -> tuple[str, int, int]:
        """Render the crafter prompt + invoke the LLM.

        Returns:
            ``(payload_text, tokens_in, tokens_out)``. The crafter does NOT
            attempt JSON parsing — its output is consumed as plain text
            (per the HouYi system prompt instructing "no JSON, no
            commentary, no explanation"). If the LLM ignores that and
            returns JSON anyway, the runner still gets a usable string —
            the victim MAS will receive whatever the LLM produced.
        """
        user_message = HOUYI_CRAFTER_USER_TEMPLATE.format(
            attack_angle=intent.attack_angle,
            rationale=intent.rationale,
            mas_id=self.victim_mas_shape.get("mas_id", "<unknown>"),
            agents_compact=_agents_compact(self.victim_mas_shape),
            entry_point=self.victim_mas_shape.get("entry_point", "<unknown>"),
            continuity_block=_continuity_block(session),
        )
        # System + user messages joined with a clear separator so the
        # responder-mode _FakeLLM can pattern-match either block.
        full_prompt = f"{HOUYI_CRAFTER_SYSTEM_PROMPT}\n\n---\n\n{user_message}"
        response_text, tokens_in, tokens_out = self._llm.invoke(full_prompt)
        # Strip surrounding whitespace; victim MAS gets a clean prompt.
        return response_text.strip(), tokens_in, tokens_out
