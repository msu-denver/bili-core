"""MASExecutor-to-PROBE coercion helpers.

The victim side of PROBE is a real AETHER MAS, invoked through
``MASExecutor.run``. Its return value (a ``MASExecutionResult`` in production,
or a plain dict from the in-process stub) has to be coerced into the small
dict shape the PROBE observer and judge consume. Keeping that coercion here,
out of :mod:`bili.aegis.probe.attacker_mas`, leaves the attacker loop focused
on turn / budget / exception logic.

All four helpers are deliberately defensive: a victim that returns an
unexpected shape yields empty signals or a ``repr`` fallback rather than
crashing the run.
"""

from __future__ import annotations

from typing import Any


def _victim_output_text(victim_output: dict[str, Any]) -> str:
    """Render the victim MAS state as a single string for the judge prompt.

    Picks the last LangChain message's content if present; falls back to
    a compact summary of agent_results' outputs; falls back further to
    ``repr(victim_output)``.
    """
    messages = victim_output.get("messages") or []
    if isinstance(messages, list) and messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        if isinstance(content, str) and content:
            return content
    agent_results = victim_output.get("agent_results") or []
    if isinstance(agent_results, list) and agent_results:
        parts = []
        for entry in agent_results:
            if isinstance(entry, dict):
                output = entry.get("output_state") or entry.get("output")
                if output:
                    parts.append(f"{entry.get('agent_id', '?')}: {output}")
        if parts:
            return "\n".join(parts)
    return repr(victim_output)


def _extract_victim_tokens(victim_output: dict[str, Any]) -> int:
    """Sum per-agent token usage when MASExecutor surfaces it; else 0.

    Defensive: any error or missing field returns 0 rather than crashing.
    """
    agent_results = victim_output.get("agent_results") or []
    if not isinstance(agent_results, list):
        return 0
    total = 0
    for entry in agent_results:
        if not isinstance(entry, dict):
            continue
        for key in ("total_tokens", "tokens_used", "token_count"):
            value = entry.get(key)
            if isinstance(value, int):
                total += value
                break
    return total


def _agent_result_to_dict(agent_result: Any) -> dict[str, Any]:
    """Best-effort conversion of an AgentExecutionResult into a plain dict.

    Falls back to ``repr`` on unknown shapes so the observer doesn't crash
    on a victim that returns weird per-agent payloads.
    """
    if isinstance(agent_result, dict):
        return agent_result
    out: dict[str, Any] = {}
    for attr in (
        "agent_id",
        "role",
        "input_state",
        "output_state",
        "output",
        "tokens_used",
        "total_tokens",
    ):
        value = getattr(agent_result, attr, None)
        if value is not None:
            out[attr] = value
    if not out:
        out = {"repr": repr(agent_result)}
    return out


def _victim_result_to_dict(victim_result: Any) -> dict[str, Any]:
    """Coerce a MASExecutionResult (or test mock) into the dict the observer
    and evaluator expect.

    The result may already be a dict (test mocks), in which case it is
    returned unchanged. A structured result is flattened via its
    ``final_state`` plus per-agent results.
    """
    if isinstance(victim_result, dict):
        return victim_result
    out: dict[str, Any] = {}
    final_state = getattr(victim_result, "final_state", None)
    if isinstance(final_state, dict):
        out.update(final_state)
    agent_results = getattr(victim_result, "agent_results", None)
    if agent_results is not None:
        out["agent_results"] = [_agent_result_to_dict(a) for a in agent_results]
    return out
