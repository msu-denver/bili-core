"""Tests for :class:`bili.aegis.probe.nodes.payload_crafter.PayloadCrafterNode`.

Covers:
  * three-tuple return shape (text, tokens_in, tokens_out)
  * HouYi system + user prompt rendering
  * intent/MAS-shape/continuity field plumbing into the LLM prompt
  * first-turn graceful boundary (no prior continuity)
  * tokens flow through from the underlying ProbeLLM
  * distinct intents → distinct payloads (responder-mode verification)
  * defensive against missing victim_mas_shape fields
"""

from typing import Any, Optional

from bili.aegis.probe._llm import _FakeLLM
from bili.aegis.probe.nodes.payload_crafter import PayloadCrafterNode
from bili.aegis.probe.schema import AttackIntent, ProbeSession, ProbeTurn
from bili.aegis.tests.conftest import (
    make_probe_intent,
    make_probe_session,
    make_probe_turn,
)


def _intent(
    angle: str = "appeal-to-authority",
    rationale: str = "agent defers to senior roles",
) -> AttackIntent:
    """Local helper for varying attack_angle / rationale in tests."""
    return make_probe_intent(attack_angle=angle, rationale=rationale)


def _turn(idx: int, angle: str = "x") -> ProbeTurn:
    """Local helper: a prior turn with payload and observation keyed to ``idx``."""
    return make_probe_turn(
        turn_index=idx,
        intent=make_probe_intent(
            target_agent_role="r", attack_angle=angle, rationale="r"
        ),
        payload_text=f"prior payload {idx}",
        propagation_path=["a"],
        observation_summary=f"observation for turn {idx}",
        tier3_reasoning="",
        tier3_confidence="low",
    )


def _session(turns: Optional[list[ProbeTurn]] = None) -> ProbeSession:
    """Local helper: a fresh session with optional prior turns."""
    return make_probe_session(turns=turns or [])


_DEFAULT_SHAPE: dict[str, Any] = {
    "mas_id": "simple_chain",
    "agents": [
        {"agent_id": "manager", "role": "community_manager"},
        {"agent_id": "reviewer", "role": "content_reviewer"},
        {"agent_id": "judge", "role": "judge"},
    ],
    "entry_point": "manager",
}


def _crafter(
    llm: _FakeLLM,
    shape: Optional[dict[str, Any]] = None,
) -> PayloadCrafterNode:
    return PayloadCrafterNode(
        model_config={},
        victim_mas_shape=shape if shape is not None else _DEFAULT_SHAPE,
        llm_override=llm,
    )


# =========================================================================
# Return shape + token flow
# =========================================================================


def test_call_returns_three_tuple_with_string_and_token_ints():
    """Universal node contract: (str, int, int)."""
    fake = _FakeLLM(responder=lambda p: ("Please review this carefully.", 10, 5))
    crafter = _crafter(fake)
    result = crafter(_intent(), _session())
    assert isinstance(result, tuple)
    assert len(result) == 3
    text, t_in, t_out = result
    assert isinstance(text, str)
    assert (t_in, t_out) == (10, 5)


def test_call_strips_surrounding_whitespace_from_llm_response():
    """Trailing/leading newlines from the LLM are stripped before return."""
    fake = _FakeLLM(responder=lambda p: ("\n\n  the actual payload  \n", 0, 0))
    crafter = _crafter(fake)
    text, _, _ = crafter(_intent(), _session())
    assert text == "the actual payload"


# =========================================================================
# Prompt-content assertions (anti-cheat via responder mode)
# =========================================================================


def _capture_prompt():
    """Helper: build a responder that records every prompt it receives."""
    seen: list[str] = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        seen.append(prompt)
        return ("payload-out", 0, 0)

    return seen, _resp


def test_prompt_includes_houyi_keywords_from_system_message():
    """FRAMEWORK / SEPARATOR / PAYLOAD must appear in the LLM input."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), _session())
    prompt = seen[0]
    assert "FRAMEWORK" in prompt
    assert "SEPARATOR" in prompt
    assert "PAYLOAD" in prompt


def test_prompt_includes_intent_attack_angle():
    """The crafter's prompt to the LLM contains the intent.attack_angle."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(angle="ROLE_REVERSAL_42"), _session())
    assert "ROLE_REVERSAL_42" in seen[0]


def test_prompt_includes_intent_rationale():
    """The intent.rationale appears in the prompt."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(
        _intent(rationale="UNIQUE_RATIONALE_STRING_xxx"),
        _session(),
    )
    assert "UNIQUE_RATIONALE_STRING_xxx" in seen[0]


def test_prompt_includes_mas_id():
    """The victim MAS mas_id reaches the crafter prompt."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), _session())
    assert "simple_chain" in seen[0]


def test_prompt_includes_compact_agent_roles_list():
    """Each (role, agent_id) pair appears as `role:id` in the prompt."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), _session())
    assert "community_manager:manager" in seen[0]
    assert "content_reviewer:reviewer" in seen[0]
    assert "judge:judge" in seen[0]


def test_prompt_includes_entry_point():
    """The MAS entry_point appears in the prompt."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), _session())
    assert "manager" in seen[0]


# =========================================================================
# Continuity block (multi-turn coherence)
# =========================================================================


def test_first_turn_uses_no_prior_context_placeholder():
    """Empty session.turns → placeholder string in continuity block."""
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), _session(turns=[]))
    assert "first turn" in seen[0].lower()


def test_subsequent_turn_includes_prior_observation_summaries():
    """Turn 2 prompt contains the prior turns' observation summaries."""
    seen, resp = _capture_prompt()
    sess = _session(
        turns=[
            _turn(0, angle="opener"),
            _turn(1, angle="follow_up"),
        ]
    )
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(angle="next_angle"), sess)
    assert "observation for turn 0" in seen[0]
    assert "observation for turn 1" in seen[0]


def test_continuity_block_uses_only_last_2_turns():
    """Turn-3 prompt does NOT include turn 0 (window is 2)."""
    seen, resp = _capture_prompt()
    sess = _session(
        turns=[
            _turn(0, angle="early"),
            _turn(1, angle="mid_a"),
            _turn(2, angle="mid_b"),
        ]
    )
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(), sess)
    assert "observation for turn 0" not in seen[0]
    assert "observation for turn 1" in seen[0]
    assert "observation for turn 2" in seen[0]


# =========================================================================
# Distinct intents → distinct payloads (responder verifies)
# =========================================================================


def test_distinct_intents_produce_distinct_prompts():
    """Anti-cheat: two intents send two different prompts to the LLM.

    Catches a degenerate implementation that ignores `intent` and
    sends the same prompt each call.
    """
    seen, resp = _capture_prompt()
    crafter = _crafter(_FakeLLM(responder=resp))
    crafter(_intent(angle="angle_A", rationale="rA"), _session())
    crafter(_intent(angle="angle_B", rationale="rB"), _session())
    assert seen[0] != seen[1]
    assert "angle_A" in seen[0] and "angle_A" not in seen[1]
    assert "angle_B" in seen[1] and "angle_B" not in seen[0]


# =========================================================================
# Defensive against missing victim_mas_shape fields
# =========================================================================


def test_missing_mas_id_renders_unknown_placeholder():
    """A shape dict with no `mas_id` produces `<unknown>` in the prompt."""
    seen, resp = _capture_prompt()
    crafter = _crafter(
        _FakeLLM(responder=resp),
        shape={"agents": [], "entry_point": "x"},
    )
    crafter(_intent(), _session())
    assert "<unknown>" in seen[0]


def test_missing_entry_point_renders_unknown_placeholder():
    """Shape without entry_point also degrades gracefully."""
    seen, resp = _capture_prompt()
    crafter = _crafter(
        _FakeLLM(responder=resp),
        shape={"mas_id": "x", "agents": []},
    )
    crafter(_intent(), _session())
    # `<unknown>` appears at least once (for entry_point); empty mas_id ok
    assert "<unknown>" in seen[0]


def test_missing_agents_list_produces_empty_compact_string():
    """An empty agents list renders as an empty `agents:` substring."""
    seen, resp = _capture_prompt()
    crafter = _crafter(
        _FakeLLM(responder=resp),
        shape={"mas_id": "x", "agents": [], "entry_point": "y"},
    )
    crafter(_intent(), _session())
    # No crash; the prompt still rendered
    assert "agents:" in seen[0]


def test_non_dict_agent_entries_are_skipped():
    """Malformed entries in shape['agents'] are silently filtered out."""
    seen, resp = _capture_prompt()
    crafter = _crafter(
        _FakeLLM(responder=resp),
        shape={
            "mas_id": "x",
            "agents": [
                {"agent_id": "good", "role": "ok"},
                "garbage_string",
                None,
                42,
            ],
            "entry_point": "good",
        },
    )
    crafter(_intent(), _session())
    assert "ok:good" in seen[0]
    assert "garbage_string" not in seen[0]
