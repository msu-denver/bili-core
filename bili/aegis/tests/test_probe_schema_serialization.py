"""Tests for :meth:`ProbeSession.to_csv_row` and :meth:`to_sidecar_json`.

Anti-cheat focus: catches sloppy implementations like ``turns[-1].tier3_score``
(should be max), ``payload_id=session_id`` (should be blank), or forgetting
to round-trip through ``json.dumps``.
"""

import json
from enum import Enum

import pytest

from bili.aegis.probe.schema import (
    PROBE_CSV_COLUMNS,
    ProbeOutcomeReason,
    ProbeSession,
    ProbeTurn,
    TurnVerdict,
)
from bili.aegis.tests.conftest import make_probe_objective as _obj
from bili.aegis.tests.conftest import make_probe_outcome as _outcome
from bili.aegis.tests.conftest import make_probe_session, make_probe_turn


def _turn(idx: int, tier3_score: int = 0, **kwargs) -> ProbeTurn:
    """Local helper: positional ``idx`` / ``tier3_score`` for score-aggregation tests.

    Accepts ``influenced=[...]`` as a friendlier alias for
    ``influenced_agents``. Other kwargs forward to :func:`make_probe_turn`.
    Uses the same per-turn payload / observation / token values the local
    builder originally hard-coded, so tests that inspect those fields keep
    their expected values.
    """
    if "influenced" in kwargs:
        kwargs["influenced_agents"] = kwargs.pop("influenced")
    return make_probe_turn(
        turn_index=idx,
        tier3_score=tier3_score,
        payload_text=f"payload for turn {idx}",
        victim_output={"messages": []},
        observation_summary=f"observation {idx}",
        duration_ms=100.0,
        tokens_attacker=10,
        tokens_victim=20,
        tokens_judge=5,
        **kwargs,
    )


def _session(**kwargs) -> ProbeSession:
    """Local helper: session with id='sess-123' for identity-roundtrip assertions."""
    defaults: dict = {"session_id": "sess-123"}
    defaults.update(kwargs)
    return make_probe_session(**defaults)


# =========================================================================
# PROBE_CSV_COLUMNS constant
# =========================================================================


def test_probe_csv_columns_has_exactly_21_entries():
    """13 cross-suite + 8 PROBE-specific = 21 columns total."""
    assert len(PROBE_CSV_COLUMNS) == 21


def test_probe_csv_columns_includes_all_13_cross_suite_columns():
    """The shared schema is preserved verbatim.

    Each column is named explicitly so a rename in :mod:`schema` surfaces
    here; the tuple-of-strings form keeps the literal under pylint's
    ``min-similarity-lines`` threshold so this test does not need a
    duplicate-code disable.
    """
    cols = set(PROBE_CSV_COLUMNS)
    # fmt: off
    cross_suite = (
        "payload_id", "injection_type", "severity", "stub_mode", "mas_id",
        "phase", "tier1_pass", "tier2_influenced", "tier2_resistant",
        "tier3_score", "tier3_confidence", "tier3_reasoning", "attack_suite",
    )
    # fmt: on
    for name in cross_suite:
        assert name in cols, f"missing cross-suite column: {name}"


def test_probe_csv_columns_includes_all_8_probe_specific_columns():
    """The PROBE-extension columns are present (same drift-detection pattern)."""
    cols = set(PROBE_CSV_COLUMNS)
    # fmt: off
    probe_specific = (
        "session_id", "objective_id", "policy", "rng_seed",
        "turns_used", "budget_used", "turns_to_compromise", "terminated_reason",
    )
    # fmt: on
    for name in probe_specific:
        assert name in cols, f"missing PROBE-specific column: {name}"


# =========================================================================
# to_csv_row — schema
# =========================================================================


def test_to_csv_row_raises_when_final_outcome_none():
    """Premature serialization raises RuntimeError.

    Catches: silently returning blanks when the session isn't finalized.
    """
    sess = _session(final_outcome=None)
    with pytest.raises(RuntimeError, match="final_outcome is None"):
        sess.to_csv_row()


def test_to_csv_row_has_all_21_expected_columns():
    """The returned dict has exactly the 21 columns declared by the constant."""
    sess = _session(final_outcome=_outcome())
    row = sess.to_csv_row()
    assert set(row.keys()) == set(PROBE_CSV_COLUMNS)


def test_to_csv_row_payload_id_blank():
    """PROBE rows leave payload_id as the empty string.

    Catches: ``payload_id = session_id`` shortcut.
    """
    sess = _session(final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["payload_id"] == ""


def test_to_csv_row_phase_blank():
    """PROBE rows leave phase blank."""
    sess = _session(final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["phase"] == ""


def test_to_csv_row_attack_suite_is_probe():
    """Literal value 'probe' (not 'PROBE', not the policy name)."""
    sess = _session(final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["attack_suite"] == "probe"


# =========================================================================
# to_csv_row — outcome → tier1_pass mapping
# =========================================================================


@pytest.mark.parametrize(
    "reason,expected",
    [
        (ProbeOutcomeReason.SUCCESS, "true"),
        (ProbeOutcomeReason.BUDGET_EXCEEDED, "false"),
        (ProbeOutcomeReason.ATTACKER_SELF_ABANDONED, "false"),
        (ProbeOutcomeReason.VICTIM_CRASHED, "false"),
        (ProbeOutcomeReason.ATTACKER_CRASHED, "false"),
        (ProbeOutcomeReason.JUDGE_UNAVAILABLE, "false"),
    ],
)
def test_to_csv_row_tier1_pass_reflects_outcome_reason(reason, expected):
    """Only SUCCESS produces tier1_pass='true'.

    Catches: tier1_pass derived from tier3_score (would be True on partial scores).
    """
    sess = _session(final_outcome=_outcome(reason=reason))
    assert sess.to_csv_row()["tier1_pass"] == expected


# =========================================================================
# to_csv_row — tier3 aggregation (CRITICAL anti-cheat)
# =========================================================================


def test_to_csv_row_tier3_score_is_max_across_turns():
    """tier3_score is the MAX across self.turns, NOT the last turn.

    Anti-cheat: catches ``turns[-1].tier3_score`` lazy implementation.
    Constructs scores [2, 3, 1]: the max is 3 (turn 1) but the LAST
    turn's score is 1. A correct implementation reports 3.
    """
    turns = [
        _turn(idx=0, tier3_score=2),
        _turn(idx=1, tier3_score=3),
        _turn(idx=2, tier3_score=1),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["tier3_score"] == 3


def test_to_csv_row_tier3_reasoning_from_max_score_turn():
    """tier3_reasoning comes from the turn that achieved the max score."""
    turns = [
        _turn(idx=0, tier3_score=1, tier3_reasoning="early partial"),
        _turn(idx=1, tier3_score=3, tier3_reasoning="success here"),
        _turn(idx=2, tier3_score=0, tier3_reasoning="regression"),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    assert sess.to_csv_row()["tier3_reasoning"] == "success here"


def test_to_csv_row_tier3_confidence_from_max_score_turn():
    """tier3_confidence comes from the same winning turn."""
    turns = [
        _turn(idx=0, tier3_score=0, tier3_confidence="high"),
        _turn(idx=1, tier3_score=2, tier3_confidence="low"),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    assert sess.to_csv_row()["tier3_confidence"] == "low"


def test_to_csv_row_tier3_score_zero_when_no_turns():
    """Empty turns list → tier3_score=0, reasoning='', confidence=''."""
    sess = _session(turns=[], final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["tier3_score"] == 0
    assert row["tier3_reasoning"] == ""
    assert row["tier3_confidence"] == ""


# =========================================================================
# to_csv_row — tier2 aggregation
# =========================================================================


def test_to_csv_row_tier2_influenced_is_json_sorted_union():
    """tier2_influenced is JSON-encoded sorted union of per-turn influenced lists.

    Catches: only using last turn's influenced list; not sorting; not deduping.
    """
    turns = [
        _turn(idx=0, tier3_score=0, influenced=["c_agent", "a_agent"]),
        _turn(idx=1, tier3_score=0, influenced=["b_agent", "a_agent"]),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    parsed = json.loads(sess.to_csv_row()["tier2_influenced"])
    assert parsed == ["a_agent", "b_agent", "c_agent"]


def test_to_csv_row_tier2_resistant_derived_from_propagation_path():
    """resistant = (propagation_path − influenced_agents), unioned across turns."""
    turns = [
        _turn(
            idx=0,
            tier3_score=0,
            propagation_path=["a", "b", "c"],
            influenced=["a"],
        ),
        _turn(
            idx=1,
            tier3_score=0,
            propagation_path=["b", "d"],
            influenced=["d"],
        ),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    parsed = json.loads(sess.to_csv_row()["tier2_resistant"])
    # Turn 0 resistant: [b, c]. Turn 1 resistant: [b]. Union sorted: [b, c].
    assert parsed == ["b", "c"]


def test_to_csv_row_tier2_resistant_empty_when_all_influenced():
    """If every agent in propagation_path is influenced, resistant is empty."""
    turns = [
        _turn(
            idx=0,
            tier3_score=0,
            propagation_path=["a", "b"],
            influenced=["a", "b"],
        ),
    ]
    sess = _session(turns=turns, final_outcome=_outcome())
    assert json.loads(sess.to_csv_row()["tier2_resistant"]) == []


# =========================================================================
# to_csv_row — PROBE-specific columns
# =========================================================================


def test_to_csv_row_session_id_objective_id_policy_seed_propagate():
    """Identity fields round-trip from ProbeSession to row."""
    sess = _session(final_outcome=_outcome())
    row = sess.to_csv_row()
    assert row["session_id"] == "sess-123"
    assert row["objective_id"] == "pr_test_001"
    assert row["policy"] == "pair"
    assert row["rng_seed"] == 0


def test_to_csv_row_turns_used_matches_len_turns():
    """turns_used = len(self.turns) exactly."""
    turns = [_turn(i, tier3_score=0) for i in range(4)]
    sess = _session(turns=turns, final_outcome=_outcome())
    assert sess.to_csv_row()["turns_used"] == 4


def test_to_csv_row_budget_used_is_final_outcome_cost():
    """budget_used reflects the final_outcome.estimated_cost_usd."""
    sess = _session(final_outcome=_outcome(estimated_cost_usd=0.42))
    assert sess.to_csv_row()["budget_used"] == 0.42


def test_to_csv_row_turns_to_compromise_blank_on_failure():
    """When TTC is None, the column is blank string (CSV-friendly)."""
    sess = _session(
        final_outcome=_outcome(
            reason=ProbeOutcomeReason.BUDGET_EXCEEDED,
            turns_to_compromise=None,
        )
    )
    assert sess.to_csv_row()["turns_to_compromise"] == ""


def test_to_csv_row_turns_to_compromise_set_on_success():
    """When TTC is an int, it propagates verbatim."""
    sess = _session(
        final_outcome=_outcome(
            reason=ProbeOutcomeReason.SUCCESS,
            turns_to_compromise=3,
        )
    )
    assert sess.to_csv_row()["turns_to_compromise"] == 3


def test_to_csv_row_terminated_reason_is_enum_value_string():
    """terminated_reason is the .value string, not the enum object."""
    sess = _session(final_outcome=_outcome(reason=ProbeOutcomeReason.JUDGE_UNAVAILABLE))
    assert sess.to_csv_row()["terminated_reason"] == "judge_unavailable"


# =========================================================================
# to_csv_row — stub_mode detection
# =========================================================================


def test_to_csv_row_stub_mode_when_model_name_missing():
    """Attacker config without model_name → stub_mode='stub'."""
    sess = _session(
        final_outcome=_outcome(),
        attacker_model_config={},
    )
    assert sess.to_csv_row()["stub_mode"] == "stub"


def test_to_csv_row_stub_mode_when_model_name_present():
    """Attacker config with a real model_name → stub_mode='real'."""
    sess = _session(
        final_outcome=_outcome(),
        attacker_model_config={"model_name": "deepseek-chat"},
    )
    assert sess.to_csv_row()["stub_mode"] == "real"


def test_to_csv_row_injection_type_is_objective_harm_class():
    """injection_type carries the objective's harm_class for cross-suite grouping."""
    sess = ProbeSession(
        session_id="x",
        objective=_obj(harm_class="safety_bypass"),
        victim_mas_id="m",
        victim_mas_path="p",
        policy_name="pair",
        rng_seed=0,
        attacker_model_config={},
        judge_model_config={},
        final_outcome=_outcome(),
    )
    assert sess.to_csv_row()["injection_type"] == "safety_bypass"


# =========================================================================
# to_sidecar_json
# =========================================================================


def test_to_sidecar_json_round_trips_through_json_dumps_loads():
    """The dict survives ``json.dumps(d) → json.loads(s)`` with default=str.

    Catches: enums or non-serializable types leaking through.
    """
    turns = [_turn(idx=0, tier3_score=2, verdict=TurnVerdict.PARTIAL_PROGRESS)]
    sess = _session(turns=turns, final_outcome=_outcome())
    data = sess.to_sidecar_json()
    # Must round-trip; default=str handles any residual non-JSON objects
    # in victim_output (LangChain messages etc).
    restored = json.loads(json.dumps(data, default=str))
    assert restored["session_id"] == "sess-123"


def test_to_sidecar_json_round_trips_without_default_argument():
    """Plain json.dumps (no default=) works for the schema's native fields.

    Catches: leaving Enum objects in the dict (would need default=).
    Note: victim_output may contain non-native types from MAS execution;
    this test uses an empty victim_output to isolate the schema itself.
    """
    sess = _session(
        turns=[_turn(idx=0, tier3_score=0)],
        final_outcome=_outcome(),
    )
    # No default=
    s = json.dumps(sess.to_sidecar_json())
    restored = json.loads(s)
    assert restored["session_id"] == "sess-123"


def test_to_sidecar_json_contains_all_required_top_level_keys():
    """session_id, objective, turns, final_outcome all present."""
    sess = _session(
        turns=[_turn(idx=0, tier3_score=0)],
        final_outcome=_outcome(),
    )
    data = sess.to_sidecar_json()
    for key in ("session_id", "objective", "turns", "final_outcome"):
        assert key in data


def test_to_sidecar_json_includes_victim_model_config():
    """The victim model config is serialized so the run records the victim used.

    Previously ProbeSession had no victim_model_config field, so the sidecar
    omitted it even when the victim ran on a real model.
    """
    sess = _session(
        victim_model_config={
            "model_type": "remote_anthropic",
            "model_name": "claude-sonnet-4-6",
        },
        turns=[_turn(idx=0, tier3_score=0)],
        final_outcome=_outcome(),
    )
    data = sess.to_sidecar_json()
    assert data["victim_model_config"]["model_name"] == "claude-sonnet-4-6"
    restored = json.loads(json.dumps(data, default=str))
    assert restored["victim_model_config"]["model_type"] == "remote_anthropic"


def test_to_sidecar_json_enum_values_are_strings_not_enum_objects():
    """ProbeOutcomeReason and TurnVerdict are pre-converted to .value strings.

    Catches: leaving Enum members in the dict (would fail strict isinstance check).
    """
    turns = [_turn(idx=0, tier3_score=3, verdict=TurnVerdict.SUCCESS)]
    sess = _session(
        turns=turns,
        final_outcome=_outcome(reason=ProbeOutcomeReason.SUCCESS),
    )
    data = sess.to_sidecar_json()
    assert data["final_outcome"]["reason"] == "success"
    assert not isinstance(data["final_outcome"]["reason"], Enum)
    assert data["turns"][0]["verdict"] == "success"
    assert not isinstance(data["turns"][0]["verdict"], Enum)


def test_to_sidecar_json_turn_count_matches_session_turns():
    """Number of serialized turns equals len(session.turns).

    Catches: silent truncation or duplication.
    """
    turns = [_turn(idx=i, tier3_score=0) for i in range(5)]
    sess = _session(turns=turns, final_outcome=_outcome())
    assert len(sess.to_sidecar_json()["turns"]) == 5


def test_to_sidecar_json_preserves_per_turn_token_breakdown():
    """tokens_attacker / tokens_victim / tokens_judge are distinct fields per turn.

    Catches: collapsing tokens into a single number.
    """
    sess = _session(
        turns=[_turn(idx=0, tier3_score=0)],
        final_outcome=_outcome(),
    )
    turn = sess.to_sidecar_json()["turns"][0]
    assert turn["tokens_attacker"] == 10
    assert turn["tokens_victim"] == 20
    assert turn["tokens_judge"] == 5


def test_to_sidecar_json_two_runs_produce_byte_identical_output():
    """Determinism: same session → same JSON.

    Catches: timestamps or random IDs accidentally introduced.
    """
    turns = [_turn(idx=0, tier3_score=1)]
    sess1 = _session(turns=turns, final_outcome=_outcome())
    sess2 = _session(turns=turns, final_outcome=_outcome())
    assert json.dumps(sess1.to_sidecar_json(), sort_keys=True) == json.dumps(
        sess2.to_sidecar_json(), sort_keys=True
    )
