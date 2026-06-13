"""Tests for :mod:`bili.aegis.probe._mas_executor_adapter`.

The four coercion helpers turn a MASExecutor result (a structured
MASExecutionResult in production, or a plain dict from the stub) into the
shapes the PROBE observer and judge consume. Every helper is defensive, so
the tests cover the unknown-shape fallbacks alongside the happy paths.
"""

from types import SimpleNamespace

from bili.aegis.probe._mas_executor_adapter import (
    _agent_result_to_dict,
    _extract_victim_tokens,
    _victim_output_text,
    _victim_result_to_dict,
)

# =========================================================================
# _victim_output_text
# =========================================================================


def test_victim_output_text_uses_last_message_content_when_present():
    """Picks the last LangChain message's content."""
    msg = SimpleNamespace(content="the victim said hello")
    out = _victim_output_text({"messages": [msg]})
    assert out == "the victim said hello"


def test_victim_output_text_falls_back_to_agent_results():
    """When no messages, render agent_results outputs."""
    out = _victim_output_text(
        {
            "agent_results": [
                {"agent_id": "a", "output_state": "first agent text"},
                {"agent_id": "b", "output_state": "second agent text"},
            ]
        }
    )
    assert "first agent text" in out
    assert "second agent text" in out


def test_victim_output_text_falls_back_to_repr_for_unknown_shape():
    """Unknown shape → repr() so the judge still has SOMETHING to score."""
    out = _victim_output_text({"weird_key": [1, 2, 3]})
    assert "weird_key" in out


def test_victim_output_text_reads_dict_shaped_message_content():
    """A dict-shaped last message exposes its content via the 'content' key."""
    out = _victim_output_text({"messages": [{"content": "dict message text"}]})
    assert out == "dict message text"


# =========================================================================
# _extract_victim_tokens
# =========================================================================


def test_extract_victim_tokens_sums_known_keys():
    """total_tokens / tokens_used / token_count keys are summed."""
    output = {
        "agent_results": [
            {"agent_id": "a", "total_tokens": 100},
            {"agent_id": "b", "tokens_used": 50},
            {"agent_id": "c", "token_count": 25},
        ]
    }
    assert _extract_victim_tokens(output) == 175


def test_extract_victim_tokens_returns_zero_when_unavailable():
    """No token keys → 0 (defensive: shouldn't crash)."""
    assert _extract_victim_tokens({"agent_results": [{"agent_id": "a"}]}) == 0
    assert _extract_victim_tokens({}) == 0


def test_extract_victim_tokens_returns_zero_for_non_list_agent_results():
    """A non-list agent_results value is tolerated and yields 0."""
    assert _extract_victim_tokens({"agent_results": "not a list"}) == 0


def test_extract_victim_tokens_skips_non_dict_entries():
    """Non-dict entries in agent_results are skipped, not summed."""
    output = {"agent_results": ["junk", {"total_tokens": 10}]}
    assert _extract_victim_tokens(output) == 10


# =========================================================================
# _agent_result_to_dict
# =========================================================================


def test_agent_result_to_dict_falls_back_to_repr():
    """An object with none of the known attrs → repr-wrapped dict.

    Exercises the fallback that keeps the observer from crashing on a victim
    that returns weird per-agent payloads.
    """
    out = _agent_result_to_dict(SimpleNamespace())
    assert out == {"repr": "namespace()"}


def test_agent_result_to_dict_extracts_named_attributes():
    """Known attributes are picked off into a dict."""
    obj = SimpleNamespace(agent_id="x", role="r", output_state="o", total_tokens=42)
    out = _agent_result_to_dict(obj)
    assert out["agent_id"] == "x"
    assert out["total_tokens"] == 42


def test_agent_result_to_dict_passes_dict_through():
    """A result already in dict form is returned unchanged."""
    payload = {"agent_id": "a", "role": "reviewer"}
    assert _agent_result_to_dict(payload) is payload


# =========================================================================
# _victim_result_to_dict
# =========================================================================


def test_victim_result_to_dict_passes_dict_through():
    """The stub path: a dict result is returned unchanged."""
    payload = {"messages": [], "agent_results": []}
    assert _victim_result_to_dict(payload) is payload


def test_victim_result_to_dict_coerces_structured_result():
    """A structured result merges final_state and coerces each agent_result."""
    result = SimpleNamespace(
        final_state={"messages": ["m"]},
        agent_results=[SimpleNamespace(agent_id="a", role="reviewer")],
    )
    out = _victim_result_to_dict(result)
    assert out["messages"] == ["m"]
    assert out["agent_results"] == [{"agent_id": "a", "role": "reviewer"}]


def test_total_tokens_flow_from_structured_result_to_extract():
    """Per-agent total_tokens surfaces end to end through the adapter chain.

    Mirrors the production path: a MASExecutionResult whose AgentExecutionResults
    carry ``total_tokens`` is coerced by ``_victim_result_to_dict`` and then
    summed by ``_extract_victim_tokens`` for the per-turn victim token count.
    """
    result = SimpleNamespace(
        final_state={"messages": []},
        agent_results=[
            SimpleNamespace(agent_id="a", role="moderator", total_tokens=120),
            SimpleNamespace(agent_id="b", role="judge", total_tokens=80),
        ],
    )
    victim_output = _victim_result_to_dict(result)
    assert _extract_victim_tokens(victim_output) == 200
