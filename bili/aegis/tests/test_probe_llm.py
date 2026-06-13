"""Tests for :mod:`bili.aegis.probe._llm`.

Covers ProbeLLM Protocol satisfaction, _FakeLLM script + responder modes,
and _LangChainLLMAdapter token extraction. Real-LLM loading via
resolve_real_llm is mocked so the tests don't require provider credentials.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bili.aegis.probe._llm import (
    _STUB_CRAFTER_OUTPUT,
    _STUB_JUDGE_JSON,
    _STUB_LADDER_JSON,
    _STUB_PLANNER_JSON,
    _STUB_REFINEMENTS_JSON,
    ProbeLLM,
    _FakeLLM,
    _LangChainLLMAdapter,
    _stub_responder,
    _StubVictimExecutor,
    resolve_real_llm,
)


def _fake_chat(content, usage_metadata):
    """Build a minimal LangChain ChatModel stub returning a fixed response.

    The returned object exposes ``.invoke(...)`` (ignoring its args) and
    yields a response carrying the given ``content`` and ``usage_metadata``.
    """
    return SimpleNamespace(
        invoke=lambda *_args, **_kwargs: SimpleNamespace(
            content=content, usage_metadata=usage_metadata
        )
    )


# =========================================================================
# _FakeLLM — mode-selection guards
# =========================================================================


def test_fake_llm_rejects_both_script_and_responder():
    """Constructor refuses both modes simultaneously.

    Catches: silently preferring one over the other.
    """
    with pytest.raises(ValueError, match="rejects both"):
        _FakeLLM(
            script={"default": ["x"]},
            responder=lambda p: ("y", 0, 0),
        )


def test_fake_llm_rejects_neither_script_nor_responder():
    """Constructor refuses both modes empty.

    Catches: a default-fallback behavior that hides config errors.
    """
    with pytest.raises(ValueError, match="exactly one"):
        _FakeLLM()


# =========================================================================
# _FakeLLM — script mode
# =========================================================================


def test_fake_llm_script_returns_scripted_responses_in_order():
    """Calls drain the bucket in order."""
    fake = _FakeLLM(script={"default": ["r1", "r2", "r3"]})
    assert fake.invoke("p1")[0] == "r1"
    assert fake.invoke("p2")[0] == "r2"
    assert fake.invoke("p3")[0] == "r3"


def test_fake_llm_script_raises_when_exhausted():
    """Past the end of the bucket raises AssertionError with the label."""
    fake = _FakeLLM(script={"default": ["r1"]})
    fake.invoke("p")
    with pytest.raises(AssertionError, match="default"):
        fake.invoke("p")


def test_fake_llm_script_raises_for_unknown_label():
    """invoke with no matching bucket raises with a clear message."""
    fake = _FakeLLM(script={"planner": ["r1"]})
    fake.set_label("nonexistent")
    with pytest.raises(AssertionError, match="nonexistent"):
        fake.invoke("p")


def test_fake_llm_set_label_switches_bucket():
    """Tests can drain different buckets in sequence."""
    fake = _FakeLLM(
        script={
            "planner": ["plan_resp"],
            "judge": ["judge_resp"],
        },
        label_when_unscripted="planner",
    )
    assert fake.invoke("p1")[0] == "plan_resp"
    fake.set_label("judge")
    assert fake.invoke("p2")[0] == "judge_resp"


def test_fake_llm_script_buckets_track_independent_cursors():
    """Switching labels mid-test preserves each bucket's cursor.

    Catches: a shared cursor across labels.
    """
    fake = _FakeLLM(
        script={"a": ["a1", "a2"], "b": ["b1"]},
        label_when_unscripted="a",
    )
    fake.invoke("p")  # a → a1
    fake.set_label("b")
    fake.invoke("p")  # b → b1
    fake.set_label("a")
    assert fake.invoke("p")[0] == "a2"


# =========================================================================
# _FakeLLM — responder mode
# =========================================================================


def test_fake_llm_responder_invoked_with_full_prompt():
    """Responder sees the exact prompt string the caller passed.

    Critical for anti-cheat prompt-content assertions in node tests.
    """
    received: list[str] = []

    def _resp(prompt: str) -> tuple[str, int, int]:
        received.append(prompt)
        return "ok", 0, 0

    fake = _FakeLLM(responder=_resp)
    fake.invoke("the actual prompt with multiple lines\nline 2")
    assert received == ["the actual prompt with multiple lines\nline 2"]


def test_fake_llm_responder_returns_tokens_unchanged():
    """Responder's token tuple flows through invoke verbatim."""
    fake = _FakeLLM(responder=lambda p: ("ok", 137, 42))
    assert fake.invoke("anything") == ("ok", 137, 42)


# =========================================================================
# _FakeLLM — token defaults
# =========================================================================


def test_fake_llm_tokens_per_call_default_zero_zero():
    """Default (0, 0) — explicit anti-cheat against accidental nonzero defaults.

    A nonzero default could mask bugs where tests expect zero tokens.
    """
    fake = _FakeLLM(script={"default": ["r"]})
    _, t_in, t_out = fake.invoke("p")
    assert (t_in, t_out) == (0, 0)


def test_fake_llm_tokens_per_call_overridable():
    """Caller can supply non-zero token defaults."""
    fake = _FakeLLM(
        script={"default": ["r"]},
        tokens_per_call=(123, 45),
    )
    _, t_in, t_out = fake.invoke("p")
    assert (t_in, t_out) == (123, 45)


# =========================================================================
# Protocol satisfaction
# =========================================================================


def test_fake_llm_satisfies_probellm_protocol():
    """isinstance(fake, ProbeLLM) is True (runtime-checkable Protocol)."""
    fake = _FakeLLM(script={"default": ["r"]})
    assert isinstance(fake, ProbeLLM)


def test_langchain_adapter_satisfies_probellm_protocol():
    """The real-LLM adapter also satisfies the Protocol."""
    adapter = _LangChainLLMAdapter(chat_model=SimpleNamespace())
    assert isinstance(adapter, ProbeLLM)


# =========================================================================
# _LangChainLLMAdapter
# =========================================================================


def test_langchain_adapter_reads_usage_metadata():
    """When response.usage_metadata is present, tokens flow through."""
    adapter = _LangChainLLMAdapter(
        _fake_chat("hello world", {"input_tokens": 50, "output_tokens": 17})
    )
    text, t_in, t_out = adapter.invoke("anything")
    assert text == "hello world"
    assert t_in == 50
    assert t_out == 17


def test_langchain_adapter_logs_warning_when_usage_metadata_absent(caplog):
    """Without usage_metadata, falls back to (0, 0) and emits a warning."""
    adapter = _LangChainLLMAdapter(_fake_chat("hi", None))
    with caplog.at_level(logging.WARNING, logger="bili.aegis.probe._llm"):
        text, t_in, t_out = adapter.invoke("p")
    assert (text, t_in, t_out) == ("hi", 0, 0)
    assert any("usage_metadata" in rec.message for rec in caplog.records)


def test_langchain_adapter_stringifies_non_string_content():
    """If chat_model.content is not a str, it's coerced via str()."""
    adapter = _LangChainLLMAdapter(
        _fake_chat(["multi", "part", "list"], {"input_tokens": 0, "output_tokens": 0})
    )
    text, _, _ = adapter.invoke("p")
    assert isinstance(text, str)


# =========================================================================
# resolve_real_llm
# =========================================================================


def test_resolve_real_llm_returns_adapter():
    """resolve_real_llm returns something satisfying ProbeLLM."""
    fake_chat = SimpleNamespace()
    with patch("bili.iris.loaders.llm_loader.load_model", return_value=fake_chat):
        result = resolve_real_llm(
            {"model_type": "remote_aws_bedrock", "model_name": "claude-x"}
        )
    assert isinstance(result, ProbeLLM)


def test_resolve_real_llm_passes_config_as_kwargs_to_load_model():
    """The full model_config dict is spread into load_model as kwargs.

    Catches: passing the dict as a positional argument (wrong signature).
    """
    captured: dict = {}

    def _fake_load_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    with patch("bili.iris.loaders.llm_loader.load_model", side_effect=_fake_load_model):
        resolve_real_llm(
            {
                "model_type": "remote_google_vertex",
                "model_name": "gemini-2.5-flash",
                "temperature": 0.2,
            }
        )

    assert captured == {
        "model_type": "remote_google_vertex",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.2,
    }


# =========================================================================
# _stub_responder — one branch per PROBE node type
#
# Anti-cheat: each branch asserts BOTH the response string and the
# (tokens_in, tokens_out) tuple, so a degenerate responder that always
# returns the planner shape (the fallback) would fail every non-planner case.
# =========================================================================


def test_stub_responder_ladder_branch():
    """A Crescendo ladder prompt returns the ladder JSON + its token cost."""
    prompt = "Build the escalation ladder. Return a JSON object with one key."
    assert _stub_responder(prompt) == (_STUB_LADDER_JSON, 80, 40)


def test_stub_responder_refinements_branch():
    """A refinements prompt returns the refinements JSON, not the ladder."""
    assert _stub_responder("Propose refinements to the prior payload.") == (
        _STUB_REFINEMENTS_JSON,
        80,
        40,
    )


def test_stub_responder_judge_branch_via_score_guide():
    """The judge prompt (matched by 'score guide') returns the judge JSON."""
    assert _stub_responder("Follow the score guide below.") == (
        _STUB_JUDGE_JSON,
        30,
        15,
    )


def test_stub_responder_judge_branch_via_original_case_marker():
    """The judge branch also matches the case-sensitive 'Score this turn'."""
    assert _stub_responder("Score this turn against the objective.") == (
        _STUB_JUDGE_JSON,
        30,
        15,
    )


def test_stub_responder_crafter_branch():
    """A crafter prompt (FRAMEWORK + SEPARATOR markers) returns the payload."""
    prompt = "Use the FRAMEWORK and the SEPARATOR to craft a payload."
    assert _stub_responder(prompt) == (_STUB_CRAFTER_OUTPUT, 60, 30)


def test_stub_responder_planner_branch():
    """A planner prompt (improvement + attack_angle) returns the planner JSON."""
    prompt = "Suggest an improvement and a new attack_angle for next turn."
    assert _stub_responder(prompt) == (_STUB_PLANNER_JSON, 50, 25)


def test_stub_responder_fallback_is_planner_shape():
    """An unrecognized prompt falls back to the planner JSON shape."""
    assert _stub_responder("totally unrelated text") == (_STUB_PLANNER_JSON, 50, 25)


# =========================================================================
# _StubVictimExecutor — recording + raise hook
# =========================================================================


def test_stub_victim_executor_records_message_content():
    """run() records the first message's content and echoes it in the result."""
    victim = _StubVictimExecutor()
    result = victim.run({"messages": [SimpleNamespace(content="hello victim")]})
    assert victim.calls == 1
    assert victim.received == ["hello victim"]
    assert result["agent_results"][0]["input_state"]["prompt"] == "hello victim"


def test_stub_victim_executor_handles_empty_input():
    """run() with no messages still counts the call and leaves prompt empty.

    Covers the ``received[-1] if self.received else ""`` empty branch.
    """
    victim = _StubVictimExecutor()
    result = victim.run({})
    assert victim.calls == 1
    assert not victim.received
    assert result["agent_results"][0]["input_state"]["prompt"] == ""


def test_stub_victim_executor_counts_calls_across_invocations():
    """``calls`` accumulates across multiple run() invocations."""
    victim = _StubVictimExecutor()
    victim.run({})
    victim.run({"messages": [SimpleNamespace(content="x")]})
    assert victim.calls == 2


def test_stub_victim_executor_raises_when_configured():
    """A ``raises`` hook makes run() raise instead of returning."""
    victim = _StubVictimExecutor(raises=RuntimeError("victim exploded"))
    with pytest.raises(RuntimeError, match="victim exploded"):
        victim.run({"messages": [SimpleNamespace(content="x")]})


def test_stub_victim_executor_satisfies_canned_shape():
    """The canned result has the messages + agent_results MASExecutor shape."""
    result = _StubVictimExecutor().run({})
    assert not result["messages"]
    assert result["agent_results"][0]["role"] == "reviewer"
