"""Tests for Anthropic prompt-caching helpers in :mod:`anthropic_provider`.

These cover both halves of the caching feature:

- The selection helpers (:func:`_is_anthropic_model`,
  :func:`build_prompt_caching_middleware`) decide *whether* caching applies and
  degrade to ``None`` for every non-Anthropic case.
- The end-to-end evidence: the middleware places an ``ephemeral`` cache
  breakpoint on the stable prefix of the request, and a reused prefix reports
  cache-read usage on the following call.  The Anthropic client is mocked, so no
  API key or network is required.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bili.iris.providers.anthropic_provider import (
    _is_anthropic_model,
    build_prompt_caching_middleware,
)

# ChatAnthropic construction reads the API key but makes no network call; a
# placeholder key keeps the whole module offline.
_FAKE_KEY = "sk-ant-test-key-not-real"
_MODEL = "claude-opus-4-8"


def _chat():
    """Return an offline ``ChatAnthropic`` instance for tests."""
    return ChatAnthropic(model=_MODEL, api_key=_FAKE_KEY, max_tokens=16)


# ---------------------------------------------------------------------------
# _is_anthropic_model
# ---------------------------------------------------------------------------


class TestIsAnthropicModel:
    """Model-type detection, including transparent-proxy unwrapping."""

    def test_true_for_chatanthropic(self):
        """A ChatAnthropic instance is detected as Anthropic."""
        assert _is_anthropic_model(_chat()) is True

    def test_false_for_plain_object(self):
        """A plain object (no primary attribute) is not Anthropic."""
        assert _is_anthropic_model(object()) is False

    def test_false_for_string(self):
        """A non-model value is not Anthropic."""
        assert _is_anthropic_model("not a model") is False

    def test_unwraps_fallback_primary_anthropic(self):
        """A FallbackLLM whose primary is Anthropic is recognised."""
        proxy = SimpleNamespace(primary=_chat())
        assert _is_anthropic_model(proxy) is True

    def test_fallback_primary_non_anthropic(self):
        """A FallbackLLM whose primary is not Anthropic is not recognised."""
        proxy = SimpleNamespace(primary=object())
        assert _is_anthropic_model(proxy) is False

    def test_false_when_langchain_anthropic_missing(self):
        """When langchain_anthropic cannot be imported, the answer is False."""
        with patch.dict(sys.modules, {"langchain_anthropic": None}):
            assert _is_anthropic_model(object()) is False


# ---------------------------------------------------------------------------
# build_prompt_caching_middleware
# ---------------------------------------------------------------------------


class TestBuildPromptCachingMiddleware:
    """The middleware factory returns a middleware only for Anthropic models."""

    def test_returns_middleware_for_chatanthropic(self):
        """A ChatAnthropic model yields a caching middleware."""
        mw = build_prompt_caching_middleware(_chat())
        assert isinstance(mw, AnthropicPromptCachingMiddleware)

    def test_returns_middleware_for_fallback_with_anthropic_primary(self):
        """A proxy whose primary is Anthropic yields a caching middleware."""
        proxy = SimpleNamespace(primary=_chat())
        assert isinstance(
            build_prompt_caching_middleware(proxy), AnthropicPromptCachingMiddleware
        )

    def test_returns_none_for_non_anthropic(self):
        """A non-Anthropic model yields no caching middleware."""
        assert build_prompt_caching_middleware(object()) is None

    def test_configured_to_ignore_unsupported_models(self):
        """The middleware never warns/raises on a non-Anthropic effective model."""
        mw = build_prompt_caching_middleware(_chat())
        assert mw.unsupported_model_behavior == "ignore"

    def test_returns_none_when_middleware_unavailable(self):
        """An older langchain_anthropic without the middleware degrades to None."""
        # Model detection still succeeds, but the middleware import fails.
        with patch.dict(sys.modules, {"langchain_anthropic.middleware": None}):
            assert build_prompt_caching_middleware(_chat()) is None


# ---------------------------------------------------------------------------
# The middleware marks the stable prefix (model_settings -> bound kwarg).
# ---------------------------------------------------------------------------


def _run_middleware(
    mw, model, messages, system_prompt="STABLE SYSTEM PREFIX", seen=None
):
    """Drive the middleware's ``wrap_model_call`` and return the model_settings
    the model call would receive (what the langchain factory binds).

    The request is langchain's real ``ModelRequest``, not a stand-in: the
    middleware reads fields of it (``system_message`` among them) that a
    hand-written stub silently lacks when the upstream schema grows. When
    *seen* is a list, the request the handler received is appended to it.
    """
    request = ModelRequest(model=model, messages=messages, system_prompt=system_prompt)
    captured: dict = {}

    def handler(req):
        captured.update(req.model_settings)
        if seen is not None:
            seen.append(req)
        return "response"

    mw.wrap_model_call(request, handler)
    return captured


def test_middleware_sets_ephemeral_cache_control():
    """The middleware records an ephemeral cache_control setting for Anthropic."""
    chat = _chat()
    mw = build_prompt_caching_middleware(chat)
    seen = []
    settings = _run_middleware(mw, chat, [HumanMessage("hello")], seen=seen)
    assert settings.get("cache_control", {}).get("type") == "ephemeral"
    # The stable system prefix itself carries the breakpoint, and the user
    # turn is passed through unchanged.
    (forwarded,) = seen
    system_blocks = forwarded.system_message.content
    assert system_blocks[-1]["text"] == "STABLE SYSTEM PREFIX"
    assert system_blocks[-1]["cache_control"]["type"] == "ephemeral"
    assert [m.content for m in forwarded.messages] == ["hello"]


# ---------------------------------------------------------------------------
# End-to-end: cache_control lands in the request, reuse reports cache reads.
# ---------------------------------------------------------------------------


class _FakeUsage:  # pylint: disable=too-few-public-methods
    """Stand-in for Anthropic's ``Usage`` model with the token fields
    ``_create_usage_metadata`` reads."""

    def __init__(self, input_tokens, output_tokens, cache_read, cache_creation):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation
        self.cache_creation = None


class _FakeRawMessage:
    """Stand-in for what ``ChatAnthropic._create`` returns.

    ``_create`` calls the client's ``with_raw_response`` API, so the chat model
    calls ``parse()`` on the result to get the Message. One object plays both
    roles here: ``parse()`` returns itself, and ``model_dump()`` and ``usage``
    are the Message fields ``_format_output`` reads.
    """

    def __init__(self, usage):
        self.usage = usage

    def parse(self):
        """Return the parsed Message (this object)."""
        return self

    def model_dump(self):
        """Return a minimal Anthropic Message dict for ``_format_output``."""
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": _MODEL,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": "ok"}],
            "usage": {},
        }


def test_cache_control_marks_prefix_and_reuse_reads_cache():
    """Full chain: the middleware's setting, bound the way the langchain agent
    factory binds it, places a cache breakpoint on the prefix of every request,
    and the model surfaces cache-creation then cache-read usage across two
    calls sharing that prefix.
    """
    chat = _chat()
    mw = build_prompt_caching_middleware(chat)
    assert mw is not None

    # 1) The middleware records the cache_control model-setting.
    settings = _run_middleware(mw, chat, [HumanMessage("hello")])
    assert "cache_control" in settings

    # 2) Bind it the way langchain's agent factory does
    #    (model.bind_tools(..., **model_settings) / model.bind(**model_settings)).
    bound = chat.bind(**settings)

    # 3) Mock the Anthropic client so no network/key is needed.  First call
    #    writes the cache; second (same prefix) reads it.
    payloads = []

    def fake_create(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            usage = _FakeUsage(10, 5, cache_read=0, cache_creation=1470)
        else:
            usage = _FakeUsage(10, 5, cache_read=1470, cache_creation=0)
        return _FakeRawMessage(usage)

    chat._create = fake_create  # pylint: disable=protected-access

    system = SystemMessage("STABLE SYSTEM PREFIX")
    first = bound.invoke([system, HumanMessage("hello")])
    second = bound.invoke(
        [
            system,
            HumanMessage("hello"),
            AIMessage("intermediate reasoning"),
            HumanMessage("Observation: tool result"),
        ]
    )

    # Both calls carry the cache breakpoint. On the direct Anthropic API the
    # chat model sends it as the top-level ``cache_control`` request field and
    # the API places the breakpoint on the prefix; it is not copied onto a
    # message block as well.
    for payload in payloads:
        assert payload["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
        for message in payload["messages"]:
            content = message["content"]
            blocks = content if isinstance(content, list) else []
            assert not any("cache_control" in block for block in blocks)
    # The system prefix is present ahead of the breakpoint (it is cached).
    assert payloads[0]["system"]

    # First call creates the cache; second call reads it.
    assert first.usage_metadata["input_token_details"]["cache_creation"] == 1470
    assert first.usage_metadata["input_token_details"]["cache_read"] == 0
    assert second.usage_metadata["input_token_details"]["cache_read"] == 1470


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
