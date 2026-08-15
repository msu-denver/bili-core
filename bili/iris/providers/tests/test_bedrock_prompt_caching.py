"""Tests for Bedrock prompt-caching helpers and middleware.

Covers both halves of the feature:

- The selection helpers (:func:`_is_cacheable_bedrock_model`,
  :func:`build_prompt_caching_middleware`) decide *whether* Bedrock caching
  applies (Claude/Nova only) and degrade to ``None`` otherwise.
- The end-to-end evidence: the middleware places a Bedrock cache point after the
  system content, so the actual Converse ``system`` payload carries it, and a
  reused prefix reports cache-read usage.  The Bedrock client is mocked, so no
  AWS credentials or network are required.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock
from langchain_core.messages import HumanMessage, SystemMessage

from bili.iris.providers.bedrock_cache import (
    BedrockSystemCachePointMiddleware,
    _system_content_with_cache_point,
)
from bili.iris.providers.bedrock_provider import (
    _is_cacheable_bedrock_model,
    build_prompt_caching_middleware,
)

_CACHE_POINT = {"cachePoint": {"type": "default"}}
_CLAUDE_ID = "us.anthropic.claude-sonnet-4-6"
_NOVA_ID = "us.amazon.nova-pro-v1:0"
_LLAMA_ID = "us.meta.llama3-3-70b-instruct-v1:0"


def _bedrock(model_id):
    """Return an offline ``ChatBedrockConverse`` (region set, no AWS call)."""
    return ChatBedrockConverse(model_id=model_id, region_name="us-east-1")


def _model_request(messages, system_prompt=None):
    """Build a real ``ModelRequest`` carrying *messages* / *system_prompt*."""
    return ModelRequest(
        model=_bedrock(_CLAUDE_ID),
        system_prompt=system_prompt,
        messages=list(messages),
        tool_choice=None,
        tools=[],
        response_format=None,
        state={},
        runtime=None,
        model_settings={},
    )


def _run_middleware(request):
    """Drive the middleware and return the messages the handler received."""
    seen = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        seen["system_prompt"] = req.system_prompt
        return "response"

    BedrockSystemCachePointMiddleware().wrap_model_call(request, handler)
    return seen


# ---------------------------------------------------------------------------
# _is_cacheable_bedrock_model
# ---------------------------------------------------------------------------


class TestIsCacheableBedrockModel:
    """Only Claude/Nova on Bedrock are cache-eligible."""

    def test_true_for_claude(self):
        """A Claude Bedrock model is cache-eligible."""
        assert _is_cacheable_bedrock_model(_bedrock(_CLAUDE_ID)) is True

    def test_true_for_nova(self):
        """A Nova Bedrock model is cache-eligible."""
        assert _is_cacheable_bedrock_model(_bedrock(_NOVA_ID)) is True

    def test_false_for_llama(self):
        """A Bedrock family without caching support is not eligible."""
        assert _is_cacheable_bedrock_model(_bedrock(_LLAMA_ID)) is False

    def test_false_for_non_bedrock(self):
        """A non-Bedrock object is not eligible."""
        assert _is_cacheable_bedrock_model(object()) is False

    def test_unwraps_fallback_primary(self):
        """A proxy whose primary is a Claude Bedrock model is eligible."""
        proxy = SimpleNamespace(primary=_bedrock(_CLAUDE_ID))
        assert _is_cacheable_bedrock_model(proxy) is True

    def test_fallback_primary_llama_not_eligible(self):
        """A proxy whose primary is a non-caching Bedrock model is not eligible."""
        proxy = SimpleNamespace(primary=_bedrock(_LLAMA_ID))
        assert _is_cacheable_bedrock_model(proxy) is False

    def test_false_when_langchain_aws_missing(self):
        """When langchain_aws cannot be imported, the answer is False."""
        with patch.dict(sys.modules, {"langchain_aws": None}):
            assert _is_cacheable_bedrock_model(object()) is False


# ---------------------------------------------------------------------------
# build_prompt_caching_middleware
# ---------------------------------------------------------------------------


class TestBuildPromptCachingMiddleware:
    """The factory returns a middleware only for a caching-eligible model."""

    def test_returns_middleware_for_claude(self):
        """A Claude Bedrock model yields the caching middleware."""
        mw = build_prompt_caching_middleware(_bedrock(_CLAUDE_ID))
        assert isinstance(mw, BedrockSystemCachePointMiddleware)

    def test_returns_middleware_for_nova(self):
        """A Nova Bedrock model yields the caching middleware."""
        assert isinstance(
            build_prompt_caching_middleware(_bedrock(_NOVA_ID)),
            BedrockSystemCachePointMiddleware,
        )

    def test_returns_none_for_llama(self):
        """A non-caching Bedrock family yields no middleware."""
        assert build_prompt_caching_middleware(_bedrock(_LLAMA_ID)) is None

    def test_returns_none_for_non_bedrock(self):
        """A non-Bedrock model yields no middleware."""
        assert build_prompt_caching_middleware(object()) is None

    def test_returns_none_when_middleware_module_unavailable(self):
        """A cacheable model degrades to None when the middleware cannot import."""
        with patch.dict(sys.modules, {"bili.iris.providers.bedrock_cache": None}):
            assert build_prompt_caching_middleware(_bedrock(_CLAUDE_ID)) is None


# ---------------------------------------------------------------------------
# The middleware places a cache point after the system content.
# ---------------------------------------------------------------------------


class TestCachePointPlacement:
    """The middleware marks the stable system prefix, both request shapes."""

    def test_leading_system_message_gets_cache_point(self):
        """A leading SystemMessage (the AETHER shape) is cache-pointed."""
        request = _model_request([SystemMessage("METHODOLOGY"), HumanMessage("hi")])
        seen = _run_middleware(request)
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        assert bedrock_system == [{"text": "METHODOLOGY"}, _CACHE_POINT]

    def test_system_prompt_string_gets_cache_point(self):
        """A create_agent system_prompt string is turned into a cache-pointed
        system message and removed from system_prompt."""
        request = _model_request([HumanMessage("hi")], system_prompt="METHODOLOGY")
        seen = _run_middleware(request)
        assert seen["system_prompt"] is None
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        assert bedrock_system == [{"text": "METHODOLOGY"}, _CACHE_POINT]

    def test_idempotent_when_already_cache_pointed(self):
        """A system message that already carries a cache point is unchanged."""
        already = SystemMessage(content=[{"type": "text", "text": "M"}, _CACHE_POINT])
        request = _model_request([already, HumanMessage("hi")])
        seen = _run_middleware(request)
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        # Exactly one cache point, not two.
        assert bedrock_system.count(_CACHE_POINT) == 1

    def test_no_system_message_is_passed_through(self):
        """With no system prefix there is nothing to cache; request unchanged."""
        request = _model_request([HumanMessage("hi")])
        seen = _run_middleware(request)
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        assert not bedrock_system

    def test_empty_system_prompt_is_noop(self):
        """An empty system_prompt string caches nothing and stays empty."""
        request = _model_request([HumanMessage("hi")], system_prompt="")
        seen = _run_middleware(request)
        # No system message is synthesised; the empty prompt is preserved as-is.
        assert seen["system_prompt"] == ""
        assert not any(isinstance(m, SystemMessage) for m in seen["messages"])

    def test_list_system_content_gets_cache_point(self):
        """A SystemMessage whose content is already a block list is appended to."""
        request = _model_request(
            [SystemMessage(content=[{"type": "text", "text": "M"}]), HumanMessage("hi")]
        )
        seen = _run_middleware(request)
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        assert bedrock_system == [{"text": "M"}, _CACHE_POINT]

    def test_awrap_model_call_places_cache_point(self):
        """The async path places the cache point the same way as the sync path."""
        request = _model_request([SystemMessage("METHODOLOGY"), HumanMessage("hi")])
        seen = {}

        async def handler(req):
            seen["messages"] = list(req.messages)
            return "response"

        asyncio.run(
            BedrockSystemCachePointMiddleware().awrap_model_call(request, handler)
        )
        _, bedrock_system = _messages_to_bedrock(seen["messages"])
        assert bedrock_system == [{"text": "METHODOLOGY"}, _CACHE_POINT]


class TestSystemContentHelper:
    """Branch coverage for the content transform."""

    def test_non_empty_string(self):
        """A non-empty string becomes a text block plus a cache point."""
        assert _system_content_with_cache_point("M") == [
            {"type": "text", "text": "M"},
            _CACHE_POINT,
        ]

    def test_empty_string_returns_none(self):
        """An empty string caches nothing."""
        assert _system_content_with_cache_point("") is None

    def test_list_without_cache_point_gets_one(self):
        """A block list without a cache point gets one appended."""
        assert _system_content_with_cache_point([{"type": "text", "text": "M"}]) == [
            {"type": "text", "text": "M"},
            _CACHE_POINT,
        ]

    def test_list_with_cache_point_returns_none(self):
        """A block list that already carries a cache point is unchanged."""
        assert (
            _system_content_with_cache_point(
                [{"type": "text", "text": "M"}, _CACHE_POINT]
            )
            is None
        )

    def test_other_type_returns_none(self):
        """Content that is neither a string nor a list caches nothing."""
        assert _system_content_with_cache_point(None) is None


# ---------------------------------------------------------------------------
# End-to-end: the cache point reaches the Converse API and reuse reads cache.
# ---------------------------------------------------------------------------


def _converse_response(cache_read, cache_write):
    """A minimal Bedrock Converse response with the given cache token counts."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
            "cacheReadInputTokens": cache_read,
            "cacheWriteInputTokens": cache_write,
        },
        "stopReason": "end_turn",
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "metrics": {"latencyMs": 1},
    }


def test_cache_point_reaches_api_and_reuse_reports_cache_read():
    """Full path: the middleware's cache point reaches the Converse ``system``
    payload, and a reused prefix reports cache-creation then cache-read usage.
    The Bedrock client is mocked; no AWS credentials or network are used.
    """
    chat = _bedrock(_CLAUDE_ID)
    calls = []

    def converse(**kwargs):
        calls.append(kwargs)
        return (
            _converse_response(0, 1470)
            if len(calls) == 1
            else _converse_response(1470, 0)
        )

    mock_client = MagicMock()
    mock_client.converse.side_effect = converse
    chat.client = mock_client

    mw = build_prompt_caching_middleware(chat)
    assert mw is not None

    request = ModelRequest(
        model=chat,
        system_prompt=None,
        messages=[SystemMessage("STABLE PREFIX"), HumanMessage("hi")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={},
        runtime=None,
        model_settings={},
    )

    transformed = {}

    def handler(req):
        transformed["messages"] = list(req.messages)
        return chat.invoke(req.messages)

    first = mw.wrap_model_call(request, handler)
    # A reuse call sends the same cache-pointed prefix.
    second = chat.invoke(transformed["messages"])

    # The cache point is in the actual Converse system payload.
    assert calls[0]["system"] == [{"text": "STABLE PREFIX"}, _CACHE_POINT]
    # First call writes the cache; second call reads it.
    assert first.usage_metadata["input_token_details"]["cache_creation"] == 1470
    assert first.usage_metadata["input_token_details"]["cache_read"] == 0
    assert second.usage_metadata["input_token_details"]["cache_read"] == 1470


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
