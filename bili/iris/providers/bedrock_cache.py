"""Bedrock prompt-caching middleware for the native tool-calling agent path.

AWS Bedrock's Converse API caches a conversation prefix when a
``{"cachePoint": {"type": "default"}}`` block is placed in the request, and
subsequent calls that share that prefix read it from cache (billed well below
the standard input rate).  This is the Bedrock analogue of
``langchain_anthropic``'s ``AnthropicPromptCachingMiddleware``: Bedrock loads
``ChatBedrockConverse`` (not ``ChatAnthropic``), so the Anthropic middleware's
``isinstance`` guard correctly skips it, and Bedrock needs its own wiring.

This middleware places a cache point on the STABLE SYSTEM PREFIX (the large
methodology/task-context block re-sent on every model call in a multi-call agent
run).  Caching the system prefix captures the dominant repeated input cost with
one cache checkpoint (well within Bedrock's four-checkpoint limit) and mutates
only the stable system message, avoiding the edge cases of rewriting mid-loop
tool turns.

The class is imported lazily by
:func:`bili.iris.providers.bedrock_provider.build_prompt_caching_middleware` so
the base provider module stays free of the ``langchain.agents`` import cost when
caching is not in use.

A newer ``langchain-aws`` ships a first-party ``BedrockPromptCachingMiddleware``
(same ``anthropic``/``amazon.nova`` model gate, plus incremental
latest-message caching); this module targets the pinned ``langchain-aws`` that
predates it without requiring a dependency bump.
"""

from typing import Any, Awaitable, Callable, List, Optional

from langchain.agents.middleware.types import (  # pylint: disable=import-error
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage


def _cache_point() -> dict:
    """Return Bedrock's cache-point content block.

    Sourced from ``ChatBedrockConverse.create_cache_point()`` so the shape stays
    in sync with the installed ``langchain-aws`` rather than being duplicated.
    """
    from langchain_aws import (  # pylint: disable=import-outside-toplevel
        ChatBedrockConverse,
    )

    return ChatBedrockConverse.create_cache_point()


def _system_content_with_cache_point(content: Any) -> Optional[List[Any]]:
    """Return *content* as a Bedrock content-block list ending in a cache point.

    Returns ``None`` (meaning "no change") when the content is empty or already
    carries a cache point, which keeps the transform idempotent.

    :param content: A system message's content (a plain string or a block list).
    :returns: The new content list, or ``None`` to leave the request unchanged.
    """
    cache_point = _cache_point()
    if isinstance(content, str):
        if not content:
            return None
        return [{"type": "text", "text": content}, cache_point]
    if isinstance(content, list):
        if any(isinstance(block, dict) and "cachePoint" in block for block in content):
            return None  # already cache-pointed; do not stack a second one
        return [*content, cache_point]
    return None


def _request_with_system_cache_point(request: ModelRequest) -> Optional[ModelRequest]:
    """Return a copy of *request* with a cache point on the system prefix.

    Handles both shapes ``create_agent`` produces: the system prompt supplied as
    a ``system_prompt`` string, and a leading ``SystemMessage`` carried in the
    request messages (which is how the AETHER agent node passes it).  Returns
    ``None`` when there is no system prefix to cache or it is already
    cache-pointed, so the caller leaves the request untouched.
    """
    # Shape 1: create_agent(system_prompt=...) -> request.system_prompt is a str.
    # ``is not None`` (rather than truthiness) so an empty system_prompt is a
    # no-op via the helper's own emptiness check rather than a silent skip.
    if request.system_prompt is not None:
        new_content = _system_content_with_cache_point(request.system_prompt)
        if new_content is None:
            return None
        system_message = SystemMessage(content=new_content)
        return request.override(
            system_prompt=None,
            messages=[system_message, *request.messages],
        )

    # Shape 2: a leading SystemMessage in the messages (the AETHER agent node).
    messages = request.messages
    if messages and isinstance(messages[0], SystemMessage):
        new_content = _system_content_with_cache_point(messages[0].content)
        if new_content is None:
            return None
        cached_system = messages[0].model_copy(update={"content": new_content})
        return request.override(messages=[cached_system, *messages[1:]])

    return None


class BedrockSystemCachePointMiddleware(AgentMiddleware):
    """Place a Bedrock cache point on the stable system prefix.

    For a ``ChatBedrockConverse`` model, this marks the system prompt with a
    cache point so subsequent model calls in the same agent run read it from
    Bedrock's cache instead of re-billing it at the full input rate.  When there
    is no system prefix (or it is already cache-pointed) the request is passed
    through unchanged, so applying the middleware is never destructive.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add the system cache point, then delegate to the handler."""
        new_request = _request_with_system_cache_point(request)
        return handler(new_request if new_request is not None else request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of :meth:`wrap_model_call`."""
        new_request = _request_with_system_cache_point(request)
        return await handler(new_request if new_request is not None else request)
