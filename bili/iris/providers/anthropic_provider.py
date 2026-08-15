"""Anthropic direct API provider for bili-core IRIS.

Wraps ``langchain_anthropic.ChatAnthropic`` behind the :class:`LLMProvider`
interface.  Supports Claude model families (Opus, Sonnet, Haiku) accessed via
the Anthropic API endpoint directly, as a complement to the existing
``remote_aws_bedrock`` provider which routes Claude through AWS Bedrock.

Authentication reads ``ANTHROPIC_API_KEY`` from the environment.

Prompt caching
--------------
:func:`build_prompt_caching_middleware` returns a LangChain agent middleware
that enables Anthropic prompt caching on the native tool-calling agent path.
An agent run makes one model call per Thought/Action/Observation turn and
re-sends the whole stable conversation prefix (system prompt plus the
accumulated tool turns) on every call.  The middleware marks that prefix with
an ``ephemeral`` cache breakpoint so subsequent calls in the same run read it
from the provider cache (billed at roughly a tenth of the standard input rate)
instead of re-billing it in full.  It is a no-op for non-Anthropic models and
degrades cleanly when the installed ``langchain_anthropic`` predates the
middleware.  See :mod:`bili.iris.nodes.react_agent_node` for the wiring.

Heavy dependencies
------------------
``langchain_anthropic`` is imported inside :meth:`AnthropicProvider.load` and
inside the caching helpers to avoid module-level import cost when this provider
is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


def _is_anthropic_model(llm_model: Any) -> bool:
    """Return ``True`` when *llm_model* is (or transparently wraps) a
    ``ChatAnthropic`` instance.

    A :class:`~bili.iris.providers.fallback.FallbackLLM` is unwrapped via its
    ``primary`` attribute so a chain whose first provider is Anthropic is still
    recognised.  Returns ``False`` (never raises) when ``langchain_anthropic``
    is not installed.

    :param llm_model: Any LangChain-compatible chat model or transparent proxy.
    :returns: ``True`` if the model is an Anthropic direct-API model.
    :rtype: bool
    """
    try:
        from langchain_anthropic import (  # pylint: disable=import-outside-toplevel
            ChatAnthropic,
        )
    except ImportError:
        return False

    if isinstance(llm_model, ChatAnthropic):
        return True
    # Unwrap a FallbackLLM (or any transparent proxy) whose primary is Anthropic.
    primary = getattr(llm_model, "primary", None)
    return isinstance(primary, ChatAnthropic) if primary is not None else False


def build_prompt_caching_middleware(llm_model: Any) -> Optional[Any]:
    """Build an Anthropic prompt-caching middleware for *llm_model*, or ``None``.

    Returns an ``AnthropicPromptCachingMiddleware`` configured to place an
    ``ephemeral`` cache breakpoint on the stable conversation prefix, so that a
    multi-call agent run re-reads that prefix from cache instead of re-billing
    it at the full input rate on every model call.

    The result is ``None`` (a graceful no-op) when either:

    - *llm_model* is not an Anthropic direct-API model (caching here is
      Anthropic specific; OpenAI caches automatically and other providers have
      their own mechanisms), or
    - the installed ``langchain_anthropic`` predates the caching middleware.

    The middleware is constructed with ``unsupported_model_behavior="ignore"``
    as a second line of defence: even if the effective model at call time turns
    out not to be a ``ChatAnthropic`` (for example a wrapped or swapped model),
    caching is skipped silently rather than warning or raising.  Anthropic
    enforces its own minimum cacheable prompt size server-side, so a prefix too
    small to cache is a silent no-op with no error.

    :param llm_model: The chat model the agent will run on.
    :returns: An ``AgentMiddleware`` instance, or ``None`` to skip caching.
    :rtype: Optional[Any]
    """
    if not _is_anthropic_model(llm_model):
        return None
    try:
        from langchain_anthropic.middleware import (  # noqa: E501  pylint: disable=import-outside-toplevel
            AnthropicPromptCachingMiddleware,
        )
    except ImportError:
        # Older langchain-anthropic without the caching middleware: no-op.
        return None
    return AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")


# pylint: disable=too-few-public-methods
class AnthropicProvider(LLMProvider):
    """LangChain-native Anthropic API provider.

    Accepted kwargs
    ---------------
    model_name : str
        Anthropic model identifier (e.g. ``"claude-opus-4-8"``,
        ``"claude-sonnet-4-6"``).
    max_tokens : int, optional
        Maximum completion tokens.  Defaults to 1024 when not set
        (``ChatAnthropic`` requires an explicit value; 1024 is a safe
        minimum compatible with all Claude models).
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    top_k : int, optional
        Top-k sampling limit.
    max_retries : int, optional
        Maximum number of automatic retries on transient errors.
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatAnthropic`` instance.

        :param model_name: Anthropic model identifier.
        :returns: A ``ChatAnthropic`` instance.
        :raises ImportError: If ``langchain_anthropic`` is not installed.
        """
        from langchain_anthropic import (  # pylint: disable=import-outside-toplevel
            ChatAnthropic,
        )

        LOGGER.info("Initializing Anthropic model: %s", model_name)

        # ChatAnthropic requires max_tokens; use 1024 as a safe default.
        config: dict = {
            "model": model_name,
            "max_tokens": max_tokens if max_tokens is not None else 1024,
        }
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if top_k is not None:
            config["top_k"] = top_k
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatAnthropic(**config)
        LOGGER.debug(llm)
        return llm
