"""Anthropic direct API provider for bili-core IRIS.

Wraps ``langchain_anthropic.ChatAnthropic`` behind the :class:`LLMProvider`
interface.  Supports Claude model families (Opus, Sonnet, Haiku) accessed via
the Anthropic API endpoint directly, as a complement to the existing
``remote_aws_bedrock`` provider which routes Claude through AWS Bedrock.

Authentication reads ``ANTHROPIC_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_anthropic`` is imported inside :meth:`AnthropicProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


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
