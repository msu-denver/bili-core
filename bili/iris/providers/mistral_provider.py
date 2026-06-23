"""Mistral AI provider for bili-core IRIS.

Wraps ``langchain_mistralai.ChatMistralAI`` behind the :class:`LLMProvider`
interface.  Supports Mistral model families (Mistral Large, Mistral Small,
Codestral, and related models) accessed via the Mistral AI API.

Authentication reads ``MISTRAL_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_mistralai`` is imported inside :meth:`MistralProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class MistralProvider(LLMProvider):
    """LangChain-native Mistral AI provider.

    Accepted kwargs
    ---------------
    model_name : str
        Mistral model identifier (e.g. ``"mistral-large-latest"``,
        ``"mistral-small-latest"``).
    max_tokens : int, optional
        Maximum completion tokens.
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    max_retries : int, optional
        Maximum number of automatic retries on transient errors.
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_retries: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatMistralAI`` instance.

        :param model_name: Mistral model identifier.
        :returns: A ``ChatMistralAI`` instance.
        :raises ImportError: If ``langchain_mistralai`` is not installed.
        """
        from langchain_mistralai import (  # pylint: disable=import-outside-toplevel,import-error
            ChatMistralAI,
        )

        LOGGER.info("Initializing Mistral model: %s", model_name)

        config: dict = {"model": model_name}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatMistralAI(**config)
        LOGGER.debug(llm)
        return llm
