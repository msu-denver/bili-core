"""xAI (Grok) provider for bili-core IRIS.

Wraps ``langchain_xai.ChatXAI`` behind the :class:`LLMProvider` interface.
Supports Grok model families accessed via the xAI API.

Authentication reads ``XAI_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_xai`` is imported inside :meth:`XAIProvider.load` to avoid
module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class XAIProvider(LLMProvider):
    """LangChain-native xAI (Grok) provider.

    Accepted kwargs
    ---------------
    model_name : str
        xAI model identifier (e.g. ``"grok-3-latest"``,
        ``"grok-beta"``).
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
        """Create and return a ``ChatXAI`` instance.

        :param model_name: xAI model identifier.
        :returns: A ``ChatXAI`` instance.
        :raises ImportError: If ``langchain_xai`` is not installed.
        """
        from langchain_xai import (  # pylint: disable=import-outside-toplevel,import-error
            ChatXAI,
        )

        LOGGER.info("Initializing xAI model: %s", model_name)

        config: dict = {"model": model_name}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatXAI(**config)
        LOGGER.debug(llm)
        return llm
