"""OpenAI provider for bili-core IRIS.

Wraps ``langchain_openai.ChatOpenAI`` behind the :class:`LLMProvider`
interface.  Supports GPT-4.x, GPT-3.5, and reasoning model families
available through the OpenAI API.

Authentication reads ``OPENAI_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_openai`` is imported inside :meth:`OpenAIProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class OpenAIProvider(LLMProvider):
    """LangChain-native OpenAI API provider.

    Accepted kwargs
    ---------------
    model_name : str
        OpenAI model identifier (e.g. ``"gpt-4o"``, ``"o3-mini"``).
    max_tokens : int, optional
        Maximum completion tokens (mapped to ``max_completion_tokens``).
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    top_k : int, optional
        Top-k sampling limit (forwarded; provider may ignore).
    seed : int, optional
        Random seed for reproducibility.
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
        seed: Optional[int] = None,
        max_retries: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatOpenAI`` instance.

        :param model_name: OpenAI model identifier.
        :returns: A ``ChatOpenAI`` instance.
        :raises ImportError: If ``langchain_openai`` is not installed.
        """
        from langchain_openai import (  # pylint: disable=import-outside-toplevel
            ChatOpenAI,
        )

        LOGGER.info("Initializing OpenAI model: %s", model_name)

        config: dict = {"model": model_name}
        if temperature:
            config["temperature"] = temperature
        if max_tokens:
            config["max_completion_tokens"] = max_tokens
        if top_p:
            config["top_p"] = top_p
        if top_k:
            config["top_k"] = top_k
        if seed:
            config["seed"] = seed
        if max_retries:
            config["max_retries"] = max_retries

        llm = ChatOpenAI(**config)
        LOGGER.debug(llm)
        return llm
