"""Cohere provider for bili-core IRIS.

Wraps ``langchain_cohere.ChatCohere`` behind the :class:`LLMProvider`
interface.  Supports Cohere Command model families (Command A+, Command A,
Command R+, Command R) accessed via the Cohere API.

Authentication reads ``COHERE_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_cohere`` is imported inside :meth:`CohereProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class CohereProvider(LLMProvider):
    """LangChain-native Cohere provider.

    Accepted kwargs
    ---------------
    model_name : str
        Cohere model identifier (e.g. ``"command-a-plus-05-2026"``,
        ``"command-r-plus"``).
    max_tokens : int, optional
        Maximum completion tokens.
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
        """Create and return a ``ChatCohere`` instance.

        :param model_name: Cohere model identifier.
        :returns: A ``ChatCohere`` instance.
        :raises ImportError: If ``langchain_cohere`` is not installed.
        """
        from langchain_cohere import (  # pylint: disable=import-outside-toplevel,import-error
            ChatCohere,
        )

        LOGGER.info("Initializing Cohere model: %s", model_name)

        config: dict = {"model": model_name}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["p"] = top_p
        if top_k is not None:
            config["k"] = top_k
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatCohere(**config)
        LOGGER.debug(llm)
        return llm
