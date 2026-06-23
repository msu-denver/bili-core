"""Groq provider for bili-core IRIS.

Wraps ``langchain_groq.ChatGroq`` behind the :class:`LLMProvider` interface.
Supports models served on Groq's low-latency inference hardware, including the
Llama, Gemma, and compound-beta families.

Authentication reads ``GROQ_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_groq`` is imported inside :meth:`GroqProvider.load` to avoid
module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class GroqProvider(LLMProvider):
    """LangChain-native Groq provider.

    Accepted kwargs
    ---------------
    model_name : str
        Groq-hosted model identifier (e.g. ``"llama-3.3-70b-versatile"``,
        ``"compound-beta"``).
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
        """Create and return a ``ChatGroq`` instance.

        :param model_name: Groq-hosted model identifier.
        :returns: A ``ChatGroq`` instance.
        :raises ImportError: If ``langchain_groq`` is not installed.
        """
        from langchain_groq import (  # pylint: disable=import-outside-toplevel,import-error
            ChatGroq,
        )

        LOGGER.info("Initializing Groq model: %s", model_name)

        config: dict = {"model_name": model_name}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatGroq(**config)
        LOGGER.debug(llm)
        return llm
