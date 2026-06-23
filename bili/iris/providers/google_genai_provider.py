"""Google Generative AI (Gemini direct API) provider for bili-core IRIS.

Wraps ``langchain_google_genai.ChatGoogleGenerativeAI`` behind the
:class:`LLMProvider` interface.  Provides access to Gemini models via the
Google AI Developer API, as a complement to the existing
``remote_google_vertex`` provider which routes through Google Cloud Vertex AI.

Authentication reads ``GOOGLE_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_google_genai`` is imported inside :meth:`GoogleGenAIProvider.load`
to avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class GoogleGenAIProvider(LLMProvider):
    """LangChain-native Google Generative AI (Gemini) provider.

    Accepted kwargs
    ---------------
    model_name : str
        Gemini model identifier (e.g. ``"gemini-2.5-flash"``,
        ``"gemini-2.0-flash"``).
    max_tokens : int, optional
        Maximum completion tokens (mapped to ``max_output_tokens``).
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
        """Create and return a ``ChatGoogleGenerativeAI`` instance.

        :param model_name: Gemini model identifier.
        :returns: A ``ChatGoogleGenerativeAI`` instance.
        :raises ImportError: If ``langchain_google_genai`` is not installed.
        """
        from langchain_google_genai import (  # pylint: disable=import-outside-toplevel,import-error
            ChatGoogleGenerativeAI,
        )

        LOGGER.info("Initializing Google GenAI model: %s", model_name)

        config: dict = {"model": model_name}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if top_k is not None:
            config["top_k"] = top_k
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatGoogleGenerativeAI(**config)
        LOGGER.debug(llm)
        return llm
