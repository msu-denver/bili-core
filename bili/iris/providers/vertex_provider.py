"""Google Vertex AI provider for bili-core IRIS.

Wraps ``langchain_google_vertexai.ChatVertexAI`` behind the
:class:`LLMProvider` interface.  Supports all Gemini model generations
catalogued in ``bili.iris.config.llm_config.LLM_MODELS``.

Authentication uses Google Application Default Credentials (ADC):
``gcloud auth application-default login`` or a service-account key pointed
to by ``GOOGLE_APPLICATION_CREDENTIALS``.

Heavy dependencies
------------------
``langchain_google_vertexai`` is imported inside :meth:`VertexAIProvider.load`
to avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class VertexAIProvider(LLMProvider):
    """LangChain-native Google Vertex AI provider.

    Accepted kwargs
    ---------------
    model_name : str
        Vertex AI model name (e.g. ``"gemini-2.5-pro"``).
    max_tokens : int, optional
        Maximum output tokens (mapped to ``max_output_tokens``).
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    top_k : int, optional
        Top-k sampling limit.
    seed : int, optional
        Random seed for reproducibility.
    response_mime_type : str, optional
        MIME type for structured output (e.g. ``"application/json"``).
    response_schema : dict, optional
        JSON schema for structured output.
    additional_headers : dict, optional
        Extra HTTP headers (Priority PayGo, Provisioned Throughput, etc.).
    location : str, optional
        GCP region override (e.g. ``"us-central1"``, ``"global"``).
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        response_schema: Optional[dict] = None,
        response_mime_type: Optional[str] = None,
        additional_headers: Optional[dict] = None,
        location: Optional[str] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatVertexAI`` instance.

        :returns: A ``ChatVertexAI`` instance.
        :raises ImportError: If ``langchain_google_vertexai`` is not installed.
        """
        from langchain_google_vertexai import (  # pylint: disable=import-outside-toplevel
            ChatVertexAI,
        )

        LOGGER.info("Initializing Google Vertex AI model: %s", model_name)

        config: dict = {"model_name": model_name}
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if temperature:
            config["temperature"] = temperature
        if top_p:
            config["top_p"] = top_p
        if top_k:
            config["top_k"] = top_k
        if seed:
            config["seed"] = seed
        if response_mime_type:
            config["response_mime_type"] = response_mime_type
        if response_schema:
            config["response_schema"] = response_schema
        if additional_headers:
            config["additional_headers"] = additional_headers
        if location:
            config["location"] = location

        llm = ChatVertexAI(**config)
        LOGGER.debug(llm)
        return llm
