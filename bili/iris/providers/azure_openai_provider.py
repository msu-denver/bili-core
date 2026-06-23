"""Azure OpenAI provider for bili-core IRIS.

Wraps ``langchain_openai.AzureChatOpenAI`` behind the :class:`LLMProvider`
interface.  Supports GPT-4.x, GPT-3.5, and reasoning (o1/o3/o4) model
families available through Azure OpenAI Service.

Authentication reads ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT``
from the environment (LangChain OpenAI defaults).

Heavy dependencies
------------------
``langchain_openai`` is imported inside :meth:`AzureOpenAIProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class AzureOpenAIProvider(LLMProvider):
    """LangChain-native Azure OpenAI Service provider.

    Accepted kwargs
    ---------------
    model_name : str
        Azure deployment name (e.g. ``"gpt-41"``).
    api_version : str
        Azure OpenAI REST API version string (e.g. ``"2025-01-01-preview"``).
        Required. No default is provided here to match the contract of
        ``load_remote_azure_openai``; the caller must supply the version
        explicitly so it cannot silently drift.
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
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        api_version: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return an ``AzureChatOpenAI`` instance.

        :param model_name: Azure deployment name.
        :param api_version: REST API version string. Required; no default is
            provided so the caller cannot silently use a stale version.
        :returns: An ``AzureChatOpenAI`` instance.
        :raises ImportError: If ``langchain_openai`` is not installed.
        """
        from langchain_openai import (  # pylint: disable=import-outside-toplevel
            AzureChatOpenAI,
        )

        LOGGER.info(
            "Initializing Azure OpenAI model: %s, API version: %s",
            model_name,
            api_version,
        )

        config: dict = {
            "azure_deployment": model_name,
            "api_version": api_version,
        }
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

        llm = AzureChatOpenAI(**config)
        LOGGER.debug(llm)
        return llm
