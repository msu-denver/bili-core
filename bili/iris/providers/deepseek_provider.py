"""DeepSeek provider for bili-core IRIS.

Wraps ``langchain_deepseek.ChatDeepSeek`` behind the :class:`LLMProvider`
interface.  Supports DeepSeek models (DeepSeek-V3/V4 chat and DeepSeek-R1
reasoner families) accessed via the DeepSeek API.

Authentication reads ``DEEPSEEK_API_KEY`` from the environment.

Heavy dependencies
------------------
``langchain_deepseek`` is imported inside :meth:`DeepSeekProvider.load` to
avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class DeepSeekProvider(LLMProvider):
    """LangChain-native DeepSeek provider.

    Accepted kwargs
    ---------------
    model_name : str
        DeepSeek model identifier (e.g. ``"deepseek-chat"``,
        ``"deepseek-reasoner"``).
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
        """Create and return a ``ChatDeepSeek`` instance.

        :param model_name: DeepSeek model identifier.
        :returns: A ``ChatDeepSeek`` instance.
        :raises ImportError: If ``langchain_deepseek`` is not installed.
        """
        from langchain_deepseek import (  # pylint: disable=import-outside-toplevel,import-error
            ChatDeepSeek,
        )

        LOGGER.info("Initializing DeepSeek model: %s", model_name)

        config: dict = {"model": model_name}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_retries is not None:
            config["max_retries"] = max_retries

        llm = ChatDeepSeek(**config)
        LOGGER.debug(llm)
        return llm
