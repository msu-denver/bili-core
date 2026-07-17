"""Local LlamaCpp provider for bili-core IRIS.

Wraps ``langchain_community.chat_models.ChatLlamaCpp`` behind the
:class:`LLMProvider` interface for loading GGUF-format models from disk.

Note: LlamaCpp does not currently support automatic tool calling.  See
``bili.iris.config.llm_config.LLM_MODELS`` for the ``supports_tools: False``
flag on this provider's entries.

Heavy dependencies
------------------
``langchain_community`` is imported inside :meth:`LlamaCppProvider.load` to
avoid loading the large community package when this provider is not in use.
"""

import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class LlamaCppProvider(LLMProvider):
    """Local LlamaCpp in-process model provider.

    Accepted kwargs
    ---------------
    model_name : str
        Absolute path to the GGUF model file.
    max_tokens : int, optional
        Maximum tokens to generate.
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    top_k : int, optional
        Top-k sampling limit.
    seed : int, optional
        Random seed for reproducibility.
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Load and return a ``ChatLlamaCpp`` instance.

        :param model_name: Path to the GGUF model file.
        :returns: A ``ChatLlamaCpp`` instance.
        :raises ImportError: If ``langchain_community`` is not installed.
        """
        from langchain_community.chat_models import (  # pylint: disable=import-outside-toplevel
            ChatLlamaCpp,
        )

        LOGGER.info("Loading LlamaCpp model from %s", model_name)

        params: dict = {
            "model_path": model_name,
            "n_ctx": 4096,
            "n_gpu_layers": 512,
            "n_batch": 30,
            "n_parts": 1,
            "repeat_penalty": 1.176,
            "f16_kv": True,
        }
        if seed:
            params["seed"] = seed
        if top_p:
            params["top_p"] = top_p
        if top_k:
            params["top_k"] = top_k
        if temperature:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = max_tokens

        llm = ChatLlamaCpp(**params)  # pylint: disable=E1102
        LOGGER.debug(llm)
        return llm
