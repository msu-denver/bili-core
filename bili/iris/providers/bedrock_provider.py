"""AWS Bedrock provider for bili-core IRIS.

Wraps ``langchain_aws.ChatBedrockConverse`` behind the :class:`LLMProvider`
interface.  Supports all models available through the AWS Bedrock Converse API
(Amazon Nova, Anthropic Claude, Meta Llama, Mistral, Cohere, DeepSeek, and
others catalogued in ``bili.iris.config.llm_config.LLM_MODELS``).

Authentication uses the standard AWS credential chain (environment variables,
``~/.aws/credentials``, IAM instance profiles, etc.).

Heavy dependencies
------------------
``langchain_aws`` and its underlying ``boto3`` SDK are imported inside
:meth:`BedrockProvider.load` to avoid module-level import cost when the
AWS provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class BedrockProvider(LLMProvider):
    """LangChain-native AWS Bedrock Converse API provider.

    Accepted kwargs
    ---------------
    model_name : str
        The Bedrock model ID (e.g. ``"us.anthropic.claude-opus-4-20250514-v1:0"``).
    max_tokens : int, optional
        Maximum tokens in the model response.
    temperature : float, optional
        Sampling temperature (0.0 = deterministic, 1.0 = creative).
    top_p : float, optional
        Nucleus sampling probability threshold.
    top_k : int, optional
        Top-k sampling limit.
    seed : int, optional
        Random seed for reproducibility.
    """

    def load(  # pylint: disable=arguments-differ
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatBedrockConverse`` instance.

        :param model_name: Bedrock model ID.
        :param max_tokens: Maximum response tokens.
        :param temperature: Sampling temperature.
        :param top_p: Nucleus sampling probability.
        :param top_k: Top-k token limit.
        :param seed: Random seed.
        :returns: A ``ChatBedrockConverse`` instance.
        :raises ImportError: If ``langchain_aws`` is not installed.
        """
        from langchain_aws import (  # pylint: disable=import-outside-toplevel
            ChatBedrockConverse,
        )

        LOGGER.info("Initializing AWS Bedrock model: %s", model_name)

        config: dict = {"model_id": model_name}
        if max_tokens:
            config["max_tokens"] = max_tokens
        if temperature:
            config["temperature"] = temperature
        if top_p:
            config["top_p"] = top_p
        if top_k:
            config["top_k"] = top_k
        if seed:
            config["seed"] = seed

        llm = ChatBedrockConverse(**config)
        LOGGER.debug(llm)
        return llm
