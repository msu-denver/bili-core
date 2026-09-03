"""AWS Bedrock provider for bili-core IRIS.

Wraps ``langchain_aws.ChatBedrockConverse`` behind the :class:`LLMProvider`
interface.  Supports all models available through the AWS Bedrock Converse API
(Amazon Nova, Anthropic Claude, Meta Llama, Mistral, Cohere, DeepSeek, and
others catalogued in ``bili.iris.config.llm_config.LLM_MODELS``).

Authentication uses the standard AWS credential chain (environment variables,
``~/.aws/credentials``, IAM instance profiles, etc.).

Prompt caching
--------------
:func:`build_prompt_caching_middleware` returns a LangChain agent middleware
that enables Bedrock Converse prompt caching on the native tool-calling agent
path, for the model families Bedrock supports it on (Anthropic Claude and
Amazon Nova).  It is the Bedrock analogue of the Anthropic-direct wiring: a
multi-call agent run re-reads its stable system prefix from Bedrock's cache
instead of re-billing it at the full input rate.  It is a no-op for every other
provider and for Bedrock model families that do not support caching (a cache
point on an unsupported model is rejected by Bedrock, so the gate matches
``langchain-aws``'s own ``anthropic``/``amazon.nova`` model-id markers).  See
:mod:`bili.iris.providers.bedrock_cache` and
:mod:`bili.iris.nodes.react_agent_node` for the wiring.

Heavy dependencies
------------------
``langchain_aws`` and its underlying ``boto3`` SDK are imported inside
:meth:`BedrockProvider.load` and inside the caching helpers to avoid
module-level import cost when the AWS provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)

#: Substrings of a Bedrock model id that mark a family whose Converse API
#: supports prompt caching.  Bedrock rejects a cache point on any other model,
#: so caching is gated on these markers.  This mirrors ``langchain-aws``'s own
#: first-party gate rather than a bespoke allowlist that would drift.
_CACHEABLE_BEDROCK_MODEL_MARKERS = ("anthropic", "amazon.nova")


def _is_cacheable_bedrock_model(llm_model: Any) -> bool:
    """Return ``True`` when *llm_model* is (or wraps) a ``ChatBedrockConverse``
    whose model id marks a caching-supported family (Claude or Nova).

    A :class:`~bili.iris.providers.fallback.FallbackLLM` is unwrapped via its
    ``primary`` attribute.  Returns ``False`` (never raises) when
    ``langchain_aws`` is not installed.

    :param llm_model: Any LangChain-compatible chat model or transparent proxy.
    :returns: ``True`` if Bedrock caching applies to this model.
    :rtype: bool
    """
    try:
        from langchain_aws import (  # pylint: disable=import-outside-toplevel
            ChatBedrockConverse,
        )
    except ImportError:
        return False

    model = llm_model
    if not isinstance(model, ChatBedrockConverse):
        primary = getattr(llm_model, "primary", None)
        if not isinstance(primary, ChatBedrockConverse):
            return False
        model = primary

    model_id = str(getattr(model, "model_id", "") or "").lower()
    return any(marker in model_id for marker in _CACHEABLE_BEDROCK_MODEL_MARKERS)


def build_prompt_caching_middleware(llm_model: Any) -> Optional[Any]:
    """Build a Bedrock prompt-caching middleware for *llm_model*, or ``None``.

    Returns a middleware that places a cache point on the stable system prefix
    of a ``ChatBedrockConverse`` request, so a multi-call agent run re-reads it
    from Bedrock's cache instead of re-billing it at the full input rate.

    The result is ``None`` (a graceful no-op) when *llm_model* is not a
    caching-supported Bedrock model (see :func:`_is_cacheable_bedrock_model`) or
    when the LangChain agent-middleware base is unavailable.  A cache point on an
    unsupported Bedrock model is rejected by the API, so this gate is what keeps
    the wiring safe.

    :param llm_model: The chat model the agent will run on.
    :returns: An ``AgentMiddleware`` instance, or ``None`` to skip caching.
    :rtype: Optional[Any]
    """
    if not _is_cacheable_bedrock_model(llm_model):
        return None
    try:
        from .bedrock_cache import (  # pylint: disable=import-outside-toplevel
            BedrockSystemCachePointMiddleware,
        )
    except ImportError:
        return None
    return BedrockSystemCachePointMiddleware()


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
