"""Register all built-in providers in the global ``PROVIDER_REGISTRY``.

Importing this module has the side effect of populating
:data:`~bili.iris.providers.registry.PROVIDER_REGISTRY` with the provider
types shipped with bili-core.  It is imported once inside
:func:`bili.iris.loaders.llm_loader.load_model` at first call.

External provider authors do NOT import this module; they call
:func:`~bili.iris.providers.registry.register_provider` directly from their
own code at application startup.

Built-in provider types
-----------------------
================================  ================================================
Provider type string               Implementation class
================================  ================================================
``remote_aws_bedrock``             :class:`~.bedrock_provider.BedrockProvider`
``remote_google_vertex``           :class:`~.vertex_provider.VertexAIProvider`
``remote_azure_openai``            :class:`~.azure_openai_provider.AzureOpenAIProvider`
``remote_openai``                  :class:`~.openai_provider.OpenAIProvider`
``remote_anthropic``               :class:`~.anthropic_provider.AnthropicProvider`
``remote_mistral``                 :class:`~.mistral_provider.MistralProvider`
``remote_cohere``                  :class:`~.cohere_provider.CohereProvider`
``remote_google_genai``            :class:`~.google_genai_provider.GoogleGenAIProvider`
``remote_deepseek``                :class:`~.deepseek_provider.DeepSeekProvider`
``remote_xai``                     :class:`~.xai_provider.XAIProvider`
``remote_groq``                    :class:`~.groq_provider.GroqProvider`
``local_llamacpp``                 :class:`~.llamacpp_provider.LlamaCppProvider`
``local_huggingface``              :class:`~.huggingface_provider.HuggingFaceProvider`
``cli``                            :class:`~.cli_provider.CliProvider`
================================  ================================================
"""

import logging

from .anthropic_provider import AnthropicProvider
from .azure_openai_provider import AzureOpenAIProvider
from .bedrock_provider import BedrockProvider
from .cli_provider import CliProvider
from .cohere_provider import CohereProvider
from .deepseek_provider import DeepSeekProvider
from .google_genai_provider import GoogleGenAIProvider
from .groq_provider import GroqProvider
from .huggingface_provider import HuggingFaceProvider
from .llamacpp_provider import LlamaCppProvider
from .mistral_provider import MistralProvider
from .openai_provider import OpenAIProvider
from .registry import PROVIDER_REGISTRY
from .vertex_provider import VertexAIProvider
from .xai_provider import XAIProvider

LOGGER = logging.getLogger(__name__)

_BUILTIN_PROVIDERS = {
    "remote_aws_bedrock": BedrockProvider,
    "remote_google_vertex": VertexAIProvider,
    "remote_azure_openai": AzureOpenAIProvider,
    "remote_openai": OpenAIProvider,
    "remote_anthropic": AnthropicProvider,
    "remote_mistral": MistralProvider,
    "remote_cohere": CohereProvider,
    "remote_google_genai": GoogleGenAIProvider,
    "remote_deepseek": DeepSeekProvider,
    "remote_xai": XAIProvider,
    "remote_groq": GroqProvider,
    "local_llamacpp": LlamaCppProvider,
    "local_huggingface": HuggingFaceProvider,
    "cli": CliProvider,
}


def _register_builtins() -> None:
    """Populate ``PROVIDER_REGISTRY`` with built-in providers (idempotent).

    Safe to call multiple times; already-registered types are skipped
    so repeated imports (e.g. in tests) do not raise ``ValueError``.
    """
    for provider_type, provider_class in _BUILTIN_PROVIDERS.items():
        if provider_type not in PROVIDER_REGISTRY:
            PROVIDER_REGISTRY.register(provider_type, provider_class)
            LOGGER.debug("Built-in provider registered: '%s'", provider_type)
        else:
            LOGGER.debug("Built-in provider already registered: '%s'", provider_type)


_register_builtins()
