"""Local Ollama provider for bili-core IRIS.

Wraps ``langchain_ollama.ChatOllama`` behind the :class:`LLMProvider`
interface so bili-core agents can run against a local Ollama server with no
API key and no network egress.  Unlike the in-process ``local_llamacpp`` and
``local_huggingface`` providers, this provider talks to a running Ollama
daemon over HTTP; the model must be pulled into Ollama beforehand
(``ollama pull <model>``).

Native tool calling
-------------------
``ChatOllama`` implements LangChain's ``bind_tools`` for tool-capable models
(e.g. Qwen3, Llama 3.1+, Mistral).  This provider is therefore wired as a
``native`` tool-calling provider in the catalog: an AETHER agent's tools bind
directly, the same path the remote API providers take.  Whether a *particular*
pulled model actually honours a tool call is a property of that model and its
Ollama template, not of this provider; select a tool-capable model for
tool-using workflows.

Model names are user-chosen because Ollama models are pulled locally, so the
catalog entry carries a placeholder ``model_id``.  Set the concrete model via
the ``model_name`` kwarg (from ``AgentSpec.model_name``) or a catalog entry's
``model_id``.  Point at a non-default daemon with the ``base_url`` kwarg (or a
catalog entry's ``kwargs.base_url``); it defaults to ``http://localhost:11434``.

Heavy dependencies
------------------
``langchain_ollama`` is imported inside :meth:`OllamaProvider.load` so this
module imports without the optional dependency installed.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)

#: Ollama's default daemon endpoint.  Used when no ``base_url`` is supplied.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


# pylint: disable=too-few-public-methods
class OllamaProvider(LLMProvider):
    """Local Ollama server provider (native tool calling).

    Accepted kwargs
    ---------------
    model_name : str
        Name of a model pulled into the local Ollama server (e.g.
        ``"qwen3"``, ``"llama3.1"``, ``"mistral"``).
    base_url : str, optional
        Base URL of the Ollama daemon.  Defaults to
        ``"http://localhost:11434"``.
    max_tokens : int, optional
        Maximum tokens to generate.  Forwarded to ``ChatOllama`` as
        ``num_predict`` (Ollama's parameter name).
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
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatOllama`` instance.

        :param model_name: Name of a model pulled into the local Ollama server.
        :param base_url: Ollama daemon base URL; defaults to
            ``"http://localhost:11434"``.
        :returns: A ``ChatOllama`` instance.
        :raises ImportError: If ``langchain_ollama`` is not installed.
        """
        from langchain_ollama import (  # pylint: disable=import-outside-toplevel,import-error
            ChatOllama,
        )

        resolved_base_url = base_url or DEFAULT_OLLAMA_BASE_URL
        LOGGER.info(
            "Initializing Ollama model '%s' at %s", model_name, resolved_base_url
        )

        config: dict = {"model": model_name, "base_url": resolved_base_url}
        # Ollama names the generation-length parameter num_predict, not
        # max_tokens; map bili-core's cross-provider max_tokens onto it.
        if max_tokens is not None:
            config["num_predict"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if top_k is not None:
            config["top_k"] = top_k
        if seed is not None:
            config["seed"] = seed

        llm = ChatOllama(**config)
        LOGGER.debug(llm)
        return llm
