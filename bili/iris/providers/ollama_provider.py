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

Schema-constrained structured output
------------------------------------
Pass ``structured_output_schema`` (a JSON schema ``dict`` or a Pydantic
``BaseModel`` subclass) to constrain generation to schema-valid JSON.  The
schema is forwarded as ``ChatOllama``'s ``format`` parameter; Ollama compiles
it to a llama.cpp GBNF grammar, so decoding is token-level constrained -- the
model cannot emit output that violates the schema.  ``.invoke()`` still
returns a message whose ``.content`` is a string (guaranteed schema-valid
JSON); parse it with
:func:`bili.iris.providers.structured_output.parse_structured_content`.
The ``format`` grammar applies to assistant *content* only; Ollama does not
constrain tool-call arguments, so combine the schema with a tool-less agent
and apply side effects deterministically from the validated object.

Resolving arbitrary user-pulled tags
-------------------------------------
An AETHER ``AgentSpec`` carries only ``model_name``, resolved to a provider via
catalog lookup and prefix heuristics (see
``bili.aether.compiler.llm_resolver``).  A tag that is not in the catalog and
matches no heuristic cannot resolve at all.  Since Ollama tags are arbitrary
and user-pulled (``qwen3:14b``, ``llama3.1:70b-instruct-q4_0``, ...), prefix
the ``model_name`` with the ``ollama:`` sentinel (e.g.
``"ollama:qwen3:14b"``) to route to this provider without a catalog entry,
mirroring the existing ``cli:`` sentinel for the subprocess CLI provider.
:meth:`load` strips the ``ollama:`` prefix before constructing ``ChatOllama``
so the daemon receives the real tag.  The prefix is optional: the single-turn
path that passes a bare tag with an explicit ``provider_type`` continues to
work unchanged.

Heavy dependencies
------------------
``langchain_ollama`` is imported inside :meth:`OllamaProvider.load` so this
module imports without the optional dependency installed.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider
from .structured_output import normalize_schema

LOGGER = logging.getLogger(__name__)

#: Ollama's default daemon endpoint.  Used when no ``base_url`` is supplied.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

#: Sentinel prefix that routes an arbitrary, non-catalog ``model_name`` to
#: this provider via the resolver's heuristic rules (mirrors the "cli:"
#: sentinel for the subprocess CLI provider).  Stripped in :meth:`load`.
OLLAMA_MODEL_PREFIX = "ollama:"


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
    structured_output_schema : dict or type, optional
        JSON schema (or Pydantic model class) to constrain generation to.
        Forwarded as ``ChatOllama``'s ``format`` parameter for
        grammar-constrained decoding.
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
        structured_output_schema: Optional[Any] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatOllama`` instance.

        :param model_name: Name of a model pulled into the local Ollama
            server (e.g. ``"qwen3"``), optionally prefixed with the
            ``"ollama:"`` sentinel (e.g. ``"ollama:qwen3:14b"``) so the
            resolver can route an arbitrary, non-catalog tag here.  The
            prefix, if present, is stripped before the tag reaches
            ``ChatOllama``.
        :param base_url: Ollama daemon base URL; defaults to
            ``"http://localhost:11434"``.
        :returns: A ``ChatOllama`` instance.
        :raises ImportError: If ``langchain_ollama`` is not installed.
        """
        try:
            from langchain_ollama import (  # pylint: disable=import-outside-toplevel,import-error
                ChatOllama,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-ollama' package is required for the Ollama "
                "provider. Install it with: pip install bili-core[ollama]"
            ) from exc

        # Strip the "ollama:" sentinel used by the resolver's heuristic
        # routing (bili.aether.compiler.llm_resolver) for non-catalog tags.
        # The single-turn path passes the bare tag directly, so the prefix
        # is optional here -- only stripped when present.
        resolved_model_name = (
            model_name[len(OLLAMA_MODEL_PREFIX) :]
            if model_name.startswith(OLLAMA_MODEL_PREFIX)
            else model_name
        )

        resolved_base_url = base_url or DEFAULT_OLLAMA_BASE_URL
        LOGGER.info(
            "Initializing Ollama model '%s' at %s",
            resolved_model_name,
            resolved_base_url,
        )

        config: dict = {"model": resolved_model_name, "base_url": resolved_base_url}
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
        if structured_output_schema is not None:
            # ChatOllama's format parameter accepts a JSON schema dict; the
            # daemon compiles it to a GBNF grammar for constrained decoding.
            config["format"] = normalize_schema(structured_output_schema)

        llm = ChatOllama(**config)
        LOGGER.debug(llm)
        return llm
