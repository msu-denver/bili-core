"""Google Generative AI (Gemini direct API) provider for bili-core IRIS.

Wraps ``langchain_google_genai.ChatGoogleGenerativeAI`` behind the
:class:`LLMProvider` interface.  Provides access to Gemini models via the
Google AI Developer API, as a complement to the existing
``remote_google_vertex`` provider which routes through Google Cloud Vertex AI.

Authentication reads ``GOOGLE_API_KEY`` from the environment.

Selecting the Developer API for an arbitrary model
--------------------------------------------------
An AETHER ``AgentSpec`` carries only ``model_name``, resolved to a provider via
catalog lookup and prefix heuristics (see
``bili.aether.compiler.llm_resolver``).  Catalog lookup runs first and reaches
``remote_google_vertex`` before this provider, so a bare Gemini ``model_id``
that both providers list (e.g. ``"gemini-2.5-flash"``) resolves to Vertex and a
caller has no way to ask for the Developer API by model id alone.  Prefix the
``model_name`` with the ``genai:`` sentinel (e.g. ``"genai:gemini-2.5-flash"``)
to force this provider for any Developer API model, catalogued or not,
mirroring the ``ollama:`` and ``cli:`` sentinels.  :meth:`load` strips the
``genai:`` prefix before constructing ``ChatGoogleGenerativeAI`` so the API
receives the real model id.  The prefix is optional: passing a bare model id
with an explicit ``provider_type`` continues to work unchanged.

Heavy dependencies
------------------
``langchain_google_genai`` is imported inside :meth:`GoogleGenAIProvider.load`
to avoid module-level import cost when this provider is not in use.
"""

# pylint: disable=duplicate-code
import logging
from typing import Any, Optional

from .base import LLMProvider
from .structured_output import gemini_response_schema

LOGGER = logging.getLogger(__name__)

#: Sentinel prefix that routes an arbitrary ``model_name`` to this provider via
#: the resolver's heuristic rules, overriding the catalog lookup that would
#: otherwise send a Vertex-catalogued Gemini id to ``remote_google_vertex``
#: (mirrors the "ollama:" and "cli:" sentinels).  Stripped in :meth:`load`.
GOOGLE_GENAI_MODEL_PREFIX = "genai:"


# pylint: disable=too-few-public-methods
class GoogleGenAIProvider(LLMProvider):
    """LangChain-native Google Generative AI (Gemini) provider.

    Accepted kwargs
    ---------------
    model_name : str
        Gemini model identifier (e.g. ``"gemini-3.1-flash-lite"``,
        ``"gemini-2.5-flash"``), optionally prefixed with the ``"genai:"``
        sentinel to force the Developer API for a model id the catalog also
        lists under Vertex.
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
    structured_output_schema : dict or type, optional
        JSON schema (or Pydantic model class) to constrain generation to.
        Sets ``response_schema`` and ``response_mime_type=
        "application/json"`` (Gemini controlled generation).
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = None,
        structured_output_schema: Optional[Any] = None,
        **_extra: Any,
    ) -> Any:
        """Create and return a ``ChatGoogleGenerativeAI`` instance.

        :param model_name: Gemini model identifier, optionally prefixed with
            the ``"genai:"`` sentinel (e.g. ``"genai:gemini-2.5-flash"``) so
            the resolver routes a Vertex-catalogued id here.  The prefix, if
            present, is stripped before the id reaches the API.
        :returns: A ``ChatGoogleGenerativeAI`` instance.
        :raises ImportError: If ``langchain_google_genai`` is not installed.
        """
        from langchain_google_genai import (  # pylint: disable=import-outside-toplevel,import-error
            ChatGoogleGenerativeAI,
        )

        # Strip the "genai:" sentinel used by the resolver's heuristic routing
        # (bili.aether.compiler.llm_resolver) to force the Developer API for an
        # id the catalog also lists under Vertex.  Callers passing an explicit
        # provider_type send a bare id, so the prefix is only stripped when
        # present.
        resolved_model_name = (
            model_name[len(GOOGLE_GENAI_MODEL_PREFIX) :]
            if model_name.startswith(GOOGLE_GENAI_MODEL_PREFIX)
            else model_name
        )

        LOGGER.info("Initializing Google GenAI model: %s", resolved_model_name)

        config: dict = {"model": resolved_model_name}
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
        if structured_output_schema is not None:
            config["response_schema"] = gemini_response_schema(structured_output_schema)
            config["response_mime_type"] = "application/json"

        llm = ChatGoogleGenerativeAI(**config)
        LOGGER.debug(llm)
        return llm
