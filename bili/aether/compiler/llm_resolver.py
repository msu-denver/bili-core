"""LLM resolution — maps AgentSpec model names to LLM provider instances.

Resolves ``AgentSpec.model_name`` to a provider type and ``model_id``
using ``bili.iris.config.llm_config.LLM_MODELS``, then instantiates the LLM
via ``bili.iris.loaders.llm_loader.load_model``.

bili-core distinguishes between a display *model_name* (e.g.
``"GPT-4o"``) and the actual *model_id* sent to the provider (e.g.
``"gpt-4o"``).  This module handles that mapping so AETHER users can
specify either form in their ``AgentSpec.model_name`` field.

A name is resolved in four steps, and :func:`describe_model_resolution`
reports which one answered and why:

1. A **qualified** name, ``<provider_type>:<model>``, names its provider
   outright.  ``AgentSpec`` has no provider field, so for a declarative run
   this is the only way to override the preference below.
2. An exact match on ``model_id`` or display ``model_name`` in
   ``LLM_MODELS``.
3. When step 2 matches **more than one provider**, the collision is broken
   by :data:`bili.iris.config.model_families.MODEL_FAMILY_OWNERS` rather
   than by the order the providers happen to appear in the catalog literal.
4. The heuristic prefix/substring rules below, for ids the catalog does not
   carry at all.

All heavy imports (torch, provider SDKs) are lazy to allow the compiler
module to load without those dependencies installed.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bili.aether.schema import AgentSpec, OutputFormat

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic provider detection (fallback when LLM_MODELS lookup fails)
# ---------------------------------------------------------------------------

_HEURISTIC_RULES = [
    # (substring_or_prefix, provider_type)
    # Order matters -- more specific patterns must come before broader ones,
    # since matching below is a plain substring test ("pattern in lower"),
    # not an anchored prefix match.  The heuristic layer only fires when
    # LLM_MODELS lookup returns nothing, so these rules apply to non-catalog
    # model IDs only.
    #
    # Explicit sentinel prefixes MUST precede every vendor substring rule
    # below.  A sentinel-prefixed tag can legitimately embed a vendor
    # substring (e.g. "ollama:deepseek-r1:14b" contains "deepseek-";
    # "ollama:llama-3.1-8b" contains "llama-3"), and since this loop takes
    # the first match, an unrelated vendor rule appearing earlier would
    # silently steal the match and misroute a local/CLI tag to a remote
    # provider. Placing both sentinels first means an explicit routing
    # prefix always wins over an incidental vendor substring.
    ("cli:", "cli"),  # Subprocess CLI provider sentinel
    # Local Ollama server sentinel.  The resolver keeps model_id unchanged
    # on a heuristic match (same as "cli:" above); OllamaProvider.load()
    # strips the "ollama:" prefix itself before passing the bare tag to
    # ChatOllama, since (unlike the CLI provider) it forwards model_name
    # straight to the client rather than taking its config from a separate
    # command kwarg.
    ("ollama:", "local_ollama"),
    # Google AI Developer API sentinel.  Unlike the vendor rules below, this
    # exists to override the *catalog* lookup rather than to name a provider
    # the heuristics could not otherwise guess: a Gemini model_id listed by
    # both remote_google_vertex and remote_google_genai resolves to Vertex
    # (catalog lookup runs before the heuristics, and Vertex is declared
    # first), so a bare id gives callers no way to select the Developer API.
    # A "genai:"-prefixed name misses the catalog, falls through to here, and
    # routes explicitly.  GoogleGenAIProvider.load() strips the prefix before
    # the id reaches the API, the same contract as "ollama:" above.
    ("genai:", "remote_google_genai"),
    ("gpt-", "remote_openai"),
    ("gpt4", "remote_openai"),
    ("o1-", "remote_openai"),
    ("o1", "remote_openai"),
    ("o3-", "remote_openai"),
    ("o3", "remote_openai"),
    # Bedrock-hosted models use dotted-namespace prefixes.  Check these first
    # so that provider-bare patterns below do not intercept them.
    ("anthropic.claude", "remote_aws_bedrock"),
    ("amazon.nova", "remote_aws_bedrock"),
    ("amazon.titan", "remote_aws_bedrock"),
    ("meta.llama", "remote_aws_bedrock"),
    ("cohere.command", "remote_aws_bedrock"),
    ("mistral.mistral", "remote_aws_bedrock"),
    # Direct-API heuristics for non-Bedrock-namespaced model IDs.
    # Each pattern is more specific than the broad fallbacks below,
    # so place them first.
    ("claude-", "remote_anthropic"),  # Anthropic direct API
    ("mistral-", "remote_mistral"),  # Mistral AI direct (not Bedrock)
    ("codestral", "remote_mistral"),  # Mistral's code model
    ("command-", "remote_cohere"),  # Cohere Command family
    # "gemini-" routes to the Google AI Developer API.  Users who want
    # Vertex AI should select a model by its catalog display name, use the
    # Vertex-registered model_id directly, or invoke load_model() with
    # provider_type="remote_google_vertex" explicitly.
    ("gemini-", "remote_google_genai"),  # Google GenAI developer API
    ("deepseek-", "remote_deepseek"),  # DeepSeek direct API
    ("grok-", "remote_xai"),  # xAI Grok
    ("llama-3", "remote_groq"),  # Groq-hosted Llama
    ("compound-beta", "remote_groq"),  # Groq compound system
    ("gemma2-", "remote_groq"),  # Groq-hosted Gemma
    # Broad pre-existing fallbacks -- preserved for backward compatibility.
    # These fire for non-catalog model IDs that match only the bare vendor
    # name (e.g. bare "gemini", legacy Bedrock-style "mistral-..." that did
    # not match "mistral.mistral-*" above).  Because "mistral-" already
    # routes to remote_mistral, the "mistral" fallback below only triggers
    # for strings containing "mistral" but NOT "mistral-" (e.g. a bare
    # "mistral" string or "mistral_v2").
    ("gemini", "remote_google_vertex"),  # bare/non-hyphenated gemini IDs
    ("mistral", "remote_aws_bedrock"),  # legacy Bedrock Mistral catch-all
]


#: Prefixes in :data:`_HEURISTIC_RULES` that name a routing target outright
#: rather than describing a vendor's model ids.  Derived from the table so a
#: fourth sentinel added there is covered without a second list.
#:
#: These win over the ``<provider_type>:<model>`` qualified form.  ``"cli:"``
#: is both a sentinel and a provider type, and its documented contract is
#: that ``model_id`` keeps the prefix (the CLI provider takes its model from
#: a separate kwarg), so reading it as a qualifier would strip a prefix the
#: provider expects.
_SENTINEL_PREFIXES: Tuple[str, ...] = tuple(
    pattern for pattern, _ in _HEURISTIC_RULES if pattern.endswith(":")
)


@dataclass(frozen=True)
class ModelResolution:
    """How a model name was resolved, and why.

    Returned by :func:`describe_model_resolution` so a caller can report
    which provider a *bare* name reached instead of discovering it at the
    provider boundary.

    :ivar model_name: The name that was asked for.
    :ivar provider_type: The ``LLM_MODELS`` key the name resolved to.
    :ivar model_id: The id to send to that provider.
    :ivar extra_kwargs: Provider-specific parameters from the catalog entry
        (empty when no entry backed the resolution).
    :ivar source: Which step answered -- one of :data:`RESOLUTION_SOURCES`.
    :ivar reason: A sentence naming why, suitable for a log line.
    :ivar candidates: Every provider whose catalog carries the name, in
        catalog order.  Empty when no catalog entry matched.
    """

    model_name: str
    provider_type: str
    model_id: str
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    source: str = "catalog"
    reason: str = ""
    candidates: Tuple[str, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        """Whether several providers catalog the name and none was preferred."""
        return self.source == "catalog-ambiguous"


#: Every value :attr:`ModelResolution.source` can take.
#:
#: ``qualified``
#:     The name carried its provider (``<provider_type>:<model>``).
#: ``catalog``
#:     Exactly one provider catalogs the name.
#: ``catalog-disambiguated``
#:     Several do, and the model family's owner was preferred.
#: ``catalog-ambiguous``
#:     Several do, and no rule covers the family, so the first in catalog
#:     order was taken.  Reported rather than raised: refusing here would
#:     break a working deployment over a catalog edit.
#: ``heuristic``
#:     No provider catalogs the name; a prefix/substring rule routed it.
RESOLUTION_SOURCES: Tuple[str, ...] = (
    "qualified",
    "catalog",
    "catalog-disambiguated",
    "catalog-ambiguous",
    "heuristic",
)


def _llm_models() -> Dict[str, Any]:
    """Return ``LLM_MODELS``, or an empty mapping when iris config is absent.

    The import is lazy and guarded so this compiler module keeps loading
    without the catalog installed; the resolver then falls through to the
    heuristic rules, which is the pre-existing degrade.
    """
    try:
        from bili.iris.config.llm_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:
        LOGGER.debug(
            "bili.iris.config.llm_config not available; skipping LLM_MODELS lookup"
        )
        return {}
    return LLM_MODELS


def _split_qualified(model_name: str) -> Optional[Tuple[str, str]]:
    """Split a ``<provider_type>:<model>`` name, or return ``None``.

    The qualifier is the provider type itself, so there is no alias table to
    keep in step with the catalog: the accepted prefixes are exactly the keys
    of ``LLM_MODELS``.  A name whose prefix is not one of those keys is not a
    qualified name and is left alone, which is what keeps an ordinary id
    carrying a colon (a local tag such as ``"qwen3:8b"``) working.

    :param model_name: The name from the caller.
    :returns: ``(provider_type, model)`` or ``None``.
    """
    if model_name.lower().startswith(_SENTINEL_PREFIXES):
        return None
    prefix, sep, rest = model_name.partition(":")
    if not sep or not rest:
        return None
    if prefix not in _llm_models():
        return None
    return prefix, rest


def _resolve_qualified(model_name: str) -> Optional[ModelResolution]:
    """Resolve a ``<provider_type>:<model>`` name, or return ``None``.

    A qualified name binds its provider even when that provider's catalog
    does not carry the model: the caller named the provider explicitly, and a
    passthrough id is legitimate (a locally pulled tag, a new model the
    catalog has not caught up with).  Falling through instead would leave the
    prefix inside ``model_id`` and send the provider a name it cannot serve.
    """
    split = _split_qualified(model_name)
    if split is None:
        return None
    provider, model = split

    for hit_provider, model_id, extra_kwargs in _lookup_in_llm_models(model):
        if hit_provider == provider:
            return ModelResolution(
                model_name=model_name,
                provider_type=provider,
                model_id=model_id,
                extra_kwargs=extra_kwargs,
                source="qualified",
                reason=f"the name named provider '{provider}' outright",
                candidates=_catalog_candidates(model),
            )

    return ModelResolution(
        model_name=model_name,
        provider_type=provider,
        model_id=model,
        source="qualified",
        reason=(
            f"the name named provider '{provider}' outright; that catalog has "
            f"no entry for '{model}', which is passed through as the model id"
        ),
        candidates=_catalog_candidates(model),
    )


def _catalog_candidates(name: str) -> Tuple[str, ...]:
    """Return each provider whose catalog carries *name*, in catalog order."""
    seen: List[str] = []
    for provider, _model_id, _kwargs in _lookup_in_llm_models(name):
        if provider not in seen:
            seen.append(provider)
    return tuple(seen)


def _resolve_from_catalog(model_name: str) -> Optional[ModelResolution]:
    """Resolve *model_name* against ``LLM_MODELS``, or return ``None``.

    When more than one provider catalogs the name, the tie is broken by the
    model family's owner rather than by catalog order.  The preferred
    provider is always one of the candidates, so a preference reorders the
    providers that carry the model and can never introduce one that does not.
    """
    hits = _lookup_in_llm_models(model_name)
    if not hits:
        return None

    by_provider: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for provider, model_id, extra_kwargs in hits:
        by_provider.setdefault(provider, (model_id, extra_kwargs))
    candidates = tuple(by_provider)

    first_provider = candidates[0]
    first_model_id, first_kwargs = by_provider[first_provider]

    if len(candidates) == 1:
        return ModelResolution(
            model_name=model_name,
            provider_type=first_provider,
            model_id=first_model_id,
            extra_kwargs=first_kwargs,
            source="catalog",
            reason=f"'{first_provider}' is the only provider cataloging this name",
            candidates=candidates,
        )

    preference = _prefer_owner(first_model_id, candidates)
    if preference is not None:
        provider, why = preference
        model_id, extra_kwargs = by_provider[provider]
        return ModelResolution(
            model_name=model_name,
            provider_type=provider,
            model_id=model_id,
            extra_kwargs=extra_kwargs,
            source="catalog-disambiguated",
            reason=why,
            candidates=candidates,
        )

    return ModelResolution(
        model_name=model_name,
        provider_type=first_provider,
        model_id=first_model_id,
        extra_kwargs=first_kwargs,
        source="catalog-ambiguous",
        reason=(
            "several providers catalog this name and no family rule covers it, "
            "so the first in catalog order was taken"
        ),
        candidates=candidates,
    )


def _prefer_owner(
    model_id: str, candidates: Tuple[str, ...]
) -> Optional[Tuple[str, str]]:
    """Return ``(provider_type, reason)`` for the family owner, or ``None``.

    Guarded and lazy for the same reason as :func:`_llm_models`: without the
    iris config package there is no table to consult and the caller keeps its
    catalog-order answer.
    """
    try:
        from bili.iris.config.model_families import (  # noqa: E402  pylint: disable=import-outside-toplevel
            disambiguate,
        )
    except ImportError:  # pragma: no cover - mirrors the _llm_models degrade
        LOGGER.debug(
            "bili.iris.config.model_families not available; "
            "a colliding name keeps its catalog-order provider"
        )
        return None
    return disambiguate(model_id, candidates)


def _resolve_heuristically(model_name: str) -> Optional[ModelResolution]:
    """Route a name no provider catalogs by prefix/substring, or ``None``."""
    lower = model_name.lower()
    for pattern, ptype in _HEURISTIC_RULES:
        if pattern in lower:
            return ModelResolution(
                model_name=model_name,
                provider_type=ptype,
                model_id=model_name,
                source="heuristic",
                reason=(
                    f"no provider catalogs this name; the '{pattern}' rule routes "
                    f"it to '{ptype}' and it is used as the model id"
                ),
            )
    return None


def describe_model_resolution(model_name: str) -> ModelResolution:
    """Resolve *model_name* and report which step answered and why.

    This is the resolution :func:`resolve_model`, :func:`resolve_provider`
    and :func:`create_llm` all run; those return only part of it.  Use this
    one to log or assert *which* provider a bare name reached, which is not
    otherwise observable until the provider call fails.

    :param model_name: A display name, a model id, or a
        ``<provider_type>:<model>`` qualified name.
    :returns: The :class:`ModelResolution`.
    :raises ValueError: If the name cannot be resolved to any provider.
    """
    resolution = (
        _resolve_qualified(model_name)
        or _resolve_from_catalog(model_name)
        or _resolve_heuristically(model_name)
    )

    if resolution is None:
        raise ValueError(
            f"Cannot resolve model '{model_name}' to a provider. "
            f"Set a recognised model_name or use bili.loaders.llm_loader directly."
        )

    if resolution.is_ambiguous:
        # Loud, because it means a catalog edit added a collision no family
        # rule covers, and the answer is then whatever the catalog literal
        # happens to order first.  It cannot fire on the shipped catalog: the
        # tests derive every collision from it and require a rule.
        LOGGER.warning(
            "Model '%s' is cataloged by %s; %s. Qualify the name as "
            "'<provider_type>:%s' to select one.",
            model_name,
            ", ".join(resolution.candidates),
            resolution.reason,
            resolution.model_id,
        )
    else:
        # A disambiguated name is announced by the caller that acts on it
        # (:func:`create_llm`), not here.  ``AgentSpec`` validation resolves a
        # name on every construction, so an INFO line at this level would
        # repeat the same sentence several times per agent.
        LOGGER.debug(
            "Resolved '%s' via %s → provider=%s, model_id=%s",
            model_name,
            resolution.source,
            resolution.provider_type,
            resolution.model_id,
        )

    return resolution


def _resolved_catalog_entry(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the catalog entry *model_name* resolves to, or ``None``.

    Every per-model field read from the catalog by name has to come from the
    entry the model will actually be LOADED from, or a caller gets one
    provider's model with another provider's declared limits.  That is not
    hypothetical: the two entries for a colliding id are written independently
    and do diverge (``gpt-4`` declares 8192 input tokens under the direct API
    and 128000 under the re-host).  Reading the first catalog match was
    consistent while the loader also took the first match, and stopped being
    consistent when the loader started preferring the family owner.

    ``None`` means no catalog entry backs the name: it is not in the catalog,
    the catalog is not importable, it routed by heuristic, or it is a
    qualified name for a model that provider does not carry.  Callers treat
    that as "no declared value", never as a zero or a default.
    """
    try:
        resolution = describe_model_resolution(model_name)
    except ValueError:
        return None
    if resolution.source == "heuristic":
        return None
    provider_info = _llm_models().get(resolution.provider_type, {})
    for entry in provider_info.get("models", []):
        if entry.get("model_id") == resolution.model_id:
            return entry
    return None


def _resolve_model_full(
    model_name: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve a model name to ``(provider_type, model_id, extra_kwargs)``.

    Thin projection of :func:`describe_model_resolution`, kept because
    several callers want only the triple.

    Raises:
        ValueError: If the model cannot be resolved to any provider.
    """
    resolution = describe_model_resolution(model_name)
    return resolution.provider_type, resolution.model_id, resolution.extra_kwargs


def resolve_model(model_name: str) -> Tuple[str, str]:
    """Resolve a model name to a ``(provider_type, model_id)`` pair.

    Search order:
        1. A ``<provider_type>:<model>`` qualified name
        2. Exact match on ``model_id`` or display ``model_name`` in
           ``LLM_MODELS``, with a name several providers catalog broken by
           the model family's owner
        3. Heuristic fallback using prefix/substring rules
           (assumes *model_name* is already the *model_id*)

    Use :func:`describe_model_resolution` when the *reason* matters -- which
    step answered, and which other providers catalog the same name.

    Args:
        model_name: The model identifier from ``AgentSpec.model_name``.
            Can be a display name (``"GPT-4o"``), a model ID
            (``"gpt-4o"``), or a qualified name
            (``"remote_azure_openai:gpt-4o"``).

    Returns:
        A ``(provider_type, model_id)`` tuple — e.g.
        ``("remote_openai", "gpt-4o")``.

    Raises:
        ValueError: If the model cannot be resolved to any provider.
    """
    provider, model_id, _ = _resolve_model_full(model_name)
    return provider, model_id


def resolve_provider(model_name: str) -> str:
    """Resolve a model name to a bili-core provider type string.

    Convenience wrapper around :func:`resolve_model` that returns only
    the provider type.
    """
    provider, _ = resolve_model(model_name)
    return provider


def _forward_cli_subprocess_kwargs(
    agent: AgentSpec, provider: str, kwargs: Dict[str, Any]
) -> None:
    """Forward ``AgentSpec.cli_subprocess_*`` fields into *kwargs* for CLI providers.

    Mutates *kwargs* in place.  A no-op entirely when *provider* is not a CLI
    subprocess type -- passing these kwargs to an API provider's loader would
    raise an unexpected-keyword-argument error, since those loader functions
    have explicit signatures without ``**kwargs``.

    :param agent: The ``AgentSpec`` whose ``cli_subprocess_*`` fields may be set.
    :param provider: The resolved provider type string (e.g. ``"cli_claude_code"``).
    :param kwargs: The in-progress ``load_model`` kwargs dict; updated in place.
    """
    if not provider.startswith("cli"):
        return

    if agent.cli_subprocess_timeout is not None:
        # 0 is the user's signal for "no timeout" (matches the ge=0 constraint
        # on the field).  Translate it to None so subprocess.run receives
        # timeout=None rather than timeout=0 (which would expire immediately).
        raw = agent.cli_subprocess_timeout
        kwargs["timeout_seconds"] = None if raw == 0.0 else raw

    if agent.cli_subprocess_cwd is not None:
        kwargs["cwd"] = agent.cli_subprocess_cwd

    if agent.cli_subprocess_max_retries is not None:
        kwargs["max_retries"] = agent.cli_subprocess_max_retries

    if agent.cli_subprocess_retry_backoff is not None:
        kwargs["retry_backoff_seconds"] = agent.cli_subprocess_retry_backoff

    if agent.cli_subprocess_model is not None:
        kwargs["model"] = agent.cli_subprocess_model

    if agent.cli_subprocess_reasoning_effort is not None:
        kwargs["reasoning_effort"] = agent.cli_subprocess_reasoning_effort


def _resolve_structured_schema(agent: AgentSpec, provider: str) -> Optional[dict]:
    """Return the JSON schema to bind for decode-time enforcement, or ``None``.

    An agent declaring ``output_format="structured"`` with an
    ``output_schema`` gets the schema bound at model-load time
    (``structured_output_schema``) so generation is constrained to
    schema-valid output, when both of these hold:

    - The agent has no tools.  Constrained generation applies to every
      assistant turn, which would also constrain the intermediate turns of a
      tool-calling loop; the two are mutually exclusive on this seam.
    - The resolved provider has decode-time enforcement wired
      (:func:`bili.iris.providers.structured_output.supports_structured_output`).

    When either condition fails the schema is not bound and a warning is
    logged; the agent still runs, and ``_build_output`` in the agent
    generator validates the output post-hoc against the same schema.  This
    graceful degradation mirrors how tool/middleware resolution failures are
    handled: a MAS config never becomes un-runnable because one model lacks
    a capability.
    """
    if agent.output_format != OutputFormat.STRUCTURED or not agent.output_schema:
        return None

    if agent.tools:
        LOGGER.warning(
            "Agent '%s': output_format='structured' is not decode-time "
            "enforced for tool-calling agents; the schema will be validated "
            "post-hoc only. Produce large structured documents with a "
            "dedicated tool-less agent to get constrained generation.",
            agent.agent_id,
        )
        return None

    from bili.iris.providers.structured_output import (  # noqa: E402  pylint: disable=import-outside-toplevel
        supports_structured_output,
    )

    if not supports_structured_output(provider):
        LOGGER.warning(
            "Agent '%s': provider '%s' has no decode-time structured-output "
            "enforcement; the schema will be validated post-hoc only.",
            agent.agent_id,
            provider,
        )
        return None

    return agent.output_schema


def _load_fallback_member(provider_type: str, member_kwargs: dict) -> Any:
    """Load one fallback-chain member through the ``load_model`` choke point.

    A ``FallbackLLM`` loads its members through an injected loader.  The primary
    LLM is created via ``load_model``, which applies catalog-derived load
    defaults (the output-token budget and temperature resilience).  A fallback
    member loaded by the bare provider would miss those, so a fail-over would
    silently drop to the provider's own small ``max_tokens`` default and lose
    the temperature handling.  Routing members through the same ``load_model``
    gives the whole chain identical treatment.

    :param provider_type: The member's provider type.
    :param member_kwargs: The member's load kwargs (already carrying only a
        ``structured_output_schema`` the provider supports, so ``load_model``'s
        fail-fast gate does not trip on an unconstrained member).
    :returns: The loaded LLM object.
    """
    from bili.iris.loaders.llm_loader import (  # noqa: E402  pylint: disable=import-outside-toplevel
        load_model,
    )

    return load_model(provider_type, **member_kwargs)


def create_llm(agent: AgentSpec) -> Any:
    """Create a LangChain-compatible chat model from an ``AgentSpec``.

    Lazy-imports ``bili.iris.loaders.llm_loader.load_model`` and
    ``bili.iris.config.llm_config.LLM_MODELS`` so the compiler module can
    be loaded without heavy provider dependencies.

    The function resolves the display ``model_name`` to the actual
    ``model_id`` expected by the provider, then delegates to ``load_model``.

    When ``agent.fallback_models`` is non-empty, the returned object is a
    :class:`~bili.iris.providers.fallback.FallbackLLM` that transparently
    tries each fallback provider on retryable errors (rate limits, transient
    API failures).  The same ``temperature`` and ``max_tokens`` values are
    applied to each fallback.  Callers see no difference — the returned
    object always exposes ``.invoke()`` / ``.stream()`` / ``.astream()``.

    When ``agent.fallback_models`` is empty (the default), this function
    returns the primary LLM object directly — behaviour is identical to
    before the fallback engine was introduced.

    Args:
        agent: An ``AgentSpec`` with ``model_name`` set.

    Returns:
        A chat model ready for ``.invoke()``.  Will be a plain LLM object
        when no fallbacks are configured, or a
        :class:`~bili.iris.providers.fallback.FallbackLLM` proxy when
        ``agent.fallback_models`` is populated.

    Raises:
        ValueError: If ``agent.model_name`` is ``None`` or unresolvable,
            or if any fallback model name cannot be resolved.
    """
    if not agent.model_name:
        raise ValueError(
            f"AgentSpec '{agent.agent_id}' has no model_name; "
            f"cannot create LLM instance."
        )

    resolution = describe_model_resolution(agent.model_name)
    provider = resolution.provider_type
    model_id = resolution.model_id

    # Build kwargs for load_model — extra_kwargs first so the resolved
    # model_id always wins if extra_kwargs ever contains a "model_name" key.
    kwargs: Dict[str, Any] = {**resolution.extra_kwargs, "model_name": model_id}
    if agent.temperature is not None:
        kwargs["temperature"] = agent.temperature
    if agent.max_tokens is not None:
        kwargs["max_tokens"] = agent.max_tokens

    # Bind the agent's output_schema for decode-time enforcement when the
    # provider supports it (see _resolve_structured_schema for conditions).
    structured_schema = _resolve_structured_schema(agent, provider)
    if structured_schema is not None:
        kwargs["structured_output_schema"] = structured_schema

    # Forward cli_subprocess_* fields (timeout, cwd, retry policy, model,
    # reasoning effort) to CLI providers only; see
    # _forward_cli_subprocess_kwargs for the per-field detail.
    _forward_cli_subprocess_kwargs(agent, provider, kwargs)

    LOGGER.info(
        "Creating LLM for agent '%s': provider=%s, model_id=%s (%s: %s)%s",
        agent.agent_id,
        provider,
        model_id,
        resolution.source,
        resolution.reason,
        (
            f". Also cataloged by "
            f"{', '.join(p for p in resolution.candidates if p != provider)}; "
            f"qualify the name as '<provider_type>:{model_id}' to select one."
            if resolution.source == "catalog-disambiguated"
            else ""
        ),
    )

    from bili.iris.loaders.llm_loader import (  # noqa: E402  pylint: disable=import-outside-toplevel
        load_model,
    )

    primary_llm = load_model(provider, **kwargs)

    # --- Fallback engine (opt-in) -------------------------------------------
    # If AgentSpec.fallback_models is empty, return the primary LLM directly.
    # No change in behaviour for callers that do not configure fallbacks.
    if not agent.fallback_models:
        return primary_llm

    # Build the ordered fallback chain from the AgentSpec's fallback_models
    # list.  Each model name is resolved exactly like the primary model_name.
    fallback_chain: List[Tuple[str, Dict[str, Any]]] = []
    for fb_model_name in agent.fallback_models:
        # Resolved through the same entry point as the primary, so a bare
        # fallback name -- which is what a fallback list usually holds, since
        # it carries no provider either -- is disambiguated and reported the
        # same way rather than silently taking a different route.
        fb_resolution = describe_model_resolution(fb_model_name)
        fb_provider = fb_resolution.provider_type
        fb_model_id = fb_resolution.model_id
        fb_kwargs: Dict[str, Any] = {
            **fb_resolution.extra_kwargs,
            "model_name": fb_model_id,
        }
        if agent.temperature is not None:
            fb_kwargs["temperature"] = agent.temperature
        if agent.max_tokens is not None:
            fb_kwargs["max_tokens"] = agent.max_tokens
        # Structured-output support is evaluated per fallback provider: a
        # chain may mix constrained and unconstrained backends, and an
        # unsupported fallback must not fail load_model's fail-fast gate when
        # the member is loaded through it (see _load_fallback_member).
        fb_schema = _resolve_structured_schema(agent, fb_provider)
        if fb_schema is not None:
            fb_kwargs["structured_output_schema"] = fb_schema
        fallback_chain.append((fb_provider, fb_kwargs))
        LOGGER.debug(
            "Agent '%s': registered fallback provider=%s, model_id=%s (%s)",
            agent.agent_id,
            fb_provider,
            fb_model_id,
            fb_resolution.source,
        )

    from bili.iris.providers.fallback import (  # noqa: E402  pylint: disable=import-outside-toplevel
        build_fallback_llm,
    )

    LOGGER.info(
        "Agent '%s': wrapping primary LLM with %d fallback(s): %s",
        agent.agent_id,
        len(fallback_chain),
        [entry[0] for entry in fallback_chain],
    )
    return build_fallback_llm(
        primary_llm=primary_llm,
        fallback_chain=fallback_chain,
        loader=_load_fallback_member,
    )


def resolve_tool_strategy(model_name: str) -> str:
    """Return the ``tool_strategy`` for *model_name* from ``LLM_MODELS``.

    The ``tool_strategy`` field classifies how agents should invoke tools for a
    given model:

    - ``"native"``     -- the model implements ``bind_tools``; use
                          ``create_agent`` + LangChain tool-calling.
    - ``"facilitated"`` -- the model cannot bind tools natively; route to the
                           prompted ReAct loop (hand-rolled Thought/Action/
                           Observation cycle described in the system message).
    - ``"mcp"``        -- the model is an agentic CLI best consumed as an MCP
                           server; until the MCP mechanism lands (#311) it runs
                           on the tool-less plain path so the model can self-
                           orchestrate.
    - ``"none"``       -- the model has no tool support at all (e.g. reasoning
                           models that reject extra kwargs); runs tool-less.

    Fall-back behaviour when the field is absent from the catalog entry:

    - If the entry has ``"supports_tools": False`` the strategy is inferred as
      ``"facilitated"`` (preserves pre-migration behaviour).
    - If the entry has ``"supports_tools": True`` or omits the field entirely,
      the strategy defaults to ``"native"``.
    - If the model is not in the catalog at all, ``"native"`` is returned so
      unknown API models continue to work as before.

    Args:
        model_name: The model identifier from ``AgentSpec.model_name``.
            Can be a display name (e.g. ``"GPT-4o"``) or a model ID
            (e.g. ``"gpt-4o"``).

    Returns:
        One of ``"native"``, ``"facilitated"``, ``"mcp"``, or ``"none"``.
    """
    entry = _resolved_catalog_entry(model_name)
    if entry is None:
        LOGGER.debug(
            "No catalog entry backs '%s'; assuming tool_strategy='native'",
            model_name,
        )
        return "native"

    if "tool_strategy" in entry:
        return entry["tool_strategy"]
    # Backward-compat: infer from legacy supports_tools flag.
    return "native" if entry.get("supports_tools", True) else "facilitated"


def resolve_prompt_length_limit(model_name: str) -> Optional[int]:
    """Return the declared maximum input-token limit for *model_name*, if known.

    Every model has a different prompt/context budget -- a small local model
    and a long-context frontier model tolerate very different prompt sizes --
    so a single hardcoded limit is wrong for the catalog as a whole regardless
    of what number is chosen. This gives callers (e.g. ``AgentSpec`` prompt
    validation, or any code composing a prompt before it knows which model
    will consume it) a way to look up the *actual* per-model limit and budget
    accordingly, rather than guessing.

    Args:
        model_name: The model identifier to look up. Can be a display name
            (e.g. ``"Claude Opus 4.8"``) or a model ID (e.g.
            ``"claude-opus-4-8"``), matched the same way as
            :func:`resolve_model`.

    Returns:
        The model's declared ``max_input_tokens`` from
        ``bili.iris.config.llm_config.LLM_MODELS``, or ``None`` when the
        model is not found in the catalog, the catalog entry does not
        declare a limit (e.g. CLI-subprocess and local providers, whose
        real limits depend on the underlying tool/hardware rather than
        bili-core's catalog), or the catalog module cannot be imported.
        ``None`` means "no known limit" -- callers should treat that as
        permissive (no cap), never as zero.
    """
    entry = _resolved_catalog_entry(model_name)
    if entry is None:
        LOGGER.debug(
            "No catalog entry backs '%s'; no known prompt length limit",
            model_name,
        )
        return None

    return entry.get("max_input_tokens")


def resolve_supports_tools(model_name: str) -> bool:
    """Return whether *model_name* supports native ``bind_tools``.

    This is a backward-compatible convenience wrapper around
    :func:`resolve_tool_strategy`.  Callers that only need a boolean — e.g.
    legacy code or the Streamlit UI — can continue using this function without
    change.  Prefer :func:`resolve_tool_strategy` for new code.

    Note: this function inspects only the primary *model_name*.  When a
    ``FallbackLLM`` chain mixes a tool-capable primary with a non-tool-capable
    fallback, this function reports the primary's capability only; the fallback
    model's strategy is not checked here.

    Args:
        model_name: The model identifier from ``AgentSpec.model_name``.

    Returns:
        ``True`` when the resolved strategy is ``"native"``; ``False``
        otherwise.
    """
    return resolve_tool_strategy(model_name) == "native"


def resolve_tools(agent: AgentSpec) -> list:
    """Resolve an ``AgentSpec``'s tool names to tool instances.

    Lazy-imports ``bili.loaders.tools_loader.initialize_tools`` and
    ``bili.config.tool_config.TOOLS`` so the compiler module can be
    loaded without those dependencies installed.

    Args:
        agent: An ``AgentSpec`` whose ``tools`` list may contain tool
            names registered in bili-core's ``TOOL_REGISTRY``.

    Returns:
        A list of LangChain ``Tool`` instances (empty if no tools
        are configured or if the tools loader is unavailable).
    """
    if not agent.tools:
        return []

    try:
        from bili.iris.config.tool_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            TOOLS as TOOL_CONFIG,
        )
        from bili.iris.loaders.tools_loader import (  # noqa: E402  pylint: disable=import-outside-toplevel
            initialize_tools,
        )
    except ImportError:
        LOGGER.warning(
            "bili.iris.loaders.tools_loader not available; "
            "skipping tool resolution for agent '%s'",
            agent.agent_id,
        )
        return []

    # Build prompts dict from tool_config defaults
    tool_prompts: Dict[str, str] = {}
    for tool_name in agent.tools:
        if tool_name in TOOL_CONFIG and "default_prompt" in TOOL_CONFIG[tool_name]:
            tool_prompts[tool_name] = TOOL_CONFIG[tool_name]["default_prompt"]

    try:
        return initialize_tools(
            active_tools=agent.tools,
            tool_prompts=tool_prompts,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.warning(
            "Failed to resolve tools %s for agent '%s'; "
            "agent will run without tools",
            agent.tools,
            agent.agent_id,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lookup_in_llm_models(
    model_name: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return every ``LLM_MODELS`` entry matching *model_name*, in catalog order.

    A match is on the entry's ``model_id`` (e.g. ``"gpt-4o"``) or its display
    ``model_name`` (e.g. ``"OpenAI GPT-4o Omni"``).  Each hit is
    ``(provider_type, model_id, extra_kwargs)``; ``extra_kwargs`` carries the
    provider-specific parameters stored in the entry's ``kwargs`` dict (e.g.
    ``api_version`` for Azure OpenAI models).

    Every hit is returned rather than the first, because the same id is
    legitimately cataloged by a first-party API and a re-host, and choosing
    between them is :func:`_resolve_from_catalog`'s decision to make with the
    full candidate set in hand.  An empty list means the catalog does not
    carry the name (or is not installed).
    """
    hits: List[Tuple[str, str, Dict[str, Any]]] = []
    for provider_type, provider_info in _llm_models().items():
        models: List[Dict[str, Any]] = provider_info.get("models", [])
        for entry in models:
            entry_model_id = entry.get("model_id", "")
            entry_display = entry.get("model_name", "")
            if model_name in (entry_model_id, entry_display):
                extra_kwargs: Dict[str, Any] = entry.get("kwargs", {})
                hits.append((provider_type, entry_model_id, extra_kwargs))
                break

    return hits
