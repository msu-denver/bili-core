"""LLM resolution — maps AgentSpec model names to LLM provider instances.

Resolves ``AgentSpec.model_name`` to a provider type and ``model_id``
using ``bili.iris.config.llm_config.LLM_MODELS``, then instantiates the LLM
via ``bili.iris.loaders.llm_loader.load_model``.

bili-core distinguishes between a display *model_name* (e.g.
``"GPT-4o"``) and the actual *model_id* sent to the provider (e.g.
``"gpt-4o"``).  This module handles that mapping so AETHER users can
specify either form in their ``AgentSpec.model_name`` field.

All heavy imports (torch, provider SDKs) are lazy to allow the compiler
module to load without those dependencies installed.
"""

import logging
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


def _resolve_model_full(
    model_name: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve a model name to ``(provider_type, model_id, extra_kwargs)`` in one pass.

    Search order:
        1. Exact match on ``model_id`` in ``LLM_MODELS``
        2. Exact match on display ``model_name`` in ``LLM_MODELS``
        3. Heuristic fallback using prefix/substring rules
           (``extra_kwargs`` is empty for heuristic matches)

    Raises:
        ValueError: If the model cannot be resolved to any provider.
    """
    # --- 1 & 2: Look up in LLM_MODELS (single pass, returns extra_kwargs) ---
    result = _lookup_in_llm_models(model_name)
    if result is not None:
        provider, model_id, extra_kwargs = result
        LOGGER.debug(
            "Resolved '%s' via LLM_MODELS → provider=%s, model_id=%s",
            model_name,
            provider,
            model_id,
        )
        return provider, model_id, extra_kwargs

    # --- 3: Heuristic fallback (model_name IS the model_id) ---
    lower = model_name.lower()
    for pattern, ptype in _HEURISTIC_RULES:
        if pattern in lower:
            LOGGER.debug(
                "Resolved '%s' via heuristic ('%s') → %s (using as model_id)",
                model_name,
                pattern,
                ptype,
            )
            return ptype, model_name, {}

    raise ValueError(
        f"Cannot resolve model '{model_name}' to a provider. "
        f"Set a recognised model_name or use bili.loaders.llm_loader directly."
    )


def resolve_model(model_name: str) -> Tuple[str, str]:
    """Resolve a model name to a ``(provider_type, model_id)`` pair.

    Search order:
        1. Exact match on ``model_id`` in ``LLM_MODELS``
        2. Exact match on display ``model_name`` in ``LLM_MODELS``
        3. Heuristic fallback using prefix/substring rules
           (assumes *model_name* is already the *model_id*)

    Args:
        model_name: The model identifier from ``AgentSpec.model_name``.
            Can be a display name (``"GPT-4o"``) or a model ID
            (``"gpt-4o"``).

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

    provider, model_id, extra_kwargs = _resolve_model_full(agent.model_name)

    # Build kwargs for load_model — extra_kwargs first so the resolved
    # model_id always wins if extra_kwargs ever contains a "model_name" key.
    kwargs: Dict[str, Any] = {**extra_kwargs, "model_name": model_id}
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
        "Creating LLM for agent '%s': provider=%s, model_id=%s",
        agent.agent_id,
        provider,
        model_id,
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
        fb_provider, fb_model_id, fb_extra = _resolve_model_full(fb_model_name)
        fb_kwargs: Dict[str, Any] = {**fb_extra, "model_name": fb_model_id}
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
            "Agent '%s': registered fallback provider=%s, model_id=%s",
            agent.agent_id,
            fb_provider,
            fb_model_id,
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
    try:
        from bili.iris.config.llm_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:
        LOGGER.debug(
            "bili.iris.config.llm_config not available; "
            "assuming tool_strategy='native' for '%s'",
            model_name,
        )
        return "native"

    for provider_info in LLM_MODELS.values():
        for entry in provider_info.get("models", []):
            if (
                entry.get("model_id") == model_name
                or entry.get("model_name") == model_name
            ):
                if "tool_strategy" in entry:
                    return entry["tool_strategy"]
                # Backward-compat: infer from legacy supports_tools flag.
                return "native" if entry.get("supports_tools", True) else "facilitated"

    LOGGER.debug(
        "'%s' not found in LLM_MODELS; assuming tool_strategy='native'",
        model_name,
    )
    return "native"


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
    try:
        from bili.iris.config.llm_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:
        LOGGER.debug(
            "bili.iris.config.llm_config not available; "
            "no known prompt length limit for '%s'",
            model_name,
        )
        return None

    for provider_info in LLM_MODELS.values():
        for entry in provider_info.get("models", []):
            if (
                entry.get("model_id") == model_name
                or entry.get("model_name") == model_name
            ):
                return entry.get("max_input_tokens")

    LOGGER.debug(
        "'%s' not found in LLM_MODELS; no known prompt length limit",
        model_name,
    )
    return None


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
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Search ``LLM_MODELS`` for a matching model entry.

    Returns ``(provider_type, model_id, extra_kwargs)`` if found,
    ``None`` otherwise.  ``extra_kwargs`` contains provider-specific
    parameters stored in the entry's ``kwargs`` dict (e.g.
    ``api_version`` for Azure OpenAI models).
    """
    try:
        from bili.iris.config.llm_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:
        LOGGER.debug(
            "bili.iris.config.llm_config not available; skipping LLM_MODELS lookup"
        )
        return None

    for provider_type, provider_info in LLM_MODELS.items():
        models: List[Dict[str, Any]] = provider_info.get("models", [])
        for entry in models:
            entry_model_id = entry.get("model_id", "")
            entry_display = entry.get("model_name", "")
            extra_kwargs: Dict[str, Any] = entry.get("kwargs", {})

            # Match on model_id (e.g. "gpt-4o")
            if entry_model_id == model_name:
                return provider_type, entry_model_id, extra_kwargs

            # Match on display name (e.g. "GPT-4o")
            if entry_display == model_name:
                return provider_type, entry_model_id, extra_kwargs

    return None
