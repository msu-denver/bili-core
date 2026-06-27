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

from bili.aether.schema import AgentSpec

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic provider detection (fallback when LLM_MODELS lookup fails)
# ---------------------------------------------------------------------------

_HEURISTIC_RULES = [
    # (substring_or_prefix, provider_type)
    # Order matters -- more specific patterns must come before broader ones.
    # The heuristic layer only fires when LLM_MODELS lookup returns nothing,
    # so these rules apply to non-catalog model IDs only.
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
    # Subprocess CLI provider -- matches the "cli:" sentinel prefix used
    # for CLI models in LLM_MODELS and any user-configured cli model_id.
    ("cli:", "cli"),
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
    return build_fallback_llm(primary_llm=primary_llm, fallback_chain=fallback_chain)


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


def resolve_supports_tools(model_name: str) -> bool:
    """Return whether *model_name* supports native ``bind_tools``.

    This is a backward-compatible convenience wrapper around
    :func:`resolve_tool_strategy`.  Callers that only need a boolean — e.g.
    legacy code or the Streamlit UI — can continue using this function without
    change.  Prefer :func:`resolve_tool_strategy` for new code.

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
