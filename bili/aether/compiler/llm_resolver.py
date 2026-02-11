"""LLM resolution — maps AgentSpec model names to LLM provider instances.

Resolves ``AgentSpec.model_name`` to a provider type and ``model_id``
using ``bili.config.llm_config.LLM_MODELS``, then instantiates the LLM
via ``bili.loaders.llm_loader.load_model``.

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
    # Order matters — more specific patterns must come before broader ones.
    ("gpt-", "remote_openai"),
    ("gpt4", "remote_openai"),
    ("o1-", "remote_openai"),
    ("o1", "remote_openai"),
    ("o3-", "remote_openai"),
    ("o3", "remote_openai"),
    ("gemini", "remote_google_vertex"),
    ("amazon.nova", "remote_aws_bedrock"),
    ("amazon.titan", "remote_aws_bedrock"),
    ("anthropic.claude", "remote_aws_bedrock"),  # Bedrock-hosted Claude
    ("meta.llama", "remote_aws_bedrock"),
    ("mistral", "remote_aws_bedrock"),
    ("cohere.command", "remote_aws_bedrock"),
    ("claude-", "remote_openai"),  # Direct Anthropic API (after Bedrock check)
]


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
    # --- 1 & 2: Look up in LLM_MODELS ---
    result = _lookup_in_llm_models(model_name)
    if result is not None:
        provider, model_id = result
        LOGGER.debug(
            "Resolved '%s' via LLM_MODELS → provider=%s, model_id=%s",
            model_name,
            provider,
            model_id,
        )
        return provider, model_id

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
            return ptype, model_name

    raise ValueError(
        f"Cannot resolve model '{model_name}' to a provider. "
        f"Set a recognised model_name or use bili.loaders.llm_loader directly."
    )


def resolve_provider(model_name: str) -> str:
    """Resolve a model name to a bili-core provider type string.

    Convenience wrapper around :func:`resolve_model` that returns only
    the provider type.
    """
    provider, _ = resolve_model(model_name)
    return provider


def create_llm(agent: AgentSpec) -> Any:
    """Create a LangChain chat model instance from an ``AgentSpec``.

    Lazy-imports ``bili.loaders.llm_loader.load_model`` and
    ``bili.config.llm_config.LLM_MODELS`` so the compiler module can be
    loaded without heavy dependencies.

    The function resolves the display ``model_name`` to the actual
    ``model_id`` expected by the provider, then delegates to
    ``load_model`` with the correct parameter name for each provider.

    Args:
        agent: An ``AgentSpec`` with ``model_name`` set.

    Returns:
        A LangChain-compatible chat model ready for ``.invoke()``.

    Raises:
        ValueError: If ``agent.model_name`` is ``None`` or unresolvable.
    """
    if not agent.model_name:
        raise ValueError(
            f"AgentSpec '{agent.agent_id}' has no model_name; "
            f"cannot create LLM instance."
        )

    provider, model_id = resolve_model(agent.model_name)

    # Build kwargs for load_model — each provider uses "model_name" as
    # the kwarg, but the *value* must be the provider's model_id.
    kwargs: Dict[str, Any] = {"model_name": model_id}
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

    from bili.loaders.llm_loader import (  # noqa: E402  pylint: disable=import-outside-toplevel
        load_model,
    )

    return load_model(provider, **kwargs)


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
        from bili.config.tool_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            TOOLS as TOOL_CONFIG,
        )
        from bili.loaders.tools_loader import (  # noqa: E402  pylint: disable=import-outside-toplevel
            initialize_tools,
        )
    except ImportError:
        LOGGER.warning(
            "bili.loaders.tools_loader not available; "
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


def _lookup_in_llm_models(model_name: str) -> Optional[Tuple[str, str]]:
    """Search ``LLM_MODELS`` for a matching model entry.

    Returns ``(provider_type, model_id)`` if found, ``None`` otherwise.
    """
    try:
        from bili.config.llm_config import (  # noqa: E402  pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:
        LOGGER.debug("bili.config.llm_config not available; skipping LLM_MODELS lookup")
        return None

    for provider_type, provider_info in LLM_MODELS.items():
        models: List[Dict[str, Any]] = provider_info.get("models", [])
        for entry in models:
            entry_model_id = entry.get("model_id", "")
            entry_display = entry.get("model_name", "")

            # Match on model_id (e.g. "gpt-4o")
            if entry_model_id == model_name:
                return provider_type, entry_model_id

            # Match on display name (e.g. "GPT-4o")
            if entry_display == model_name:
                return provider_type, entry_model_id

    return None
