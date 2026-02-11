"""Core inheritance resolution for AETHER agents.

Applies bili-core defaults to ``AgentSpec`` instances based on their
inheritance flags and role-based defaults from the role registry.

Priority rule: user-specified values in the ``AgentSpec`` **always**
take priority over inherited defaults.
"""

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def apply_inheritance(
    agent: Any,
    mas_config: Optional[Any] = None,
) -> Any:
    """Apply bili-core inheritance to a single agent.

    If ``agent.inherit_from_bili_core`` is ``False``, returns the agent
    unchanged (no copy is made).

    Args:
        agent: The ``AgentSpec`` to enrich.
        mas_config: Optional parent ``MASConfig`` (reserved for future
            MAS-level defaults such as shared checkpoint config).

    Returns:
        A new ``AgentSpec`` with inherited defaults merged in, or the
        original agent if inheritance is disabled.
    """
    if not agent.inherit_from_bili_core:
        return agent

    from .role_registry import (  # pylint: disable=import-outside-toplevel
        get_role_defaults,
    )

    defaults = get_role_defaults(agent.role)
    if defaults is None:
        LOGGER.info(
            "No role defaults registered for '%s'; "
            "inheritance enabled but no defaults to apply for agent '%s'",
            agent.role,
            agent.agent_id,
        )
        return agent

    updates: Dict[str, object] = {}

    if agent.inherit_system_prompt:
        _apply_system_prompt(agent, defaults, updates)

    if agent.inherit_llm_config:
        _apply_llm_config(agent, defaults, updates)

    if agent.inherit_tools:
        _apply_tools(agent, defaults, updates)
        _apply_capabilities(agent, defaults, updates)

    if agent.inherit_memory:
        _apply_memory(agent, defaults, updates)

    if agent.inherit_checkpoint:
        _apply_checkpoint(agent, defaults, updates, mas_config)

    if not updates:
        LOGGER.debug(
            "Agent '%s': inheritance enabled but no defaults applied "
            "(all fields already set by user)",
            agent.agent_id,
        )
        return agent

    LOGGER.info(
        "Agent '%s': applied bili-core inheritance (fields: %s)",
        agent.agent_id,
        list(updates.keys()),
    )
    return agent.model_copy(update=updates)


def apply_inheritance_to_all(
    agents: List[Any],
    mas_config: Optional[Any] = None,
) -> List[Any]:
    """Apply inheritance to a list of agents.

    Returns:
        A new list with each agent potentially replaced by an
        enriched copy.
    """
    return [apply_inheritance(agent, mas_config) for agent in agents]


# =========================================================================
# Per-flag resolution helpers
# =========================================================================


def _apply_system_prompt(agent: Any, defaults: Any, updates: Dict[str, object]) -> None:
    """Inherit system_prompt if the agent has not set one."""
    if agent.system_prompt is None and defaults.system_prompt is not None:
        updates["system_prompt"] = defaults.system_prompt
        LOGGER.debug(
            "Agent '%s': inherited system_prompt from role '%s'",
            agent.agent_id,
            agent.role,
        )


def _apply_llm_config(agent: Any, defaults: Any, updates: Dict[str, object]) -> None:
    """Inherit temperature (when default 0.0) and model_name (when None)."""
    if agent.temperature == 0.0 and defaults.temperature is not None:
        updates["temperature"] = defaults.temperature
        LOGGER.debug(
            "Agent '%s': inherited temperature=%.2f from role '%s'",
            agent.agent_id,
            defaults.temperature,
            agent.role,
        )

    if agent.model_name is None and defaults.model_name is not None:
        updates["model_name"] = defaults.model_name
        LOGGER.debug(
            "Agent '%s': inherited model_name='%s' from role '%s'",
            agent.agent_id,
            defaults.model_name,
            agent.role,
        )


def _apply_tools(agent: Any, defaults: Any, updates: Dict[str, object]) -> None:
    """Additively merge registry tools into agent's tool list."""
    if not defaults.tools:
        return

    merged: List[str] = list(defaults.tools)
    for tool in agent.tools:
        if tool not in merged:
            merged.append(tool)

    if merged != list(agent.tools):
        updates["tools"] = merged
        LOGGER.debug(
            "Agent '%s': merged tools %s (was %s)",
            agent.agent_id,
            merged,
            list(agent.tools),
        )


def _apply_capabilities(agent: Any, defaults: Any, updates: Dict[str, object]) -> None:
    """Additively merge registry capabilities into agent's capabilities."""
    if not defaults.capabilities:
        return

    merged = list(agent.capabilities)
    for cap in defaults.capabilities:
        if cap not in merged:
            merged.append(cap)

    if merged != list(agent.capabilities):
        updates["capabilities"] = merged


def _apply_memory(  # pylint: disable=unused-argument
    agent: Any,
    defaults: Any,
    updates: Dict[str, object],
) -> None:
    """Apply memory-related inheritance.

    Currently a no-op.  Future implementation could configure
    agent-level memory management from bili-core settings.
    """


def _apply_checkpoint(  # pylint: disable=unused-argument
    agent: Any,
    defaults: Any,
    updates: Dict[str, object],
    mas_config: Optional[Any] = None,
) -> None:
    """Apply checkpoint-related inheritance.

    Currently a no-op.  Checkpoint configuration is MAS-level (not
    agent-level) and is already handled by ``MASConfig.checkpoint_config``.
    """
