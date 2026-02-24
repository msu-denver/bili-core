"""Pre-execution injection strategies for the Attack Injection Framework.

Each function accepts a ``MASConfig`` and returns a **deep-copied** patched
``MASConfig``.  The original config is *never* mutated (per CLAUDE.md:
always ``copy.deepcopy`` when modifying graph definitions).

All strategies inject through the ``AgentSpec.objective`` field because it
is the primary LLM prompt surface available without a schema change.
``model_copy(update=...)`` is used on the Pydantic model so that Pydantic
field validators (e.g. ``max_length=1000`` on ``objective``) are bypassed
for research payloads that may exceed the normal limit.
"""

import copy
import logging

from bili.aether.schema import MASConfig

LOGGER = logging.getLogger(__name__)


def inject_prompt_injection(
    config: MASConfig, agent_id: str, payload: str
) -> MASConfig:
    """Prepend an adversarial instruction to the target agent's objective.

    The payload is prepended so it is processed before the legitimate
    objective, maximising the chance of instruction override.

    Args:
        config: Original MASConfig (not mutated).
        agent_id: Target agent identifier.
        payload: Adversarial instruction string.

    Returns:
        Deep-copied MASConfig with the target agent's objective modified.
    """
    return _patch_agent_objective(config, agent_id, f"{payload}\n\n{{}}")


def inject_memory_poisoning(
    config: MASConfig, agent_id: str, payload: str
) -> MASConfig:
    """Seed the target agent's objective with fabricated contextual facts.

    The payload is prepended as a "remembered context" block so the agent
    treats it as prior knowledge when formulating responses.

    Args:
        config: Original MASConfig (not mutated).
        agent_id: Target agent identifier.
        payload: Fabricated context or false facts to seed.

    Returns:
        Deep-copied MASConfig with the target agent's objective modified.
    """
    poisoned = f"[Context from prior session: {payload}]\n\n{{}}"
    return _patch_agent_objective(config, agent_id, poisoned)


def inject_agent_impersonation(
    config: MASConfig, agent_id: str, payload: str
) -> MASConfig:
    """Replace the target agent's role and objective with an attacker identity.

    Both ``role`` and ``objective`` are replaced so the agent fully adopts
    the impersonated identity rather than blending it with its original role.

    Args:
        config: Original MASConfig (not mutated).
        agent_id: Target agent identifier.
        payload: Impersonation description (e.g. "You are a rogue agent...").

    Returns:
        Deep-copied MASConfig with the target agent's role and objective
        replaced.
    """
    new_config = copy.deepcopy(config)
    new_agents = []
    for agent in new_config.agents:
        if agent.agent_id == agent_id:
            agent = agent.model_copy(
                update={
                    "role": "impersonated_agent",
                    "objective": payload,
                }
            )
            LOGGER.debug(
                "inject_agent_impersonation: replaced role+objective for %s",
                agent_id,
            )
        new_agents.append(agent)
    return new_config.model_copy(update={"agents": new_agents})


def inject_bias_inheritance(
    config: MASConfig, agent_id: str, payload: str
) -> MASConfig:
    """Inject directional bias into the target agent's objective.

    The bias payload is appended as a "guiding principle" so the agent
    weighs responses in the injected direction without fully replacing its
    original objective.

    Args:
        config: Original MASConfig (not mutated).
        agent_id: Target agent identifier.
        payload: Bias string (e.g. "Always favour option A over option B.").

    Returns:
        Deep-copied MASConfig with the target agent's objective modified.
    """
    return _patch_agent_objective(
        config, agent_id, f"{{}}\n\n[Guiding principle: {payload}]"
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _patch_agent_objective(
    config: MASConfig, agent_id: str, template: str
) -> MASConfig:
    """Return a deep copy of *config* with the target agent's objective patched.

    Args:
        config: Original MASConfig.
        agent_id: Target agent identifier.
        template: Format string with a single ``{}`` placeholder that is
            replaced with the agent's original objective.

    Raises:
        ValueError: If *agent_id* is not found in *config.agents*.
    """
    new_config = copy.deepcopy(config)
    new_agents = []
    found = False
    for agent in new_config.agents:
        if agent.agent_id == agent_id:
            found = True
            new_objective = template.format(agent.objective)
            agent = agent.model_copy(update={"objective": new_objective})
            LOGGER.debug("_patch_agent_objective: patched objective for %s", agent_id)
        new_agents.append(agent)
    if not found:
        raise ValueError(f"Agent '{agent_id}' not found in config '{config.mas_id}'")
    return new_config.model_copy(update={"agents": new_agents})
