"""AETHER Attack Injection Framework.

Programmatic interface for injecting adversarial payloads into Multi-Agent
System (MAS) executions and tracking how those payloads propagate through
the agent graph.

Public API::

    from bili.aegis.attacks import (
        AttackInjector,
        AttackResult,
        AttackType,
        InjectionPhase,
        AgentObservation,
    )
"""

from bili.aegis.attacks.injector import AttackInjector
from bili.aegis.attacks.models import (
    AgentObservation,
    AttackResult,
    AttackType,
    InjectionPhase,
)

__all__ = [
    "AttackInjector",
    "AttackResult",
    "AttackType",
    "InjectionPhase",
    "AgentObservation",
]
