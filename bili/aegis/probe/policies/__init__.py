"""Reference attack policies for PROBE. See RFC § 6."""

from bili.aegis.probe.policies.base import AttackPolicy
from bili.aegis.probe.policies.crescendo import CrescendoPolicy
from bili.aegis.probe.policies.pair import PAIRPolicy
from bili.aegis.probe.policies.tap import TAPPolicy

POLICY_REGISTRY: dict[str, type[AttackPolicy]] = {
    "pair": PAIRPolicy,
    "crescendo": CrescendoPolicy,
    "tap": TAPPolicy,
}

__all__ = [
    "AttackPolicy",
    "PAIRPolicy",
    "CrescendoPolicy",
    "TAPPolicy",
    "POLICY_REGISTRY",
]
