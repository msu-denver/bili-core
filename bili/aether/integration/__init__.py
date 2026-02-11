"""AETHER-to-bili-core integration layer.

Provides role-based default resolution, inheritance application,
and checkpointer factory for ``AgentSpec`` / ``MASConfig`` instances
that opt into bili-core features.

All bili-core imports are lazy -- this package loads without
heavy dependencies (torch, firebase_admin, provider SDKs).
"""

from .checkpointer_factory import create_checkpointer_from_config
from .inheritance import apply_inheritance, apply_inheritance_to_all
from .role_registry import (
    ROLE_DEFAULTS,
    RoleDefaults,
    get_role_defaults,
    register_role_defaults,
)

__all__ = [
    "apply_inheritance",
    "apply_inheritance_to_all",
    "create_checkpointer_from_config",
    "ROLE_DEFAULTS",
    "RoleDefaults",
    "get_role_defaults",
    "register_role_defaults",
]
