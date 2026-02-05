"""AETHER-to-bili-core integration layer.

Provides role-based default resolution and inheritance application
for ``AgentSpec`` instances that opt into bili-core features via
``inherit_from_bili_core=True``.

All bili-core imports are lazy -- this package loads without
heavy dependencies (torch, firebase_admin, provider SDKs).
"""

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
    "ROLE_DEFAULTS",
    "RoleDefaults",
    "get_role_defaults",
    "register_role_defaults",
]
