"""Tests for the role and profile provider implementations.

Covers the in-memory role and profile providers plus the base
RoleProvider.is_authorized membership check.
"""

from bili.auth.providers.profile.in_memory_profile_provider import (
    InMemoryProfileProvider,
)
from bili.auth.providers.role.in_memory_role_provider import InMemoryRoleProvider
from bili.auth.providers.role.role_provider import RoleProvider


class TestInMemoryRoleProvider:
    """The in-memory role provider grants the researcher role to everyone."""

    def test_get_user_role_returns_researcher(self):
        """Every user is assigned the researcher role."""
        provider = InMemoryRoleProvider()
        assert provider.get_user_role("any-uid", "any-token") == "researcher"

    def test_is_authorized_always_true(self):
        """Authorization always succeeds regardless of required roles."""
        provider = InMemoryRoleProvider()
        assert provider.is_authorized("uid", "token", ["admin"]) is True
        assert provider.is_authorized("uid", "token", []) is True


class TestRoleProviderBaseIsAuthorized:
    """The base RoleProvider.is_authorized checks role membership.

    InMemoryRoleProvider overrides is_authorized, so the base implementation
    is exercised here via a minimal subclass that only supplies get_user_role.
    """

    def _provider_returning(self, role):
        class _FixedRoleProvider(RoleProvider):
            def get_user_role(self, uid, token):
                return role

        return _FixedRoleProvider()

    def test_authorized_when_role_in_required(self):
        """A user whose role is in the required list is authorized."""
        provider = self._provider_returning("admin")
        assert provider.is_authorized("uid", "token", ["admin", "editor"]) is True

    def test_not_authorized_when_role_absent(self):
        """A user whose role is absent from the required list is denied."""
        provider = self._provider_returning("viewer")
        assert provider.is_authorized("uid", "token", ["admin", "editor"]) is False


class TestInMemoryProfileProvider:
    """The in-memory profile provider stores and retrieves user profiles."""

    def test_create_then_get_round_trips_profile(self):
        """A created profile is retrievable with all stored fields."""
        provider = InMemoryProfileProvider()
        provider.create_user_profile("u1", "u1@example.com", "Alice", "Garcia", "token")
        profile = provider.get_user_profile("u1", "token")
        assert profile == {
            "uid": "u1",
            "email": "u1@example.com",
            "first_name": "Alice",
            "last_name": "Garcia",
        }

    def test_get_unknown_profile_returns_none(self):
        """An unknown uid yields None."""
        provider = InMemoryProfileProvider()
        assert provider.get_user_profile("missing", "token") is None
