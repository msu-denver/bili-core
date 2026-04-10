"""Tests for bili.auth.auth_manager.AuthManager.

Covers initialization with various provider names, JWT token
creation/verification, header extraction, and request token
verification.
"""

import os

import pytest

from bili.auth.auth_manager import AuthManager


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    """Ensure JWT_SECRET_KEY is set for all tests."""
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-auth-mgr"
    yield
    os.environ.pop("JWT_SECRET_KEY", None)


# ------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------


class TestAuthManagerInit:
    """AuthManager initialization with different providers."""

    def test_default_providers(self):
        """Default provider names produce a valid manager."""
        mgr = AuthManager()
        assert mgr.auth_provider is not None
        assert mgr.profile_provider is not None
        assert mgr.role_provider is not None

    def test_in_memory_providers(self):
        """In-memory providers initialise without error."""
        mgr = AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )
        assert mgr.auth_provider is not None

    def test_sqlite_providers(self):
        """Explicit sqlite provider names work."""
        mgr = AuthManager(
            auth_provider_name="sqlite",
            profile_provider_name="sqlite",
            role_provider_name="sqlite",
        )
        assert mgr.auth_provider is not None

    def test_unknown_auth_provider_raises(self):
        """Unknown auth provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown auth"):
            AuthManager(auth_provider_name="bogus")

    def test_unknown_profile_provider_raises(self):
        """Unknown profile provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            AuthManager(profile_provider_name="bogus")

    def test_unknown_role_provider_raises(self):
        """Unknown role provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown role"):
            AuthManager(role_provider_name="bogus")


# ------------------------------------------------------------------
# create_jwt_token / verify_jwt_token
# ------------------------------------------------------------------


class TestJwtTokenRoundTrip:
    """create_jwt_token and verify_jwt_token round-trip."""

    def test_create_returns_dict_with_token(self):
        """create_jwt_token returns a dict containing 'token'."""
        mgr = AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )
        result = mgr.create_jwt_token({"uid": "u1"})
        assert "token" in result
        assert isinstance(result["token"], str)

    def test_verify_valid_token(self):
        """Valid token decodes back to original payload."""
        mgr = AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )
        payload = {"uid": "u1", "email": "a@b.com"}
        token = mgr.create_jwt_token(payload)["token"]
        decoded = mgr.verify_jwt_token(token)
        assert decoded["uid"] == "u1"
        assert decoded["email"] == "a@b.com"

    def test_verify_invalid_token_raises(self):
        """Tampered token raises ValueError."""
        mgr = AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )
        with pytest.raises(ValueError):
            mgr.verify_jwt_token("not.a.real.token")


# ------------------------------------------------------------------
# extract_token_from_headers
# ------------------------------------------------------------------


class TestExtractTokenFromHeaders:
    """extract_token_from_headers parses HTTP headers."""

    def _make_mgr(self):
        """Create a manager with in-memory providers."""
        return AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )

    def test_bearer_header(self):
        """Extracts token from 'Authorization: Bearer xxx'."""
        mgr = self._make_mgr()
        headers = {"Authorization": "Bearer my-token-123"}
        assert mgr.extract_token_from_headers(headers) == ("my-token-123")

    def test_x_auth_token_fallback(self):
        """Falls back to X-Auth-Token header."""
        mgr = self._make_mgr()
        headers = {"X-Auth-Token": "fallback-token"}
        assert mgr.extract_token_from_headers(headers) == ("fallback-token")

    def test_bearer_takes_precedence(self):
        """Bearer header is preferred over X-Auth-Token."""
        mgr = self._make_mgr()
        headers = {
            "Authorization": "Bearer primary",
            "X-Auth-Token": "secondary",
        }
        assert mgr.extract_token_from_headers(headers) == ("primary")

    def test_missing_headers_raises(self):
        """No auth headers raises ValueError."""
        mgr = self._make_mgr()
        with pytest.raises(ValueError, match="No valid"):
            mgr.extract_token_from_headers({})

    def test_authorization_without_bearer_raises(self):
        """Authorization header without 'Bearer ' prefix raises."""
        mgr = self._make_mgr()
        with pytest.raises(ValueError, match="No valid"):
            mgr.extract_token_from_headers({"Authorization": "Basic abc123"})


# ------------------------------------------------------------------
# verify_request_token
# ------------------------------------------------------------------


class TestVerifyRequestToken:
    """verify_request_token extracts and verifies in one call."""

    def _make_mgr(self):
        """Create a manager with in-memory providers."""
        return AuthManager(
            auth_provider_name="in_memory",
            profile_provider_name="in_memory",
            role_provider_name="in_memory",
        )

    def test_valid_bearer_token(self):
        """Valid Bearer token is extracted and verified."""
        mgr = self._make_mgr()
        payload = {"uid": "u1", "email": "x@y.com"}
        token = mgr.create_jwt_token(payload)["token"]
        headers = {"Authorization": f"Bearer {token}"}
        decoded = mgr.verify_request_token(headers)
        assert decoded["uid"] == "u1"

    def test_missing_token_raises(self):
        """Missing token in headers raises ValueError."""
        mgr = self._make_mgr()
        with pytest.raises(ValueError):
            mgr.verify_request_token({})

    def test_invalid_token_raises(self):
        """Invalid token string raises ValueError."""
        mgr = self._make_mgr()
        headers = {"Authorization": "Bearer bad.token.here"}
        with pytest.raises(ValueError):
            mgr.verify_request_token(headers)
