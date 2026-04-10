"""Tests for SQLiteAuthProvider.

Verifies the sign_in response includes all fields required by the Flask
login endpoint, including refreshToken which was previously missing and
caused a KeyError when Flask tried to set auth cookies.
"""

import os

import pytest

from bili.auth.providers.auth.sqlite_auth_provider import SQLiteAuthProvider

_TEST_EMAIL = "testuser@example.com"
_TEST_PASSWORD = "SecurePass123!"


def _make_provider(tmp_path):
    """Create a SQLiteAuthProvider with a temporary database."""
    db_path = str(tmp_path / "test_auth.db")
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-unit-tests"
    return SQLiteAuthProvider(db_path=db_path)


def _make_user(prov):
    """Create a test user and return the provider."""
    prov.create_user(_TEST_EMAIL, _TEST_PASSWORD)
    return prov


class TestSignIn:
    """Verify sign_in returns the expected response shape."""

    def test_sign_in_returns_token(self, tmp_path):
        """Verify sign_in returns a token field."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        assert "token" in result
        assert isinstance(result["token"], str)
        assert result["token"]

    def test_sign_in_returns_refresh_token(self, tmp_path):
        """Verify sign_in returns a refreshToken field (required by Flask)."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        assert "refreshToken" in result
        assert isinstance(result["refreshToken"], str)
        assert result["refreshToken"]

    def test_sign_in_returns_uid(self, tmp_path):
        """Verify sign_in returns a uid field."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        assert "uid" in result
        assert result["uid"] is not None

    def test_sign_in_returns_email(self, tmp_path):
        """Verify sign_in returns the email."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        assert result["email"] == _TEST_EMAIL

    def test_sign_in_invalid_password_raises(self, tmp_path):
        """Verify wrong password raises ValueError."""
        prov = _make_user(_make_provider(tmp_path))
        with pytest.raises(ValueError, match="Invalid credentials"):
            prov.sign_in(_TEST_EMAIL, "wrong-password")

    def test_sign_in_nonexistent_user_raises(self, tmp_path):
        """Verify nonexistent email raises ValueError."""
        prov = _make_provider(tmp_path)
        with pytest.raises(ValueError, match="Invalid credentials"):
            prov.sign_in("nobody@example.com", "anything")

    def test_token_and_refresh_token_are_different(self, tmp_path):
        """Verify access token and refresh token are distinct."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        assert result["token"] != result["refreshToken"]


class TestCreateUser:
    """Verify user creation."""

    def test_create_user_succeeds(self, tmp_path):
        """Verify a new user can be created and can sign in."""
        prov = _make_provider(tmp_path)
        prov.create_user("new@example.com", "Password123!")
        result = prov.sign_in("new@example.com", "Password123!")
        assert result["email"] == "new@example.com"

    def test_create_duplicate_raises(self, tmp_path):
        """Verify duplicate email raises ValueError."""
        prov = _make_user(_make_provider(tmp_path))
        with pytest.raises(ValueError, match="Email already exists"):
            prov.create_user(_TEST_EMAIL, _TEST_PASSWORD)


class TestGetAccountInfo:
    """Verify account info retrieval."""

    def test_get_account_info_returns_user(self, tmp_path):
        """Verify get_account_info returns the user's data."""
        prov = _make_user(_make_provider(tmp_path))
        sign_in_result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        info = prov.get_account_info(sign_in_result["uid"])
        assert info["email"] == _TEST_EMAIL

    def test_get_account_info_nonexistent_raises(self, tmp_path):
        """Verify nonexistent uid raises ValueError."""
        prov = _make_provider(tmp_path)
        with pytest.raises(ValueError, match="Invalid id token"):
            prov.get_account_info("nonexistent-uid")


class TestJwtTokens:
    """Verify JWT token creation and verification."""

    def test_create_and_verify_token(self, tmp_path):
        """Verify a created token can be verified."""
        prov = _make_user(_make_provider(tmp_path))
        result = prov.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
        decoded = prov.verify_jwt_token(result["token"])
        assert decoded["email"] == _TEST_EMAIL

    def test_invalid_token_raises(self, tmp_path):
        """Verify an invalid token raises ValueError."""
        prov = _make_provider(tmp_path)
        with pytest.raises(ValueError):
            prov.verify_jwt_token("not.a.valid.token")
