"""Tests for InMemoryAuthProvider.

Tests the in-memory authentication provider:
- User creation and deletion
- Sign in with credentials
- Account info retrieval
- Email verification
- JWT token creation and verification
"""

import os
import time

import jwt
import pytest

from bili.auth.providers.auth.in_memory_auth_provider import InMemoryAuthProvider


def _make_provider():
    """Create a fresh InMemoryAuthProvider with JWT secret set."""
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    return InMemoryAuthProvider()


def _make_user(provider):
    """Create a test user on the given provider and return (provider, user)."""
    user = provider.create_user(email="test@example.com", password="password123")
    return provider, user


class TestCreateUser:
    """Tests for initialization and create_user method."""

    def test_initializes_empty_users_dict(self):
        """Test that provider initializes with empty users dict."""
        provider = _make_provider()
        assert isinstance(provider.users, dict)
        assert len(provider.users) == 0

    def test_creates_user_with_email_and_password(self):
        """Test creating a user with email and password."""
        provider = _make_provider()
        user = provider.create_user(email="newuser@example.com", password="mypassword")

        assert user["email"] == "newuser@example.com"
        assert user["password"] == "mypassword"
        assert "uid" in user
        assert "token" in user
        assert user["emailVerified"] is False

    def test_creates_unique_uid_for_each_user(self):
        """Test that each user gets a unique UID."""
        provider = _make_provider()
        user1 = provider.create_user(email="user1@example.com", password="pass1")
        user2 = provider.create_user(email="user2@example.com", password="pass2")

        assert user1["uid"] != user2["uid"]

    def test_stores_user_by_both_uid_and_email(self):
        """Test that user is stored in dict by both UID and email."""
        provider = _make_provider()
        user = provider.create_user(email="test@example.com", password="password")

        assert provider.users[user["uid"]] == user
        assert provider.users["test@example.com"] == user

    def test_duplicate_email_raises_error(self):
        """Test that creating user with existing email raises ValueError."""
        provider = _make_provider()
        provider.create_user(email="test@example.com", password="password")

        with pytest.raises(ValueError, match="Email already exists"):
            provider.create_user(email="test@example.com", password="different")

    def test_creates_jwt_token_for_user(self):
        """Test that user creation includes JWT token."""
        provider = _make_provider()
        user = provider.create_user(email="test@example.com", password="password")

        assert "token" in user
        assert isinstance(user["token"], str)
        assert len(user["token"]) > 0


class TestSignIn:
    """Tests for sign_in method."""

    def test_sign_in_with_valid_credentials(self):
        """Test signing in with correct email and password."""
        provider, user = _make_user(_make_provider())
        result = provider.sign_in(email="test@example.com", password="password123")

        assert result["email"] == "test@example.com"
        assert result["uid"] == user["uid"]

    def test_sign_in_with_invalid_email(self):
        """Test that sign in with non-existent email raises error."""
        provider = _make_provider()
        with pytest.raises(ValueError, match="Invalid credentials"):
            provider.sign_in(email="nonexistent@example.com", password="password")

    def test_sign_in_with_wrong_password(self):
        """Test that sign in with wrong password raises error."""
        provider, _ = _make_user(_make_provider())
        with pytest.raises(ValueError, match="Invalid credentials"):
            provider.sign_in(email="test@example.com", password="wrongpassword")

    def test_sign_in_returns_token(self):
        """Test that sign_in result contains a token field."""
        provider, _ = _make_user(_make_provider())
        result = provider.sign_in(email="test@example.com", password="password123")
        assert "token" in result
        assert isinstance(result["token"], str)
        assert result["token"]

    def test_sign_in_response_has_uid(self):
        """Test that sign_in result contains uid field."""
        provider, user = _make_user(_make_provider())
        result = provider.sign_in(email="test@example.com", password="password123")
        assert "uid" in result
        assert result["uid"] == user["uid"]

    def test_sign_in_response_has_email(self):
        """Test that sign_in result contains email field."""
        provider, _ = _make_user(_make_provider())
        result = provider.sign_in(email="test@example.com", password="password123")
        assert result["email"] == "test@example.com"


class TestGetAccountInfo:
    """Tests for get_account_info method."""

    def test_get_account_info_with_valid_uid(self):
        """Test retrieving account info with valid UID."""
        provider, user = _make_user(_make_provider())
        info = provider.get_account_info(uid=user["uid"])

        assert info["email"] == "test@example.com"
        assert info["uid"] == user["uid"]

    def test_get_account_info_with_invalid_uid(self):
        """Test that invalid UID raises ValueError."""
        provider = _make_provider()
        with pytest.raises(ValueError, match="Invalid id token"):
            provider.get_account_info(uid="nonexistent-uid")


class TestEmailAndPasswordReset:
    """Tests for send_email_verification and send_password_reset_email."""

    def test_marks_email_as_verified(self):
        """Test that email verification marks email as verified."""
        provider, user = _make_user(_make_provider())
        assert user["emailVerified"] is False

        provider.send_email_verification(auth_details=user)

        assert provider.users[user["uid"]]["emailVerified"] is True
        assert provider.users[user["email"]]["emailVerified"] is True

    def test_password_reset_raises_not_implemented(self):
        """Test that password reset raises NotImplementedError."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="Password reset email not"):
            provider.send_password_reset_email(uid="any-uid")


class TestDeleteUser:
    """Tests for delete_user method."""

    def test_deletes_user_by_uid(self):
        """Test deleting user removes them from users dict."""
        provider, user = _make_user(_make_provider())
        uid = user["uid"]
        email = user["email"]

        assert uid in provider.users
        assert email in provider.users

        provider.delete_user(uid=uid)

        assert uid not in provider.users
        assert email not in provider.users

    def test_delete_nonexistent_user_no_error(self):
        """Test that deleting non-existent user doesn't raise error."""
        provider = _make_provider()
        provider.delete_user(uid="nonexistent-uid")


class TestCreateJwtToken:
    """Tests for create_jwt_token method."""

    def test_creates_valid_jwt_token(self):
        """Test that JWT token is created and can be decoded."""
        provider = _make_provider()
        payload = {"uid": "test-uid", "email": "test@example.com"}
        result = provider.create_jwt_token(payload)

        assert "token" in result
        secret = os.getenv("JWT_SECRET_KEY")
        decoded = jwt.decode(result["token"], secret, algorithms=["HS256"])

        assert decoded["uid"] == "test-uid"
        assert decoded["email"] == "test@example.com"

    def test_token_includes_expiration(self):
        """Test that JWT token includes expiration time."""
        provider = _make_provider()
        payload = {"uid": "test-uid"}
        result = provider.create_jwt_token(payload)

        secret = os.getenv("JWT_SECRET_KEY")
        decoded = jwt.decode(result["token"], secret, algorithms=["HS256"])

        assert "exp" in decoded
        assert decoded["exp"] > time.time()


class TestVerifyJwtToken:
    """Tests for verify_jwt_token method."""

    def test_verifies_valid_token(self):
        """Test that valid token can be verified and decoded."""
        provider = _make_provider()
        payload = {"uid": "test-uid", "email": "test@example.com"}
        token_result = provider.create_jwt_token(payload)

        decoded = provider.verify_jwt_token(token_result["token"])

        assert decoded["uid"] == "test-uid"
        assert decoded["email"] == "test@example.com"

    def test_rejects_tampered_token(self):
        """Test that tampered token raises ValueError."""
        provider = _make_provider()
        payload = {"uid": "test-uid"}
        token_result = provider.create_jwt_token(payload)

        tampered = token_result["token"] + "corrupted"

        with pytest.raises(ValueError, match="Invalid token"):
            provider.verify_jwt_token(tampered)

    def test_rejects_token_with_wrong_secret(self):
        """Test that token signed with different secret is rejected."""
        provider = _make_provider()
        payload = {"uid": "test-uid"}
        wrong_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")

        with pytest.raises(ValueError, match="Invalid token"):
            provider.verify_jwt_token(wrong_token)


class TestRefreshAndLifecycle:
    """Tests for refresh_jwt_token and end-to-end flows."""

    def test_refresh_raises_not_implemented(self):
        """Test that JWT refresh raises NotImplementedError."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="JWT refresh not implemented"):
            provider.refresh_jwt_token(refresh_token="any-token")

    def test_complete_user_lifecycle(self):
        """Test complete user lifecycle: create, sign in, verify, delete."""
        provider = _make_provider()
        user = provider.create_user(email="lifecycle@example.com", password="testpass")
        assert user["emailVerified"] is False

        signed_in = provider.sign_in(email="lifecycle@example.com", password="testpass")
        assert signed_in["uid"] == user["uid"]

        provider.send_email_verification(auth_details=user)
        verified_user = provider.get_account_info(uid=user["uid"])
        assert verified_user["emailVerified"] is True

        provider.delete_user(uid=user["uid"])
        with pytest.raises(ValueError):
            provider.get_account_info(uid=user["uid"])

    def test_multiple_users_isolated(self):
        """Test that multiple users don't interfere with each other."""
        provider = _make_provider()
        user1 = provider.create_user(email="user1@example.com", password="pass1")
        user2 = provider.create_user(email="user2@example.com", password="pass2")

        auth1 = provider.sign_in(email="user1@example.com", password="pass1")
        auth2 = provider.sign_in(email="user2@example.com", password="pass2")

        assert auth1["uid"] == user1["uid"]
        assert auth2["uid"] == user2["uid"]
        assert auth1["uid"] != auth2["uid"]
