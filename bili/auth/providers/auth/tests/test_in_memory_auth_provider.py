"""Tests for InMemoryAuthProvider.

Tests the in-memory authentication provider:
- User creation and deletion
- Sign in with credentials
- Account info retrieval
- Email verification
- JWT token creation and verification
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,too-few-public-methods,unused-argument

import os
import time

import jwt
import pytest

from bili.auth.providers.auth.in_memory_auth_provider import InMemoryAuthProvider


@pytest.fixture
def auth_provider():
    """Create a fresh InMemoryAuthProvider for each test."""
    # Set JWT secret for testing
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    return InMemoryAuthProvider()


@pytest.fixture
def test_user(auth_provider):
    """Create a test user."""
    return auth_provider.create_user(email="test@example.com", password="password123")


class TestInit:
    """Tests for InMemoryAuthProvider initialization."""

    def test_initializes_empty_users_dict(self, auth_provider):
        """Test that provider initializes with empty users dict."""
        assert isinstance(auth_provider.users, dict)
        assert len(auth_provider.users) == 0


class TestCreateUser:
    """Tests for create_user method."""

    def test_creates_user_with_email_and_password(self, auth_provider):
        """Test creating a user with email and password."""
        user = auth_provider.create_user(
            email="newuser@example.com", password="mypassword"
        )

        assert user["email"] == "newuser@example.com"
        assert user["password"] == "mypassword"
        assert "uid" in user
        assert "token" in user
        assert user["emailVerified"] is False

    def test_creates_unique_uid_for_each_user(self, auth_provider):
        """Test that each user gets a unique UID."""
        user1 = auth_provider.create_user(email="user1@example.com", password="pass1")
        user2 = auth_provider.create_user(email="user2@example.com", password="pass2")

        assert user1["uid"] != user2["uid"]

    def test_stores_user_by_both_uid_and_email(self, auth_provider):
        """Test that user is stored in dict by both UID and email."""
        user = auth_provider.create_user(email="test@example.com", password="password")

        # Should be accessible by both uid and email
        assert auth_provider.users[user["uid"]] == user
        assert auth_provider.users["test@example.com"] == user

    def test_duplicate_email_raises_error(self, auth_provider):
        """Test that creating user with existing email raises ValueError."""
        auth_provider.create_user(email="test@example.com", password="password")

        with pytest.raises(ValueError, match="Email already exists"):
            auth_provider.create_user(email="test@example.com", password="different")

    def test_creates_jwt_token_for_user(self, auth_provider):
        """Test that user creation includes JWT token."""
        user = auth_provider.create_user(email="test@example.com", password="password")

        assert "token" in user
        assert isinstance(user["token"], str)
        assert len(user["token"]) > 0


class TestSignIn:
    """Tests for sign_in method."""

    def test_sign_in_with_valid_credentials(self, auth_provider, test_user):
        """Test signing in with correct email and password."""
        result = auth_provider.sign_in(email="test@example.com", password="password123")

        assert result["email"] == "test@example.com"
        assert result["uid"] == test_user["uid"]

    def test_sign_in_with_invalid_email(self, auth_provider):
        """Test that sign in with non-existent email raises error."""
        with pytest.raises(ValueError, match="Invalid credentials"):
            auth_provider.sign_in(email="nonexistent@example.com", password="password")

    def test_sign_in_with_wrong_password(self, auth_provider, test_user):
        """Test that sign in with wrong password raises error."""
        with pytest.raises(ValueError, match="Invalid credentials"):
            auth_provider.sign_in(email="test@example.com", password="wrongpassword")


class TestGetAccountInfo:
    """Tests for get_account_info method."""

    def test_get_account_info_with_valid_uid(self, auth_provider, test_user):
        """Test retrieving account info with valid UID."""
        info = auth_provider.get_account_info(uid=test_user["uid"])

        assert info["email"] == "test@example.com"
        assert info["uid"] == test_user["uid"]

    def test_get_account_info_with_invalid_uid(self, auth_provider):
        """Test that invalid UID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid id token"):
            auth_provider.get_account_info(uid="nonexistent-uid")


class TestSendEmailVerification:
    """Tests for send_email_verification method."""

    def test_marks_email_as_verified(self, auth_provider, test_user):
        """Test that email verification marks email as verified."""
        # Initially not verified
        assert test_user["emailVerified"] is False

        # Verify email
        auth_provider.send_email_verification(auth_details=test_user)

        # Should be verified in both uid and email lookups
        assert auth_provider.users[test_user["uid"]]["emailVerified"] is True
        assert auth_provider.users[test_user["email"]]["emailVerified"] is True


class TestSendPasswordResetEmail:
    """Tests for send_password_reset_email method."""

    def test_raises_not_implemented_error(self, auth_provider):
        """Test that password reset raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Password reset email not"):
            auth_provider.send_password_reset_email(uid="any-uid")


class TestDeleteUser:
    """Tests for delete_user method."""

    def test_deletes_user_by_uid(self, auth_provider, test_user):
        """Test deleting user removes them from users dict."""
        uid = test_user["uid"]
        email = test_user["email"]

        # User exists
        assert uid in auth_provider.users
        assert email in auth_provider.users

        # Delete user
        auth_provider.delete_user(uid=uid)

        # User should be removed (both uid and email keys)
        assert uid not in auth_provider.users
        assert email not in auth_provider.users

    def test_delete_nonexistent_user_no_error(self, auth_provider):
        """Test that deleting non-existent user doesn't raise error."""
        # Should not raise
        auth_provider.delete_user(uid="nonexistent-uid")


class TestCreateJwtToken:
    """Tests for create_jwt_token method."""

    def test_creates_valid_jwt_token(self, auth_provider):
        """Test that JWT token is created and can be decoded."""
        payload = {"uid": "test-uid", "email": "test@example.com"}
        result = auth_provider.create_jwt_token(payload)

        assert "token" in result
        # Decode to verify it's valid
        secret = os.getenv("JWT_SECRET_KEY")
        decoded = jwt.decode(result["token"], secret, algorithms=["HS256"])

        assert decoded["uid"] == "test-uid"
        assert decoded["email"] == "test@example.com"

    def test_token_includes_expiration(self, auth_provider):
        """Test that JWT token includes expiration time."""
        payload = {"uid": "test-uid"}
        result = auth_provider.create_jwt_token(payload)

        secret = os.getenv("JWT_SECRET_KEY")
        decoded = jwt.decode(result["token"], secret, algorithms=["HS256"])

        assert "exp" in decoded
        # Expiration should be in the future
        assert decoded["exp"] > time.time()


class TestVerifyJwtToken:
    """Tests for verify_jwt_token method."""

    def test_verifies_valid_token(self, auth_provider):
        """Test that valid token can be verified and decoded."""
        payload = {"uid": "test-uid", "email": "test@example.com"}
        token_result = auth_provider.create_jwt_token(payload)

        # Verify the token
        decoded = auth_provider.verify_jwt_token(token_result["token"])

        assert decoded["uid"] == "test-uid"
        assert decoded["email"] == "test@example.com"

    def test_rejects_tampered_token(self, auth_provider):
        """Test that tampered token raises ValueError."""
        payload = {"uid": "test-uid"}
        token_result = auth_provider.create_jwt_token(payload)

        # Tamper with token
        tampered = token_result["token"] + "corrupted"

        with pytest.raises(ValueError, match="Invalid token"):
            auth_provider.verify_jwt_token(tampered)

    def test_rejects_token_with_wrong_secret(self, auth_provider):
        """Test that token signed with different secret is rejected."""
        payload = {"uid": "test-uid"}
        # Create token with different secret
        wrong_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")

        with pytest.raises(ValueError, match="Invalid token"):
            auth_provider.verify_jwt_token(wrong_token)


class TestRefreshJwtToken:
    """Tests for refresh_jwt_token method."""

    def test_raises_not_implemented_error(self, auth_provider):
        """Test that JWT refresh raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="JWT refresh not implemented"):
            auth_provider.refresh_jwt_token(refresh_token="any-token")


class TestEndToEndUserFlow:
    """End-to-end tests for typical user workflows."""

    def test_complete_user_lifecycle(self, auth_provider):
        """Test complete user lifecycle: create, sign in, verify, delete."""
        # Create user
        user = auth_provider.create_user(
            email="lifecycle@example.com", password="testpass"
        )
        assert user["emailVerified"] is False

        # Sign in
        signed_in = auth_provider.sign_in(
            email="lifecycle@example.com", password="testpass"
        )
        assert signed_in["uid"] == user["uid"]

        # Verify email
        auth_provider.send_email_verification(auth_details=user)
        verified_user = auth_provider.get_account_info(uid=user["uid"])
        assert verified_user["emailVerified"] is True

        # Delete user
        auth_provider.delete_user(uid=user["uid"])
        with pytest.raises(ValueError):
            auth_provider.get_account_info(uid=user["uid"])

    def test_multiple_users_isolated(self, auth_provider):
        """Test that multiple users don't interfere with each other."""
        user1 = auth_provider.create_user(email="user1@example.com", password="pass1")
        user2 = auth_provider.create_user(email="user2@example.com", password="pass2")

        # Each user can sign in independently
        auth1 = auth_provider.sign_in(email="user1@example.com", password="pass1")
        auth2 = auth_provider.sign_in(email="user2@example.com", password="pass2")

        assert auth1["uid"] == user1["uid"]
        assert auth2["uid"] == user2["uid"]
        assert auth1["uid"] != auth2["uid"]
