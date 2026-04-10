"""Tests for bili.streamlit_ui.ui.ui_auth_manager.UIAuthManager.

All Streamlit session_state and provider interactions are mocked.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest

from bili.streamlit_ui.tests.conftest import FakeSessionState


@pytest.fixture()
def ui_mgr():
    """Yield (UIAuthManager, mock_st) with st patched."""
    with patch("bili.streamlit_ui.ui.ui_auth_manager.st") as mock_st:
        mock_st.session_state = FakeSessionState()
        mock_st.rerun = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.error = MagicMock()

        from bili.streamlit_ui.ui.ui_auth_manager import (  # pylint: disable=import-outside-toplevel
            UIAuthManager,
        )

        with patch.object(UIAuthManager, "__init__", return_value=None):
            mgr = UIAuthManager.__new__(UIAuthManager)
            mgr.auth_provider = MagicMock()
            mgr.profile_provider = MagicMock()
            mgr.role_provider = MagicMock()

        yield mgr, mock_st


# =========================================================================
# sign_out
# =========================================================================


class TestSignOut:  # pylint: disable=too-few-public-methods
    """Tests for UIAuthManager.sign_out."""

    def test_clears_and_sets_success(self, ui_mgr):
        """sign_out clears session and sets success message."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["some_key"] = "value"

        mgr.sign_out()

        # After clear + set, only auth_success should remain
        assert mock_st.session_state.auth_success == "You have successfully signed out"
        assert "some_key" not in mock_st.session_state


# =========================================================================
# reset_password
# =========================================================================


class TestResetPassword:
    """Tests for UIAuthManager.reset_password."""

    def test_success_sets_session_message(self, ui_mgr):
        """Successful reset sets auth_success and reruns."""
        mgr, mock_st = ui_mgr

        mgr.reset_password("user@example.com")

        mgr.auth_provider.send_password_reset_email.assert_called_once_with(
            "user@example.com"
        )
        assert "reset link" in mock_st.session_state.auth_success
        mock_st.rerun.assert_called_once()

    def test_error_sets_warning(self, ui_mgr):
        """Exception during reset sets auth_warning."""
        mgr, mock_st = ui_mgr
        mgr.auth_provider.send_password_reset_email.side_effect = ValueError(
            "No such user"
        )

        mgr.reset_password("bad@example.com")

        assert "Error" in mock_st.session_state.auth_warning
        mock_st.rerun.assert_called_once()


# =========================================================================
# delete_account
# =========================================================================


class TestDeleteAccount:
    """Tests for UIAuthManager.delete_account."""

    def test_success_clears_session(self, ui_mgr):
        """Successful deletion clears session and sets message."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["user_info"] = {"email": "u@test.com"}
        mgr.auth_provider.sign_in.return_value = {"uid": "uid1"}

        mgr.delete_account("password123")

        mgr.auth_provider.delete_account.assert_called_once_with("uid1")
        assert "deleted" in mock_st.session_state.auth_success

    def test_error_sets_warning(self, ui_mgr):
        """Exception during deletion sets auth_warning."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["user_info"] = {"email": "u@test.com"}
        mgr.auth_provider.sign_in.side_effect = ValueError("Wrong")

        mgr.delete_account("wrong")

        assert "Error" in mock_st.session_state.auth_warning
        mock_st.rerun.assert_called_once()


# =========================================================================
# attempt_reauthentication
# =========================================================================


class TestAttemptReauthentication:
    """Tests for UIAuthManager.attempt_reauthentication."""

    def test_restores_session_from_auth_info(self, ui_mgr):
        """Populates session from auth_info when user_info absent."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["auth_info"] = {
            "uid": "uid1",
            "token": "tok1",
        }

        mgr.auth_provider.get_account_info.return_value = {"email": "u@test.com"}
        mgr.role_provider.get_user_role.return_value = "researcher"

        mgr.attempt_reauthentication()

        assert mock_st.session_state.user_info == {"email": "u@test.com"}
        assert mock_st.session_state.role == "researcher"

    def test_clears_session_on_failure(self, ui_mgr):
        """Clears session when reauthentication fails."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["auth_info"] = {
            "uid": "uid1",
            "token": "tok1",
        }

        mgr.auth_provider.get_account_info.side_effect = Exception("expired")

        mgr.attempt_reauthentication()

        assert "user_info" not in mock_st.session_state

    def test_no_op_when_user_info_exists(self, ui_mgr):
        """Does nothing when user_info already in session."""
        mgr, mock_st = ui_mgr
        mock_st.session_state["auth_info"] = {
            "uid": "uid1",
            "token": "tok1",
        }
        mock_st.session_state["user_info"] = {"email": "u@test.com"}

        mgr.attempt_reauthentication()

        mgr.auth_provider.get_account_info.assert_not_called()

    def test_no_op_when_no_auth_info(self, ui_mgr):
        """Does nothing when auth_info is not in session."""
        mgr, _ = ui_mgr

        mgr.attempt_reauthentication()

        mgr.auth_provider.get_account_info.assert_not_called()


# =========================================================================
# create_account
# =========================================================================


class TestCreateAccount:
    """Tests for UIAuthManager.create_account."""

    def test_new_user_creates_and_verifies(self, ui_mgr):
        """New user triggers create_user and email verification."""
        mgr, mock_st = ui_mgr
        mgr.auth_provider.create_user.return_value = {
            "uid": "uid1",
            "token": "tok1",
        }

        mgr.create_account("u@test.com", "pass", "John", "Doe", False)

        mgr.auth_provider.create_user.assert_called_once()
        mgr.auth_provider.send_email_verification.assert_called_once()
        mgr.profile_provider.create_user_profile.assert_called_once()
        mock_st.rerun.assert_called_once()

    def test_existing_user_signs_in(self, ui_mgr):
        """Existing user triggers sign_in instead of create_user."""
        mgr, _ = ui_mgr
        mgr.auth_provider.sign_in.return_value = {
            "uid": "uid1",
            "token": "tok1",
        }

        mgr.create_account("u@test.com", "pass", "John", "Doe", True)

        mgr.auth_provider.sign_in.assert_called_once()
        mgr.auth_provider.create_user.assert_not_called()

    def test_error_sets_warning(self, ui_mgr):
        """Exception sets auth_warning and reruns."""
        mgr, mock_st = ui_mgr
        mgr.auth_provider.create_user.side_effect = ValueError("exists")

        mgr.create_account("u@test.com", "pass", "John", "Doe", False)

        assert "error" in mock_st.session_state.auth_warning.lower()
        mock_st.rerun.assert_called_once()
