"""
ui_auth_manager.py
------------------

This module defines the `UIAuthManager` class, which handles user authentication workflows
including signing in, account creation, password reset, and account deletion. It interacts
with external authentication and profile providers to manage user sessions and roles, ensuring
a seamless user authentication experience within a Streamlit application.

Classes:
--------
- UIAuthManager:
    Manages user authentication processes and updates the application's session state based
    on the outcomes of various operations, including error handling and role management.

Functions:
----------
- sign_in(email: str, password: str, first_name: Optional[str] = None,
last_name: Optional[str] = None) -> None:
    Handles the process of signing in a user, verifying the user's email, and
    determining their role within the system.

- create_account(email: str, password: str, first_name: str, last_name: str,
existing_user: bool) -> None:
    Creates a user account or signs in an existing user, then sets up the user's profile
    and updates the application state accordingly.

- reset_password(email: str) -> None:
    Resets a user's password by sending a reset link to the provided email address.

- sign_out() -> None:
    Clears the session state and updates the authentication status to indicate the user
    has successfully signed out.

- delete_account(password: str) -> None:
    Deletes the authenticated user's account from the system, requiring the user to
    confirm their password.

- attempt_reauthentication() -> None:
    Attempts to reauthenticate a user with their existing ID token, fetching the user's
    account information and role
    if necessary.

Dependencies:
-------------
- time: Provides time-related functions.
- requests: Used for making HTTP requests to external services.
- streamlit: Provides the Streamlit library for managing session state and UI elements.
- bili.auth.auth_manager.AuthManager: Base class for managing authentication.
- bili.auth.providers.auth.in_memory_auth_provider.InMemoryAuthProvider: In-memory
implementation of the authentication
provider.

Usage:
------
To use the `UIAuthManager` class, instantiate it and call its methods to manage user
authentication within a Streamlit
application.

Example:
--------
from bili.streamlit_ui.ui.ui_auth_manager import UIAuthManager

auth_manager = UIAuthManager(auth_provider, role_provider, profile_provider)

# Sign in a user
auth_manager.sign_in(email="user@example.com", password="password123")

# Create a new account
auth_manager.create_account(email="newuser@example.com", password="password123",
first_name="John", last_name="Doe",
existing_user=False)

# Reset a user's password
auth_manager.reset_password(email="user@example.com")

# Sign out the current user
auth_manager.sign_out()

# Delete the current user's account
auth_manager.delete_account(password="password123")

# Attempt to reauthenticate the current user
auth_manager.attempt_reauthentication()
"""

import time

import requests
import streamlit as st

from bili.auth.auth_manager import AuthManager
from bili.auth.providers.auth.in_memory_auth_provider import InMemoryAuthProvider


class UIAuthManager(AuthManager):
    """
    Handles the user authentication workflows including signing in, account creation,
    password reset, and account deletion. This manager interacts with an external
    authentication and profile provider to manage user sessions and roles, ensuring
    a seamless user authentication experience.

    Updates the application's session state based on the outcomes of various operations,
    including error handling, role"""

    def sign_in(
        self, email: str, password: str, first_name=None, last_name=None
    ) -> None:
        """
        Handles the process of signing in a user, verifying the user's email, and determining
        their role within the system. Performs additional profile creation steps if the user
        is not found in the API.

        :param email: The email address of the user attempting to sign in.
        :type email: str
        :param password: The password of the user attempting to sign in.
        :type password: str
        :param first_name: The first name of the user, required for profile creation if needed.
            Defaults to None.
        :type first_name: Optional[str]
        :param last_name: The last name of the user, required for profile creation if needed.
            Defaults to None.
        :type last_name: Optional[str]
        :return: None
        :rtype: None
        :raises Exception: If any unexpected error occurs during the sign-in process.
        """
        try:
            auth_response = self.auth_provider.sign_in(email, password)
            uid = auth_response["uid"]
            token = auth_response["token"]

            st.session_state["uid"] = uid
            user_info = self.auth_provider.get_account_info(uid)

            if not user_info["emailVerified"]:
                self.auth_provider.send_email_verification(auth_response)
                st.session_state.auth_warning = (
                    "Please verify your email before signing in."
                )
                st.rerun()

            st.session_state.user_info = user_info

            try:
                user_role = self.role_provider.get_user_role(uid, token)
                st.session_state.role = user_role

                if user_role in ["researcher", "admin"]:
                    st.session_state.auth_success = "Signed in successfully"
                    st.rerun()
                elif user_role == "user":
                    st.warning(
                        "Your account is under review. Please contact an administrator."
                    )
                else:
                    st.error("You are not authorized to access this page.")
                    st.stop()

            except requests.exceptions.HTTPError as error:
                if error.response.status_code == 404:
                    st.warning("User not found in API. Creating profile now...")
                    if first_name is None or last_name is None:
                        st.session_state.needs_profile_creation = True
                        st.session_state.auth_warning = (
                            "Please provide your name to complete your profile."
                        )
                        st.rerun()
                    else:
                        self.profile_provider.create_user_profile(
                            uid, email, first_name, last_name, token
                        )
                        st.session_state.role = "user"
                        st.warning(
                            "Your account is under review. Please contact an administrator."
                        )
                        time.sleep(3)
                        st.session_state.needs_profile_creation = False
                        st.rerun()
                else:
                    st.warning(f"Error fetching user role: {error}")
                    raise

        except Exception as error:
            st.session_state.clear()
            st.session_state.auth_warning = f"Unexpected error: {error}"
            st.rerun()

    def create_account(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        existing_user: bool,
    ) -> None:
        """
        Creates a user account or signs in an existing user, then sets up the user's
        profile and updates the application state accordingly.

        This function uses an authentication provider to create or sign in a user
        based on the provided parameters. If a new account is created, the function
        triggers an email verification process. After successful authentication, it
        creates a user profile using a profile provider. Depending on whether the
        user is logging in for the first time or is an existing user, the function
        updates the session state variables to reflect the account's status.

        If errors occur during the process, the function gracefully handles them by
        updating the session state with a warning message and restarting the app flow.

        :param email: The email address for the user account.
        :type email: str
        :param password: The password associated with the user account.
        :type password: str
        :param first_name: The first name of the user, used for profile creation.
        :type first_name: str
        :param last_name: The last name of the user, used for profile creation.
        :type last_name: str
        :param existing_user: A Boolean flag indicating if an account already exists
            for the email.
        :type existing_user: bool
        :return: None
        """
        try:
            if not existing_user:
                auth_response = self.auth_provider.create_user(email, password)
                self.auth_provider.send_email_verification(auth_response)
            else:
                auth_response = self.auth_provider.sign_in(email, password)

            uid = auth_response["uid"]
            token = auth_response["token"]

            self.profile_provider.create_user_profile(
                uid, email, first_name, last_name, token
            )

            if not isinstance(self.auth_provider, InMemoryAuthProvider):
                st.session_state.auth_success = (
                    "Check your inbox to verify your email"
                    if not existing_user
                    else "Account created successfully and is now under review"
                )
            else:
                st.session_state.auth_success = "Account created successfully"

            st.session_state.needs_profile_creation = False
            st.rerun()

        except Exception as error:
            st.session_state.auth_warning = f"Unexpected error: {error}"
            st.rerun()

    def reset_password(self, email: str) -> None:
        """
        Reset a user's password by sending a reset link to the provided email address.

        This method interacts with an external authentication provider to send a
        password reset email to the specified address. If the operation is successful,
        a success message will be set in the session state. In case of any errors during
        the process, a warning message with error details will be set instead. The
        method ensures immediate feedback to the user by rerunning the application state.

        :param email: The email address to send the password reset link to.
        :type email: str
        :return: None
        """
        try:
            self.auth_provider.send_password_reset(email)
            st.session_state.auth_success = "Password reset link sent to your email"
            st.rerun()
        except Exception as error:
            st.session_state.auth_warning = f"Error: {error}"
            st.rerun()

    def sign_out(self) -> None:
        """
        Clears the session state and updates the authentication status to indicate the
        user has successfully signed out.

        :return: None
        """
        st.session_state.clear()
        st.session_state.auth_success = "You have successfully signed out"

    def delete_account(self, password: str) -> None:
        """
        Deletes the authenticated user's account from the system. This method requires
        the user to confirm their password. Upon successful deletion, it clears the
        user session state and sets a success message for the session. If an error
        occurs during the process, it updates the session state with the error message
        and triggers a rerun of the application.

        :param password: The password of the user's account to confirm deletion.
        :type password: str
        :return: None
        """
        try:
            uid = self.auth_provider.sign_in(
                st.session_state.user_info["email"], password
            )["uid"]
            self.auth_provider.delete_account(uid)
            st.session_state.clear()
            st.session_state.auth_success = "You have successfully deleted your account"
        except Exception as error:
            st.session_state.auth_warning = f"Error: {error}"
            st.rerun()

    def attempt_reauthentication(self):
        """
        Attempts to reauthenticate a user with their existing ID token. If "auth_info"
        exists in the Streamlit session state but "user_info" does not, the function will
        fetch the user's account information and role from respective providers. Upon
        successful reauthentication, user details, role, and an authentication success
        message are stored back in the session state. If an exception occurs during
        reatuhentication, the session state will be cleared.

        :raises Exception: When reauthentication fails due to the unavailability of valid
                           account or role data during the authentication process.

        :rtype: None
        :return: This function does not return any value; it modifies the session state
                 with either updated user information or clears it upon failure.
        """
        if "auth_info" in st.session_state and "user_info" not in st.session_state:
            try:
                auth_info = st.session_state.auth_info
                uid = auth_info["uid"]
                token = auth_info["token"]

                user_info = self.auth_provider.get_account_info(uid)
                st.session_state.user_info = user_info

                role = self.role_provider.get_user_role(uid, token)
                st.session_state.role = role
                st.session_state.auth_success = "Reauthenticated successfully"
            except Exception:
                st.session_state.clear()
