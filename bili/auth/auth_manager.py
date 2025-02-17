"""
Module: AuthManager

This module defines the `AuthManager` class, which manages user authentication,
profile handling, and role-based access control. It interfaces with various
providers to perform tasks such as account creation, sign-in, password reset,
session management, and role verification.

Classes:
    - AuthManager: Manages user authentication, profile handling, and role-based
      access control.

Methods:
    - __init__(auth_provider_name="default", profile_provider_name="default",
               role_provider_name="default"):
      Initializes the AuthManager with specified providers.
    - sign_in(email, password, first_name=None, last_name=None):
      Signs in a user and handles profile creation if needed.
    - create_account(email, password, first_name, last_name, existing_user):
      Creates a user account or signs in an existing user.
    - reset_password(email):
      Sends a password reset link to the provided email address.
    - sign_out():
      Signs out the user and clears the session state.
    - delete_account(password):
      Deletes the authenticated user's account.
    - attempt_reauthentication():
      Attempts to reauthenticate a user with their existing ID token.

Usage:
    This module is intended for managing user authentication, profile handling,
    and role-based access control in a Streamlit application. It provides methods
    to handle various authentication-related tasks and manage session states.

Example:
    from bili.auth.AuthManager import get_auth_manager

    auth_manager = get_auth_manager()

    # Sign in a user
    auth_manager.sign_in(email="user@example.com", password="password")

    # Create a new account
    auth_manager.create_account(
        email="newuser@example.com", password="password",
        first_name="John", last_name="Doe", existing_user=False
    )

    # Reset password
    auth_manager.reset_password(email="user@example.com")

    # Sign out
    auth_manager.sign_out()

    # Delete account
    auth_manager.delete_account(password="password")

    # Attempt reauthentication
    auth_manager.attempt_reauthentication()
"""

import time

import requests
import streamlit as st

from bili.auth.providers.auth.firebase_auth_provider import FirebaseAuthProvider
from bili.auth.providers.auth.in_memory_auth_provider import InMemoryAuthProvider
from bili.auth.providers.profile.in_memory_profile_provider import (
    InMemoryProfileProvider,
)
from bili.auth.providers.role.in_memory_role_provider import InMemoryRoleProvider
from bili.streamlit.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Providers for authentication, profile management, and role checking
# This allows for easy swapping of providers based on the environment, and for users
# to define their own providers if needed.
AUTH_PROVIDERS = {"default": InMemoryAuthProvider, "firebase": FirebaseAuthProvider}
PROFILE_PROVIDERS = {"default": InMemoryProfileProvider}
ROLE_PROVIDERS = {"default": InMemoryRoleProvider}

# Initialize the logger for the module
LOGGER = get_logger(__name__)


class AuthManager:
    """
    Manages user authentication, profile handling, and role-based access control.

    The AuthManager class provides methods for user account creation, sign-in,
    password resetting, session management, and role verification. It interfaces
    with authentication, profile, and role providers to accomplish these tasks,
    and utilizes session state to persist and manage authentication states.

    :ivar auth_provider: The authentication provider to handle user authentication tasks.
    :type auth_provider: Any

    :ivar profile_provider: The profile provider to manage user profile operations via API.
    :type profile_provider: Any

    :ivar role_provider: The role provider to retrieve and manage user roles.
    :type role_provider: Any
    """

    def __init__(
        self,
        auth_provider_name="default",
        profile_provider_name="default",
        role_provider_name="default",
    ):
        """
        AuthManager handles the initialization of authentication, profile, and role
        providers based on the provided provider names. It ensures the selected
        providers are valid and sets up the corresponding components for further use.

        :param auth_provider_name: Name of the authentication provider to be used.
            Must match one of the available options in AUTH_PROVIDERS.
        :type auth_provider_name: str
        :param profile_provider_name: Name of the profile provider to be used.
            Must match one of the available options in PROFILE_PROVIDERS.
        :type profile_provider_name: str
        :param role_provider_name: Name of the role provider to be used.
            Must match one of the available options in ROLE_PROVIDERS.
        :type role_provider_name: str

        :raises ValueError: If the `auth_provider_name` is not a valid
            entry in AUTH_PROVIDERS.
        :raises ValueError: If the `profile_provider_name` is not a valid
            entry in PROFILE_PROVIDERS.
        :raises ValueError: If the `role_provider_name` is not a valid
            entry in ROLE_PROVIDERS.
        """
        if auth_provider_name not in AUTH_PROVIDERS:
            raise ValueError(f"Unknown authentication provider: {auth_provider_name}")
        if profile_provider_name not in PROFILE_PROVIDERS:
            raise ValueError(f"Unknown profile provider: {profile_provider_name}")
        if role_provider_name not in ROLE_PROVIDERS:
            raise ValueError(f"Unknown role provider: {role_provider_name}")

        self.auth_provider = AUTH_PROVIDERS[auth_provider_name]()
        self.profile_provider = PROFILE_PROVIDERS[profile_provider_name]()
        self.role_provider = ROLE_PROVIDERS[role_provider_name]()

        LOGGER.info(
            f"Initialized AuthManager with auth={auth_provider_name}, "
            f"profile={profile_provider_name}, "
            f"and role={role_provider_name} providers"
        )

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
            id_token = auth_response["idToken"]
            user_id = auth_response["localId"]

            st.session_state["id_token"] = id_token
            user_info = self.auth_provider.get_account_info(id_token)

            if not user_info["emailVerified"]:
                self.auth_provider.send_email_verification(id_token)
                st.session_state.auth_warning = (
                    "Please verify your email before signing in."
                )
                st.rerun()

            st.session_state.user_info = user_info

            try:
                user_role = self.role_provider.get_user_role(id_token)
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
                            id_token, user_id, first_name, last_name
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
                id_token = auth_response["idToken"]
                user_id = auth_response["localId"]
                self.auth_provider.send_email_verification(id_token)
            else:
                auth_response = self.auth_provider.sign_in(email, password)
                id_token = auth_response["idToken"]
                user_id = auth_response["localId"]

            self.profile_provider.create_user_profile(
                id_token, user_id, first_name, last_name
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
            id_token = self.auth_provider.sign_in(
                st.session_state.user_info["email"], password
            )["idToken"]
            self.auth_provider.delete_account(id_token)
            st.session_state.clear()
            st.session_state.auth_success = "You have successfully deleted your account"
        except Exception as error:
            st.session_state.auth_warning = f"Error: {error}"
            st.rerun()

    def attempt_reauthentication(self):
        """
        Attempts to reauthenticate a user with their existing ID token. If the "id_token"
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
        if "id_token" in st.session_state and "user_info" not in st.session_state:
            try:
                user_info = self.auth_provider.get_account_info(
                    st.session_state["id_token"]
                )
                st.session_state.user_info = user_info
                role = self.role_provider.get_user_role(st.session_state["id_token"])
                st.session_state.role = role
                st.session_state.auth_success = "Reauthenticated successfully"
            except Exception:
                st.session_state.clear()


@conditional_cache_resource()
def get_auth_manager(
    auth_provider_name="default",
    profile_provider_name="default",
    role_provider_name="default",
):
    """
    Obtains an instance of `AuthManager` using specified provider names for
    authentication, profile, and role services. This function allows conditional
    caching of the resource to optimize performance depending on the context it is
    used in. Default provider names are set to "default" if not specified.

    :param auth_provider_name: The name of the authentication provider to use.
    :param profile_provider_name: The name of the profile provider to use.
    :param role_provider_name: The name of the role provider to use.
    :return: An instance of `AuthManager` configured with the specified provider
             names.
    """
    return AuthManager(auth_provider_name, profile_provider_name, role_provider_name)
