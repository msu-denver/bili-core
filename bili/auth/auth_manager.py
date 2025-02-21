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

from bili.auth.providers.auth.firebase_auth_provider import FirebaseAuthProvider
from bili.auth.providers.auth.in_memory_auth_provider import InMemoryAuthProvider
from bili.auth.providers.profile.in_memory_profile_provider import (
    InMemoryProfileProvider,
)
from bili.auth.providers.role.in_memory_role_provider import InMemoryRoleProvider
from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
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

    def create_jwt_token(self, payload: dict) -> str:
        """
        Creates a JSON Web Token (JWT) based on the provided payload.

        The method uses the `auth_provider` attribute to generate a JWT token.
        This token can be used for authentication or securing communication
        between systems or users.

        :param payload: A dictionary containing the claims and data to
            encode within the JWT token.
        :return: A string representation of the generated JWT token.
        :rtype: str
        """
        return self.auth_provider.create_jwt_token(payload)

    def verify_jwt_token(self, token: str) -> dict:
        """
        Verifies a JWT (JSON Web Token) using the configured authentication provider.

        This function is responsible for validating the given JWT token. It uses the
        authentication provider associated with the object to perform the verification
        process. Upon successful verification, it extracts and returns the decoded
        payload of the token, which contains information embedded within the JWT.

        :param token: The JWT string to be verified.
        :type token: str
        :return: A dictionary containing the decoded payload from the verified JWT.
        :rtype: dict
        """
        return self.auth_provider.verify_jwt_token(token)

    def extract_token_from_headers(self, headers) -> str:
        """
        Extracts an authentication token from the provided HTTP headers. The method attempts to
        retrieve the token using two approaches: first, by checking the "Authorization" header
        if it starts with "Bearer", and second, by retrieving the token from the "X-Auth-Token"
        header. If neither approach provides a valid token, a ValueError is raised.

        :param headers: The HTTP headers from which the token will be extracted.
        :type headers: dict
        :return: The extracted authentication token.
        :rtype: str
        :raises ValueError: If neither the "Authorization" header nor the "X-Auth-Token"
            header contain a valid token.
        """
        auth_header = headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]  # Extract token from "Bearer <token>"

        # Fallback: Try "X-Auth-Token" header
        token = headers.get("X-Auth-Token")
        if token:
            return token

        raise ValueError("No valid authentication token found in headers.")

    def verify_request_token(self, headers):
        """
        Verifies the authenticity of a request token extracted from the provided
        headers. The method uses an authentication provider to validate the token
        and ensure its integrity and compliance with authentication requirements.

        This method is commonly used in contexts where the incoming request headers
        contain a JWT token that needs verification to authenticate and authorize user
        access.

        :param headers: The HTTP headers where the token will be extracted from.
        :type headers: dict
        :return: The result of the verification process as determined by the
            authentication provider.
        :rtype: Any
        """
        token = self.extract_token_from_headers(headers)
        return self.auth_provider.verify_jwt_token(token)


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
