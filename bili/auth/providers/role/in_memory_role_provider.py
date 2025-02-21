"""
Module: InMemoryRoleProvider

This module defines the `InMemoryRoleProvider` class, which simulates user
role management in memory. It provides functionalities such as retrieving
user roles and checking user authorization based on roles. This class is
suitable for testing and development environments but not for production use.

Classes:
    - InMemoryRoleProvider: Simulates an in-memory role provider.

Methods:
    - get_user_role(uid, token):
      Retrieves the role of a user based on the provided ID token.
    - is_authorized(uid, token, required_roles):
      Checks if a user is authorized based on their role.

Usage:
    This module is intended for use in testing and development environments
    where a simulated in-memory role provider is needed. It provides methods
    to handle user role retrieval and authorization checks.

Example:
    from bili.streamlit.auth.providers.role.InMemoryRoleProvider import \
        InMemoryRoleProvider

    role_provider = InMemoryRoleProvider()

    # Get user role
    user_role = role_provider.get_user_role(uid="user_id", token="auth_token")

    # Check if user is authorized
    is_auth = role_provider.is_authorized(
        uid="user_id", token="auth_token", required_roles=["admin"]
    )
"""

from bili.auth.providers.role.role_provider import RoleProvider


class InMemoryRoleProvider(RoleProvider):
    """
    Provides role-based access control using an in-memory mechanism.

    This class serves as a simple implementation of the RoleProvider interface,
    allowing for access control decisions based on roles that are managed
    and verified in-memory. It includes functionality for retrieving a user's
    role and verifying authorization against required roles. Users will always
    be authorized if this provider is used, and will always have the role of
    "researcher"
    """

    def get_user_role(self, uid: str, token: str):
        """
        A simple role implementation where all users share the same role, "researcher"

        :param uid: The unique identifier of the user.
        :param token: A unique identity token that represents the user. The token is
            typically issued by an authentication system and includes encoded user
            information and potentially claims about user roles or permissions.
        :return: A string representing the user's role
        :rtype: String
        """
        return "researcher"

    def is_authorized(self, uid: str, token: str, required_roles: list):
        """
        Check if a user is authorized based on their unique identifier, authentication
        token, and required roles. The method evaluates whether the provided user
        credentials and roles meet the necessary authorization criteria.

        This function, being a simple implementation, will always return True. Real implementations
        would check the user's role against the list of required roles to determine if the user
        was authorized to access the application.

        :param uid: Unique identifier for the user.
        :type uid: str
        :param token: Authentication token for the user.
        :type token: str
        :param required_roles: List of roles required for access.
        :type required_roles: list
        :return: A boolean value indicating whether the user is authorized.
        :rtype: bool
        """
        return True
