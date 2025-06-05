"""
sqlite_role_provider.py
------------------------

This module provides a class for managing user roles and authorization using an SQLite database.
It includes methods for retrieving user roles and checking user authorization based on roles.

Classes:
--------
- SQLiteRoleProvider:
    Implements the RoleProvider interface to manage user roles and authorization using SQLite.

Methods:
--------
- __init__(db_path=None):
    Initializes the SQLiteRoleProvider with the specified database path or a default path.

- get_user_role(uid: str, token: str) -> str:
    Retrieves the role of a user based on their unique identifier.

- is_authorized(uid: str, token: str, required_roles: list) -> bool:
    Checks if a user is authorized based on their role.

Dependencies:
-------------
- os: Provides a way of using operating system dependent functionality.
- sqlite3: Provides a SQL interface compliant with the DB-API 2.0 specification described by PEP 249.
- bili.auth.providers.role.role_provider.RoleProvider: Interface for role-based access control.

Usage:
------
To use the SQLiteRoleProvider, create an instance of the class and call its methods to manage user roles and
authorization in the SQLite database.

Example:
--------
from bili.auth.providers.role.sqlite_role_provider import SQLiteRoleProvider

# Initialize the role provider
role_provider = SQLiteRoleProvider()

# Get user role
user_role = role_provider.get_user_role(uid="user123", token="auth_token")

# Check if user is authorized
is_authorized = role_provider.is_authorized(uid="user123", token="auth_token", required_roles=["admin"])
"""

import os
import sqlite3

from bili.auth.providers.role.role_provider import RoleProvider


class SQLiteRoleProvider(RoleProvider):
    """
    Implementation of the RoleProvider interface that uses SQLite for role management.

    This class provides methods to retrieve user roles and check user authorization based
    on roles stored in an SQLite database.

    :ivar db_path: Path to the SQLite database file containing user profiles.
    :type db_path: str
    """

    def __init__(self, db_path=None):
        """
        Initializes an instance of the class responsible for managing user profiles.

        The class provides functionality to interact with a database where
        user profiles are stored. The database path can be either provided directly
        or derived from the environment variable "PROFILE_DB_PATH". In case neither
        is provided, the default path "/tmp/user_profiles.db" will be used.

        This constructor sets up the primary database file path which will be
        used throughout the instance for performing operations on user profiles.

        :param db_path: Path to the database file. If not provided, it will attempt
            to use the value of the "PROFILE_DB_PATH" environment variable. If
            the variable is not set, it defaults to "/tmp/user_profiles.db".
        :type db_path: Optional[str]
        """
        self.db_path = db_path or os.getenv("PROFILE_DB_PATH", "/tmp/user_profiles.db")

        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_user_role(self, uid: str, token: str) -> str:
        """
        Retrieve the role of a user based on their unique identifier.

        This function connects to a SQLite database and queries the user_profiles
        table to fetch the role associated with the given user ID (uid). The function
        returns the role as a string if the user exists in the database, otherwise,
        returns None.

        :param uid: Unique identifier of the user.
        :param token: Authorization token for validating the user.
        :return: The role of the user as a string, or None if the user does not exist.
        """
        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM user_profiles WHERE uid = ?", (uid,))
            result = cursor.fetchone()
            return result[0] if result else None

    def is_authorized(self, uid: str, token: str, required_roles: list) -> bool:
        """
        Check if a user is authorized based on their role.

        This function determines whether a user, identified by their unique identifier
        (uid) and authentication token (token), has the required roles necessary to
        access a specified resource or perform a specific action. It retrieves the
        user's role by using the `get_user_role` method and checks if the role exists
        within the list of required roles. If the role is not found, the function
        returns `False`.

        :param uid: The unique identifier for the user.
        :type uid: str
        :param token: The authentication token of the user.
        :type token: str
        :param required_roles: A list of roles necessary for authorization.
        :type required_roles: list
        :return: True if the user has one of the required roles, otherwise False.
        :rtype: bool
        """
        user_role = self.get_user_role(uid, token)
        return user_role in required_roles if user_role else False
