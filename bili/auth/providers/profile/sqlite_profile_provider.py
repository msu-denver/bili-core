"""
sqlite_profile_provider.py
---------------------------

This module provides a class for managing user profile data stored in an SQLite database.
It includes methods for creating, retrieving, and updating user profiles.

Classes:
--------
- SQLiteProfileProvider:
    Handles operations for managing user profile data stored in an SQLite database.

Methods:
--------
- __init__(db_path=None):
    Initializes the SQLiteProfileProvider with the specified database path or a default path.

- _initialize_db():
    Initializes the database by creating the required table if it does not exist.

- create_user_profile(uid, email, first_name, last_name, token):
    Creates or updates a user profile in the database based on the provided information.

- get_user_profile(uid, token):
    Fetches the user profile information from the database for the given user ID.

Dependencies:
-------------
- os: Provides a way of using operating system dependent functionality.
- sqlite3: Provides a SQL interface compliant with the DB-API 2.0
specification described by PEP 249.

Usage:
------
To use the SQLiteProfileProvider, create an instance of the class and
call its methods to manage user
profiles in the SQLite database.

Example:
--------
from bili.auth.providers.profile.sqlite_profile_provider import SQLiteProfileProvider

# Initialize the profile provider
profile_provider = SQLiteProfileProvider()

# Create a new user profile
profile_provider.create_user_profile(
    uid="user123",
    email="test@example.com",
    first_name="John",
    last_name="Doe",
    token="auth_token"
)

# Get user profile
user_profile = profile_provider.get_user_profile(uid="user123", token="auth_token")
"""

import os
import sqlite3


class SQLiteProfileProvider:
    """
    Handles operations for managing user profile data stored in an SQLite database.

    This class provides an interface for interacting with an SQLite database to
    store, retrieve, and update user profile information. The data is stored in a
    table named `user_profiles`, which includes fields such as uid, email,
    first_name, and last_name.

    :ivar db_path: Path to the SQLite database file. Defaults to the value of the
        `PROFILE_DB_PATH` environment variable or `/tmp/user_profiles.db` if the
        environment variable is not set.
    :type db_path: str
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv("PROFILE_DB_PATH", "/tmp/user_profiles.db")
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the database by creating the required table if it does not exist.
        This method ensures that the database contains a `user_profiles` table with
        columns for user identification, email, and name information.

        The `uid` field serves as the primary key while `email`, `first_name`,
        and `last_name` fields must all contain non-null values. `role` will
        default to 'researcher' if not specified.

        Raises an exception if database access fails or if the table creation encounters
        an error.

        :raises sqlite3.Error: If there is an error while connecting to or modifying
            the database.
        """
        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    uid TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'researcher'
                )
            """
            )
            conn.commit()

    def create_user_profile(
        self, uid: str, email: str, first_name: str, last_name: str, token: str
    ):
        """
        Creates or updates a user profile in the database based on the provided
        information. This function ensures that the user profile with the given
        unique identifier (uid) is created or replaced in the user_profiles
        database table. The email, first name, and last name values are directly
        updated in this process. Token is not stored but used for processing.

        :param uid: A unique identifier for the user profile.
        :type uid: str
        :param email: The email address of the user.
        :type email: str
        :param first_name: The first name of the user.
        :type first_name: str
        :param last_name: The last name of the user.
        :type last_name: str
        :param token: Authentication or initialization token for internal validation.
        :type token: str
        :return: None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_profiles (uid, email, first_name, last_name)
                VALUES (?, ?, ?, ?)
            """,
                (uid, email, first_name, last_name),
            )
            conn.commit()

    def get_user_profile(self, uid: str, token: str):
        """
        Fetches the user profile information from the database for the given user ID.

        The function connects to the database, retrieves the requested user profile data
        based on the provided user ID (`uid`) and token (`token`), and returns the profile
        details as a dictionary if the user exists in the database. If the user ID is not
        found, the function returns None.

        :param uid: The unique identifier of the user whose profile is being requested.
        :type uid: str
        :param token: The security token for user or session authentication.
        :type token: str
        :return: A dictionary containing user profile information with keys
            `uid`, `email`, `first_name`, and `last_name` if the user exists.
            Returns None if the user is not found in the database.
        :rtype: dict or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT uid, email, first_name, last_name, role FROM user_profiles WHERE uid = ?",
                (uid,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "uid": row[0],
                    "email": row[1],
                    "first_name": row[2],
                    "last_name": row[3],
                    "role": row[4],
                }
            return None
