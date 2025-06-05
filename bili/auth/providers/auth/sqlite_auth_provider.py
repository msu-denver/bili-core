"""
sqlite_auth_provider.py
------------------------

This module provides a class for managing user authentication using an SQLite database.
It includes methods for creating, retrieving, and updating user credentials, as well as
generating and verifying JSON Web Tokens (JWT).

Classes:
--------
- SQLiteAuthProvider:
    Handles operations for managing user authentication and credentials stored in an SQLite database.

Methods:
--------
- __init__(db_path: str = "/tmp/auth_provider.db"):
    Initializes the SQLiteAuthProvider with the specified database path or a default path.

- _get_connection():
    Establishes and returns a connection to the SQLite database.

- _initialize_db():
    Initializes the database by creating the required table if it does not exist.

- sign_in(email: str, password: str):
    Authenticates a user based on their email and password.

- get_account_info(uid: str):
    Retrieves the account information for a user based on the provided unique user ID (uid).

- send_email_verification(auth_details: dict):
    Updates the email verification status for a user.

- send_password_reset_email(uid: str):
    Placeholder for sending a password reset email to the user.

- create_user(email: str, password: str):
    Creates a new user in the system with the provided email and password.

- delete_user(uid: str):
    Deletes a user account based on the provided ID token.

- create_jwt_token(payload: dict) -> dict:
    Encodes a JSON Web Token (JWT) using the provided payload.

- verify_jwt_token(token: str) -> dict:
    Verifies a JSON Web Token (JWT) to decode and validate its payload.

- refresh_jwt_token(refresh_token: str) -> dict:
    Placeholder for refreshing a JWT token using the provided refresh token.

Dependencies:
-------------
- datetime: Provides classes for manipulating dates and times.
- os: Provides a way of using operating system dependent functionality.
- sqlite3: Provides a SQL interface compliant with the DB-API 2.0 specification described by PEP 249.
- uuid: Provides functions for generating universally unique identifiers.
- jwt: Provides functions for encoding and decoding JSON Web Tokens.

Usage:
------
To use the SQLiteAuthProvider, create an instance of the class and call its methods to manage user authentication and credentials in the SQLite database.

Example:
--------
from bili.auth.providers.auth.sqlite_auth_provider import SQLiteAuthProvider

# Initialize the authentication provider
auth_provider = SQLiteAuthProvider()

# Create a new user
auth_provider.create_user(email="test@example.com", password="securepassword")

# Authenticate a user
user = auth_provider.sign_in(email="test@example.com", password="securepassword")

# Get user account information
account_info = auth_provider.get_account_info(uid=user["uid"])

# Generate a JWT token
token = auth_provider.create_jwt_token(payload={"uid": user["uid"]})

# Verify a JWT token
decoded_payload = auth_provider.verify_jwt_token(token=token["token"])
"""

import datetime
import os
import sqlite3
import uuid

import jwt

from bili.auth.providers.auth.auth_provider import AuthProvider


class SQLiteAuthProvider(AuthProvider):
    """
    Handles authentication using a SQLite database.

    This class implements a SQLite-based authentication provider, maintaining user
    credentials and related information. It provides methods for user management,
    authentication, and generation of JSON Web Tokens (JWT). This class is particularly
    useful for simple, file-based storage scenarios.

    :ivar db_path: Path to the SQLite database file used by the authentication provider.
    :type db_path: str
    """

    def __init__(self, db_path=None):
        """
        Initializes the authentication provider class and sets up the database path. Ensures
        the database is initialized upon creating an instance of this class. This class handles
        user authentication operations backed by a file-based database.

        :ivar db_path: Represents the path to the database file. It can be specified during
            initialization or uses a default value if not provided.
        """
        self.db_path = db_path or os.getenv("PROFILE_DB_PATH", "/tmp/user_profiles.db")
        self._initialize_db()

    def _get_connection(self):
        """
        Establishes and returns a connection to the SQLite database.

        This private method is used to create a new connection to the SQLite
        database using the file path specified during initialization. It
        ensures consistent access to the database throughout the application.

        :return: A new SQLite database connection object.
        :rtype: sqlite3.Connection
        """
        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        """
        Initializes the database by creating the `users` table if it does not already exist.

        The table consists of the following fields:
          - `uid`: A unique identifier, serving as the primary key of the table.
          - `email`: The user's unique email address.
          - `password`: The user's password.
          - `email_verified`: An indicator of whether the user's email is verified.
            Defaults to 0, which represents "not verified."

        This method uses a database connection obtained via `_get_connection` and ensures the schema
        integrity of the `users` table.

        :return: None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    uid TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    password TEXT,
                    email_verified INTEGER DEFAULT 0
                )
            """
            )
            conn.commit()

    def sign_in(self, email: str, password: str):
        """
        Authenticates the user by verifying provided email and password, and retrieves
        account details if the credentials are valid.

        :param email: The email address of the user to authenticate.
        :type email: str
        :param password: The password associated with the email for verification.
        :type password: str
        :return: A dictionary containing user account details:
                 - **uid** (*int*): The unique identifier of the user.
                 - **email** (*str*): The email address of the user.
                 - **password** (*str*): The password associated with the user account.
                 - **emailVerified** (*bool*): Whether the user's email has been verified.
        :rtype: dict
        :raises ValueError: If the provided credentials (email and/or password) are invalid.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT uid, email, password, email_verified FROM users WHERE email=?",
                (email,),
            )
            user = cursor.fetchone()
            if not user or user[2] != password:
                raise ValueError("Invalid credentials")
            profile = {
                "uid": user[0],
                "email": user[1],
                "password": user[2],
                "emailVerified": bool(user[3]),
            }
            token = self.create_jwt_token(profile)
            profile["token"] = token["token"]
            return profile

    def get_account_info(self, uid: str):
        """
        Fetches account information for a given user ID from the database.

        This method retrieves account details such as UID, email, password, and
        email verification status by querying the database with the provided user ID.

        :param uid: User ID for which the account information is to be retrieved.
        :type uid: str
        :return: A dictionary containing user account details:
            - `uid` (str): The user's unique identifier.
            - `email` (str): The user's email address.
            - `password` (str): The user's hashed password.
            - `emailVerified` (bool): Flag indicating if the user's email is verified.
        :rtype: dict
        :raises ValueError: If no user is found with the provided UID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT uid, email, password, email_verified FROM users WHERE uid=?",
                (uid,),
            )
            user = cursor.fetchone()
            if not user:
                raise ValueError("Invalid id token")
            return {
                "uid": user[0],
                "email": user[1],
                "password": user[2],
                "emailVerified": bool(user[3]),
            }

    def send_email_verification(self, auth_details):
        """
        Updates the email verification status for a user. This method retrieves the
        user ID from the provided authentication details and updates the corresponding
        user record in the database by marking their email as verified.

        :param auth_details: Dictionary containing authentication details. It must
            include the "uid" key to identify the user whose email verification status
            is to be updated.
        :type auth_details: dict
        :return: None. The method does not return any value.
        """
        uid = auth_details["uid"]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET email_verified = 1 WHERE uid=?", (uid,))
            conn.commit()

    def send_password_reset_email(self, email):
        """
        Sends a password reset email to a user identified by their unique ID.

        This method is intended to send a password reset email for a user using their
        unique ID. The actual implementation must be provided in a specific email
        provider. By default, raises a ``NotImplementedError`` as SQLite provider does
        not support email sending. Use an email-capable backend to utilize this
        functionality.

        :param email: Unique identifier of the user to send the password reset email to
        :type email: str
        :raises NotImplementedError: Raised when the function is not implemented for
                                     the current backend
        """
        raise NotImplementedError(
            "Password reset email not implemented for SQLite provider."
        )

    def create_user(self, email: str, password: str):
        """
        Creates a new user with the provided email and password, storing the information in
        a database. Each user is assigned a unique identifier (UID). The email address must
        be unique in the database. If the email already exists, the operation is not
        completed.

        :param email: The email address of the new user. Must be unique.
        :type email: str
        :param password: The password for the new user.
        :type password: str
        :return: A dictionary containing the generated UID, the email, the password provided,
            and a flag indicating whether the email is verified (defaults to False).
        :rtype: dict
        :raises ValueError: If the provided email address already exists in the database.
        """
        uid = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (uid, email, password) VALUES (?, ?, ?)",
                    (uid, email, password),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError("Email already exists")
        profile = {
            "uid": uid,
            "email": email,
            "password": password,
            "emailVerified": False,
        }
        token = self.create_jwt_token(profile)
        profile["token"] = token["token"]
        return profile

    def delete_user(self, uid: str):
        """
        Deletes a user from the database based on the provided unique identifier (UID).

        This method uses a database connection to remove a user entry matching the given UID
        from the 'users' table. The ID must be passed as a string, representing the user's
        unique identifier in the system.

        :param uid: The unique identifier (UID) of the user to be deleted from the database.
        :type uid: str
        :return: None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE uid=?", (uid,))
            conn.commit()

    def create_jwt_token(self, payload: dict) -> dict:
        """
        Generates a JSON Web Token (JWT) for the given payload. The token is signed with
        a secret key retrieved from environment variables and is set to expire in one hour
        from the time of creation.

        :param payload: A dictionary containing the payload data to encode in the JWT.
        :return: A dictionary containing the generated JWT under the key "token".
        """
        secret_key = os.getenv("JWT_SECRET_KEY")
        payload["exp"] = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            hours=1
        )
        return {"token": jwt.encode(payload, secret_key, algorithm="HS256")}

    def verify_jwt_token(self, token: str) -> dict:
        """
        Verifies a JWT (JSON Web Token) by decoding it using a predefined secret key and algorithm.
        This method ensures that the provided token matches the expected signature
        and has not expired. If the token is invalid or expired, it raises an appropriate
        error to indicate the issue.

        :param token: The JWT in string format to be verified.
        :type token: str
        :return: A dictionary containing the decoded payload of the JWT if verification is successful.
        :rtype: dict
        :raises ValueError: If the token is expired.
        :raises ValueError: If the token is invalid.
        """
        secret_key = os.getenv("JWT_SECRET_KEY")
        try:
            return jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired.")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token.")

    def refresh_jwt_token(self, refresh_token: str) -> dict:
        """
        Refreshes the JWT token by using the provided refresh token. This function is
        intended to handle the generation of a new JWT token when the existing token
        has expired. It raises a `NotImplementedError` indicating that this operation
        is not implemented for the current provider.

        :param refresh_token: The refresh token used to generate a new JWT.
        :type refresh_token: str
        :return: A dictionary containing the new JWT token details.
        :rtype: dict
        :raises NotImplementedError: Indicates that the token refresh operation is
            not supported or implemented for the current provider.
        """
        raise NotImplementedError("JWT refresh not implemented for this provider.")
