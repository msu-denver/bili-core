"""
Module: InMemoryAuthProvider

This module defines the `InMemoryAuthProvider` class, which simulates user
authentication and management in memory. It provides functionalities such as
signing in, account creation, email verification, and password reset. This
class is suitable for testing and development environments but not for
production use.

Classes:
    - InMemoryAuthProvider: Simulates an in-memory authentication provider.

Methods:
    - __init__():
      Initializes the in-memory user storage.
    - sign_in(email, password):
      Authenticates a user based on their email and password.
    - get_account_info(uid):
      Fetches account information for the given uid.
    - send_email_verification(auth_details):
      Marks the email associated with the given auth_details as verified.
    - send_password_reset_email(email):
      Simulates sending a password reset email to the user.
    - create_user(email, password):
      Creates a new user in the system with the provided email and password.
    - delete_account(uid):
      Deletes a user account based on the provided ID token.

Usage:
    This module is intended for use in testing and development environments
    where a simulated in-memory authentication provider is needed. It provides
    methods to handle user authentication, account management, and email-based
    operations.

Example:
    from bili.auth.providers.auth.InMemoryAuthProvider import \
        InMemoryAuthProvider

    auth_provider = InMemoryAuthProvider()

    # Create a new user
    new_user = auth_provider.create_user(email="newuser@example.com", password="password")

    # Sign in a user
    user_info = auth_provider.sign_in(email="newuser@example.com", password="password")

    # Get account information
    account_info = auth_provider.get_account_info(uid=new_user["uid"])

    # Send email verification
    auth_provider.send_email_verification(uid=new_user["uid"])

    # Send password reset email
    auth_provider.send_password_reset_email(email="newuser@example.com")

    # Delete a user
    auth_provider.delete_account(uid=new_user["uid"])
"""

import datetime
import os
import uuid

import jwt

from bili.auth.providers.auth.auth_provider import AuthProvider


class InMemoryAuthProvider(AuthProvider):
    """
    Simulated in-memory authentication provider.

    This class is used to simulate user authentication and management in a
    system. It provides functionalities including signing in, account creation,
    email verification, and password reset, all executed in memory. It uses a
    dictionary to store user data during runtime, making it suitable for
    testing and development environments, but not for production.

    :ivar users: Dictionary to store user data during runtime indexed by both
        email and uid.
    :type users: dict
    """

    def __init__(self):
        """
        Class to manage users using an in-memory dictionary.

        The class provides functionality to handle users stored in a
        dictionary format. Users are represented as keys within a
        dictionary, enabling straightforward management and retrieval.
        """
        self.users = {}  # Store users in memory using a dictionary

    def sign_in(self, email: str, password: str):
        """
        Authenticates a user based on their email and password.

        This method checks the provided email and password against the stored
        user data. If the authentication is successful, the user information
        is returned. Otherwise, an exception is raised.

        :param email: The user's email address used for authentication.
        :type email: str
        :param password: The user's password associated with the provided email.
        :type password: str
        :return: The user data if authentication is successful.
        :rtype: dict
        :raises ValueError: If the email is not found or the password does not match.
        """
        if email in self.users and self.users[email]["password"] == password:
            return self.users[email]
        raise ValueError("Invalid credentials")

    def get_account_info(self, uid):
        """
        Retrieves the account information for a user based on the provided unique user ID (uid).

        This method checks if the given uid is present in the list of users. If the
        uid is not found, an error is raised indicating invalidity of the id token.
        Otherwise, it retrieves and returns the user's account information.

        :param uid: The unique identifier of the user
        :type uid: str

        :return: A dictionary containing the account information of the user
        :rtype: dict

        :raises ValueError: If the provided uid does not exist in the users list
        """
        if uid not in self.users:
            raise ValueError("Invalid id token")
        return self.users.get(uid)

    def send_email_verification(self, auth_details):
        """
        Sends an email verification for a user's email address. This function marks the email as verified
        for the user identified by the provided `auth_details`. The operation updates the email verification
        status in the internal `users` database.

        :param auth_details: A dictionary containing user authentication details, where "uid" is the unique
            identifier of the targeted user.
        :type auth_details: dict
        :return: None
        """
        uid = auth_details["uid"]
        email = self.users[uid]["email"]
        self.users[uid]["emailVerified"] = True
        self.users[email]["emailVerified"] = True

    def send_password_reset_email(self, uid):
        """
        Simulates sending a password reset email to the user. Since this is an in-memory
        authentication provider, the email is not actually sent. Instead, this function
        serves as a placeholder for the actual email sending functionality.

        :param email: The email address of the user who requested a password reset.
        :type email: str
        :returns: None
        """
        raise NotImplementedError(
            "Password reset email not implemented for in-memory provider."
        )

    def create_user(self, email: str, password: str):
        """
        Creates a new user in the system with the provided email and password. If the email
        already exists in the system, raises a ValueError. Generates a unique ID token
        for the user, then stores the user's information with the generated token and
        email as keys.

        :param email: The email address of the new user.
        :type email: str
        :param password: The password of the new user.
        :type password: str
        :raises ValueError: If the provided email already exists in the system.
        :return: A dictionary containing the created user's information.
        :rtype: dict
        """
        if email in self.users:
            raise ValueError("Email already exists")
        uid = str(uuid.uuid4())
        user = {
            "uid": uid,
            "email": email,
            "password": password,
            "emailVerified": False,
        }
        self.users[uid] = user
        self.users[email] = user
        token = self.create_jwt_token(user)
        user["token"] = token["token"]
        return user

    def delete_user(self, uid: str):
        """
        Deletes a user account based on the provided ID token.

        This function removes the user data associated with the provided ID token
        from the users dictionary. Both the token-to-email mapping and the
        email-to-token mapping are deleted to ensure the account is fully removed.

        :param uid: A unique identifier token associated with a user account.
        :type uid: str
        :return: None
        """
        if uid in self.users:
            email = self.users[uid]["email"]
            del self.users[email]
            del self.users[uid]

    def create_jwt_token(self, payload: dict) -> dict:
        """
        Encodes a JSON Web Token (JWT) using the provided payload. The token is signed
        with a secret key and includes an expiration time of 1 hour from the current
        time.

        :param payload: A dictionary containing the claims to be encoded in the JWT.
                        This must include any necessary claims required by the
                        application.
        :return: A signed and encoded JWT as a string.

        """
        secret_key = os.getenv("JWT_SECRET_KEY")
        payload["exp"] = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            hours=1
        )  # 1-hour expiry
        return {"token": jwt.encode(payload, secret_key, algorithm="HS256")}

    def verify_jwt_token(self, token: str) -> dict:
        """
        Verifies a JSON Web Token (JWT) to decode and validate its payload. This method
        utilizes a secret key retrieved from the environment variable "JWT_SECRET_KEY"
        to decode the token, ensuring its integrity and authenticity. It supports the
        HS256 algorithm for decoding.

        :param token: The JWT string to be decoded and verified.
        :type token: str
        :return: A dictionary containing the decoded payload of the JWT if successfully
            validated.
        :rtype: dict
        :raises ValueError: If the token has expired or is invalid (e.g., tampered or
            malformed).
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
        Refresh a JWT token using the provided refresh token.

        This method takes a refresh token as input and attempts to
        refresh the corresponding JWT token. It raises a
        NotImplementedError, indicating that the functionality for
        refreshing JWT tokens has not been implemented for this
        provider. The output of this method is a dictionary
        that may contain the refreshed token and any associated
        metadata when implemented.

        :param refresh_token: The refresh token used to request a new
            JWT token.
        :type refresh_token: str
        :return: A dictionary containing the refreshed JWT token and
            associated metadata.
        :rtype: dict
        :raises NotImplementedError: When the method is called, as the
            functionality has not been implemented.
        """
        raise NotImplementedError("JWT refresh not implemented for this provider.")
