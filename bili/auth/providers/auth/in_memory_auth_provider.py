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
    - get_account_info(id_token):
      Fetches account information for the given id_token.
    - send_email_verification(id_token):
      Marks the email associated with the given id_token as verified.
    - send_password_reset_email(email):
      Simulates sending a password reset email to the user.
    - create_user(email, password):
      Creates a new user in the system with the provided email and password.
    - delete_account(id_token):
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
    account_info = auth_provider.get_account_info(id_token=new_user["idToken"])

    # Send email verification
    auth_provider.send_email_verification(id_token=new_user["idToken"])

    # Send password reset email
    auth_provider.send_password_reset_email(email="newuser@example.com")

    # Delete a user
    auth_provider.delete_account(id_token=new_user["idToken"])
"""

import uuid

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
        email and id_token.
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

    def get_account_info(self, id_token):
        """
        Fetches account information for the given id_token.

        This method takes an id_token as input and returns the associated
        user account information if the token is valid. If the provided
        id_token does not exist, it raises a ValueError indicating that
        the token is invalid.

        :param id_token: The id token provided by the user to retrieve account
                         information.
        :type id_token: str
        :return: The account information associated with the given id_token.
        :rtype: dict
        :raises ValueError: If the provided id_token is not found in the
                            user database.
        """
        if id_token not in self.users:
            raise ValueError("Invalid id token")
        return self.users.get(id_token)

    def send_email_verification(self, id_token):
        """
        Marks the email associated with the given `id_token` as verified. This
        function updates the user's email verification status within the
        system's data storage.

        :param id_token: A string representing a token that identifies a user.
        :type id_token: str
        :return: None
        :rtype: None
        """
        email = self.users[id_token]["email"]
        self.users[id_token]["emailVerified"] = True
        self.users[email]["emailVerified"] = True

    def send_password_reset_email(self, email):
        """
        Simulates sending a password reset email to the user. Since this is an in-memory
        authentication provider, the email is not actually sent. Instead, this function
        serves as a placeholder for the actual email sending functionality.

        :param email: The email address of the user who requested a password reset.
        :type email: str
        :returns: None
        """

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
        id_token = str(uuid.uuid4())
        user = {
            "idToken": id_token,
            "localId": email,
            "email": email,
            "password": password,
            "emailVerified": False,
        }
        self.users[id_token] = user
        self.users[email] = user
        return self.users[id_token]

    def delete_user(self, id_token: str):
        """
        Deletes a user account based on the provided ID token.

        This function removes the user data associated with the provided ID token
        from the users dictionary. Both the token-to-email mapping and the
        email-to-token mapping are deleted to ensure the account is fully removed.

        :param id_token: A unique identifier token associated with a user account.
        :type id_token: str
        :return: None
        """
        if id_token in self.users:
            email = self.users[id_token]["email"]
            del self.users[email]
            del self.users[id_token]
