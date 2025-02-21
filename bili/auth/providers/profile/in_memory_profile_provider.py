"""
Module: InMemoryProfileProvider

This module defines the `InMemoryProfileProvider` class, which provides a
mechanism to manage user profiles in memory. It is an implementation of the
`ProfileProvider` interface, where profiles are stored and accessed within
the application's memory without relying on external databases or storage
systems. This class is suitable for lightweight or ephemeral use cases.

Classes:
    - InMemoryProfileProvider: Manages user profiles in memory.

Methods:
    - __init__():
      Initializes the in-memory profile storage.
    - create_user_profile(uid, email, first_name, last_name, token):
      Creates a new user profile and stores it in memory.
    - get_user_profile(uid, token):
      Fetches the profile of a user based on the provided user ID.

Usage:
    This module is intended for use in scenarios where user profiles need to
    be managed in memory, such as in testing or development environments. It
    provides methods to create and retrieve user profiles without relying on
    external storage systems.

Example:
    from bili.auth.providers.profile.InMemoryProfileProvider import \
        InMemoryProfileProvider

    profile_provider = InMemoryProfileProvider()

    # Create a new user profile
    profile_provider.create_user_profile(
        uid="user123",
        email="test@example.com",
        first_name="John",
        last_name="Doe"
        token="auth_token"
    )

    # Get user profile
    user_profile = profile_provider.get_user_profile(uid="user123", token="auth_token")
"""

from bili.auth.providers.profile.profile_provider import ProfileProvider


class InMemoryProfileProvider(ProfileProvider):
    """
    Provides a mechanism to manage user profiles in memory. This class is an implementation
    of the ProfileProvider, where profiles are stored and accessed within the application's
    memory without relying on external databases or storage systems. It is suitable for
    lightweight or ephemeral use cases.

    :ivar profiles: Dictionary to store user profiles with uid as the key and another
        dictionary containing profile information (id_token, first_name, last_name) as the value.
    :type profiles: dict
    """

    def __init__(self):
        self.profiles = {}  # Store profiles in memory

    def create_user_profile(
        self, uid: str, email: str, first_name: str, last_name: str, token: str
    ):
        """
        Creates a user profile in an internal data store. This method associates
        a unique identifier (UID) with user-specific details such as first name
        and last name, storing them as a profile for later access. The function
        ensures that the required user information is correctly structured and
        maintained within the profiles data structure.

        :param uid: A unique identifier for the user.
        :type uid: str
        :param email: The user's email address.
        :type email: str
        :param first_name: The first name of the user.
        :type first_name: str
        :param last_name: The last name of the user.
        :type last_name: str
        :param token: Authentication or identification token for the user.
        :type token: str

        :return: None
        :rtype: None
        """
        self.profiles[uid] = {
            "uid": uid,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
        }

    def get_user_profile(self, uid: str, token: str):
        """
        Fetches the user profile based on the given unique identifier and authentication token.
        This method retrieves the user's information stored in the `profiles` dictionary.
        If the profile corresponding to the provided identifier is not found,
        it returns an empty dictionary.

        :param uid: The unique identifier of the user whose profile is to be retrieved.
        :type uid: str
        :param token: The authentication token required to access the user profile.
        :type token: str
        :return: A dictionary containing the user's profile information if the user exists;
        otherwise, an empty dictionary.
        :rtype: dict
        """
        return self.profiles.get(uid, {})
