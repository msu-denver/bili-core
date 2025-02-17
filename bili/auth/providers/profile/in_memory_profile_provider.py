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
    - create_user_profile(id_token, user_id, first_name, last_name):
      Creates a new user profile and stores it in memory.
    - get_user_profile(user_id):
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
        id_token="unique_id_token",
        user_id="user123",
        first_name="John",
        last_name="Doe"
    )

    # Get user profile
    user_profile = profile_provider.get_user_profile(user_id="user123")
"""

from bili.auth.providers.profile.profile_provider import ProfileProvider


class InMemoryProfileProvider(ProfileProvider):
    """
    Provides a mechanism to manage user profiles in memory. This class is an implementation
    of the ProfileProvider, where profiles are stored and accessed within the application's
    memory without relying on external databases or storage systems. It is suitable for
    lightweight or ephemeral use cases.

    :ivar profiles: Dictionary to store user profiles with user_id as the key and another
        dictionary containing profile information (id_token, first_name, last_name) as the value.
    :type profiles: dict
    """

    def __init__(self):
        self.profiles = {}  # Store profiles in memory

    def create_user_profile(
        self, id_token: str, user_id: str, first_name: str, last_name: str
    ):
        """
        Creates a new user profile and stores it in the profiles dictionary. This method
        associates a unique user ID with their ID token and personal name information,
        including the first and last names.

        :param id_token: A unique string token associated with the user's identity.
        :param user_id: A unique string identifier for the user, used as the key in the
            profiles dictionary.
        :param first_name: The first name of the user, as a string.
        :param last_name: The last name of the user, as a string.
        :return: None
        """
        self.profiles[user_id] = {
            "id_token": id_token,
            "first_name": first_name,
            "last_name": last_name,
        }

    def get_user_profile(self, id_token: str):
        """
        Fetches the profile of a user based on the provided user ID.

        This method retrieves the profile of a user from the 'profiles' attribute
        using the given user ID. If the user ID does not exist in the 'profiles',
        it returns an empty dictionary.

        :param id_token: The unique identifier of the user whose profile is to be retrieved.
        :type id_token: str
        :return: The profile corresponding to the given user ID, or an empty dictionary if no
                 matching profile is found.
        :rtype: dict
        """
        return self.profiles.get(id_token, {})
