"""
Module: ProfileProvider

This module defines the `ProfileProvider` interface, which outlines the
methods required for managing user profiles. Implementations of this
interface should provide the actual functionality for creating and
retrieving user profiles.

Classes:
    - ProfileProvider: Interface for user profile management.

Methods:
    - create_user_profile(uid, email, first_name, last_name, token):
      Creates a user profile with the provided details.
    - get_user_profile(uid, token):
      Retrieves user profile information based on the provided ID token.

Usage:
    This module is intended to be used as a base class for implementing
    user profile management. Subclasses should implement the methods to
    provide the actual functionality.

Example:
    class MyProfileProvider(ProfileProvider):
        def create_user_profile(uid, email, first_name, last_name, token):
            # Implement profile creation logic here
            pass

        def get_user_profile(self, uid, token):
            # Implement profile retrieval logic here
            pass
"""


class ProfileProvider:
    """
    Interface for user profile management.

    This class defines the methods required for creating and retrieving
    user profiles. Subclasses should implement these methods to provide
    the actual functionality.

    Methods:
        - create_user_profile(uid, email, first_name, last_name, token):
          Creates a user profile with the provided details.
        - get_user_profile(uid, token):
          Retrieves user profile information based on the provided ID token.
    """

    def create_user_profile(
        self, uid: str, email: str, first_name: str, last_name: str, token: str
    ):
        """
        Creates a user profile with the provided user details and token. This function is
        intended to handle the necessary steps to initialize and store a user profile based
        on the given data. The implementation is currently not provided and must be
        handled by the derived or overriding class.

        :param uid: Unique identifier of the user.
        :type uid: str
        :param email: Email address of the user.
        :type email: str
        :param first_name: First name of the user.
        :type first_name: str
        :param last_name: Last name of the user.
        :type last_name: str
        :param token: Authentication or session token for the user.
        :type token: str

        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def get_user_profile(self, uid, token):
        """
        Retrieve the profile of a user given their unique identifier and authentication token.

        This method is designed to fetch user details based on the provided user identifier and
        an authentication token. It raises a NotImplementedError as it is intended to be
        overridden or implemented elsewhere.

        :param uid: A unique identifier of the user whose profile is being retrieved.
        :type uid: str
        :param token: A valid authentication token for user verification.
        :type token: str
        :return: The profile details of the user.
        :rtype: dict
        :raises NotImplementedError: if the method is not overridden.
        """
        raise NotImplementedError
