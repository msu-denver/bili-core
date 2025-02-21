"""
Module: ProfileProvider

This module defines the `ProfileProvider` interface, which outlines the
methods required for managing user profiles. Implementations of this
interface should provide the actual functionality for creating and
retrieving user profiles.

Classes:
    - ProfileProvider: Interface for user profile management.

Methods:
    - create_user_profile(uid, email, first_name, last_name):
      Creates a user profile with the provided details.
    - get_user_profile(uid, token):
      Retrieves user profile information based on the provided ID token.

Usage:
    This module is intended to be used as a base class for implementing
    user profile management. Subclasses should implement the methods to
    provide the actual functionality.

Example:
    class MyProfileProvider(ProfileProvider):
        def create_user_profile(self, uid, email, first_name, last_name):
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
        - create_user_profile(uid, email, first_name, last_name):
          Creates a user profile with the provided details.
        - get_user_profile(uid, token):
          Retrieves user profile information based on the provided ID token.
    """

    def create_user_profile(self, uid, email, first_name, last_name):
        """
        Creates a user profile in the system by taking user identification details
        and personal information. This method is a placeholder and must be implemented
        in subclasses or elsewhere to provide the desired functionality.

        :param uid: A unique authentication token identifying the user.
        :type uid: str
        :param email: A unique identifier for the user.
        :type email: str
        :param first_name: The first name of the user.
        :type first_name: str
        :param last_name: The last name of the user.
        :type last_name: str
        :return: None
        :rtype: NoneType
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
