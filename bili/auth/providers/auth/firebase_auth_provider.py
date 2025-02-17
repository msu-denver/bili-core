"""
Module: FirebaseAuthProvider

This module defines the `FirebaseAuthProvider` class, which extends the
`AuthProvider` base class to provide authentication functionalities using
Firebase. It includes methods for signing in, retrieving account information,
sending email verifications, sending password reset emails, creating users,
and deleting users. The class interacts with Firebase's REST API to perform
these operations.

Classes:
    - FirebaseAuthProvider: Extends `AuthProvider` to implement Firebase
      authentication.

Methods:
    - __init__():
      Initializes the Firebase configuration from environment variables.
    - sign_in(email, password):
      Authenticates a user with the provided email and password.
    - get_account_info(id_token):
      Retrieves account information for the provided ID token.
    - send_email_verification(id_token):
      Sends an email verification for a given user.
    - send_password_reset_email(email):
      Sends a password reset email to a user.
    - create_user(email, password):
      Creates a user with the specified email and password.
    - delete_user(id_token):
      Deletes a user from the system.

Usage:
    This module is intended to be used for Firebase authentication operations.
    It provides methods to handle user authentication, account management, and
    email-based operations using Firebase's REST API.

Example:
    from bili.auth.providers.auth.FirebaseAuthProvider import \
        FirebaseAuthProvider

    auth_provider = FirebaseAuthProvider()

    # Sign in a user
    user_info = auth_provider.sign_in(email="user@example.com", password="password")

    # Get account information
    account_info = auth_provider.get_account_info(id_token="user_id_token")

    # Send email verification
    auth_provider.send_email_verification(id_token="user_id_token")

    # Send password reset email
    auth_provider.send_password_reset_email(email="user@example.com")

    # Create a new user
    new_user = auth_provider.create_user(email="newuser@example.com", password="password")

    # Delete a user
    auth_provider.delete_user(id_token="user_id_token")
"""

import base64
import json
import os

import requests

from bili.auth.providers.auth.auth_provider import AuthProvider


def _raise_detailed_error(request_object):
    """
    Raises a detailed HTTPError with the error details and response text if the
    request_object does not pass the `raise_for_status` check.

    The function attempts to call the `raise_for_status` method on the
    provided `request_object`. If an HTTPError is encountered, it re-raises
    the exception with additional context that includes the response text.

    :param request_object: The request object to check and potentially throw
        an error for. Should be a Response object from the `requests` library.
    :type request_object: requests.models.Response
    :raises requests.exceptions.HTTPError: Re-raised error with additional
        response context.
    """
    try:
        request_object.raise_for_status()
    except requests.exceptions.HTTPError as error:
        raise requests.exceptions.HTTPError(error, request_object.text)


class FirebaseAuthProvider(AuthProvider):
    """
    Represents a Firebase authentication provider.

    This class is designed to interact with Firebase Authentication capabilities. It provides
    methods to authenticate users, retrieve account information, send email verifications,
    handle password resets, manage user creation and deletion functionalities using Firebase's
    REST API. The class depends on a Firebase configuration specified as a JSON string, which
    should be provided through the "GOOGLE_CREDENTIALS" environment variable. The configuration
    string can be either a base64-encoded JSON or a raw JSON.

    :ivar firebase_web_api_key: Firebase Web API key extracted from the configuration file.
    :type firebase_web_api_key: str
    """

    def __init__(self):
        """
        Initialize Firebase configuration by loading it from environment variables and setting up
        the Firebase Web API key. The configuration can either be a base64-encoded JSON string or
        a raw JSON string. If the required environment variable is not set or empty, an exception
        is raised. This setup is essential for ensuring Firebase services can be accessed securely
        and properly.

        :param firebase_config_json: The Firebase configuration JSON string loaded from the
            `GOOGLE_CREDENTIALS` environment variable. Can be a base64-encoded or plain JSON string.
        :type firebase_config_json: str

        :raises ValueError: If the environment variable `GOOGLE_CREDENTIALS` is not set or empty.
        """
        # Load Firebase configuration from environment variables
        firebase_config_json = os.getenv("GOOGLE_CREDENTIALS")

        if firebase_config_json:
            # Check if the string is base64 encoded
            try:
                decoded_bytes = base64.b64decode(firebase_config_json)
                firebase_config = json.loads(decoded_bytes)
            except (base64.binascii.Error, json.JSONDecodeError):
                firebase_config = json.loads(firebase_config_json)
        else:
            raise ValueError(
                "Environment variable GOOGLE_CREDENTIALS is not set or is empty"
            )

        # Get the Firebase Web API Key from the config JSON
        self.firebase_web_api_key = firebase_config.get("apiKey")

    def sign_in(self, email, password):
        """
        Authenticates a user with the provided email and password using Firebase Authentication.

        The function sends a POST request to Firebase Authentication to verify the provided email
        and password. Upon success, it returns the response in JSON format including a token if
        authentication is successful. If the authentication fails, it raises an appropriate error
        with detailed information.

        The request includes the Firebase web API key, which should be configured during
        initialization. The response should contain user identification and
        authentication-related details if successful.

        :param email: The email address of the user for authentication.
        :type email: str
        :param password: The password of the user for authentication.
        :type password: str
        :return: The JSON response from the Firebase Authentication API
                 containing user authentication details.
        :rtype: dict
        :raises HTTPError: If the HTTP request returns a status code indicating an error.
        :raises RequestException: If there is an issue with the HTTP request itself.
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key="
            f"{self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps(
            {"email": email, "password": password, "returnSecureToken": True}
        )
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()

    def get_account_info(self, id_token):
        """
        Retrieves user account information associated with the given ID token from Firebase.

        This method fetches account details by making a POST request to the Firebase
        Identity Toolkit API. It sends the ID token provided in the request payload
        to retrieve the associated account's information. The method assumes that the
        provided `id_token` is valid and associated with an existing user in Firebase.

        :param id_token: A string representing the Firebase identity token required
            to fetch user account information.
        :return: A dictionary representing the account information of the user, with
            the data extracted from the API response.
        :raises ValueError: If the response from the Firebase API indicates an error
            or does not contain the expected data.
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/getAccountInfo?"
            f"key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps({"idToken": id_token})
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()["users"][0]

    def send_email_verification(self, id_token):
        """
        Sends an email verification request to the specified user based on the provided
        ID token. This function utilizes Firebase's Identity Toolkit API to initiate the
        process of sending a verification email to the user's registered email address.

        The function constructs the appropriate API endpoint with the Firebase web API
        key, sets the necessary headers for the request, and sends a POST request with
        the required payload that specifies the verification request. If the request
        fails, a detailed error is raised. Upon success, the response data in JSON
        format is returned.

        :param id_token: The ID token of the user for whom the email verification is
                         requested.
        :type id_token: str

        :return: A dictionary containing the JSON response from the Firebase API,
                 which includes information about the email verification request.
        :rtype: dict
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/"
            f"getOobConfirmationCode?key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps({"requestType": "VERIFY_EMAIL", "idToken": id_token})
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()

    def send_password_reset_email(self, email):
        """
        Sends a password reset email to the specified email address using the Firebase
        Identity Toolkit API. This method constructs the required API request with
        proper headers and body, sends an HTTP POST request to the Firebase endpoint,
        and handles potential errors using a helper method.

        :param email: The email address to send the password reset request to.
        :type email: str
        :return: A dictionary containing the JSON response from the Firebase API.
        :rtype: dict
        :raises HTTPError: If there is an HTTP-related error in the request.
        :raises Timeout: If the request times out.
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/"
            f"getOobConfirmationCode?key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps({"requestType": "PASSWORD_RESET", "email": email})
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()

    def create_user(self, email, password):
        """
        Creates a new user account using the provided email and password. It communicates
        with Firebase Authentication REST API to register the user and obtain the
        authentication credentials.

        :param email: The email address for the new user account.
        :type email: str
        :param password: The password for the new user account.
        :type password: str
        :return: A dictionary containing the response data with user details and
            authentication tokens.
        :rtype: dict
        :raises requests.exceptions.RequestException: If there is an issue with the HTTP request.
        :raises Exception: If the API responds with an error or the response cannot be processed.
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?"
            f"key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps(
            {"email": email, "password": password, "returnSecureToken": True}
        )
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()

    def delete_user(self, id_token):
        """
        Delete a user account from Firebase using their ID token.

        This method interacts with Firebase's REST API to delete a specific user account.
        It requires the ID token of the user to perform the operation. The method sends
        a POST request to the Firebase Identity Toolkit API endpoint with the user's ID
        token in the request body. If the request fails, the method raises a detailed
        error based on the response. On success, it returns the JSON response from the API.

        :param id_token: The ID token of the user whose account is to be deleted.
        :type id_token: str
        :return: The JSON response from the Firebase API after deleting the user.
        :rtype: dict
        :raises Exception: Raises a detailed exception if the request fails or the API
            returns an error.
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/deleteAccount?"
            f"key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps({"idToken": id_token})
        request_object = requests.post(
            request_ref, headers=headers, data=data, timeout=10
        )
        _raise_detailed_error(request_object)
        return request_object.json()
