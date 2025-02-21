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
    - get_account_info(uid):
      Retrieves account information for the provided ID token.
    - send_email_verification(auth_details):
      Sends an email verification for a given user.
    - send_password_reset_email(email):
      Sends a password reset email to a user.
    - create_user(email, password):
      Creates a user with the specified email and password.
    - delete_user(uid):
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
    account_info = auth_provider.get_account_info(uid="user_uid")

    # Send email verification
    auth_provider.send_email_verification(auth_details)

    # Send password reset email
    auth_provider.send_password_reset_email(email="user@example.com")

    # Create a new user
    new_user = auth_provider.create_user(email="newuser@example.com", password="password")

    # Delete a user
    auth_provider.delete_user(uid="user_uid")
"""

import base64
import json
import os

import firebase_admin
import requests
from firebase_admin import auth

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

        :raises ValueError: If the environment variable `GOOGLE_CREDENTIALS` is not set or empty.
        """
        # Load Firebase configuration from environment variables
        firebase_config_json = os.getenv("GOOGLE_CREDENTIALS")

        if firebase_config_json:
            # Check if the config is base64-encoded and decode it if necessary
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

        # Initialize Firebase Admin SDK
        firebase_admin.initialize_app()

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
        response_data = request_object.json()
        response_object = {
            "token": response_data.get("idToken"),
            "refreshToken": response_data.get("refreshToken"),
            "uid": response_data.get("localId"),
        }
        return response_object

    def get_account_info(self, uid):
        """
        Retrieves account information for a given user ID (UID).

        This method fetches user data from Firebase Authentication using the provided
        UID. It constructs a dictionary containing user details such as email, display
        name, phone number, and other relevant attributes, if available. If the user
        does not exist or an error occurs during the retrieval process, an exception
        will be raised.

        :param uid: The unique identifier of the user for whom account details
                    are to be retrieved.
        :type uid: str
        :return: A dictionary containing the retrieved account details, including
                 fields such as localId, email, email_verified, display_name,
                 photo_url, phone_number, disabled, custom_claims, and uid (alias for
                 localId), if available.
        :rtype: dict
        :raises ValueError: If no user matching the given UID is found or if an error
                            occurs while retrieving user information.
        """
        try:
            user = auth.get_user(uid)
            user_dict = {
                "localId": user.uid,
                "email": user.email,
                "emailVerified": user.email_verified,
                "displayName": user.display_name,
                "photoURL": user.photo_url,
                "phoneNumber": user.phone_number,
                "disabled": user.disabled,
                "customClaims": user.custom_claims,
                "uid": user.uid,
            }
            return user_dict
        except firebase_admin.auth.UserNotFoundError:
            raise ValueError(f"User with UID {uid} not found")
        except Exception as e:
            raise ValueError(f"Error retrieving user info: {e}")

    def send_email_verification(self, auth_details):
        """
        Sends an email verification request to the Firebase Identity Toolkit API. The method
        constructs a POST request with the provided authentication details, including the
        user's ID token, to trigger a verification email for the corresponding account.

        The response from the API is returned in JSON format. Any errors raised during
        communication with the API are handled by an internal error-handling method.

        :param auth_details: A dictionary containing authentication details, specifically
            the key "token" representing the user's ID token required for the API request.
        :type auth_details: dict
        :return: A dictionary containing the JSON response from the Firebase API.
        :rtype: dict
        """
        request_ref = (
            f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/"
            f"getOobConfirmationCode?key={self.firebase_web_api_key}"
        )
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps(
            {"requestType": "VERIFY_EMAIL", "idToken": auth_details["token"]}
        )
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
        Creates a new user with the specified email and password using the
        authentication service. This function interacts with the authentication
        service to register a new user. If successful, it returns a dictionary
        containing the unique user ID and the email of the newly created user.
        If an error occurs during the creation process, it raises a ValueError
        containing the error message.

        :param email: The email address of the user to be created.
        :param password: The password for the new user.
        :return: A dictionary with keys 'uid' and 'email', representing the unique
            identifier and the email of the newly created user.
        :raises ValueError: If an error occurs during user creation, this exception
            contains the error details.
        """
        try:
            auth.create_user(email=email, password=password)
            return self.sign_in(email, password)
        except Exception as e:
            raise ValueError(f"Error creating user: {e}")

    def delete_user(self, uid):
        """
        Deletes a user account from Firebase using their UID.

        This method uses Firebase Admin SDK to delete a specific user account.
        The operation does not require an ID token since it is a privileged
        server-side operation.

        :param uid: The Firebase UID of the user to be deleted.
        :type uid: str
        :return: A dictionary confirming the deletion.
        :rtype: dict
        :raises ValueError: If the user is not found or an error occurs.
        """
        try:
            auth.delete_user(uid)
            return {"message": "User deleted successfully"}
        except firebase_admin.auth.UserNotFoundError:
            raise ValueError(f"User with UID {uid} not found")
        except Exception as e:
            raise ValueError(f"Error deleting user: {e}")

    def create_jwt_token(self, payload: dict) -> dict:
        """
        Create a Firebase JWT token using the provided payload.

        This function extracts the `uid` from the input payload and attempts to create a
        custom Firebase JWT token associated with that `uid`. If the `uid` is missing
        from the payload or if an error occurs during token creation, it raises a
        ValueError.

        :param payload: A dictionary containing the user data for whom the JWT token
            needs to be created. The dictionary must contain a `uid` field.
        :type payload: dict
        :return: A Firebase JWT token as a string.
        :rtype: str
        :raises ValueError: If the payload does not contain a `uid` field.
        :raises ValueError: If there is an error during the JWT token generation process.
        """
        id_token = payload.get("idToken")
        if not id_token:
            raise ValueError("Payload must contain a 'id_token' field")

        refresh_token = payload.get("refreshToken", None)

        return {"token": id_token, "refreshToken": refresh_token}

    def verify_jwt_token(self, token: str) -> dict:
        """
        Verifies the given Firebase JSON Web Token (JWT) to ensure its authenticity and validity.
        This function uses Firebase Admin SDK to decode and validate the token. If the token
        is valid, it processes the information embedded in it. Otherwise, it raises suitable
        exceptions to indicate invalid or failed token verification.

        :param token: The Firebase ID token JWT that needs to be verified.
        :type token: str

        :return: A dictionary containing the decoded information from the valid Firebase ID
            Token, including user information such as UID, email, etc.
        :rtype: dict

        :raises ValueError: Raised if the provided token is invalid or an error occurs
            during the token verification process. It includes specific details about
            the error encountered.
        """
        try:
            decoded_token = auth.verify_id_token(token)
            return decoded_token  # Contains user info (e.g., uid, email, etc.)
        except firebase_admin.auth.InvalidIdTokenError as e:
            print(e)
            raise ValueError("Invalid Firebase ID Token")
        except Exception as e:
            raise ValueError(f"Error verifying Firebase ID Token: {e}")

    def refresh_jwt_token(self, refresh_token: str) -> dict:
        """
        Refreshes a JWT token using a given refresh token.

        This function communicates with the Firebase secure token endpoint to obtain a
        new JWT token. It sends an HTTP POST request with the `refresh_token` and
        Firebase API key as payload. If the request fails, it raises a ValueError.
        On successful execution, it returns the refreshed token data as a dictionary.

        :param refresh_token: The refresh token provided for obtaining a new JWT token.
        :type refresh_token: str
        :returns: A dictionary containing the refreshed JWT token and related data.
        :rtype: dict
        :raises ValueError: If the HTTP request to refresh the token fails.
        """
        url = f"https://securetoken.googleapis.com/v1/token?key={self.firebase_web_api_key}"
        payload = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        response = requests.post(url, data=payload)

        if response.status_code != 200:
            raise ValueError("Failed to refresh token")

        response_data = response.json()
        resp = {
            "token": response_data.get("id_token"),
        }
        if "refresh_token" in response_data:
            resp["refreshToken"] = response_data["refresh_token"]
        else:
            resp["refreshToken"] = refresh_token
