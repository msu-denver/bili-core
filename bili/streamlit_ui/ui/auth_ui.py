"""
Module: auth_ui

This module provides the user interface for authentication within a Streamlit
application. It includes functions to display login and signup forms, check
user authentication status, and manage the authentication manager.

Functions:
    - display_login_signup():
      Displays a login and signup interface for the application.
    - is_authenticated():
      Determines if the current user is authenticated based on session state.
    - check_auth():
      Checks if a user is authenticated and manages login status accordingly.
    - initialize_auth_manager(auth_provider_name="default",
                              profile_provider_name="default",
                              role_provider_name="default"):
      Initializes the authentication manager with specified providers.

Dependencies:
    - streamlit: Provides the Streamlit library for building web applications.
    - bili.auth.AuthManager: Imports AuthManager for managing authentication.
    - bili.streamlit.utils.streamlit_utils: Imports conditional_cache_resource
      for caching resources conditionally.

Usage:
    This module is intended to be used within a Streamlit application to manage
    user authentication. It provides functions to display login and signup
    forms, check authentication status, and initialize the authentication
    manager.

Example:
    from bili.streamlit.ui.auth_ui import check_auth, initialize_auth_manager

    # Initialize the authentication manager
    initialize_auth_manager()

    # Check if the user is authenticated
    check_auth()
"""

import streamlit as st

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource


def display_login_signup():
    """
    Displays a login and signup interface for the application.

    This function uses Streamlit to render a graphical interface for users to either
    log in to their account or create a new one. It dynamically handles session states
    to display appropriate warnings, success messages, and adjust the UI based on
    profile creation needs or prior authentication actions.

    Session keys used in `st.session_state` include:
    - `auth_warning`: Displays a warning message if authentication issues occur.
    - `auth_success`: Displays a success message upon correct authentication.
    - `needs_profile_creation`: Determines whether the user needs to create a profile
      and adjusts the default selected option accordingly.

    The function handles login and signup logic based on the selection made in the UI.
    User-provided input for email, password, and additional fields (for signup) is
    captured and used to interact with an external authentication manager.

    Authentication actions include:
    - Calling `st.session_state.auth_manager.create_account` during the signup process.
    - Calling `st.session_state.auth_manager.sign_in` during the login process.

    :return: None
    """
    st.title("Login/Signup")
    if "auth_warning" in st.session_state and st.session_state.auth_warning:
        st.warning(st.session_state.auth_warning)
        st.session_state.auth_warning = ""
    if "auth_success" in st.session_state and st.session_state.auth_success:
        st.success(st.session_state.auth_success)
        st.session_state.auth_success = ""

    index = 0
    existing_user = False
    if (
        "needs_profile_creation" in st.session_state
        and st.session_state.needs_profile_creation
    ):
        index = 1
        existing_user = True

    option = st.selectbox("Login or Signup", ["Login", "Signup"], index=index)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        first_name = st.text_input("First Name", key="first_name")
        last_name = st.text_input("Last Name", key="last_name")
        if st.button("Create Account"):
            st.session_state.auth_manager.create_account(
                email, password, first_name, last_name, existing_user
            )
    else:
        if st.button("Log In"):
            st.session_state.password = password
            st.session_state.auth_manager.sign_in(email, password)


def is_authenticated():
    """
    Determines if the current user is authenticated based on session state.

    The function checks if the user's information and role are stored within
    the Streamlit session state. If the user's role is identified as either
    'researcher' or 'admin', the user is considered authenticated.

    :return: True if the user is authenticated, False otherwise.
    :rtype: bool
    """
    if "user_info" in st.session_state and "role" in st.session_state:
        if st.session_state.role in ["researcher", "admin"]:
            return True
    return False


def check_auth():
    """
    Checks if a user is authenticated and manages login status accordingly.

    This function ensures that a user is properly authenticated before allowing
    access to further functionalities. If not authenticated, it prompts the
    login/signup interface and stops the streamlit script. On successful
    authentication, it displays a welcome message and provides the option
    for the user to sign out. Signing out clears the user's session and
    restarts the script.

    :raises RuntimeError: Stops the execution of the Streamlit app if the user
                          is not authenticated.
    """
    if not is_authenticated():
        display_login_signup()
        st.stop()
    else:
        st.success(f"Welcome {st.session_state.user_info['email']}!")
        if st.button("Sign Out", use_container_width=True):
            st.session_state.auth_manager.sign_out()
            st.success("You have signed out.")
            st.rerun()
        st.markdown("---")


@conditional_cache_resource()
def initialize_auth_manager(
    auth_provider_name="default",
    profile_provider_name="default",
    role_provider_name="default",
):
    """
    Initialize the authentication manager with specified provider names for
    authentication, profile, and role providers. This function initializes an
    instance of `AuthManager` with the provided parameters, allowing integration
    and customization of authentication workflows. If no specific provider names
    are supplied, the default providers will be used.

    :param auth_provider_name: The name of the authentication provider to be used. Defaults to "default".
    :type auth_provider_name: str
    :param profile_provider_name: The name of the profile provider to be used. Defaults to "default".
    :type profile_provider_name: str
    :param role_provider_name: The name of the role provider to be used. Defaults to "default".
    :type role_provider_name: str
    :return: An initialized `AuthManager` instance configured with the specified
             providers.
    :rtype: AuthManager
    """
    from bili.streamlit_ui.ui.ui_auth_manager import UIAuthManager

    auth_manager = UIAuthManager(
        auth_provider_name=auth_provider_name,
        profile_provider_name=profile_provider_name,
        role_provider_name=role_provider_name,
    )
    return auth_manager
