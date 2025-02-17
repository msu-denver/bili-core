"""
Module: state_management

This module provides functions to manage the state of forms within a Streamlit
application. It includes functions to enable and disable forms by modifying
the session state.

Functions:
    - disable_form():
      Disables a form by setting its processing state in the session state.
    - enable_form():
      Enables a form by resetting its processing state in the session state.

Dependencies:
    - streamlit: Provides the Streamlit library for building web applications.

Usage:
    This module is intended to be used within a Streamlit application to manage
    the state of forms, particularly to prevent user interaction during
    processing.

Example:
    from bili.streamlit.utils.state_management import disable_form, enable_form

    # Disable the form
    disable_form()

    # Enable the form
    enable_form()
"""

import streamlit as st


def disable_form():
    """
    Disables a form by setting its processing state.

    This function modifies the Streamlit session state by marking the query
    as being processed. It is typically used to prevent user interaction
    with the form while a query or process is ongoing.

    :raises AttributeError: If `st.session_state` is not initialized or
        does not have the attribute `is_processing_query`.
    :return: None
    """
    st.session_state.is_processing_query = True


def enable_form():
    """
    Disables the processing query state by setting the application form
    to a non-processing status.

    This function modifies the `is_processing_query` attribute of the
    Streamlit session state to indicate that the application is no longer
    in a state of processing queries.

    :return: None
    """
    st.session_state.is_processing_query = False
