"""
Module: streamlit_utils

This module provides utility functions for conditional caching in Streamlit
applications. It includes decorators to apply Streamlit's caching mechanisms
based on the presence of specific environment variables, allowing for flexible
caching behavior depending on the runtime environment.

Functions:
    - conditional_cache_resource():
      Decorator to conditionally apply the Streamlit `st.cache_resource`
      decorator based on the presence of the `STREAMLIT_SERVER_ADDRESS`
      environment variable.
    - conditional_cache_data():
      Decorator to conditionally cache the output of a function using
      Streamlit's `st.cache_data` decorator if the `STREAMLIT_SERVER_ADDRESS`
      environment variable is present.
    - conditional_cache():
      Decorator to conditionally cache computations or function results using
      Streamlit's `st.cache` decorator if the `STREAMLIT_SERVER_ADDRESS`
      environment variable is present.

Dependencies:
    - os: Provides a way of using operating system dependent functionality.
    - streamlit: Provides the Streamlit library for caching resources.

Usage:
    This module is intended to be used within Streamlit applications to
    conditionally apply caching based on the runtime environment. It provides
    decorators to enable or disable caching depending on the presence of
    specific environment variables.

Example:
    from bili.utils.streamlit_utils import conditional_cache_data

    @conditional_cache_data
    def load_data():
        # Function implementation
        pass
"""

import os


def conditional_cache_resource():
    """
    Determines the appropriate cache resource decoration based on the environment and applies the
    decorator to the given function. Automatically selects the @st.cache_resource decorator when
    executed in a Streamlit environment and preserves the original function otherwise.

    This decorator facilitates differentiation between Streamlit and non-Streamlit execution
    environments, ensuring compatibility and optimized behavior based on the execution context.

    :return: A decorated function, either cached with Streamlit's cache mechanism or left
        unmodified depending on the environment.
    :rtype: Callable
    """

    def decorator(func):
        # If STREAMLIT_SERVER_ADDRESS exists in the environment, then we are running in Streamlit
        # and should use the @st.cache_resource decorator.
        # Otherwise, we are running in a different environment and should not use the decorator.
        if "STREAMLIT_SERVER_ADDRESS" in os.environ:
            import streamlit as st

            return st.cache_resource(func)

        return func

    return decorator


def conditional_cache_data():
    """
    Determines whether to use caching for a function based on the runtime environment. Specifically,
    if the runtime environment is Streamlit (as identified by the presence of the
    STREAMLIT_SERVER_ADDRESS environment variable), it applies the Streamlit cache_data decorator.
    Otherwise, it returns the function unchanged.

    The choice of caching ensures optimized performance in Streamlit environments while maintaining
    compatibility with other runtime contexts.

    :return: A decorator that conditionally applies Streamlit's cache_data if the environment is
        Streamlit, or returns the function unchanged in other environments.
    :rtype: Callable
    """

    def decorator(func):
        # If STREAMLIT_SERVER_ADDRESS exists in the environment, then we are running in Streamlit
        # and should use the @st.cache decorator.
        # Otherwise, we are running in a different environment and should not use the decorator.
        if "STREAMLIT_SERVER_ADDRESS" in os.environ:
            import streamlit as st

            return st.cache_data(func)
        else:
            return func

    return decorator


def conditional_cache():
    """
    Determines whether to apply caching functionality to a function based on the environment.
    The decorator applies the Streamlit `@st.cache` caching mechanism if the current environment
    is identified as a Streamlit environment (by the presence of the "STREAMLIT_SERVER_ADDRESS"
    environment variable). Otherwise, the original function is returned without modification.

    :return: A decorator function that applies Streamlit's caching mechanism if executed in
             a Streamlit environment; returns the original function otherwise.
    :rtype: Callable
    """

    def decorator(func):
        # If STREAMLIT_SERVER_ADDRESS exists in the environment, then we are running in Streamlit
        # and should use the @st.cache decorator.
        # Otherwise, we are running in a different environment and should not use the decorator.
        if "STREAMLIT_SERVER_ADDRESS" in os.environ:
            import streamlit as st

            return st.cache(func)

        return func

    return decorator
