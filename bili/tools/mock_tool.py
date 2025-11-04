"""
Module: mock_tool

This module initializes and configures the Mock Tool for use within the
application. The Mock Tool is defined in the centralized configuration file
and includes options for simulating responses with specified delays. The tool
is initialized based on its specific parameters and prompts, which are provided
as arguments to the `init_mock_tool` function.

Functions:
    - init_mock_tool(name, description, mock_response, response_time) -> Tool:
      Initializes the Mock Tool with the provided name, description, mock
      response, and response time.

Dependencies:
    - time: Standard library module for time-related functions.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain.tools: Imports `Tool` for creating tools.
    - bili.streamlit.utils.streamlit_utils: Imports `conditional_cache_resource`
      for caching resources.

Usage:
    This module is intended to be used within the application to initialize and
    configure the Mock Tool based on the provided configuration. It supports
    simulating responses with specified delays.

Example:
    from bili.tools.mock_tool import init_mock_tool

    # Initialize the Mock Tool
    mock_tool = init_mock_tool(
        name="Mock Tool",
        description="Tool for simulating responses",
        mock_response="This is a mock response",
        response_time=2
    )
"""

import time

from langchain_core.tools import Tool

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource


@conditional_cache_resource()
def init_mock_tool(
    name, description, mock_response, response_time, middleware=None
) -> Tool:
    """
    Initializes a mock tool with a given name, description, mock response, and an optional
    response time to delay the responses. This function wraps the creation of a `Tool` object
    and ensures that the response time is properly parsed into a numeric value, defaulting
    to 0 if it cannot be converted.

    The created `Tool` object simulates responses by returning the provided mock response
    after an optional sleep duration specified by the response time.

    :param name: Name of the mock tool.
    :type name: str
    :param description: Description of the tool's purpose or functionality.
    :type description: str
    :param mock_response: The simulated response that the tool will return.
    :type mock_response: Any
    :param response_time: The time in seconds to delay the response. Can be passed as a
        number or a string convertible to a number. Defaults to 0 if validation fails.
    :type response_time: Union[int, float, str]
    :param middleware: List of middleware to be applied to the tool. Middleware can
        intercept and modify tool execution.
    :type middleware: list
    :return: Returns a `Tool` object initialized with the specified name, description,
        and simulation logic.
    :rtype: Tool
    """

    # Check if response_time is a number, if not convert it
    converted_response_time = response_time
    if not isinstance(converted_response_time, (int, float)):
        try:
            converted_response_time = float(converted_response_time)
        except ValueError:
            converted_response_time = 0

    # Define the tool's function to simulate responses
    def simulate_response(_):
        if converted_response_time > 0:
            time.sleep(converted_response_time)
        return mock_response

    return Tool(
        name=name,
        func=simulate_response,
        description=description,
        middleware=middleware or [],
    )
