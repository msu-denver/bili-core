"""
Module: api_serp

This module provides tools for interacting with the SERP API to retrieve search
results for a given query. It includes functions to execute queries and
initialize a Streamlit `Tool` for the SERP API.

Functions:
    - execute_query(query: str) -> str:
      Executes the query to retrieve search results from the SERP API and
      returns the results in JSON format.
    - init_serp_api_tool(name, description) -> Tool:
      Initializes and returns an instance of the Tool class for the SERP API.

Dependencies:
    - json: Standard library module for JSON operations.
    - os: Standard library module for interacting with the operating system.
    - requests: Third-party library for making HTTP requests.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain.tools: Imports `Tool` for creating tools.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within a Streamlit application to
    retrieve and display search results from the SERP API.

Example:
    from bili.tools.api_serp import init_serp_api_tool

    # Initialize the SERP API tool
    serp_tool = init_serp_api_tool(
        name="SERP API Tool",
        description="Tool for retrieving search results from SERP API"
    )
"""

import json
import os
import re
import urllib.parse

import requests
from langchain_core.tools import Tool

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def sanitize_serp_query(query: str) -> str:
    """
    Sanitizes user input for SERP API queries to prevent injection attacks.

    This function performs the following sanitization steps:
    1. Removes control characters and null bytes
    2. Removes potentially harmful characters (<, >, ", ')
    3. Limits query length to prevent abuse
    4. Normalizes whitespace
    5. URL-encodes the result

    :param query: The raw search query string from user input.
    :type query: str
    :return: A sanitized and URL-encoded query string safe for API use.
    :rtype: str
    """
    if not query:
        return ""

    # Remove control characters and null bytes
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", query)

    # Remove potential injection patterns (HTML/script tags, quotes)
    sanitized = re.sub(r"[<>\"']", "", sanitized)

    # Limit length to prevent abuse (SERP API handles ~2048 chars, we limit to 500)
    sanitized = sanitized[:500]

    # Strip and normalize whitespace
    sanitized = " ".join(sanitized.split())

    # URL-encode the result
    return urllib.parse.quote(sanitized)


def execute_query(query: str) -> str:
    """
    Executes a search query using the SERP API and returns the result as a JSON-formatted string.

    This function interacts with the SERP API by sending an HTTP GET request with the specified
    search query. The function expects the API key to be defined in the environment variables
    under "SERP_API_KEY". The query is sanitized before being sent to prevent injection attacks.
    The response is parsed and converted to a JSON-formatted string before being returned.

    :param query: The search query string to be sent to the SERP API.
    :type query: str
    :return: A JSON-formatted string containing the API response for the given query.
    :rtype: str
    """
    api_key = os.environ["SERP_API_KEY"]
    sanitized_query = sanitize_serp_query(query)
    LOGGER.debug(
        "Executing SERP query: %s (sanitized from: %s)", sanitized_query, query[:50]
    )
    url = f"https://serpapi.com/search.json?q={sanitized_query}&api_key={api_key}"
    response = requests.get(url, timeout=10)
    data = response.json()
    return json.dumps(data)


@conditional_cache_resource()
def init_serp_api_tool(name, description, middleware=None) -> Tool:
    """
    Initialize and configure a SERP API tool resource with caching.

    This function checks if the `SERP_API_KEY` is set in the environment variables, which
    is mandatory for the tool to function. It creates and returns an instance of `Tool`
    configured with the provided name, a function for query execution, and a description
    specifying its purpose.

    :param name: The name assigned to the tool instance.
    :param description: Details describing the purpose and functionality of the tool.
    :return: An instance of Tool configured with the specified properties.
    """
    # If SERP_API_KEY not set in environment, raise an error
    if "SERP_API_KEY" not in os.environ:
        raise ValueError("Please set the SERP_API_KEY environment variable.")

    return Tool(
        name=name,
        func=execute_query,
        description=description,
        middleware=middleware or [],
    )
