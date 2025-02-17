"""
Module: api_weather_gov

This module provides tools for retrieving and processing weather forecast data
from the weather.gov API. It includes functions to fetch forecast data based on
geographic coordinates and initialize a Streamlit `Tool` for the weather.gov API.

Functions:
    - get_forecast_context_4k(url):
      Retrieves the forecast for the given latitude and longitude coordinates,
      returning the forecast in the form of a string.
    - execute_query(query: str) -> str:
      Executes the query to retrieve weather forecast data based on latitude
      and longitude.
    - init_weather_gov_api_tool(name, description) -> Tool:
      Initializes and returns an instance of the Tool class for the weather.gov
      API.

Dependencies:
    - json: Standard library module for JSON operations.
    - re: Standard library module for regular expressions.
    - requests: Third-party library for making HTTP requests.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain.tools: Imports `Tool` for creating tools.
    - langchain_core.tools: Imports `ToolException` for handling tool-specific
      exceptions.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within a Streamlit application to
    retrieve and display weather forecast data from the weather.gov API.

Example:
    from bili.tools.api_weather_gov import init_weather_gov_api_tool

    # Initialize the weather.gov API tool
    weather_gov_tool = init_weather_gov_api_tool(
        name="Weather.gov API Tool",
        description="Tool for retrieving weather forecast data from weather.gov API"
    )
"""

import json
import re

import requests
from langchain.tools import Tool
from langchain_core.tools import ToolException

from bili.streamlit.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def get_forecast_context_4k(url):
    """
    Fetches and processes weather forecast data from a specified URL. The method initially
    sends a request to retrieve the forecast JSON data from the NOAA API's forecast endpoint.
    It further narrows down the content by retaining only the `detailedForecast` field for
    each forecast period and removes all other unwanted metadata to simplify the data.

    :param url: The API endpoint to retrieve the weather forecast data
                as a JSON response.
    :type url: str
    :return: A JSON string containing processed weather forecast data
             with detailed forecasts for each period.
    :rtype: str
    :raises ToolException: If there are any issues during the API request
                           or response processing, particularly connection
                           errors or a non-successful status code response.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # we'll use the forecast endpoint instead of the hourly forecast endpoint, this returns 12
        forecast_url = data["properties"]["forecast"]
        # hour periods instead of 1 hour periods.
        response = requests.get(forecast_url, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()["properties"]["periods"]
        # now we trim a lot, lets drop some data that we're less concerned with.
        # drop everything in each period except these keys: name, startTime,
        # endTime,detailedForecast
        for period in forecast_data:
            period.pop("number", None)
            period.pop("isDaytime", None)
            period.pop("startTime", None)
            period.pop("endTime", None)
            period.pop("temperature", None)
            period.pop("temperatureUnit", None)
            period.pop("temperatureTrend", None)
            period.pop("probabilityOfPrecipitation", None)
            period.pop("relativeHumidity", None)
            period.pop("dewpoint", None)
            period.pop("windSpeed", None)
            period.pop("windDirection", None)
            period.pop("icon", None)
            period.pop("shortForecast", None)
        result = json.dumps(forecast_data)
        return result
    except requests.exceptions.RequestException as e:
        raise ToolException(f"Error in API_WeatherGOVTool: {e}") from e


def execute_query(query: str) -> str:
    """
    Processes a geographic coordinate query and returns forecast data from the
    National Weather Service API. The function validates and ensures the input
    query is in a proper latitude,longitude format. If incorrectly formatted,
    an error message is returned instead. The forecast data is adapted for specific
    context length requirements before being returned.

    :param query: Geographic coordinates in the format "latitude,longitude".
    :type query: str
    :return: Forecast data as a string or an error message if the query is invalid.
    :rtype: str
    """

    # validate and fix the query to ensure it's in lat,lon format and contains only one comma
    query = re.sub(
        r"[^\d.,-]", "", query
    )  # Remove any characters that are not digits, dots, commas, or hyphens
    # Ensure only one comma is present
    parts = query.split(",")
    if len(parts) > 2:
        query = ",".join(
            parts[:2]
        )  # Keep only the first two parts separated by a comma

    # Print the query for debugging purposes
    LOGGER.debug(query)

    if not re.match(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$", query):
        return (
            "Invalid query. Please provide the latitude and longitude in the format "
            "'latitude,longitude' and try again."
        )  # return an error message to the LLM if the query is invalid.
    # if we raise an error, the application crashes, so we'll return an error message instead,
    # which allows the LLM to try again.

    url = f"https://api.weather.gov/points/{query}"

    # TODO: Then we'll check the context length of the model, and slice the
    # forecast data accordingly.

    # Then we'll call the get_forecast_context_4k function to get the forecast
    # data in 4k context length.
    forecast_data = get_forecast_context_4k(url)

    # TODO: Or, if the context length is longer than 16k, we can return the full forecast data.
    #  Do we need to? not necessarily, but it's an option.

    # Then we'll return the forecast data
    return forecast_data


@conditional_cache_resource()
def init_weather_gov_api_tool(name, description) -> Tool:
    """
    Initializes the Weather.gov API Tool.

    This function sets up a Tool instance configured with the provided name,
    description, and a predefined execution function. It serves as a wrapper
    to initialize the tool and attach the relevant execution method for handling
    queries to the Weather.gov API.

    :param name: The name of the tool instance.
    :type name: str
    :param description: The description explaining the tool's purpose.
    :type description: str
    :return: An initialized instance of the Tool configured for Weather.gov API usage.
    :rtype: Tool
    """
    return Tool(
        name=name,
        func=execute_query,
        description=description,
    )
