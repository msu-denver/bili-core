"""
Module: api_open_weather

This module provides tools for interacting with the OpenWeather API to retrieve
weather data for a given location. It includes functions to get geocode data,
fetch weather information, execute queries, and initialize a Streamlit `Tool`
for the OpenWeather API.

Functions:
    - sanitize_input(input_str):
      Sanitizes an input string by removing unwanted characters and formatting
      it to be URL-safe.
    - get_geocode(city_name, state_code="CO", country_code="US"):
      Retrieves the latitude and longitude for the specified city.
    - get_weather(lat, lon):
      Retrieves weather data for the specified latitude and longitude.
    - execute_query(query: str) -> str:
      Executes the query to retrieve weather data for the specified location.
    - init_weather_api_tool(name, description) -> Tool:
      Initializes and returns an instance of the Tool class for the OpenWeather
      API.

Dependencies:
    - os: Standard library module for interacting with the operating system.
    - re: Standard library module for regular expressions.
    - urllib.parse: Standard library module for URL parsing.
    - requests: Third-party library for making HTTP requests.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain.tools: Imports `Tool` for creating tools.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within a Streamlit application to
    retrieve and display weather data from the OpenWeather API.

Example:
    from bili.tools.api_open_weather import init_weather_api_tool

    # Initialize the weather API tool
    weather_tool = init_weather_api_tool(
        name="Weather API Tool",
        description="Tool for retrieving weather data from OpenWeather API"
    )
"""

import os
import re
import urllib.parse

import requests
from langchain.tools import Tool

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def sanitize_input(input_str):
    """
    Sanitizes an input string by removing unwanted characters and formatting it
    to be URL-safe. The function removes non-alphanumeric characters except spaces
    and hyphens, trims excessive spaces/newlines, and URL-encodes the result.

    :param input_str: The string to sanitize.
    :type input_str: str
    :return: A sanitized and URL-encoded version of the input string.
    :rtype: str
    """
    # Remove unwanted characters (newline, tabs, excessive whitespace, commas,
    # and other URL-unsafe characters)
    # Remove all non-alphanumeric characters except spaces and hyphens
    sanitized = re.sub(r"[^\w\s-]", "", input_str.strip())
    # Replace multiple spaces/newlines with a single space
    sanitized = re.sub(r"\s+", " ", sanitized)

    return urllib.parse.quote(sanitized)  # URL-encode the cleaned string


def get_geocode(city_name, state_code="CO", country_code="US"):
    """
    Retrieve the geocode (latitude and longitude) for a given city, state, and country.

    The function contacts the OpenWeatherMap Geocoding API using the provided city name,
    state code, and country code to obtain the geographic coordinates. It requires an
    API key, which should be stored in the `OPENWEATHERMAP_API_KEY` environment variable.
    The inputs are sanitized to ensure a robust query to the API. The function returns
    the latitude and longitude of the first matching location from the API response.
    If no location is found or an error occurs during the request, it returns `None, None`.

    :param city_name: Name of the city for which geocode is to be retrieved.
    :type city_name: str
    :param state_code: Abbreviation of the state where the city is located. By default,
        it is set to "CO".
    :type state_code: str, optional
    :param country_code: Abbreviation of the country where the city is located. By default,
        it is set to "US".
    :type country_code: str, optional
    :return: A tuple containing two elements: the latitude and longitude of the city. If
        no location is found or an error occurs, returns (None, None).
    :rtype: tuple[float, float] | tuple[None, None]
    :raises requests.exceptions.RequestException: If any error occurs during the API request.
    """
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]

    # Sanitize each input component
    sanitized_city = sanitize_input(city_name)
    sanitized_state = sanitize_input(state_code)
    sanitized_country = sanitize_input(country_code)

    url = (
        f"http://api.openweathermap.org/geo/1.0/direct?"
        f"q={sanitized_city},{sanitized_state},{sanitized_country}&limit=5&appid={api_key}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            return data[0]["lat"], data[0]["lon"]
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Error occurred while making the API request: {e}")
        raise e
    return None, None

def get_geocode_from_zip(zip_code, country_code="US"):
    """
    Retrieves the geocode (latitude and longitude) for a given zip code.

    Uses OpenWeather's ZIP code geocoding API to fetch the latitude and longitude
    based on the zip code and country.

    :param zip_code: ZIP code to look up.
    :type zip_code: str
    :param country_code: Country code (defaults to 'US').
    :type country_code: str, optional
    :return: Tuple containing (latitude, longitude), or (None, None) if an error occurs.
    :rtype: tuple[float, float] | tuple[None, None]
    """
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]

    # Sanitize zip code input
    sanitized_zip = sanitize_input(zip_code)
    sanitized_country = sanitize_input(country_code)

    url = f"http://api.openweathermap.org/geo/1.0/zip?zip={sanitized_zip},{sanitized_country}&appid={api_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data.get("lat"), data.get("lon")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Error occurred while fetching geocode by ZIP code: {e}")
        return None, None


def get_weather(lat, lon):
    """
    Fetches current weather data from the OpenWeatherMap API for a given latitude and longitude.

    This function makes a GET request to the OpenWeatherMap API using the provided geographic
    coordinates (latitude and longitude) and an API key accessed from the environment variables.
    It retrieves the weather information in JSON format and returns it. Logs and raises an exception
    if any issues occur during the API request.

    :param lat: The latitude of the location for which weather data is required.
        Must be a float or a type convertible to float.
    :param lon: The longitude of the location for which weather data is required.
        Must be a float or a type convertible to float.
    :return: The JSON data containing the weather information retrieved from the API.
    :rtype: dict
    :raises requests.exceptions.RequestException: If any error occurs during the HTTP request to
        the OpenWeatherMap API, such as a timeout or an invalid response.
    """
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Error occurred while making the API request: {e}")
        raise e


def execute_query(query: str) -> str:
    """
    Executes a query to retrieve weather data for a specified location. The query
    may include a city and state separated by a comma or space. If the state is
    not specified, it defaults to "CO". The method sanitizes the input, retrieves
    geographical coordinates, and fetches associated weather data. If any step
    fails, an appropriate error message is returned.

    :param query: A string representing the location query that includes the city
                  and optionally the state, separated by a comma or space.
    :type query: str
    :return: A string representing the retrieved weather data or an error message
             if the data could not be obtained.
    :rtype: str
    """
    # Detect if the query is a ZIP code (5-digit numeric)
    if query.isdigit() and len(query) == 5:
        lat, lon = get_geocode_from_zip(query)
    else:
        # Assume it's a city/state input, split the query into city and state if a comma or space is present
        if "," in query:
            city, state = map(str.strip, query.split(",", 1))
        elif " " in query:
            city, state = map(str.strip, query.split(" ", 1))
        else:
            city, state = query, "CO"  # Default to "CO" if no state is provided

    # Sanitize and encode the city and state
    city = sanitize_input(city)
    state = sanitize_input(state)

    # Get the latitude and longitude for the city and state
    lat, lon = get_geocode(city, state)
    if lat is None or lon is None:
        return "Could not find the location."

    # Get the weather data for the latitude and longitude
    weather_data = get_weather(lat, lon)
    if weather_data is None:
        return "Failed to retrieve weather data."

    return str(weather_data)


@conditional_cache_resource()
def init_weather_api_tool(name, description) -> Tool:
    """
    Initializes a Tool object configured for interacting with the OpenWeatherMap API.

    This function creates a new Tool instance for accessing the OpenWeatherMap API
    and specifies the behavior for querying the API's weather data functionalities.
    If the required API key is not found in the environment variables, the function
    raises an error, requiring users to properly configure their setup with the
    `OPENWEATHERMAP_API_KEY`.

    :param name: Name of the tool to be initialized.
    :type name: str
    :param description: Description of the tool to indicate its purpose or functionality.
    :type description: str
    :return: A Tool instance ready for accessing the OpenWeatherMap API with the given
        configuration.
    :rtype: Tool
    :raises ValueError: If `OPENWEATHERMAP_API_KEY` is missing from environment variables.
    """
    # If OPENWEATHERMAP_API_KEY not set in environment, raise an error
    if "OPENWEATHERMAP_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENWEATHERMAP_API_KEY environment variable.")

    return Tool(
        name=name,
        func=execute_query,
        description=description,
    )
