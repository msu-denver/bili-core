"""
api_free_weather_api.py
----------

This module provides functionality for fetching and processing weather information
using an external weather API. It includes tools for integrating weather data into
a conversational AI system and registering the weather tool in the tool registry.

Features:
---------
- **Weather Fetching**: Retrieves the latest weather information for a specified city.
- **Error Handling**: Handles API errors, timeouts, and invalid responses gracefully.
- **Tool Integration**: Initializes and registers the weather tool for use in AI workflows.

Key Components:
---------------
- `fetch_weather`: An asynchronous function that fetches weather data from an external API.
- `init_weather_tool`: Initializes a weather tool for querying weather information.
- `initialize_weather_tool`: Registers the weather tool in the global tool registry.

Dependencies:
-------------
- `asyncio`: For asynchronous operations.
- `requests`: For making HTTP requests to the weather API.
- `bili.loaders.tools_loader.TOOL_REGISTRY`: For registering the weather tool.
- `bili.streamlit_ui.utils.streamlit_utils.conditional_cache_resource`: For caching resources.
- `langchain.tools.Tool`: For defining the weather tool.

Environment Variables:
----------------------
- `WEATHER_API_KEY`: The API key for accessing the weather service.
- `FREE_WEATHER_API`: The base URL for the weather API.

Usage:
------
This module is designed to be used as part of a conversational AI system. The weather tool
can be initialized and registered for use in workflows that require weather information.

Example:
--------
To fetch weather data for a city:
```python
from katie.tools.weather import fetch_weather

weather_data = asyncio.run(fetch_weather("New York"))
print(weather_data)

To initialize and register the weather tool:

"""

import asyncio
import logging
import os

import requests
from langchain_core.tools import Tool

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource

logger = logging.getLogger(__name__)


async def fetch_weather(city: str = "denver") -> dict:
    """
    Fetches the latest weather information.

    :return: A dictionary containing weather information.
    :rtype: dict
    """
    weather_api = os.environ.get("WEATHER_API_KEY", "")
    weather_url = os.environ.get("FREE_WEATHER_API", "")
    # Check if there is a weather url and API key
    if not weather_api:
        raise ValueError("Weather API key is not set in the environment variables.")
    if not weather_url:
        raise ValueError("Weather URL is not set in the environment variables.")

    # create weather endpoint
    weather_endpoint = f"{weather_url}{weather_api}&q={city}&aqi=no"
    logger.debug("Fetching weather from API: %s", weather_endpoint)
    try:
        # fetch the latest weather information
        response = requests.get(
            weather_endpoint, timeout=5
        )  # may need to change the timeout
        response.raise_for_status()
        data = response.json()
        logger.debug("Weather data: %s", data)

        # check if the response is empty
        if not data:
            logger.error("No data returned from Weather API")
            return {}

        # parse the data
        return {"weather": data}
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching weather data: %s", e)
        return {}
    except Exception as e:
        logger.error("Error parsing weather data: %s", e)
        return {}


@conditional_cache_resource()
def init_weather_tool(name, description) -> Tool:
    """
    Initialize the weather tool.

    :param name: The name of the tool.
    :type name: str
    :param description: The description for the tool.
    :type description: str
    :return: A Tool object for fetching and parsing weather data.
    :rtype: Tool
    """
    return Tool(
        name=name,
        func=lambda query: asyncio.run(fetch_weather(query)),
        description=description,
    )
