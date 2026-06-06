"""Tests for bili.iris.tools.api_free_weather_api.

Covers the async fetch_weather function (endpoint construction, env-var
guards, empty-response and error handling) and tool initialization. The
async tests use anyio, matching the project's async test convention.
"""

# pylint: disable=missing-function-docstring

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import Tool

from bili.iris.tools.api_free_weather_api import fetch_weather, init_weather_tool

pytestmark = pytest.mark.anyio


class TestFetchWeather:
    """Verify async weather fetching against the free weather API."""

    @patch.dict(
        "os.environ",
        {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": "http://api/?key="},
        clear=False,
    )
    @patch("bili.iris.tools.api_free_weather_api.requests.get")
    async def test_builds_endpoint_and_returns_wrapped_data(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {"current": {"temp_c": 20}}
        mock_get.return_value = resp

        result = await fetch_weather("boulder")

        assert result == {"weather": {"current": {"temp_c": 20}}}
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://api/?key=abc&q=boulder&aqi=no"
        assert mock_get.call_args[1]["timeout"] == 5
        resp.raise_for_status.assert_called_once()

    @patch.dict(
        "os.environ",
        {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": "http://api/?key="},
        clear=False,
    )
    @patch("bili.iris.tools.api_free_weather_api.requests.get")
    async def test_empty_response_returns_empty_dict(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {}
        mock_get.return_value = resp

        assert await fetch_weather("denver") == {}

    @patch.dict(
        "os.environ",
        {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": "http://api/?key="},
        clear=False,
    )
    @patch("bili.iris.tools.api_free_weather_api.requests.get")
    async def test_request_exception_returns_empty_dict(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("down")

        assert await fetch_weather("denver") == {}

    @patch.dict(
        "os.environ",
        {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": "http://api/?key="},
        clear=False,
    )
    @patch("bili.iris.tools.api_free_weather_api.requests.get")
    async def test_generic_exception_returns_empty_dict(self, mock_get):
        # A non-request error during parsing is caught by the broad handler.
        resp = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp

        assert await fetch_weather("denver") == {}

    @patch.dict("os.environ", {"WEATHER_API_KEY": "", "FREE_WEATHER_API": "u"})
    async def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Weather API key is not set"):
            await fetch_weather("denver")

    @patch.dict("os.environ", {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": ""})
    async def test_missing_url_raises_value_error(self):
        with pytest.raises(ValueError, match="Weather URL is not set"):
            await fetch_weather("denver")


class TestInitWeatherTool:
    """Verify tool construction wraps the async fetch in asyncio.run."""

    def test_builds_tool(self):
        tool = init_weather_tool("free_weather", "desc")
        assert isinstance(tool, Tool)
        assert tool.name == "free_weather"

    @patch.dict(
        "os.environ",
        {"WEATHER_API_KEY": "abc", "FREE_WEATHER_API": "http://api/?key="},
        clear=False,
    )
    @patch("bili.iris.tools.api_free_weather_api.requests.get")
    def test_tool_func_runs_fetch_synchronously(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {"current": {"temp_c": 5}}
        mock_get.return_value = resp

        tool = init_weather_tool("free_weather", "desc")
        result = tool.func("denver")

        assert result == {"weather": {"current": {"temp_c": 5}}}
