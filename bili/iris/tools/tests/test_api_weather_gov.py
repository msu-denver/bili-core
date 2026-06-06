"""Tests for bili.iris.tools.api_weather_gov.

Covers the two-hop forecast fetch and field trimming, query validation and
sanitization in execute_query, and tool initialization. HTTP traffic is
mocked at the requests layer.
"""

# pylint: disable=missing-function-docstring

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import Tool, ToolException

from bili.iris.tools.api_weather_gov import (
    execute_query,
    get_forecast_context_4k,
    init_weather_gov_api_tool,
)


def _points_response(forecast_url):
    resp = MagicMock()
    resp.json.return_value = {"properties": {"forecast": forecast_url}}
    return resp


def _forecast_response(periods):
    resp = MagicMock()
    resp.json.return_value = {"properties": {"periods": periods}}
    return resp


class TestGetForecastContext4k:
    """Verify the two-request forecast fetch and field trimming."""

    @patch("bili.iris.tools.api_weather_gov.requests.get")
    def test_trims_to_name_and_detailed_forecast(self, mock_get):
        period = {
            "name": "Tonight",
            "number": 1,
            "isDaytime": False,
            "startTime": "2026-01-01T18:00",
            "endTime": "2026-01-02T06:00",
            "temperature": 30,
            "temperatureUnit": "F",
            "temperatureTrend": None,
            "probabilityOfPrecipitation": {"value": 10},
            "relativeHumidity": {"value": 80},
            "dewpoint": {"value": 1},
            "windSpeed": "5 mph",
            "windDirection": "N",
            "icon": "url",
            "shortForecast": "Clear",
            "detailedForecast": "Clear skies overnight.",
        }
        mock_get.side_effect = [
            _points_response("https://api.weather.gov/forecast/1"),
            _forecast_response([period]),
        ]

        result = get_forecast_context_4k("https://api.weather.gov/points/39,-104")

        parsed = json.loads(result)
        assert parsed == [
            {"name": "Tonight", "detailedForecast": "Clear skies overnight."}
        ]
        # Two GET calls: first for points, then for the forecast URL.
        assert mock_get.call_args_list[0][0][0].endswith("points/39,-104")
        assert mock_get.call_args_list[1][0][0] == (
            "https://api.weather.gov/forecast/1"
        )

    @patch("bili.iris.tools.api_weather_gov.requests.get")
    def test_request_exception_becomes_tool_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("boom")

        with pytest.raises(ToolException, match="API_WeatherGOVTool"):
            get_forecast_context_4k("https://api.weather.gov/points/1,2")


class TestExecuteQuery:
    """Verify coordinate validation, sanitization, and dispatch."""

    @patch("bili.iris.tools.api_weather_gov.get_forecast_context_4k")
    def test_valid_coordinates_call_forecast(self, mock_forecast):
        mock_forecast.return_value = "[]"

        result = execute_query("39.7,-104.9")

        mock_forecast.assert_called_once_with(
            "https://api.weather.gov/points/39.7,-104.9"
        )
        assert result == "[]"

    @patch("bili.iris.tools.api_weather_gov.get_forecast_context_4k")
    def test_strips_unsafe_characters_before_matching(self, mock_forecast):
        mock_forecast.return_value = "[]"

        # Letters and spaces are stripped, leaving a valid lat,lon pair.
        execute_query("lat 39.7, lon -104.9")

        mock_forecast.assert_called_once_with(
            "https://api.weather.gov/points/39.7,-104.9"
        )

    @patch("bili.iris.tools.api_weather_gov.get_forecast_context_4k")
    def test_extra_commas_truncated_to_two_parts(self, mock_forecast):
        mock_forecast.return_value = "[]"

        execute_query("39.7,-104.9,500")

        mock_forecast.assert_called_once_with(
            "https://api.weather.gov/points/39.7,-104.9"
        )

    def test_invalid_query_returns_error_message(self):
        result = execute_query("not-a-coordinate-pair")
        assert result.startswith("Invalid query.")


class TestInitWeatherGovApiTool:
    """Verify tool construction."""

    def test_builds_tool(self):
        tool = init_weather_gov_api_tool("wgov", "desc")
        assert isinstance(tool, Tool)
        assert tool.name == "wgov"
        assert tool.func is execute_query
