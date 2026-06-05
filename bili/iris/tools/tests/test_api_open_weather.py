"""Tests for bili.iris.tools.api_open_weather.

Covers input sanitization, geocoding by city and ZIP, weather retrieval,
the execute_query dispatch logic, and tool initialization. All HTTP traffic
is mocked at the requests layer; assertions check request URLs, parsed
fields, and error handling.
"""

# pylint: disable=missing-function-docstring

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import Tool

from bili.iris.tools import api_open_weather
from bili.iris.tools.api_open_weather import (
    execute_query,
    get_geocode,
    get_geocode_from_zip,
    get_weather,
    init_weather_api_tool,
    sanitize_input,
)


class TestSanitizeInput:
    """Verify input sanitization and URL encoding."""

    def test_strips_unsafe_characters_and_collapses_whitespace(self):
        # The comma is removed as unsafe; the newline and the run of spaces
        # collapse to single spaces, each URL-encoded to %20.
        assert sanitize_input("New\nYork,  CO") == "New%20York%20CO"

    def test_preserves_hyphen_and_alphanumerics(self):
        assert sanitize_input("Winston-Salem") == "Winston-Salem"

    def test_trims_leading_and_trailing_whitespace(self):
        assert sanitize_input("  Denver  ") == "Denver"


class TestGetGeocode:
    """Verify city geocoding against the OpenWeather direct API."""

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k123"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_returns_lat_lon_from_first_result(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [{"lat": 39.7, "lon": -104.9}]
        mock_get.return_value = resp

        lat, lon = get_geocode("Denver", "CO", "US")

        assert (lat, lon) == (39.7, -104.9)
        called_url = mock_get.call_args[0][0]
        assert "geo/1.0/direct?q=Denver,CO,US" in called_url
        assert "appid=k123" in called_url
        assert mock_get.call_args[1]["timeout"] == 10
        resp.raise_for_status.assert_called_once()

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k123"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_returns_none_pair_for_empty_response(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = []
        mock_get.return_value = resp

        assert get_geocode("Nowhere") == (None, None)

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k123"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_reraises_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("down")

        with pytest.raises(requests.exceptions.RequestException, match="down"):
            get_geocode("Denver")


class TestGetGeocodeFromZip:
    """Verify ZIP geocoding against the OpenWeather zip API."""

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "zk"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_returns_lat_lon_from_zip(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {"lat": 40.0, "lon": -105.0}
        mock_get.return_value = resp

        lat, lon = get_geocode_from_zip("80202")

        assert (lat, lon) == (40.0, -105.0)
        called_url = mock_get.call_args[0][0]
        assert "geo/1.0/zip?zip=80202,US" in called_url
        assert "appid=zk" in called_url

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "zk"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_returns_none_pair_on_request_exception(self, mock_get):
        # Unlike get_geocode, the ZIP variant swallows the error and
        # returns (None, None) rather than re-raising.
        mock_get.side_effect = requests.exceptions.RequestException("oops")

        assert get_geocode_from_zip("80202") == (None, None)


class TestGetWeather:
    """Verify current-weather retrieval."""

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "wk"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_returns_parsed_json(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {"weather": [{"main": "Clear"}]}
        mock_get.return_value = resp

        data = get_weather(39.7, -104.9)

        assert data == {"weather": [{"main": "Clear"}]}
        called_url = mock_get.call_args[0][0]
        assert "data/2.5/weather?lat=39.7&lon=-104.9" in called_url
        assert "appid=wk" in called_url

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "wk"}, clear=False)
    @patch("bili.iris.tools.api_open_weather.requests.get")
    def test_reraises_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("net")

        with pytest.raises(requests.exceptions.RequestException, match="net"):
            get_weather(1.0, 2.0)


class TestExecuteQuery:
    """Verify the query-dispatch and error-message paths."""

    @patch("bili.iris.tools.api_open_weather.get_weather")
    @patch("bili.iris.tools.api_open_weather.get_geocode_from_zip")
    def test_zip_code_path(self, mock_zip, mock_weather):
        mock_zip.return_value = (40.0, -105.0)
        mock_weather.return_value = {"main": "Clouds"}

        result = execute_query("80202")

        mock_zip.assert_called_once_with("80202")
        mock_weather.assert_called_once_with(40.0, -105.0)
        assert result == "{'main': 'Clouds'}"

    @patch("bili.iris.tools.api_open_weather.get_weather")
    @patch("bili.iris.tools.api_open_weather.get_geocode")
    def test_city_comma_state_path(self, mock_geo, mock_weather):
        mock_geo.return_value = (39.7, -104.9)
        mock_weather.return_value = {"main": "Sun"}

        result = execute_query("Denver, CO")

        # City and state are sanitized before geocoding.
        mock_geo.assert_called_once_with("Denver", "CO")
        assert result == "{'main': 'Sun'}"

    @patch("bili.iris.tools.api_open_weather.get_weather")
    @patch("bili.iris.tools.api_open_weather.get_geocode")
    def test_city_space_state_path(self, mock_geo, mock_weather):
        mock_geo.return_value = (1.0, 2.0)
        mock_weather.return_value = {"main": "Rain"}

        execute_query("Boulder CO")

        mock_geo.assert_called_once_with("Boulder", "CO")

    @patch("bili.iris.tools.api_open_weather.get_weather")
    @patch("bili.iris.tools.api_open_weather.get_geocode")
    def test_city_only_defaults_state_to_co(self, mock_geo, mock_weather):
        mock_geo.return_value = (1.0, 2.0)
        mock_weather.return_value = {"main": "Snow"}

        execute_query("Aspen")

        mock_geo.assert_called_once_with("Aspen", "CO")

    @patch("bili.iris.tools.api_open_weather.get_geocode")
    def test_location_not_found_returns_message(self, mock_geo):
        mock_geo.return_value = (None, None)

        assert execute_query("Atlantis") == "Could not find the location."

    @patch("bili.iris.tools.api_open_weather.get_weather")
    @patch("bili.iris.tools.api_open_weather.get_geocode")
    def test_weather_none_returns_failure_message(self, mock_geo, mock_weather):
        mock_geo.return_value = (1.0, 2.0)
        mock_weather.return_value = None

        assert execute_query("Denver") == "Failed to retrieve weather data."


class TestInitWeatherApiTool:
    """Verify tool construction and missing-key guard."""

    @patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "x"}, clear=False)
    def test_builds_tool(self):
        tool = init_weather_api_tool("ow", "desc")
        assert isinstance(tool, Tool)
        assert tool.name == "ow"
        assert tool.func is execute_query

    def test_missing_key_raises_value_error(self):
        env = {k: v for k, v in api_open_weather.os.environ.items()}
        env.pop("OPENWEATHERMAP_API_KEY", None)
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="OPENWEATHERMAP_API_KEY"):
                init_weather_api_tool("ow", "desc")
