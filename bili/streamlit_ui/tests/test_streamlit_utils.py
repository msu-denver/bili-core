"""Tests for streamlit_ui.utils.streamlit_utils conditional cache decorators."""

import importlib.util
from unittest.mock import patch

import pytest

from bili.streamlit_ui.utils.streamlit_utils import (
    conditional_cache_data,
    conditional_cache_resource,
)


def _sample():
    return "value"


# The in-Streamlit cases patch attributes of the real streamlit module, so they
# need the optional [streamlit] extra. The outside-Streamlit cases check the
# behavior without it and always run.
needs_streamlit = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="requires the [streamlit] extra (streamlit is not installed)",
)


class TestConditionalCacheResource:
    """conditional_cache_resource wraps with st.cache_resource only in Streamlit."""

    @needs_streamlit
    def test_wraps_with_cache_resource_in_streamlit_env(self):
        """In a Streamlit environment the function is cache_resource wrapped."""
        sentinel = object()
        with patch.dict("os.environ", {"STREAMLIT_SERVER_ADDRESS": "0.0.0.0"}), patch(
            "streamlit.cache_resource", return_value=sentinel
        ) as mock_cache:
            decorated = conditional_cache_resource()(_sample)
        assert decorated is sentinel
        mock_cache.assert_called_once_with(_sample)

    def test_returns_function_unchanged_outside_streamlit(self):
        """Outside Streamlit the original function is returned unmodified."""
        with patch.dict("os.environ", {}, clear=True):
            decorated = conditional_cache_resource()(_sample)
        assert decorated is _sample


class TestConditionalCacheData:
    """conditional_cache_data wraps with st.cache_data only in Streamlit."""

    @needs_streamlit
    def test_wraps_with_cache_data_in_streamlit_env(self):
        """In a Streamlit environment the function is cache_data wrapped."""
        sentinel = object()
        with patch.dict("os.environ", {"STREAMLIT_SERVER_ADDRESS": "0.0.0.0"}), patch(
            "streamlit.cache_data", return_value=sentinel
        ) as mock_cache:
            decorated = conditional_cache_data()(_sample)
        assert decorated is sentinel
        mock_cache.assert_called_once_with(_sample)

    def test_returns_function_unchanged_outside_streamlit(self):
        """Outside Streamlit the original function is returned unmodified."""
        with patch.dict("os.environ", {}, clear=True):
            decorated = conditional_cache_data()(_sample)
        assert decorated is _sample
