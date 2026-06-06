"""Tests for bili.iris.tools.api_serp.

Covers query sanitization, response filtering, the execute_query HTTP path,
and tool initialization including the missing-key guard.
"""

# pylint: disable=missing-function-docstring

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import Tool

from bili.iris.tools import api_serp
from bili.iris.tools.api_serp import (
    execute_query,
    filter_serp_results,
    init_serp_api_tool,
    sanitize_serp_query,
)


class TestSanitizeSerpQuery:
    """Verify SERP query sanitization."""

    def test_empty_query_returns_empty_string(self):
        assert sanitize_serp_query("") == ""

    def test_strips_injection_characters_and_control_chars(self):
        # Angle brackets and quotes are removed; a control char (bell) too.
        assert sanitize_serp_query("a<b>\"c'\x07d") == "abcd"

    def test_normalizes_whitespace_and_url_encodes(self):
        assert sanitize_serp_query("hello   world") == "hello%20world"

    def test_truncates_to_500_characters_before_encoding(self):
        raw = "a" * 600
        # 500 'a' characters, none needing encoding.
        assert sanitize_serp_query(raw) == "a" * 500


class TestFilterSerpResults:
    """Verify the token-reducing result filter."""

    def test_extracts_query_and_top_results(self):
        data = {
            "search_parameters": {"q": "weather"},
            "organic_results": [
                {
                    "title": "T1",
                    "link": "http://a",
                    "snippet": "s1",
                    "source": "src1",
                },
                {"title": "T2", "link": "http://b", "snippet": "s2"},
            ],
        }
        filtered = filter_serp_results(data, max_results=1)

        assert filtered["query"] == "weather"
        assert len(filtered["results"]) == 1
        assert filtered["results"][0] == {
            "title": "T1",
            "link": "http://a",
            "snippet": "s1",
            "source": "src1",
        }

    def test_truncates_long_snippet_to_300_chars(self):
        data = {"organic_results": [{"title": "x", "snippet": "y" * 400}]}
        filtered = filter_serp_results(data)
        assert len(filtered["results"][0]["snippet"]) == 300

    def test_includes_knowledge_panel_when_present(self):
        data = {
            "organic_results": [],
            "knowledge_graph": {"title": "Denver", "description": "A city."},
        }
        filtered = filter_serp_results(data)
        assert filtered["knowledge_panel"] == {
            "title": "Denver",
            "description": "A city.",
        }

    def test_includes_answer_box_with_snippet_fallback(self):
        data = {
            "organic_results": [],
            "answer_box": {"title": "Ans", "snippet": "42"},
        }
        filtered = filter_serp_results(data)
        # answer falls back to snippet when no explicit answer key.
        assert filtered["answer_box"] == {"title": "Ans", "answer": "42"}

    def test_omits_optional_sections_when_absent(self):
        filtered = filter_serp_results({"organic_results": []})
        assert "knowledge_panel" not in filtered
        assert "answer_box" not in filtered


class TestExecuteQuery:
    """Verify the SERP HTTP execution path."""

    @patch.dict("os.environ", {"SERP_API_KEY": "sk"}, clear=False)
    @patch("bili.iris.tools.api_serp.requests.get")
    def test_builds_url_and_returns_filtered_json(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = {
            "search_parameters": {"q": "cats"},
            "organic_results": [{"title": "Cats", "link": "http://c"}],
        }
        mock_get.return_value = resp

        result = execute_query("cats")

        called_url = mock_get.call_args[0][0]
        assert "serpapi.com/search.json?q=cats" in called_url
        assert "api_key=sk" in called_url
        assert mock_get.call_args[1]["timeout"] == 10
        parsed = json.loads(result)
        assert parsed["query"] == "cats"
        assert parsed["results"][0]["title"] == "Cats"


class TestInitSerpApiTool:
    """Verify tool construction and missing-key guard."""

    @patch.dict("os.environ", {"SERP_API_KEY": "sk"}, clear=False)
    def test_builds_tool(self):
        tool = init_serp_api_tool("serp", "desc")
        assert isinstance(tool, Tool)
        assert tool.name == "serp"
        assert tool.func is execute_query

    def test_missing_key_raises_value_error(self):
        env = {k: v for k, v in api_serp.os.environ.items()}
        env.pop("SERP_API_KEY", None)
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="SERP_API_KEY"):
                init_serp_api_tool("serp", "desc")
