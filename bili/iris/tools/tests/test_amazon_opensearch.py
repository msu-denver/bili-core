"""Tests for bili.iris.tools.amazon_opensearch.

Covers the query-builder closure (similarity search dispatch, list joining,
and error swallowing) and tool initialization. The OpenSearch vector store is
mocked at load_opensearch_vector_search.
"""

# pylint: disable=missing-function-docstring

from unittest.mock import MagicMock, patch

from langchain_core.tools import Tool

from bili.iris.tools.amazon_opensearch import (
    build_query_opensearch,
    init_amazon_opensearch,
)


class TestBuildQueryOpensearch:
    """Verify the closure returned by build_query_opensearch."""

    @patch("bili.iris.tools.amazon_opensearch.load_opensearch_vector_search")
    def test_joins_list_results_into_string(self, mock_load):
        docsearch = MagicMock()
        docsearch.similarity_search.return_value = ["doc one", "doc two"]
        mock_load.return_value = docsearch
        embed = MagicMock()

        query_fn = build_query_opensearch(embed, "idx", k=3, score_threshold=0.2)
        result = query_fn("hello")

        assert result == "doc one doc two"
        mock_load.assert_called_once_with(embed, "idx")
        docsearch.similarity_search.assert_called_once_with(
            "hello", k=3, score_threshold=0.2
        )

    @patch("bili.iris.tools.amazon_opensearch.load_opensearch_vector_search")
    def test_non_list_result_is_stringified(self, mock_load):
        docsearch = MagicMock()
        docsearch.similarity_search.return_value = 12345
        mock_load.return_value = docsearch

        query_fn = build_query_opensearch(MagicMock(), "idx")
        assert query_fn("q") == "12345"

    @patch("bili.iris.tools.amazon_opensearch.load_opensearch_vector_search")
    def test_search_error_returns_empty_string(self, mock_load):
        docsearch = MagicMock()
        docsearch.similarity_search.side_effect = RuntimeError("boom")
        mock_load.return_value = docsearch

        query_fn = build_query_opensearch(MagicMock(), "idx")
        assert query_fn("q") == ""


class TestInitAmazonOpensearch:
    """Verify tool construction."""

    @patch("bili.iris.tools.amazon_opensearch.load_opensearch_vector_search")
    def test_builds_tool_with_query_function(self, mock_load):
        mock_load.return_value = MagicMock()

        tool = init_amazon_opensearch("os_tool", "desc", MagicMock(), "my_index")

        assert isinstance(tool, Tool)
        assert tool.name == "os_tool"
        assert callable(tool.func)
        mock_load.assert_called_once()
