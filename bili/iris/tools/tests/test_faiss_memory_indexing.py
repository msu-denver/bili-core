"""Tests for bili.iris.tools.faiss_memory_indexing.

Covers retriever construction, the cached index loader, path validation
security rules, and init_faiss dispatch. FAISS, the embedding loader, and
directory preprocessing are all mocked.
"""

# pylint: disable=missing-function-docstring

import os
from unittest.mock import MagicMock, patch

import pytest

from bili.iris.tools.faiss_memory_indexing import (
    ALLOWED_PREFIXES,
    create_faiss_retriever,
    init_faiss,
    load_faiss_index,
    validate_path,
)


class TestCreateFaissRetriever:
    """Verify retriever construction from documents."""

    @patch("bili.iris.tools.faiss_memory_indexing.FAISS")
    @patch("bili.iris.tools.faiss_memory_indexing.load_embedding_function")
    def test_builds_retriever_with_expected_search_kwargs(self, mock_embed, mock_faiss):
        mock_embed.return_value = "embed_fn"
        faiss_index = MagicMock()
        retriever = MagicMock()
        faiss_index.as_retriever.return_value = retriever
        mock_faiss.from_documents.return_value = faiss_index

        docs = ["d1", "d2"]
        result = create_faiss_retriever(docs)

        assert result is retriever
        mock_embed.assert_called_once_with("sentence_transformer")
        mock_faiss.from_documents.assert_called_once_with(
            documents=docs, embedding="embed_fn"
        )
        faiss_index.as_retriever.assert_called_once_with(
            search_kwargs={"k": 50, "fetch_k": 500}
        )


class TestLoadFaissIndex:
    """Verify the cached preprocessing-and-index path."""

    @patch("bili.iris.tools.faiss_memory_indexing.create_faiss_retriever")
    @patch("bili.iris.tools.faiss_memory_indexing.preprocess_directory")
    def test_preprocesses_then_builds_retriever(self, mock_prep, mock_create):
        mock_prep.return_value = ["doc"]
        mock_create.return_value = "retriever"

        result = load_faiss_index("data/memory")

        assert result == "retriever"
        mock_prep.assert_called_once_with("data/memory")
        mock_create.assert_called_once_with(["doc"])


class TestValidatePath:
    """Verify the path-validation security rules."""

    def test_allows_path_under_allowed_prefix(self):
        allowed = os.path.join(ALLOWED_PREFIXES[0], "subdir")
        assert validate_path(allowed) is True

    def test_allows_relative_data_path(self):
        assert validate_path("data/memory") is True

    def test_rejects_path_outside_allowed_locations(self):
        assert validate_path("/etc/passwd") is False

    def test_rejects_traversal_escaping_data(self):
        # Normalization resolves the traversal to a path outside data/.
        assert validate_path("data/../../etc/passwd") is False


class TestInitFaiss:
    """Verify init_faiss validation and dispatch."""

    @patch("bili.iris.tools.faiss_memory_indexing.load_faiss_index")
    def test_valid_path_returns_retriever(self, mock_load):
        mock_load.return_value = "retriever"

        result = init_faiss("data/memory")

        assert result == "retriever"
        mock_load.assert_called_once_with("data/memory")

    def test_invalid_path_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid path"):
            init_faiss("/not/allowed")

    def test_default_data_dir_is_valid(self):
        with patch(
            "bili.iris.tools.faiss_memory_indexing.load_faiss_index"
        ) as mock_load:
            mock_load.return_value = "r"
            assert init_faiss() == "r"
            mock_load.assert_called_once_with("data")
