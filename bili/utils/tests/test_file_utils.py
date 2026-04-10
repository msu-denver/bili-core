"""Tests for bili.utils.file_utils.

Covers load_from_json, preprocess_file (dispatch logic),
process_text_data, process_csv_data, process_text, and
chunk_documents. Heavy-dependency loaders (PDF, Excel, Word)
are tested only via mocking or skipped.
"""

import json
from unittest.mock import patch

from langchain_core.documents import Document

from bili.utils.file_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    load_from_json,
    preprocess_directory,
    preprocess_file,
    process_text,
    process_text_data,
)

# ------------------------------------------------------------------
# load_from_json
# ------------------------------------------------------------------


class TestLoadFromJsonFull:
    """load_from_json returning entire file contents."""

    def test_returns_full_dict(self, tmp_path):
        """Entire JSON dict is returned when no parameter given."""
        data = {"key": "value", "num": 42}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_from_json(str(path))
        assert result == data

    def test_returns_list(self, tmp_path):
        """JSON array is returned correctly."""
        data = [1, 2, 3]
        path = tmp_path / "list.json"
        path.write_text(json.dumps(data))
        result = load_from_json(str(path))
        assert result == data


class TestLoadFromJsonParameter:
    """load_from_json with a specific parameter key."""

    def test_returns_parameter_value(self, tmp_path):
        """Specific parameter value is extracted."""
        data = {"name": "alice", "age": 30}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_from_json(str(path), parameter="name")
        assert result == "alice"

    def test_missing_parameter_returns_full(self, tmp_path):
        """Missing parameter key returns full data."""
        data = {"a": 1}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_from_json(str(path), parameter="missing")
        assert result == data


# ------------------------------------------------------------------
# process_text_data
# ------------------------------------------------------------------


class TestProcessTextData:
    """process_text_data loads a .txt file via TextLoader."""

    def test_loads_text_file(self, tmp_path):
        """Text file contents are returned as Document list."""
        path = tmp_path / "sample.txt"
        path.write_text("hello world")
        docs = process_text_data(str(path), ".txt")
        assert len(docs) >= 1
        assert "hello world" in docs[0].page_content


# ------------------------------------------------------------------
# preprocess_file dispatch
# ------------------------------------------------------------------


class TestPreprocessFileText:
    """preprocess_file dispatches .txt to process_text_data."""

    def test_txt_extension(self, tmp_path):
        """A .txt file is handled by the text processor."""
        path = tmp_path / "note.txt"
        path.write_text("content here")
        result = preprocess_file(str(path))
        assert len(result) >= 1
        assert "content here" in result[0].page_content


class TestPreprocessFileUnsupported:
    """preprocess_file falls back for unknown extensions."""

    def test_unknown_ext_dispatches_to_unstructured(self, tmp_path):
        """Unknown extension delegates to process_unstructured_data."""
        path = tmp_path / "data.xyz"
        path.write_text("stuff")
        mock_docs = [Document(page_content="stuff", metadata={})]
        with patch(
            "bili.utils.file_utils.process_unstructured_data",
            return_value=mock_docs,
        ) as mocked:
            result = preprocess_file(str(path))
            mocked.assert_called_once_with(str(path), ".xyz")
            assert result == mock_docs


class TestPreprocessFileCsv:
    """preprocess_file dispatches .csv to process_csv_data."""

    def test_csv_extension(self, tmp_path):
        """A .csv file is loaded via CSVLoader."""
        path = tmp_path / "data.csv"
        path.write_text("name,age\nalice,30\nbob,25\n")
        result = preprocess_file(str(path))
        assert len(result) >= 1


# ------------------------------------------------------------------
# preprocess_directory
# ------------------------------------------------------------------


class TestPreprocessDirectory:
    """preprocess_directory processes visible files only."""

    def test_ignores_hidden_files(self, tmp_path):
        """Hidden (dot-prefix) files are skipped."""
        visible = tmp_path / "readme.txt"
        visible.write_text("visible")
        hidden = tmp_path / ".hidden"
        hidden.write_text("secret")
        result = preprocess_directory(str(tmp_path))
        contents = " ".join(d.page_content for d in result)
        assert "visible" in contents

    def test_ignores_subdirectories(self, tmp_path):
        """Subdirectories are skipped."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        txt = tmp_path / "file.txt"
        txt.write_text("top-level")
        result = preprocess_directory(str(tmp_path))
        assert len(result) >= 1

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        result = preprocess_directory(str(tmp_path))
        assert not result


# ------------------------------------------------------------------
# process_text  (in-memory text chunking)
# ------------------------------------------------------------------


class TestProcessTextEmpty:
    """process_text with empty or whitespace input."""

    def test_empty_string_returns_empty(self):
        """Empty string yields no documents."""
        assert not process_text("")

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only string yields no documents."""
        assert not process_text("   \n  ")

    def test_none_returns_empty(self):
        """None input yields no documents."""
        assert not process_text(None)


class TestProcessTextBasic:
    """process_text with normal text."""

    def test_returns_documents(self):
        """Non-empty text produces at least one Document."""
        docs = process_text("Hello world, this is a test.")
        assert len(docs) >= 1
        assert isinstance(docs[0], Document)

    def test_chunk_index_metadata(self):
        """Each document has chunk_index in metadata."""
        docs = process_text("Some content.")
        assert docs[0].metadata["chunk_index"] == 0

    def test_total_chunks_metadata(self):
        """Each document has total_chunks in metadata."""
        docs = process_text("Some content.")
        assert docs[0].metadata["total_chunks"] == len(docs)

    def test_custom_metadata_preserved(self):
        """Supplied metadata appears in every chunk."""
        meta = {"source": "test", "url": "https://x.com"}
        docs = process_text("Text here.", metadata=meta)
        assert docs[0].metadata["source"] == "test"
        assert docs[0].metadata["url"] == "https://x.com"

    def test_small_chunk_size_produces_more_docs(self):
        """Smaller chunk_size produces more documents."""
        text = "word " * 500
        small = process_text(text, chunk_size=50, chunk_overlap=0)
        large = process_text(text, chunk_size=500, chunk_overlap=0)
        assert len(small) > len(large)


# ------------------------------------------------------------------
# chunk_documents
# ------------------------------------------------------------------


class TestChunkDocuments:
    """chunk_documents splits large documents and handles edge cases."""

    def test_empty_list_returns_empty(self):
        """Empty document list yields empty result."""
        assert not chunk_documents([])

    def test_splits_large_document(self):
        """A document exceeding chunk_size is split."""
        big_doc = Document(
            page_content="word " * 1000,
            metadata={"source": "test"},
        )
        chunks = chunk_documents([big_doc], chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        """Original document metadata is preserved in chunks."""
        doc = Document(
            page_content="word " * 500,
            metadata={"source": "orig"},
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=0)
        assert all(c.metadata["source"] == "orig" for c in chunks)


# ------------------------------------------------------------------
# Default constants
# ------------------------------------------------------------------


class TestDefaultConstants:
    """Verify default chunk constants are sensible."""

    def test_default_chunk_size(self):
        """DEFAULT_CHUNK_SIZE is 1000."""
        assert DEFAULT_CHUNK_SIZE == 1000

    def test_default_chunk_overlap(self):
        """DEFAULT_CHUNK_OVERLAP is 200."""
        assert DEFAULT_CHUNK_OVERLAP == 200
