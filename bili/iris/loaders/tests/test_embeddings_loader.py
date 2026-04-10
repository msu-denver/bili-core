"""Tests for bili.iris.loaders.embeddings_loader module.

Covers load_embedding_function routing and each provider-specific
embedding function creator with mocked external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from bili.iris.loaders.embeddings_loader import (
    create_amazon_bedrock_embedding_function,
    create_azure_openai_embedding_function,
    create_google_vertexai_embedding_function,
    create_openai_embedding_function,
    create_sentence_transformer_embedding_function,
    load_embedding_function,
)

# ---------------------------------------------------------------------------
# load_embedding_function — routing
# ---------------------------------------------------------------------------


class TestLoadEmbeddingFunction:
    """Verify load_embedding_function dispatches correctly."""

    @patch(
        "bili.iris.loaders.embeddings_loader"
        ".create_google_vertexai_embedding_function"
    )
    def test_routes_to_vertex(self, mock_fn):
        """Verify vertex provider routes correctly."""
        mock_fn.return_value = MagicMock()
        result = load_embedding_function(
            provider="vertex", model_name="text-embedding-005"
        )
        mock_fn.assert_called_once_with(model_name="text-embedding-005")
        assert result is mock_fn.return_value

    @patch(
        "bili.iris.loaders.embeddings_loader"
        ".create_amazon_bedrock_embedding_function"
    )
    def test_routes_to_amazon(self, mock_fn):
        """Verify amazon provider routes correctly."""
        mock_fn.return_value = MagicMock()
        result = load_embedding_function(provider="amazon", model_name="titan-v2")
        mock_fn.assert_called_once_with(model_name="titan-v2")
        assert result is mock_fn.return_value

    @patch("bili.iris.loaders.embeddings_loader.create_azure_openai_embedding_function")
    def test_routes_to_azure(self, mock_fn):
        """Verify azure provider routes correctly."""
        mock_fn.return_value = MagicMock()
        result = load_embedding_function(provider="azure", model_name="embed-3-large")
        mock_fn.assert_called_once_with(model_name="embed-3-large")
        assert result is mock_fn.return_value

    @patch("bili.iris.loaders.embeddings_loader.create_openai_embedding_function")
    def test_routes_to_openai(self, mock_fn):
        """Verify openai provider routes correctly."""
        mock_fn.return_value = MagicMock()
        result = load_embedding_function(
            provider="openai", model_name="text-embedding-3-large"
        )
        mock_fn.assert_called_once_with(model_name="text-embedding-3-large")
        assert result is mock_fn.return_value

    @patch(
        "bili.iris.loaders.embeddings_loader"
        ".create_sentence_transformer_embedding_function"
    )
    def test_routes_to_sentence_transformer(self, mock_fn):
        """Verify sentence_transformer provider routes correctly."""
        mock_fn.return_value = MagicMock()
        result = load_embedding_function(
            provider="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
        )
        mock_fn.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert result is mock_fn.return_value

    def test_invalid_provider_raises_value_error(self):
        """Verify unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Invalid index name"):
            load_embedding_function(provider="unknown")

    @patch(
        "bili.iris.loaders.embeddings_loader"
        ".create_google_vertexai_embedding_function"
    )
    def test_none_model_name_passes_through(self, mock_fn):
        """Verify None model_name is forwarded to creator."""
        mock_fn.return_value = MagicMock()
        load_embedding_function(provider="vertex", model_name=None)
        mock_fn.assert_called_once_with(model_name=None)


# ---------------------------------------------------------------------------
# Individual creator functions
# ---------------------------------------------------------------------------


class TestCreateSentenceTransformerEmbeddingFunction:
    """Verify SentenceTransformer embedding function creation."""

    @patch("bili.iris.loaders.embeddings_loader.SentenceTransformerEmbeddings")
    def test_default_model(self, mock_cls):
        """Verify default model name is used."""
        mock_cls.return_value = MagicMock()
        result = create_sentence_transformer_embedding_function()
        mock_cls.assert_called_once_with(model_name="bert-large-nli-mean-tokens")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.embeddings_loader.SentenceTransformerEmbeddings")
    def test_custom_model(self, mock_cls):
        """Verify custom model name is passed through."""
        mock_cls.return_value = MagicMock()
        create_sentence_transformer_embedding_function(model_name="all-MiniLM-L6-v2")
        mock_cls.assert_called_once_with(model_name="all-MiniLM-L6-v2")


class TestCreateAzureOpenaiEmbeddingFunction:
    """Verify Azure OpenAI embedding function creation."""

    @patch("bili.iris.loaders.embeddings_loader.AzureOpenAIEmbeddings")
    def test_default_model(self, mock_cls):
        """Verify default model name for Azure embeddings."""
        mock_cls.return_value = MagicMock()
        result = create_azure_openai_embedding_function()
        mock_cls.assert_called_once_with(model="azure_text-embedding-3-large")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.embeddings_loader.AzureOpenAIEmbeddings")
    def test_custom_model(self, mock_cls):
        """Verify custom model name for Azure embeddings."""
        mock_cls.return_value = MagicMock()
        create_azure_openai_embedding_function(model_name="custom-embed")
        mock_cls.assert_called_once_with(model="custom-embed")


class TestCreateOpenaiEmbeddingFunction:
    """Verify OpenAI embedding function creation."""

    @patch("bili.iris.loaders.embeddings_loader.OpenAIEmbeddings")
    def test_default_model(self, mock_cls):
        """Verify default model name for OpenAI embeddings."""
        mock_cls.return_value = MagicMock()
        result = create_openai_embedding_function()
        mock_cls.assert_called_once_with(model="text-embedding-3-large")
        assert result is mock_cls.return_value


class TestCreateAmazonBedrockEmbeddingFunction:
    """Verify Amazon Bedrock embedding function creation."""

    @patch("bili.iris.loaders.embeddings_loader.BedrockEmbeddings")
    def test_default_model(self, mock_cls):
        """Verify default model name for Bedrock embeddings."""
        mock_cls.return_value = MagicMock()
        result = create_amazon_bedrock_embedding_function()
        mock_cls.assert_called_once_with(model_id="amazon_titan-embed-text-v2")
        assert result is mock_cls.return_value


class TestCreateGoogleVertexaiEmbeddingFunction:
    """Verify Google Vertex AI embedding function creation."""

    @patch("bili.iris.loaders.embeddings_loader.VertexAIEmbeddings")
    def test_default_model(self, mock_cls):
        """Verify default model name for Vertex embeddings."""
        mock_cls.return_value = MagicMock()
        result = create_google_vertexai_embedding_function()
        mock_cls.assert_called_once_with(model="azure_text-embedding-3-large")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.embeddings_loader.VertexAIEmbeddings")
    def test_custom_model(self, mock_cls):
        """Verify custom model name for Vertex embeddings."""
        mock_cls.return_value = MagicMock()
        create_google_vertexai_embedding_function(model_name="text-embedding-005")
        mock_cls.assert_called_once_with(model="text-embedding-005")
