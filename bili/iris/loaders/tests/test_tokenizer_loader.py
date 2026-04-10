"""Tests for bili.iris.loaders.tokenizer_loader module.

Covers load_huggingface_tokenizer with mocked HuggingFace
AutoTokenizer dependency.
"""

from unittest.mock import MagicMock, patch

from bili.iris.loaders.tokenizer_loader import load_huggingface_tokenizer

# ---------------------------------------------------------------------------
# load_huggingface_tokenizer
# ---------------------------------------------------------------------------


class TestLoadHuggingfaceTokenizer:
    """Verify HuggingFace tokenizer loading and configuration."""

    @patch("bili.iris.loaders.tokenizer_loader.AutoTokenizer")
    def test_calls_from_pretrained_with_fast(self, mock_auto):
        """Verify from_pretrained is called with use_fast=True."""
        mock_tokenizer = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        load_huggingface_tokenizer(model_name="gpt2")
        mock_auto.from_pretrained.assert_called_once_with("gpt2", use_fast=True)

    @patch("bili.iris.loaders.tokenizer_loader.AutoTokenizer")
    def test_sets_model_max_length(self, mock_auto):
        """Verify model_max_length is set to 5120."""
        mock_tokenizer = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_huggingface_tokenizer(model_name="gpt2")
        assert result.model_max_length == 5120

    @patch("bili.iris.loaders.tokenizer_loader.AutoTokenizer")
    def test_sets_truncation(self, mock_auto):
        """Verify truncation is set to True."""
        mock_tokenizer = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_huggingface_tokenizer(model_name="gpt2")
        assert result.truncation is True

    @patch("bili.iris.loaders.tokenizer_loader.AutoTokenizer")
    def test_returns_tokenizer_object(self, mock_auto):
        """Verify the function returns the tokenizer object."""
        mock_tokenizer = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_huggingface_tokenizer(model_name="test-model")
        assert result is mock_tokenizer

    @patch("bili.iris.loaders.tokenizer_loader.AutoTokenizer")
    def test_different_model_names(self, mock_auto):
        """Verify different model names are passed correctly."""
        mock_auto.from_pretrained.return_value = MagicMock()
        load_huggingface_tokenizer(model_name="meta-llama/Llama-2-7b")
        mock_auto.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-2-7b", use_fast=True
        )
