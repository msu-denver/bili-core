"""Tests for the ``required_input_modalities`` precondition on ``load_model``.

This is the seam that turns a per-model capability record into a refusal at
*selection*.  Without it, a caller that intends to send an image builds the
message, hands it to a text-only model, and finds out at the provider call --
or, worse, does not find out at all because the provider ignores the part.

Backwards compatibility is a load-bearing property here, so the kwarg's
absence is pinned as hard as its presence: a call that does not mention
modalities must reach the loader with byte-identical kwargs.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from bili.iris.loaders.llm_loader import load_model
from bili.iris.providers.modality import UnsupportedInputModalityError


class TestRequiredInputModalitiesIsAPrecondition:
    """The kwarg gates the load and never reaches the provider."""

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_supported_modality_loads_the_model(self, mock_loader):
        """Supported modality loads the model."""
        mock_loader.return_value = MagicMock()
        result = load_model(
            "remote_openai",
            model_name="gpt-4o",
            required_input_modalities=["image"],
        )
        assert result is not None
        mock_loader.assert_called_once()

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_the_kwarg_is_not_forwarded_to_the_loader(self, mock_loader):
        """It is a precondition on the selection, not a provider parameter;
        forwarding it would raise TypeError in every loader."""
        mock_loader.return_value = MagicMock()
        load_model(
            "remote_openai",
            model_name="gpt-4o",
            required_input_modalities=["image"],
        )
        assert "required_input_modalities" not in mock_loader.call_args.kwargs

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_unsupported_modality_refuses_before_dispatch(self, mock_loader):
        """The refusal happens at selection: the loader is never called, so no
        client is constructed and no credential is used."""
        with pytest.raises(UnsupportedInputModalityError):
            load_model(
                "remote_openai",
                model_name="gpt-35-turbo",
                required_input_modalities=["image"],
            )
        mock_loader.assert_not_called()

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_a_bare_string_is_accepted(self, mock_loader):
        """A bare string is accepted."""
        mock_loader.return_value = MagicMock()
        load_model(
            "remote_openai",
            model_name="gpt-4o",
            required_input_modalities="image",
        )
        mock_loader.assert_called_once()

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_undeclared_model_warns_and_loads(self, mock_loader, caplog):
        """Undeclared model warns and loads."""
        mock_loader.return_value = MagicMock()
        with caplog.at_level(logging.WARNING, logger="bili.iris.providers.modality"):
            load_model(
                "remote_openai",
                model_name="some-unlisted-model",
                required_input_modalities=["image"],
            )
        assert "declares no input modalities" in caplog.text
        mock_loader.assert_called_once()

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_refusal_precedes_the_structured_output_check(self, mock_loader):
        """Both are fail-fast preconditions; either one refusing means no
        provider client is built."""
        with pytest.raises(ValueError):
            load_model(
                "remote_openai",
                model_name="gpt-35-turbo",
                required_input_modalities=["image"],
                structured_output_schema={"type": "object"},
            )
        mock_loader.assert_not_called()


class TestExistingCallersAreUnchanged:
    """Omitting the kwarg leaves every existing call byte-for-byte the same."""

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_kwargs_are_identical_without_the_new_kwarg(self, mock_loader):
        """Kwargs are identical without the new kwarg."""
        mock_loader.return_value = MagicMock()
        load_model("remote_openai", model_name="gpt-4o", max_tokens=100)
        assert mock_loader.call_args.kwargs == {
            "model_name": "gpt-4o",
            "max_tokens": 100,
        }

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    @pytest.mark.parametrize("empty", [None, [], ()])
    def test_an_empty_requirement_runs_no_check(self, mock_loader, empty):
        """A falsy value must not resolve to 'require nothing successfully' via
        a warning storm, nor refuse; it is simply not a request."""
        mock_loader.return_value = MagicMock()
        load_model(
            "remote_openai",
            model_name="gpt-35-turbo",
            required_input_modalities=empty,
        )
        mock_loader.assert_called_once()
        assert "required_input_modalities" not in mock_loader.call_args.kwargs
