"""Tests for bili.utils.opensearch_utils.load_opensearch_vector_search."""

from unittest.mock import MagicMock, patch

import pytest

from bili.utils.opensearch_utils import load_opensearch_vector_search

_MODULE = "bili.utils.opensearch_utils"


def _patches():
    """Return the standard set of external-dependency patchers as a context tuple."""
    return (
        patch(f"{_MODULE}.boto3"),
        patch(f"{_MODULE}.AWS4Auth"),
        patch(f"{_MODULE}.OpenSearchVectorSearch"),
    )


class TestLoadOpenSearchVectorSearch:
    """The loader configures credentials and builds an OpenSearchVectorSearch."""

    def test_local_environment_uses_test_credentials(self):
        """LOCALSTACK_HOSTNAME triggers the localstack credential path without SSL."""
        env = {
            "LOCALSTACK_HOSTNAME": "localhost",
            "OPENSEARCH_URL": "http://opensearch:9200",
        }
        p_boto, p_auth, p_oss = _patches()
        with patch.dict(
            "os.environ", env, clear=True
        ), p_boto as boto3_mock, p_auth, p_oss as oss_mock:
            embed = MagicMock()
            result = load_opensearch_vector_search(embed, "my-index")

        boto3_mock.Session.assert_called_once()
        # Local path passes explicit test credentials to the session.
        assert "aws_access_key_id" in boto3_mock.Session.call_args.kwargs
        kwargs = oss_mock.call_args.kwargs
        assert kwargs["use_ssl"] is False
        assert kwargs["verify_certs"] is False
        assert kwargs["index_name"] == "my-index"
        assert kwargs["opensearch_url"] == "http://opensearch:9200"
        assert result is oss_mock.return_value

    def test_aws_environment_uses_default_credential_chain(self):
        """Without LOCALSTACK the default boto3 session and SSL are used."""
        env = {
            "OPENSEARCH_URL": "https://opensearch.aws:443",
            "AWS_REGION": "us-west-2",
        }
        p_boto, p_auth, p_oss = _patches()
        with patch.dict(
            "os.environ", env, clear=True
        ), p_boto as boto3_mock, p_auth, p_oss as oss_mock:
            load_opensearch_vector_search(MagicMock(), "idx")

        # Default credential chain: Session() called with no keyword arguments.
        boto3_mock.Session.assert_called_once_with()
        assert oss_mock.call_args.kwargs["use_ssl"] is True
        assert oss_mock.call_args.kwargs["verify_certs"] is True

    def test_missing_opensearch_url_raises(self):
        """A missing OPENSEARCH_URL raises a descriptive ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENSEARCH_URL"):
                load_opensearch_vector_search(MagicMock(), "idx")

    def test_missing_aws_region_raises_in_aws_environment(self):
        """A non-local environment without AWS_REGION raises a ValueError."""
        env = {"OPENSEARCH_URL": "https://opensearch.aws:443"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="AWS_REGION"):
                load_opensearch_vector_search(MagicMock(), "idx")
