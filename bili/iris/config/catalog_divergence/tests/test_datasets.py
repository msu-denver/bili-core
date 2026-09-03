"""Tests for fetching and parsing the two community capability datasets.

The parsers are asserted against the RECORDED payloads, so what they measure
is agreement with what the upstream actually served rather than agreement with
a shape this repository invented.

Every degradation branch is executed rather than excluded, because the one
outcome this must never produce is a silent empty dataset: an empty merge
reports "no divergence", which is indistinguishable from a clean catalog.
"""

import json
import urllib.error
from pathlib import Path

import pytest

from bili.iris.config.catalog_divergence.datasets import (
    COMPARABLE_MODALITIES,
    LITELLM,
    MODELS_DEV,
    REASON_AUTH,
    REASON_HTTP,
    REASON_MALFORMED,
    REASON_NETWORK,
    Dataset,
    Unavailable,
    fetch_json,
    load_litellm,
    load_models_dev,
    parse_litellm,
    parse_models_dev,
    read_json_file,
)


class _FakeResponse:
    """Minimal context-manager stand-in for a urlopen result."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        """Return the canned payload.

        :returns: The bytes the fake response carries.
        :rtype: bytes
        """
        return self._payload


class TestFetchJson:
    """Every network failure must become a reason, never an exception."""

    def test_a_successful_fetch_returns_the_decoded_document(self, monkeypatch):
        """A 200 with valid JSON decodes to the document."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeResponse(b'{"hello": "world"}'),
        )
        assert fetch_json("https://example.invalid/x", MODELS_DEV) == {"hello": "world"}

    @pytest.mark.parametrize("code", [401, 403])
    def test_an_auth_failure_is_reported_as_auth(self, monkeypatch, code):
        """A credential rejection is distinguishable from any other failure."""

        def _raise(*_args, **_kwargs):
            raise urllib.error.HTTPError(
                "https://example.invalid/x", code, "no", {}, None
            )

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = fetch_json("https://example.invalid/x", MODELS_DEV)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_AUTH
        assert result.source == MODELS_DEV
        assert str(code) in result.detail

    def test_a_non_auth_http_failure_is_reported_as_http(self, monkeypatch):
        """A server-side failure is its own reason, not an auth failure."""

        def _raise(*_args, **_kwargs):
            raise urllib.error.HTTPError(
                "https://example.invalid/x", 503, "down", {}, None
            )

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = fetch_json("https://example.invalid/x", LITELLM)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_HTTP

    def test_a_transport_failure_is_reported_as_network(self, monkeypatch):
        """An unreachable host is a network reason, not an empty dataset."""

        def _raise(*_args, **_kwargs):
            raise urllib.error.URLError("name resolution failed")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = fetch_json("https://example.invalid/x", MODELS_DEV)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_NETWORK

    def test_a_socket_failure_is_reported_as_network(self, monkeypatch):
        """A timeout surfaces as an OSError and must not escape."""

        def _raise(*_args, **_kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = fetch_json("https://example.invalid/x", MODELS_DEV)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_NETWORK

    def test_a_truncated_payload_is_reported_as_malformed(self, monkeypatch):
        """A half-written document is malformed, not an empty dataset."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeResponse(b'{"openai": {"models": {"gpt-4"'),
        )
        result = fetch_json("https://example.invalid/x", MODELS_DEV)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED

    def test_undecodable_bytes_are_reported_as_malformed(self, monkeypatch):
        """Bytes that are not text at all must not raise."""
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _FakeResponse(b"\xff\xfe\x00")
        )
        result = fetch_json("https://example.invalid/x", LITELLM)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED


class TestReadJsonFile:
    """The local-file path degrades the same way the network path does."""

    def test_a_readable_file_decodes(self, tmp_path):
        """A well-formed file decodes to its document."""
        path = tmp_path / "d.json"
        path.write_text('{"a": 1}', encoding="utf-8")
        assert read_json_file(path, MODELS_DEV) == {"a": 1}

    def test_a_missing_file_is_unavailable(self, tmp_path):
        """A path that does not exist is a reason, not an exception."""
        result = read_json_file(tmp_path / "nope.json", MODELS_DEV)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_NETWORK

    def test_a_truncated_file_is_malformed(self, tmp_path):
        """A half-written file is malformed, not an empty dataset."""
        path = tmp_path / "d.json"
        path.write_text('{"a": ', encoding="utf-8")
        result = read_json_file(path, LITELLM)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED


class TestParseModelsDev:
    """The provider-keyed dataset, against its recorded payload."""

    def test_the_recorded_payload_parses(self, models_dev_dataset):
        """The capture parses into a dataset with records."""
        assert isinstance(models_dev_dataset, Dataset)
        assert models_dev_dataset.source == MODELS_DEV
        assert models_dev_dataset.model_count > 0

    def test_records_are_indexed_provider_first(self, models_dev_dataset):
        """A record is reachable only under its own provider."""
        assert models_dev_dataset.lookup("openai", "gpt-4") is not None
        assert models_dev_dataset.lookup("anthropic", "gpt-4") is None

    def test_a_missing_provider_looks_up_to_nothing(self, models_dev_dataset):
        """An unknown provider resolves to nothing rather than raising."""
        assert models_dev_dataset.lookup("no-such-provider", "gpt-4") is None

    def test_an_explicit_modality_array_is_marked_explicit(self, models_dev_dataset):
        """This dataset states modalities, which is what licenses a denial."""
        record = models_dev_dataset.lookup("openai", "gpt-4")
        assert record.states_input_modalities is True
        assert record.input_modalities == frozenset({"text"})

    def test_modalities_outside_the_vocabulary_are_dropped(self, models_dev_dataset):
        """A kind this framework cannot declare is not a divergence.

        The restricted set is what the comparison uses; the verbatim set is
        kept for the report so the reader sees what the upstream really said.
        """
        record = models_dev_dataset.lookup("google", "gemini-2.5-flash")
        assert record.input_modalities <= COMPARABLE_MODALITIES
        assert "pdf" in record.input_modalities_verbatim
        assert "pdf" not in record.input_modalities

    def test_the_two_axes_carry_their_own_explicitness(self):
        """One axis's explicitness must not license the other's denial.

        A record can state one array and not the other, and reading the input
        flag for the output axis either invents a denial the upstream never
        made or silences one it did.
        """
        result = parse_models_dev(
            {"openai": {"models": {"m": {"modalities": {"output": ["text"]}}}}}
        )
        record = result.lookup("openai", "m")
        assert record.states_output_modalities is True
        assert record.states_input_modalities is False

    def test_an_axis_view_reports_that_axis_and_no_other(self):
        """The view is what a consumer reads, so it must not cross the axes."""
        result = parse_models_dev(
            {
                "openai": {
                    "models": {
                        "m": {
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["video"],
                            }
                        }
                    }
                }
            }
        )
        record = result.lookup("openai", "m")
        assert record.axis("input_modalities").declared == frozenset({"text", "image"})
        assert record.axis("output_modalities").declared == frozenset()
        assert record.axis("output_modalities").verbatim == frozenset({"video"})
        assert record.axis("output_modalities").explicit is True

    def test_an_array_of_only_uncomparable_kinds_is_empty_not_absent(self):
        """Empty and absent are different upstream statements.

        An array listing only kinds outside this vocabulary is a statement
        that was made; a missing array is one that was not. Collapsing them
        would either silence a real over-claim or invent one.
        """
        result = parse_models_dev(
            {"openai": {"models": {"m": {"modalities": {"input": ["pdf", "video"]}}}}}
        )
        record = result.lookup("openai", "m")
        assert record.input_modalities == frozenset()
        assert record.input_modalities is not None
        assert record.input_modalities_verbatim == frozenset({"pdf", "video"})

    def test_the_temperature_field_is_read(self, models_dev_dataset):
        """This dataset carries a temperature boolean and it is parsed."""
        record = models_dev_dataset.lookup("openai", "gpt-4")
        assert isinstance(record.temperature, bool)

    def test_the_context_window_is_read(self, models_dev_dataset):
        """The context limit is parsed as a positive integer."""
        record = models_dev_dataset.lookup("openai", "gpt-4")
        assert record.context_window == 8192

    @pytest.mark.parametrize("payload", [None, [], {}, "text", 7])
    def test_a_document_of_the_wrong_shape_is_malformed(self, payload):
        """Anything but a non-empty object is a reason, not an empty dataset."""
        result = parse_models_dev(payload)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED

    def test_a_document_with_no_parseable_models_is_malformed(self):
        """A well-formed object carrying nothing usable must not read clean."""
        result = parse_models_dev(
            {
                "openai": "not-an-object",
                "anthropic": {"models": "not-an-object"},
                "google": {"models": {}},
                "xai": {"models": {"a": "not-an-object"}},
            }
        )
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED

    def test_unstated_fields_parse_to_none_rather_than_a_default(self):
        """An absent field is unrecorded; it must never become a value."""
        result = parse_models_dev({"openai": {"models": {"m": {}}}})
        record = result.lookup("openai", "m")
        assert record.input_modalities is None
        assert record.temperature is None
        assert record.context_window is None
        assert record.states_input_modalities is False
        assert record.states_output_modalities is False

    @pytest.mark.parametrize(
        "modalities",
        [{"input": []}, {"input": "text"}, {"input": [1, 2]}, "not-a-dict"],
    )
    def test_an_unusable_modality_value_parses_to_none(self, modalities):
        """A malformed modality value yields nothing, never a partial set."""
        result = parse_models_dev(
            {"openai": {"models": {"m": {"modalities": modalities}}}}
        )
        assert result.lookup("openai", "m").input_modalities is None

    @pytest.mark.parametrize("value", [True, False, "8192", None, 0, -1])
    def test_a_non_positive_integer_context_is_rejected(self, value):
        """A boolean, a string, or a non-positive number is not a window."""
        result = parse_models_dev(
            {"openai": {"models": {"m": {"limit": {"context": value}}}}}
        )
        assert result.lookup("openai", "m").context_window is None

    def test_a_non_boolean_temperature_is_rejected(self):
        """Only a real boolean is read as a temperature answer."""
        result = parse_models_dev({"openai": {"models": {"m": {"temperature": "yes"}}}})
        assert result.lookup("openai", "m").temperature is None

    def test_a_non_dict_limit_is_tolerated(self):
        """A limit of the wrong shape yields no window rather than raising."""
        result = parse_models_dev({"openai": {"models": {"m": {"limit": 5}}}})
        assert result.lookup("openai", "m").context_window is None


class TestParseLitellm:
    """The flat dataset, re-keyed by the provider each record names."""

    def test_the_recorded_payload_parses(self, litellm_dataset):
        """The capture parses into a dataset with records."""
        assert isinstance(litellm_dataset, Dataset)
        assert litellm_dataset.source == LITELLM
        assert litellm_dataset.model_count > 0

    def test_records_are_scoped_by_the_provider_they_name(self, litellm_dataset):
        """A gateway that re-lists a model is not an authority on it.

        The recorded slice carries a record for this model under the vendor's
        own provider; a lookup under a different provider must not find it.
        """
        assert litellm_dataset.lookup("openai", "gpt-4") is not None
        assert litellm_dataset.lookup("anthropic", "gpt-4") is None

    def test_a_short_name_resolves_to_the_shallowest_key(self, litellm_dataset):
        """A canonical key outranks a region-qualified one.

        This dataset lists the same model under several
        "<provider>/<region>/<id>" keys beside one bare "<id>" key. A
        first-wins alias would let an arbitrary regional record stand in for
        the canonical one and would cite a key the reader cannot find.
        """
        record = litellm_dataset.lookup("bedrock", "mistral.mistral-large-2402-v1:0")
        assert record is not None
        assert record.key == "mistral.mistral-large-2402-v1:0"
        assert "/" not in record.key

    def test_the_shallowest_key_wins_whatever_the_document_order(self):
        """A canonical key stated last must still outrank a qualified one.

        A first-wins alias happens to be right when the canonical key comes
        first, so an assertion that only ever sees that order agrees with the
        broken implementation too.
        """
        result = parse_litellm(
            {
                "bedrock/eu-west-3/m": {
                    "litellm_provider": "bedrock",
                    "max_input_tokens": 111,
                },
                "bedrock/us-east-1/m": {
                    "litellm_provider": "bedrock",
                    "max_input_tokens": 222,
                },
                "m": {"litellm_provider": "bedrock", "max_input_tokens": 999},
            }
        )
        record = result.lookup("bedrock", "m")
        assert record.key == "m"
        assert record.context_window == 999

    def test_a_prefixed_key_is_also_reachable_verbatim(self, litellm_dataset):
        """The full key still resolves, so a caller can address either form."""
        assert litellm_dataset.lookup("azure", "azure/gpt-4") is not None
        assert litellm_dataset.lookup("azure", "gpt-4") is not None

    def test_this_dataset_never_licenses_a_denial(self, litellm_dataset):
        """Its silence is unrecorded, so no record here may deny a modality."""
        for models in litellm_dataset.records.values():
            for record in models.values():
                assert record.states_input_modalities is False
                assert record.states_output_modalities is False

    def test_this_dataset_carries_no_temperature_answer(self, litellm_dataset):
        """It has no such field, so it must never assert one."""
        for models in litellm_dataset.records.values():
            for record in models.values():
                assert record.temperature is None

    def test_a_positive_vision_flag_confirms_image_input(self):
        """A stated flag is a positive signal and is read as one."""
        result = parse_litellm(
            {"m": {"litellm_provider": "openai", "supports_vision": True}}
        )
        record = result.lookup("openai", "m")
        assert record.input_modalities == frozenset({"text", "image"})
        assert record.states_input_modalities is False

    @pytest.mark.parametrize("flag", [False, None, "true"])
    def test_an_absent_or_negative_vision_flag_is_not_a_denial(self, flag):
        """Absence here is unrecorded, never "does not accept images"."""
        entry = {"litellm_provider": "openai"}
        if flag is not None:
            entry["supports_vision"] = flag
        result = parse_litellm({"m": entry})
        assert result.lookup("openai", "m").input_modalities is None

    def test_a_stated_modality_array_outranks_the_vision_flag(self):
        """When the dataset states modalities, the flag is not consulted."""
        result = parse_litellm(
            {
                "m": {
                    "litellm_provider": "openai",
                    "supported_modalities": ["text", "audio"],
                    "supports_vision": True,
                }
            }
        )
        assert result.lookup("openai", "m").input_modalities == frozenset(
            {"text", "audio"}
        )

    def test_the_context_window_is_read(self):
        """The input window is parsed as a positive integer."""
        result = parse_litellm(
            {"m": {"litellm_provider": "openai", "max_input_tokens": 4096}}
        )
        assert result.lookup("openai", "m").context_window == 4096

    def test_output_modalities_are_read_when_stated(self):
        """The output axis parses when the dataset states it."""
        result = parse_litellm(
            {
                "m": {
                    "litellm_provider": "openai",
                    "supported_output_modalities": ["text", "image"],
                }
            }
        )
        assert result.lookup("openai", "m").output_modalities == frozenset(
            {"text", "image"}
        )

    @pytest.mark.parametrize("payload", [None, [], {}, "text", 7])
    def test_a_document_of_the_wrong_shape_is_malformed(self, payload):
        """Anything but a non-empty object is a reason, not an empty dataset."""
        result = parse_litellm(payload)
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED

    def test_records_without_a_provider_are_skipped(self):
        """A record naming no provider cannot be scoped, so it is not indexed."""
        result = parse_litellm(
            {
                "doc": "not-an-object",
                "a": {"max_input_tokens": 10},
                "b": {"litellm_provider": "", "max_input_tokens": 10},
                "c": {"litellm_provider": 7, "max_input_tokens": 10},
                "d": {"litellm_provider": "openai", "max_input_tokens": 10},
            }
        )
        assert set(result.records) == {"openai"}
        assert result.lookup("openai", "d") is not None

    def test_a_document_with_no_scopeable_records_is_malformed(self):
        """A document where nothing can be scoped must not read clean."""
        result = parse_litellm({"a": {"max_input_tokens": 10}})
        assert isinstance(result, Unavailable)
        assert result.reason == REASON_MALFORMED


class TestLoaders:
    """The loaders choose a source and hand any failure straight back."""

    def test_loading_models_dev_from_a_file_records_the_origin(self, models_dev_path):
        """A file-sourced dataset names the file it came from."""
        dataset = load_models_dev(models_dev_path)
        assert isinstance(dataset, Dataset)
        assert dataset.origin == str(models_dev_path)

    def test_loading_litellm_from_a_file_records_the_origin(self, litellm_path):
        """A file-sourced dataset names the file it came from."""
        dataset = load_litellm(litellm_path)
        assert isinstance(dataset, Dataset)
        assert dataset.origin == str(litellm_path)

    def test_a_file_failure_is_returned_unwrapped(self, tmp_path):
        """A read failure reaches the caller as its own reason."""
        assert isinstance(load_models_dev(tmp_path / "nope.json"), Unavailable)
        assert isinstance(load_litellm(tmp_path / "nope.json"), Unavailable)

    def test_the_network_path_records_the_upstream_url(self, monkeypatch):
        """A fetched dataset names the URL it came from."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeResponse(
                json.dumps({"openai": {"models": {"m": {}}}}).encode()
            ),
        )
        dataset = load_models_dev()
        assert isinstance(dataset, Dataset)
        assert dataset.origin.startswith("https://")

    def test_the_litellm_network_path_records_the_upstream_url(self, monkeypatch):
        """A fetched dataset names the URL it came from."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeResponse(
                json.dumps({"m": {"litellm_provider": "openai"}}).encode()
            ),
        )
        dataset = load_litellm()
        assert isinstance(dataset, Dataset)
        assert dataset.origin.startswith("https://")

    def test_a_network_failure_is_returned_unwrapped(self, monkeypatch):
        """A fetch failure reaches the caller instead of an empty dataset."""

        def _raise(*_args, **_kwargs):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        assert isinstance(load_models_dev(), Unavailable)
        assert isinstance(load_litellm(), Unavailable)


class TestFixtureProvenance:
    """The recorded slices are what the tests measure against."""

    def test_both_fixtures_name_their_capture_date(self, models_dev_path, litellm_path):
        """A fixture whose vintage is unknown cannot be reasoned about."""
        for path in (models_dev_path, litellm_path):
            assert path.name.endswith("_2026-09-03.json")

    def test_the_fixture_directory_carries_its_licenses(self, models_dev_path):
        """MIT requires the notices travel with the data."""
        readme = (Path(models_dev_path).parent / "README.md").read_text(
            encoding="utf-8"
        )
        assert "MIT License" in readme
        assert "Copyright (c) 2025 models.dev" in readme
        assert "Copyright (c) 2023 Berri AI" in readme
