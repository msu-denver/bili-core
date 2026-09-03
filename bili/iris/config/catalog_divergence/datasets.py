"""Fetching and parsing the two community capability datasets.

Two datasets are read, both MIT-licensed and both fetched over plain HTTPS
with no credentials:

``models.dev``
    ``https://models.dev/api.json``.  Provider-keyed
    (``{provider: {id, name, models: {model_id: record}}}``) and the only one
    of the two with explicit ``modalities.input`` / ``modalities.output``
    arrays on essentially every record.  It also carries a ``temperature``
    boolean the issue's survey does not note.

``LiteLLM``
    ``model_prices_and_context_window.json``.  A flat ``{key: record}`` map
    where each record names its own provider in ``litellm_provider``, so it is
    re-keyed by provider here for the same reason the other one already is.
    Broader coverage of context windows, thinner modality coverage.

An absent field is not a denial
-------------------------------
The two datasets express "no" differently, and the difference decides how
strong a disagreement may be.  models.dev states an explicit modality array,
so a modality absent from that array is a *denial*.  LiteLLM omits fields it
has no value for, so the absence of ``supports_vision`` means "unrecorded",
not "does not accept images".  :func:`parse_litellm` therefore yields input
modalities only when the record states them positively, and never synthesises
a denial from an omission.

Failure is a result, not an exception
-------------------------------------
Every failure path returns :class:`Unavailable` carrying a reason, because the
one outcome this must never produce is a silent empty dataset: an empty merge
reports "no divergence", which is indistinguishable from a clean catalog and
is the worst answer a checker can give.  A caller that cannot tell a fetch
failure from a clean result will eventually report a broken job as a passing
one.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

MODELS_DEV = "models.dev"
LITELLM = "litellm"

#: The input-modality vocabulary this framework can express.  An upstream
#: record may declare kinds outside it (``pdf``, ``video``); those are dropped
#: before comparison, because a catalog entry cannot declare them and a
#: difference there is not a divergence in the catalog's own terms.
COMPARABLE_MODALITIES = frozenset({"text", "image", "audio"})

REASON_NETWORK = "network"
REASON_AUTH = "auth"
REASON_HTTP = "http"
REASON_MALFORMED = "malformed"

DEFAULT_TIMEOUT_SECONDS = 30
_USER_AGENT = "bili-core-catalog-divergence/1"


@dataclass(frozen=True)
class Unavailable:
    """A dataset that could not be read, and why.

    :ivar source: The dataset name (:data:`MODELS_DEV` or :data:`LITELLM`).
    :ivar reason: One of the ``REASON_*`` constants.
    :ivar detail: A human-readable explanation for the report.
    """

    source: str
    reason: str
    detail: str


@dataclass(frozen=True)
class CapabilityRecord:  # pylint: disable=too-many-instance-attributes
    """One upstream model record, reduced to the axes this compares.

    Every field is tri-state: ``None`` means the dataset records nothing,
    which is never read as a denial.

    :ivar source: The dataset the record came from.
    :ivar provider_id: The upstream provider the record was found under.
    :ivar key: The upstream key the record was found at.
    :ivar input_modalities: Declared input kinds, restricted to
        :data:`COMPARABLE_MODALITIES`.
    :ivar input_modalities_verbatim: Declared input kinds as the upstream
        states them, before that restriction, for the report.
    :ivar output_modalities: Declared output kinds, similarly restricted.
    :ivar temperature: Whether the upstream records the model as accepting a
        ``temperature`` parameter.
    :ivar context_window: The upstream's input context window in tokens.
    :ivar states_modalities_explicitly: ``True`` only when the upstream states
        input modalities as an explicit array, which is what makes an absent
        modality a denial rather than an omission.
    """

    source: str
    provider_id: str
    key: str
    input_modalities: Optional[FrozenSet[str]] = None
    input_modalities_verbatim: Optional[FrozenSet[str]] = None
    output_modalities: Optional[FrozenSet[str]] = None
    temperature: Optional[bool] = None
    context_window: Optional[int] = None
    states_modalities_explicitly: bool = False


@dataclass(frozen=True)
class Dataset:
    """A parsed dataset, indexed provider-first.

    :ivar source: The dataset name.
    :ivar origin: Where the bytes came from (a URL or a file path), for the
        report's provenance line.
    :ivar records: ``{provider_id: {key: CapabilityRecord}}``.
    """

    source: str
    origin: str
    records: Dict[str, Dict[str, CapabilityRecord]] = field(default_factory=dict)

    @property
    def model_count(self) -> int:
        """Total records across every provider.

        :returns: The number of parsed model records.
        :rtype: int
        """
        return sum(len(models) for models in self.records.values())

    def lookup(self, provider_id: str, key: str) -> Optional[CapabilityRecord]:
        """Return the record at *key* under *provider_id*, or ``None``.

        :param provider_id: An upstream provider id.
        :param key: An upstream model key.
        :returns: The record, or ``None`` when the provider or key is absent.
        :rtype: Optional[CapabilityRecord]
        """
        return self.records.get(provider_id, {}).get(key)


DatasetResult = Union[Dataset, Unavailable]


def fetch_json(
    url: str, source: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> Union[Any, Unavailable]:
    """Fetch and decode a JSON document, returning :class:`Unavailable` on any failure.

    :param url: The document URL.
    :param source: The dataset name, used in the failure result.
    :param timeout: Socket timeout in seconds.
    :returns: The decoded JSON document, or an :class:`Unavailable`.
    :rtype: Union[Any, Unavailable]
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(  # nosec B310 - fixed https:// constants
            request, timeout=timeout
        ) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        reason = REASON_AUTH if exc.code in (401, 403) else REASON_HTTP
        return Unavailable(source, reason, f"HTTP {exc.code} from {url}")
    except (urllib.error.URLError, OSError) as exc:
        return Unavailable(source, REASON_NETWORK, f"{type(exc).__name__}: {exc}")

    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return Unavailable(
            source, REASON_MALFORMED, f"undecodable payload from {url}: {exc}"
        )


def read_json_file(path: Path, source: str) -> Union[Any, Unavailable]:
    """Read and decode a JSON document from disk.

    :param path: The file to read.
    :param source: The dataset name, used in the failure result.
    :returns: The decoded JSON document, or an :class:`Unavailable`.
    :rtype: Union[Any, Unavailable]
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return Unavailable(source, REASON_NETWORK, f"cannot read {path}: {exc}")
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return Unavailable(source, REASON_MALFORMED, f"undecodable file {path}: {exc}")


def _restrict(values: Any) -> Optional[FrozenSet[str]]:
    """Reduce an upstream modality list to the comparable vocabulary.

    :param values: The upstream value, of unverified shape.
    :returns: The restricted set, or ``None`` when the upstream states nothing
        usable.
    :rtype: Optional[FrozenSet[str]]
    """
    if not isinstance(values, list) or not values:
        return None
    strings = {v for v in values if isinstance(v, str)}
    if not strings:
        return None
    return frozenset(strings & COMPARABLE_MODALITIES)


def _verbatim(values: Any) -> Optional[FrozenSet[str]]:
    """Return an upstream modality list unrestricted, for the report.

    :param values: The upstream value, of unverified shape.
    :returns: The declared strings, or ``None``.
    :rtype: Optional[FrozenSet[str]]
    """
    if not isinstance(values, list) or not values:
        return None
    strings = {v for v in values if isinstance(v, str)}
    return frozenset(strings) or None


def _positive_int(value: Any) -> Optional[int]:
    """Return *value* as a positive int, or ``None`` if it is not one.

    :param value: The upstream value, of unverified shape.
    :returns: The integer, or ``None``.
    :rtype: Optional[int]
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def parse_models_dev(payload: Any, origin: str = MODELS_DEV_URL) -> DatasetResult:
    """Parse a models.dev ``api.json`` document.

    :param payload: The decoded document.
    :param origin: Where the bytes came from, recorded on the result.
    :returns: A :class:`Dataset`, or :class:`Unavailable` when the document
        does not have the documented shape or carries no models.
    :rtype: DatasetResult
    """
    if not isinstance(payload, dict) or not payload:
        return Unavailable(
            MODELS_DEV, REASON_MALFORMED, "top level is not a non-empty object"
        )

    records: Dict[str, Dict[str, CapabilityRecord]] = {}
    for provider_id, provider in payload.items():
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue
        bucket: Dict[str, CapabilityRecord] = {}
        for key, model in models.items():
            if not isinstance(model, dict):
                continue
            modalities = model.get("modalities")
            modalities = modalities if isinstance(modalities, dict) else {}
            declared_input = modalities.get("input")
            temperature = model.get("temperature")
            limit = model.get("limit")
            limit = limit if isinstance(limit, dict) else {}
            bucket[key] = CapabilityRecord(
                source=MODELS_DEV,
                provider_id=provider_id,
                key=key,
                input_modalities=_restrict(declared_input),
                input_modalities_verbatim=_verbatim(declared_input),
                output_modalities=_restrict(modalities.get("output")),
                temperature=(
                    bool(temperature) if isinstance(temperature, bool) else None
                ),
                context_window=_positive_int(limit.get("context")),
                states_modalities_explicitly=isinstance(declared_input, list),
            )
        if bucket:
            records[provider_id] = bucket

    if not records:
        return Unavailable(
            MODELS_DEV, REASON_MALFORMED, "document carried no parseable models"
        )
    return Dataset(MODELS_DEV, origin, records)


def parse_litellm(payload: Any, origin: str = LITELLM_URL) -> DatasetResult:
    """Parse a LiteLLM ``model_prices_and_context_window.json`` document.

    Each record is indexed under its ``litellm_provider`` at both its full key
    and its last path segment, because the catalog spells a model id without
    the provider prefix LiteLLM sometimes carries.

    :param payload: The decoded document.
    :param origin: Where the bytes came from, recorded on the result.
    :returns: A :class:`Dataset`, or :class:`Unavailable` when the document
        does not have the documented shape or carries no models.
    :rtype: DatasetResult
    """
    if not isinstance(payload, dict) or not payload:
        return Unavailable(
            LITELLM, REASON_MALFORMED, "top level is not a non-empty object"
        )

    records: Dict[str, Dict[str, CapabilityRecord]] = {}
    # Depth of the key each short-name alias currently points at, so a
    # canonical key always outranks a region-qualified one (see below).
    alias_depth: Dict[Tuple[str, str], int] = {}
    for key, model in payload.items():
        if not isinstance(model, dict):
            continue
        provider_id = model.get("litellm_provider")
        if not isinstance(provider_id, str) or not provider_id:
            continue

        declared_input = model.get("supported_modalities")
        restricted = _restrict(declared_input)
        verbatim = _verbatim(declared_input)
        if restricted is None and model.get("supports_vision") is True:
            # A positive vision flag confirms image input.  Its ABSENCE is
            # never read as a denial: this dataset omits fields it has no
            # value for, so an omission is unrecorded, not "no".
            restricted = frozenset({"text", "image"})
            verbatim = restricted

        record = CapabilityRecord(
            source=LITELLM,
            provider_id=provider_id,
            key=key,
            input_modalities=restricted,
            input_modalities_verbatim=verbatim,
            output_modalities=_restrict(model.get("supported_output_modalities")),
            temperature=None,  # this dataset carries no temperature field
            context_window=_positive_int(model.get("max_input_tokens")),
            # Never explicit: an omission here is unrecorded, not a denial.
            states_modalities_explicitly=False,
        )
        bucket = records.setdefault(provider_id, {})
        short = key.split("/")[-1]
        depth = key.count("/")
        if key != short:
            bucket.setdefault(key, record)

        # This dataset lists the same model several times under
        # region-qualified keys ("<provider>/<region>/<id>") beside one
        # canonical key ("<id>").  Indexing the short name first-wins would
        # let an arbitrary regional record stand in for the canonical one,
        # which is the resale-gateway contamination this module scopes by
        # provider to avoid, one level down.  The SHALLOWEST key wins, so a
        # canonical record always outranks a qualified one however the
        # document happens to be ordered, and the report cites the key a
        # reader can find.
        alias_key = (provider_id, short)
        if depth < alias_depth.get(alias_key, depth + 1):
            alias_depth[alias_key] = depth
            bucket[short] = record

    if not records:
        return Unavailable(
            LITELLM, REASON_MALFORMED, "document carried no parseable models"
        )
    return Dataset(LITELLM, origin, records)


def load_models_dev(
    path: Optional[Path] = None, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> DatasetResult:
    """Load models.dev from *path* when given, else from the network.

    :param path: A local copy to read instead of fetching.
    :param timeout: Socket timeout in seconds for the network path.
    :returns: A :class:`Dataset` or an :class:`Unavailable`.
    :rtype: DatasetResult
    """
    if path is not None:
        payload = read_json_file(path, MODELS_DEV)
        origin = str(path)
    else:
        payload = fetch_json(MODELS_DEV_URL, MODELS_DEV, timeout=timeout)
        origin = MODELS_DEV_URL
    if isinstance(payload, Unavailable):
        return payload
    return parse_models_dev(payload, origin=origin)


def load_litellm(
    path: Optional[Path] = None, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> DatasetResult:
    """Load LiteLLM from *path* when given, else from the network.

    :param path: A local copy to read instead of fetching.
    :param timeout: Socket timeout in seconds for the network path.
    :returns: A :class:`Dataset` or an :class:`Unavailable`.
    :rtype: DatasetResult
    """
    if path is not None:
        payload = read_json_file(path, LITELLM)
        origin = str(path)
    else:
        payload = fetch_json(LITELLM_URL, LITELLM, timeout=timeout)
        origin = LITELLM_URL
    if isinstance(payload, Unavailable):
        return payload
    return parse_litellm(payload, origin=origin)
