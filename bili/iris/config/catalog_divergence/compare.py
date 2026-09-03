"""Compare the declared catalog against the community datasets, report-only.

This module changes nothing.  It reads
:data:`bili.iris.config.llm_config.LLM_MODELS`, resolves each entry to its
authoritative upstream records under the provider-scoped mapping, and emits a
finding for every disagreement.  A dataset never wins: the rule this
implements is that a dataset may *fill a gap* but may not *overturn a
record*, so a disagreement becomes a finding for a human to adjudicate rather
than a value anything adopts.

Why the severities are what they are
------------------------------------
The two ways to be wrong do not cost the same.  A *false refusal* blocks a
call the model would have served: recoverable, visible, overridable.  A *false
assurance* lets a call proceed that the model cannot serve, and the failure
then surfaces at the provider boundary, opaquely, which is the outcome the
modality gate exists to prevent.  So:

``ERROR``
    The catalog declares a modality that a dataset stating an explicit
    modality array does not list.  This is the false-assurance direction, and
    it is the shape of a real defect this check was written after: an entry
    whose declared image input had been read off a display name rather than
    off the model's actual capability.

``WARNING`` (catalog narrower)
    The dataset lists a modality the catalog omits.  Under-declaring is a
    defensible position -- the framework may deliberately decline to claim an
    input kind it ships no content-part builder for -- so this is a prompt to
    review, not a defect.

``WARNING`` (temperature, either direction)
    A live probe settled this axis: the parameter is accepted normally and is
    constrained only alongside extended thinking, so a dataset boolean of
    ``false`` can encode a mode-conditional restriction rather than a flat
    rejection.  A community capability field can flatten a conditional
    semantics into an unconditional boolean, so a disagreement here is a
    *semantic mismatch candidate*, never proof that either side is wrong.

``WARNING`` / ``INFO`` (context window)
    Over-declaring is the direction that matters, because a declared window is
    read as an approximate prompt budget, so it is a ``WARNING``.
    Under-declaring is conservative and is ``INFO``.  Neither is an ``ERROR``:
    a window is a quantity rather than a capability, it gates no call, and the
    two datasets routinely disagree with each other about it.

``INFO`` (unmatched)
    No authoritative dataset carries the entry.  Expected for the provider
    families that have no upstream listing at all, which are excluded, and
    otherwise a note that this entry is unverifiable from these sources.

An omission is never a denial, which is why an ``ERROR`` can only come from a
dataset that states modalities as an explicit array.  A dataset that simply
has no value for a field says nothing, and reading its silence as "no" would
manufacture the exact false finding this check exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bili.iris.config.llm_config import LLM_MODELS

from .datasets import Dataset, DatasetResult, Unavailable
from .mapping import (
    ADVISORY_ONLY_PROVIDER_TYPES,
    LITELLM_PROVIDERS,
    MODELS_DEV_PROVIDERS,
    RECORDED_MATCH_FLOORS,
    UNLISTED_PROVIDER_TYPES,
    id_candidates,
)

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

#: Ordered most severe first, which is also the report's sort order.
SEVERITY_ORDER = (ERROR, WARNING, INFO)

FIELD_INPUT_MODALITIES = "input_modalities"
FIELD_OUTPUT_MODALITIES = "output_modalities"
FIELD_TEMPERATURE = "supports_temperature"
FIELD_CONTEXT_WINDOW = "max_input_tokens"
FIELD_COVERAGE = "coverage"
FIELD_COVERAGE_FLOOR = "coverage_floor"


@dataclass(frozen=True)
class DatasetValue:
    """One upstream dataset's answer for a compared field.

    :ivar source: The dataset name.
    :ivar provider_id: The upstream provider the record was found under.
    :ivar key: The upstream key the record was found at.
    :ivar value: The upstream value, rendered for the report.
    """

    source: str
    provider_id: str
    key: str
    value: Any


@dataclass(frozen=True)
class Finding:  # pylint: disable=too-many-instance-attributes
    """One disagreement between the catalog and the datasets.

    :ivar severity: One of :data:`SEVERITY_ORDER`.
    :ivar provider_type: The catalog provider type key.
    :ivar model_id: The catalog ``model_id``.
    :ivar model_name: The catalog ``model_name``, for legibility only.
    :ivar field_name: The catalog field in disagreement.
    :ivar catalog_value: What the catalog declares.
    :ivar dataset_values: What each dataset that has an opinion says.
    :ivar message: A one-line explanation.
    """

    severity: str
    provider_type: str
    model_id: str
    model_name: str
    field_name: str
    catalog_value: Any
    dataset_values: Tuple[DatasetValue, ...]
    message: str


@dataclass(frozen=True)
class ProviderMatch:
    """Per-provider-family coverage of the catalog by the datasets.

    :ivar provider_type: The catalog provider type key.
    :ivar entries: Catalog entries in this family.
    :ivar matched_models_dev: Entries resolved to a models.dev record.
    :ivar matched_litellm: Entries resolved to a LiteLLM record.
    :ivar matched_either: Entries resolved to at least one record.
    """

    provider_type: str
    entries: int
    matched_models_dev: int
    matched_litellm: int
    matched_either: int

    @property
    def match_rate(self) -> float:
        """Fraction of this family's entries resolved to at least one record.

        :returns: A value in ``[0.0, 1.0]``; ``0.0`` for an empty family.
        :rtype: float
        """
        if not self.entries:
            return 0.0
        return self.matched_either / self.entries


@dataclass(frozen=True)
class DivergenceReport:
    """The whole comparison: findings, coverage, and what could not be read.

    :ivar findings: Every disagreement, unsorted.
    :ivar matches: Per-provider coverage, keyed by provider type.
    :ivar unavailable: Datasets that could not be read, with reasons.
    :ivar catalog_entries: Total catalog entries examined.
    :ivar generated_at: UTC ISO-8601 timestamp of the run.
    :ivar dataset_origins: Where each readable dataset's bytes came from.
    """

    findings: Tuple[Finding, ...] = ()
    matches: Dict[str, ProviderMatch] = field(default_factory=dict)
    unavailable: Tuple[Unavailable, ...] = ()
    catalog_entries: int = 0
    generated_at: str = ""
    dataset_origins: Dict[str, str] = field(default_factory=dict)

    def count(self, severity: str) -> int:
        """Number of findings at *severity*.

        :param severity: One of :data:`SEVERITY_ORDER`.
        :returns: The count.
        :rtype: int
        """
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def has_errors(self) -> bool:
        """Whether any finding is an ``ERROR``.

        :returns: ``True`` when at least one finding is an ``ERROR``.
        :rtype: bool
        """
        return self.count(ERROR) > 0

    @property
    def coverage_regressed(self) -> bool:
        """Whether any family resolved fewer entries than it did at capture.

        This is the check noticing that it has stopped working. An upstream
        that renames a provider key or changes an id scheme still serves a
        well-formed document, so nothing fails to parse; the lookups simply
        stop hitting, every entry reads as uncovered, and the run reports no
        divergence. That is the same silence a failed fetch would produce and
        it has to be as loud.

        :returns: ``True`` when at least one family is below its floor.
        :rtype: bool
        """
        return any(f.field_name == FIELD_COVERAGE_FLOOR for f in self.findings)

    @property
    def any_unavailable(self) -> bool:
        """Whether any dataset could not be read.

        :returns: ``True`` when at least one dataset is unavailable.
        :rtype: bool
        """
        return bool(self.unavailable)


def _cap(severity: str, provider_type: str) -> str:
    """Lower an ``ERROR`` to a ``WARNING`` for an advisory-only family.

    A provider type whose catalog id is an operator-chosen deployment name has
    no vendor model behind that string, so an upstream record carrying the
    same string is about something else and cannot establish that the catalog
    is wrong.

    :param severity: The severity the rule produced.
    :param provider_type: The catalog provider type key.
    :returns: The severity to report.
    :rtype: str
    """
    if severity == ERROR and provider_type in ADVISORY_ONLY_PROVIDER_TYPES:
        return WARNING
    return severity


def _resolve(
    dataset: Optional[Dataset],
    provider_ids: Sequence[str],
    provider_type: str,
    model_id: str,
):
    """Find *model_id*'s record in *dataset*, provider-scoped.

    :param dataset: The dataset to search, or ``None`` when unavailable.
    :param provider_ids: The authoritative upstream providers, in order.
    :param provider_type: The catalog provider type key.
    :param model_id: The catalog model id.
    :returns: The matching :class:`~.datasets.CapabilityRecord`, or ``None``.
    """
    if dataset is None:
        return None
    candidates = id_candidates(provider_type, model_id)
    for provider_id in provider_ids:
        for candidate in candidates:
            record = dataset.lookup(provider_id, candidate)
            if record is not None:
                return record
    return None


def _modality_findings(  # pylint: disable=too-many-locals
    provider_type: str,
    entry: dict,
    records: Sequence[Any],
    field_name: str,
    catalog_key: str,
) -> List[Finding]:
    """Compare one declared modality set against every dataset that states one.

    :param provider_type: The catalog provider type key.
    :param entry: The catalog entry.
    :param records: The resolved upstream records.
    :param field_name: The reported field name.
    :param catalog_key: The catalog key holding the declaration.
    :returns: Zero or more findings.
    :rtype: List[Finding]
    """
    declared = entry.get(catalog_key)
    if not declared:
        return []
    declared_set = frozenset(declared)
    attribute = (
        "input_modalities" if catalog_key == "input_modalities" else "output_modalities"
    )

    findings: List[Finding] = []
    for record in records:
        dataset_set = getattr(record, attribute)
        if dataset_set is None:
            continue
        if dataset_set == declared_set:
            continue

        # The untrimmed upstream set is shown so the reader sees what the
        # dataset really said, including the kinds this vocabulary drops.
        # Only the input axis keeps one; the output axis has nothing trimmed.
        if attribute == "input_modalities":
            shown = sorted(record.input_modalities_verbatim or dataset_set)
        else:
            shown = sorted(dataset_set)
        value = DatasetValue(record.source, record.provider_id, record.key, shown)

        overclaimed = declared_set - dataset_set
        underclaimed = dataset_set - declared_set
        # An over-claim is a finding only when the dataset ENUMERATES its
        # modalities; a dataset that merely confirms one cannot deny another.
        denies = bool(overclaimed) and record.states_modalities_explicitly

        # Both directions can hold at once, and reporting only the more severe
        # half would leave the other stated nowhere.  They are one finding
        # rather than two, at the higher severity, because they are one fact
        # about one field.
        parts = []
        if denies:
            parts.append(
                f"catalog claims {sorted(overclaimed)}, which the dataset "
                f"does not list"
            )
        if underclaimed:
            parts.append(
                f"catalog omits {sorted(underclaimed)}, which the dataset lists"
            )
        if not parts:
            continue

        findings.append(
            Finding(
                severity=_cap(ERROR, provider_type) if denies else WARNING,
                provider_type=provider_type,
                model_id=entry["model_id"],
                model_name=entry.get("model_name", ""),
                field_name=field_name,
                catalog_value=sorted(declared_set),
                dataset_values=(value,),
                message="; ".join(parts),
            )
        )
    return findings


def _temperature_findings(
    provider_type: str, entry: dict, records: Sequence[Any]
) -> List[Finding]:
    """Compare the declared temperature support against the datasets.

    :param provider_type: The catalog provider type key.
    :param entry: The catalog entry.
    :param records: The resolved upstream records.
    :returns: Zero or more findings.
    :rtype: List[Finding]
    """
    if "supports_temperature" not in entry:
        return []
    declared = bool(entry["supports_temperature"])
    findings: List[Finding] = []
    for record in records:
        if record.temperature is None or record.temperature == declared:
            continue
        findings.append(
            Finding(
                severity=WARNING,
                provider_type=provider_type,
                model_id=entry["model_id"],
                model_name=entry.get("model_name", ""),
                field_name=FIELD_TEMPERATURE,
                catalog_value=declared,
                dataset_values=(
                    DatasetValue(
                        record.source,
                        record.provider_id,
                        record.key,
                        record.temperature,
                    ),
                ),
                message=(
                    "semantic mismatch candidate: a dataset boolean can encode "
                    "a mode-conditional restriction rather than a flat rejection"
                ),
            )
        )
    return findings


def _context_findings(
    provider_type: str, entry: dict, records: Sequence[Any]
) -> List[Finding]:
    """Compare the declared context window against the datasets.

    Every dataset that records a window contributes one finding, so a
    disagreement *between* the datasets is visible rather than hidden behind
    whichever one was consulted first.

    :param provider_type: The catalog provider type key.
    :param entry: The catalog entry.
    :param records: The resolved upstream records.
    :returns: Zero or more findings.
    :rtype: List[Finding]
    """
    declared = entry.get("max_input_tokens")
    # bool is a subclass of int, so an unguarded isinstance check reads True
    # as the window 1.  The dataset side already rejects a boolean, and the
    # two sides of one comparison have to mean the same thing by "a window".
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        return []
    findings: List[Finding] = []
    for record in records:
        window = record.context_window
        if window is None or window == declared:
            continue
        over = declared > window
        findings.append(
            Finding(
                severity=WARNING if over else INFO,
                provider_type=provider_type,
                model_id=entry["model_id"],
                model_name=entry.get("model_name", ""),
                field_name=FIELD_CONTEXT_WINDOW,
                catalog_value=declared,
                dataset_values=(
                    DatasetValue(record.source, record.provider_id, record.key, window),
                ),
                message=(
                    "catalog declares a larger window than the dataset records"
                    if over
                    else "catalog declares a smaller window than the dataset records"
                ),
            )
        )
    return findings


def merge_findings(findings: Sequence[Finding]) -> List[Finding]:
    """Coalesce findings that state the same fact about the same field.

    Both datasets are asked about every entry, so when they agree the same
    disagreement is raised twice.  Printing it twice is noise, and worse, it
    hides the part a maintainer most wants: that two independent sources
    corroborate each other.  Findings identical but for their citation are
    merged into one carrying every citation, so agreement reads as one
    stronger finding and a genuine dataset-vs-dataset disagreement (a
    different message, or a different severity) stays two.

    :param findings: The findings to coalesce.
    :returns: Merged findings, in first-seen order.
    :rtype: List[Finding]
    """
    merged: Dict[Tuple[Any, ...], Finding] = {}
    for finding in findings:
        key = (
            finding.severity,
            finding.provider_type,
            finding.model_id,
            finding.field_name,
            finding.message,
            repr(finding.catalog_value),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = finding
        else:
            merged[key] = replace(
                existing,
                dataset_values=existing.dataset_values + finding.dataset_values,
            )
    return list(merged.values())


def _coverage_floor_findings(matches: Dict[str, ProviderMatch]) -> List[Finding]:
    """Report any family resolving fewer entries than it did at capture.

    A family is skipped when the catalog no longer carries it, because a
    provider type that was removed cannot be measured against a floor.

    :param matches: The per-family coverage this run measured.
    :returns: Zero or more findings.
    :rtype: List[Finding]
    """
    findings: List[Finding] = []
    for provider_type, floor in sorted(RECORDED_MATCH_FLOORS.items()):
        match = matches.get(provider_type)
        if match is None or match.matched_either >= floor:
            continue
        findings.append(
            Finding(
                severity=WARNING,
                provider_type=provider_type,
                model_id="",
                model_name="",
                field_name=FIELD_COVERAGE_FLOOR,
                catalog_value=match.matched_either,
                dataset_values=(),
                message=(
                    f"only {match.matched_either} of {match.entries} entries "
                    f"resolved, below the recorded floor of {floor}: either an "
                    f"upstream moved out from under the id mapping, or the "
                    f"mapping regressed"
                ),
            )
        )
    return findings


def compare_catalog(  # pylint: disable=too-many-locals
    models_dev: DatasetResult,
    litellm: DatasetResult,
    catalog: Optional[Dict[str, dict]] = None,
) -> DivergenceReport:
    """Compare the catalog against both datasets and return every finding.

    Either dataset may be an :class:`~.datasets.Unavailable`; the comparison
    proceeds against whichever is readable and records the failure, because a
    partial comparison reported as partial is useful and a failed fetch
    reported as "no divergence" is not.

    :param models_dev: A parsed models.dev dataset, or an ``Unavailable``.
    :param litellm: A parsed LiteLLM dataset, or an ``Unavailable``.
    :param catalog: The catalog to compare; defaults to the shipped
        ``LLM_MODELS``. The recorded coverage floors are checked only for
        that default, because they were measured against it.
    :returns: The full report.
    :rtype: DivergenceReport
    """
    # The recorded floors were measured against the SHIPPED catalog, so they
    # say nothing about a caller-supplied one: a smaller catalog resolving
    # fewer entries is not a regression, it is a different question.
    compares_shipped_catalog = catalog is None
    catalog = LLM_MODELS if catalog is None else catalog

    unavailable = tuple(d for d in (models_dev, litellm) if isinstance(d, Unavailable))
    md = models_dev if isinstance(models_dev, Dataset) else None
    ll = litellm if isinstance(litellm, Dataset) else None
    origins = {d.source: d.origin for d in (md, ll) if d is not None}

    findings: List[Finding] = []
    matches: Dict[str, ProviderMatch] = {}
    total_entries = 0

    for provider_type, provider in sorted(catalog.items()):
        entries = provider.get("models") or []
        total_entries += len(entries)
        md_ids = MODELS_DEV_PROVIDERS.get(provider_type, ())
        ll_ids = LITELLM_PROVIDERS.get(provider_type, ())
        md_hits = ll_hits = either_hits = 0

        for entry in entries:
            model_id = entry.get("model_id")
            if not model_id:
                continue
            md_record = _resolve(md, md_ids, provider_type, model_id)
            ll_record = _resolve(ll, ll_ids, provider_type, model_id)
            records = [r for r in (md_record, ll_record) if r is not None]
            md_hits += md_record is not None
            ll_hits += ll_record is not None
            either_hits += bool(records)

            if not records:
                if provider_type not in UNLISTED_PROVIDER_TYPES:
                    findings.append(
                        Finding(
                            severity=INFO,
                            provider_type=provider_type,
                            model_id=model_id,
                            model_name=entry.get("model_name", ""),
                            field_name=FIELD_COVERAGE,
                            catalog_value=None,
                            dataset_values=(),
                            message=(
                                "no authoritative dataset record; this entry is "
                                "unverifiable from these sources"
                            ),
                        )
                    )
                continue

            findings.extend(
                _modality_findings(
                    provider_type,
                    entry,
                    records,
                    FIELD_INPUT_MODALITIES,
                    "input_modalities",
                )
            )
            findings.extend(
                _modality_findings(
                    provider_type,
                    entry,
                    records,
                    FIELD_OUTPUT_MODALITIES,
                    "output_modalities",
                )
            )
            findings.extend(_temperature_findings(provider_type, entry, records))
            findings.extend(_context_findings(provider_type, entry, records))

        matches[provider_type] = ProviderMatch(
            provider_type=provider_type,
            entries=len(entries),
            matched_models_dev=md_hits,
            matched_litellm=ll_hits,
            matched_either=either_hits,
        )

    if compares_shipped_catalog:
        findings.extend(_coverage_floor_findings(matches))

    return DivergenceReport(
        findings=tuple(merge_findings(findings)),
        matches=matches,
        unavailable=unavailable,
        catalog_entries=total_entries,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_origins=origins,
    )
