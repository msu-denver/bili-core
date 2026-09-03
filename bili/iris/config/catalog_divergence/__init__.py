"""Report-only cross-check of the declared model catalog against community data.

The model catalog is hand-maintained, so it drifts, and the drift that costs
most is the kind that reads as a capability the model does not have: a
declaration made from a display name rather than from the model's behaviour
will pass a capability gate and then fail opaquely at the provider boundary.
This package is the detector for that class.  It fetches two MIT-licensed
community capability datasets, resolves every catalog entry to its
*authoritative* upstream record, and reports every disagreement.

It changes nothing.  A dataset may fill a gap in the catalog and may never
overturn a declared value, so a disagreement is raised for a human to
adjudicate rather than resolved in the dataset's favour.  That posture is not
caution for its own sake: the community data is measurably wrong in places,
and a live probe has already shown a case where adopting a dataset value would
have replaced a correct declaration with an incorrect one.

Entry point::

    python -m bili.iris.config.catalog_divergence.cli
"""

from .compare import (
    ERROR,
    INFO,
    SEVERITY_ORDER,
    WARNING,
    DatasetValue,
    DivergenceReport,
    Finding,
    ProviderMatch,
    compare_catalog,
)
from .datasets import (
    LITELLM,
    MODELS_DEV,
    CapabilityRecord,
    Dataset,
    Unavailable,
    load_litellm,
    load_models_dev,
    parse_litellm,
    parse_models_dev,
)
from .mapping import (
    ADVISORY_ONLY_PROVIDER_TYPES,
    LITELLM_PROVIDERS,
    MODELS_DEV_PROVIDERS,
    UNLISTED_PROVIDER_TYPES,
    id_candidates,
)
from .report import STICKY_MARKER, render_issue_body, render_json, render_text, to_dict

__all__ = [
    "ADVISORY_ONLY_PROVIDER_TYPES",
    "STICKY_MARKER",
    "CapabilityRecord",
    "Dataset",
    "DatasetValue",
    "DivergenceReport",
    "ERROR",
    "Finding",
    "INFO",
    "LITELLM",
    "LITELLM_PROVIDERS",
    "MODELS_DEV",
    "MODELS_DEV_PROVIDERS",
    "ProviderMatch",
    "SEVERITY_ORDER",
    "UNLISTED_PROVIDER_TYPES",
    "Unavailable",
    "WARNING",
    "compare_catalog",
    "id_candidates",
    "load_litellm",
    "load_models_dev",
    "parse_litellm",
    "parse_models_dev",
    "render_issue_body",
    "render_json",
    "render_text",
    "to_dict",
]
