"""Per-model load defaults read from the model catalog.

The catalog (:data:`bili.iris.config.llm_config.LLM_MODELS`) declares per-model
capabilities that the loader should apply as defaults when the caller supplies
none.  Historically some of these were metadata only: a value was declared but
never reached the provider.

:func:`model_max_output_tokens` is the output-budget case.  Without it, a
provider's own fallback caps output: ``langchain_anthropic`` defaults
``max_tokens`` to 1024 when none is passed, far too small for an agent to emit a
multi-KB document as a required tool argument, so the tool call is truncated, its
argument-schema validation fails, and the agent loop retries fruitlessly.  The
catalog already carries the correct ``max_output_tokens`` per model; the loader
reads it here and passes it through.

(The sibling temperature case lives in
:mod:`bili.iris.providers.temperature_resilience`; both are catalog-derived load
defaults applied at the ``load_model`` choke point.)
"""

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def model_max_output_tokens(
    model_type: str, model_name: Optional[str]
) -> Optional[int]:
    """Return the cataloged ``max_output_tokens`` for a model, or ``None``.

    Returns the value only when the catalog entry declares it AND does not
    declare ``supports_max_output_tokens: False`` (a model that rejects an
    explicit output cap must not be given one).  Returns ``None`` for a
    passthrough (uncataloged) model, an entry without the value, or one that
    opts out, so the caller leaves ``max_tokens`` alone.

    :param model_type: The provider type key (e.g. ``"remote_anthropic"``).
    :param model_name: The model id the loader was given.
    :returns: The per-model output-token cap, or ``None`` to leave it unset.
    :rtype: Optional[int]
    """
    if not model_name:
        return None
    try:
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:  # pragma: no cover - config always importable in practice
        return None
    provider_info = LLM_MODELS.get(model_type)
    if not provider_info:
        return None
    for entry in provider_info.get("models", []):
        if entry.get("model_id") != model_name:
            continue
        if entry.get("supports_max_output_tokens") is False:
            return None
        value = entry.get("max_output_tokens")
        return value if isinstance(value, int) and value > 0 else None
    return None
