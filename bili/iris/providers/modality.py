"""Per-model input-modality capability, read from the model catalog.

A caller that attaches an image to a run needs to know, at *selection* time,
whether the model it picked will accept one.  Without a machine-readable
record, an image-bearing message is built and handed to a text-only model,
which then errors (or ignores the image) at invoke time, at the provider
boundary, where the failure is opaque, instead of at selection, where it is
actionable.  This module supplies that record and the check that reads it.

The catalog field
-----------------
Each entry in :data:`bili.iris.config.llm_config.LLM_MODELS` may declare
``input_modalities``: the set of input kinds the model accepts, drawn from
:data:`KNOWN_INPUT_MODALITIES` (``text``, ``image``, ``audio``)::

    {
        "model_name": "Some Vision Model",
        "model_id": "vendor.some-vision-model",
        "input_modalities": ["text", "image"],
        ...
    }

Input and output modality are independent axes.  This module answers the
*input* question only; an output-side record (what a model can emit) is a
separate axis and is deliberately not modelled here.

Declared, absent, and unknown
-----------------------------
The reader is deliberately tri-state, and the third state is load-bearing:

``declared and contains the modality``
    :func:`require_input_modality` returns.

``declared and does not contain it``
    :func:`require_input_modality` raises
    :exc:`UnsupportedInputModalityError`, naming the model and the modality.

``not declared`` (an uncataloged/passthrough model id, or a catalog entry
without the key)
    :func:`require_input_modality` logs a warning and allows the load.
    bili-core cannot assert a capability it has no record of, and refusing
    would block every passthrough model, including a locally-pulled
    vision-capable model reached by a sentinel-prefixed name, which the
    catalog cannot enumerate by construction.  The warning keeps the gap
    visible rather than silent.

A catalog entry omits the key when bili-core has no defensible record of
what the model accepts: a moving ``-latest`` alias whose capability changes
under it, or a model whose input modality is outside this vocabulary (e.g.
video).  Omission is a deliberate "not declared", not an oversight: it
degrades to the warning above rather than asserting a claim that would
produce either a false refusal or a false assurance.

Requesting a modality
---------------------
``load_model`` accepts ``required_input_modalities`` and calls
:func:`require_input_modality` for each before dispatch, so the refusal
happens at selection::

    from bili.iris.loaders.llm_loader import load_model

    llm = load_model(
        "remote_openai",
        model_name="gpt-4o",
        required_input_modalities=["image"],
    )

Routing
-------
:func:`models_supporting_input_modality` answers the other direction, which
cataloged models accept a given input kind, so a caller can *select* a
vision-capable model instead of guessing at one.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

LOGGER = logging.getLogger(__name__)

#: Plain text input.  Every model accepts it; it is declared explicitly so a
#: declared ``input_modalities`` value is always a complete statement.
TEXT = "text"

#: Still-image input (a photo, screenshot, chart, scanned page).
IMAGE = "image"

#: Audio input.  Part of the vocabulary so a catalog entry can declare it;
#: bili-core ships no audio content-part builder yet.
AUDIO = "audio"

#: The input kinds a catalog entry may declare.
KNOWN_INPUT_MODALITIES = frozenset({TEXT, IMAGE, AUDIO})

#: The catalog key this module reads.
CATALOG_KEY = "input_modalities"


class UnsupportedInputModalityError(ValueError):
    """A model or transport cannot accept a requested input modality.

    Raised at model selection when the catalog declares that the chosen model
    does not accept the requested input kind, and at a text-only transport
    boundary when a message carrying such a part reaches it.  A ``ValueError``
    subclass so existing ``except ValueError`` handling around ``load_model``
    keeps working.
    """


def _catalog_entry(model_type: str, model_name: Optional[str]) -> Optional[dict]:
    """Return the catalog entry for a model, or ``None`` when uncataloged."""
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
        if entry.get("model_id") == model_name:
            return entry
    return None


def model_input_modalities(
    model_type: str, model_name: Optional[str]
) -> Optional[frozenset]:
    """Return the input modalities a model declares, or ``None``.

    :param model_type: The provider type key (e.g. ``"remote_openai"``).
    :param model_name: The model id the loader was given.
    :returns: A frozenset of modality strings when the catalog entry declares
        them, or ``None`` when the model is uncataloged or the entry declares
        nothing (the "not declared" state; see the module docstring).
    :rtype: Optional[frozenset]
    """
    entry = _catalog_entry(model_type, model_name)
    if entry is None:
        return None
    declared = entry.get(CATALOG_KEY)
    if not declared:
        return None
    return frozenset(declared)


def supports_input_modality(
    model_type: str, model_name: Optional[str], modality: str
) -> Optional[bool]:
    """Return whether a model accepts *modality*, tri-state.

    :param model_type: The provider type key.
    :param model_name: The model id.
    :param modality: One of :data:`KNOWN_INPUT_MODALITIES`.
    :returns: ``True`` / ``False`` when the catalog declares the model's input
        modalities, ``None`` when it does not.
    :rtype: Optional[bool]
    """
    declared = model_input_modalities(model_type, model_name)
    if declared is None:
        return None
    return modality in declared


def require_input_modality(
    model_type: str, model_name: Optional[str], modality: str
) -> None:
    """Raise when the catalog declares that a model rejects *modality*.

    Called by ``load_model`` before dispatch so a caller that intends to send
    an image is refused at selection rather than at the provider call.  An
    undeclared model degrades to a warning; see the module docstring for why
    refusing there would be wrong.

    :param model_type: The provider type key being loaded.
    :param model_name: The model id being loaded.
    :param modality: The input kind the caller intends to send.
    :raises ValueError: If *modality* is not a known input modality.
    :raises UnsupportedInputModalityError: If the catalog declares the model's
        input modalities and *modality* is not among them.
    """
    if modality not in KNOWN_INPUT_MODALITIES:
        raise ValueError(
            f"Unknown input modality {modality!r}. "
            f"Known modalities: {', '.join(sorted(KNOWN_INPUT_MODALITIES))}."
        )

    declared = model_input_modalities(model_type, model_name)
    if declared is None:
        LOGGER.warning(
            "Model '%s' (%s) declares no input modalities in the catalog; "
            "cannot verify that it accepts %r input. Proceeding; the provider "
            "will reject the request if it does not.",
            model_name,
            model_type,
            modality,
        )
        return

    if modality in declared:
        return

    alternatives = models_supporting_input_modality(modality).get(model_type)
    hint = (
        f" Models of type '{model_type}' that do: {', '.join(alternatives)}."
        if alternatives
        else ""
    )
    raise UnsupportedInputModalityError(
        f"Model '{model_name}' ({model_type}) does not accept {modality!r} "
        f"input; it declares {sorted(declared)}.{hint}"
    )


def require_input_modalities(
    model_type: str,
    model_name: Optional[str],
    modalities: Union[str, Iterable[str]],
) -> None:
    """Apply :func:`require_input_modality` to each of *modalities*.

    :param model_type: The provider type key being loaded.
    :param model_name: The model id being loaded.
    :param modalities: A single modality string or an iterable of them.
    :raises UnsupportedInputModalityError: On the first unsupported modality.
    """
    if isinstance(modalities, str):
        modalities = [modalities]
    for modality in modalities:
        require_input_modality(model_type, model_name, modality)


def models_supporting_input_modality(modality: str) -> Dict[str, List[str]]:
    """Return the cataloged models that declare support for *modality*.

    The routing direction: a caller holding an image asks which models can
    take it, instead of picking one and hoping.

    :param modality: One of :data:`KNOWN_INPUT_MODALITIES`.
    :returns: ``{provider_type: [model_id, ...]}`` for every entry whose
        declared ``input_modalities`` contains *modality*.  Providers with no
        such model are omitted.  Entries that declare nothing are omitted too:
        this answers "known to accept it", never "might".
    :rtype: Dict[str, List[str]]
    """
    try:
        from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
            LLM_MODELS,
        )
    except ImportError:  # pragma: no cover - config always importable in practice
        return {}

    result: Dict[str, List[str]] = {}
    for provider_type, provider_info in LLM_MODELS.items():
        matches = [
            entry["model_id"]
            for entry in provider_info.get("models", [])
            if modality in (entry.get(CATALOG_KEY) or ()) and entry.get("model_id")
        ]
        if matches:
            result[provider_type] = matches
    return result


def describe_message_modalities(messages: Iterable[Any]) -> List[str]:
    """Return the non-text input modalities present in *messages*.

    Maps the content-part vocabulary onto the catalog vocabulary, so a caller
    can derive ``required_input_modalities`` from the messages it has actually
    built rather than restating them by hand.

    :param messages: An iterable of ``BaseMessage`` objects.
    :returns: A sorted list of modality strings (a subset of
        :data:`KNOWN_INPUT_MODALITIES`, excluding ``text``).
    :rtype: List[str]
    """
    from bili.iris.multimodal import (  # pylint: disable=import-outside-toplevel
        AUDIO_PART_TYPES,
        IMAGE_PART_TYPES,
        non_text_part_types,
    )

    found = set()
    for message in messages:
        for kind in non_text_part_types(getattr(message, "content", None)):
            if kind in IMAGE_PART_TYPES:
                found.add(IMAGE)
            elif kind in AUDIO_PART_TYPES:
                found.add(AUDIO)
    return sorted(found)
