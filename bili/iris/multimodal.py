"""Multimodal message construction for bili-core.

This module is the single seam through which a caller attaches a non-text
input part (today: an image) to a message that an IRIS or AETHER run will
send to a model.

Why this exists
---------------
bili-core's provider contract is already modality-agnostic: a provider takes
a list of ``langchain_core.messages.BaseMessage`` and returns a response with
a ``.content`` attribute, and ``HumanMessage.content`` is
``Union[str, list]``.  A message whose content is a list of content parts
therefore reaches a vision-capable chat model with no per-provider change.
What was missing was everything above the provider: the public entry points
typed their input as ``str`` and wrapped it as a plain-string message, so a
caller had no supported affordance for attaching an image, and the
orchestration flatteners coerced list-shaped content back to text before the
model call, which dropped any non-text part.

This module supplies the construction half (content parts and the message
builder); the flatteners consult :func:`message_has_non_text_parts` so an
image-bearing message is carried through instead of stringified.

Content-part vocabulary
-----------------------
A content part is a plain ``dict`` carrying a ``"type"`` key.  bili-core does
not define its own wire format: it recognises the shapes the LangChain
ecosystem already exchanges, so a part built here is passed to the provider
verbatim.

============================  ==========================================
Part ``type``                  Meaning
============================  ==========================================
``text``                       A text span (:func:`text_part`).
``image_url`` / ``image`` /    An image, by URL or inline base64 data
``input_image``                (:func:`image_part`).
``audio`` / ``input_audio``    Audio input.  Recognised as non-text so it
                               is never silently dropped; bili-core ships
                               no builder for it yet.
``file``                       A file attachment.  Recognised, no builder.
============================  ==========================================

:func:`image_part` emits the ``image_url`` form, which is the shape the
widest set of LangChain chat integrations accept directly.  ``langchain-core``
normalises it to its own standard ``image`` block on
``BaseMessage.content_blocks``, so both spellings are recognised on the way
back in.

Routing
-------
Building an image-bearing message does not by itself guarantee the selected
model accepts one.  That is a separate, per-model question answered by
:mod:`bili.iris.providers.modality`, which reads the catalog's declared
``input_modalities`` and lets ``load_model`` reject a text-only model up
front rather than at the provider call.

Usage::

    from bili.iris.multimodal import build_human_message, image_part
    from bili.iris.loaders.streaming_utils import invoke_agent

    message = build_human_message(
        text="What is in this picture?",
        images=["https://example.invalid/chart.png"],
    )
    answer = invoke_agent(agent, message.content, thread_id="user1")

    # Equivalent, building the parts explicitly:
    parts = [
        {"type": "text", "text": "What is in this picture?"},
        image_part(url="https://example.invalid/chart.png"),
    ]
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)

#: Part types carrying plain text.
TEXT_PART_TYPES = frozenset({"text"})

#: Part types carrying an image.  ``image_url`` is the OpenAI-style shape
#: :func:`image_part` emits; ``image`` is ``langchain-core``'s standard
#: content block; ``input_image`` is the Responses-API spelling.  All three
#: are recognised so a message built by any of them survives the flatteners.
IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})

#: Part types carrying audio.  Recognised (so such a part is never silently
#: stringified) even though bili-core ships no builder for one.
AUDIO_PART_TYPES = frozenset({"audio", "input_audio"})

#: Part types carrying an opaque file attachment.
FILE_PART_TYPES = frozenset({"file"})

#: Every recognised non-text part type.
NON_TEXT_PART_TYPES = IMAGE_PART_TYPES | AUDIO_PART_TYPES | FILE_PART_TYPES

#: Content accepted by :func:`build_human_message` and by the widened entry
#: points: either a plain string (unchanged behaviour) or a list of parts.
MessageContent = Union[str, List[Any]]


class MultimodalContentError(ValueError):
    """A content part could not be built from the arguments supplied."""


# ---------------------------------------------------------------------------
# Part builders
# ---------------------------------------------------------------------------


def text_part(text: str) -> dict:
    """Return a text content part.

    :param text: The text span.
    :returns: ``{"type": "text", "text": <text>}``.
    :raises MultimodalContentError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise MultimodalContentError(
            f"text_part() requires a str, got {type(text).__name__}."
        )
    return {"type": "text", "text": text}


def image_part(
    url: Optional[str] = None,
    *,
    data: Optional[Union[bytes, str]] = None,
    mime_type: Optional[str] = None,
    detail: Optional[str] = None,
) -> dict:
    """Return an image content part, by URL or by inline data.

    Exactly one of *url* and *data* must be supplied.  Inline data is encoded
    as a ``data:`` URI, which is how every LangChain chat integration that
    accepts an ``image_url`` part takes raw bytes.

    :param url: An ``http(s)://`` or ``data:`` URL for the image.
    :param data: Raw image bytes, or an already-base64-encoded ``str``.
    :param mime_type: The image media type (e.g. ``"image/png"``).  Required
        with *data*: bili-core will not guess a media type from bytes, because
        a wrong guess is rejected by the provider with an opaque error.
    :param detail: Optional provider-specific fidelity hint (e.g. ``"low"``).
    :returns: An ``image_url`` content part dict.
    :raises MultimodalContentError: If neither or both of *url* and *data* are
        supplied, or if *data* is supplied without *mime_type*.
    """
    if (url is None) == (data is None):
        raise MultimodalContentError(
            "image_part() requires exactly one of url= or data=."
        )

    if data is not None:
        if not mime_type:
            raise MultimodalContentError(
                "image_part(data=...) requires mime_type= (e.g. 'image/png'); "
                "bili-core does not guess a media type from raw bytes."
            )
        encoded = (
            base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
        )
        url = f"data:{mime_type};base64,{encoded}"

    image_url: dict = {"url": url}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def image_part_from_path(
    path: Union[str, Path],
    *,
    mime_type: Optional[str] = None,
    detail: Optional[str] = None,
) -> dict:
    """Return an image content part built from a file on disk.

    :param path: Path to the image file.
    :param mime_type: Media type override.  Guessed from the file suffix when
        omitted; a suffix that does not map to an ``image/*`` type raises
        rather than sending an unlabelled payload to the provider.
    :param detail: Optional provider-specific fidelity hint.
    :returns: An ``image_url`` content part dict carrying a ``data:`` URI.
    :raises MultimodalContentError: If the media type cannot be determined.
    :raises OSError: If the file cannot be read.
    """
    file_path = Path(path)
    resolved = mime_type or mimetypes.guess_type(file_path.name)[0]
    if not resolved or not resolved.startswith("image/"):
        raise MultimodalContentError(
            f"Cannot determine an image media type for {file_path.name!r}"
            f"{f' (guessed {resolved!r})' if resolved else ''}. "
            "Pass mime_type= explicitly."
        )
    return image_part(data=file_path.read_bytes(), mime_type=resolved, detail=detail)


def _coerce_image(image: Any) -> dict:
    """Coerce one caller-supplied image into a content part.

    Accepts an already-built part dict (returned unchanged) or a URL/``data:``
    URI string.
    """
    if isinstance(image, dict):
        if part_type(image) not in IMAGE_PART_TYPES:
            raise MultimodalContentError(
                f"Not an image content part: type={part_type(image)!r}. "
                f"Expected one of {sorted(IMAGE_PART_TYPES)}."
            )
        return image
    if isinstance(image, str):
        return image_part(url=image)
    raise MultimodalContentError(
        f"Cannot build an image part from {type(image).__name__}; pass a URL "
        "string or a content-part dict (see image_part / image_part_from_path)."
    )


def build_human_message(
    text: Optional[str] = None,
    images: Optional[Iterable[Any]] = None,
    *,
    content: Optional[MessageContent] = None,
) -> Any:
    """Build a ``HumanMessage``, optionally carrying image parts.

    With *text* only, this produces exactly ``HumanMessage(content=<text>)``,
    the same message every text-only caller has always built, so a text-only
    call is unchanged.  Supplying *images* (or a list *content*) produces the
    multimodal list form.

    :param text: The text span of the turn.
    :param images: Images to attach.  Each entry is a URL/``data:`` URI string
        or an image content-part dict (from :func:`image_part` or
        :func:`image_part_from_path`).
    :param content: A pre-built content value (string or parts list) used
        verbatim.  Mutually exclusive with *text*/*images*.
    :returns: A ``langchain_core.messages.HumanMessage``.
    :raises MultimodalContentError: If nothing was supplied, if *content* is
        combined with *text*/*images*, or if an image cannot be coerced.
    """
    from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
        HumanMessage,
    )

    if content is not None:
        if text is not None or images is not None:
            raise MultimodalContentError(
                "build_human_message() takes either content= or text=/images=, "
                "not both."
            )
        return HumanMessage(content=content)

    image_parts = [_coerce_image(image) for image in (images or [])]
    if not image_parts:
        if text is None:
            raise MultimodalContentError(
                "build_human_message() requires text=, images=, or content=."
            )
        # Byte-for-byte the historical text-only message.
        return HumanMessage(content=text)

    parts: List[Any] = []
    if text is not None:
        parts.append(text_part(text))
    parts.extend(image_parts)
    return HumanMessage(content=parts)


# ---------------------------------------------------------------------------
# Part predicates
# ---------------------------------------------------------------------------


def part_type(part: Any) -> Optional[str]:
    """Return the ``"type"`` of a content part, or ``None``.

    ``None`` for anything that is not a dict carrying a string ``"type"``:
    a bare string part, or an unrecognised object.
    """
    if isinstance(part, dict):
        value = part.get("type")
        return value if isinstance(value, str) else None
    return None


def is_text_part(part: Any) -> bool:
    """Return whether *part* is a recognised text content part."""
    return part_type(part) in TEXT_PART_TYPES


def is_image_part(part: Any) -> bool:
    """Return whether *part* is a recognised image content part."""
    return part_type(part) in IMAGE_PART_TYPES


def is_non_text_part(part: Any) -> bool:
    """Return whether *part* is a recognised non-text content part.

    Only *recognised* non-text types count.  An unrecognised part is not
    claimed to be multimodal, so list content that carries one keeps the
    existing text-coercion behaviour rather than being forwarded to a
    provider that may reject it.
    """
    return part_type(part) in NON_TEXT_PART_TYPES


def content_has_non_text_parts(content: Any) -> bool:
    """Return whether a message ``content`` value carries a non-text part.

    ``False`` for string content and for a list of text-only parts, which is
    what keeps the existing flatteners byte-for-byte unchanged on the
    text-only path they were written for.
    """
    if not isinstance(content, list):
        return False
    return any(is_non_text_part(part) for part in content)


def message_has_non_text_parts(message: Any) -> bool:
    """Return whether *message* carries a non-text content part."""
    return content_has_non_text_parts(getattr(message, "content", None))


def non_text_part_types(content: Any) -> List[str]:
    """Return the distinct recognised non-text part types in *content*.

    Used to name what a text-only transport is being asked to carry, so the
    refusal says which modality it cannot accept.
    """
    if not isinstance(content, list):
        return []
    seen: List[str] = []
    for part in content:
        kind = part_type(part)
        if kind in NON_TEXT_PART_TYPES and kind not in seen:
            seen.append(kind)
    return seen


def message_text(message: Any) -> str:
    """Return the text of *message*, ignoring any non-text parts.

    For string content this is the content itself.  For list content it is
    the concatenation of the text parts.  Callers that only need to inspect
    the words of a message (a prefix check, a summary heuristic) use this so
    list-shaped content does not raise.
    """
    return content_text(getattr(message, "content", ""))


def content_text(content: Any) -> str:
    """Return the text carried by a message ``content`` value.

    :param content: A string, a list of content parts, or anything else.
    :returns: The string itself, the joined text parts, or ``""``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if is_text_part(part))
    return ""


def normalise_prompt(prompt: Union[str, Sequence[Any]]) -> MessageContent:
    """Validate and return a caller-supplied prompt as message content.

    Accepts the historical ``str`` unchanged, or a sequence of content parts
    which is returned as a list.

    :param prompt: A prompt string or a sequence of content parts.
    :returns: The value to hand to ``HumanMessage(content=...)``.
    :raises MultimodalContentError: If *prompt* is neither.
    """
    if isinstance(prompt, str):
        return prompt
    # Deliberately (list, tuple) rather than Sequence: ``str`` and ``bytes``
    # are both Sequences, so an abstract check would accept a bytes payload
    # and forward it to the provider as a "parts list" of integers.
    if isinstance(prompt, (list, tuple)):
        return list(prompt)
    raise MultimodalContentError(
        f"prompt must be a str or a list of content parts, got "
        f"{type(prompt).__name__}."
    )
