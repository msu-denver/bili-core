"""Image delivery for CLI (subprocess) providers.

A CLI LLM tool consumes a single text prompt, so an image cannot ride inside
the request the way it does for a message-based provider.  Every CLI harness
bili-core presets, though, sits on a vision-capable model and can open a file
it is pointed at.  This module is the seam that turns an image content part
into something such a harness can actually read: the bytes are written into
the directory the subprocess runs in, the invocation is rewritten to point at
that file, and the file is removed once the call returns.

Two honest consequences
-----------------------
**The delivery kind is different, and it is reported.**  A message-based
provider is handed the BYTES; a CLI harness is merely OFFERED A PATH, and
whether the agent behind it actually opened that file is not verifiable from
the response text.  Those are different facts about a turn, so they are
reported under different names
(:data:`~bili.iris.providers.modality.IMAGE_DELIVERY_BYTES` versus
:data:`~bili.iris.providers.modality.IMAGE_DELIVERY_OFFERED_BY_PATH`) rather
than collapsed into one "the image was sent" claim.

**A harness with no file-read route still refuses.**  The route is a property
of a specific CLI tool, not of "CLI tools" in general: the generic
``cli`` provider type drives an arbitrary executable that bili-core knows
nothing about.  With no :class:`CliImageRoute` configured, the pre-existing
:exc:`~bili.iris.providers.modality.UnsupportedInputModalityError` refusal is
what happens, unchanged.

Where the file is written
-------------------------
Into the working directory the subprocess is already pinned to (the
provider's configured ``cwd``, or the calling process's cwd when it has
none), never a system temporary directory.  Two reasons, and the first is
decisive:

1. A CLI harness commonly gates filesystem access by directory.  A path
   outside the directory it was pointed at is one it may simply refuse to
   open, so a temp-dir path can produce a turn that looks successful while
   the agent reports it could not read the file.  The directory the
   subprocess runs in is the one place it is known to reach.
2. That directory is the workspace the caller already chose and consented to
   for these subprocesses.  Writing there exposes nothing that was not
   already exposed; scattering image bytes into a shared system temp
   directory would.

The file is referenced by its bare filename rather than an absolute path, so
the prompt (which is sent to a third party) carries no part of the host's
directory layout.

Filename
--------
Generated, neutral, and origin-free: a fixed prefix, a random token, and an
extension derived from the part's own declared media type.  Nothing from the
image's source (an original filename, a URL, a caller-supplied label) reaches
the filename, because the filename is visible to the harness and, through the
prompt, to the model behind it.

Cleanup
-------
:func:`materialized_images` removes what it wrote on the way out of the
``with`` block, on the success path and on the failure path alike, so a raised
:exc:`CliLLMError`, a timeout, or a cancellation does not leave image bytes
behind in the caller's workspace.  Materialization wraps the provider's whole
retry loop rather than one attempt, because a file deleted between attempts
would leave the retry pointing at nothing.
"""

import base64
import binascii
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from bili.iris.multimodal import is_image_part, part_type

LOGGER = logging.getLogger(__name__)

#: Prefix for every materialized image filename.  Present so a stray file is
#: attributable to bili-core by a human reading a directory listing; it says
#: nothing about the image itself.
IMAGE_FILENAME_PREFIX = "bili-image-"

#: Media type -> file extension, for the image types a CLI harness is likely
#: to recognise.  An explicit table rather than :func:`mimetypes.guess_extension`
#: alone because that function reads the host's MIME registry, so the
#: extension of a written file would vary by machine; the harness decides what
#: it will open partly from that extension, so it must not.
_MEDIA_TYPE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/svg+xml": ".svg",
}

#: The ``data:`` URI scheme prefix an inline image part carries.
_DATA_URI_PREFIX = "data:"


class CliImageMaterializationError(RuntimeError):
    """An image part could not be written to the subprocess's workspace.

    Raised when an image part carries no inline bytes this transport can
    write (a remote URL), when its media type is unknown, or when the write
    itself fails.

    Deliberately **not** named in the fallback engine's default retryable set
    (:data:`bili.iris.providers.fallback._DEFAULT_RETRYABLE_NAMES`): every
    cause is a local, permanent condition (an unwritable directory, an
    unsupported part shape) that another attempt or another provider does not
    fix, and silently falling through would hide a misconfigured workspace
    behind a slower answer.  A caller that would rather re-route such a turn
    can add this class to its own ``FallbackPolicy``.
    """


@dataclass(frozen=True)
class CliImageRoute:
    """How one CLI harness is pointed at an image file on disk.

    Each harness exposes a different mechanism, and the mechanism is a fact
    about that tool rather than a preference, so it lives beside the argv and
    cwd the provider already owns for that tool (see
    :mod:`bili.iris.providers.cli_presets`).

    :param name: Identifier for the route, used in log lines.
    :param argv_template: Argument tokens added to the command line, rendered
        once per image with ``{path}`` replaced by the image's filename.
        ``None`` when the harness takes no image flag.

        A flag whose value is *attached* (``--flag=value``) is strongly
        preferred over the separated form (``--flag value``): a variadic
        image flag consumes following words, so the separated form can
        swallow the prompt positional that comes after it.
    :param prompt_template: A fragment added to the prompt text, rendered once
        per image with ``{path}`` replaced by the image's filename.  ``None``
        when the harness takes the image through argv alone.  Used both for a
        harness whose file reference is prompt syntax and for one that must be
        *instructed* to open the file with its own read tool.
    :param prompt_separator: What joins the rendered fragments to the original
        prompt text.  Per-harness because the verified shapes differ: an
        inline reference token reads as part of the sentence, an instruction
        reads as its own paragraph.
    :param verified_against: The tool and version this route was checked
        against, so a later reader can tell a verified mechanism from an
        assumed one and knows what to re-check.  These are facts about
        third-party software that nothing in bili-core can detect changing.
    """

    name: str
    argv_template: Optional[Tuple[str, ...]] = None
    prompt_template: Optional[str] = None
    prompt_separator: str = "\n\n"
    verified_against: str = ""

    def __post_init__(self) -> None:
        if not self.argv_template and not self.prompt_template:
            raise ValueError(
                f"CliImageRoute {self.name!r} declares neither argv_template nor "
                "prompt_template, so it points the harness at nothing. A harness "
                "with no file-read route must configure no route at all, which "
                "keeps the image refusal."
            )


@dataclass(frozen=True)
class MaterializedImage:
    """One image written into the subprocess's working directory.

    :param path: Absolute path of the written file, used for cleanup.
    :param filename: The bare filename, which is what the invocation
        references so no part of the host's directory layout reaches the
        prompt.
    """

    path: str
    filename: str


@dataclass(frozen=True)
class ImagePayload:
    """Decoded image bytes plus the media type they were declared as.

    :param data: The raw image bytes.
    :param media_type: The declared media type (e.g. ``"image/png"``), which
        decides the written file's extension.
    """

    data: bytes = field(repr=False)
    media_type: str


def _extension_for(media_type: str) -> str:
    """Return the file extension for *media_type*.

    :param media_type: A declared image media type.
    :returns: A dot-prefixed extension.
    :raises CliImageMaterializationError: When the media type is not a known
        image type.  Writing the bytes under a guessed or generic extension
        would hand the harness a file it cannot identify, which fails as an
        unhelpful "I could not read that" mid-turn rather than here.
    """
    normalised = (media_type or "").split(";", 1)[0].strip().lower()
    extension = _MEDIA_TYPE_EXTENSIONS.get(normalised)
    if extension:
        return extension
    raise CliImageMaterializationError(
        f"Cannot write an image of media type {media_type!r} for a CLI harness: "
        f"no known file extension. Supported: "
        f"{', '.join(sorted(_MEDIA_TYPE_EXTENSIONS))}."
    )


def _decode_data_uri(url: str) -> ImagePayload:
    """Decode a ``data:<media-type>;base64,<payload>`` URI.

    :param url: The data URI.
    :returns: The decoded :class:`ImagePayload`.
    :raises CliImageMaterializationError: When the URI is not base64-encoded
        or its payload does not decode.
    """
    header, _, payload = url[len(_DATA_URI_PREFIX) :].partition(",")
    if not header.lower().endswith(";base64"):
        raise CliImageMaterializationError(
            "Only base64-encoded data: URIs can be written to a file for a CLI "
            f"harness; got a URI declared as {header!r}."
        )
    media_type = header[: -len(";base64")]
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CliImageMaterializationError(
            f"The image part's base64 payload could not be decoded: {exc}"
        ) from exc
    return ImagePayload(data=data, media_type=media_type)


def _payload_from_url_value(url: Any) -> ImagePayload:
    """Turn the ``url``-shaped value of an image part into a payload.

    :param url: The value carried by an ``image_url`` part, either a
        ``data:`` URI or a remote URL.
    :returns: The decoded :class:`ImagePayload`.
    :raises CliImageMaterializationError: When the value is a remote URL or
        is not a string.
    """
    if not isinstance(url, str):
        raise CliImageMaterializationError(
            f"An image part's url must be a string, got {type(url).__name__}."
        )
    if url.startswith(_DATA_URI_PREFIX):
        return _decode_data_uri(url)
    raise CliImageMaterializationError(
        "A CLI harness is handed an image as a file, so the image part must "
        "carry its bytes inline; this one carries a remote URL, which "
        "bili-core would have to fetch. Build the part from local bytes "
        "instead (bili.iris.multimodal.image_part_from_path, or "
        "image_part(data=..., mime_type=...)), or route this turn to a "
        "provider that accepts a URL directly."
    )


def image_payload(part: Any) -> ImagePayload:
    """Extract writable image bytes from an image content part.

    Recognises the same part shapes as :mod:`bili.iris.multimodal`: the
    OpenAI-style ``image_url`` part (whose ``image_url`` is either a dict
    carrying ``url`` or a bare string), the ``langchain-core`` standard
    ``image`` block (``source_type`` of ``base64``), and the Responses-API
    ``input_image`` spelling.

    :param part: An image content part.
    :returns: The decoded :class:`ImagePayload`.
    :raises CliImageMaterializationError: When *part* is not an image part, or
        carries no inline bytes this transport can write.
    """
    if not is_image_part(part):
        raise CliImageMaterializationError(
            f"Not an image content part: type={part_type(part)!r}."
        )

    # langchain-core standard image block: {"type": "image", "source_type":
    # "base64", "data": "...", "mime_type": "image/png"}.
    if part.get("source_type") == "base64" or ("data" in part and "mime_type" in part):
        media_type = part.get("mime_type") or part.get("media_type")
        if not media_type:
            raise CliImageMaterializationError(
                "An inline image block must declare its media type "
                "(mime_type); bili-core does not guess one from raw bytes."
            )
        raw = part.get("data")
        if isinstance(raw, bytes):
            return ImagePayload(data=raw, media_type=media_type)
        if isinstance(raw, str):
            try:
                return ImagePayload(
                    data=base64.b64decode(raw, validate=True), media_type=media_type
                )
            except (binascii.Error, ValueError) as exc:
                raise CliImageMaterializationError(
                    f"The image block's base64 data could not be decoded: {exc}"
                ) from exc
        raise CliImageMaterializationError(
            f"An inline image block's data must be bytes or a base64 string, "
            f"got {type(raw).__name__}."
        )

    value = part.get("image_url", part.get("url"))
    if isinstance(value, dict):
        value = value.get("url")
    return _payload_from_url_value(value)


def image_payloads(content: Any) -> List[ImagePayload]:
    """Return a payload for every image part in a message ``content`` value.

    :param content: A message ``content`` value.
    :returns: One :class:`ImagePayload` per image part, in order.  Empty for
        string content and for a parts list carrying no image.
    """
    if not isinstance(content, list):
        return []
    return [image_payload(part) for part in content if is_image_part(part)]


@contextmanager
def materialized_images(
    payloads: Sequence[ImagePayload], directory: Optional[str] = None
) -> Iterator[List[MaterializedImage]]:
    """Write *payloads* into *directory* for the duration of the block.

    Every file written is removed on the way out, whether the block completed
    or raised, so a failed CLI call leaves no image bytes behind.  A partial
    write (the second of three images fails) removes the files already
    written for the same reason.

    :param payloads: The images to write.
    :param directory: The directory to write into.  ``None`` means the
        calling process's current working directory, which is the directory
        the subprocess inherits when the provider pins no ``cwd``.
    :yields: One :class:`MaterializedImage` per payload, in order.
    :raises CliImageMaterializationError: When a media type is unknown or the
        write fails.
    """
    target = Path(directory) if directory else Path.cwd()
    written: List[MaterializedImage] = []
    try:
        for payload in payloads:
            filename = (
                f"{IMAGE_FILENAME_PREFIX}{uuid.uuid4().hex}"
                f"{_extension_for(payload.media_type)}"
            )
            path = target / filename
            try:
                path.write_bytes(payload.data)
            except OSError as exc:
                raise CliImageMaterializationError(
                    f"Could not write an image into the CLI subprocess's working "
                    f"directory {str(target)!r}: {exc}. This is the directory the "
                    "subprocess runs in and the only one it is known to be able "
                    "to read."
                ) from exc
            LOGGER.debug(
                "CliLLM: materialized an image as %s (%d bytes)",
                filename,
                len(payload.data),
            )
            written.append(MaterializedImage(path=str(path), filename=filename))
        yield written
    finally:
        for image in written:
            try:
                os.unlink(image.path)
            except OSError as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "CliLLM: could not remove the materialized image %s: %s",
                    image.path,
                    exc,
                )


def apply_route(
    route: CliImageRoute, prompt: str, images: Sequence[MaterializedImage]
) -> Tuple[str, List[str]]:
    """Rewrite the invocation so the harness is pointed at *images*.

    :param route: The harness's file-read route.
    :param prompt: The rendered prompt text.
    :param images: The materialized images, referenced by bare filename.
    :returns: ``(prompt, extra_argv)`` -- the prompt to send and the argument
        tokens to add to the command line.  With no images, the prompt is
        returned unchanged and ``extra_argv`` is empty, so a text-only turn
        through a route-carrying preset is byte-for-byte what it always was.
    """
    if not images:
        return prompt, []

    extra_argv: List[str] = []
    if route.argv_template:
        for image in images:
            extra_argv.extend(
                token.replace("{path}", image.filename) for token in route.argv_template
            )

    if route.prompt_template:
        fragments = [
            route.prompt_template.replace("{path}", image.filename) for image in images
        ]
        reference = " ".join(fragments)
        # With no caller text there is nothing to separate the reference
        # from, and joining anyway sends the harness a trailing separator
        # followed by an empty prompt.
        prompt = (
            route.prompt_separator.join([reference, prompt]) if prompt else reference
        )

    return prompt, extra_argv
