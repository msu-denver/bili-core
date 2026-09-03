"""Tests for bili.iris.multimodal: content parts and message construction.

The negative space matters as much as the happy path here: the whole point of
the module is that a non-text part is never *silently* lost, so the tests below
pin both that an image survives and that a malformed request is refused by
name rather than producing a plausible-looking message.
"""

import base64

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bili.iris.multimodal import (
    AUDIO_PART_TYPES,
    IMAGE_PART_TYPES,
    NON_TEXT_PART_TYPES,
    MultimodalContentError,
    build_human_message,
    content_has_non_text_parts,
    content_text,
    image_part,
    image_part_from_path,
    is_image_part,
    is_non_text_part,
    is_text_part,
    message_has_non_text_parts,
    message_text,
    non_text_part_types,
    normalise_prompt,
    part_type,
    text_part,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n-not-a-real-png"
URL = "https://example.invalid/chart.png"


# ---------------------------------------------------------------------------
# text_part / image_part
# ---------------------------------------------------------------------------


class TestTextPart:
    """The text content part."""

    def test_builds_text_part(self):
        """A text span becomes a text content part."""
        assert text_part("hello") == {"type": "text", "text": "hello"}

    def test_empty_string_is_allowed(self):
        """Empty string is allowed."""
        assert text_part("") == {"type": "text", "text": ""}

    @pytest.mark.parametrize("value", [None, 42, ["hi"], {"text": "hi"}])
    def test_rejects_non_string(self, value):
        """A non-string is refused rather than coerced."""
        with pytest.raises(MultimodalContentError):
            text_part(value)


class TestImagePart:
    """The image content part, by URL and by inline data."""

    def test_url_form(self):
        """A URL builds the image_url part shape."""
        assert image_part(url=URL) == {
            "type": "image_url",
            "image_url": {"url": URL},
        }

    def test_url_is_positional(self):
        """The URL may be passed positionally."""
        assert image_part(URL) == image_part(url=URL)

    def test_detail_is_carried(self):
        """Detail is carried."""
        part = image_part(url=URL, detail="low")
        assert part["image_url"] == {"url": URL, "detail": "low"}

    def test_detail_omitted_when_not_given(self):
        """Detail omitted when not given."""
        assert "detail" not in image_part(url=URL)["image_url"]

    def test_bytes_are_encoded_as_a_data_uri(self):
        """Bytes are encoded as a data URI."""
        part = image_part(data=PNG_BYTES, mime_type="image/png")
        expected = base64.b64encode(PNG_BYTES).decode("ascii")
        assert part["image_url"]["url"] == f"data:image/png;base64,{expected}"

    def test_pre_encoded_string_is_used_verbatim(self):
        """A pre-encoded base64 string is used verbatim."""
        part = image_part(data="QUJD", mime_type="image/jpeg")
        assert part["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_neither_url_nor_data_is_refused(self):
        """Neither URL nor data is refused."""
        with pytest.raises(MultimodalContentError, match="exactly one"):
            image_part()

    def test_both_url_and_data_is_refused(self):
        """Both URL and data is refused."""
        with pytest.raises(MultimodalContentError, match="exactly one"):
            image_part(url=URL, data=PNG_BYTES, mime_type="image/png")

    def test_data_without_mime_type_is_refused(self):
        """bili-core will not guess a media type: a wrong guess fails opaquely."""
        with pytest.raises(MultimodalContentError, match="mime_type"):
            image_part(data=PNG_BYTES)


class TestImagePartFromPath:
    """Reading an image off disk."""

    def test_guesses_mime_from_suffix(self, tmp_path):
        """Guesses MIME from suffix."""
        path = tmp_path / "shot.png"
        path.write_bytes(PNG_BYTES)
        part = image_part_from_path(path)
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_accepts_a_string_path(self, tmp_path):
        """Accepts a string path."""
        path = tmp_path / "shot.png"
        path.write_bytes(PNG_BYTES)
        assert is_image_part(image_part_from_path(str(path)))

    def test_explicit_mime_type_overrides_the_guess(self, tmp_path):
        """Explicit MIME type overrides the guess."""
        path = tmp_path / "shot.bin"
        path.write_bytes(PNG_BYTES)
        part = image_part_from_path(path, mime_type="image/webp")
        assert part["image_url"]["url"].startswith("data:image/webp;base64,")

    def test_detail_is_carried(self, tmp_path):
        """Detail is carried."""
        path = tmp_path / "shot.png"
        path.write_bytes(PNG_BYTES)
        assert (
            image_part_from_path(path, detail="high")["image_url"]["detail"] == "high"
        )

    def test_unknown_suffix_is_refused(self, tmp_path):
        """Unknown suffix is refused."""
        path = tmp_path / "shot.unknownext"
        path.write_bytes(PNG_BYTES)
        with pytest.raises(MultimodalContentError, match="mime_type"):
            image_part_from_path(path)

    def test_non_image_media_type_is_refused(self, tmp_path):
        """A .txt guesses text/plain, which is not an image; refuse rather than send."""
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        with pytest.raises(MultimodalContentError, match="text/plain"):
            image_part_from_path(path)

    def test_missing_file_raises_oserror(self, tmp_path):
        """Missing file raises OSError."""
        with pytest.raises(OSError):
            image_part_from_path(tmp_path / "absent.png")


# ---------------------------------------------------------------------------
# build_human_message
# ---------------------------------------------------------------------------


class TestBuildHumanMessage:
    """Message construction, including the unchanged text-only path."""

    def test_text_only_builds_the_historical_string_message(self):
        """The backwards-compatibility claim: text alone is a plain str content."""
        message = build_human_message(text="Hello")
        assert isinstance(message, HumanMessage)
        assert message.content == "Hello"
        assert message.content == HumanMessage(content="Hello").content

    def test_text_plus_image_builds_parts(self):
        """Text plus image builds parts."""
        message = build_human_message(text="What is this?", images=[URL])
        assert message.content == [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": URL}},
        ]

    def test_image_only_omits_the_text_part(self):
        """Image only omits the text part."""
        message = build_human_message(images=[URL])
        assert message.content == [{"type": "image_url", "image_url": {"url": URL}}]

    def test_multiple_images_are_ordered_after_the_text(self):
        """Multiple images are ordered after the text."""
        message = build_human_message(text="two", images=[URL, URL + "?2"])
        assert [part_type(p) for p in message.content] == [
            "text",
            "image_url",
            "image_url",
        ]

    def test_prebuilt_part_dict_is_used_verbatim(self):
        """Prebuilt part dict is used verbatim."""
        part = image_part(url=URL, detail="low")
        message = build_human_message(images=[part])
        assert message.content == [part]

    def test_empty_images_iterable_is_the_text_only_path(self):
        """Empty images iterable is the text only path."""
        assert build_human_message(text="Hi", images=[]).content == "Hi"

    def test_content_is_passed_through(self):
        """Content is passed through."""
        parts = [text_part("a"), image_part(url=URL)]
        assert build_human_message(content=parts).content == parts

    def test_content_string_is_passed_through(self):
        """Content string is passed through."""
        assert build_human_message(content="plain").content == "plain"

    def test_content_with_text_is_refused(self):
        """Content with text is refused."""
        with pytest.raises(MultimodalContentError, match="not both"):
            build_human_message(text="a", content="b")

    def test_content_with_images_is_refused(self):
        """Content with images is refused."""
        with pytest.raises(MultimodalContentError, match="not both"):
            build_human_message(images=[URL], content="b")

    def test_nothing_supplied_is_refused(self):
        """Nothing supplied is refused."""
        with pytest.raises(MultimodalContentError, match="requires"):
            build_human_message()

    def test_non_image_part_dict_is_refused(self):
        """A part dict that is not an image is refused."""
        with pytest.raises(MultimodalContentError, match="Not an image"):
            build_human_message(images=[text_part("not an image")])

    @pytest.mark.parametrize("value", [42, None, b"bytes"])
    def test_uncoercible_image_is_refused(self, value):
        """A value no builder accepts is refused."""
        with pytest.raises(MultimodalContentError):
            build_human_message(images=[value])


# ---------------------------------------------------------------------------
# Part predicates
# ---------------------------------------------------------------------------


class TestPartPredicates:
    """What counts as text, as an image, and as non-text."""

    @pytest.mark.parametrize(
        "part,expected",
        [
            ({"type": "text", "text": "x"}, "text"),
            ({"type": "image_url", "image_url": {}}, "image_url"),
            ({"type": 7}, None),
            ({}, None),
            ("bare string part", None),
            (None, None),
        ],
    )
    def test_part_type(self, part, expected):
        """part_type reads the type of a content part, or None."""
        assert part_type(part) == expected

    def test_is_text_part(self):
        """is_text_part recognises a text part and nothing else."""
        assert is_text_part({"type": "text", "text": "x"})
        assert not is_text_part({"type": "image_url", "image_url": {}})

    @pytest.mark.parametrize("kind", sorted(IMAGE_PART_TYPES))
    def test_every_image_spelling_is_recognised(self, kind):
        """All three ecosystem spellings count, so a message built by any of
        them survives the flatteners."""
        assert is_image_part({"type": kind})
        assert is_non_text_part({"type": kind})

    @pytest.mark.parametrize("kind", sorted(AUDIO_PART_TYPES))
    def test_audio_is_non_text_but_not_an_image(self, kind):
        """Audio is non text but not an image."""
        assert is_non_text_part({"type": kind})
        assert not is_image_part({"type": kind})

    def test_file_part_is_non_text(self):
        """File part is non text."""
        assert is_non_text_part({"type": "file"})

    def test_unrecognised_part_is_not_claimed_as_non_text(self):
        """An unknown part keeps the existing text-coercion behaviour rather
        than being forwarded to a provider that may reject it."""
        assert not is_non_text_part({"type": "some_future_part"})

    def test_non_text_part_types_is_a_closed_union(self):
        """Non text part types is a closed union."""
        assert NON_TEXT_PART_TYPES == IMAGE_PART_TYPES | AUDIO_PART_TYPES | frozenset(
            {"file"}
        )


class TestContentHasNonTextParts:
    """The predicate the flatteners branch on."""

    def test_string_content_is_text(self):
        """String content is text."""
        assert content_has_non_text_parts("hello") is False

    def test_text_only_list_is_text(self):
        """This is the case the existing flatteners were written for; it must
        keep taking the coercion path."""
        assert content_has_non_text_parts([text_part("a"), text_part("b")]) is False

    def test_empty_list_is_text(self):
        """Empty content is not multimodal."""
        assert content_has_non_text_parts([]) is False

    def test_list_with_an_image_is_non_text(self):
        """List with an image is non text."""
        assert content_has_non_text_parts([text_part("a"), image_part(url=URL)]) is True

    def test_none_is_text(self):
        """Absent content is not multimodal."""
        assert content_has_non_text_parts(None) is False

    def test_message_form(self):
        """The message-level predicate agrees with the content-level one."""
        assert message_has_non_text_parts(HumanMessage(content="hi")) is False
        assert (
            message_has_non_text_parts(HumanMessage(content=[image_part(url=URL)]))
            is True
        )

    def test_message_form_tolerates_a_non_message(self):
        """Message form tolerates a non message."""
        assert message_has_non_text_parts(object()) is False


class TestNonTextPartTypes:
    """Naming what a text-only transport is being asked to carry."""

    def test_lists_distinct_types_in_order(self):
        """Lists distinct types in order."""
        content = [
            text_part("a"),
            {"type": "image_url"},
            {"type": "audio"},
            {"type": "image_url"},
        ]
        assert non_text_part_types(content) == ["image_url", "audio"]

    def test_text_only_is_empty(self):
        """Text only is empty."""
        assert non_text_part_types([text_part("a")]) == []

    def test_string_content_is_empty(self):
        """String content is empty."""
        assert non_text_part_types("hello") == []


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestMessageText:
    """Reading the words of a message without raising on list content."""

    def test_string_content(self):
        """String content is its own text."""
        assert message_text(HumanMessage(content="USER PROFILE: x")) == (
            "USER PROFILE: x"
        )

    def test_list_content_joins_text_parts_and_ignores_images(self):
        """List content joins text parts and ignores images."""
        message = HumanMessage(
            content=[text_part("USER "), image_part(url=URL), text_part("PROFILE: x")]
        )
        assert message_text(message) == "USER PROFILE: x"

    def test_image_only_content_has_no_text(self):
        """Image only content has no text."""
        assert message_text(HumanMessage(content=[image_part(url=URL)])) == ""

    def test_ai_message_works_too(self):
        """An AIMessage reads the same way a HumanMessage does."""
        assert message_text(AIMessage(content="reply")) == "reply"

    def test_missing_content_is_empty(self):
        """Missing content is empty."""
        assert message_text(object()) == ""

    def test_content_text_on_an_unexpected_type(self):
        """Content text on an unexpected type."""
        assert content_text(42) == ""

    def test_content_text_tolerates_a_text_part_without_a_text_key(self):
        """Content text tolerates a text part without a text key."""
        assert content_text([{"type": "text"}]) == ""


# ---------------------------------------------------------------------------
# normalise_prompt
# ---------------------------------------------------------------------------


class TestNormalisePrompt:
    """What the widened entry points accept."""

    def test_string_is_returned_unchanged(self):
        """String is returned unchanged."""
        assert normalise_prompt("Hello") == "Hello"

    def test_list_is_returned_as_a_list(self):
        """List is returned as a list."""
        parts = [text_part("a"), image_part(url=URL)]
        assert normalise_prompt(parts) == parts

    def test_tuple_becomes_a_list(self):
        """Tuple becomes a list."""
        assert normalise_prompt((text_part("a"),)) == [text_part("a")]

    def test_bytes_is_refused(self):
        """bytes is a Sequence; accepting it would forward a byte payload to
        the provider as a list of integers."""
        with pytest.raises(MultimodalContentError):
            normalise_prompt(b"raw")

    @pytest.mark.parametrize("value", [42, None, {"type": "text"}])
    def test_other_types_are_refused(self, value):
        """Other types are refused."""
        with pytest.raises(MultimodalContentError):
            normalise_prompt(value)
