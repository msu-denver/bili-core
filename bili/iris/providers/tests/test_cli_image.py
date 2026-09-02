"""Tests for delivering an image to a CLI provider by materializing a file.

The property under test is not "an image reaches the model" -- a CLI harness
is a subprocess whose behaviour is not observable from here -- but the four
things bili-core is actually responsible for: the file exists where the
subprocess can read it while the call runs, it is gone afterwards on both the
success and the failure path, nothing about the image's origin leaks into the
filename or the prompt, and the weaker delivery kind is reported rather than
implied.
"""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
import base64
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.loaders.llm_loader import load_model
from bili.iris.multimodal import image_part, image_part_from_path, text_part
from bili.iris.providers.cli_image import (
    IMAGE_FILENAME_PREFIX,
    CliImageMaterializationError,
    CliImageRoute,
    ImagePayload,
    apply_route,
    image_payload,
    image_payloads,
    materialized_images,
)
from bili.iris.providers.cli_presets import (
    CLAUDE_CODE_IMAGE_ROUTE,
    CLI_PRESET_CATALOG,
    CODEX_IMAGE_ROUTE,
    GEMINI_CLI_IMAGE_ROUTE,
)
from bili.iris.providers.cli_provider import (
    CliLLM,
    CliLLMError,
    CliProvider,
    messages_rendered_by,
)
from bili.iris.providers.modality import (
    IMAGE_DELIVERY_OFFERED_BY_PATH,
    IMAGE_DELIVERY_RESPONSE_KEY,
    UnsupportedInputModalityError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: A real 1x1 PNG, so a test that writes it produces a file a tool would
#: actually recognise rather than arbitrary bytes with a .png suffix.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _png_part(**kwargs):
    """Return an image content part carrying PNG_BYTES inline."""
    return image_part(data=PNG_BYTES, mime_type="image/png", **kwargs)


def _human_with_image(text="what is this?", **kwargs):
    """Return a HumanMessage carrying text plus one inline PNG."""
    return HumanMessage(content=[text_part(text), _png_part(**kwargs)])


def _completed(stdout="answer", returncode=0, stderr=""):
    """Return a stand-in for subprocess.CompletedProcess."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def _capture_argv(seen):
    """Return a subprocess.run stand-in that records the argv it was given.

    A plain lambda cannot do this: ``seen.setdefault(...) or _completed()``
    returns the stored list whenever it is non-empty, so the caller gets a
    list where it expects a CompletedProcess and the test fails for a reason
    that has nothing to do with the code under test.
    """

    def _run(cmd, **_kwargs):
        """Record the argv and return a successful result."""
        seen["argv"] = list(cmd)
        return _completed()

    return _run


def _capture_workspace(seen, tmp_path):
    """Return a subprocess.run stand-in recording the workspace mid-call."""

    def _run(*_args, **kwargs):
        """Record what exists in the workspace while the call is running."""
        seen["names"] = _materialized_names(tmp_path)
        seen["cwd"] = kwargs.get("cwd")
        return _completed()

    return _run


def _routed_llm(tmp_path, route=CLAUDE_CODE_IMAGE_ROUTE, **kwargs):
    """Return a CliLLM pinned to tmp_path with an image route configured."""
    config = {
        "command": ["some-cli"],
        "prompt_via": "arg",
        "cwd": str(tmp_path),
        "image_route": route,
        "max_retries": 0,
    }
    config.update(kwargs)
    return CliProvider().load(**config)


def _materialized_names(tmp_path):
    """Return the names of the materialized images currently on disk."""
    return [
        p.name for p in tmp_path.iterdir() if p.name.startswith(IMAGE_FILENAME_PREFIX)
    ]


# ---------------------------------------------------------------------------
# The refusal, which is what happens without a route
# ---------------------------------------------------------------------------


class TestAHarnessWithNoFileReadRouteStillRefuses:
    """The negative space the whole mechanism is scoped against.

    A route is a fact about one specific tool.  The generic CLI provider
    drives an arbitrary executable, so bili-core has no basis for believing
    it can open a file, and dropping the image into the prompt would produce
    a turn that looks successful and silently answered without it.
    """

    def test_an_image_is_refused_when_no_route_is_configured(self, tmp_path):
        """No route means the pre-existing refusal, unchanged."""
        llm = CliProvider().load(command=["some-cli"], cwd=str(tmp_path))
        assert llm.image_route is None
        with pytest.raises(UnsupportedInputModalityError):
            llm._call_cli([_human_with_image()])

    def test_the_refusal_still_names_the_modality(self, tmp_path):
        """The caller learns what was refused, not merely that it failed."""
        llm = CliProvider().load(command=["some-cli"], cwd=str(tmp_path))
        with pytest.raises(UnsupportedInputModalityError) as excinfo:
            llm._call_cli([_human_with_image()])
        assert "image" in str(excinfo.value)

    def test_nothing_is_written_when_the_image_is_refused(self, tmp_path):
        """A refusal materializes nothing; the refusal comes first."""
        llm = CliProvider().load(command=["some-cli"], cwd=str(tmp_path))
        with pytest.raises(UnsupportedInputModalityError):
            llm._call_cli([_human_with_image()])
        assert list(tmp_path.iterdir()) == []

    def test_stripping_a_preset_route_restores_the_refusal(self, tmp_path):
        """A caller can opt a preset back out of image delivery."""
        llm = load_model("cli_claude_code", cwd=str(tmp_path), image_route=None)
        with pytest.raises(UnsupportedInputModalityError):
            llm._call_cli([_human_with_image()])

    def test_a_route_declaring_no_mechanism_is_refused_at_construction(self):
        """A route that points the harness at nothing is not a route.

        Silently accepting one would produce the exact failure the refusal
        exists to prevent: the image is stripped out of the prompt for
        delivery through a channel that does not exist.
        """
        with pytest.raises(ValueError, match="neither"):
            CliImageRoute(name="empty")


# ---------------------------------------------------------------------------
# Materialization: where the file goes, and that it goes away
# ---------------------------------------------------------------------------


class TestTheFileIsWrittenWhereTheSubprocessCanReadIt:
    """A CLI harness commonly gates filesystem access by directory."""

    def test_the_image_lands_in_the_configured_working_directory(self, tmp_path):
        """The image is written into the cwd the subprocess is pinned to."""
        llm = _routed_llm(tmp_path)
        seen = {}

        def _record(*_args, **kwargs):
            seen["cwd"] = kwargs.get("cwd")
            seen["names"] = _materialized_names(tmp_path)
            return _completed()

        with patch("subprocess.run", side_effect=_record):
            llm._call_cli([_human_with_image()])

        assert seen["cwd"] == str(tmp_path)
        assert len(seen["names"]) == 1

    def test_the_file_holds_the_image_bytes_during_the_call(self, tmp_path):
        """What is on disk is the image, not a placeholder."""
        llm = _routed_llm(tmp_path)
        captured = {}

        def _read_it(*_args, **_kwargs):
            name = _materialized_names(tmp_path)[0]
            captured["bytes"] = (tmp_path / name).read_bytes()
            return _completed()

        with patch("subprocess.run", side_effect=_read_it):
            llm._call_cli([_human_with_image()])

        assert captured["bytes"] == PNG_BYTES

    def test_no_system_temp_directory_is_used(self, tmp_path):
        """The workspace is the consented directory; a shared system temp
        directory is neither consented to nor necessarily readable by a
        harness that gates access by directory."""
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            with patch("tempfile.mkstemp") as mkstemp, patch(
                "tempfile.NamedTemporaryFile"
            ) as named, patch("tempfile.mkdtemp") as mkdtemp:
                llm._call_cli([_human_with_image()])
        mkstemp.assert_not_called()
        named.assert_not_called()
        mkdtemp.assert_not_called()

    def test_with_no_configured_cwd_the_file_lands_in_the_inherited_one(
        self, tmp_path, monkeypatch
    ):
        """``cwd=None`` means the subprocess inherits the caller's directory,
        so that is the directory the file has to be in."""
        monkeypatch.chdir(tmp_path)
        llm = CliProvider().load(
            command=["some-cli"],
            prompt_via="arg",
            image_route=CLAUDE_CODE_IMAGE_ROUTE,
            max_retries=0,
        )
        seen = {}
        with patch("subprocess.run", side_effect=_capture_workspace(seen, tmp_path)):
            llm._call_cli([_human_with_image()])
        assert len(seen["names"]) == 1


class TestTheFileIsRemovedOnBothPaths:
    """A materialized image that outlives its call is a leak into a workspace
    the caller reuses."""

    def test_removed_after_a_successful_call(self, tmp_path):
        """Removed after a successful call."""
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            llm._call_cli([_human_with_image()])
        assert _materialized_names(tmp_path) == []

    def test_removed_after_a_failing_call(self, tmp_path):
        """A non-zero exit still cleans up.

        Pinned with a call that actually fails rather than by inspecting the
        cleanup code, because the failure path is the one nobody exercises by
        hand.
        """
        llm = _routed_llm(tmp_path)
        with patch(
            "subprocess.run", return_value=_completed(returncode=1, stderr="nope")
        ):
            with pytest.raises(CliLLMError):
                llm._call_cli([_human_with_image()])
        assert _materialized_names(tmp_path) == []

    def test_removed_after_a_timeout(self, tmp_path):
        """A timeout raises from inside the with-block and still cleans up."""
        llm = _routed_llm(tmp_path)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="some-cli", timeout=1),
        ):
            with pytest.raises(CliLLMError):
                llm._call_cli([_human_with_image()])
        assert _materialized_names(tmp_path) == []

    def test_removed_when_output_parsing_fails(self, tmp_path):
        """A parse failure after the subprocess returns still cleans up."""
        llm = _routed_llm(tmp_path, output_format="json")
        with patch("subprocess.run", return_value=_completed(stdout="not json")):
            with pytest.raises(CliLLMError):
                llm._call_cli([_human_with_image()])
        assert _materialized_names(tmp_path) == []

    def test_the_file_survives_across_a_transient_retry(self, tmp_path):
        """Materialization wraps the whole retry loop.

        A file removed between attempts would leave the retry pointing at
        nothing, so the retry would fail for a reason that has nothing to do
        with why the first attempt failed.
        """
        llm = _routed_llm(tmp_path, max_retries=1, retry_backoff_seconds=0.0)
        names_per_attempt = []

        def _fail_then_succeed(*_args, **_kwargs):
            names_per_attempt.append(_materialized_names(tmp_path))
            if len(names_per_attempt) == 1:
                return _completed(returncode=1, stderr="429 rate limit")
            return _completed()

        with patch("subprocess.run", side_effect=_fail_then_succeed):
            llm._call_cli([_human_with_image()])

        assert len(names_per_attempt) == 2
        assert names_per_attempt[0] == names_per_attempt[1]
        assert len(names_per_attempt[0]) == 1
        assert _materialized_names(tmp_path) == []

    def test_a_partial_write_removes_what_it_already_wrote(self, tmp_path):
        """The second of three images failing must not strand the first."""
        good = ImagePayload(data=PNG_BYTES, media_type="image/png")
        bad = ImagePayload(data=PNG_BYTES, media_type="image/nonesuch")
        with pytest.raises(CliImageMaterializationError):
            with materialized_images([good, bad], str(tmp_path)):
                pass  # pragma: no cover - the with-block is never entered
        assert _materialized_names(tmp_path) == []

    def test_an_unwritable_directory_is_named(self, tmp_path):
        """The error says which directory, and why it matters."""
        payload = ImagePayload(data=PNG_BYTES, media_type="image/png")
        with patch.object(Path, "write_bytes", side_effect=OSError("read-only")):
            with pytest.raises(CliImageMaterializationError) as excinfo:
                with materialized_images([payload], str(tmp_path)):
                    pass  # pragma: no cover - never entered
        assert str(tmp_path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Nothing about the image's origin escapes
# ---------------------------------------------------------------------------


class TestTheFilenameAndPromptCarryNothingAboutTheImage:
    """The filename is visible to the harness and, through the prompt, to the
    model behind it."""

    def test_the_filename_is_generated_not_derived_from_the_source(self, tmp_path):
        """A source filename must not reach the workspace or the prompt."""
        source = tmp_path / "originating-source-name.png"
        source.write_bytes(PNG_BYTES)
        part = image_part_from_path(source)
        llm = _routed_llm(tmp_path)
        seen = {}

        def _capture(cmd, **_kwargs):
            seen["argv"] = list(cmd)
            seen["names"] = _materialized_names(tmp_path)
            return _completed()

        with patch("subprocess.run", side_effect=_capture):
            llm._call_cli([HumanMessage(content=[text_part("describe"), part])])

        written = seen["names"][0]
        assert "originating-source-name" not in written
        assert "originating-source-name" not in " ".join(seen["argv"])

    def test_the_filename_is_a_neutral_prefix_and_a_random_token(self, tmp_path):
        """Two materializations of the same bytes produce different names, so
        the name cannot be a function of the content either."""
        payload = ImagePayload(data=PNG_BYTES, media_type="image/png")
        names = []
        for _ in range(2):
            with materialized_images([payload], str(tmp_path)) as images:
                names.append(images[0].filename)
        assert names[0] != names[1]
        for name in names:
            assert name.startswith(IMAGE_FILENAME_PREFIX)
            assert name.endswith(".png")

    def test_the_extension_comes_from_the_declared_media_type(self, tmp_path):
        """The harness decides what it will open partly from the extension."""
        payload = ImagePayload(data=PNG_BYTES, media_type="image/jpeg")
        with materialized_images([payload], str(tmp_path)) as images:
            assert images[0].filename.endswith(".jpg")

    def test_the_prompt_carries_a_path_and_not_the_image_data(self, tmp_path):
        """A base64 payload in the prompt would be the drop this replaces,
        wearing a different shape."""
        llm = _routed_llm(tmp_path)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image()])
        argv = " ".join(seen["argv"])
        assert base64.b64encode(PNG_BYTES).decode("ascii")[:24] not in argv

    def test_the_reference_is_a_bare_filename_not_an_absolute_path(self, tmp_path):
        """The prompt goes to a third party; the host's directory layout does
        not need to go with it, and the file is in the subprocess's own cwd."""
        llm = _routed_llm(tmp_path)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image()])
        argv = " ".join(seen["argv"])
        assert IMAGE_FILENAME_PREFIX in argv
        assert str(tmp_path) not in argv


# ---------------------------------------------------------------------------
# The delivery kind is reported, not implied
# ---------------------------------------------------------------------------


class TestTheDeliveryKindIsReported:
    """A message-based provider is handed bytes; this one offers a path and
    cannot verify the harness opened it."""

    def test_generate_reports_offered_by_path(self, tmp_path):
        """_generate reports offered_by_path for an image-bearing turn."""
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            result = llm._generate([_human_with_image()])
        message = result.generations[0].message
        assert (
            message.response_metadata[IMAGE_DELIVERY_RESPONSE_KEY]
            == IMAGE_DELIVERY_OFFERED_BY_PATH
        )
        assert result.llm_output[IMAGE_DELIVERY_RESPONSE_KEY] == (
            IMAGE_DELIVERY_OFFERED_BY_PATH
        )

    def test_stream_reports_offered_by_path(self, tmp_path):
        """The streaming path reports the same kind as the batch one."""
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            chunks = list(llm._stream([_human_with_image()]))
        assert (
            chunks[0].message.response_metadata[IMAGE_DELIVERY_RESPONSE_KEY]
            == IMAGE_DELIVERY_OFFERED_BY_PATH
        )

    def test_astream_reports_offered_by_path(self, tmp_path):
        """The async path reports it too."""
        llm = _routed_llm(tmp_path)

        async def _run():
            """Collect the chunks from _astream."""
            return [chunk async for chunk in llm._astream([_human_with_image()])]

        with patch("subprocess.run", return_value=_completed()):
            chunks = asyncio.run(_run())
        assert (
            chunks[0].message.response_metadata[IMAGE_DELIVERY_RESPONSE_KEY]
            == IMAGE_DELIVERY_OFFERED_BY_PATH
        )

    def test_a_text_only_turn_reports_no_delivery_at_all(self, tmp_path):
        """Absence of the key is "no image", never "an image, somehow".

        A key reporting ``bytes`` or ``none`` on every text turn would make
        the signal unreadable, because a caller auditing image handling could
        no longer tell a turn that carried one from a turn that did not.
        """
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            result = llm._generate([HumanMessage(content="no image here")])
        message = result.generations[0].message
        assert IMAGE_DELIVERY_RESPONSE_KEY not in message.response_metadata
        assert result.llm_output is None


# ---------------------------------------------------------------------------
# A text-only turn is untouched
# ---------------------------------------------------------------------------


class TestATextOnlyTurnIsUnchanged:
    """Every CLI turn that carries no image must be byte-for-byte what it was
    before this path existed, route configured or not."""

    @pytest.mark.parametrize("message_format", ["last", "roles", "chatml"])
    def test_the_prompt_is_identical_with_and_without_a_route(
        self, tmp_path, message_format
    ):
        """The prompt is identical with and without a route."""
        messages = [
            SystemMessage(content="be brief"),
            HumanMessage(content="first"),
            AIMessage(content="answer"),
            HumanMessage(content="second"),
        ]
        prompts = {}
        for label, route in (("none", None), ("routed", CLAUDE_CODE_IMAGE_ROUTE)):
            llm = CliProvider().load(
                command=["some-cli"],
                prompt_via="arg",
                cwd=str(tmp_path),
                message_format=message_format,
                image_route=route,
            )
            seen = {}
            with patch("subprocess.run", side_effect=_capture_argv(seen)):
                llm._call_cli(messages)
            prompts[label] = seen["argv"]
        assert prompts["none"] == prompts["routed"]

    def test_no_file_is_written_for_a_text_only_turn(self, tmp_path):
        """No file is written for a text-only turn."""
        llm = _routed_llm(tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            llm._call_cli([HumanMessage(content="hello")])
        assert list(tmp_path.iterdir()) == []

    def test_a_text_only_parts_list_is_still_rendered_as_before(self, tmp_path):
        """List content carrying only text is not multimodal, and the route
        must not change how it renders."""
        parts = [text_part("hi")]
        prompts = {}
        for label, route in (("none", None), ("routed", CLAUDE_CODE_IMAGE_ROUTE)):
            llm = CliProvider().load(
                command=["some-cli"],
                prompt_via="arg",
                cwd=str(tmp_path),
                image_route=route,
            )
            seen = {}
            with patch("subprocess.run", side_effect=_capture_argv(seen)):
                llm._call_cli([HumanMessage(content=parts)])
            prompts[label] = seen["argv"]
        assert prompts["none"] == prompts["routed"] == ["some-cli", str(parts)]


# ---------------------------------------------------------------------------
# Scope: which messages a format carries
# ---------------------------------------------------------------------------


class TestOnlyTheRenderedMessagesAreMaterialized:
    """The refusal was already scoped to the messages a format renders;
    materialization is scoped the same way by sharing the same function."""

    def test_last_does_not_materialize_an_image_it_would_not_carry(self, tmp_path):
        """An image earlier in history is not written for ``last``."""
        llm = _routed_llm(tmp_path)
        messages = [_human_with_image(), AIMessage(content="ok"), HumanMessage("next")]
        seen = {}
        with patch("subprocess.run", side_effect=_capture_workspace(seen, tmp_path)):
            llm._call_cli(messages)
        assert seen["names"] == []

    @pytest.mark.parametrize("message_format", ["roles", "chatml"])
    def test_a_history_format_does_materialize_an_earlier_image(
        self, tmp_path, message_format
    ):
        """A format that renders the whole list carries the image too."""
        llm = _routed_llm(tmp_path, message_format=message_format)
        messages = [_human_with_image(), AIMessage(content="ok"), HumanMessage("next")]
        seen = {}
        with patch("subprocess.run", side_effect=_capture_workspace(seen, tmp_path)):
            llm._call_cli(messages)
        assert len(seen["names"]) == 1

    def test_messages_rendered_by_agrees_with_what_render_produces(self):
        """One answer to "which messages does this format carry"."""
        history = [
            SystemMessage(content="s"),
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
        ]
        assert messages_rendered_by(history, "last") == [history[-1]]
        assert messages_rendered_by(history, "roles") == history
        assert messages_rendered_by(history, "chatml") == history

    def test_last_falls_back_to_the_final_message_with_no_human_turn(self):
        """The fallback the renderer has always had is preserved."""
        history = [SystemMessage(content="s"), AIMessage(content="b")]
        assert messages_rendered_by(history, "last") == [history[-1]]

    def test_an_empty_list_is_rejected(self):
        """An empty list is rejected."""
        with pytest.raises(ValueError, match="empty"):
            messages_rendered_by([], "last")

    def test_an_unknown_format_is_rejected(self):
        """An unknown format is rejected."""
        with pytest.raises(ValueError, match="Unknown message_format"):
            messages_rendered_by([HumanMessage(content="x")], "nope")

    def test_the_text_of_an_image_bearing_message_survives(self, tmp_path):
        """Stripping the image must not strip the question with it."""
        llm = _routed_llm(tmp_path)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("what colour is the sky here?")])
        assert "what colour is the sky here?" in " ".join(seen["argv"])

    def test_a_role_label_survives_stripping_in_a_history_format(self, tmp_path):
        """The stripped message keeps its class, so it keeps its role label."""
        llm = _routed_llm(tmp_path, message_format="roles")
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("look")])
        assert "User: " in " ".join(seen["argv"])


# ---------------------------------------------------------------------------
# The per-harness routes
# ---------------------------------------------------------------------------


class TestTheShippedRoutesMatchWhatEachHarnessTakes:
    """Each route was verified against the tool itself; these pin the shape
    that verification established."""

    def test_every_shipped_preset_carries_a_route(self):
        """Each preset's harness is a known, vision-capable agent."""
        for provider_type, preset in CLI_PRESET_CATALOG.items():
            assert preset.image_route is not None, provider_type

    def test_a_loaded_preset_model_carries_its_route(self):
        """The route has to survive the preset -> provider -> model handoff.

        Asserting it on the preset object alone would pass with the field
        never threaded through ``CliPresetProvider``, which is where a
        silently route-less model would come from: the catalog would declare
        image, selection would allow it, and the call would then refuse.
        """
        for provider_type, preset in CLI_PRESET_CATALOG.items():
            llm = load_model(provider_type)
            assert isinstance(llm, CliLLM)
            assert llm.image_route is preset.image_route, provider_type

    def test_every_shipped_route_records_what_it_was_verified_against(self):
        """A route is a claim about third-party software that nothing here
        can detect changing, so it says which version it was checked on."""
        for preset in CLI_PRESET_CATALOG.values():
            assert preset.image_route.verified_against

    def test_claude_code_names_the_path_in_the_prompt(self, tmp_path):
        """No image flag exists on that CLI; it reads a path it is given."""
        llm = _routed_llm(tmp_path, route=CLAUDE_CODE_IMAGE_ROUTE)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("describe")])
        prompt = seen["argv"][-1]
        assert prompt.startswith("Read the image file ")
        assert prompt.endswith("describe")
        assert len(seen["argv"]) == 2  # no flag was added

    def test_codex_attaches_the_value_to_its_image_flag(self, tmp_path):
        """The flag is variadic (``-i, --image <FILE>...``), so the separated
        form would consume the prompt positional that follows it as a second
        image path and the CLI would report no prompt at all."""
        llm = _routed_llm(tmp_path, route=CODEX_IMAGE_ROUTE)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("describe")])
        argv = seen["argv"]
        flags = [token for token in argv if token.startswith("--image=")]
        assert len(flags) == 1
        assert "--image" not in argv  # never the separated form
        assert argv[-1] == "describe"  # the prompt is still last and intact

    def test_gemini_puts_an_at_reference_in_the_prompt(self, tmp_path):
        """Its headless path runs the same at-command file injection as the
        interactive one."""
        llm = _routed_llm(tmp_path, route=GEMINI_CLI_IMAGE_ROUTE)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("describe")])
        prompt = seen["argv"][-1]
        assert prompt.startswith(f"@{IMAGE_FILENAME_PREFIX}")
        assert prompt.endswith(" describe")

    def test_the_image_flag_precedes_the_prompt_positional(self, tmp_path):
        """A prompt delivered as a positional argument has to stay last."""
        llm = _routed_llm(tmp_path, route=CODEX_IMAGE_ROUTE, model="some-model")
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([_human_with_image("describe")])
        argv = seen["argv"]
        assert argv.index("--model") < argv.index(
            next(t for t in argv if t.startswith("--image="))
        )
        assert argv[-1] == "describe"

    def test_several_images_each_get_their_own_reference(self, tmp_path):
        """Two images are two files and two references, not one of each."""
        llm = _routed_llm(tmp_path, route=CODEX_IMAGE_ROUTE)
        message = HumanMessage(
            content=[text_part("compare these"), _png_part(), _png_part()]
        )
        seen = {}

        def _capture(cmd, **_kwargs):
            seen["argv"] = list(cmd)
            seen["names"] = _materialized_names(tmp_path)
            return _completed()

        with patch("subprocess.run", side_effect=_capture):
            llm._call_cli([message])

        flags = [t for t in seen["argv"] if t.startswith("--image=")]
        assert len(flags) == 2
        assert len(set(flags)) == 2
        assert len(seen["names"]) == 2

    def test_an_image_only_turn_through_an_argv_route_is_refused(self, tmp_path):
        """With no text and no prompt reference the invocation would carry no
        instruction, and the CLI would answer something that is not a model
        response at all."""
        llm = _routed_llm(tmp_path, route=CODEX_IMAGE_ROUTE)
        message = HumanMessage(content=[_png_part()])
        with patch("subprocess.run", return_value=_completed()) as run:
            with pytest.raises(CliLLMError, match="no text"):
                llm._call_cli([message])
        run.assert_not_called()
        assert _materialized_names(tmp_path) == []

    def test_an_image_only_turn_through_a_prompt_route_still_asks(self, tmp_path):
        """A prompt-reference route produces a non-empty instruction on its
        own, so an image with no question is a legitimate turn there."""
        llm = _routed_llm(tmp_path, route=CLAUDE_CODE_IMAGE_ROUTE)
        seen = {}
        with patch("subprocess.run", side_effect=_capture_argv(seen)):
            llm._call_cli([HumanMessage(content=[_png_part()])])
        assert seen["argv"][-1].startswith("Read the image file ")


# ---------------------------------------------------------------------------
# apply_route as a unit
# ---------------------------------------------------------------------------


class TestApplyRoute:
    """The pure half: given files, what does the invocation become."""

    IMAGES = []

    def test_no_images_changes_nothing(self):
        """A text-only turn through a routed preset is untouched."""
        prompt, argv = apply_route(CODEX_IMAGE_ROUTE, "hello", [])
        assert prompt == "hello"
        assert argv == []

    def test_a_route_with_both_mechanisms_applies_both(self, tmp_path):
        """Nothing in the shape forbids a tool that wants both."""
        route = CliImageRoute(
            name="both",
            argv_template=("--img", "{path}"),
            prompt_template="see {path}",
            prompt_separator=" | ",
        )
        payload = ImagePayload(data=PNG_BYTES, media_type="image/png")
        with materialized_images([payload], str(tmp_path)) as images:
            prompt, argv = apply_route(route, "question", images)
        assert argv == ["--img", images[0].filename]
        assert prompt == f"see {images[0].filename} | question"


# ---------------------------------------------------------------------------
# Reading bytes out of a content part
# ---------------------------------------------------------------------------


class TestImagePayloadExtraction:
    """The part shapes bili.iris.multimodal recognises, and the ones this
    transport cannot turn into a file."""

    def test_an_openai_style_data_uri_part(self):
        """The shape image_part() emits."""
        payload = image_payload(_png_part())
        assert payload.data == PNG_BYTES
        assert payload.media_type == "image/png"

    def test_a_bare_string_image_url(self):
        """Some integrations spell image_url as a plain string."""
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        part = {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
        assert image_payload(part).data == PNG_BYTES

    def test_a_langchain_standard_block_with_base64_string(self):
        """langchain-core's own standard image block."""
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        part = {
            "type": "image",
            "source_type": "base64",
            "data": encoded,
            "mime_type": "image/png",
        }
        payload = image_payload(part)
        assert payload.data == PNG_BYTES
        assert payload.media_type == "image/png"

    def test_a_standard_block_carrying_raw_bytes(self):
        """Raw bytes are taken as-is rather than base64-decoded."""
        part = {"type": "image", "data": PNG_BYTES, "mime_type": "image/png"}
        assert image_payload(part).data == PNG_BYTES

    def test_a_standard_block_with_no_media_type_is_refused(self):
        """bili-core does not guess a media type from raw bytes."""
        part = {"type": "image", "source_type": "base64", "data": "aGk="}
        with pytest.raises(CliImageMaterializationError, match="media type"):
            image_payload(part)

    def test_a_standard_block_with_undecodable_data_is_refused(self):
        """A malformed payload is named rather than written as garbage."""
        part = {
            "type": "image",
            "source_type": "base64",
            "data": "!!!not base64!!!",
            "mime_type": "image/png",
        }
        with pytest.raises(CliImageMaterializationError, match="decoded"):
            image_payload(part)

    def test_a_standard_block_with_a_non_string_payload_is_refused(self):
        """A number is neither bytes nor base64."""
        part = {
            "type": "image",
            "source_type": "base64",
            "data": 7,
            "mime_type": "image/png",
        }
        with pytest.raises(CliImageMaterializationError, match="bytes or a base64"):
            image_payload(part)

    def test_a_remote_url_is_refused_and_says_what_to_do(self):
        """This transport writes a file; it does not fetch one.

        Fetching remote content inside the provider is a network egress this
        transport never had, so the refusal names the local alternatives
        rather than quietly acquiring one.
        """
        part = image_part(url="https://example.invalid/chart.png")
        with pytest.raises(CliImageMaterializationError) as excinfo:
            image_payload(part)
        assert "image_part_from_path" in str(excinfo.value)

    def test_a_non_base64_data_uri_is_refused(self):
        """A percent-encoded data URI is not something to write blindly."""
        part = {"type": "image_url", "image_url": {"url": "data:image/png,raw"}}
        with pytest.raises(CliImageMaterializationError, match="base64"):
            image_payload(part)

    def test_an_undecodable_data_uri_is_refused(self):
        """An undecodable data URI is refused."""
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}}
        with pytest.raises(CliImageMaterializationError, match="decoded"):
            image_payload(part)

    def test_a_non_string_url_is_refused(self):
        """A non-string url is refused."""
        part = {"type": "image_url", "image_url": {"url": 5}}
        with pytest.raises(CliImageMaterializationError, match="must be a string"):
            image_payload(part)

    def test_a_non_image_part_is_refused(self):
        """A non-image part is refused."""
        with pytest.raises(CliImageMaterializationError, match="Not an image"):
            image_payload(text_part("hello"))

    def test_an_unknown_media_type_is_refused(self, tmp_path):
        """Writing under a guessed extension hands the harness a file it
        cannot identify, which fails later and less clearly."""
        payload = ImagePayload(data=PNG_BYTES, media_type="image/nonesuch")
        with pytest.raises(CliImageMaterializationError, match="no known file"):
            with materialized_images([payload], str(tmp_path)):
                pass  # pragma: no cover - never entered

    def test_a_media_type_with_parameters_still_resolves(self, tmp_path):
        """``image/png; charset=binary`` is still a PNG."""
        payload = ImagePayload(data=PNG_BYTES, media_type="image/png; charset=binary")
        with materialized_images([payload], str(tmp_path)) as images:
            assert images[0].filename.endswith(".png")

    def test_string_content_yields_no_payloads(self):
        """String content yields no payloads."""
        assert image_payloads("just text") == []

    def test_a_parts_list_with_no_image_yields_no_payloads(self):
        """A parts list with no image yields no payloads."""
        assert image_payloads([text_part("a"), text_part("b")]) == []

    def test_payloads_come_back_in_order(self):
        """Order matters: the prompt references them in the order given."""
        other = image_part(data=b"\x00", mime_type="image/gif")
        payloads = image_payloads([_png_part(), text_part("x"), other])
        assert [p.media_type for p in payloads] == ["image/png", "image/gif"]


# ---------------------------------------------------------------------------
# End to end against a real subprocess
# ---------------------------------------------------------------------------


class TestARealSubprocessCanOpenTheFile:
    """Everything above mocks subprocess.run.  These two spawn a real process
    that opens the file by the reference it was given, which is the only way
    to show the reference actually resolves from the subprocess's own cwd.
    """

    READER = (
        "import sys, pathlib;"
        "ref=[t for t in sys.argv[1:] if 'bili-image-' in t][0];"
        "name=ref.split('=')[-1].strip('@').split()[0];"
        "print(pathlib.Path(name).read_bytes().hex())"
    )

    def test_a_prompt_reference_resolves_from_the_subprocess_cwd(self, tmp_path):
        """The file named in the prompt is openable by the process."""
        llm = CliProvider().load(
            command=[sys.executable, "-c", self.READER],
            prompt_via="arg",
            cwd=str(tmp_path),
            image_route=GEMINI_CLI_IMAGE_ROUTE,
            max_retries=0,
        )
        content, delivery = llm._call_cli([_human_with_image("hi")])
        assert bytes.fromhex(content.strip()) == PNG_BYTES
        assert delivery == IMAGE_DELIVERY_OFFERED_BY_PATH
        assert _materialized_names(tmp_path) == []

    def test_an_argv_reference_resolves_from_the_subprocess_cwd(self, tmp_path):
        """The file named in an argv flag is openable by the process."""
        llm = CliProvider().load(
            command=[sys.executable, "-c", self.READER],
            prompt_via="arg",
            cwd=str(tmp_path),
            image_route=CODEX_IMAGE_ROUTE,
            model_flag_template=None,
            max_retries=0,
        )
        content, _ = llm._call_cli([_human_with_image("hi")])
        assert bytes.fromhex(content.strip()) == PNG_BYTES
        assert _materialized_names(tmp_path) == []
