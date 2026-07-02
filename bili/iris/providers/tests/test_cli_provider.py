"""Tests for the CLI subprocess provider.

Covers :class:`~bili.iris.providers.cli_provider.CliProvider`,
:class:`~bili.iris.providers.cli_provider.CliLLM`, and the supporting
helpers :func:`~bili.iris.providers.cli_provider.render_messages`,
:func:`~bili.iris.providers.cli_provider.strip_ansi`, and
:func:`~bili.iris.providers.cli_provider.extract_json_path`.

No real subprocess is spawned; :mod:`subprocess` is mocked throughout
so the tests run in any environment without a CLI LLM tool installed.
"""

# pylint: disable=too-few-public-methods,protected-access

import asyncio
import json
import os
import subprocess
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.aether.compiler.llm_resolver import resolve_provider
from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.providers.base import KNOWN_PROVIDER_TYPES
from bili.iris.providers.cli_provider import (
    CliLLM,
    CliLLMError,
    CliProvider,
    extract_json_path,
    render_messages,
    strip_ansi,
)
from bili.iris.providers.fallback import (
    _DEFAULT_RETRYABLE_NAMES,
    DEFAULT_POLICY,
    FallbackLLM,
)
from bili.iris.providers.registry import PROVIDER_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human(text: str) -> HumanMessage:
    """Return a HumanMessage with the given content."""
    return HumanMessage(content=text)


def _system(text: str) -> SystemMessage:
    """Return a SystemMessage with the given content."""
    return SystemMessage(content=text)


def _ai(text: str) -> AIMessage:
    """Return an AIMessage with the given content."""
    return AIMessage(content=text)


def _make_completed_proc(stdout: str = "", returncode: int = 0, stderr: str = ""):
    """Return a mock subprocess.CompletedProcess object."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def _llm(**kwargs) -> CliLLM:
    """Shorthand to build a CliLLM via CliProvider.load with a default command."""
    defaults: Dict = {"command": ["echo"]}
    defaults.update(kwargs)
    return CliProvider().load(**defaults)


# ---------------------------------------------------------------------------
# render_messages
# ---------------------------------------------------------------------------


class TestRenderMessages:
    """Unit tests for the message rendering helper."""

    def test_last_single_human(self):
        """last format returns the content of the only human message."""
        result = render_messages([_human("hello")], "last")
        assert result == "hello"

    def test_last_picks_last_human_in_multi_turn(self):
        """last format returns the last HumanMessage in a multi-turn conversation."""
        msgs = [_system("sys"), _human("first"), _ai("resp"), _human("second")]
        result = render_messages(msgs, "last")
        assert result == "second"

    def test_last_falls_back_to_final_message_when_no_human(self):
        """last format falls back to the last message when no HumanMessage is present."""
        msgs = [_system("sys"), _ai("response")]
        result = render_messages(msgs, "last")
        assert result == "response"

    def test_roles_format(self):
        """roles format prefixes each message with its role label."""
        msgs = [_system("be helpful"), _human("hi"), _ai("hello")]
        result = render_messages(msgs, "roles")
        assert result == "System: be helpful\nUser: hi\nAssistant: hello"

    def test_roles_single_message(self):
        """roles format works for a single message."""
        result = render_messages([_human("ping")], "roles")
        assert result == "User: ping"

    def test_chatml_format_structure(self):
        """chatml format includes the required im_start/im_end markers."""
        msgs = [_system("system"), _human("question")]
        result = render_messages(msgs, "chatml")
        assert "<|im_start|>system" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        lines = result.splitlines()
        assert lines[-1] == "<|im_start|>assistant"

    def test_chatml_full_conversation(self):
        """chatml format emits correct marker counts for a full conversation."""
        msgs = [_system("sys"), _human("q"), _ai("a"), _human("q2")]
        result = render_messages(msgs, "chatml")
        # 4 message blocks + 1 open assistant turn = 5 im_start markers
        assert result.count("<|im_start|>") == 5
        assert result.count("<|im_end|>") == 4

    def test_empty_messages_raises(self):
        """render_messages raises ValueError for an empty message list."""
        with pytest.raises(ValueError, match="empty message list"):
            render_messages([], "last")

    def test_unknown_format_raises(self):
        """render_messages raises ValueError for an unsupported format string."""
        with pytest.raises(ValueError, match="Unknown message_format"):
            render_messages([_human("x")], "unknown_format")


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    """Tests for ANSI escape code stripping."""

    def test_strips_color_codes(self):
        """Common SGR color codes are removed."""
        text = "\x1b[32mhello\x1b[0m world"
        assert strip_ansi(text) == "hello world"

    def test_strips_bold(self):
        """Bold ANSI codes are removed."""
        text = "\x1b[1mbold\x1b[0m"
        assert strip_ansi(text) == "bold"

    def test_no_ansi_passthrough(self):
        """Plain text without ANSI codes is returned unchanged."""
        text = "plain text"
        assert strip_ansi(text) == "plain text"

    def test_empty_string(self):
        """Empty string input returns empty string."""
        assert strip_ansi("") == ""

    def test_strips_cursor_movement(self):
        """Cursor movement codes used by progress bars are removed."""
        text = "\x1b[2Kprogress done\x1b[0m"
        assert strip_ansi(text) == "progress done"


# ---------------------------------------------------------------------------
# extract_json_path
# ---------------------------------------------------------------------------


class TestExtractJsonPath:
    """Tests for the dot-separated JSON path extractor."""

    def test_top_level_key(self):
        """A single-segment path extracts a top-level key."""
        assert extract_json_path({"content": "hello"}, "content") == "hello"

    def test_nested_key(self):
        """A multi-segment path traverses nested dicts and lists."""
        obj = {"choices": [{"text": "hi"}]}
        assert extract_json_path(obj, "choices.0.text") == "hi"

    def test_deeply_nested(self):
        """A deeply nested path is traversed correctly."""
        obj = {"a": {"b": {"c": "value"}}}
        assert extract_json_path(obj, "a.b.c") == "value"

    def test_list_index(self):
        """A numeric path segment indexes into a list."""
        obj = [{"x": "first"}, {"x": "second"}]
        assert extract_json_path(obj, "1.x") == "second"

    def test_missing_key_raises(self):
        """A missing dict key raises KeyError."""
        with pytest.raises(KeyError):
            extract_json_path({"a": "b"}, "missing")

    def test_out_of_range_raises(self):
        """An out-of-range list index raises IndexError."""
        with pytest.raises(IndexError):
            extract_json_path([1, 2], "5")

    def test_non_container_traversal_raises(self):
        """Attempting to traverse a scalar node raises TypeError."""
        with pytest.raises(TypeError):
            extract_json_path({"a": "scalar"}, "a.nested")

    def test_converts_to_str(self):
        """The returned value is always a string regardless of the JSON type."""
        result = extract_json_path({"n": 42}, "n")
        assert result == "42"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# CliProvider.load validation
# ---------------------------------------------------------------------------


class TestCliProviderLoad:
    """Tests for CliProvider.load() parameter validation."""

    def test_empty_command_raises(self):
        """An empty command list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CliProvider().load(command=[])

    def test_invalid_prompt_via_raises(self):
        """An unsupported prompt_via value raises ValueError."""
        with pytest.raises(ValueError, match="prompt_via"):
            CliProvider().load(command=["echo"], prompt_via="unknown")

    def test_invalid_message_format_raises(self):
        """An unsupported message_format value raises ValueError."""
        with pytest.raises(ValueError, match="message_format"):
            CliProvider().load(command=["echo"], message_format="bad")

    def test_invalid_output_format_raises(self):
        """An unsupported output_format value raises ValueError."""
        with pytest.raises(ValueError, match="output_format"):
            CliProvider().load(command=["echo"], output_format="xml")

    def test_returns_cli_llm_instance(self):
        """load() returns a CliLLM instance."""
        llm = CliProvider().load(command=["echo", "hello"])
        assert isinstance(llm, CliLLM)

    def test_defaults_propagated(self):
        """Default config values are set correctly on the returned CliLLM."""
        llm = CliProvider().load(command=["my-cli"])
        assert llm.prompt_via == "stdin"
        assert llm.message_format == "last"
        assert llm.output_format == "text"
        assert llm.strip_ansi_output is True
        assert llm.timeout_seconds == 1800.0
        assert llm.cwd is None

    def test_none_timeout_accepted(self):
        """CliProvider.load() accepts timeout_seconds=None to disable the timeout."""
        llm = CliProvider().load(command=["my-cli"], timeout_seconds=None)
        assert llm.timeout_seconds is None

    def test_custom_values_propagated(self):
        """Custom config values are forwarded to the CliLLM instance."""
        llm = CliProvider().load(
            command=["my-cli", "--json"],
            prompt_via="arg",
            message_format="roles",
            output_format="json",
            json_path="result.text",
            strip_ansi=False,
            timeout_seconds=30.0,
            cwd="/opt/sandbox",
        )
        assert llm.command == ["my-cli", "--json"]
        assert llm.prompt_via == "arg"
        assert llm.message_format == "roles"
        assert llm.output_format == "json"
        assert llm.json_path == "result.text"
        assert llm.strip_ansi_output is False
        assert llm.timeout_seconds == 30.0
        assert llm.cwd == "/opt/sandbox"

    def test_custom_cwd_propagated(self):
        """CliProvider.load() forwards a caller-supplied cwd to the CliLLM."""
        llm = CliProvider().load(command=["my-cli"], cwd="/workspace/fixed")
        assert llm.cwd == "/workspace/fixed"

    def test_extra_kwargs_ignored(self):
        """Extra kwargs from an llm_config catalog entry do not raise."""
        llm = CliProvider().load(
            command=["my-cli"],
            model_name="cli:custom",
            temperature=0.7,
        )
        assert isinstance(llm, CliLLM)


# ---------------------------------------------------------------------------
# CliLLM._build_run_args
# ---------------------------------------------------------------------------


class TestBuildRunArgs:
    """Tests for command-line construction logic."""

    def test_stdin_returns_stdin_text(self):
        """stdin mode returns the prompt as stdin_text and no temp file."""
        llm = _llm(prompt_via="stdin")
        cmd, stdin_text, tmp = llm._build_run_args("hello")
        assert cmd == ["echo"]
        assert stdin_text == "hello"
        assert tmp is None

    def test_arg_appends_prompt(self):
        """arg mode appends the prompt to the command and leaves stdin empty."""
        llm = _llm(prompt_via="arg")
        cmd, stdin_text, tmp = llm._build_run_args("hello")
        assert cmd == ["echo", "hello"]
        assert stdin_text is None
        assert tmp is None

    def test_file_creates_temp_and_appends_path(self):
        """file mode writes the prompt to a temp file and appends its path."""
        llm = _llm(prompt_via="file")
        cmd, stdin_text, tmp = llm._build_run_args("write me to a file")
        assert stdin_text is None
        assert tmp is not None
        assert os.path.exists(tmp)
        with open(tmp, encoding="utf-8") as f:
            assert f.read() == "write me to a file"
        os.unlink(tmp)
        assert cmd[-1] == tmp

    def test_command_is_not_mutated(self):
        """_build_run_args does not mutate the CliLLM.command list."""
        llm = _llm(command=["my-llm", "--flag"], prompt_via="arg")
        original = list(llm.command)
        llm._build_run_args("prompt")
        assert llm.command == original


# ---------------------------------------------------------------------------
# CliLLM subprocess execution
# ---------------------------------------------------------------------------


class TestRunSubprocess:
    """Tests for the subprocess execution layer."""

    def test_successful_run_returns_stdout(self):
        """A zero-exit subprocess returns its stdout unchanged (pre-strip)."""
        llm = _llm()
        with patch(
            "subprocess.run", return_value=_make_completed_proc("hello world\n")
        ):
            result = llm._run_subprocess("prompt")
        assert result == "hello world\n"

    def test_ansi_stripped_when_flag_set(self):
        """ANSI codes are removed from stdout when strip_ansi_output is True."""
        llm = _llm(strip_ansi=True)
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc("\x1b[32mgreen\x1b[0m text\n"),
        ):
            result = llm._run_subprocess("prompt")
        assert result == "green text\n"

    def test_ansi_preserved_when_flag_false(self):
        """ANSI codes are preserved when strip_ansi_output is False."""
        llm = _llm(strip_ansi=False)
        raw = "\x1b[32mgreen\x1b[0m text"
        with patch("subprocess.run", return_value=_make_completed_proc(raw)):
            result = llm._run_subprocess("prompt")
        assert "\x1b[32m" in result

    def test_non_zero_exit_raises_cli_llm_error(self):
        """A non-zero exit code raises CliLLMError."""
        llm = _llm()
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc("", returncode=1, stderr="oops"),
        ):
            with pytest.raises(CliLLMError, match="exited with code 1"):
                llm._run_subprocess("prompt")

    def test_timeout_raises_cli_llm_error(self):
        """A subprocess timeout raises CliLLMError."""
        llm = _llm(timeout_seconds=5.0)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["echo"], timeout=5),
        ):
            with pytest.raises(CliLLMError, match="timed out"):
                llm._run_subprocess("prompt")

    def test_none_timeout_passes_none_to_subprocess(self):
        """When timeout_seconds=None, subprocess.run receives timeout=None (no limit)."""
        llm = _llm(timeout_seconds=None)
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm._run_subprocess("prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("timeout") is None

    def test_default_cwd_inherits_calling_process(self):
        """By default (cwd unset), subprocess.run receives cwd=None, which
        makes it inherit the calling process's current working directory --
        today's behaviour, preserved for backward compatibility."""
        llm = _llm()
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm._run_subprocess("prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("cwd") is None

    def test_explicit_cwd_is_honored(self):
        """A caller-supplied cwd is forwarded verbatim to subprocess.run."""
        llm = _llm(cwd="/opt/fixed-workspace")
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm._run_subprocess("prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("cwd") == "/opt/fixed-workspace"

    def test_env_is_passed(self):
        """The subprocess is called with a copy of os.environ."""
        llm = _llm()
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm._run_subprocess("prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert "env" in call_kwargs
        assert isinstance(call_kwargs["env"], dict)

    def test_stdin_mode_passes_prompt_as_input(self):
        """stdin mode passes the prompt as the subprocess input kwarg."""
        llm = _llm(prompt_via="stdin")
        with patch(
            "subprocess.run", return_value=_make_completed_proc("out")
        ) as mock_run:
            llm._run_subprocess("my prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("input") == "my prompt"

    def test_arg_mode_does_not_pass_stdin(self):
        """arg mode does not pass stdin to the subprocess."""
        llm = _llm(prompt_via="arg")
        with patch(
            "subprocess.run", return_value=_make_completed_proc("out")
        ) as mock_run:
            llm._run_subprocess("my prompt")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("input") is None

    def test_stderr_truncated_in_error_message(self):
        """The error message includes at most 500 chars of stderr."""
        long_stderr = "X" * 600
        llm = _llm()
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc("", returncode=2, stderr=long_stderr),
        ):
            with pytest.raises(CliLLMError) as exc_info:
                llm._run_subprocess("p")
        assert len(str(exc_info.value)) < 600 + 100


# ---------------------------------------------------------------------------
# CliLLM._parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    """Tests for output parsing (text vs JSON)."""

    def test_text_strips_whitespace(self):
        """text format strips leading/trailing whitespace."""
        llm = _llm(output_format="text")
        assert llm._parse_output("  hello world  \n") == "hello world"

    def test_json_extracts_default_content_path(self):
        """json format extracts the value at the default 'content' path."""
        llm = _llm(output_format="json")
        raw = json.dumps({"content": "extracted"})
        assert llm._parse_output(raw) == "extracted"

    def test_json_custom_path(self):
        """json format extracts the value at a custom dot-path."""
        llm = _llm(output_format="json", json_path="choices.0.text")
        raw = json.dumps({"choices": [{"text": "the answer"}]})
        assert llm._parse_output(raw) == "the answer"

    def test_invalid_json_raises(self):
        """Non-JSON stdout raises CliLLMError when output_format='json'."""
        llm = _llm(output_format="json")
        with pytest.raises(CliLLMError, match="not valid JSON"):
            llm._parse_output("not json")

    def test_missing_json_path_raises(self):
        """A missing path in JSON output raises CliLLMError."""
        llm = _llm(output_format="json", json_path="nonexistent")
        raw = json.dumps({"other": "field"})
        with pytest.raises(CliLLMError, match="json_path"):
            llm._parse_output(raw)


# ---------------------------------------------------------------------------
# CliLLM._generate (LangChain interface)
# ---------------------------------------------------------------------------


class TestGenerate:
    """Tests for the LangChain _generate() entrypoint."""

    def test_returns_chat_result(self):
        """_generate returns a ChatResult instance."""
        llm = _llm()
        # type: ignore[method-assign]
        llm._run_subprocess = MagicMock(return_value="Hello from CLI")
        result = llm._generate(messages=[_human("hi")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1

    def test_generation_contains_ai_message(self):
        """The single generation contains an AIMessage with the CLI output."""
        llm = _llm()
        llm._run_subprocess = MagicMock(return_value="CLI response")  # type: ignore[method-assign]
        result = llm._generate(messages=[_human("ping")])
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == "CLI response"

    def test_uses_parsed_output(self):
        """_generate delegates parsing to _call_cli, not raw stdout."""
        llm = _llm()
        with patch.object(llm, "_call_cli", return_value="parsed") as mock_call:
            result = llm._generate(messages=[_human("q")])
        mock_call.assert_called_once()
        assert result.generations[0].message.content == "parsed"


# ---------------------------------------------------------------------------
# CliLLM._stream (single-chunk streaming)
# ---------------------------------------------------------------------------


class TestStream:
    """Tests for the synchronous _stream() method."""

    def test_yields_single_chunk(self):
        """_stream yields exactly one ChatGenerationChunk."""
        llm = _llm()
        with patch.object(llm, "_call_cli", return_value="response text"):
            chunks = list(llm._stream(messages=[_human("hi")]))
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatGenerationChunk)

    def test_chunk_contains_full_content(self):
        """The single chunk contains the complete CLI response."""
        llm = _llm()
        with patch.object(llm, "_call_cli", return_value="the full text"):
            chunks = list(llm._stream(messages=[_human("hi")]))
        msg = chunks[0].message
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == "the full text"

    def test_stream_propagates_cli_error(self):
        """_stream propagates CliLLMError from the subprocess."""
        llm = _llm()
        with patch.object(llm, "_call_cli", side_effect=CliLLMError("boom")):
            with pytest.raises(CliLLMError, match="boom"):
                list(llm._stream(messages=[_human("hi")]))


# ---------------------------------------------------------------------------
# CliLLM._astream (async streaming)
# ---------------------------------------------------------------------------


class TestAStream:
    """Tests for the async _astream() method."""

    def test_async_yields_single_chunk(self):
        """_astream yields exactly one chunk containing the full CLI response."""

        async def _run():
            """Async inner coroutine to collect chunks from _astream."""
            llm = _llm()
            with patch.object(llm, "_call_cli", return_value="async result"):
                chunks = []
                async for chunk in llm._astream(messages=[_human("hi")]):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 1
        assert chunks[0].message.content == "async result"

    def test_astream_does_not_block_event_loop(self):
        """_astream runs the blocking subprocess in a thread executor.

        Verify that _call_cli is invoked via asyncio.to_thread by patching
        asyncio.to_thread and asserting it is awaited with the correct
        callable.  This ensures the event loop is not blocked for the
        duration of the subprocess call.
        """

        async def _run():
            """Async inner coroutine that patches asyncio.to_thread."""
            llm = _llm()
            to_thread_calls = []

            async def fake_to_thread(func, *args, **kwargs):
                """Capture the call and return a fixed value."""
                to_thread_calls.append((func, args, kwargs))
                return "threaded result"

            with patch("asyncio.to_thread", side_effect=fake_to_thread):
                chunks = []
                async for chunk in llm._astream(messages=[_human("hi")]):
                    chunks.append(chunk)
            return to_thread_calls, chunks

        calls, chunks = asyncio.run(_run())
        # asyncio.to_thread must have been called exactly once
        assert len(calls) == 1
        # The first positional arg to to_thread is the callable (_call_cli)
        func, _args, _ = calls[0]
        assert callable(func)
        # The result from to_thread is used as the chunk content
        assert len(chunks) == 1
        assert chunks[0].message.content == "threaded result"


# ---------------------------------------------------------------------------
# Integration: full invoke() path
# ---------------------------------------------------------------------------


class TestInvokeEndToEnd:
    """Integration tests: full .invoke() path with subprocess.run mocked."""

    def test_invoke_returns_ai_message(self):
        """invoke() returns an AIMessage containing the CLI output."""
        llm = _llm(command=["echo", "hello"])
        with patch("subprocess.run", return_value=_make_completed_proc("AI output")):
            response = llm.invoke([_human("what is 2+2")])
        assert isinstance(response, AIMessage)
        assert "AI output" in response.content

    def test_invoke_json_output_format(self):
        """invoke() parses JSON output and extracts the configured path."""
        llm = _llm(
            command=["my-cli", "--json"],
            output_format="json",
            json_path="message",
        )
        stdout = json.dumps({"message": "the answer is 4"})
        with patch("subprocess.run", return_value=_make_completed_proc(stdout)):
            response = llm.invoke([_human("what is 2+2")])
        assert response.content == "the answer is 4"

    def test_invoke_roles_message_format_sends_full_history(self):
        """roles format includes System/User/Assistant prefixes in the subprocess input."""
        llm = _llm(message_format="roles")
        messages = [_system("be helpful"), _human("hi"), _ai("hello"), _human("bye")]
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm.invoke(messages)
        call_kwargs = mock_run.call_args.kwargs
        stdin_text = call_kwargs.get("input", "")
        assert "System:" in stdin_text
        assert "User:" in stdin_text
        assert "Assistant:" in stdin_text

    def test_invoke_chatml_message_format(self):
        """chatml format includes im_start markers in the subprocess input."""
        llm = _llm(message_format="chatml")
        messages = [_system("sys"), _human("q")]
        with patch(
            "subprocess.run", return_value=_make_completed_proc("ok")
        ) as mock_run:
            llm.invoke(messages)
        call_kwargs = mock_run.call_args.kwargs
        stdin_text = call_kwargs.get("input", "")
        assert "<|im_start|>system" in stdin_text
        assert "<|im_start|>user" in stdin_text

    def test_invoke_arg_mode_no_stdin(self):
        """arg mode appends the prompt to the command and passes no stdin."""
        llm = _llm(prompt_via="arg")
        with patch(
            "subprocess.run", return_value=_make_completed_proc("answer")
        ) as mock_run:
            llm.invoke([_human("question")])
        positional_cmd = mock_run.call_args.args[0]
        assert positional_cmd[-1] == "question"
        assert mock_run.call_args.kwargs.get("input") is None


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Verify the 'cli' type is registered in the built-in provider registry."""

    def test_cli_registered_in_builtin_registry(self):
        """The 'cli' provider type is present in PROVIDER_REGISTRY."""
        assert "cli" in PROVIDER_REGISTRY

    def test_cli_provider_class_correct(self):
        """The 'cli' type maps to CliProvider in the registry."""
        assert PROVIDER_REGISTRY.get("cli") is CliProvider

    def test_cli_in_known_provider_types(self):
        """'cli' appears in KNOWN_PROVIDER_TYPES."""
        assert "cli" in KNOWN_PROVIDER_TYPES


# ---------------------------------------------------------------------------
# Heuristic resolver: "cli:" prefix
# ---------------------------------------------------------------------------


class TestHeuristicResolution:
    """Verify the heuristic resolver maps 'cli:' prefixed IDs to the 'cli' provider."""

    def test_cli_prefix_resolves_to_cli(self):
        """Model IDs prefixed with 'cli:' resolve to the cli provider type."""
        assert resolve_provider("cli:custom") == "cli"
        assert resolve_provider("cli:my-local-tool") == "cli"


# ---------------------------------------------------------------------------
# LLM_MODELS catalog entry
# ---------------------------------------------------------------------------


class TestLLMModelsCatalog:
    """Verify the 'cli' section is present and structurally valid."""

    def test_cli_section_exists(self):
        """LLM_MODELS contains a 'cli' top-level key."""
        assert "cli" in LLM_MODELS

    def test_cli_section_has_models(self):
        """The 'cli' section has at least one model entry."""
        models = LLM_MODELS["cli"]["models"]
        assert len(models) > 0

    def test_cli_model_has_required_fields(self):
        """Each cli model entry has model_name and model_id fields."""
        for entry in LLM_MODELS["cli"]["models"]:
            assert "model_name" in entry
            assert "model_id" in entry

    def test_cli_model_id_uses_sentinel(self):
        """CLI catalog model IDs use the 'cli:' sentinel prefix."""
        ids = [e["model_id"] for e in LLM_MODELS["cli"]["models"]]
        assert any(mid.startswith("cli:") for mid in ids)

    def test_cli_supports_tools_is_false(self):
        """CLI models set supports_tools=False (bind_tools() is not available)."""
        for entry in LLM_MODELS["cli"]["models"]:
            assert entry.get("supports_tools") is False


# ---------------------------------------------------------------------------
# CliLLM._llm_type property
# ---------------------------------------------------------------------------


class TestLLMType:
    """Verify the _llm_type property returns the correct identifier."""

    def test_llm_type_returns_cli(self):
        """The _llm_type property returns 'cli'."""
        llm = _llm()
        assert llm._llm_type == "cli"


# ---------------------------------------------------------------------------
# FallbackLLM integration: CliLLMError retryability
# ---------------------------------------------------------------------------


class TestCliLLMErrorRetryability:
    """Verify CliLLMError is treated as retryable by the fallback engine.

    The fallback engine's DEFAULT_POLICY must classify CliLLMError as
    retryable so that a CLI provider can be placed at the front of a fallback
    chain (e.g. try the CLI first, fall through to an API provider if it
    fails) without requiring consumers to write a custom FallbackPolicy.
    """

    def test_default_policy_treats_cli_llm_error_as_retryable(self):
        """DEFAULT_POLICY.should_fallback(CliLLMError(...)) returns True."""
        err = CliLLMError("subprocess exited with code 1")
        assert DEFAULT_POLICY.should_fallback(err) is True

    def test_cli_llm_error_name_in_retryable_names(self):
        """'CliLLMError' appears in the fallback engine's retryable name set."""
        assert "CliLLMError" in _DEFAULT_RETRYABLE_NAMES

    def test_fallback_llm_falls_over_on_cli_error(self):
        """A FallbackLLM with a failing CliLLM primary falls through to the next
        provider.

        Construct a FallbackLLM where:
        - primary is a CliLLM that always raises CliLLMError
        - fallback is a mock LLM that succeeds

        Assert that the FallbackLLM's invoke() returns the fallback's response,
        not the primary's error.
        """
        # Primary: a CliLLM configured to always fail
        primary = _llm()
        primary._call_cli = MagicMock(  # type: ignore[method-assign]
            side_effect=CliLLMError("CLI tool crashed")
        )

        # Fallback: a simple mock that succeeds
        fallback_response = AIMessage(content="API fallback succeeded")
        fallback_llm = MagicMock()
        fallback_llm.invoke = MagicMock(return_value=fallback_response)

        chain = FallbackLLM(primary=primary, fallbacks=[fallback_llm])
        result = chain.invoke([_human("hello")])

        # The fallback response must be returned
        assert result is fallback_response
        # The primary must have been tried
        primary._call_cli.assert_called_once()
        # The fallback must have been invoked
        fallback_llm.invoke.assert_called_once()

    def test_fallback_not_triggered_for_non_retryable_errors(self):
        """A FallbackLLM does NOT fall over on a ValueError from the primary.

        ValueError is not in the retryable set, so the chain must re-raise it
        immediately rather than trying the fallback.
        """
        primary = _llm()
        primary._call_cli = MagicMock(  # type: ignore[method-assign]
            side_effect=ValueError("bad configuration")
        )

        fallback_llm = MagicMock()
        fallback_llm.invoke = MagicMock(
            return_value=AIMessage(content="should not reach")
        )

        chain = FallbackLLM(primary=primary, fallbacks=[fallback_llm])

        with pytest.raises(ValueError, match="bad configuration"):
            chain.invoke([_human("hello")])

        # The fallback must NOT have been called
        fallback_llm.invoke.assert_not_called()
