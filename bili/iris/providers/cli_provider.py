"""CLI (subprocess) provider for bili-core IRIS.

Drives any command-line LLM tool as a stateless text-in / text-out model
behind the :class:`LLMProvider` interface.  The provider spawns the
configured executable as a subprocess, sends the rendered prompt via stdin
or a command-line argument, captures stdout, and returns the result.

This enables "BYO-CLI" usage: users who have a CLI LLM tool but not an API
key can plug it into any bili-core agent or AETHER multi-agent workflow
without writing integration code.  The provider is intentionally generic --
no specific CLI tool is hard-coded or special-cased.

Design
------
A non-zero exit code or subprocess timeout raises :class:`CliLLMError`, which
propagates up to the caller.  When used with the fallback engine
(:class:`~bili.iris.providers.fallback.FallbackLLM`), ``CliLLMError`` is
treated as retryable so the chain can fall through to an API provider.

The LLM object returned by :meth:`CliProvider.load` is a
:class:`CliLLM` -- a concrete :class:`~langchain_core.language_models.BaseChatModel`
subclass.  This makes it a drop-in replacement inside any LangChain/LangGraph
pipeline, AETHER agent, or IRIS node without modification.

Message rendering
-----------------
CLI tools accept a single text prompt, not a structured message list.  The
provider supports three ``message_format`` strategies:

``"last"`` (default)
    Send only the content of the last ``HumanMessage``.  Use this for
    interactive CLIs that handle their own conversation state, or for
    one-shot question-answer usage.  This is the honest stateless
    behaviour -- the provider does not pretend to carry history.

``"roles"``
    Render the full message list as plain text with role prefixes::

        System: <system content>
        User: <first user turn>
        Assistant: <first assistant turn>
        User: <second user turn>

    Suitable for CLI tools that accept a full conversation context as a
    single string.

``"chatml"``
    Render the full message list in ChatML format::

        <|im_start|>system
        <system content>
        <|im_end|>
        <|im_start|>user
        <first user turn>
        <|im_end|>
        ...

    Some local model runners (e.g. llama.cpp with a chat template) parse
    this format natively.

Auth
----
The subprocess inherits the calling process's environment (``os.environ``).
Whatever credential the CLI tool requires (OAuth session, API key file, etc.)
must already be present in the environment -- bili-core never touches it.

No new optional dependency is required; the module uses stdlib ``subprocess``
and ``re`` only.
"""

import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Iterator, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported values for the ``message_format`` config key.
SUPPORTED_MESSAGE_FORMATS = frozenset({"last", "roles", "chatml"})

#: Supported values for the ``prompt_via`` config key.
SUPPORTED_PROMPT_VIA = frozenset({"stdin", "arg", "file"})

#: Supported values for the ``output_format`` config key.
SUPPORTED_OUTPUT_FORMATS = frozenset({"text", "json"})

#: ANSI escape-sequence pattern for stripping colour codes from output.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CliLLMError(RuntimeError):
    """Raised when the CLI subprocess exits non-zero or times out.

    The fallback engine's default retryable-name set includes
    ``CliLLMError`` so failed CLI calls fall over to the next provider in
    the chain automatically.
    """


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def _role_label(msg: BaseMessage) -> str:
    """Return a human-readable role label for *msg*."""
    if isinstance(msg, SystemMessage):
        return "System"
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    # Generic fallback for any other message type
    msg_type = type(msg).__name__
    return msg_type.replace("Message", "").replace("message", "") or "Unknown"


def render_messages(
    messages: List[BaseMessage],
    message_format: str = "last",
) -> str:
    """Collapse a list of LangChain messages into a single prompt string.

    :param messages: The message list from a LangChain invocation.
    :param message_format: One of ``"last"``, ``"roles"``, or ``"chatml"``.
    :returns: A single string ready to send to the CLI tool.
    :raises ValueError: If ``message_format`` is not recognised, or if the
        message list is empty.
    """
    if not messages:
        raise ValueError("Cannot render an empty message list")
    if message_format not in SUPPORTED_MESSAGE_FORMATS:
        raise ValueError(
            f"Unknown message_format {message_format!r}. "
            f"Supported: {sorted(SUPPORTED_MESSAGE_FORMATS)}"
        )

    if message_format == "last":
        # Find the last human message (most common case)
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        # Fall back to the very last message if no HumanMessage found
        return str(messages[-1].content)

    if message_format == "roles":
        parts = []
        for msg in messages:
            label = _role_label(msg)
            parts.append(f"{label}: {msg.content}")
        return "\n".join(parts)

    # chatml
    parts = []
    for msg in messages:
        role = _role_label(msg).lower()
        # Normalise "User" -> "user", "System" -> "system", etc.
        parts.append(f"<|im_start|>{role}\n{msg.content}\n<|im_end|>")
    # Append the open assistant turn so the model knows to continue
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ANSI stripping
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*.

    CLI tools that emit colour output (progress bars, formatting) leave
    garbage in the captured stdout.  This function strips all standard ANSI
    SGR / cursor-movement codes.
    """
    return _ANSI_ESCAPE.sub("", text)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json_path(obj: Any, path: str) -> str:
    """Traverse a nested dict/list using a dot-separated *path*.

    Examples::

        extract_json_path({"content": "hello"}, "content")          # "hello"
        extract_json_path({"choices": [{"text": "hi"}]}, "choices.0.text")  # "hi"

    :param obj: The parsed JSON object (dict, list, or scalar).
    :param path: Dot-separated key/index path.
    :returns: The value at *path* as a string.
    :raises KeyError: If any segment of the path is missing.
    :raises IndexError: If a numeric segment is out of range.
    :raises TypeError: If a non-container node is traversed as a container.
    """
    current: Any = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current[segment]
        elif isinstance(current, list):
            current = current[int(segment)]
        else:
            raise TypeError(
                f"Cannot index into {type(current).__name__!r} "
                f"at path segment {segment!r}"
            )
    return str(current)


# ---------------------------------------------------------------------------
# CliLLM — the LangChain-compatible model object
# ---------------------------------------------------------------------------


class CliLLM(BaseChatModel):
    """A :class:`~langchain_core.language_models.BaseChatModel` backed by a
    CLI subprocess.

    Instances are created by :class:`CliProvider` and returned to the caller.
    Do not instantiate directly -- use ``CliProvider().load(...)`` instead.

    The model is **stateless**: each call spawns a fresh subprocess for a
    fresh prompt.  No conversation history is persisted across calls.

    **Streaming note:** ``_stream`` and ``_astream`` both yield the complete
    response as a SINGLE chunk.  This is honest -- the subprocess runs to
    completion before any output is available, so true per-token streaming
    is not possible here.  The single-chunk contract is fully compatible with
    LangChain's streaming consumers (they iterate over whatever chunks are
    produced).

    Config fields (all set by ``CliProvider.load``)
    -----------------------------------------------
    command : list[str]
        The executable and its arguments (e.g. ``["claude", "--output-format",
        "json"]``).  Do not include the prompt here; it is added by the
        provider according to ``prompt_via``.
    prompt_via : str
        How the prompt is sent to the process.  One of:

        ``"stdin"`` (default)
            The prompt is written to the process's stdin pipe.

        ``"arg"``
            The prompt is appended as an extra positional argument to
            ``command``.

        ``"file"``
            The prompt is written to a temporary file and its path is
            appended as an extra argument.  Useful for CLIs that do not
            read stdin.

    message_format : str
        Message rendering strategy -- ``"last"`` (default), ``"roles"``,
        or ``"chatml"``.  See :func:`render_messages`.
    output_format : str
        How to parse the subprocess stdout.  ``"text"`` (default) returns
        the raw stripped output; ``"json"`` parses JSON and extracts the
        value at ``json_path``.
    json_path : str
        Dot-separated path into the parsed JSON object.  Only used when
        ``output_format == "json"``.  Default ``"content"``.
    strip_ansi_output : bool
        Strip ANSI escape codes from stdout before parsing.  Default
        ``True``.
    timeout_seconds : float
        Subprocess wall-clock timeout.  Default 120 s.  A timeout raises
        :class:`CliLLMError`.
    """

    # ------------------------------------------------------------------
    # Pydantic fields (must be declared for BaseChatModel subclasses)
    # ------------------------------------------------------------------

    command: List[str]
    prompt_via: str = "stdin"
    message_format: str = "last"
    output_format: str = "text"
    json_path: str = "content"
    strip_ansi_output: bool = True
    timeout_seconds: float = 120.0

    # ------------------------------------------------------------------
    # BaseChatModel required property
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "cli"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_run_args(
        self, prompt: str
    ) -> Tuple[List[str], Optional[str], Optional[str]]:
        """Return ``(cmd, stdin_text, tmp_file_path)`` for the subprocess call.

        The caller is responsible for cleaning up ``tmp_file_path`` if set.
        """
        if self.prompt_via == "stdin":
            return list(self.command), prompt, None
        if self.prompt_via == "arg":
            return list(self.command) + [prompt], None, None
        # "file"
        # Write to a temp file and pass the path as an extra arg
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(prompt)
            tmp_path = tmp.name
        return list(self.command) + [tmp_path], None, tmp_path

    def _run_subprocess(self, prompt: str) -> str:
        """Execute the CLI with *prompt* and return the captured output text.

        :raises CliLLMError: On non-zero exit code or timeout.
        """
        cmd, stdin_text, tmp_path = self._build_run_args(prompt)
        LOGGER.debug(
            "CliLLM: running %s (prompt_via=%s, timeout=%ss)",
            cmd[0],
            self.prompt_via,
            self.timeout_seconds,
        )
        try:
            result = subprocess.run(  # pylint: disable=subprocess-run-check
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            raise CliLLMError(
                f"CLI subprocess timed out after {self.timeout_seconds}s: {cmd[0]}"
            ) from exc
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Best-effort cleanup

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise CliLLMError(
                f"CLI subprocess exited with code {result.returncode}: "
                f"{cmd[0]}\nstderr: {stderr_snippet}"
            )

        output = result.stdout
        if self.strip_ansi_output:
            output = strip_ansi(output)
        return output

    def _parse_output(self, raw: str) -> str:
        """Parse *raw* stdout according to ``output_format``."""
        if self.output_format == "text":
            return raw.strip()

        # json
        import json  # pylint: disable=import-outside-toplevel

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliLLMError(
                f"CLI output is not valid JSON (output_format='json'): {exc}"
            ) from exc
        try:
            return extract_json_path(parsed, self.json_path)
        except (KeyError, IndexError, TypeError) as exc:
            raise CliLLMError(
                f"json_path={self.json_path!r} not found in CLI output: {exc}"
            ) from exc

    def _call_cli(self, messages: List[BaseMessage]) -> str:
        """Render messages, run the CLI, and return the parsed response text."""
        prompt = render_messages(messages, self.message_format)
        raw = self._run_subprocess(prompt)
        return self._parse_output(raw)

    # ------------------------------------------------------------------
    # BaseChatModel abstract method implementation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # pylint: disable=unused-argument
        run_manager: Optional[Any] = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> ChatResult:
        """Run the CLI and return the result as a :class:`ChatResult`."""
        content = self._call_cli(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    # ------------------------------------------------------------------
    # Streaming: yield the full response as ONE chunk (honest, not faked)
    # ------------------------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # pylint: disable=unused-argument
        run_manager: Optional[Any] = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the CLI response as a single chunk.

        The subprocess runs to completion before any output is available, so
        true per-token streaming is not possible.  Yielding one chunk is the
        honest contract: callers receive all content in the first (and only)
        iteration, which is fully compatible with LangChain streaming
        consumers.
        """
        content = self._call_cli(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # pylint: disable=unused-argument
        run_manager: Optional[Any] = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ):
        """Async stream — delegates to the sync path (subprocess is blocking).

        Yields the full response as a single :class:`AIMessageChunk`.
        """
        content = self._call_cli(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))


# ---------------------------------------------------------------------------
# CliProvider — the LLMProvider factory
# ---------------------------------------------------------------------------


# pylint: disable=too-few-public-methods
class CliProvider(LLMProvider):
    """LLM provider that drives a command-line tool as a stateless text model.

    Returns a :class:`CliLLM` instance configured from the supplied kwargs.
    No optional SDK is required -- the implementation uses ``subprocess``
    from the Python standard library.

    **Auth note:** The subprocess inherits ``os.environ`` in full.  Whatever
    authentication the CLI tool requires (OAuth session written to a config
    file, environment variable, etc.) must already be present in the calling
    process's environment.  bili-core does not manage credentials for CLI
    tools.

    Accepted kwargs
    ---------------
    command : list[str]
        **Required.** The executable and any fixed arguments.  Example::

            command=["my-llm-cli", "--model", "fast", "--no-color"]

    prompt_via : str, optional
        How the prompt is delivered to the process.
        ``"stdin"`` (default), ``"arg"``, or ``"file"``.

    message_format : str, optional
        How the LangChain message list is rendered into a prompt string.
        ``"last"`` (default), ``"roles"``, or ``"chatml"``.

    output_format : str, optional
        How the subprocess stdout is parsed.
        ``"text"`` (default) or ``"json"``.

    json_path : str, optional
        Dot-separated extraction path for ``output_format="json"``.
        Default ``"content"``.

    strip_ansi : bool, optional
        Strip ANSI colour codes from stdout.  Default ``True``.

    timeout_seconds : float, optional
        Per-call wall-clock timeout in seconds.  Default ``120``.
    """

    # The `strip_ansi` parameter intentionally shares its name with the
    # module-level helper function.  The parameter is the public API kwarg
    # documented in the class docstring; the module function is an
    # implementation detail.  Pylint sees the parameter as shadowing the
    # outer-scope name, which is benign here.
    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments,redefined-outer-name
        self,
        command: List[str],
        prompt_via: str = "stdin",
        message_format: str = "last",
        output_format: str = "text",
        json_path: str = "content",
        strip_ansi: bool = True,
        timeout_seconds: float = 120.0,
        **_extra: Any,
    ) -> CliLLM:
        """Create and return a :class:`CliLLM` instance.

        :param command: The executable and fixed arguments list.
        :param prompt_via: How the prompt is sent -- ``"stdin"``,
            ``"arg"``, or ``"file"``.
        :param message_format: Message rendering strategy -- ``"last"``,
            ``"roles"``, or ``"chatml"``.
        :param output_format: Output parsing strategy -- ``"text"`` or
            ``"json"``.
        :param json_path: Extraction path for JSON output.
        :param strip_ansi: Strip ANSI escape codes from stdout.
        :param timeout_seconds: Per-call subprocess timeout in seconds.
        :returns: A configured :class:`CliLLM` instance.
        :raises ValueError: If ``command`` is empty, or any config value is
            not among the supported options.
        """
        if not command:
            raise ValueError("CliProvider requires a non-empty 'command' list")
        if prompt_via not in SUPPORTED_PROMPT_VIA:
            raise ValueError(
                f"prompt_via={prompt_via!r} is not supported. "
                f"Choose from: {sorted(SUPPORTED_PROMPT_VIA)}"
            )
        if message_format not in SUPPORTED_MESSAGE_FORMATS:
            raise ValueError(
                f"message_format={message_format!r} is not supported. "
                f"Choose from: {sorted(SUPPORTED_MESSAGE_FORMATS)}"
            )
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format={output_format!r} is not supported. "
                f"Choose from: {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )

        LOGGER.info("Initializing CliLLM: command=%s", command[0])

        llm = CliLLM(
            command=command,
            prompt_via=prompt_via,
            message_format=message_format,
            output_format=output_format,
            json_path=json_path,
            strip_ansi_output=strip_ansi,
            timeout_seconds=timeout_seconds,
        )
        LOGGER.debug(llm)
        return llm
