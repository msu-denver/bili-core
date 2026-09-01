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

Before it gets that far, though, the provider itself retries a failure that
looks *transient* (rate limiting, temporary overload, a transient 5xx from
whatever backend the CLI tool talks to) in-process, with exponential
backoff, up to a configurable number of attempts.  This matters most for a
consumer running a CLI provider with no API-provider fallback configured at
all: without in-process retry, a single rate-limit blip from the CLI's own
upstream call fails the whole turn immediately, with nothing to fall through
to.  See :data:`_TRANSIENT_ERROR_PATTERNS` for the transient-vs-permanent
detection approach.

The LLM object returned by :meth:`CliProvider.load` is a
:class:`CliLLM` -- a concrete :class:`~langchain_core.language_models.BaseChatModel`
subclass.  This makes it a drop-in replacement inside any LangChain/LangGraph
pipeline, AETHER agent, or IRIS node without modification.

Message rendering
-----------------
CLI tools accept a single text prompt, not a structured message list.  This
transport is therefore text-only by construction: whichever format is chosen
below, the message list ends up as one prompt string.  A message carrying an
image content part is REFUSED with
:class:`~bili.iris.providers.modality.UnsupportedInputModalityError` rather
than stringified, because a stringified image is a dropped image that looks
like a successful turn.

The provider supports three ``message_format`` strategies:

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

Model and reasoning-effort selection
-------------------------------------
``CliLLM.model`` and ``CliLLM.reasoning_effort`` pin a specific model and
reasoning depth for the spawned CLI, instead of inheriting whatever the CLI
tool's own global default or interactive session is set to.  Both default to
``None`` (no override -- unconfigured behaviour is unchanged).  See
:mod:`bili.iris.providers.cli_model_flags` for the full rationale and the
shared argv-templating mechanism used by both this module's direct
subprocess path and the MCP tool-strategy path
(:func:`bili.iris.mcp.server.build_mcp_node`).

No new optional dependency is required; the module uses stdlib ``subprocess``
and ``re`` only.
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from bili.iris.multimodal import non_text_part_types

from .base import LLMProvider
from .cli_model_flags import DEFAULT_MODEL_FLAG_TEMPLATE, build_model_and_effort_args
from .modality import UnsupportedInputModalityError, describe_message_modalities

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
# Transient-vs-permanent failure detection
# ---------------------------------------------------------------------------

#: Regex patterns checked (case-insensitively) against the text of a failed
#: CLI invocation -- the ``CliLLMError`` message built from the captured
#: stderr -- to decide whether the failure looks TRANSIENT and is therefore
#: worth retrying in-process, versus PERMANENT (fail fast, no retry).
#:
#: Detection approach: text patterns against stderr, not exit codes.  CLI
#: tools are a heterogeneous grab-bag of processes with no shared convention
#: for exit codes -- one tool might exit 1 for "rate limited", another might
#: exit 1 for "bad flag" or "binary not found".  Exit code alone cannot
#: distinguish those cases across arbitrary CLI tools.  Stderr text is a far
#: more portable signal: CLI tools that wrap a hosted LLM API commonly
#: surface the upstream provider's own error text verbatim (e.g. "429",
#: "rate_limit_error", "Overloaded", "503 Service Unavailable") regardless of
#: which CLI emits it.
#:
#: The list is deliberately narrow and conservative.  A broken command, a
#: missing binary, a bad flag, an authentication failure, or a malformed-
#: output parse error are all permanent and must fail on the first attempt;
#: none of those match these patterns.  When in doubt, a failure is treated
#: as permanent -- an unnecessary retry is merely slower, but a retry of a
#: genuinely permanent failure wastes the full backoff schedule for no
#: benefit and delays the caller from seeing (and acting on) the real error.
_TRANSIENT_ERROR_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"rate[\s_-]?limit",
        r"too many requests",
        r"overloaded",
        r"resource[\s_-]?exhausted",
        r"quota exceeded",
        r"throttl",
        r"service unavailable",
        r"temporarily unavailable",
        r"\b429\b",
        r"\b500\b",
        r"\b502\b",
        r"\b503\b",
        r"\b504\b",
        r"\b529\b",
    )
)


def _is_transient_failure(error_text: str) -> bool:
    """Return ``True`` if *error_text* matches a known transient-failure
    signature.

    :param error_text: The message text of a raised :class:`CliLLMError`
        (which embeds the captured stderr snippet for non-zero exits).
    :returns: ``True`` if *error_text* looks like a transient upstream
        condition (rate limit, overload, transient 5xx) safe to retry;
        ``False`` otherwise, including subprocess timeouts, whose messages
        never match these patterns and are therefore always treated as
        permanent -- a timeout already waited out the full configured
        budget, so retrying would only double the wall-clock cost without a
        clear signal that the retry will fare any better.
    """
    return any(pattern.search(error_text) for pattern in _TRANSIENT_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def _require_text_only(msg: BaseMessage) -> BaseMessage:
    """Return *msg*, or raise if it carries a non-text content part.

    A CLI tool consumes text on stdin or as an argv value, so this transport
    has no channel for an image.  ``str(msg.content)`` on a multimodal message
    yields the *repr* of the parts list, which loses the image while producing
    a plausible-looking prompt: the subprocess succeeds, the caller gets an
    answer, and nothing anywhere says the image was never sent.  Refusing by
    name is the honest behaviour -- the caller can then route that turn to an
    image-capable provider.

    :param msg: The message about to be rendered into the prompt string.
    :returns: *msg*, unchanged, when its content is text.
    :raises UnsupportedInputModalityError: When it carries a recognised
        non-text content part.
    """
    kinds = non_text_part_types(getattr(msg, "content", None))
    if not kinds:
        return msg
    # Name the modality where the part type maps to one ("image"), and fall
    # back to the raw part types for a part outside that vocabulary, so the
    # refusal always says what it refused.
    named = describe_message_modalities([msg]) or kinds
    raise UnsupportedInputModalityError(
        f"A CLI provider cannot carry {', '.join(named)} content "
        f"(content part(s): {', '.join(kinds)}): CLI tools consume a single "
        "text prompt, so the part would be dropped. Route this turn to a "
        "provider whose model accepts it (see "
        "bili.iris.providers.modality.models_supporting_input_modality)."
    )


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
    :raises UnsupportedInputModalityError: If a message that would be rendered
        carries a non-text content part.  The check is scoped to the messages
        this format actually renders, so ``"last"`` does not refuse over an
        image in history it was never going to carry.
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
                return str(_require_text_only(msg).content)
        # Fall back to the very last message if no HumanMessage found
        return str(_require_text_only(messages[-1]).content)

    if message_format == "roles":
        parts = []
        for msg in messages:
            checked = _require_text_only(msg)
            parts.append(f"{_role_label(checked)}: {checked.content}")
        return "\n".join(parts)

    # chatml
    parts = []
    for msg in messages:
        checked = _require_text_only(msg)
        # Normalise "User" -> "user", "System" -> "system", etc.
        role = _role_label(checked).lower()
        parts.append(f"<|im_start|>{role}\n{checked.content}\n<|im_end|>")
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
    timeout_seconds : float or None
        Subprocess wall-clock timeout in seconds.  Default ``1800`` (30 min),
        which is intentionally generous for long-running agentic turns where
        the CLI tool may spend several minutes reasoning, searching, or
        generating large artifacts before producing output.  Set to ``None``
        to disable the timeout entirely (the subprocess runs until it exits
        or the process is killed).  A finite timeout raises
        :class:`CliLLMError` when exceeded.
    cwd : str or None
        Working directory for the subprocess.  Default ``None``, which
        preserves historical behaviour: the subprocess inherits the calling
        process's current working directory (``subprocess.run``'s own
        default).  Set to a fixed path to pin every invocation to a
        caller-controlled directory instead -- useful when the CLI tool
        gates filesystem access by directory (a one-time trust decision per
        directory rather than one per caller cwd) or when the caller wants
        to scope the tool's filesystem reach to a dedicated sandbox rather
        than exposing whatever directory the calling process happens to be
        running from.
    max_retries : int
        Number of additional attempts after an initial transient failure,
        before giving up and raising :class:`CliLLMError`.  Default ``2`` --
        small enough that a persistently-broken CLI does not multiply the
        caller's wall-clock cost many times over, but enough that a rate
        limit or overload signal (which typically clears within a few
        seconds) usually succeeds on retry.  Set to ``0`` to disable retry
        entirely, matching the historical behaviour of raising immediately
        on the first failure.  Only failures that match a known transient
        signature are retried (see :data:`_TRANSIENT_ERROR_PATTERNS`);
        permanent failures (bad command, auth error, malformed output)
        always fail on the first attempt regardless of this setting.
    retry_backoff_seconds : float
        Base delay, in seconds, before the first retry.  Each subsequent
        retry doubles the delay (``retry_backoff_seconds * 2 ** attempt``).
        Default ``1.0``.
    model : str or None
        Model name/ID to pass to the CLI, overriding whatever model the
        CLI's own global default or interactive session would otherwise use.
        Default ``None`` (no override -- the CLI's own default model is
        used, matching historical behaviour).  Applied via
        ``model_flag_template``; see :func:`build_model_and_effort_args`.
    reasoning_effort : str or None
        Reasoning-effort / thinking-budget value to pass to the CLI (the
        accepted vocabulary is CLI-specific, e.g. ``"low"``/``"medium"``/
        ``"high"``/``"max"``).  Default ``None`` (no override).  Applied via
        ``reasoning_effort_flag_template``; a value set with no template
        configured for the target CLI is a documented no-op (a warning is
        logged).  See :func:`build_model_and_effort_args`.
    model_flag_template : list[str] or None
        Argv template used to render ``model`` into command-line tokens (the
        literal substring ``"{value}"`` is replaced by ``model``).  Default
        ``["--model", "{value}"]`` -- the near-universal convention across
        CLI LLM tools.  Set to ``None`` to disable model-flag injection
        entirely for this CLI.
    reasoning_effort_flag_template : list[str] or None
        Argv template used to render ``reasoning_effort`` into command-line
        tokens, analogous to ``model_flag_template``.  Default ``None`` --
        there is no cross-CLI convention for this control, so named presets
        that know a specific CLI's syntax configure this explicitly (see
        :mod:`bili.iris.providers.cli_presets`).
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
    timeout_seconds: Optional[float] = 1800.0
    cwd: Optional[str] = None
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    model_flag_template: Optional[List[str]] = list(DEFAULT_MODEL_FLAG_TEMPLATE)
    reasoning_effort_flag_template: Optional[List[str]] = None

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

        The base command is ``self.command`` with any configured ``model`` /
        ``reasoning_effort`` flags appended (see
        :func:`build_model_and_effort_args`), before the prompt itself is
        added according to ``prompt_via``.

        The caller is responsible for cleaning up ``tmp_file_path`` if set.
        """
        base_command = list(self.command) + build_model_and_effort_args(
            self.model,
            self.model_flag_template,
            self.reasoning_effort,
            self.reasoning_effort_flag_template,
            cli_name=self.command[0] if self.command else "",
        )
        if self.prompt_via == "stdin":
            return base_command, prompt, None
        if self.prompt_via == "arg":
            return base_command + [prompt], None, None
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
        return base_command + [tmp_path], None, tmp_path

    def _run_subprocess(self, prompt: str) -> str:
        """Execute the CLI with *prompt*, retrying transient failures.

        Calls :meth:`_run_subprocess_once` up to ``max_retries + 1`` times.
        A failure is retried (with exponential backoff) only when
        :func:`_is_transient_failure` matches the raised
        :class:`CliLLMError`'s message; any other failure -- or the final
        attempt of a persistently-transient one -- is re-raised immediately.

        :raises CliLLMError: On a non-transient failure, or once transient
            retries are exhausted.
        """
        max_attempts = max(self.max_retries, 0) + 1
        last_error: Optional[CliLLMError] = None
        for attempt in range(max_attempts):
            try:
                return self._run_subprocess_once(prompt)
            except CliLLMError as exc:
                last_error = exc
                attempts_remaining = max_attempts - attempt - 1
                if attempts_remaining <= 0 or not _is_transient_failure(str(exc)):
                    raise
                delay = self.retry_backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "CliLLM: transient failure on attempt %d/%d (%s); "
                    "retrying in %.1fs",
                    attempt + 1,
                    max_attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        # Unreachable: the loop above always returns from a successful
        # attempt or raises via the `raise` statement above.  Kept only to
        # satisfy static analysis that every code path returns or raises.
        raise last_error  # pragma: no cover

    def _run_subprocess_once(self, prompt: str) -> str:
        """Execute the CLI once with *prompt* and return captured output text.

        A single subprocess invocation, no retry.  Called by
        :meth:`_run_subprocess`, which wraps this in the retry loop.

        :raises CliLLMError: On non-zero exit code or timeout.
        """
        cmd, stdin_text, tmp_path = self._build_run_args(prompt)
        timeout_display: Union[str, float] = (
            self.timeout_seconds if self.timeout_seconds is not None else "none"
        )
        LOGGER.debug(
            "CliLLM: running %s (prompt_via=%s, timeout=%ss, cwd=%s)",
            cmd[0],
            self.prompt_via,
            timeout_display,
            self.cwd or "<inherited>",
        )
        try:
            result = subprocess.run(  # pylint: disable=subprocess-run-check
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,  # None disables the timeout
                cwd=self.cwd,  # None inherits the calling process's cwd
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
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream the CLI response as a single chunk without blocking
        the event loop.

        The subprocess is run in a thread via :func:`asyncio.to_thread` so the
        event loop remains free for other coroutines while the CLI tool
        executes.  Yields the full response as a single
        :class:`ChatGenerationChunk` once the subprocess completes.
        """
        content = await asyncio.to_thread(self._call_cli, messages)
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

    timeout_seconds : float or None, optional
        Per-call wall-clock timeout in seconds.  Default ``1800`` (30 min).
        Pass ``None`` to disable the timeout entirely for CLI tools whose
        runtime is unbounded (e.g. long-form generation or multi-step
        agentic reasoning).

    cwd : str or None, optional
        Working directory the subprocess is spawned in.  Default ``None``,
        which preserves the historical behaviour of inheriting the calling
        process's current working directory.  Pass a fixed path to pin the
        subprocess to a caller-controlled directory instead -- for CLI tools
        that gate filesystem access per directory (so trust is granted once
        for a known directory rather than re-triggered by every caller cwd)
        or to scope the tool's filesystem reach to a dedicated directory
        rather than whatever directory the calling process happens to be
        running from.

    max_retries : int, optional
        Number of additional in-process attempts after an initial transient
        failure (rate limit, overload, transient 5xx -- see
        :data:`_TRANSIENT_ERROR_PATTERNS`), before raising
        :class:`CliLLMError`.  Default ``2``.  Pass ``0`` to disable retry
        entirely, matching the historical (pre-retry) behaviour of raising
        immediately.  Permanent failures always fail on the first attempt
        regardless of this setting.

    retry_backoff_seconds : float, optional
        Base delay, in seconds, before the first retry; doubles on each
        subsequent retry.  Default ``1.0``.

    model : str or None, optional
        Model name/ID to pass to the CLI, overriding whatever model the
        CLI's own global default or interactive session would otherwise use.
        Default ``None`` (no override -- today's behaviour, unchanged).
        Applied via ``model_flag_template``.

    reasoning_effort : str or None, optional
        Reasoning-effort / thinking-budget value to pass to the CLI (the
        accepted vocabulary -- e.g. ``"low"``/``"medium"``/``"high"``/
        ``"max"`` -- is defined by the target CLI).  Default ``None`` (no
        override).  Applied via ``reasoning_effort_flag_template``; a value
        set with no template configured for this CLI is a documented no-op
        (a warning is logged, no flag is added).

    model_flag_template : sequence of str or None, optional
        Argv template used to render ``model`` into command-line tokens (the
        literal substring ``"{value}"`` is replaced by ``model``).  Default
        ``("--model", "{value}")`` -- the near-universal convention across
        CLI LLM tools.  Pass ``None`` to disable model-flag injection
        entirely for this CLI.

    reasoning_effort_flag_template : sequence of str or None, optional
        Argv template used to render ``reasoning_effort`` into command-line
        tokens, analogous to ``model_flag_template``.  Default ``None`` --
        there is no cross-CLI convention for this control, so named presets
        that know a specific CLI's syntax configure this explicitly (see
        :mod:`bili.iris.providers.cli_presets`).
    """

    # The `strip_ansi` parameter intentionally shares its name with the
    # module-level helper function.  The parameter is the public API kwarg
    # documented in the class docstring; the module function is an
    # implementation detail.  Pylint sees the parameter as shadowing the
    # outer-scope name, which is benign here.
    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments,too-many-locals,redefined-outer-name
        self,
        command: List[str],
        prompt_via: str = "stdin",
        message_format: str = "last",
        output_format: str = "text",
        json_path: str = "content",
        strip_ansi: bool = True,
        timeout_seconds: Optional[float] = 1800.0,
        cwd: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        model_flag_template: Optional[Sequence[str]] = DEFAULT_MODEL_FLAG_TEMPLATE,
        reasoning_effort_flag_template: Optional[Sequence[str]] = None,
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
            ``None`` disables the timeout.
        :param cwd: Working directory for the subprocess.  ``None`` (default)
            inherits the calling process's current working directory,
            matching historical behaviour.  Pass a fixed path to pin the
            subprocess to a caller-controlled directory.
        :param max_retries: Additional attempts after an initial transient
            failure, before raising.  ``0`` disables retry.
        :param retry_backoff_seconds: Base delay in seconds before the first
            retry; doubles on each subsequent retry.
        :param model: Model name/ID to pass to the CLI.  ``None`` (default)
            inherits the CLI's own default model.
        :param reasoning_effort: Reasoning-effort / thinking-budget value to
            pass to the CLI.  ``None`` (default) inherits the CLI's own
            default reasoning depth.
        :param model_flag_template: Argv template for *model*.  Default
            ``("--model", "{value}")``.  Pass ``None`` to disable model-flag
            injection for this CLI.
        :param reasoning_effort_flag_template: Argv template for
            *reasoning_effort*.  Default ``None`` (no known cross-CLI
            convention); named presets configure this explicitly for CLIs
            that support a reasoning-effort control.
        :returns: A configured :class:`CliLLM` instance.
        :raises ValueError: If ``command`` is empty, any config value is not
            among the supported options, or ``max_retries``/
            ``retry_backoff_seconds`` is negative.
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
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        if retry_backoff_seconds < 0:
            raise ValueError(
                f"retry_backoff_seconds must be >= 0, got {retry_backoff_seconds!r}"
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
            cwd=cwd,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            model=model,
            reasoning_effort=reasoning_effort,
            model_flag_template=(
                list(model_flag_template) if model_flag_template is not None else None
            ),
            reasoning_effort_flag_template=(
                list(reasoning_effort_flag_template)
                if reasoning_effort_flag_template is not None
                else None
            ),
        )
        LOGGER.debug(llm)
        return llm
