"""Named presets for common CLI LLM tools.

Each preset bundles the known one-shot invocation pattern, prompt-delivery
strategy, output format, and sensible defaults for a specific CLI tool, so
callers can load a CLI-backed LLM **by name** instead of reverse-engineering
the tool's flags.

Usage
-----
Register a preset provider type at application startup and then pass the
preset's provider-type string to :func:`~bili.iris.loaders.llm_loader.load_model`:

.. code-block:: python

    import bili.iris.providers.builtin  # noqa: F401  (registers built-ins)
    from bili.iris.loaders.llm_loader import load_model

    # Load Claude Code in one-shot mode -- no flags to look up.
    llm = load_model("cli_claude_code")

    # Override the timeout for a slow query.
    llm = load_model("cli_claude_code", timeout_seconds=300.0)

Each preset maps to a :class:`~bili.iris.providers.preset_provider.CliPresetProvider`
registered under its type string in
:data:`~bili.iris.providers.registry.PROVIDER_REGISTRY`.

Built-in presets
----------------
``cli_claude_code``
    Drives ``claude -p <prompt>`` (Anthropic Claude Code CLI) in one-shot
    print mode.  The subprocess inherits the calling process's environment,
    which supplies the OAuth session or API key that the Claude CLI already
    holds.

``cli_codex``
    Drives ``codex exec <prompt>`` (OpenAI Codex CLI) in non-interactive
    mode.  Requires the Codex CLI to be installed and authenticated.

``cli_gemini_cli``
    Drives ``gemini -p <prompt>`` (Google Gemini CLI) in non-interactive
    headless mode.  Requires the Gemini CLI to be installed and authenticated.

Adding custom presets
---------------------
Call :func:`register_cli_preset` at application startup to register an
additional preset under its own provider-type string:

.. code-block:: python

    from bili.iris.providers.cli_presets import CliPreset, register_cli_preset

    register_cli_preset(
        "cli_my_tool",
        CliPreset(
            command=["my-llm-tool", "--no-color"],
            prompt_via="stdin",
            output_format="text",
        ),
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# CliPreset dataclass
# ---------------------------------------------------------------------------


@dataclass
class CliPreset:  # pylint: disable=too-many-instance-attributes
    """Configuration bundle for a specific CLI LLM tool.

    Each field mirrors a parameter of
    :meth:`~bili.iris.providers.cli_provider.CliProvider.load`, with a
    sensible per-tool default.  Callers can override any field at
    ``load_model()`` call-time.

    :param command: The executable and its fixed arguments.  The prompt is
        appended separately according to ``prompt_via``.
    :param prompt_via: How the rendered prompt is delivered to the process.
        ``"arg"`` (default for most CLIs -- appended as a positional argument),
        ``"stdin"``, or ``"file"``.

        **Security note:** ``"arg"`` exposes the full prompt text as a
        command-line argument.  On shared or multi-user hosts, process listings
        (``ps aux``, ``/proc/<pid>/cmdline``) may reveal prompt content to
        other users.  Use ``"stdin"`` instead when the CLI supports it and
        prompt confidentiality matters.

    :param message_format: How the LangChain message list is rendered before
        delivery.  ``"last"`` (default) sends only the final human message.

        **Note:** ``"last"`` silently drops all prior conversation turns.
        This is intentional for single-shot CLI tools that have no session
        state, but it means a CLI-backed model used in a multi-turn
        conversation will not see its own earlier responses.  If conversation
        continuity matters, use ``"roles"`` (renders each turn as
        ``Human: … / Assistant: …`` lines) or ``"chatml"`` (ChatML
        ``<|im_start|>`` / ``<|im_end|>`` tokens) where the CLI accepts
        structured multi-turn input.
    :param output_format: How the subprocess stdout is parsed.  ``"text"``
        (default) returns the raw stripped output; ``"json"`` parses JSON and
        extracts the value at ``json_path``.
    :param json_path: Extraction path for JSON output.  Default ``"content"``.
    :param strip_ansi: Strip ANSI escape codes from stdout.  Default ``True``.
    :param timeout_seconds: Per-call subprocess timeout in seconds.
        Default ``1800.0`` (30 min), generous enough for long-running agentic
        turns.  Set to ``None`` to disable the timeout entirely.
    :param cwd: Working directory the subprocess is spawned in.  Default
        ``None``, which preserves the historical behaviour of inheriting the
        calling process's current working directory.  Set to a fixed path to
        pin every invocation of this preset to a caller-controlled directory
        instead -- useful for CLI tools that gate filesystem access per
        directory (a one-time trust decision for a known directory rather
        than one triggered by every caller cwd) or to scope the tool's
        filesystem reach to a dedicated directory.
    """

    command: List[str] = field(default_factory=list)
    prompt_via: str = "arg"
    message_format: str = "last"
    output_format: str = "text"
    json_path: str = "content"
    strip_ansi: bool = True
    timeout_seconds: Optional[float] = 1800.0
    cwd: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in preset definitions
# ---------------------------------------------------------------------------

#: Claude Code CLI -- ``claude -p <prompt>``
#:
#: Invokes the ``claude`` executable in one-shot print mode (``-p`` / ``--print``).
#: Output is written to stdout as plain text and the process exits.  The
#: subprocess inherits the calling process's environment, so whatever OAuth
#: session or ``ANTHROPIC_API_KEY`` the CLI holds is reused automatically.
#:
#: Reference: ``claude --help`` (look for ``-p, --print``).
CLAUDE_CODE_PRESET = CliPreset(
    command=["claude", "-p"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
)

#: OpenAI Codex CLI -- ``codex exec <prompt>``
#:
#: Invokes ``codex exec`` in non-interactive mode.  The prompt is passed as a
#: positional argument.  The CLI prints the final agent message to stdout and
#: exits.  Requires the Codex CLI to be installed and authenticated via
#: ``OPENAI_API_KEY`` or its interactive login flow.
#:
#: Reference: ``codex --help`` / OpenAI Codex CLI documentation.
CODEX_PRESET = CliPreset(
    command=["codex", "exec"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
)

#: Google Gemini CLI -- ``gemini -p <prompt>``
#:
#: Invokes the ``gemini`` executable in non-interactive headless mode
#: (``-p`` / ``--prompt``).  Output is written to stdout as plain text and
#: the process exits.  Requires the Gemini CLI to be installed and
#: authenticated (Google account OAuth or ``GEMINI_API_KEY``).
#:
#: Reference: ``gemini --help`` (look for ``-p, --prompt``).
GEMINI_CLI_PRESET = CliPreset(
    command=["gemini", "-p"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
)

# ---------------------------------------------------------------------------
# Public preset catalog
# ---------------------------------------------------------------------------

#: Mapping from provider-type string to :class:`CliPreset`.
#:
#: Used by :mod:`bili.iris.providers.builtin` to register each preset as a
#: :class:`~bili.iris.providers.preset_provider.CliPresetProvider` in the
#: global :data:`~bili.iris.providers.registry.PROVIDER_REGISTRY`.
CLI_PRESET_CATALOG: dict = {
    "cli_claude_code": CLAUDE_CODE_PRESET,
    "cli_codex": CODEX_PRESET,
    "cli_gemini_cli": GEMINI_CLI_PRESET,
}


def register_cli_preset(provider_type: str, preset: "CliPreset") -> None:
    """Register a custom :class:`CliPreset` in the global provider registry.

    Creates a :class:`~bili.iris.providers.preset_provider.CliPresetProvider`
    for *preset* and registers it under *provider_type* in the global
    :data:`~bili.iris.providers.registry.PROVIDER_REGISTRY`.

    Call this at application startup, before the application begins serving
    requests.

    :param provider_type: The provider-type key to register (e.g.
        ``"cli_my_tool"``).  Must not already be registered.
    :param preset: The :class:`CliPreset` configuration bundle.

    Example::

        from bili.iris.providers.cli_presets import CliPreset, register_cli_preset

        register_cli_preset(
            "cli_my_tool",
            CliPreset(command=["my-llm", "--no-color"], prompt_via="stdin"),
        )
    """
    # Lazy import to avoid circular dependency: cli_presets -> preset_provider
    # -> cli_provider -> base; keeping cli_presets free of provider imports
    # also lets callers import CliPreset without pulling in LangChain.
    from bili.iris.providers.preset_provider import (  # pylint: disable=import-outside-toplevel
        CliPresetProvider,
    )
    from bili.iris.providers.registry import (  # pylint: disable=import-outside-toplevel
        PROVIDER_REGISTRY,
    )

    provider_class = CliPresetProvider.for_preset(preset)
    PROVIDER_REGISTRY.register(provider_type, provider_class)
