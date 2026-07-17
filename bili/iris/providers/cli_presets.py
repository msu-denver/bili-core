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

Per-CLI model / reasoning-effort support
-----------------------------------------
Each preset optionally supports pinning the underlying ``model`` and a
``reasoning_effort`` (reasoning-effort / thinking-budget) value, overriding
whatever the CLI's own global default or interactive session would
otherwise use.  Support and mechanism are CLI-specific:

``cli_claude_code``
    ``model`` via ``--model <value>``.  ``reasoning_effort`` via
    ``--effort <value>`` (e.g. ``low``/``medium``/``high``/``xhigh``/``max``).

``cli_codex``
    ``model`` via ``--model <value>``.  ``reasoning_effort`` via
    ``-c model_reasoning_effort="<value>"`` (e.g.
    ``low``/``medium``/``high``/``xhigh``).

``cli_gemini_cli``
    ``model`` via ``--model <value>``.  ``reasoning_effort`` is **not
    CLI-settable**: the Gemini CLI only exposes a thinking-budget control
    via ``.gemini/settings.json`` (or interactive ``/think`` / ``/budget``
    slash commands), not a headless-mode (``-p``) flag or env var.  Setting
    ``reasoning_effort`` on this preset is a documented no-op -- a warning
    is logged and the CLI's own default is used.

Both settings default to ``None`` (no override -- the CLI's own default
model and reasoning depth are used, matching historical behaviour).

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

# pylint: disable=duplicate-code  # CliPreset fields intentionally mirror CliLLM by design
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
    :param max_retries: Additional in-process attempts after an initial
        transient failure (rate limit, overload, transient 5xx), before
        raising ``CliLLMError``.  Default ``2``.  Set to ``0`` to disable
        retry entirely.  Only clearly-transient failures are retried;
        permanent failures always fail on the first attempt.
    :param retry_backoff_seconds: Base delay in seconds before the first
        retry; doubles on each subsequent retry.  Default ``1.0``.
    :param model: Model name/ID to pass to the CLI, overriding whatever
        model the CLI's own global default or interactive session would
        otherwise use.  Default ``None`` (no override).  Applied via
        ``model_flag_template``.
    :param reasoning_effort: Reasoning-effort / thinking-budget value to
        pass to the CLI (the accepted vocabulary is CLI-specific).  Default
        ``None`` (no override).  Applied via
        ``reasoning_effort_flag_template``; a value set with no template
        configured is a documented no-op (a warning is logged).
    :param model_flag_template: Argv template used to render ``model`` into
        command-line tokens (``"{value}"`` is replaced by ``model``).
        Default ``["--model", "{value}"]`` -- the near-universal convention
        across CLI LLM tools.  Set to ``None`` to disable model-flag
        injection for this preset.
    :param reasoning_effort_flag_template: Argv template used to render
        ``reasoning_effort`` into command-line tokens, analogous to
        ``model_flag_template``.  Default ``None`` (no known cross-CLI
        convention); set explicitly for CLIs that support a
        reasoning-effort control.
    """

    command: List[str] = field(default_factory=list)
    prompt_via: str = "arg"
    message_format: str = "last"
    output_format: str = "text"
    json_path: str = "content"
    strip_ansi: bool = True
    timeout_seconds: Optional[float] = 1800.0
    cwd: Optional[str] = None
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    model_flag_template: Optional[List[str]] = field(
        default_factory=lambda: ["--model", "{value}"]
    )
    reasoning_effort_flag_template: Optional[List[str]] = None


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
#: ``model`` is applied via ``--model <value>``.  ``reasoning_effort`` is
#: applied via ``--effort <value>`` (accepted values are defined by the
#: Claude Code CLI, e.g. ``"low"``/``"medium"``/``"high"``/``"xhigh"``/
#: ``"max"``).
#:
#: Reference: ``claude --help`` (look for ``-p, --print``, ``--model``,
#: ``--effort``).
CLAUDE_CODE_PRESET = CliPreset(
    command=["claude", "-p"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
    model_flag_template=["--model", "{value}"],
    reasoning_effort_flag_template=["--effort", "{value}"],
)

#: OpenAI Codex CLI -- ``codex exec <prompt>``
#:
#: Invokes ``codex exec`` in non-interactive mode.  The prompt is passed as a
#: positional argument.  The CLI prints the final agent message to stdout and
#: exits.  Requires the Codex CLI to be installed and authenticated via
#: ``OPENAI_API_KEY`` or its interactive login flow.
#:
#: ``model`` is applied via ``--model <value>``.  ``reasoning_effort`` is
#: applied via a ``-c model_reasoning_effort="<value>"`` config override
#: (accepted values are defined by the Codex CLI, e.g. ``"low"``/
#: ``"medium"``/``"high"``/``"xhigh"``).
#:
#: Reference: ``codex exec --help`` / OpenAI Codex CLI configuration reference.
CODEX_PRESET = CliPreset(
    command=["codex", "exec"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
    model_flag_template=["--model", "{value}"],
    reasoning_effort_flag_template=["-c", 'model_reasoning_effort="{value}"'],
)

#: Google Gemini CLI -- ``gemini -p <prompt>``
#:
#: Invokes the ``gemini`` executable in non-interactive headless mode
#: (``-p`` / ``--prompt``).  Output is written to stdout as plain text and
#: the process exits.  Requires the Gemini CLI to be installed and
#: authenticated (Google account OAuth or ``GEMINI_API_KEY``).
#:
#: ``model`` is applied via ``--model <value>``.  ``reasoning_effort`` has
#: **no CLI-settable equivalent** for this preset: the Gemini CLI only
#: exposes a thinking-budget control via ``.gemini/settings.json`` (or the
#: interactive ``/think`` / ``/budget`` slash commands), not a headless-mode
#: flag or environment variable.  Setting ``reasoning_effort`` on this
#: preset is a documented no-op -- a warning is logged and the CLI's own
#: default is used.
#:
#: Reference: ``gemini --help`` (look for ``-p, --prompt``, ``--model``).
GEMINI_CLI_PRESET = CliPreset(
    command=["gemini", "-p"],
    prompt_via="arg",
    message_format="last",
    output_format="text",
    strip_ansi=True,
    timeout_seconds=1800.0,
    model_flag_template=["--model", "{value}"],
    reasoning_effort_flag_template=None,
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
