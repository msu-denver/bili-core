"""Model / reasoning-effort argv-templating for CLI-backed LLM providers.

Extracted from :mod:`bili.iris.providers.cli_provider` into its own module so
the same templating logic can be shared, without duplication, by both CLI
subprocess execution paths in bili-core:

- The direct path: :meth:`~bili.iris.providers.cli_provider.CliLLM._build_run_args`.
- The MCP tool-strategy path: :func:`~bili.iris.mcp.server.build_mcp_node`.

Without any further configuration, a CLI subprocess uses whatever model and
reasoning depth the CLI tool's own global default or interactive session is
set to.  For a consumer that wants a specific, cost-controlled model for a
given role -- or wants to avoid a heavy default reasoner turning a single
mechanical call into a multi-minute wait -- ``model`` and ``reasoning_effort``
settings (declared on
:class:`~bili.iris.providers.cli_provider.CliLLM`) pin both explicitly:

``model``
    The model name/ID to pass to the CLI.  Applied via ``model_flag_template``
    (default ``("--model", "{value}")``, the near-universal convention across
    CLI LLM tools).

``reasoning_effort``
    The reasoning-effort / thinking-budget level to pass to the CLI (the
    accepted vocabulary -- e.g. ``"low"``/``"medium"``/``"high"``/``"max"`` --
    is defined by the target CLI, not by bili-core).  Applied via
    ``reasoning_effort_flag_template``, which has no generic default because
    there is no cross-CLI convention for this control.  Named presets that
    know a specific CLI's flag syntax (see
    :mod:`bili.iris.providers.cli_presets`) configure this template; CLIs
    with no CLI-settable reasoning-effort control leave it ``None``, which is
    a documented no-op (a warning is logged and the setting is dropped).

Both settings default to ``None`` (no flag added -- unconfigured behaviour is
unchanged).
"""

import logging
from typing import List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

#: Default argv template used to pass a ``model`` override to a CLI tool when
#: the caller sets ``model`` but does not override ``model_flag_template``.
#: ``--model <value>`` is the near-universal convention across CLI LLM tools
#: (Claude Code, Codex, Gemini CLI, and most others), so it is a safe default
#: for the generic (non-preset)
#: :class:`~bili.iris.providers.cli_provider.CliProvider` path.  Pass
#: ``model_flag_template=None`` to disable model-flag injection entirely
#: (e.g. when the model is already baked into ``command``).
DEFAULT_MODEL_FLAG_TEMPLATE: Tuple[str, ...] = ("--model", "{value}")


def build_model_and_effort_args(
    model: Optional[str],
    model_flag_template: Optional[Sequence[str]],
    reasoning_effort: Optional[str],
    reasoning_effort_flag_template: Optional[Sequence[str]],
    cli_name: str = "",
) -> List[str]:
    """Return extra argv tokens for a configured model and/or reasoning effort.

    Each ``*_flag_template`` is a sequence of argv tokens in which the literal
    substring ``"{value}"`` is replaced by the corresponding value -- e.g.
    ``["--model", "{value}"]`` renders to ``["--model", "gpt-5"]``, and
    ``["-c", 'model_reasoning_effort="{value}"']`` renders to
    ``["-c", 'model_reasoning_effort="high"']``.  This lets each CLI preset
    describe its own flag syntax (a plain ``--flag value`` pair, a combined
    ``-c key=value`` config override, or anything else expressible as argv
    tokens) without any CLI-specific branching in the caller -- the same
    templating is reused by both the direct subprocess path
    (:meth:`~bili.iris.providers.cli_provider.CliLLM._build_run_args`) and the
    MCP tool-strategy path (:func:`~bili.iris.mcp.server.build_mcp_node`).

    A configured value with no corresponding template is a documented no-op:
    a warning is logged and the setting is dropped rather than guessing at a
    flag the target CLI may not support.  This is the "no CLI-settable
    control" case -- e.g. the Gemini CLI has no headless-mode flag for its
    thinking-budget setting, so ``reasoning_effort`` is accepted on that
    preset but has no effect.

    :param model: The model name/ID to pass to the CLI, or ``None``.
    :param model_flag_template: Argv template for *model*, or ``None`` if
        this CLI has no configured model-selection flag.
    :param reasoning_effort: The reasoning-effort / thinking-budget value to
        pass to the CLI, or ``None``.
    :param reasoning_effort_flag_template: Argv template for
        *reasoning_effort*, or ``None`` if this CLI has no known
        CLI-settable reasoning-effort / thinking-budget control.
    :param cli_name: The CLI executable name, used only in the warning
        message logged when a value is configured but has no template.
    :returns: A list of extra argv tokens to append to the base command
        (empty if neither *model* nor *reasoning_effort* is set).
    """
    extra: List[str] = []

    if model:
        if model_flag_template:
            extra.extend(part.format(value=model) for part in model_flag_template)
        else:
            LOGGER.warning(
                "CliLLM: model=%r requested for %r but no model_flag_template "
                "is configured for this CLI; the override is ignored and the "
                "CLI's own default model will be used.",
                model,
                cli_name or "<cli>",
            )

    if reasoning_effort:
        if reasoning_effort_flag_template:
            extra.extend(
                part.format(value=reasoning_effort)
                for part in reasoning_effort_flag_template
            )
        else:
            LOGGER.warning(
                "CliLLM: reasoning_effort=%r requested for %r but this CLI "
                "has no known CLI-settable reasoning-effort / "
                "thinking-budget control; the setting is ignored.",
                reasoning_effort,
                cli_name or "<cli>",
            )

    return extra


__all__ = [
    "DEFAULT_MODEL_FLAG_TEMPLATE",
    "build_model_and_effort_args",
]
