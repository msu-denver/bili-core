"""Provider wrapper that applies a :class:`~bili.iris.providers.cli_presets.CliPreset`
as default configuration and delegates to
:class:`~bili.iris.providers.cli_provider.CliProvider`.

This module exists as a thin adapter layer between the named-preset catalog
and the generic :class:`~bili.iris.providers.cli_provider.CliProvider`.  It
is not part of the public API; use
:data:`~bili.iris.providers.cli_presets.CLI_PRESET_CATALOG` and
:func:`~bili.iris.providers.cli_presets.register_cli_preset` instead.

Design note
-----------
A dynamically-generated subclass per preset (via :meth:`CliPresetProvider.for_preset`)
is used so that each preset can be registered as a separate *class* in the
:class:`~bili.iris.providers.registry.ProviderRegistry`.  The registry stores
*classes* (not instances), and each registered class must be distinct --
registering the same class under two different keys would mean both keys share
the same preset defaults.  The ``for_preset`` factory avoids a proliferation of
hand-written subclasses.
"""

import logging
from typing import Any, List, Optional, Type

from .base import LLMProvider
from .cli_provider import CliLLM, CliProvider

LOGGER = logging.getLogger(__name__)

# Sentinel used by _resolve_kwargs to distinguish "caller passed no value"
# from "caller explicitly passed None" (which means "disable the timeout").
_UNSET = object()


class CliPresetProvider(LLMProvider):
    """An :class:`~bili.iris.providers.base.LLMProvider` that wraps a
    :class:`~bili.iris.providers.cli_presets.CliPreset`.

    Instances supply the preset's defaults for every parameter that the caller
    does not override, then delegate to
    :class:`~bili.iris.providers.cli_provider.CliProvider` for validation and
    :class:`~bili.iris.providers.cli_provider.CliLLM` construction.

    Do not instantiate this class directly.  Use
    :meth:`for_preset` to obtain a subclass bound to a specific preset, or
    call :func:`~bili.iris.providers.cli_presets.register_cli_preset` to
    register a preset and let the registry instantiate it on demand.
    """

    #: The :class:`~bili.iris.providers.cli_presets.CliPreset` this provider
    #: is bound to.  Set by :meth:`for_preset`; ``None`` in the base class.
    _preset: Optional[Any] = None  # CliPreset -- kept as Any to avoid circular import

    def _resolve_kwargs(self, overrides: dict) -> dict:
        """Merge caller-supplied *overrides* with preset defaults.

        Caller-supplied values win over the preset default.  ``None`` is a
        valid caller value for ``timeout_seconds`` (meaning "no timeout") and
        for ``cwd`` (meaning "inherit the calling process's cwd"), so the
        sentinel :data:`_UNSET` is used to distinguish "not provided" from
        "explicitly set to None".

        :param overrides: Mapping of parameter name to caller-supplied value,
            where the sentinel :data:`_UNSET` indicates "not provided by
            caller, use the preset default".
        """
        preset = self._preset
        # command is special: copy the list to avoid mutating the preset.
        cmd = overrides.get("command", _UNSET)
        resolved: dict = {
            "command": cmd if cmd is not _UNSET else list(preset.command),
        }
        for field in (
            "prompt_via",
            "message_format",
            "output_format",
            "json_path",
            "strip_ansi",
            "timeout_seconds",
            "cwd",
            "max_retries",
            "retry_backoff_seconds",
            "model",
            "reasoning_effort",
        ):
            val = overrides.get(field, _UNSET)
            resolved[field] = val if val is not _UNSET else getattr(preset, field)

        # model_flag_template / reasoning_effort_flag_template are list-valued
        # (like command): copy them when falling back to the preset default so
        # a caller mutating the returned CliLLM's list in place does not
        # corrupt the shared preset object.
        for list_field in ("model_flag_template", "reasoning_effort_flag_template"):
            val = overrides.get(list_field, _UNSET)
            if val is not _UNSET:
                resolved[list_field] = val
            else:
                preset_val = getattr(preset, list_field)
                resolved[list_field] = (
                    list(preset_val) if preset_val is not None else None
                )
        return resolved

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        command: Optional[List[str]] = _UNSET,
        prompt_via: Optional[str] = _UNSET,
        message_format: Optional[str] = _UNSET,
        output_format: Optional[str] = _UNSET,
        json_path: Optional[str] = _UNSET,
        strip_ansi: Optional[bool] = _UNSET,
        timeout_seconds: Optional[float] = _UNSET,
        cwd: Optional[str] = _UNSET,
        max_retries: Optional[int] = _UNSET,
        retry_backoff_seconds: Optional[float] = _UNSET,
        model: Optional[str] = _UNSET,
        reasoning_effort: Optional[str] = _UNSET,
        model_flag_template: Optional[List[str]] = _UNSET,
        reasoning_effort_flag_template: Optional[List[str]] = _UNSET,
        **extra: Any,
    ) -> CliLLM:
        """Create a :class:`~bili.iris.providers.cli_provider.CliLLM` using
        the bound preset's defaults, overridden by any supplied kwargs.

        :param command: Override the preset's command list.
        :param prompt_via: Override the preset's ``prompt_via``.
        :param message_format: Override the preset's ``message_format``.
        :param output_format: Override the preset's ``output_format``.
        :param json_path: Override the preset's ``json_path``.
        :param strip_ansi: Override the preset's ``strip_ansi``.
        :param timeout_seconds: Override the preset's ``timeout_seconds``.
            Pass ``None`` to disable the subprocess timeout entirely.
        :param cwd: Override the preset's ``cwd``.  Pass ``None`` to force
            the subprocess back to inheriting the calling process's current
            working directory even if the bound preset sets a fixed one.
        :param max_retries: Override the preset's ``max_retries``.  Pass
            ``0`` to disable in-process retry entirely.
        :param retry_backoff_seconds: Override the preset's
            ``retry_backoff_seconds``.
        :param model: Override the preset's ``model``.  Pins the CLI to a
            specific model instead of inheriting the CLI's own default.
        :param reasoning_effort: Override the preset's ``reasoning_effort``.
            Pins the CLI's reasoning depth; a value set with no
            corresponding flag template configured for this preset is a
            documented no-op (a warning is logged).
        :param model_flag_template: Override the preset's
            ``model_flag_template``.
        :param reasoning_effort_flag_template: Override the preset's
            ``reasoning_effort_flag_template``.
        :returns: A configured :class:`~bili.iris.providers.cli_provider.CliLLM`.
        :raises RuntimeError: If the provider was not created via
            :meth:`for_preset` (i.e. ``_preset`` is ``None``).
        """
        if self._preset is None:
            raise RuntimeError(
                "CliPresetProvider must be created via CliPresetProvider.for_preset(). "
                "Do not instantiate the base class directly."
            )
        kwargs = self._resolve_kwargs(
            {
                "command": command,
                "prompt_via": prompt_via,
                "message_format": message_format,
                "output_format": output_format,
                "json_path": json_path,
                "strip_ansi": strip_ansi,
                "timeout_seconds": timeout_seconds,
                "cwd": cwd,
                "max_retries": max_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "model_flag_template": model_flag_template,
                "reasoning_effort_flag_template": reasoning_effort_flag_template,
            }
        )
        LOGGER.info(
            "CliPresetProvider: loading preset command=%s",
            kwargs["command"][0] if kwargs["command"] else "<empty>",
        )
        return CliProvider().load(**kwargs, **extra)

    @classmethod
    def for_preset(cls, preset: Any) -> Type["CliPresetProvider"]:
        """Return a new :class:`CliPresetProvider` subclass bound to *preset*.

        Each call returns a **distinct class** so that the same preset class
        can be registered under multiple provider-type strings in the
        :class:`~bili.iris.providers.registry.ProviderRegistry` without
        collision.

        :param preset: A :class:`~bili.iris.providers.cli_presets.CliPreset`
            instance.
        :returns: A new :class:`CliPresetProvider` subclass.

        Example::

            preset = CliPreset(command=["my-llm"], prompt_via="arg")
            MyPresetProvider = CliPresetProvider.for_preset(preset)
            PROVIDER_REGISTRY.register("cli_my_llm", MyPresetProvider)
        """
        command_display = preset.command[0] if preset.command else "unknown"
        # The class name is informational only; it appears in repr() and logs.
        new_class = type(
            f"CliPresetProvider[{command_display}]",
            (cls,),
            {"_preset": preset},
        )
        return new_class
