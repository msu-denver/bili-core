"""Tests for CLI preset providers.

Covers :mod:`bili.iris.providers.cli_presets`,
:mod:`bili.iris.providers.preset_provider`, the three built-in preset
entries in :data:`~bili.iris.config.llm_config.LLM_MODELS`, and the
end-to-end ``load_model("<preset_type>")`` path via the provider registry.

No real subprocess is spawned; :mod:`subprocess` is mocked so tests run
in any environment without a CLI LLM tool installed.
"""

# pylint: disable=protected-access

import subprocess
from dataclasses import fields
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.config.llm_config import LLM_MODELS
from bili.iris.loaders.llm_loader import load_model
from bili.iris.providers.base import KNOWN_PROVIDER_TYPES
from bili.iris.providers.cli_presets import (
    CLAUDE_CODE_PRESET,
    CLI_PRESET_CATALOG,
    CODEX_PRESET,
    GEMINI_CLI_PRESET,
    CliPreset,
    register_cli_preset,
)
from bili.iris.providers.cli_provider import CliLLM, CliLLMError
from bili.iris.providers.preset_provider import CliPresetProvider
from bili.iris.providers.registry import PROVIDER_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed_proc(stdout: str = "", returncode: int = 0, stderr: str = ""):
    """Return a mock subprocess.CompletedProcess-like object."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def _run_preset_roundtrip(
    provider_type: str,
    prompt: str = "Hello",
    response: str = "World",
) -> str:
    """Load a preset via load_model, invoke it, and return the response text."""
    with patch(
        "subprocess.run",
        return_value=_make_completed_proc(stdout=response),
    ):
        llm = load_model(provider_type)
        result = llm.invoke([HumanMessage(content=prompt)])
    return result.content


# ---------------------------------------------------------------------------
# CliPreset dataclass
# ---------------------------------------------------------------------------


class TestCliPreset:
    """Unit tests for the CliPreset dataclass."""

    def test_default_values(self):
        """Default field values match the documented preset defaults."""
        preset = CliPreset(command=["my-tool"])
        assert preset.prompt_via == "arg"
        assert preset.message_format == "last"
        assert preset.output_format == "text"
        assert preset.json_path == "content"
        assert preset.strip_ansi is True
        assert preset.timeout_seconds == 1800.0
        assert preset.cwd is None
        assert preset.max_retries == 2
        assert preset.retry_backoff_seconds == 1.0
        assert preset.model is None
        assert preset.reasoning_effort is None
        assert preset.model_flag_template == ["--model", "{value}"]
        assert preset.reasoning_effort_flag_template is None

    def test_none_timeout_accepted(self):
        """timeout_seconds=None disables the per-call timeout."""
        preset = CliPreset(command=["my-tool"], timeout_seconds=None)
        assert preset.timeout_seconds is None

    def test_custom_values_stored(self):
        """Custom values supplied to the constructor are stored correctly."""
        preset = CliPreset(
            command=["llm", "--fast"],
            prompt_via="stdin",
            message_format="roles",
            output_format="json",
            json_path="result.text",
            strip_ansi=False,
            timeout_seconds=60.0,
            cwd="/opt/sandbox",
            max_retries=5,
            retry_backoff_seconds=2.5,
            model="fast-model",
            reasoning_effort="low",
            model_flag_template=["-m", "{value}"],
            reasoning_effort_flag_template=["--reasoning", "{value}"],
        )
        assert preset.command == ["llm", "--fast"]
        assert preset.prompt_via == "stdin"
        assert preset.message_format == "roles"
        assert preset.output_format == "json"
        assert preset.json_path == "result.text"
        assert preset.strip_ansi is False
        assert preset.timeout_seconds == 60.0
        assert preset.cwd == "/opt/sandbox"
        assert preset.max_retries == 5
        assert preset.retry_backoff_seconds == 2.5
        assert preset.model == "fast-model"
        assert preset.reasoning_effort == "low"
        assert preset.model_flag_template == ["-m", "{value}"]
        assert preset.reasoning_effort_flag_template == ["--reasoning", "{value}"]

    def test_none_model_flag_template_accepted(self):
        """model_flag_template=None disables model-flag injection for a preset."""
        preset = CliPreset(command=["my-tool"], model_flag_template=None)
        assert preset.model_flag_template is None

    def test_dataclass_has_expected_fields(self):
        """CliPreset exposes exactly the fields that CliProvider.load accepts."""
        field_names = {f.name for f in fields(CliPreset)}
        expected = {
            "command",
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
            "model_flag_template",
            "reasoning_effort_flag_template",
        }
        assert expected == field_names


# ---------------------------------------------------------------------------
# Built-in preset definitions
# ---------------------------------------------------------------------------


class TestBuiltinPresets:
    """Verify the three built-in preset definitions have correct defaults."""

    def test_claude_code_preset_command(self):
        """CLAUDE_CODE_PRESET uses 'claude -p' as the command."""
        assert CLAUDE_CODE_PRESET.command == ["claude", "-p"]

    def test_claude_code_preset_prompt_via(self):
        """CLAUDE_CODE_PRESET delivers the prompt as a positional argument."""
        assert CLAUDE_CODE_PRESET.prompt_via == "arg"

    def test_claude_code_preset_output_format(self):
        """CLAUDE_CODE_PRESET uses plain-text output."""
        assert CLAUDE_CODE_PRESET.output_format == "text"

    def test_claude_code_preset_strip_ansi(self):
        """CLAUDE_CODE_PRESET strips ANSI codes by default."""
        assert CLAUDE_CODE_PRESET.strip_ansi is True

    def test_codex_preset_command(self):
        """CODEX_PRESET uses 'codex exec' as the command."""
        assert CODEX_PRESET.command == ["codex", "exec"]

    def test_codex_preset_prompt_via(self):
        """CODEX_PRESET delivers the prompt as a positional argument."""
        assert CODEX_PRESET.prompt_via == "arg"

    def test_codex_preset_output_format(self):
        """CODEX_PRESET uses plain-text output."""
        assert CODEX_PRESET.output_format == "text"

    def test_codex_preset_timeout(self):
        """CODEX_PRESET uses the agentic-turn default timeout (1800 s)."""
        assert CODEX_PRESET.timeout_seconds == 1800.0

    def test_claude_code_preset_timeout(self):
        """CLAUDE_CODE_PRESET uses the agentic-turn default timeout (1800 s)."""
        assert CLAUDE_CODE_PRESET.timeout_seconds == 1800.0

    def test_gemini_cli_preset_timeout(self):
        """GEMINI_CLI_PRESET uses the agentic-turn default timeout (1800 s)."""
        assert GEMINI_CLI_PRESET.timeout_seconds == 1800.0

    def test_gemini_cli_preset_command(self):
        """GEMINI_CLI_PRESET uses 'gemini -p' as the command."""
        assert GEMINI_CLI_PRESET.command == ["gemini", "-p"]

    def test_gemini_cli_preset_prompt_via(self):
        """GEMINI_CLI_PRESET delivers the prompt as a positional argument."""
        assert GEMINI_CLI_PRESET.prompt_via == "arg"

    def test_gemini_cli_preset_output_format(self):
        """GEMINI_CLI_PRESET uses plain-text output."""
        assert GEMINI_CLI_PRESET.output_format == "text"


# ---------------------------------------------------------------------------
# Built-in preset model / reasoning-effort flag mapping
# ---------------------------------------------------------------------------


class TestBuiltinPresetModelAndReasoningEffort:
    """Verify each built-in preset's model / reasoning-effort flag templates.

    Per-CLI mapping (see cli_presets.py module docstring for the full
    rationale):

    - claude-code: ``--model <value>`` and ``--effort <value>``.
    - codex: ``--model <value>`` and ``-c model_reasoning_effort="<value>"``.
    - gemini-cli: ``--model <value>``; no CLI-settable reasoning-effort
      control (``reasoning_effort_flag_template`` is ``None``).
    """

    def test_claude_code_model_flag_template(self):
        """CLAUDE_CODE_PRESET applies model via '--model <value>'."""
        assert CLAUDE_CODE_PRESET.model_flag_template == ["--model", "{value}"]

    def test_claude_code_reasoning_effort_flag_template(self):
        """CLAUDE_CODE_PRESET applies reasoning_effort via '--effort <value>'."""
        assert CLAUDE_CODE_PRESET.reasoning_effort_flag_template == [
            "--effort",
            "{value}",
        ]

    def test_codex_model_flag_template(self):
        """CODEX_PRESET applies model via '--model <value>'."""
        assert CODEX_PRESET.model_flag_template == ["--model", "{value}"]

    def test_codex_reasoning_effort_flag_template(self):
        """CODEX_PRESET applies reasoning_effort via a '-c' config override."""
        assert CODEX_PRESET.reasoning_effort_flag_template == [
            "-c",
            'model_reasoning_effort="{value}"',
        ]

    def test_gemini_cli_model_flag_template(self):
        """GEMINI_CLI_PRESET applies model via '--model <value>'."""
        assert GEMINI_CLI_PRESET.model_flag_template == ["--model", "{value}"]

    def test_gemini_cli_reasoning_effort_not_cli_settable(self):
        """GEMINI_CLI_PRESET has no reasoning-effort flag template.

        The Gemini CLI exposes a thinking-budget control only via
        .gemini/settings.json (or interactive slash commands), not a
        headless-mode flag -- this preset documents that as an explicit
        ``None`` template.
        """
        assert GEMINI_CLI_PRESET.reasoning_effort_flag_template is None

    def test_all_builtin_presets_default_model_unset(self):
        """None of the built-in presets pin a model by default."""
        for preset in (CLAUDE_CODE_PRESET, CODEX_PRESET, GEMINI_CLI_PRESET):
            assert preset.model is None

    def test_all_builtin_presets_default_reasoning_effort_unset(self):
        """None of the built-in presets pin a reasoning effort by default."""
        for preset in (CLAUDE_CODE_PRESET, CODEX_PRESET, GEMINI_CLI_PRESET):
            assert preset.reasoning_effort is None


# ---------------------------------------------------------------------------
# CLI_PRESET_CATALOG
# ---------------------------------------------------------------------------


class TestCliPresetCatalog:
    """Verify the catalog contains exactly the expected preset keys."""

    def test_catalog_contains_all_presets(self):
        """All three built-in presets are present in CLI_PRESET_CATALOG."""
        assert "cli_claude_code" in CLI_PRESET_CATALOG
        assert "cli_codex" in CLI_PRESET_CATALOG
        assert "cli_gemini_cli" in CLI_PRESET_CATALOG

    def test_catalog_values_are_cli_preset_instances(self):
        """Every catalog entry is a CliPreset instance."""
        for key, value in CLI_PRESET_CATALOG.items():
            assert isinstance(
                value, CliPreset
            ), f"CLI_PRESET_CATALOG[{key!r}] is {type(value)}, expected CliPreset"

    def test_catalog_presets_match_module_constants(self):
        """Catalog entries are the same objects as the module-level constants."""
        assert CLI_PRESET_CATALOG["cli_claude_code"] is CLAUDE_CODE_PRESET
        assert CLI_PRESET_CATALOG["cli_codex"] is CODEX_PRESET
        assert CLI_PRESET_CATALOG["cli_gemini_cli"] is GEMINI_CLI_PRESET


# ---------------------------------------------------------------------------
# CliPresetProvider
# ---------------------------------------------------------------------------


class TestCliPresetProvider:
    """Unit tests for CliPresetProvider."""

    def test_for_preset_returns_subclass(self):
        """for_preset() returns a class that is a subclass of CliPresetProvider."""
        preset = CliPreset(command=["echo"])
        klass = CliPresetProvider.for_preset(preset)
        assert issubclass(klass, CliPresetProvider)

    def test_for_preset_binds_preset(self):
        """The returned subclass has _preset set to the supplied CliPreset."""
        preset = CliPreset(command=["echo"])
        klass = CliPresetProvider.for_preset(preset)
        assert klass._preset is preset

    def test_for_preset_creates_distinct_classes(self):
        """Two calls to for_preset return distinct classes even for the same preset."""
        preset = CliPreset(command=["echo"])
        klass_a = CliPresetProvider.for_preset(preset)
        klass_b = CliPresetProvider.for_preset(preset)
        assert klass_a is not klass_b

    def test_load_applies_preset_defaults(self):
        """load() uses the preset's command and config when no overrides are given."""
        preset = CliPreset(command=["myltool"], prompt_via="arg", output_format="text")
        klass = CliPresetProvider.for_preset(preset)
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="response"),
        ) as mock_run:
            llm = klass().load()
            llm.invoke([HumanMessage(content="hi")])
        cmd_used: List[str] = mock_run.call_args[0][0]
        assert cmd_used[0] == "myltool"
        assert cmd_used[-1] == "hi"  # prompt appended as arg

    def test_load_overrides_command(self):
        """load() accepts a caller-supplied command that overrides the preset."""
        preset = CliPreset(command=["preset-tool"])
        klass = CliPresetProvider.for_preset(preset)
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = klass().load(command=["override-tool"])
            llm.invoke([HumanMessage(content="test")])
        cmd_used: List[str] = mock_run.call_args[0][0]
        assert cmd_used[0] == "override-tool"

    def test_load_overrides_timeout(self):
        """load() accepts a caller-supplied timeout_seconds override."""
        preset = CliPreset(command=["tool"], timeout_seconds=60.0)
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(timeout_seconds=300.0)
        assert llm.timeout_seconds == 300.0

    def test_load_none_timeout_disables_timeout(self):
        """load() accepts timeout_seconds=None to disable the subprocess timeout."""
        preset = CliPreset(command=["tool"], timeout_seconds=600.0)
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(timeout_seconds=None)
        assert llm.timeout_seconds is None

    def test_load_default_cwd_matches_preset_default(self):
        """load() with no cwd override falls back to the preset's cwd default
        (None, i.e. inherit the calling process's cwd)."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.cwd is None

    def test_load_overrides_cwd(self):
        """load() accepts a caller-supplied cwd override."""
        preset = CliPreset(command=["tool"], cwd="/preset/default/dir")
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(cwd="/caller/override/dir")
        assert llm.cwd == "/caller/override/dir"

    def test_load_none_cwd_resets_to_inherited(self):
        """Explicitly passing cwd=None overrides a preset-fixed cwd back to
        inheriting the calling process's cwd."""
        preset = CliPreset(command=["tool"], cwd="/preset/default/dir")
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(cwd=None)
        assert llm.cwd is None

    def test_load_default_max_retries_matches_preset_default(self):
        """load() with no override falls back to the preset's max_retries."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.max_retries == 2

    def test_load_overrides_max_retries(self):
        """load() accepts a caller-supplied max_retries override."""
        preset = CliPreset(command=["tool"], max_retries=2)
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(max_retries=5)
        assert llm.max_retries == 5

    def test_load_overrides_retry_backoff_seconds(self):
        """load() accepts a caller-supplied retry_backoff_seconds override."""
        preset = CliPreset(command=["tool"], retry_backoff_seconds=1.0)
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(retry_backoff_seconds=3.0)
        assert llm.retry_backoff_seconds == 3.0

    def test_load_without_preset_raises(self):
        """load() raises RuntimeError if _preset is None (base class used directly)."""
        provider = CliPresetProvider()
        with pytest.raises(RuntimeError, match="for_preset"):
            provider.load(command=["echo"])

    def test_load_returns_cli_llm(self):
        """load() returns a CliLLM instance."""
        preset = CliPreset(command=["echo"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert isinstance(llm, CliLLM)

    def test_load_default_model_matches_preset_default(self):
        """load() with no override falls back to the preset's model (None)."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.model is None

    def test_load_overrides_model(self):
        """load() accepts a caller-supplied model override."""
        preset = CliPreset(command=["tool"], model="preset-default-model")
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(model="caller-model")
        assert llm.model == "caller-model"

    def test_load_overrides_reasoning_effort(self):
        """load() accepts a caller-supplied reasoning_effort override."""
        preset = CliPreset(
            command=["tool"], reasoning_effort_flag_template=["--effort", "{value}"]
        )
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(reasoning_effort="high")
        assert llm.reasoning_effort == "high"

    def test_load_default_model_flag_template_matches_preset(self):
        """load() with no override falls back to the preset's model_flag_template."""
        preset = CliPreset(command=["tool"], model_flag_template=["-m", "{value}"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.model_flag_template == ["-m", "{value}"]

    def test_load_default_model_flag_template_is_copied_not_aliased(self):
        """The preset's model_flag_template list is copied, not shared, so a
        caller mutating the returned CliLLM's list cannot corrupt the preset."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        llm.model_flag_template.append("mutated")
        assert preset.model_flag_template == ["--model", "{value}"]

    def test_load_overrides_model_flag_template(self):
        """load() accepts a caller-supplied model_flag_template override."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(model_flag_template=["--use-model", "{value}"])
        assert llm.model_flag_template == ["--use-model", "{value}"]

    def test_load_none_model_flag_template_disables_injection(self):
        """Passing model_flag_template=None disables model-flag injection
        even if the bound preset has one configured."""
        preset = CliPreset(command=["tool"], model_flag_template=["--model", "{value}"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(model_flag_template=None)
        assert llm.model_flag_template is None

    def test_load_default_reasoning_effort_flag_template_matches_preset(self):
        """load() with no override falls back to the preset's
        reasoning_effort_flag_template."""
        preset = CliPreset(
            command=["tool"], reasoning_effort_flag_template=["--effort", "{value}"]
        )
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.reasoning_effort_flag_template == ["--effort", "{value}"]

    def test_load_default_reasoning_effort_flag_template_is_none_by_default(self):
        """A preset with no reasoning_effort_flag_template configured yields
        None on the loaded CliLLM (documented no-op if reasoning_effort is
        also set)."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load()
        assert llm.reasoning_effort_flag_template is None

    def test_load_overrides_reasoning_effort_flag_template(self):
        """load() accepts a caller-supplied reasoning_effort_flag_template
        override."""
        preset = CliPreset(command=["tool"])
        klass = CliPresetProvider.for_preset(preset)
        llm = klass().load(reasoning_effort_flag_template=["--think", "{value}"])
        assert llm.reasoning_effort_flag_template == ["--think", "{value}"]


# ---------------------------------------------------------------------------
# Provider registry integration
# ---------------------------------------------------------------------------


class TestBuiltinPresetsRegistered:
    """Verify that importing builtin.py registers all three preset types."""

    def test_cli_claude_code_in_registry(self):
        """cli_claude_code is registered in the global provider registry."""
        assert "cli_claude_code" in PROVIDER_REGISTRY

    def test_cli_codex_in_registry(self):
        """cli_codex is registered in the global provider registry."""
        assert "cli_codex" in PROVIDER_REGISTRY

    def test_cli_gemini_cli_in_registry(self):
        """cli_gemini_cli is registered in the global provider registry."""
        assert "cli_gemini_cli" in PROVIDER_REGISTRY

    def test_preset_providers_are_preset_subclasses(self):
        """Each registered preset provider is a CliPresetProvider subclass."""
        for key in ("cli_claude_code", "cli_codex", "cli_gemini_cli"):
            klass = PROVIDER_REGISTRY.get(key)
            assert klass is not None, f"{key!r} not in registry"
            assert issubclass(
                klass, CliPresetProvider
            ), f"{key!r} maps to {klass}, not a CliPresetProvider subclass"

    def test_preset_types_in_known_provider_types(self):
        """All three preset type strings appear in KNOWN_PROVIDER_TYPES."""
        for key in ("cli_claude_code", "cli_codex", "cli_gemini_cli"):
            assert (
                key in KNOWN_PROVIDER_TYPES
            ), f"{key!r} missing from KNOWN_PROVIDER_TYPES"


# ---------------------------------------------------------------------------
# LLM_MODELS catalog entries
# ---------------------------------------------------------------------------


class TestLlmModelsCatalog:
    """Verify that LLM_MODELS contains accurate entries for each preset."""

    def test_cli_claude_code_entry_present(self):
        """LLM_MODELS contains a 'cli_claude_code' entry."""
        assert "cli_claude_code" in LLM_MODELS

    def test_cli_codex_entry_present(self):
        """LLM_MODELS contains a 'cli_codex' entry."""
        assert "cli_codex" in LLM_MODELS

    def test_cli_gemini_cli_entry_present(self):
        """LLM_MODELS contains a 'cli_gemini_cli' entry."""
        assert "cli_gemini_cli" in LLM_MODELS

    def test_preset_entries_have_required_catalog_fields(self):
        """Each preset entry has name, description, model_help, and models keys."""
        for key in ("cli_claude_code", "cli_codex", "cli_gemini_cli"):
            entry = LLM_MODELS[key]
            assert "name" in entry, f"{key} missing 'name'"
            assert "description" in entry, f"{key} missing 'description'"
            assert "model_help" in entry, f"{key} missing 'model_help'"
            assert "models" in entry, f"{key} missing 'models'"
            assert len(entry["models"]) >= 1, f"{key} has empty models list"

    def test_preset_entries_disable_unsupported_features(self):
        """CLI preset model entries mark temperature, seed, tools etc. as unsupported."""
        for key in ("cli_claude_code", "cli_codex", "cli_gemini_cli"):
            model = LLM_MODELS[key]["models"][0]
            assert (
                model.get("supports_temperature") is False
            ), f"{key} should have supports_temperature=False"
            assert (
                model.get("supports_tools") is False
            ), f"{key} should have supports_tools=False"


# ---------------------------------------------------------------------------
# End-to-end load_model round-trips
# ---------------------------------------------------------------------------


class TestLoadModelRoundtrip:
    """End-to-end tests: load_model(preset_type) returns a working CliLLM."""

    def test_claude_code_roundtrip(self):
        """load_model('cli_claude_code') returns a CliLLM that produces a response."""
        response = _run_preset_roundtrip("cli_claude_code", response="Claude response")
        assert response == "Claude response"

    def test_codex_roundtrip(self):
        """load_model('cli_codex') returns a CliLLM that produces a response."""
        response = _run_preset_roundtrip("cli_codex", response="Codex response")
        assert response == "Codex response"

    def test_gemini_cli_roundtrip(self):
        """load_model('cli_gemini_cli') returns a CliLLM that produces a response."""
        response = _run_preset_roundtrip("cli_gemini_cli", response="Gemini response")
        assert response == "Gemini response"

    def test_claude_code_command_is_claude(self):
        """The subprocess command for cli_claude_code starts with 'claude'."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_claude_code")
            llm.invoke([HumanMessage(content="test")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd[0] == "claude"

    def test_codex_command_is_codex_exec(self):
        """The subprocess command for cli_codex starts with 'codex exec'."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_codex")
            llm.invoke([HumanMessage(content="test")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"

    def test_gemini_cli_command_is_gemini(self):
        """The subprocess command for cli_gemini_cli starts with 'gemini'."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_gemini_cli")
            llm.invoke([HumanMessage(content="test")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd[0] == "gemini"

    def test_prompt_appended_as_arg_for_all_presets(self):
        """Each preset appends the prompt as the final positional argument."""
        for preset_type in ("cli_claude_code", "cli_codex", "cli_gemini_cli"):
            with patch(
                "subprocess.run",
                return_value=_make_completed_proc(stdout="ok"),
            ) as mock_run:
                llm = load_model(preset_type)
                llm.invoke([HumanMessage(content="my-prompt")])
            cmd: List[str] = mock_run.call_args[0][0]
            assert (
                cmd[-1] == "my-prompt"
            ), f"{preset_type}: expected prompt as last arg, got cmd={cmd}"

    def test_timeout_override_respected(self):
        """A timeout_seconds override passed to load_model is applied to CliLLM."""
        llm = load_model("cli_claude_code", timeout_seconds=300.0)
        assert llm.timeout_seconds == 300.0

    def test_none_timeout_override_disables_timeout(self):
        """Passing timeout_seconds=None to load_model disables the subprocess timeout."""
        llm = load_model("cli_claude_code", timeout_seconds=None)
        assert llm.timeout_seconds is None

    def test_default_preset_timeout_is_1800(self):
        """The default timeout for a preset CliLLM loaded via load_model is 1800 s."""
        llm = load_model("cli_claude_code")
        assert llm.timeout_seconds == 1800.0

    def test_default_preset_cwd_is_none(self):
        """The default cwd for a preset CliLLM loaded via load_model is None,
        i.e. it inherits the calling process's current working directory."""
        llm = load_model("cli_claude_code")
        assert llm.cwd is None

    def test_cwd_override_respected(self):
        """A cwd override passed to load_model is applied to the CliLLM and
        forwarded to the subprocess call."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_claude_code", cwd="/fixed/workspace")
            assert llm.cwd == "/fixed/workspace"
            llm.invoke([HumanMessage(content="hello")])
        assert mock_run.call_args.kwargs.get("cwd") == "/fixed/workspace"

    def test_subprocess_error_raises_cli_llm_error(self):
        """A non-zero subprocess exit for a preset raises CliLLMError."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(returncode=1, stderr="fail"),
        ):
            llm = load_model("cli_claude_code")
            with pytest.raises(CliLLMError, match="exited with code 1"):
                llm.invoke([HumanMessage(content="hello")])

    def test_subprocess_timeout_raises_cli_llm_error(self):
        """A subprocess timeout for a preset raises CliLLMError."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["claude", "-p"], timeout=120),
        ):
            llm = load_model("cli_claude_code")
            with pytest.raises(CliLLMError, match="timed out"):
                llm.invoke([HumanMessage(content="hello")])

    def test_claude_code_model_and_effort_reach_subprocess_command(self):
        """model + reasoning_effort overrides reach the spawned claude command."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model(
                "cli_claude_code", model="claude-sonnet-5", reasoning_effort="high"
            )
            llm.invoke([HumanMessage(content="hello")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd == [
            "claude",
            "-p",
            "--model",
            "claude-sonnet-5",
            "--effort",
            "high",
            "hello",
        ]

    def test_codex_model_and_effort_reach_subprocess_command(self):
        """model + reasoning_effort overrides reach the spawned codex command,
        with reasoning_effort applied via the '-c model_reasoning_effort=' config
        override."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_codex", model="gpt-5-codex", reasoning_effort="low")
            llm.invoke([HumanMessage(content="hello")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd == [
            "codex",
            "exec",
            "--model",
            "gpt-5-codex",
            "-c",
            'model_reasoning_effort="low"',
            "hello",
        ]

    def test_gemini_cli_model_reaches_subprocess_command(self):
        """model override reaches the spawned gemini command."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_gemini_cli", model="gemini-3-pro")
            llm.invoke([HumanMessage(content="hello")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd == ["gemini", "-p", "--model", "gemini-3-pro", "hello"]

    def test_gemini_cli_reasoning_effort_is_no_op(self):
        """reasoning_effort on the gemini preset is a documented no-op: no
        extra flag is added, and the CLI's own default is used."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="ok"),
        ) as mock_run:
            llm = load_model("cli_gemini_cli", reasoning_effort="high")
            llm.invoke([HumanMessage(content="hello")])
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd == ["gemini", "-p", "hello"]

    def test_unset_model_and_reasoning_effort_add_no_flags(self):
        """With neither model nor reasoning_effort set, no extra flags are
        added -- today's behaviour, unchanged."""
        for preset_type, expected_prefix in (
            ("cli_claude_code", ["claude", "-p"]),
            ("cli_codex", ["codex", "exec"]),
            ("cli_gemini_cli", ["gemini", "-p"]),
        ):
            with patch(
                "subprocess.run",
                return_value=_make_completed_proc(stdout="ok"),
            ) as mock_run:
                llm = load_model(preset_type)
                llm.invoke([HumanMessage(content="hello")])
            cmd: List[str] = mock_run.call_args[0][0]
            assert cmd == expected_prefix + [
                "hello"
            ], f"{preset_type}: expected no extra flags, got cmd={cmd}"


# ---------------------------------------------------------------------------
# register_cli_preset
# ---------------------------------------------------------------------------


class TestRegisterCliPreset:
    """Tests for the register_cli_preset() helper."""

    def test_register_and_invoke(self):
        """A custom preset registered via register_cli_preset is callable via load_model."""
        # Use a unique type string for this test to avoid registry collisions.
        provider_type = "cli_test_custom_preset_xyz"
        if provider_type in PROVIDER_REGISTRY:
            PROVIDER_REGISTRY.unregister(provider_type)

        preset = CliPreset(command=["custom-cli"], prompt_via="arg")
        register_cli_preset(provider_type, preset)

        with patch(
            "subprocess.run",
            return_value=_make_completed_proc(stdout="custom response"),
        ) as mock_run:
            llm = load_model(provider_type)
            result = llm.invoke([HumanMessage(content="hi")])

        assert result.content == "custom response"
        cmd: List[str] = mock_run.call_args[0][0]
        assert cmd[0] == "custom-cli"
        # Cleanup so other tests are not affected.
        PROVIDER_REGISTRY.unregister(provider_type)

    def test_register_duplicate_raises(self):
        """Registering the same type twice raises ValueError."""
        provider_type = "cli_test_dup_preset_xyz"
        if provider_type in PROVIDER_REGISTRY:
            PROVIDER_REGISTRY.unregister(provider_type)

        preset = CliPreset(command=["dup-cli"])
        register_cli_preset(provider_type, preset)

        with pytest.raises(ValueError, match="already registered"):
            register_cli_preset(provider_type, preset)

        # Cleanup.
        PROVIDER_REGISTRY.unregister(provider_type)
