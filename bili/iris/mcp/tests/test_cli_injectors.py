"""Tests for bili/iris/mcp/cli_injectors.py.

Tests per-CLI MCP configuration injectors and the injector registry:

- ClaudeCodeInjector: writes temp JSON file, injects --mcp-config/--strict-mcp-config
- CodexInjector: injects -c flags + unique env var (no temp files)
- GeminiCliInjector: writes .gemini/settings.json in temp dir, sets cwd sentinel
- get_injector: returns correct class or None for unknown CLIs
- register_cli_mcp_injector: registers a custom injector

All tests verify:
  - The Bearer token is embedded in the MCP config (not missing)
  - Temp resources are cleaned up by the cleanup() callable
  - No unauthenticated exposure (token always present in config)
"""

# pylint: disable=too-few-public-methods,protected-access,missing-function-docstring,missing-class-docstring,import-outside-toplevel

import json
import os
import re
import stat
import tempfile
from pathlib import Path

from bili.iris.mcp.cli_injectors import (
    _GEMINI_CWD_KEY,
    INJECTORS,
    ClaudeCodeInjector,
    CodexInjector,
    GeminiCliInjector,
    InjectionResult,
    McpCliInjector,
    get_injector,
    register_cli_mcp_injector,
)
from bili.iris.mcp.server import EphemeralMcpHandle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handle(
    server_url: str = "http://127.0.0.1:9001/mcp",
    token: str = "test-secret-token-xyz",
    server_name: str = "bili_tools_a1b2",
) -> EphemeralMcpHandle:
    return EphemeralMcpHandle(
        server_url=server_url, token=token, server_name=server_name
    )


BASE_CMD = ["claude", "-p"]


# ---------------------------------------------------------------------------
# ClaudeCodeInjector
# ---------------------------------------------------------------------------


class TestClaudeCodeInjector:
    def test_returns_injection_result(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        assert isinstance(result, InjectionResult)
        result.cleanup()

    def test_augmented_command_contains_mcp_config_flag(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        assert "--mcp-config" in result.augmented_command
        result.cleanup()

    def test_augmented_command_contains_strict_mcp_config(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        assert "--strict-mcp-config" in result.augmented_command
        result.cleanup()

    def test_base_command_preserved(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        assert result.augmented_command[:2] == BASE_CMD
        result.cleanup()

    def test_temp_file_exists_before_cleanup(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        tmp_path = result.augmented_command[idx]
        assert os.path.isfile(tmp_path)
        result.cleanup()

    def test_temp_file_deleted_by_cleanup(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        tmp_path = result.augmented_command[idx]
        result.cleanup()
        assert not os.path.exists(tmp_path)

    def test_cleanup_idempotent(self):
        """Calling cleanup() twice must not raise."""
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        result.cleanup()
        result.cleanup()  # Should not raise

    def test_config_file_contains_server_url(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle(server_url="http://127.0.0.1:12345/mcp")
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        config = json.loads(
            Path(result.augmented_command[idx]).read_text(encoding="utf-8")
        )
        urls = [v["url"] for v in config.get("mcpServers", {}).values()]
        assert "http://127.0.0.1:12345/mcp" in urls
        result.cleanup()

    def test_config_file_embeds_bearer_token(self):
        """Token MUST appear in the MCP config — no unauthenticated exposure."""
        injector = ClaudeCodeInjector()
        handle = _make_handle(token="super-secret-bearer")
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        config = json.loads(
            Path(result.augmented_command[idx]).read_text(encoding="utf-8")
        )
        auth_headers = [
            v.get("headers", {}).get("Authorization", "")
            for v in config.get("mcpServers", {}).values()
        ]
        assert any("super-secret-bearer" in h for h in auth_headers)
        result.cleanup()

    def test_config_file_uses_server_name_as_key(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle(server_name="bili_tools_c3d4")
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        config = json.loads(
            Path(result.augmented_command[idx]).read_text(encoding="utf-8")
        )
        assert "bili_tools_c3d4" in config.get("mcpServers", {})
        result.cleanup()

    def test_config_declares_http_transport_type(self):
        """The 'type':'http' field is required; without it Claude Code defaults to stdio."""
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        idx = result.augmented_command.index("--mcp-config") + 1
        config = json.loads(
            Path(result.augmented_command[idx]).read_text(encoding="utf-8")
        )
        types = [v.get("type") for v in config.get("mcpServers", {}).values()]
        assert "http" in types, "Claude Code config must declare type='http'"
        result.cleanup()

    def test_no_extra_env(self):
        injector = ClaudeCodeInjector()
        handle = _make_handle()
        result = injector.inject(command=BASE_CMD, handle=handle)
        assert result.extra_env == {}
        result.cleanup()


# ---------------------------------------------------------------------------
# CodexInjector
# ---------------------------------------------------------------------------


class TestCodexInjector:
    def test_returns_injection_result(self):
        injector = CodexInjector()
        handle = _make_handle()
        result = injector.inject(command=["codex", "exec"], handle=handle)
        assert isinstance(result, InjectionResult)

    def test_augmented_command_contains_config_flag(self):
        injector = CodexInjector()
        handle = _make_handle()
        result = injector.inject(command=["codex", "exec"], handle=handle)
        assert "-c" in result.augmented_command

    def test_config_flag_contains_server_url(self):
        injector = CodexInjector()
        handle = _make_handle(server_url="http://127.0.0.1:9999/mcp")
        result = injector.inject(command=["codex", "exec"], handle=handle)
        combined = " ".join(result.augmented_command)
        assert "http://127.0.0.1:9999/mcp" in combined

    def test_config_flag_contains_bearer_token_env_var(self):
        injector = CodexInjector()
        handle = _make_handle()
        result = injector.inject(command=["codex", "exec"], handle=handle)
        combined = " ".join(result.augmented_command)
        assert "bearer_token_env_var" in combined

    def test_extra_env_contains_token(self):
        """Token must be in extra_env so Codex can send it as Authorization header."""
        injector = CodexInjector()
        handle = _make_handle(token="codex-bearer-xyz")
        result = injector.inject(command=["codex", "exec"], handle=handle)
        # Find the env var name from the -c flag.
        env_var_name = None
        for part in result.augmented_command:
            if "bearer_token_env_var" in part:
                # Extract the var name from the TOML value e.g. BILI_MCP_TOKEN_ABCD
                m = re.search(r'bearer_token_env_var="(BILI_MCP_TOKEN_\w+)"', part)
                if m:
                    env_var_name = m.group(1)
                break
        assert (
            env_var_name is not None
        ), "bearer_token_env_var name not found in command"
        assert result.extra_env.get(env_var_name) == "codex-bearer-xyz"

    def test_env_var_name_unique_per_call(self):
        """Different calls must get different env var names (collision safety)."""
        injector = CodexInjector()
        handle = _make_handle()
        r1 = injector.inject(command=["codex", "exec"], handle=handle)
        r2 = injector.inject(command=["codex", "exec"], handle=handle)
        assert set(r1.extra_env.keys()) != set(r2.extra_env.keys())

    def test_no_temp_files_created(self):
        """Codex injector must not write any temp files."""
        injector = CodexInjector()
        handle = _make_handle()
        result = injector.inject(command=["codex", "exec"], handle=handle)
        assert result.cleanup is None

    def test_base_command_preserved(self):
        injector = CodexInjector()
        handle = _make_handle()
        result = injector.inject(command=["codex", "exec"], handle=handle)
        assert result.augmented_command[:2] == ["codex", "exec"]

    def test_server_name_in_config_flag(self):
        injector = CodexInjector()
        handle = _make_handle(server_name="bili_tools_e5f6")
        result = injector.inject(command=["codex", "exec"], handle=handle)
        combined = " ".join(result.augmented_command)
        assert "bili_tools_e5f6" in combined


# ---------------------------------------------------------------------------
# GeminiCliInjector
# ---------------------------------------------------------------------------


class TestGeminiCliInjector:
    def test_returns_injection_result(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        assert isinstance(result, InjectionResult)
        result.cleanup()

    def test_extra_env_contains_cwd_sentinel(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        assert _GEMINI_CWD_KEY in result.extra_env
        result.cleanup()

    def test_cwd_is_a_real_directory(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        cwd = result.extra_env[_GEMINI_CWD_KEY]
        assert os.path.isdir(cwd)
        result.cleanup()

    def test_gemini_settings_file_exists_before_cleanup(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        cwd = result.extra_env[_GEMINI_CWD_KEY]
        settings_path = Path(cwd) / ".gemini" / "settings.json"
        assert settings_path.is_file()
        result.cleanup()

    def test_gemini_settings_contains_server_url(self):
        injector = GeminiCliInjector()
        handle = _make_handle(server_url="http://127.0.0.1:54321/mcp")
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        cwd = result.extra_env[_GEMINI_CWD_KEY]
        settings = json.loads(
            (Path(cwd) / ".gemini" / "settings.json").read_text(encoding="utf-8")
        )
        urls = [v.get("httpUrl", "") for v in settings.get("mcpServers", {}).values()]
        assert "http://127.0.0.1:54321/mcp" in urls
        result.cleanup()

    def test_gemini_settings_embeds_bearer_token(self):
        """Token MUST appear in settings — no unauthenticated exposure."""
        injector = GeminiCliInjector()
        handle = _make_handle(token="gemini-bearer-secret")
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        cwd = result.extra_env[_GEMINI_CWD_KEY]
        settings = json.loads(
            (Path(cwd) / ".gemini" / "settings.json").read_text(encoding="utf-8")
        )
        auth_headers = [
            v.get("headers", {}).get("Authorization", "")
            for v in settings.get("mcpServers", {}).values()
        ]
        assert any("gemini-bearer-secret" in h for h in auth_headers)
        result.cleanup()

    def test_temp_dir_deleted_by_cleanup(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        cwd = result.extra_env[_GEMINI_CWD_KEY]
        result.cleanup()
        assert not os.path.exists(cwd)

    def test_cleanup_idempotent(self):
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        result.cleanup()
        result.cleanup()  # Must not raise

    def test_base_command_preserved(self):
        """Gemini injector must not add flags (it uses cwd, not command flags)."""
        injector = GeminiCliInjector()
        handle = _make_handle()
        result = injector.inject(command=["gemini", "-p"], handle=handle)
        assert result.augmented_command == ["gemini", "-p"]
        result.cleanup()


# ---------------------------------------------------------------------------
# Token-delivery confidentiality, derived over the whole registry
# ---------------------------------------------------------------------------

_PROBE_TOKEN = "probe-token-Nn3rQ1s7ZkVt-do-not-reuse"


def _token_bearing_files(root: Path) -> "list[Path]":
    """Return every regular file under *root* whose bytes contain the token.

    Derived rather than enumerated: whatever an injector wrote is found by
    reading what is on disk, so an injector added later is covered without
    anyone extending a list of filenames.
    """
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:  # pragma: no cover — unreadable temp artefact
            continue
        if _PROBE_TOKEN.encode() in data:
            found.append(path)
    return found


def _run_every_injector(tmp_root: Path, monkeypatch) -> "list[Path]":
    """Run every registered injector with tempdir redirected under *tmp_root*.

    :returns: every file any injector wrote that carries the token.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
    handle = _make_handle(token=_PROBE_TOKEN)
    for cli_name, injector in INJECTORS.items():
        injector.inject(command=[cli_name, "-p"], handle=handle)
    return _token_bearing_files(tmp_root)


class TestTokenDeliveryConfidentiality:
    """Every channel carrying the token must be same-user-confidential.

    The token is the only thing standing between a local caller and the
    agent's tools, so the channels that deliver it to the subprocess set the
    boundary.  Both properties below are asserted over the whole registry,
    not per injector, so a newly registered injector inherits them.
    """

    def test_every_token_bearing_file_is_owner_only(self, tmp_path, monkeypatch):
        """A file carrying the token must be 0600, never the process umask.

        A default umask yields 0644, which would publish the token to every
        user on the host.
        """
        written = _run_every_injector(tmp_path, monkeypatch)
        modes = {p: stat.S_IMODE(p.stat().st_mode) for p in written}
        assert modes, "no injector wrote a token-bearing file; the walk found nothing"
        assert all(m == 0o600 for m in modes.values()), {
            str(p): oct(m) for p, m in modes.items() if m != 0o600
        }

    def test_the_walk_detects_a_widened_file(self, tmp_path):
        """Control: the collector must flag a 0644 token file.

        Without this, a walk that silently found nothing would satisfy the
        test above forever.
        """
        decoy = tmp_path / "decoy.json"
        decoy.write_text(json.dumps({"Authorization": f"Bearer {_PROBE_TOKEN}"}))
        os.chmod(decoy, 0o644)
        found = _token_bearing_files(tmp_path)
        assert decoy in found
        assert stat.S_IMODE(decoy.stat().st_mode) != 0o600

    def test_no_injector_puts_the_token_in_argv(self, tmp_path, monkeypatch):
        """A process command line is world-readable, so the token must not be in it.

        The Codex injector passes the *name* of an environment variable; the
        value must reach the subprocess through the environment only.
        """
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        handle = _make_handle(token=_PROBE_TOKEN)
        for cli_name, injector in INJECTORS.items():
            result = injector.inject(command=[cli_name, "-p"], handle=handle)
            assert not any(
                _PROBE_TOKEN in arg for arg in result.augmented_command
            ), f"{cli_name} injector leaked the token into argv"


# ---------------------------------------------------------------------------
# get_injector / registry
# ---------------------------------------------------------------------------


class TestGetInjector:
    """Tests for get_injector lookup against the built-in INJECTORS registry."""

    def test_claude_resolves(self):
        """claude must map to ClaudeCodeInjector."""
        assert isinstance(get_injector("claude"), ClaudeCodeInjector)

    def test_codex_resolves(self):
        """codex must map to CodexInjector."""
        assert isinstance(get_injector("codex"), CodexInjector)

    def test_gemini_resolves(self):
        """gemini must map to GeminiCliInjector."""
        assert isinstance(get_injector("gemini"), GeminiCliInjector)

    def test_unknown_returns_none(self):
        """An unrecognised CLI name must return None (unknown CLIs cannot use MCP path)."""
        assert get_injector("unknown-llm-xyz-9999") is None

    def test_empty_string_returns_none(self):
        """An empty-string CLI name must return None (no registered injector)."""
        assert get_injector("") is None


class TestRegisterCliMcpInjector:
    """Tests for register_cli_mcp_injector and the INJECTORS registry."""

    def test_custom_injector_resolves_after_registration(self):
        """A custom injector registered under a new name must be returned by get_injector."""

        class MyInjector(McpCliInjector):
            """Stub injector for testing registry insertion."""

            def inject(self, command, handle):
                """Return a no-op InjectionResult."""
                return InjectionResult(augmented_command=command)

        register_cli_mcp_injector("my-custom-cli", MyInjector())
        result = get_injector("my-custom-cli")
        assert isinstance(result, MyInjector)
        # Cleanup: remove the custom entry so it doesn't affect other tests.
        del INJECTORS["my-custom-cli"]

    def test_registration_overwrites_existing(self):
        """Registering under an existing name replaces the prior injector."""

        class OverrideInjector(McpCliInjector):
            """Stub injector for testing registry overwrite."""

            def inject(self, command, handle):
                """Return a no-op InjectionResult."""
                return InjectionResult(augmented_command=command)

        original = INJECTORS.get("claude")
        try:
            register_cli_mcp_injector("claude", OverrideInjector())
            assert isinstance(get_injector("claude"), OverrideInjector)
        finally:
            # Restore original
            if original is not None:
                INJECTORS["claude"] = original
