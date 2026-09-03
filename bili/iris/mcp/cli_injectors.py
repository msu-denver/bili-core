"""Per-CLI MCP configuration injectors for the ephemeral MCP server.

Each injector knows how to configure one specific CLI LLM tool to connect to
an ephemeral MCP server described by an
:class:`~bili.iris.mcp.server.EphemeralMcpHandle`.  Injectors are keyed by
the CLI executable's basename (e.g. ``"claude"``, ``"codex"``, ``"gemini"``).

All injectors embed the per-call Bearer token in the MCP configuration so the
spawned subprocess can authenticate to the ephemeral server.

Token delivery is same-user-confidential, not per-process
---------------------------------------------------------
The token reaches the subprocess through a file or an environment variable,
so any process running as the **same user** can read it and authenticate;
other users cannot.  Files carrying the token are therefore created ``0600``
rather than at the process umask, and no injector puts the token in ``argv``
(the Codex injector passes the *name* of an environment variable, never its
value), because a process command line is world-readable.  Keep both
properties when adding an injector.  See the security-model section of
:mod:`bili.iris.mcp.server` for what this does and does not defend against.

Auth mechanisms by CLI
----------------------

``claude`` (Claude Code):
    Writes a temporary JSON file consumed by ``--mcp-config <path>
    --strict-mcp-config``.  The JSON carries a ``headers`` map with
    ``Authorization: Bearer <token>`` so every MCP request from Claude Code
    is authenticated.

``codex`` (OpenAI Codex CLI):
    Injects the server URL via ``-c mcp_servers.<name>.url="<url>"`` and sets
    a unique per-call environment variable (``BILI_MCP_TOKEN_<call_id>``)
    containing the token.  Instructs Codex to use that env var as the bearer
    token via ``-c mcp_servers.<name>.bearer_token_env_var="<var_name>"``.
    The extra env var is added to the subprocess environment by
    :func:`~bili.iris.mcp.server.build_mcp_node`.

``gemini`` (Google Gemini CLI):
    Writes a ``.gemini/settings.json`` into a temporary directory and runs the
    subprocess with ``cwd`` set to that directory so Gemini picks up the
    project-scoped settings file.  The JSON includes a ``headers`` map with
    the Bearer token.

Hard rule: unknown CLIs
-----------------------
If a CLI does not have a registered injector, it **cannot** be used on the
MCP path — :func:`get_injector` returns ``None`` and the caller must fall back
to the tool-less path.  An unauthenticated ephemeral server is never
acceptable; this is the enforced safe default.

Extension point
---------------
Third parties can register injectors for additional CLI tools via
:func:`register_cli_mcp_injector` at application startup.

Usage
-----
::

    from bili.iris.mcp.cli_injectors import get_injector

    injector = get_injector("claude")
    if injector is None:
        # Unknown CLI — fall back to tool-less path.
        ...

    with EphemeralMcpServer(tools) as handle:
        result = injector.inject(command=cmd, handle=handle)
        subprocess.run(result.augmented_command, env={**os.environ, **result.extra_env})
        result.cleanup()
"""

import json
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InjectionResult
# ---------------------------------------------------------------------------


@dataclass
class InjectionResult:
    """Return value from :meth:`McpCliInjector.inject`.

    :param augmented_command: The CLI command with MCP flags injected.
    :param extra_env: Additional environment variables to add to the subprocess
        environment (merged over ``os.environ`` by the caller).
    :param cleanup: A zero-argument callable that removes any temporary files or
        directories created by the injector.  ``None`` if no cleanup is needed.
    """

    augmented_command: List[str]
    extra_env: Dict[str, str] = field(default_factory=dict)
    cleanup: Optional[Callable[[], None]] = None


# ---------------------------------------------------------------------------
# Base class / protocol
# ---------------------------------------------------------------------------


class McpCliInjector:  # pylint: disable=too-few-public-methods
    """Base class for per-CLI MCP configuration injectors.

    Subclasses implement :meth:`inject` to produce an :class:`InjectionResult`
    that augments the CLI command and sets up any required temp resources so
    the spawned subprocess connects to the ephemeral MCP server with the
    correct Bearer token.
    """

    def inject(
        self,
        command: List[str],
        handle: Any,
    ) -> "InjectionResult":
        """Inject MCP configuration into *command* for a specific CLI tool.

        :param command: The base CLI command list (e.g. ``["claude", "-p"]``).
        :param handle: An :class:`~bili.iris.mcp.server.EphemeralMcpHandle`
            carrying ``server_url``, ``token``, and ``server_name``.
        :returns: An :class:`InjectionResult`.
        """
        raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# Claude Code injector
# ---------------------------------------------------------------------------


class ClaudeCodeInjector(McpCliInjector):  # pylint: disable=too-few-public-methods
    """MCP injector for the Claude Code CLI (``claude``).

    Writes a temporary JSON file in the format consumed by ``--mcp-config``
    and injects ``--mcp-config <path> --strict-mcp-config`` into the command.

    Config format (Streamable HTTP transport — ``"type": "http"`` is required;
    without it, Claude Code defaults to ``stdio`` transport and ignores ``url``)::

        {
          "mcpServers": {
            "<server_name>": {
              "type": "http",
              "url": "<server_url>",
              "headers": {
                "Authorization": "Bearer <token>"
              }
            }
          }
        }

    ``--strict-mcp-config`` ensures Claude Code ONLY connects to the
    ephemeral server for this call, ignoring any MCP servers the user has
    configured globally.

    The temporary JSON file is created with ``delete=False`` so it survives
    until after the subprocess exits.  The returned ``cleanup`` callable
    removes it.
    """

    def inject(self, command: List[str], handle: Any) -> InjectionResult:
        config_payload = {
            "mcpServers": {
                handle.server_name: {
                    "type": "http",
                    "url": handle.server_url,
                    "headers": {
                        "Authorization": f"Bearer {handle.token}",
                    },
                }
            }
        }

        # Write to a named temp file (not deleted on close; cleanup removes it).
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="bili_mcp_claude_", text=True
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(config_payload, fh)
        except (
            Exception
        ):  # pragma: no cover — json.dump failure is not realistically reachable
            os.unlink(tmp_path)
            raise

        LOGGER.debug(
            "ClaudeCodeInjector: wrote MCP config to %s (server=%s)",
            tmp_path,
            handle.server_name,
        )

        augmented = list(command) + [
            "--mcp-config",
            tmp_path,
            "--strict-mcp-config",
        ]

        def _cleanup() -> None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return InjectionResult(
            augmented_command=augmented,
            extra_env={},
            cleanup=_cleanup,
        )


# ---------------------------------------------------------------------------
# Codex injector
# ---------------------------------------------------------------------------


class CodexInjector(McpCliInjector):  # pylint: disable=too-few-public-methods
    """MCP injector for the OpenAI Codex CLI (``codex``).

    Codex has no per-call ``--mcp-config`` flag.  Instead it reads MCP server
    configuration from ``~/.codex/config.toml``, but its ``-c/--config``
    flag can override individual config values for a single invocation.

    This injector:

    1. Generates a unique per-call environment variable name
       (``BILI_MCP_TOKEN_<hex>``).
    2. Sets that variable to the Bearer token in the subprocess environment
       (via :attr:`InjectionResult.extra_env`).
    3. Injects two ``-c`` flags:
       ``-c mcp_servers.<name>.url="<server_url>"`` (Streamable HTTP endpoint)
       ``-c mcp_servers.<name>.bearer_token_env_var="<var_name>"``

    Codex reads the env var at startup and sends ``Authorization: Bearer
    <value>`` on every MCP request, which the server's auth middleware
    validates.

    No temp files are written; the unique env var name is cleaned up
    automatically when the subprocess exits.
    """

    def inject(self, command: List[str], handle: Any) -> InjectionResult:
        # Unique env var per call prevents collisions between concurrent calls.
        call_id = secrets.token_hex(4)
        env_var_name = f"BILI_MCP_TOKEN_{call_id.upper()}"

        server_key = handle.server_name

        LOGGER.debug(
            "CodexInjector: injecting MCP config via -c flags "
            "(server=%s, env_var=%s)",
            server_key,
            env_var_name,
        )

        augmented = list(command) + [
            "-c",
            f'mcp_servers.{server_key}.url="{handle.server_url}"',
            "-c",
            f'mcp_servers.{server_key}.bearer_token_env_var="{env_var_name}"',
        ]

        return InjectionResult(
            augmented_command=augmented,
            extra_env={env_var_name: handle.token},
            cleanup=None,  # No temp files.
        )


# ---------------------------------------------------------------------------
# Gemini CLI injector
# ---------------------------------------------------------------------------


class GeminiCliInjector(McpCliInjector):  # pylint: disable=too-few-public-methods
    """MCP injector for the Google Gemini CLI (``gemini``).

    Gemini CLI has no per-call ``--mcp-config`` flag.  It reads
    ``mcpServers`` from ``.gemini/settings.json`` in the current working
    directory (project scope) or from ``~/.gemini/settings.json`` (user
    scope).

    This injector:

    1. Creates a temporary directory.
    2. Writes ``<tmpdir>/.gemini/settings.json`` (mode ``0600``) using the
       ``httpUrl`` key, which maps to Gemini's Streamable HTTP transport
       (``httpUrl`` = MCP Streamable HTTP; ``url`` = deprecated SSE transport
       in the Gemini config schema).  The Bearer token is embedded in the
       ``headers`` map.
    3. Sets the subprocess ``cwd`` to *tmpdir* so Gemini picks up the
       project-scoped settings file automatically.

    The ``cleanup`` callable removes the temp directory and its contents.

    .. note::
        Running the subprocess in a different ``cwd`` is safe for the ``-p``
        (headless) invocation used by the Gemini CLI preset — the prompt is
        passed as an argument and does not depend on the working directory.
    """

    def inject(self, command: List[str], handle: Any) -> InjectionResult:
        settings_payload = {
            "mcpServers": {
                handle.server_name: {
                    "httpUrl": handle.server_url,
                    "headers": {
                        "Authorization": f"Bearer {handle.token}",
                    },
                }
            }
        }

        # Create temp dir with the required .gemini sub-directory.
        tmp_dir = tempfile.mkdtemp(prefix="bili_mcp_gemini_")
        gemini_dir = Path(tmp_dir) / ".gemini"
        gemini_dir.mkdir()
        settings_path = gemini_dir / "settings.json"
        # The file carries the Bearer token, so it is created 0600 rather than
        # at the process umask (which yields 0644 on a default configuration).
        # mkdtemp's own 0700 also keeps other users out, but that protection is
        # a property of the enclosing directory: a caller that relocates or
        # copies this file must not be able to widen the token's exposure by
        # doing so.  Opened O_CREAT|O_EXCL so the mode is applied at creation
        # and never exists briefly at the umask.
        with os.fdopen(
            os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(json.dumps(settings_payload))

        LOGGER.debug(
            "GeminiCliInjector: wrote settings to %s (server=%s, cwd=%s)",
            settings_path,
            handle.server_name,
            tmp_dir,
        )

        # Inject cwd via the command; subprocess.run doesn't take it as a
        # command flag.  The build_mcp_node caller must honour this.
        # We use a sentinel wrapper to carry the cwd alongside the result.
        augmented = list(command)

        def _cleanup() -> None:
            import shutil  # pylint: disable=import-outside-toplevel

            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except (
                OSError
            ):  # pragma: no cover — shutil.rmtree(ignore_errors=True) absorbs OSErrors
                pass

        # Stash cwd in extra_env under a private sentinel key so build_mcp_node
        # can extract it.  This is the cleanest way to pass subprocess kwargs
        # without changing InjectionResult's public interface.
        extra_env = {_GEMINI_CWD_KEY: tmp_dir}

        return InjectionResult(
            augmented_command=augmented,
            extra_env=extra_env,
            cleanup=_cleanup,
        )


#: Sentinel env-var key used by GeminiCliInjector to pass the subprocess cwd
#: to build_mcp_node.  This key is extracted and removed before the env is
#: forwarded to the subprocess.
_GEMINI_CWD_KEY = "__BILI_INTERNAL_GEMINI_CWD__"


# ---------------------------------------------------------------------------
# Injector registry
# ---------------------------------------------------------------------------

#: Built-in mapping from CLI basename to its injector instance.
#: Access via :func:`get_injector`.
INJECTORS: Dict[str, McpCliInjector] = {
    "claude": ClaudeCodeInjector(),
    "codex": CodexInjector(),
    "gemini": GeminiCliInjector(),
}


def get_injector(cli_name: str) -> Optional[McpCliInjector]:
    """Return the registered :class:`McpCliInjector` for *cli_name*, or ``None``.

    *cli_name* should be the basename of the CLI executable (e.g. ``"claude"``,
    ``"codex"``, ``"gemini"``).

    Returns ``None`` for unknown CLIs.  The caller must fall back to the
    tool-less path — never spawn an unauthenticated ephemeral server.

    :param cli_name: CLI executable basename.
    :returns: A :class:`McpCliInjector` instance, or ``None``.
    """
    return INJECTORS.get(cli_name)


def register_cli_mcp_injector(cli_name: str, injector: McpCliInjector) -> None:
    """Register a custom :class:`McpCliInjector` for *cli_name*.

    Call at application startup to support additional CLI tools on the MCP
    path.  If *cli_name* is already registered, the existing entry is replaced.

    :param cli_name: CLI executable basename (e.g. ``"my-custom-llm-cli"``).
    :param injector: The injector instance to register.
    """
    INJECTORS[cli_name] = injector
    LOGGER.debug(
        "Registered MCP CLI injector for '%s': %s", cli_name, type(injector).__name__
    )


__all__ = [
    "InjectionResult",
    "McpCliInjector",
    "ClaudeCodeInjector",
    "CodexInjector",
    "GeminiCliInjector",
    "INJECTORS",
    "get_injector",
    "register_cli_mcp_injector",
    "_GEMINI_CWD_KEY",
]
