"""Async MCP client session wrapper for bili-core.

Provides :class:`McpClient` -- a thin async context manager that wraps the
``mcp`` Python SDK's session lifecycle (connect -> initialize -> list_tools ->
call_tool -> close) for both ``stdio`` (subprocess) and ``http``
(SSE/streamable-HTTP) transports.

Callers are not expected to use :class:`McpClient` directly; instead use the
higher-level :func:`~bili.iris.mcp.loader.initialize_mcp_servers` and
:func:`~bili.iris.mcp.loader.register_mcp_tools` helpers.

Lazy imports
------------
All ``mcp`` SDK imports are deferred to the connect path so that this module
can be imported without the ``mcp`` package installed.  The base bili-core
install stays lean; the SDK is only required when MCP is actually used
(``pip install bili-core[mcp]``).

Config validation is pure Python and runs *before* any SDK import, so
``ValueError`` is raised on bad config even when ``mcp`` is not installed.

Supported transports
--------------------
``"stdio"``
    Spawns the configured executable as a subprocess.  The subprocess is the
    MCP server process; the client communicates with it over stdin/stdout.
    The subprocess always inherits the full caller environment as a baseline.
    ``auth="none"`` means *no auth credentials are forwarded* (e.g. no API
    keys in the env), not "no environment at all" -- the base env (PATH etc.)
    is always preserved so the subprocess can locate executables.
    ``env_passthrough`` *restricts* the forwarded env to that list; it does
    NOT strip the base env below PATH.

``"http"``
    Connects to a running HTTP MCP server via the SSE transport.  The server
    URL is set in the config entry's ``url`` field.

Design note -- lifecycle
------------------------
:class:`McpClient` is an *async* context manager.  Callers ``async with``-
enter it to get a live :class:`~mcp.ClientSession`, then call
``session.list_tools()`` / ``session.call_tool(...)`` as needed.  The context
manager handles subprocess teardown (stdio) or connection close (http) on
``__aexit__``.

Typical usage inside an ``async`` function::

    config = {
        "transport": "stdio",
        "command": "my-cli",
        "args": ["mcp", "serve"],
        "auth": "inherited",
        "env_passthrough": None,
        "startup_timeout": 10.0,
    }
    async with McpClient("my_server", config) as session:
        result = await session.list_tools()
        tools = result.tools
"""

import contextlib
import logging
import os
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Transports supported by McpClient.
_SUPPORTED_TRANSPORTS = frozenset({"stdio", "http"})


# ---------------------------------------------------------------------------
# Config validation (pure Python -- no SDK required)
# ---------------------------------------------------------------------------


def _validate_config(server_name: str, config: Dict[str, Any]) -> None:
    """Validate the server config dict.  Raises :exc:`ValueError` on bad input.

    This runs in pure Python with no ``mcp`` SDK import so that callers
    receive a descriptive :exc:`ValueError` even when the optional extra is
    not installed.

    :param server_name: Used in error messages.
    :param config: The server config dict.
    :raises ValueError: On any config error.
    """
    transport = config.get("transport", "stdio")
    if transport not in _SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"McpClient '{server_name}': unsupported transport {transport!r}. "
            "Use 'stdio' or 'http'."
        )
    if transport == "stdio" and not config.get("command"):
        raise ValueError(
            f"McpClient '{server_name}': 'command' is required for stdio transport"
        )
    if transport == "http" and not config.get("url"):
        raise ValueError(
            f"McpClient '{server_name}': 'url' is required for http transport"
        )


# ---------------------------------------------------------------------------
# McpClient
# ---------------------------------------------------------------------------


class McpClient:
    """Async context manager that manages a single MCP server session.

    Opens the transport (stdio subprocess or HTTP connection), performs the
    MCP ``initialize`` handshake, and exposes the live
    :class:`~mcp.ClientSession` to the caller.  Tears down the transport on
    exit.

    Config validation is performed at construction time in pure Python -- no
    ``mcp`` SDK import required.  :exc:`ValueError` for bad config is raised
    before any SDK call.  :exc:`ImportError` for a missing ``mcp`` package is
    only raised on the connect path (inside ``__aenter__``).

    :param server_name: Human-readable name used in log messages.
    :param config: Server config dict; see :mod:`bili.iris.mcp.config`.
    :raises ValueError: If the config is invalid (at construction time).
    :raises ImportError: If the ``mcp`` package is not installed (at connect time).
    """

    def __init__(self, server_name: str, config: Dict[str, Any]) -> None:
        self._name = server_name
        self._config = config
        self._exit_stack: Any = None  # contextlib.AsyncExitStack
        self._session: Any = None  # mcp.ClientSession

        # Validate config immediately -- pure Python, no SDK needed.
        _validate_config(server_name, config)

    # ------------------------------------------------------------------
    # Async context manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Any:
        """Open the transport, initialise the session, and return it.

        :raises ImportError: If the ``mcp`` package is not installed.
        """
        try:
            from mcp import ClientSession  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for MCP client support. "
                "Install it with: pip install bili-core[mcp]"
            ) from exc

        transport = self._config.get("transport", "stdio")
        self._exit_stack = contextlib.AsyncExitStack()

        LOGGER.debug("McpClient '%s': opening %s transport", self._name, transport)

        if transport == "stdio":
            read_stream, write_stream = await self._open_stdio()
        else:
            # transport == "http" (validated at construction; no other value possible)
            read_stream, write_stream = await self._open_http()

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        LOGGER.info("McpClient '%s': session initialized", self._name)
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the session and tear down the transport."""
        LOGGER.debug("McpClient '%s': closing session", self._name)
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None
        return False  # Do not suppress exceptions

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> Optional[Dict[str, str]]:
        """Build the environment dict for a stdio subprocess.

        The subprocess always receives at least the full current process
        environment as a baseline so it can locate executables (PATH etc.).

        - ``auth="inherited"`` (default) + no ``env_passthrough``: pass
          ``None`` to the SDK so the subprocess inherits ``os.environ`` in
          full (including OAuth session files, API key vars, etc.).
        - ``auth="none"``: means "forward no auth credentials" (e.g. strip
          API key vars), NOT "give the subprocess an empty environment".  The
          full ``os.environ`` is still the base; any credential filtering is
          the caller's responsibility via ``env_passthrough``.
        - ``env_passthrough`` (list of var names): forward ONLY those vars
          from ``os.environ``.  Always includes basic execution vars (PATH,
          HOME, USER, LANG) as a safety baseline so the subprocess can run.

        :returns: ``None`` to inherit the full env, or a filtered dict.
        """
        auth = self._config.get("auth", "inherited")
        passthrough = self._config.get("env_passthrough")

        if passthrough is not None:
            # Forward only the listed variables, but always include the
            # minimal execution baseline so the subprocess can run.
            baseline = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TERM"}
            keys_to_forward = set(passthrough) | baseline
            return {k: os.environ[k] for k in keys_to_forward if k in os.environ}

        # auth="inherited" or auth="none" with no explicit passthrough:
        # pass None so the SDK forwards the full os.environ.
        # "none" means no auth credentials, not no environment.
        if auth == "none":
            LOGGER.debug(
                "McpClient '%s': auth='none' with no env_passthrough -- "
                "subprocess inherits full env; use env_passthrough to "
                "restrict specific credential vars.",
                self._name,
            )
        return None

    async def _open_stdio(self):
        """Open a stdio transport and return ``(read_stream, write_stream)``."""
        try:
            from mcp import (  # pylint: disable=import-outside-toplevel
                StdioServerParameters,
            )
            from mcp.client.stdio import (  # pylint: disable=import-outside-toplevel
                stdio_client,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for stdio transport. "
                "Install it with: pip install bili-core[mcp]"
            ) from exc

        args: List[str] = self._config.get("args", [])
        env = self._build_env()

        params = StdioServerParameters(
            command=self._config["command"],  # validated at construction
            args=args,
            env=env,
        )

        LOGGER.debug(
            "McpClient '%s': spawning stdio server %s %s",
            self._name,
            self._config["command"],
            args,
        )

        return await self._exit_stack.enter_async_context(stdio_client(params))

    async def _open_http(self):
        """Open an HTTP/SSE transport and return ``(read_stream, write_stream)``."""
        try:
            from mcp.client.sse import (  # pylint: disable=import-outside-toplevel
                sse_client,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for HTTP transport. "
                "Install it with: pip install bili-core[mcp]"
            ) from exc

        timeout = self._config.get("startup_timeout", 10.0)
        LOGGER.debug(
            "McpClient '%s': connecting to %s", self._name, self._config["url"]
        )

        return await self._exit_stack.enter_async_context(
            sse_client(
                self._config["url"], timeout=timeout
            )  # validated at construction
        )
