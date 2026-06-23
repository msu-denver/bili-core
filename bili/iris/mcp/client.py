"""Async MCP client session wrapper for bili-core.

Provides :class:`McpClient` — a thin async context manager that wraps the
``mcp`` Python SDK's session lifecycle (connect → initialize → list_tools →
call_tool → close) for both ``stdio`` (subprocess) and ``http``
(SSE/streamable-HTTP) transports.

Callers are not expected to use :class:`McpClient` directly; instead use the
higher-level :func:`~bili.iris.mcp.loader.initialize_mcp_servers` and
:func:`~bili.iris.mcp.loader.register_mcp_tools` helpers.

Lazy imports
------------
All ``mcp`` SDK imports are deferred to method bodies so that this module can
be imported without the ``mcp`` package installed.  The base bili-core install
stays lean; the SDK is only required when MCP is actually used
(``pip install bili-core[mcp]``).

Supported transports
--------------------
``"stdio"``
    Spawns the configured executable as a subprocess.  The subprocess is the
    MCP server process; the client communicates with it over stdin/stdout.
    Credentials are inherited from ``os.environ`` (``auth="inherited"``) or
    restricted to a specified subset (``env_passthrough``).

``"http"``
    Connects to a running HTTP MCP server via the SSE transport.  The server
    URL is set in the config entry's ``url`` field.

Design note — lifecycle
-----------------------
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


# ---------------------------------------------------------------------------
# McpClient
# ---------------------------------------------------------------------------


class McpClient:
    """Async context manager that manages a single MCP server session.

    Opens the transport (stdio subprocess or HTTP connection), performs the
    MCP ``initialize`` handshake, and exposes the live
    :class:`~mcp.ClientSession` to the caller.  Tears down the transport on
    exit.

    :param server_name: Human-readable name used in log messages.
    :param config: Server config dict; see :mod:`bili.iris.mcp.config`.
    :raises ImportError: If the ``mcp`` package is not installed.
    :raises ValueError: If the transport is not ``"stdio"`` or ``"http"``.
    """

    def __init__(self, server_name: str, config: Dict[str, Any]) -> None:
        self._name = server_name
        self._config = config
        self._exit_stack: Any = None  # contextlib.AsyncExitStack
        self._session: Any = None  # mcp.ClientSession

    # ------------------------------------------------------------------
    # Async context manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Any:
        """Open the transport, initialise the session, and return it."""
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
        elif transport == "http":
            read_stream, write_stream = await self._open_http()
        else:
            raise ValueError(
                f"McpClient '{self._name}': unsupported transport {transport!r}. "
                "Use 'stdio' or 'http'."
            )

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

    async def _build_env(self) -> Optional[Dict[str, str]]:
        """Build the environment dict for a stdio subprocess.

        Returns ``None`` (inherit everything) when ``auth="inherited"`` and
        ``env_passthrough`` is unset, so that the subprocess sees the full
        caller environment (including OAuth session files, API key vars, etc.).
        """
        auth = self._config.get("auth", "inherited")
        passthrough = self._config.get("env_passthrough")

        if auth == "none":
            return {}  # Empty env: subprocess sees nothing

        if passthrough is not None:
            # Forward only the listed variables
            return {k: os.environ[k] for k in passthrough if k in os.environ}

        # auth == "inherited" with no explicit passthrough: let the subprocess
        # inherit the full os.environ (pass None to StdioServerParameters so
        # the SDK uses get_default_environment(), which mirrors os.environ).
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

        command = self._config.get("command")
        if not command:
            raise ValueError(
                f"McpClient '{self._name}': 'command' is required for stdio transport"
            )

        args: List[str] = self._config.get("args", [])
        env = await self._build_env()

        params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        LOGGER.debug(
            "McpClient '%s': spawning stdio server %s %s",
            self._name,
            command,
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

        url = self._config.get("url")
        if not url:
            raise ValueError(
                f"McpClient '{self._name}': 'url' is required for http transport"
            )

        timeout = self._config.get("startup_timeout", 10.0)
        LOGGER.debug("McpClient '%s': connecting to %s", self._name, url)

        return await self._exit_stack.enter_async_context(
            sse_client(url, timeout=timeout)
        )
