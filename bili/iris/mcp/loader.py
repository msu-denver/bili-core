"""MCP server initialization and tool registration for bili-core.

This module provides the two top-level functions that consumers call to add
MCP tools to a bili-core agent session:

:func:`initialize_mcp_servers`
    Connects to a set of MCP servers (as declared in
    :data:`~bili.iris.mcp.config.MCP_SERVERS`), performs the ``initialize``
    handshake on each, and returns a mapping of
    ``server_name → McpServerSession``.  Each session holds the live
    :class:`~mcp.ClientSession` and the list of tools the server advertises.

:func:`register_mcp_tools`
    Takes the sessions returned by :func:`initialize_mcp_servers`, adapts
    every tool to a LangChain :class:`~langchain_core.tools.StructuredTool`,
    inserts them into :data:`~bili.iris.loaders.tools_loader.TOOL_REGISTRY`
    under namespaced keys (``<server>__<tool>``), and returns a
    :class:`McpLifecycle` context manager that the caller holds open for the
    duration of the agent session.

:class:`McpLifecycle`
    An async context manager whose ``__aenter__`` returns the registered tool
    names and whose ``__aexit__`` tears down all open server sessions.  Callers
    that do not use ``async with`` can call :meth:`McpLifecycle.close` directly.

Typical usage (async session)
------------------------------
::

    import asyncio
    from bili.iris.mcp.loader import initialize_mcp_servers, register_mcp_tools
    from bili.iris.mcp.config import MCP_SERVERS

    async def run_agent():
        servers = await initialize_mcp_servers(
            active_servers=["my_server"],
            server_configs=MCP_SERVERS,
        )
        async with register_mcp_tools(servers) as tool_names:
            # tool_names: ["my_server__tool_a", "my_server__tool_b", ...]
            # Those keys are now in TOOL_REGISTRY — pass them to initialize_tools()
            ...

Integration with TOOL_REGISTRY
--------------------------------
:func:`register_mcp_tools` is an *opt-in extension* of
:data:`~bili.iris.loaders.tools_loader.TOOL_REGISTRY`.  It does NOT modify
the existing ``initialize_tools()`` signature, ``AgentSpec``, or
``MASExecutor`` — it simply inserts additional keys into the registry dict
before the agent's tools are resolved.  An agent that declares
``tools: ["my_server__tool_a"]`` in its ``AgentSpec`` will pick it up
automatically via the existing tools loader path.

Backward compatibility
-----------------------
All changes are additive.  Sessions that do not call ``initialize_mcp_servers``
or ``register_mcp_tools`` are unaffected.  The ``mcp`` optional dependency is
lazy-imported so the base install remains lean.
"""

import contextlib
import logging
from typing import Any, Dict, List, Optional

from .client import McpClient  # noqa: F401 -- imported here so tests can patch it

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# McpServerSession — holds a live session + its advertised tools
# ---------------------------------------------------------------------------


class McpServerSession:
    """Container for a live MCP server session and the tools it advertises.

    Created by :func:`initialize_mcp_servers`; consumed by
    :func:`register_mcp_tools`.

    :param server_name: The server's config key.
    :param session: The live ``mcp.ClientSession`` (an async context manager).
    :param mcp_tools: The list of ``mcp.types.Tool`` objects discovered from
        ``session.list_tools()``.
    :param client: The :class:`~bili.iris.mcp.client.McpClient` context
        manager that owns the transport (needed for teardown).
    """

    def __init__(
        self,
        server_name: str,
        session: Any,
        mcp_tools: List[Any],
        client: Any,
    ) -> None:
        self.server_name = server_name
        self.session = session
        self.mcp_tools = mcp_tools
        self._client = client  # McpClient context manager instance

    @property
    def tool_count(self) -> int:
        """The number of tools advertised by this server."""
        return len(self.mcp_tools)

    async def close(self) -> None:
        """Tear down the server session and its transport."""
        LOGGER.debug("McpServerSession '%s': closing", self.server_name)
        if hasattr(self._client, "aclose"):
            # AsyncExitStack path (production code)
            await self._client.aclose()
        else:
            # Legacy / test path: _client is a raw async context manager
            await self._client.__aexit__(
                None, None, None
            )  # pylint: disable=unnecessary-dunder-call


# ---------------------------------------------------------------------------
# McpLifecycle — context manager for registered MCP tools
# ---------------------------------------------------------------------------


class McpLifecycle:
    """Async context manager that owns open MCP server sessions.

    Returned by :func:`register_mcp_tools`.  The caller holds this open for
    the duration of the agent session and closes it when done to free the
    server subprocess / HTTP connection.

    ``async with`` usage::

        async with register_mcp_tools(servers) as tool_names:
            # tool_names is a list of registered TOOL_REGISTRY keys
            # Sessions are live here
        # Sessions are torn down here; TOOL_REGISTRY keys are removed

    Manual usage (for sync callers)::

        lifecycle = register_mcp_tools(servers)
        tool_names = await lifecycle.aopen()
        try:
            ...
        finally:
            await lifecycle.close()
    """

    def __init__(
        self,
        server_sessions: List[McpServerSession],
        registered_tool_names: List[str],
    ) -> None:
        self._sessions = server_sessions
        self._tool_names = registered_tool_names

    async def __aenter__(self) -> List[str]:
        """Return the list of registered tool names."""
        return self._tool_names

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Tear down all sessions and remove tools from TOOL_REGISTRY."""
        await self.close()
        return False  # Do not suppress exceptions

    async def aopen(self) -> List[str]:
        """Open the lifecycle and return the registered tool names.

        Use this when not using ``async with`` syntax.
        """
        return self._tool_names

    async def close(self) -> None:
        """Close all open MCP server sessions and remove their tools from the registry."""
        from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
            TOOL_REGISTRY,
        )

        for name in self._tool_names:
            if name in TOOL_REGISTRY:
                del TOOL_REGISTRY[name]
                LOGGER.debug("McpLifecycle: removed TOOL_REGISTRY key '%s'", name)

        for server_session in self._sessions:
            try:
                await server_session.close()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    "Error closing MCP server session '%s': %s",
                    server_session.server_name,
                    exc,
                )

    @property
    def tool_names(self) -> List[str]:
        """The list of namespaced tool keys registered in TOOL_REGISTRY."""
        return list(self._tool_names)


# ---------------------------------------------------------------------------
# initialize_mcp_servers
# ---------------------------------------------------------------------------


async def initialize_mcp_servers(
    active_servers: Optional[List[str]] = None,
    server_configs: Optional[Dict[str, Any]] = None,
) -> List[McpServerSession]:
    """Connect to MCP servers and return their live sessions.

    Filters to enabled servers in *active_servers* (or all enabled servers if
    *active_servers* is ``None``), opens each transport, performs the MCP
    ``initialize`` handshake, and calls ``list_tools`` to discover available
    tools.

    :param active_servers: Optional list of server names to activate.  If
        ``None``, all enabled servers in *server_configs* are started.
    :param server_configs: Server config dict in the shape of
        :data:`~bili.iris.mcp.config.MCP_SERVERS`.  Defaults to the built-in
        config when ``None``.
    :returns: A list of :class:`McpServerSession` objects, one per connected
        server.
    :raises ImportError: If the ``mcp`` package is not installed.
    """
    if server_configs is None:
        from bili.iris.mcp.config import (  # pylint: disable=import-outside-toplevel
            MCP_SERVERS,
        )

        server_configs = MCP_SERVERS

    sessions: List[McpServerSession] = []

    for server_name, config in server_configs.items():
        if not config.get("enabled", False):
            LOGGER.debug("MCP server '%s' is disabled; skipping", server_name)
            continue
        if active_servers is not None and server_name not in active_servers:
            LOGGER.debug(
                "MCP server '%s' not in active_servers list; skipping", server_name
            )
            continue

        LOGGER.info("Initializing MCP server: '%s'", server_name)
        client = McpClient(server_name, config)
        exit_stack = contextlib.AsyncExitStack()

        try:
            session = await exit_stack.enter_async_context(client)
            result = await session.list_tools()
            mcp_tools = result.tools
            LOGGER.info(
                "MCP server '%s': discovered %d tool(s): %s",
                server_name,
                len(mcp_tools),
                [t.name for t in mcp_tools],
            )
            # Transfer ownership of the exit_stack to the session container
            # so that McpServerSession.close() tears it down.
            sessions.append(
                McpServerSession(
                    server_name=server_name,
                    session=session,
                    mcp_tools=mcp_tools,
                    client=exit_stack,
                )
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed to initialize MCP server '%s': %s", server_name, exc)
            # Attempt clean teardown via the exit stack before re-raising
            try:
                await exit_stack.aclose()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise

    return sessions


# ---------------------------------------------------------------------------
# register_mcp_tools
# ---------------------------------------------------------------------------


def register_mcp_tools(
    server_sessions: List[McpServerSession],
    active_tool_names: Optional[List[str]] = None,
) -> McpLifecycle:
    """Adapt MCP tools to LangChain tools and register them in TOOL_REGISTRY.

    For each server session, converts every discovered ``mcp.types.Tool`` to a
    LangChain ``StructuredTool`` (namespaced ``<server>__<tool>``) and inserts
    it into :data:`~bili.iris.loaders.tools_loader.TOOL_REGISTRY`.

    Existing TOOL_REGISTRY entries with the same key are overwritten with a
    warning (avoids silently shadowing a native tool).

    :param server_sessions: Sessions returned by :func:`initialize_mcp_servers`.
    :param active_tool_names: Optional allowlist of namespaced tool names to
        register.  If ``None``, all tools from all sessions are registered.
    :returns: A :class:`McpLifecycle` context manager.  Call ``async with``
        on it (or ``await lifecycle.aopen()``) to obtain the tool name list,
        and ensure ``lifecycle.close()`` is called at session end.
    """
    from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
        TOOL_REGISTRY,
    )

    from .adapters.tool_adapter import (  # pylint: disable=import-outside-toplevel
        extract_text_from_result,
        mcp_tools_to_langchain,
    )

    registered_names: List[str] = []

    for server_session in server_sessions:
        sname = server_session.server_name
        live_session = server_session.session

        # Build a closure that captures the live session for this server.
        # The closure is what the StructuredTool calls on each invocation.
        def _make_call_tool_fn(sess, s_name):
            async def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
                LOGGER.debug(
                    "MCP call: server='%s' tool='%s' args=%s",
                    s_name,
                    tool_name,
                    list(arguments.keys()),
                )
                result = await sess.call_tool(tool_name, arguments=arguments)
                return extract_text_from_result(result)

            return _call_tool

        call_tool_fn = _make_call_tool_fn(live_session, sname)

        lc_tools = mcp_tools_to_langchain(
            server_name=sname,
            mcp_tools=server_session.mcp_tools,
            call_tool_fn=call_tool_fn,
        )

        for lc_tool in lc_tools:
            namespaced_name = lc_tool.name
            if (
                active_tool_names is not None
                and namespaced_name not in active_tool_names
            ):
                LOGGER.debug(
                    "MCP tool '%s' not in active_tool_names; skipping", namespaced_name
                )
                continue

            if namespaced_name in TOOL_REGISTRY:
                LOGGER.warning(
                    "MCP tool '%s' overwrites an existing TOOL_REGISTRY entry",
                    namespaced_name,
                )

            # Insert into TOOL_REGISTRY using the same lambda signature as
            # native tools: (name, prompt, params) -> LangChain tool.
            # The MCP StructuredTool ignores the prompt and params since its
            # behavior is fully determined by the server.
            captured_tool = lc_tool  # avoid late-binding capture in lambda
            TOOL_REGISTRY[namespaced_name] = (
                lambda _n, _p, _params, _t=captured_tool: _t
            )
            registered_names.append(namespaced_name)
            LOGGER.info("TOOL_REGISTRY: registered MCP tool '%s'", namespaced_name)

    return McpLifecycle(
        server_sessions=server_sessions,
        registered_tool_names=registered_names,
    )


# ---------------------------------------------------------------------------
# Convenience: sync wrapper for initialize_mcp_servers
# ---------------------------------------------------------------------------


def initialize_mcp_servers_sync(
    active_servers: Optional[List[str]] = None,
    server_configs: Optional[Dict[str, Any]] = None,
) -> List[McpServerSession]:
    """Synchronous wrapper around :func:`initialize_mcp_servers`.

    Provided for callers that cannot use ``await`` (e.g. legacy sync code,
    unit tests).  Uses the same async-bridge strategy as the tool adapters.

    :param active_servers: Passed through to :func:`initialize_mcp_servers`.
    :param server_configs: Passed through to :func:`initialize_mcp_servers`.
    :returns: List of :class:`McpServerSession` objects.
    """
    from .adapters.tool_adapter import (  # pylint: disable=import-outside-toplevel
        _run_async_sync,
    )

    return _run_async_sync(
        initialize_mcp_servers(
            active_servers=active_servers,
            server_configs=server_configs,
        )
    )
