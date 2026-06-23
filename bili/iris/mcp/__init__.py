"""bili-core MCP client subsystem (``bili/iris/mcp/``).

Lets bili-core agents consume tools from MCP servers.

Public API
----------
:func:`~bili.iris.mcp.loader.initialize_mcp_servers`
    Connect to MCP servers and return live sessions.
:func:`~bili.iris.mcp.loader.register_mcp_tools`
    Adapt MCP tools to LangChain tools and register them in ``TOOL_REGISTRY``.
:class:`~bili.iris.mcp.loader.McpLifecycle`
    Async context manager for session lifecycle management.
:class:`~bili.iris.mcp.loader.McpServerSession`
    Container for a live server session and its advertised tools.
:class:`~bili.iris.mcp.client.McpClient`
    Low-level async context manager for a single MCP server session.

Optional dependency
-------------------
The ``mcp`` Python SDK is required at runtime (lazy-imported so the base
bili-core install stays lean).  Install it with::

    pip install bili-core[mcp]

Quick start
-----------
::

    import asyncio
    from bili.iris.mcp import initialize_mcp_servers, register_mcp_tools
    from bili.iris.mcp.config import MCP_SERVERS

    async def run():
        servers = await initialize_mcp_servers(
            active_servers=["my_server"],
            server_configs=MCP_SERVERS,
        )
        async with register_mcp_tools(servers) as tool_names:
            # tool_names: ["my_server__tool_a", ...]
            ...

    asyncio.run(run())
"""

from .loader import (
    McpLifecycle,
    McpServerSession,
    initialize_mcp_servers,
    initialize_mcp_servers_sync,
    register_mcp_tools,
)

__all__ = [
    "initialize_mcp_servers",
    "initialize_mcp_servers_sync",
    "register_mcp_tools",
    "McpLifecycle",
    "McpServerSession",
]
