"""bili-core MCP subsystem (``bili/iris/mcp/``).

Two complementary directions:

**MCP client** (``#205``): bili-core agents *consume* tools from external MCP
servers.  An MCP server's tools are adapted into LangChain
:class:`~langchain_core.tools.StructuredTool` objects and registered in the
agent's tool registry.

**MCP server** (``#311``): an agent's registered LangChain tools are *exposed*
as an ephemeral MCP server so that an MCP-capable CLI model (Claude Code,
Codex, Gemini CLI) can call them via its own native tool-calling interface.
The server is authenticated with a cryptographically-random per-call Bearer
token, lives only for the duration of the CLI call, and binds to localhost
only.

Public API — client side
------------------------
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

Public API — server side
------------------------
:class:`~bili.iris.mcp.server.EphemeralMcpServer`
    Synchronous context manager that serves a list of LangChain tools as an
    ephemeral, per-call authenticated MCP server on a dynamic localhost port.
:class:`~bili.iris.mcp.server.EphemeralMcpHandle`
    Handle returned by the context manager (SSE URL + Bearer token).
:func:`~bili.iris.mcp.server.build_mcp_node`
    Factory that builds a LangGraph node callable for the ``mcp``
    tool-strategy path.
:func:`~bili.iris.mcp.server.resolve_mcp_injector`
    Resolve the appropriate CLI injector from a
    :class:`~bili.iris.providers.cli_provider.CliLLM` model instance.
:func:`~bili.iris.mcp.cli_injectors.get_injector`
    Look up a per-CLI injector by executable basename.
:func:`~bili.iris.mcp.cli_injectors.register_cli_mcp_injector`
    Register a custom injector for an additional CLI tool.

:class:`~bili.iris.mcp.peer_identity.PeerAuthorization`
    Decides whether an inbound connection belongs to the spawned subprocess
    tree, which is what the Bearer token cannot establish on its own.

Optional dependency
-------------------
The ``mcp`` Python SDK, ``uvicorn``, and ``psutil`` are required at runtime
(lazy-imported so the base bili-core install stays lean).  Install with::

    pip install bili-core[mcp]

Quick start — client
--------------------
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

Quick start — server (ephemeral, for MCP-capable CLI models)
------------------------------------------------------------
::

    from bili.iris.mcp import EphemeralMcpServer, EphemeralMcpHandle
    from bili.iris.mcp.cli_injectors import get_injector

    injector = get_injector("claude")  # or "codex", "gemini"
    server = EphemeralMcpServer(tools)
    with server as handle:
        result = injector.inject(command=["claude", "-p"], handle=handle)
        proc = subprocess.Popen(
            result.augmented_command + [prompt],
            env={**os.environ, **result.extra_env},
        )
        # Required. The server refuses every request, valid token included,
        # until it knows which process it is serving; a bearer token cannot
        # tell the spawned subprocess apart from any other process of the
        # same user, all of which can read it.
        server.authorize_subprocess(proc.pid)
        proc.wait()

:func:`~bili.iris.mcp.server.build_mcp_node` does this for callers on the
``mcp`` tool-strategy path.
"""

from .cli_injectors import (
    ClaudeCodeInjector,
    CodexInjector,
    GeminiCliInjector,
    InjectionResult,
    McpCliInjector,
    get_injector,
    register_cli_mcp_injector,
)
from .loader import (
    McpLifecycle,
    McpServerSession,
    initialize_mcp_servers,
    initialize_mcp_servers_sync,
    register_mcp_tools,
)
from .peer_identity import PeerAuthorization, ProcessIdentity
from .server import (
    EphemeralMcpHandle,
    EphemeralMcpServer,
    build_mcp_node,
    resolve_mcp_injector,
)

__all__ = [
    # Client side
    "initialize_mcp_servers",
    "initialize_mcp_servers_sync",
    "register_mcp_tools",
    "McpLifecycle",
    "McpServerSession",
    # Server side
    "EphemeralMcpHandle",
    "EphemeralMcpServer",
    "PeerAuthorization",
    "ProcessIdentity",
    "build_mcp_node",
    "resolve_mcp_injector",
    # Injectors
    "InjectionResult",
    "McpCliInjector",
    "ClaudeCodeInjector",
    "CodexInjector",
    "GeminiCliInjector",
    "get_injector",
    "register_cli_mcp_injector",
]
