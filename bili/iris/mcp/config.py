"""MCP server configuration for bili-core.

Declares which MCP servers are available for agents to consume tools from.
The shape mirrors :mod:`bili.iris.config.tool_config` so that operators can
configure MCP servers the same way they configure regular tools.

Each entry in :data:`MCP_SERVERS` describes one MCP server:

- ``transport``: ``"stdio"`` (spawn a subprocess) or ``"http"`` (connect to a
  URL).
- ``command`` / ``args``: for ``stdio`` — the executable and its arguments.
- ``url``: for ``http`` — the server base URL.
- ``auth``: ``"inherited"`` (subprocess inherits ``os.environ``) or ``"none"``.
- ``enabled``: ``True`` / ``False``.  Disabled servers are silently skipped.
- ``env_passthrough``: environment variable names to forward explicitly; when
  ``None`` the subprocess inherits the full environment.
- ``startup_timeout``: seconds to wait for the server to accept the first
  ``initialize`` handshake.

This file ships as a generic empty/example config.  Consumers declare their
own servers here (or by extending ``MCP_SERVERS`` at runtime before calling
:func:`~bili.iris.mcp.loader.initialize_mcp_servers`).

Example
-------
::

    MCP_SERVERS = {
        "my_server": {
            "transport": "stdio",
            "command": "my-llm-cli",
            "args": ["mcp", "serve"],
            "auth": "inherited",
            "enabled": True,
            "env_passthrough": None,
            "startup_timeout": 10.0,
        },
    }
"""

# ---------------------------------------------------------------------------
# MCP server catalog
# ---------------------------------------------------------------------------

#: Maps server name -> server config dict.
#: Extend this dict at application startup to register additional servers.
MCP_SERVERS: dict = {
    # Example entry (disabled by default; enable by setting enabled=True).
    # Replace "my_server" with your server name and fill in the transport
    # details.  Multiple servers can be registered here.
    "example_server": {
        # Transport: "stdio" spawns the command as a subprocess.
        # "http" connects to an HTTP/SSE MCP server at `url`.
        "transport": "stdio",
        # For stdio: the command to spawn.
        "command": "my-cli-llm",
        # Additional arguments passed to the command.
        "args": ["mcp", "serve"],
        # For http: the server's base URL.
        # "url": "http://localhost:8080",
        # Auth strategy.  "inherited" reuses the calling process's env
        # (OAuth session files, API key env vars, etc.).  "none" passes an
        # empty environment.
        "auth": "inherited",
        # Set to True to activate this server.
        "enabled": False,
        # List of env var names to forward explicitly.  None = inherit all.
        "env_passthrough": None,
        # Seconds to wait for the server to complete the initialize handshake.
        "startup_timeout": 10.0,
    },
}
