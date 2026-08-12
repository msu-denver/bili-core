"""Integration tests for the MCP loader and lifecycle (bili/iris/mcp/loader.py).

Uses in-process mock MCP sessions -- no real subprocess or network connection.
All tests pass without a running MCP server.

Real-CLI integration tests (spawning an actual MCP server process) are gated
by the ``MCP_INTEGRATION_TEST`` environment variable and are skipped by default
in CI.
"""

# pylint: disable=too-few-public-methods, import-outside-toplevel

import asyncio
import os
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bili.iris.mcp.loader import (
    McpServerSession,
    initialize_mcp_servers,
    initialize_mcp_servers_sync,
    register_mcp_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(name: str, description: str = "A test tool") -> Any:
    """Return a minimal mock mcp.types.Tool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = MagicMock()
    tool.input_schema.model_dump = MagicMock(
        return_value={"type": "object", "properties": {}}
    )
    return tool


def _make_mock_session(tool_names: List[str]) -> Any:
    """Return a mock mcp.ClientSession with list_tools and call_tool stubbed."""
    session = AsyncMock()
    mcp_tools = [_make_mcp_tool(n) for n in tool_names]
    list_result = MagicMock()
    list_result.tools = mcp_tools
    session.list_tools = AsyncMock(return_value=list_result)

    async def _call_tool(name, arguments=None):  # pylint: disable=unused-argument
        result = MagicMock()
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = f"result_from_{name}"
        result.content = [content_block]
        result.is_error = False
        return result

    session.call_tool = _call_tool
    return session, mcp_tools


def _make_mock_client(session: Any) -> Any:
    """Return a mock McpClient that returns the given session on __aenter__."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=session)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _make_server_session(name: str, tool_names: List[str]) -> McpServerSession:
    """Build a McpServerSession backed by mock objects."""
    session, mcp_tools = _make_mock_session(tool_names)
    client = _make_mock_client(session)
    return McpServerSession(
        server_name=name,
        session=session,
        mcp_tools=mcp_tools,
        client=client,
    )


# ---------------------------------------------------------------------------
# McpServerSession
# ---------------------------------------------------------------------------


class TestMcpServerSession:
    """Tests for McpServerSession construction and teardown."""

    def test_attributes_set(self):
        """McpServerSession stores server_name, session, and mcp_tools."""
        ss = _make_server_session("srv", ["tool_a"])
        assert ss.server_name == "srv"
        assert ss.session is not None
        assert len(ss.mcp_tools) == 1

    def test_tool_count_property(self):
        """tool_count returns the number of tools in the session."""
        ss = _make_server_session("srv", ["a", "b", "c"])
        assert ss.tool_count == 3

    def test_close_tears_down_client(self):
        """close() tears down the underlying client via aclose or __aexit__."""
        ss = _make_server_session("srv", ["t"])
        asyncio.run(ss.close())
        # AsyncMock auto-creates aclose; the production path calls aclose().
        # pylint: disable=protected-access
        assert ss._client.aclose.called or ss._client.__aexit__.called
        # pylint: enable=protected-access


# ---------------------------------------------------------------------------
# register_mcp_tools + McpLifecycle
# ---------------------------------------------------------------------------


class TestRegisterMcpTools:
    """Tests for register_mcp_tools() and the resulting McpLifecycle."""

    def test_registers_tools_in_tool_registry(self):
        """register_mcp_tools inserts namespaced keys into TOOL_REGISTRY."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss = _make_server_session("myserver", ["tool_one", "tool_two"])
        lifecycle = register_mcp_tools([ss])
        try:
            assert "myserver__tool_one" in TOOL_REGISTRY
            assert "myserver__tool_two" in TOOL_REGISTRY
        finally:
            asyncio.run(lifecycle.close())

    def test_lifecycle_tool_names(self):
        """McpLifecycle.tool_names lists the registered namespaced keys."""
        ss = _make_server_session("srv", ["alpha", "beta"])
        lifecycle = register_mcp_tools([ss])
        try:
            assert "srv__alpha" in lifecycle.tool_names
            assert "srv__beta" in lifecycle.tool_names
        finally:
            asyncio.run(lifecycle.close())

    def test_lifecycle_close_removes_from_registry(self):
        """lifecycle.close() removes the registered keys from TOOL_REGISTRY."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss = _make_server_session("cleanup_srv", ["tool_x"])
        lifecycle = register_mcp_tools([ss])
        assert "cleanup_srv__tool_x" in TOOL_REGISTRY

        asyncio.run(lifecycle.close())

        assert "cleanup_srv__tool_x" not in TOOL_REGISTRY

    def test_lifecycle_close_calls_session_close(self):
        """lifecycle.close() tears down the server session."""
        ss = _make_server_session("srv_tear", ["t"])
        lifecycle = register_mcp_tools([ss])
        asyncio.run(lifecycle.close())
        # aclose or __aexit__ called confirms teardown
        # pylint: disable=protected-access
        assert ss._client.aclose.called or ss._client.__aexit__.called
        # pylint: enable=protected-access

    def test_async_context_manager_returns_tool_names(self):
        """async with lifecycle as tool_names yields the registered names."""

        async def _run():
            ss = _make_server_session("ctx_srv", ["t1", "t2"])
            lifecycle = register_mcp_tools([ss])
            async with lifecycle as names:
                return names

        names = asyncio.run(_run())
        assert "ctx_srv__t1" in names
        assert "ctx_srv__t2" in names

    def test_async_context_manager_cleans_up_on_exit(self):
        """The async context manager removes tools from TOOL_REGISTRY on exit."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        async def _run():
            ss = _make_server_session("cleanup2", ["tx"])
            lifecycle = register_mcp_tools([ss])
            async with lifecycle:
                assert "cleanup2__tx" in TOOL_REGISTRY
            assert "cleanup2__tx" not in TOOL_REGISTRY

        asyncio.run(_run())

    def test_active_tool_names_filter(self):
        """active_tool_names allows registering a subset of discovered tools."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss = _make_server_session("filter_srv", ["keep_me", "skip_me"])
        lifecycle = register_mcp_tools([ss], active_tool_names=["filter_srv__keep_me"])
        try:
            assert "filter_srv__keep_me" in TOOL_REGISTRY
            assert "filter_srv__skip_me" not in TOOL_REGISTRY
        finally:
            asyncio.run(lifecycle.close())

    def test_multiple_servers(self):
        """Tools from multiple servers are all registered under their namespaces."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss1 = _make_server_session("s1", ["t1"])
        ss2 = _make_server_session("s2", ["t2"])
        lifecycle = register_mcp_tools([ss1, ss2])
        try:
            assert "s1__t1" in TOOL_REGISTRY
            assert "s2__t2" in TOOL_REGISTRY
        finally:
            asyncio.run(lifecycle.close())

    def test_registered_tool_is_callable(self):
        """The lambda stored in TOOL_REGISTRY returns the StructuredTool."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss = _make_server_session("call_srv", ["callable_tool"])
        lifecycle = register_mcp_tools([ss])
        try:
            fn = TOOL_REGISTRY["call_srv__callable_tool"]
            result = fn("call_srv__callable_tool", "prompt", {})
            assert result is not None
            assert result.name == "call_srv__callable_tool"
        finally:
            asyncio.run(lifecycle.close())


# ---------------------------------------------------------------------------
# Tool round-trip: call_tool on a registered MCP tool
# ---------------------------------------------------------------------------


class TestToolRoundTrip:
    """End-to-end: a registered MCP tool can be invoked (sync and async)."""

    def _setup_registered_tool(self, server_name: str, tool_name: str):
        """Register a single MCP tool and return the StructuredTool + lifecycle."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        ss = _make_server_session(server_name, [tool_name])
        lifecycle = register_mcp_tools([ss])
        namespaced = f"{server_name}__{tool_name}"
        lc_tool = TOOL_REGISTRY[namespaced](namespaced, "prompt", {})
        return lc_tool, lifecycle

    def test_sync_tool_invocation(self):
        """The sync func path returns a string result from the MCP session."""
        lc_tool, lifecycle = self._setup_registered_tool("roundtrip_srv", "greet")
        try:
            result = lc_tool.func()
            assert isinstance(result, str)
        finally:
            asyncio.run(lifecycle.close())

    def test_async_tool_invocation(self):
        """The async coroutine path returns a string result from the MCP session."""
        lc_tool, lifecycle = self._setup_registered_tool("async_srv", "echo")

        async def _run():
            return await lc_tool.coroutine()

        try:
            result = asyncio.run(_run())
            assert isinstance(result, str)
        finally:
            asyncio.run(lifecycle.close())

    def test_tool_result_contains_server_output(self):
        """The result string contains the mock server's response text."""
        lc_tool, lifecycle = self._setup_registered_tool("content_srv", "mytool")
        try:
            result = lc_tool.func()
            assert "result_from_mytool" in result
        finally:
            asyncio.run(lifecycle.close())


# ---------------------------------------------------------------------------
# initialize_mcp_servers
# ---------------------------------------------------------------------------


def _make_patched_client(tool_names: List[str]):
    """Return a mock McpClient tuple (client, session, mcp_tools)."""
    session, mcp_tools = _make_mock_session(tool_names)
    client = _make_mock_client(session)
    return client, session, mcp_tools


class TestInitializeMcpServers:
    """Tests for initialize_mcp_servers() with a mock McpClient."""

    def test_disabled_server_skipped(self):
        """A server with enabled=False is not initialized."""
        configs = {
            "disabled_srv": {
                "transport": "stdio",
                "command": "fake",
                "args": [],
                "enabled": False,
            }
        }

        async def _run():
            return await initialize_mcp_servers(server_configs=configs)

        sessions = asyncio.run(_run())
        assert not sessions

    def test_not_in_active_servers_skipped(self):
        """A server not in active_servers is skipped even if enabled."""
        configs = {
            "other_srv": {
                "transport": "stdio",
                "command": "fake",
                "args": [],
                "enabled": True,
            }
        }

        async def _run():
            return await initialize_mcp_servers(
                active_servers=["different_name"], server_configs=configs
            )

        sessions = asyncio.run(_run())
        assert not sessions

    def test_enabled_server_initialized(self):
        """An enabled server in active_servers returns a McpServerSession."""
        client_mock, _session, _tools = _make_patched_client(["tool_a"])

        configs = {
            "live_srv": {
                "transport": "stdio",
                "command": "fake",
                "args": [],
                "enabled": True,
            }
        }

        with patch(
            "bili.iris.mcp.loader.McpClient",
            return_value=client_mock,
        ):

            async def _run():
                return await initialize_mcp_servers(
                    active_servers=["live_srv"], server_configs=configs
                )

            sessions = asyncio.run(_run())

        assert len(sessions) == 1
        assert sessions[0].server_name == "live_srv"
        assert len(sessions[0].mcp_tools) == 1

    def test_session_has_discovered_tools(self):
        """The returned McpServerSession holds the tools from list_tools()."""
        client_mock, _s, _t = _make_patched_client(["tool_x", "tool_y", "tool_z"])
        configs = {
            "multi_srv": {
                "transport": "stdio",
                "command": "f",
                "args": [],
                "enabled": True,
            }
        }

        with patch("bili.iris.mcp.loader.McpClient", return_value=client_mock):

            async def _run():
                return await initialize_mcp_servers(server_configs=configs)

            sessions = asyncio.run(_run())

        tool_names = [t.name for t in sessions[0].mcp_tools]
        assert set(tool_names) == {"tool_x", "tool_y", "tool_z"}

    def test_multiple_servers_all_initialized(self):
        """Multiple enabled servers all produce McpServerSessions."""
        client1, _s1, _t1 = _make_patched_client(["t1"])
        client2, _s2, _t2 = _make_patched_client(["t2"])

        configs = {
            "srv1": {
                "transport": "stdio",
                "command": "f1",
                "args": [],
                "enabled": True,
            },
            "srv2": {
                "transport": "stdio",
                "command": "f2",
                "args": [],
                "enabled": True,
            },
        }

        call_count = [0]
        clients = [client1, client2]

        def _client_factory(_name, _cfg):
            c = clients[call_count[0] % 2]
            call_count[0] += 1
            return c

        with patch("bili.iris.mcp.loader.McpClient", side_effect=_client_factory):

            async def _run():
                return await initialize_mcp_servers(server_configs=configs)

            sessions = asyncio.run(_run())

        assert len(sessions) == 2
        names = {s.server_name for s in sessions}
        assert names == {"srv1", "srv2"}


# ---------------------------------------------------------------------------
# initialize_mcp_servers_sync
# ---------------------------------------------------------------------------


class TestInitializeMcpServersSync:
    """Tests for the synchronous wrapper."""

    def test_sync_wrapper_returns_sessions(self):
        """initialize_mcp_servers_sync returns the same result as the async version."""
        client_mock, _s, _t = _make_patched_client(["t"])
        configs = {
            "sync_srv": {
                "transport": "stdio",
                "command": "f",
                "args": [],
                "enabled": True,
            }
        }

        with patch("bili.iris.mcp.loader.McpClient", return_value=client_mock):
            sessions = initialize_mcp_servers_sync(server_configs=configs)

        assert len(sessions) == 1
        assert sessions[0].server_name == "sync_srv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestMcpConfig:
    """Smoke test: the built-in config loads and has the expected shape."""

    def test_config_loads(self):
        """MCP_SERVERS can be imported without error."""
        from bili.iris.mcp.config import MCP_SERVERS

        assert isinstance(MCP_SERVERS, dict)

    def test_example_server_entry(self):
        """The example server entry has the required keys."""
        from bili.iris.mcp.config import MCP_SERVERS

        assert "example_server" in MCP_SERVERS
        entry = MCP_SERVERS["example_server"]
        assert "transport" in entry
        assert "enabled" in entry
        # Example is disabled by default
        assert entry["enabled"] is False


# ---------------------------------------------------------------------------
# McpClient environment building
# ---------------------------------------------------------------------------


class TestMcpClientEnv:
    """Tests for McpClient._build_env() -- no real subprocess spawned."""

    def test_inherited_no_passthrough_returns_none(self):
        """auth='inherited' with no env_passthrough returns None (full env inherit)."""
        from bili.iris.mcp.client import McpClient

        client = McpClient(
            "srv", {"transport": "stdio", "command": "fake", "auth": "inherited"}
        )
        result = client._build_env()  # pylint: disable=protected-access
        assert result is None

    def test_default_auth_returns_none(self):
        """No auth key (defaults to 'inherited') returns None."""
        from bili.iris.mcp.client import McpClient

        client = McpClient("srv", {"transport": "stdio", "command": "fake"})
        result = client._build_env()  # pylint: disable=protected-access
        assert result is None

    def test_auth_none_also_returns_none(self):
        """auth='none' with no env_passthrough still returns None.

        'none' means no auth credentials, not no environment.  The subprocess
        must inherit at least PATH so it can locate executables.
        """
        from bili.iris.mcp.client import McpClient

        client = McpClient(
            "srv", {"transport": "stdio", "command": "fake", "auth": "none"}
        )
        result = client._build_env()  # pylint: disable=protected-access
        # Must NOT return {} (empty env) -- the subprocess needs PATH etc.
        assert result is None

    def test_env_passthrough_restricts_to_listed_vars_plus_baseline(self):
        """env_passthrough forwards only listed vars and the safety baseline."""
        from bili.iris.mcp.client import McpClient

        client = McpClient(
            "srv",
            {
                "transport": "stdio",
                "command": "fake",
                "env_passthrough": ["MY_API_KEY"],
            },
        )
        env = client._build_env()  # pylint: disable=protected-access
        assert env is not None
        # "PATH" is always in the baseline (if present in os.environ)
        if "PATH" in os.environ:
            assert "PATH" in env
        # Listed var is forwarded if present
        if "MY_API_KEY" in os.environ:
            assert "MY_API_KEY" in env

    def test_env_passthrough_never_empty(self):
        """env_passthrough result always contains at least the PATH baseline."""
        from bili.iris.mcp.client import McpClient

        client = McpClient(
            "srv",
            {
                "transport": "stdio",
                "command": "fake",
                "env_passthrough": [],  # empty explicit list
            },
        )
        env = client._build_env()  # pylint: disable=protected-access
        # Even with an empty passthrough list, PATH (if set) must be present
        if "PATH" in os.environ:
            assert env is not None
            assert "PATH" in env


# ---------------------------------------------------------------------------
# McpClient config validation
# ---------------------------------------------------------------------------


class TestMcpClientConfig:
    """Tests for McpClient's config validation.

    Validation is pure Python (no mcp SDK import), so these tests pass
    even when the mcp optional extra is NOT installed.  Errors are raised
    at construction time, not at connect time.
    """

    def test_stdio_missing_command_raises(self):
        """A stdio config without 'command' raises ValueError at construction."""
        from bili.iris.mcp.client import McpClient

        with pytest.raises(ValueError, match="'command' is required"):
            McpClient("bad_srv", {"transport": "stdio", "args": []})

    def test_http_missing_url_raises(self):
        """An http config without 'url' raises ValueError at construction."""
        from bili.iris.mcp.client import McpClient

        with pytest.raises(ValueError, match="'url' is required"):
            McpClient("bad_http", {"transport": "http"})

    def test_unsupported_transport_raises(self):
        """An unknown transport raises ValueError at construction."""
        from bili.iris.mcp.client import McpClient

        with pytest.raises(ValueError, match="unsupported transport"):
            McpClient("bad_t", {"transport": "grpc"})


# ---------------------------------------------------------------------------
# Loader branch coverage: aopen, exception paths, overwrite warning,
# server_configs=None default, legacy __aexit__ session close path
# ---------------------------------------------------------------------------


class TestLoaderBranchCoverage:
    """Cover branches in loader.py that are not hit by the main scenario tests."""

    def test_lifecycle_aopen(self):
        """McpLifecycle.aopen() returns the registered tool names."""
        ss = _make_server_session("aopen_srv", ["t1"])
        lifecycle = register_mcp_tools([ss])
        try:
            names = asyncio.run(lifecycle.aopen())
            assert "aopen_srv__t1" in names
        finally:
            asyncio.run(lifecycle.close())

    def test_lifecycle_close_handles_session_close_failure(self):
        """McpLifecycle.close() logs and continues when a session fails to close."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        # Build a server session whose close() raises
        ss = _make_server_session("fail_srv", ["tool"])
        ss._client.aclose = AsyncMock(
            side_effect=RuntimeError("close failed")
        )  # pylint: disable=protected-access

        lifecycle = register_mcp_tools([ss])
        # close() must NOT propagate the exception -- it logs and continues
        asyncio.run(lifecycle.close())
        # Tool must still be removed from TOOL_REGISTRY despite the session error
        assert "fail_srv__tool" not in TOOL_REGISTRY

    def test_register_mcp_tools_overwrites_existing_entry(self):
        """register_mcp_tools logs a warning when overwriting an existing TOOL_REGISTRY key."""
        from bili.iris.loaders.tools_loader import TOOL_REGISTRY

        # Pre-seed a key that will be overwritten
        TOOL_REGISTRY["overwrite_srv__mytool"] = lambda n, p, params: None

        ss = _make_server_session("overwrite_srv", ["mytool"])
        lifecycle = register_mcp_tools([ss])
        try:
            # No exception; the warning was logged (not asserted here -- just coverage)
            assert "overwrite_srv__mytool" in TOOL_REGISTRY
        finally:
            asyncio.run(lifecycle.close())

    def test_initialize_mcp_servers_uses_default_config_when_none(self):
        """initialize_mcp_servers uses MCP_SERVERS when server_configs=None."""

        # All entries in MCP_SERVERS are disabled by default, so no sessions are
        # returned -- but the branch (importing MCP_SERVERS) still executes.
        async def _run():
            return await initialize_mcp_servers(server_configs=None)

        sessions = asyncio.run(_run())
        # The built-in example_server is disabled; no sessions should be created
        assert isinstance(sessions, list)

    def test_initialize_mcp_servers_cleans_up_on_connect_failure(self):
        """initialize_mcp_servers tears down the exit_stack when a server fails."""
        bad_client = AsyncMock()
        bad_client.__aenter__ = AsyncMock(side_effect=RuntimeError("connect failed"))
        bad_client.__aexit__ = AsyncMock(return_value=False)

        configs = {
            "fail_srv": {
                "transport": "stdio",
                "command": "fake",
                "args": [],
                "enabled": True,
            }
        }

        with patch("bili.iris.mcp.loader.McpClient", return_value=bad_client):

            async def _run():
                return await initialize_mcp_servers(server_configs=configs)

            with pytest.raises(RuntimeError, match="connect failed"):
                asyncio.run(_run())

    def test_initialize_mcp_servers_cleanup_survives_aclose_failure(self):
        """The inner except in initialize_mcp_servers swallows aclose() errors.

        When exit_stack.aclose() itself raises, the inner except: pass swallows
        it and the original connection error propagates.
        """
        import contextlib

        bad_client = AsyncMock()
        bad_client.__aenter__ = AsyncMock(side_effect=RuntimeError("enter failed"))
        bad_client.__aexit__ = AsyncMock(return_value=False)

        configs = {
            "double_fail": {
                "transport": "stdio",
                "command": "fake",
                "args": [],
                "enabled": True,
            }
        }

        async def _broken_aclose(self):
            raise RuntimeError("aclose also failed")

        with patch("bili.iris.mcp.loader.McpClient", return_value=bad_client):
            with patch.object(contextlib.AsyncExitStack, "aclose", _broken_aclose):

                async def _run():
                    return await initialize_mcp_servers(server_configs=configs)

                # Original "enter failed" propagates; "aclose also failed" is swallowed
                with pytest.raises(RuntimeError, match="enter failed"):
                    asyncio.run(_run())

    def test_mcpserversession_close_legacy_path(self):
        """McpServerSession.close() uses __aexit__ when _client has no aclose."""
        # Build a mock client WITHOUT aclose to hit the legacy __aexit__ branch
        legacy_client = MagicMock(spec=["__aexit__"])
        legacy_client.__aexit__ = AsyncMock(return_value=False)

        session, mcp_tools = _make_mock_session(["t"])
        ss = McpServerSession(
            server_name="legacy",
            session=session,
            mcp_tools=mcp_tools,
            client=legacy_client,
        )
        asyncio.run(ss.close())
        legacy_client.__aexit__.assert_awaited_once()


# ---------------------------------------------------------------------------
# Real-CLI integration test (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MCP_INTEGRATION_TEST"),
    reason="Set MCP_INTEGRATION_TEST=1 to run real MCP server integration tests",
)
class TestRealCliIntegration:
    """Integration tests that spawn a real MCP server process.

    Gated by the MCP_INTEGRATION_TEST environment variable.  These tests
    are skipped by default in CI.  To run them locally::

        MCP_INTEGRATION_TEST=1 pytest bili/iris/mcp/tests/test_loader.py::TestRealCliIntegration -v

    The tests require the CLI tool under test to be installed in the PATH
    and to implement the MCP server protocol.  Set MCP_CLI_COMMAND to the
    executable name (default: ``"echo"`` as a placeholder).
    """

    def test_placeholder(self):
        """Placeholder: replace with a real server spawn when MCP_INTEGRATION_TEST=1."""
        cli_command = os.environ.get("MCP_CLI_COMMAND", "echo")
        configs = {
            "integration_srv": {
                "transport": "stdio",
                "command": cli_command,
                "args": [],
                "enabled": True,
                "startup_timeout": 5.0,
            }
        }

        async def _run():
            servers = await initialize_mcp_servers(server_configs=configs)
            if not servers:
                pytest.skip("No servers initialized")
            lifecycle = register_mcp_tools(servers)
            async with lifecycle as tool_names:
                assert isinstance(tool_names, list)

        asyncio.run(_run())
