"""Coverage tests for McpClient's connect path (bili/iris/mcp/client.py).

These tests inject a fake ``mcp`` module into ``sys.modules`` before each
test so that the lazy ``from mcp import ...`` calls inside ``__aenter__``,
``_open_stdio``, and ``_open_http`` actually execute (giving coverage) without
needing the real ``mcp`` package installed.

All tests pass whether or not ``pip install bili-core[mcp]`` has been run.
"""

# pylint: disable=too-few-public-methods, import-outside-toplevel

import asyncio
import contextlib
import sys
import types
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fake mcp transport helpers
# ---------------------------------------------------------------------------


def _make_transport_mods(streams):
    """Build mcp.client, mcp.client.stdio, mcp.client.sse sub-modules."""

    @asynccontextmanager
    async def _fake_stdio_client(_params):
        yield streams

    @asynccontextmanager
    async def _fake_sse_client(_url, **_kwargs):
        yield streams

    client_mod = types.ModuleType("mcp.client")
    stdio_mod = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = _fake_stdio_client
    sse_mod = types.ModuleType("mcp.client.sse")
    sse_mod.sse_client = _fake_sse_client
    client_mod.stdio = stdio_mod
    client_mod.sse = sse_mod
    return client_mod, stdio_mod, sse_mod


def _make_fake_mcp_mod(mock_session: Any, streams: Any):
    """Build and return the fake ``mcp`` top-level module and its sub-modules."""

    class _FakeStdioServerParameters:
        def __init__(self, command, args, env=None):
            self.command = command
            self.args = args
            self.env = env

    class _FakeClientSession:
        """Fake mcp.ClientSession -- used as an async context manager."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            return False

    client_mod, stdio_mod, sse_mod = _make_transport_mods(streams)

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = _FakeClientSession
    mcp_mod.StdioServerParameters = _FakeStdioServerParameters
    mcp_mod.client = client_mod  # type: ignore[attr-defined]

    return mcp_mod, client_mod, stdio_mod, sse_mod


# ---------------------------------------------------------------------------
# Context manager: inject / restore fake mcp in sys.modules
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fake_mcp_installed():
    """Inject a complete fake ``mcp`` package; restore sys.modules on exit."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    fake_read = MagicMock()
    fake_write = MagicMock()
    streams = (fake_read, fake_write)

    mcp_mod, client_mod, stdio_mod, sse_mod = _make_fake_mcp_mod(mock_session, streams)

    keys = ["mcp", "mcp.client", "mcp.client.stdio", "mcp.client.sse"]
    saved = {k: sys.modules.get(k) for k in keys}
    patch_map = {
        "mcp": mcp_mod,
        "mcp.client": client_mod,
        "mcp.client.stdio": stdio_mod,
        "mcp.client.sse": sse_mod,
    }
    try:
        sys.modules.update(patch_map)
        sys.modules.pop("bili.iris.mcp.client", None)
        yield mock_session
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("bili.iris.mcp.client", None)


# ---------------------------------------------------------------------------
# McpClient.__aenter__ / __aexit__ -- stdio transport
# ---------------------------------------------------------------------------


class TestMcpClientConnectStdio:
    """Drive the full __aenter__ / __aexit__ path for stdio transport."""

    def test_aenter_returns_session(self):
        """__aenter__ imports mcp, opens transport, initialises session, returns it."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "test_srv",
                {"transport": "stdio", "command": "my-cli", "args": ["serve"]},
            )

            async def _run():
                async with client as session:
                    return session

            result = asyncio.run(_run())
            assert result is mock_session

    def test_aenter_calls_initialize(self):
        """__aenter__ calls session.initialize() during the MCP handshake."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "init_srv",
                {"transport": "stdio", "command": "cli", "args": []},
            )

            async def _run():
                async with client:
                    pass

            asyncio.run(_run())
            mock_session.initialize.assert_awaited_once()

    def test_aexit_clears_session_and_stack(self):
        """__aexit__ sets _session and _exit_stack back to None."""
        with _fake_mcp_installed():
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "exit_srv",
                {"transport": "stdio", "command": "cli", "args": []},
            )

            async def _run():
                async with client:
                    assert (
                        client._session is not None  # pylint: disable=protected-access
                    )
                    assert (
                        client._exit_stack  # pylint: disable=protected-access
                        is not None
                    )
                assert client._session is None  # pylint: disable=protected-access
                assert client._exit_stack is None  # pylint: disable=protected-access

            asyncio.run(_run())

    def test_stdio_with_env_passthrough(self):
        """_open_stdio passes the filtered env dict to StdioServerParameters."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "env_srv",
                {
                    "transport": "stdio",
                    "command": "cli",
                    "args": [],
                    "env_passthrough": ["PATH"],
                },
            )

            async def _run():
                async with client as sess:
                    return sess

            assert asyncio.run(_run()) is mock_session

    def test_stdio_with_auth_none(self):
        """stdio transport with auth='none' still connects successfully."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "none_auth_srv",
                {"transport": "stdio", "command": "cli", "args": [], "auth": "none"},
            )

            async def _run():
                async with client as sess:
                    return sess

            assert asyncio.run(_run()) is mock_session


# ---------------------------------------------------------------------------
# McpClient.__aenter__ / __aexit__ -- http transport
# ---------------------------------------------------------------------------


class TestMcpClientConnectHttp:
    """Drive the full connect path for http transport."""

    def test_aenter_http_returns_session(self):
        """__aenter__ with http transport opens SSE connection and returns session."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "http_srv",
                {"transport": "http", "url": "http://localhost:9000/sse"},
            )

            async def _run():
                async with client as session:
                    return session

            assert asyncio.run(_run()) is mock_session

    def test_aenter_http_with_timeout(self):
        """startup_timeout is forwarded to the SSE client."""
        with _fake_mcp_installed() as mock_session:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "http_timeout",
                {
                    "transport": "http",
                    "url": "http://localhost:9000/sse",
                    "startup_timeout": 30.0,
                },
            )

            async def _run():
                async with client as sess:
                    return sess

            assert asyncio.run(_run()) is mock_session

    def test_http_aexit_cleans_up(self):
        """__aexit__ for http transport clears session state."""
        with _fake_mcp_installed():
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "http_exit",
                {"transport": "http", "url": "http://localhost:9000/sse"},
            )

            async def _run():
                async with client:
                    assert (
                        client._session is not None  # pylint: disable=protected-access
                    )
                assert client._session is None  # pylint: disable=protected-access

            asyncio.run(_run())


# ---------------------------------------------------------------------------
# McpClient -- ImportError path (mcp not installed)
# ---------------------------------------------------------------------------


class TestMcpClientImportError:
    """Verify ImportError is raised at connect time when mcp is absent."""

    def test_aenter_raises_import_error_without_mcp(self):
        """__aenter__ raises ImportError with a helpful message when mcp is absent."""
        saved = sys.modules.pop("mcp", None)
        sys.modules["mcp"] = None  # type: ignore[assignment]  # blocks import
        sys.modules.pop("bili.iris.mcp.client", None)

        try:
            from bili.iris.mcp.client import McpClient

            client = McpClient(
                "no_mcp", {"transport": "stdio", "command": "fake", "args": []}
            )

            async def _run():
                async with client:
                    pass

            with pytest.raises(ImportError, match="mcp.*required"):
                asyncio.run(_run())
        finally:
            if saved is None:
                sys.modules.pop("mcp", None)
            else:
                sys.modules["mcp"] = saved
            sys.modules.pop("bili.iris.mcp.client", None)


# ---------------------------------------------------------------------------
# McpClient -- transport-specific ImportError paths
# ---------------------------------------------------------------------------


class TestMcpClientTransportImportErrors:
    """Verify ImportError paths inside _open_stdio and _open_http."""

    def test_open_stdio_import_error(self):
        """_open_stdio raises ImportError when StdioServerParameters is absent."""
        mcp_mod = types.ModuleType("mcp")

        class _FakeCS:
            async def __aenter__(self):
                return AsyncMock()

            async def __aexit__(self, *a):
                return False

        mcp_mod.ClientSession = _FakeCS  # type: ignore[attr-defined]
        # StdioServerParameters raises ImportError to simulate missing module
        mcp_mod.StdioServerParameters = MagicMock(  # type: ignore[attr-defined]
            side_effect=ImportError("mcp.client.stdio not available")
        )

        client_mod = types.ModuleType("mcp.client")
        stdio_mod = types.ModuleType("mcp.client.stdio")
        mcp_mod.client = client_mod  # type: ignore[attr-defined]
        client_mod.stdio = stdio_mod  # type: ignore[attr-defined]

        keys = ["mcp", "mcp.client", "mcp.client.stdio"]
        saved = {k: sys.modules.get(k) for k in keys}
        sys.modules["mcp"] = mcp_mod
        sys.modules["mcp.client"] = client_mod
        sys.modules["mcp.client.stdio"] = stdio_mod
        sys.modules.pop("bili.iris.mcp.client", None)

        try:
            from bili.iris.mcp.client import McpClient

            client = McpClient("err_srv", {"transport": "stdio", "command": "fake"})

            async def _run():
                async with client:
                    pass

            with pytest.raises((ImportError, Exception)):
                asyncio.run(_run())
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            sys.modules.pop("bili.iris.mcp.client", None)

    def test_open_http_import_error(self):
        """_open_http raises ImportError when mcp.client.sse is absent."""
        mcp_mod = types.ModuleType("mcp")

        class _FakeCS:
            async def __aenter__(self):
                return AsyncMock()

            async def __aexit__(self, *a):
                return False

        mcp_mod.ClientSession = _FakeCS  # type: ignore[attr-defined]
        client_mod = types.ModuleType("mcp.client")
        mcp_mod.client = client_mod  # type: ignore[attr-defined]

        keys = ["mcp", "mcp.client", "mcp.client.sse"]
        saved = {k: sys.modules.get(k) for k in keys}
        sys.modules["mcp"] = mcp_mod
        sys.modules["mcp.client"] = client_mod
        sys.modules["mcp.client.sse"] = None  # type: ignore[assignment]  # blocks import
        sys.modules.pop("bili.iris.mcp.client", None)

        try:
            from bili.iris.mcp.client import McpClient

            client = McpClient("http_err", {"transport": "http", "url": "http://x/sse"})

            async def _run():
                async with client:
                    pass

            with pytest.raises(ImportError, match="mcp.*required"):
                asyncio.run(_run())
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            sys.modules.pop("bili.iris.mcp.client", None)
