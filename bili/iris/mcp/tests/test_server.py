"""Tests for bili/iris/mcp/server.py.

Tests the ephemeral MCP server and its components:

- _build_mcp_fn: function generation from LangChain tools (structured + plain)
- _TokenAuthMiddleware: request authorization (valid, invalid, lifespan pass-through)
- _find_free_port: allocates an integer port
- _wait_for_server_ready: accepts TCP or raises TimeoutError
- EphemeralMcpServer: lifecycle (start, register tools, stop; teardown on error)
- EphemeralMcpHandle: dataclass fields
- build_mcp_node: node callable exercises the server, injector, subprocess, and cleanup
- resolve_mcp_injector: returns injector for CliLLM, None for non-CLI models
- _parse_output: text and json paths

All tests run without a real MCP server or CLI binary.  Network I/O is mocked.
"""

# pylint: disable=too-few-public-methods,protected-access,missing-function-docstring,missing-class-docstring,import-outside-toplevel

import asyncio
import inspect
import json
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from bili.iris.mcp.server import (
    EphemeralMcpHandle,
    EphemeralMcpServer,
    _build_mcp_fn,
    _build_tool_preamble,
    _find_free_port,
    _parse_output,
    _TokenAuthMiddleware,
    _wait_for_server_ready,
    build_mcp_node,
    resolve_mcp_injector,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_plain_tool(
    name: str = "my_tool", description: str = "Does something"
) -> BaseTool:
    """Return a plain (no args_schema) BaseTool mock."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = description
    tool.args_schema = None
    tool.invoke = MagicMock(return_value="plain result")
    return tool


class _SearchArgs(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Max number of results")


def _make_structured_tool(
    name: str = "search_tool", description: str = "Searches the web"
) -> StructuredTool:
    """Return a StructuredTool with a Pydantic args_schema."""

    def _search(query: str, max_results: int = 5) -> str:
        return f"results for {query}"

    return StructuredTool(
        name=name,
        description=description,
        func=_search,
        args_schema=_SearchArgs,
    )


# ---------------------------------------------------------------------------
# _build_mcp_fn
# ---------------------------------------------------------------------------


class TestBuildMcpFn:
    """Tests for _build_mcp_fn."""

    def test_plain_tool_returns_callable(self):
        tool = _make_plain_tool()
        fn = _build_mcp_fn(tool)
        assert callable(fn)

    def test_plain_tool_name(self):
        tool = _make_plain_tool(name="my_tool")
        fn = _build_mcp_fn(tool)
        assert fn.__name__ == "my_tool"

    def test_plain_tool_single_str_param(self):
        tool = _make_plain_tool()
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ["tool_input"]
        assert sig.parameters["tool_input"].annotation is str

    def test_plain_tool_invokes_tool_invoke(self):
        tool = _make_plain_tool()
        fn = _build_mcp_fn(tool)
        result = fn("hello")
        tool.invoke.assert_called_once_with("hello")
        assert result == "plain result"

    def test_structured_tool_returns_callable(self):
        tool = _make_structured_tool()
        fn = _build_mcp_fn(tool)
        assert callable(fn)

    def test_structured_tool_name(self):
        tool = _make_structured_tool(name="search_tool")
        fn = _build_mcp_fn(tool)
        assert fn.__name__ == "search_tool"

    def test_structured_tool_has_schema_params(self):
        tool = _make_structured_tool()
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        assert "query" in param_names
        assert "max_results" in param_names

    def test_structured_tool_required_param_no_default(self):
        tool = _make_structured_tool()
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        assert sig.parameters["query"].default is inspect.Parameter.empty

    def test_structured_tool_optional_param_has_default(self):
        tool = _make_structured_tool()
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        assert sig.parameters["max_results"].default == 5

    def test_structured_tool_invokes_tool_with_kwargs(self):
        # Use a MagicMock as the underlying func so we can verify call args without
        # patching StructuredTool's Pydantic-managed attributes.
        mock_func = MagicMock(return_value="found stuff")
        tool = StructuredTool(
            name="search_tool",
            description="Searches the web",
            func=mock_func,
            args_schema=_SearchArgs,
        )
        fn = _build_mcp_fn(tool)
        result = fn(query="python", max_results=3)
        # _build_mcp_fn calls tool.invoke(kwargs); BaseTool.invoke routes to _run → func.
        mock_func.assert_called_once_with(query="python", max_results=3)
        assert result == "found stuff"

    def test_tool_with_none_args_schema_uses_plain_path(self):
        tool = MagicMock()
        tool.name = "plain"
        tool.description = "desc"
        tool.args_schema = None
        tool.invoke = MagicMock(return_value="x")
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        assert list(sig.parameters.keys()) == ["tool_input"]

    def test_tool_with_schema_no_model_fields_uses_plain_path(self):
        tool = MagicMock()
        tool.name = "nt"
        tool.description = "desc"
        tool.args_schema = object()  # no model_fields
        tool.invoke = MagicMock(return_value="x")
        fn = _build_mcp_fn(tool)
        sig = inspect.signature(fn)
        assert list(sig.parameters.keys()) == ["tool_input"]


# ---------------------------------------------------------------------------
# _TokenAuthMiddleware
# ---------------------------------------------------------------------------

#: Arbitrary ports for the middleware's peer lookup. The tests in
#: TestTokenAuthMiddleware drive the TOKEN check, so their authorization is
#: stubbed and these are never resolved against a real connection. The real
#: lookup is exercised in TestConnectionIsBoundToTheSpawnedProcessTree.
_SERVER_PORT = 51000
_PEER_PORT = 51001

#: Stand-in PID for the spawned subprocess in the build_mcp_node tests, where
#: both the server and Popen are mocked and nothing is really spawned.
_SPAWNED_PID = 424242


class _PermissiveAuth:  # pylint: disable=too-few-public-methods
    """Authorization stub that permits, isolating the token check under test."""

    def permits(self, peer_port: int, server_port: int) -> bool:
        """Permit every caller."""
        del peer_port, server_port
        return True


class _DenyingAuth:  # pylint: disable=too-few-public-methods
    """Authorization stub that refuses, isolating the identity check."""

    def permits(self, peer_port: int, server_port: int) -> bool:
        """Refuse every caller."""
        del peer_port, server_port
        return False


class TestTokenAuthMiddleware:
    """Tests for _TokenAuthMiddleware ASGI middleware."""

    TOKEN = "test-secret-token-abc123"

    def _make_scope(self, scope_type: str = "http", auth_header: bytes = b"") -> dict:
        headers = [(b"authorization", auth_header)] if auth_header else []
        return {
            "type": scope_type,
            "path": "/mcp",
            "headers": headers,
            "client": ("127.0.0.1", _PEER_PORT),
        }

    async def _collect_response(self, scope, middleware) -> dict:
        """Run the middleware and collect the ASGI response events."""
        events = []

        async def _send(event):
            events.append(event)

        async def _receive():
            return {}

        await middleware(scope, _receive, _send)
        return events

    def test_valid_token_passes_to_inner_app(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = self._make_scope(auth_header=f"Bearer {self.TOKEN}".encode())
        asyncio.run(mw(scope, AsyncMock(), AsyncMock()))
        inner.assert_called_once()

    def test_missing_auth_header_returns_401(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = self._make_scope()
        events = asyncio.run(self._collect_response(scope, mw))
        inner.assert_not_called()
        assert events[0]["status"] == 401

    def test_wrong_token_returns_401(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = self._make_scope(auth_header=b"Bearer wrong-token")
        events = asyncio.run(self._collect_response(scope, mw))
        inner.assert_not_called()
        assert events[0]["status"] == 401

    def test_lifespan_scope_passes_through(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = {"type": "lifespan"}
        asyncio.run(mw(scope, AsyncMock(), AsyncMock()))
        inner.assert_called_once()

    def test_websocket_scope_passes_through(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = {"type": "websocket", "path": "/ws", "headers": []}
        asyncio.run(mw(scope, AsyncMock(), AsyncMock()))
        inner.assert_called_once()

    def test_401_response_body_is_valid_json(self):
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = self._make_scope()
        events = asyncio.run(self._collect_response(scope, mw))
        body = events[1]["body"]
        assert json.loads(body)  # Must be valid JSON

    def test_401_content_length_matches_body(self):
        """Declared Content-Length must equal actual body length (prevents uvicorn error)."""
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = self._make_scope()
        events = asyncio.run(self._collect_response(scope, mw))
        headers = dict(events[0]["headers"])
        declared = int(headers[b"content-length"])
        actual = len(events[1]["body"])
        assert (
            declared == actual
        ), f"Content-Length {declared} does not match body length {actual}"

    def test_header_case_insensitive_lookup(self):
        """Auth header lookup must be case-insensitive (ASGI headers are bytes)."""
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        # ASGI normalizes header names to lowercase bytes, but test explicitly.
        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"Authorization", f"Bearer {self.TOKEN}".encode())],
            "client": ("127.0.0.1", _PEER_PORT),
        }
        # With the standard lowercase key used by our middleware:
        scope2 = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", f"Bearer {self.TOKEN}".encode())],
            "client": ("127.0.0.1", _PEER_PORT),
        }
        asyncio.run(mw(scope2, AsyncMock(), AsyncMock()))
        inner.assert_called_once()

    def test_valid_token_from_an_unauthorized_caller_returns_403(self):
        """A correct token is necessary and not sufficient.

        The two refusals carry different statuses so an operator can tell a
        wrong token apart from the right token used by the wrong process; they
        are different problems with different fixes.
        """
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _DenyingAuth(), _SERVER_PORT)
        scope = self._make_scope(auth_header=f"Bearer {self.TOKEN}".encode())
        events = asyncio.run(self._collect_response(scope, mw))
        inner.assert_not_called()
        assert events[0]["status"] == 403

    def test_a_request_with_no_client_address_is_refused(self):
        """An unattributable caller is refused rather than waved through.

        A request whose peer cannot be identified is precisely the case the
        identity check exists for, so the absent-client branch must fail
        closed; treating it as "cannot check, therefore allow" would reopen
        the whole gap to anyone who can suppress the field.
        """
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _PermissiveAuth(), _SERVER_PORT)
        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", f"Bearer {self.TOKEN}".encode())],
            "client": None,
        }
        events = asyncio.run(self._collect_response(scope, mw))
        inner.assert_not_called()
        assert events[0]["status"] == 403

    def test_403_content_length_matches_body(self):
        """The 403 shares the 401's response builder, so it shares its contract."""
        inner = AsyncMock()
        mw = _TokenAuthMiddleware(inner, self.TOKEN, _DenyingAuth(), _SERVER_PORT)
        scope = self._make_scope(auth_header=f"Bearer {self.TOKEN}".encode())
        events = asyncio.run(self._collect_response(scope, mw))
        headers = dict(events[0]["headers"])
        assert int(headers[b"content-length"]) == len(events[1]["body"])
        assert json.loads(events[1]["body"])


# ---------------------------------------------------------------------------
# _find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_integer(self):
        port = _find_free_port()
        assert isinstance(port, int)

    def test_returns_valid_port_range(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535

    def test_two_calls_may_differ(self):
        # Not guaranteed to differ, but usually will; mostly checks no exception.
        p1 = _find_free_port()
        p2 = _find_free_port()
        assert isinstance(p1, int) and isinstance(p2, int)


# ---------------------------------------------------------------------------
# _wait_for_server_ready
# ---------------------------------------------------------------------------


class TestWaitForServerReady:
    def test_accepts_when_port_is_open(self):
        """Server that accepts immediately should pass readiness."""
        # Bind a real socket so _wait_for_server_ready can connect.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            _wait_for_server_ready("127.0.0.1", port, timeout=2.0)

    def test_raises_timeout_when_port_closed(self):
        port = _find_free_port()
        with pytest.raises(TimeoutError, match="did not accept connections"):
            _wait_for_server_ready("127.0.0.1", port, timeout=0.2, interval=0.05)


# ---------------------------------------------------------------------------
# EphemeralMcpHandle
# ---------------------------------------------------------------------------


class TestEphemeralMcpHandle:
    def test_fields_accessible(self):
        h = EphemeralMcpHandle(
            server_url="http://127.0.0.1:9999/mcp",
            token="tok",
            server_name="bili_tools_abc",
        )
        assert h.server_url == "http://127.0.0.1:9999/mcp"
        assert h.token == "tok"
        assert h.server_name == "bili_tools_abc"


# ---------------------------------------------------------------------------
# EphemeralMcpServer
# ---------------------------------------------------------------------------


class _FakeUvicornServer:
    """Minimal fake uvicorn.Server that accepts connections after a short delay."""

    def __init__(self, config):
        self.config = config
        self.should_exit = False
        self._port = config.port if hasattr(config, "port") else None

    async def serve(self):
        if self._port:
            # Bind to trigger _wait_for_server_ready to pass.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", self._port))
                s.listen(1)
                while not self.should_exit:
                    time.sleep(0.02)


class TestEphemeralMcpServer:
    """Tests for EphemeralMcpServer lifecycle (mocked uvicorn + FastMCP)."""

    def _patch_deps(self, fake_port: int = 9876):
        """Return a context-manager-compatible patch dict."""

        fake_fmcp = MagicMock()
        fake_fmcp.streamable_http_app.return_value = MagicMock()

        class FakeUvicornConfig:
            def __init__(self, app, host, port, **kwargs):
                self.app = app
                self.host = host
                self.port = port

        fake_server = _FakeUvicornServer(
            FakeUvicornConfig(None, "127.0.0.1", fake_port)
        )

        return {
            "FastMCP": fake_fmcp,
            "UvicornServer": fake_server,
            "UvicornConfig": FakeUvicornConfig,
        }

    def test_enter_returns_handle(self):
        tool = _make_plain_tool()
        port = _find_free_port()
        patches = self._patch_deps(port)

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch(
                "bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]
            ) as mock_fastmcp_cls,
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
        ):
            mock_uv.Config = patches["UvicornConfig"]
            mock_uv.Server = lambda cfg: patches["UvicornServer"]

            with EphemeralMcpServer([tool]) as handle:
                assert isinstance(handle, EphemeralMcpHandle)
                assert f"127.0.0.1:{port}" in handle.server_url
                assert handle.server_url.endswith("/mcp")
                assert len(handle.token) > 20
                assert "bili_tools_" in handle.server_name

    def test_tool_registered_on_fastmcp(self):
        tool = _make_plain_tool(name="search")
        port = _find_free_port()
        patches = self._patch_deps(port)

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch("bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]),
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
        ):
            mock_uv.Config = patches["UvicornConfig"]
            mock_uv.Server = lambda cfg: patches["UvicornServer"]

            with EphemeralMcpServer([tool]) as _:
                patches["FastMCP"].add_tool.assert_called_once()
                call_args = patches["FastMCP"].add_tool.call_args
                assert call_args.kwargs.get("name") == "search" or (
                    len(call_args.args) > 0 and call_args.kwargs.get("name") == "search"
                )

    def test_exit_signals_shutdown(self):
        tool = _make_plain_tool()
        port = _find_free_port()
        patches = self._patch_deps(port)

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch("bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]),
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
        ):
            mock_uv.Config = patches["UvicornConfig"]
            fake_server = patches["UvicornServer"]
            mock_uv.Server = lambda cfg: fake_server

            with EphemeralMcpServer([tool]):
                pass  # __exit__ called here

            assert fake_server.should_exit is True

    def test_exit_called_on_exception(self):
        """__exit__ must run (signal shutdown) even when the body raises."""
        tool = _make_plain_tool()
        port = _find_free_port()
        patches = self._patch_deps(port)

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch("bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]),
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
        ):
            mock_uv.Config = patches["UvicornConfig"]
            fake_server = patches["UvicornServer"]
            mock_uv.Server = lambda cfg: fake_server

            with pytest.raises(RuntimeError, match="test error"):
                with EphemeralMcpServer([tool]):
                    raise RuntimeError("test error")

            assert fake_server.should_exit is True

    def test_fastmcp_bound_to_localhost(self):
        """FastMCP must be initialised with host='127.0.0.1'."""
        tool = _make_plain_tool()
        port = _find_free_port()
        patches = self._patch_deps(port)
        captured_kwargs = {}

        def _capture_fastmcp(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return patches["FastMCP"]

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch("bili.iris.mcp.server.FastMCP", side_effect=_capture_fastmcp),
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
        ):
            mock_uv.Config = patches["UvicornConfig"]
            mock_uv.Server = lambda cfg: patches["UvicornServer"]

            with EphemeralMcpServer([tool]):
                pass

        assert captured_kwargs.get("host") == "127.0.0.1"

    def test_import_error_when_mcp_missing(self):
        # _MCP_AVAILABLE is set at module load; simulate unavailability by patching the flag.
        with patch("bili.iris.mcp.server._MCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="bili-core\\[mcp\\]"):
                server = EphemeralMcpServer([_make_plain_tool()])
                server.__enter__()

    def test_startup_timeout_calls_stop_and_reraises(self):
        """A TimeoutError from _wait_for_server_ready must stop the server and re-raise."""
        tool = _make_plain_tool()
        port = _find_free_port()
        patches = self._patch_deps(port)

        with (
            patch("bili.iris.mcp.server._find_free_port", return_value=port),
            patch("bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]),
            patch("bili.iris.mcp.server.uvicorn") as mock_uv,
            patch(
                "bili.iris.mcp.server._wait_for_server_ready",
                side_effect=TimeoutError("server did not start"),
            ),
        ):
            mock_uv.Config = patches["UvicornConfig"]
            fake_server = patches["UvicornServer"]
            mock_uv.Server = lambda cfg: fake_server

            with pytest.raises(TimeoutError):
                s = EphemeralMcpServer([tool])
                s.__enter__()

            # should_exit must be set so uvicorn cleans up.
            assert fake_server.should_exit is True

    def test_token_differs_per_instance(self):
        """Each instance must generate a unique token (auth isolation)."""
        port1 = _find_free_port()
        port2 = _find_free_port()
        patches1 = self._patch_deps(port1)
        patches2 = self._patch_deps(port2)
        tokens = []

        for port, patches in [(port1, patches1), (port2, patches2)]:
            with (
                patch("bili.iris.mcp.server._find_free_port", return_value=port),
                patch("bili.iris.mcp.server.FastMCP", return_value=patches["FastMCP"]),
                patch("bili.iris.mcp.server.uvicorn") as mock_uv,
            ):
                mock_uv.Config = patches["UvicornConfig"]
                mock_uv.Server = lambda cfg: patches["UvicornServer"]
                with EphemeralMcpServer([_make_plain_tool()]) as handle:
                    tokens.append(handle.token)

        assert tokens[0] != tokens[1], "Tokens must be unique per call"


# ---------------------------------------------------------------------------
# _build_tool_preamble
# ---------------------------------------------------------------------------


class TestBuildToolPreamble:
    def test_empty_tools_returns_empty_string(self):
        assert _build_tool_preamble([], "srv") == ""

    def test_contains_server_name(self):
        tool = _make_plain_tool("search", "Searches things")
        preamble = _build_tool_preamble([tool], "bili_tools_abc")
        assert "bili_tools_abc" in preamble

    def test_contains_tool_name_and_description(self):
        tool = _make_plain_tool("my_search", "Searches the web")
        preamble = _build_tool_preamble([tool], "srv")
        assert "my_search" in preamble
        assert "Searches the web" in preamble

    def test_multiple_tools_listed(self):
        tools = [_make_plain_tool(f"tool_{i}", f"Desc {i}") for i in range(3)]
        preamble = _build_tool_preamble(tools, "srv")
        for i in range(3):
            assert f"tool_{i}" in preamble


# ---------------------------------------------------------------------------
# _parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    def test_text_format_strips_whitespace(self):
        result = _parse_output("  hello  \n", "text", "content", "cli")
        assert result == "hello"

    def test_json_format_extracts_path(self):
        raw = json.dumps({"content": "extracted"})
        result = _parse_output(raw, "json", "content", "cli")
        assert result == "extracted"

    def test_json_format_nested_path(self):
        raw = json.dumps({"a": {"b": "deep"}})
        result = _parse_output(raw, "json", "a.b", "cli")
        assert result == "deep"

    def test_json_invalid_raises_cli_error(self):
        from bili.iris.providers.cli_provider import CliLLMError

        with pytest.raises(CliLLMError, match="not valid JSON"):
            _parse_output("not json", "json", "content", "cli")

    def test_json_missing_path_raises_cli_error(self):
        from bili.iris.providers.cli_provider import CliLLMError

        raw = json.dumps({"other": "value"})
        with pytest.raises(CliLLMError, match="not found"):
            _parse_output(raw, "json", "missing.key", "cli")


# ---------------------------------------------------------------------------
# build_mcp_node
# ---------------------------------------------------------------------------


class TestBuildMcpNode:
    """Tests for the build_mcp_node factory and the node callable it returns."""

    def _make_cli_llm(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        command=None,
        cwd=None,
        model=None,
        reasoning_effort=None,
        model_flag_template=None,
        reasoning_effort_flag_template=None,
    ):
        llm = MagicMock()
        llm.command = command or ["claude", "-p"]
        llm.message_format = "last"
        llm.output_format = "text"
        llm.json_path = "content"
        llm.strip_ansi_output = False
        llm.timeout_seconds = 30.0
        llm.cwd = cwd
        llm.model = model
        llm.reasoning_effort = reasoning_effort
        llm.model_flag_template = model_flag_template
        llm.reasoning_effort_flag_template = reasoning_effort_flag_template
        return llm

    def _make_injector(self, extra_env=None, cleanup=None, cwd=None):
        from bili.iris.mcp.cli_injectors import _GEMINI_CWD_KEY, InjectionResult

        env = dict(extra_env or {})
        if cwd:
            env[_GEMINI_CWD_KEY] = cwd

        result = InjectionResult(
            augmented_command=["claude", "-p", "--mcp-config", "/tmp/x.json"],
            extra_env=env,
            cleanup=cleanup,
        )
        injector = MagicMock()
        injector.inject.return_value = result
        return injector

    def _make_state(self, text: str = "hello"):
        return {"messages": [HumanMessage(content=text)]}

    def _run_node_with_mocks(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        tool=None,
        injector=None,
        stdout="CLI response",
        returncode=0,
        extra_env=None,
        cwd=None,
        llm_cwd=None,
        model=None,
        reasoning_effort=None,
        model_flag_template=None,
        reasoning_effort_flag_template=None,
    ):
        """Patch EphemeralMcpServer and subprocess to run the node callable.

        :param cwd: Injector-side cwd sentinel (e.g. the Gemini injector's
            temp-dir cwd requirement), forwarded via the injector's extra env.
        :param llm_cwd: The CliLLM's own configured ``cwd`` attribute --
            simulates a caller-configured subprocess working directory.
        :param model: The CliLLM's own configured ``model`` attribute.
        :param reasoning_effort: The CliLLM's own configured
            ``reasoning_effort`` attribute.
        :param model_flag_template: The CliLLM's own configured
            ``model_flag_template`` attribute.
        :param reasoning_effort_flag_template: The CliLLM's own configured
            ``reasoning_effort_flag_template`` attribute.
        """
        tool = tool or _make_plain_tool()
        llm = self._make_cli_llm(
            cwd=llm_cwd,
            model=model,
            reasoning_effort=reasoning_effort,
            model_flag_template=model_flag_template,
            reasoning_effort_flag_template=reasoning_effort_flag_template,
        )
        injector = injector or self._make_injector(extra_env=extra_env, cwd=cwd)

        node = build_mcp_node(llm_model=llm, tools=[tool], injector=injector)

        mock_handle = EphemeralMcpHandle(
            server_url="http://127.0.0.1:9001/mcp",
            token="tok",
            server_name="bili_tools_abc",
        )
        mock_proc = MagicMock()
        mock_proc.returncode = returncode
        mock_proc.pid = _SPAWNED_PID
        mock_proc.communicate = MagicMock(return_value=(stdout, ""))

        with (
            patch("bili.iris.mcp.server.EphemeralMcpServer") as mock_server_cls,
            patch(
                "bili.iris.mcp.server.subprocess.Popen", return_value=mock_proc
            ) as mock_popen,
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_handle)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_server_cls.return_value = mock_ctx

            result = node(self._make_state())
            # The server object is what authorizes, so hand it back for the
            # test that pins the grant.
            mock_popen.server = mock_ctx
            mock_popen.proc = mock_proc
            return result, mock_popen, injector

    def test_node_returns_dict_with_messages(self):
        result, _, _ = self._run_node_with_mocks()
        assert "messages" in result
        assert len(result["messages"]) == 1

    def test_node_returns_ai_message(self):
        from langchain_core.messages import AIMessage

        result, _, _ = self._run_node_with_mocks(stdout="hello")
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "hello"

    def test_subprocess_called_with_augmented_command(self):
        _, mock_popen, _ = self._run_node_with_mocks()
        called_cmd = mock_popen.call_args[0][0]
        assert called_cmd == ["claude", "-p", "--mcp-config", "/tmp/x.json"]

    def test_the_spawned_process_is_authorized(self):
        """The node must grant the server the PID it just spawned.

        Nothing else in this class can see this: the server is mocked here, so
        a node that spawns the CLI and never authorizes it produces identical
        output and every other assertion still holds.  Without this the whole
        path ships denying every tool call the CLI makes.
        """
        _, mock_popen, _ = self._run_node_with_mocks()
        mock_popen.server.authorize_subprocess.assert_called_once_with(_SPAWNED_PID)

    def test_authorization_precedes_writing_the_prompt(self):
        """The grant must be in place before the CLI can act on the prompt.

        The CLI may connect as soon as it starts, so a grant that lands after
        communicate() returns arrives after every tool call the run makes.
        The ordering is observed from inside communicate() rather than read
        off either mock's call list: the two calls are on different objects,
        so neither list records the relationship, and an assertion over one of
        them stays green with the grant moved after the call.
        """
        tool = _make_plain_tool()
        llm = self._make_cli_llm()
        injector = self._make_injector()
        node = build_mcp_node(llm_model=llm, tools=[tool], injector=injector)

        mock_handle = EphemeralMcpHandle("http://127.0.0.1:9001/mcp", "tok", "srv")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_handle)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        seen = {}

        def _communicate(**_kwargs):
            seen["authorized_first"] = mock_ctx.authorize_subprocess.called
            return ("out", "")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = _SPAWNED_PID
        mock_proc.communicate = MagicMock(side_effect=_communicate)

        with (
            patch("bili.iris.mcp.server.EphemeralMcpServer", return_value=mock_ctx),
            patch("bili.iris.mcp.server.subprocess.Popen", return_value=mock_proc),
        ):
            node(self._make_state())

        assert seen.get("authorized_first") is True, (
            "the prompt was written to the CLI before the server was told which "
            "process to serve"
        )

    def test_injector_inject_called_with_handle(self):
        _, _, injector = self._run_node_with_mocks()
        injector.inject.assert_called_once()
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["handle"].server_url == "http://127.0.0.1:9001/mcp"

    def test_cleanup_called_after_subprocess(self):
        cleanup = MagicMock()
        injector = self._make_injector(cleanup=cleanup)
        self._run_node_with_mocks(injector=injector)
        cleanup.assert_called_once()

    def test_cleanup_called_on_subprocess_error(self):
        """Cleanup must run even when the subprocess exits non-zero."""
        from bili.iris.providers.cli_provider import CliLLMError

        cleanup = MagicMock()
        injector = self._make_injector(cleanup=cleanup)
        with pytest.raises(CliLLMError):
            self._run_node_with_mocks(injector=injector, returncode=1)
        cleanup.assert_called_once()

    def test_subprocess_passed_extra_env(self):
        _, mock_popen, _ = self._run_node_with_mocks(extra_env={"MY_KEY": "MY_VAL"})
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["env"]["MY_KEY"] == "MY_VAL"

    def test_gemini_cwd_extracted_from_env(self):
        """Gemini injector's CWD sentinel must be extracted and used as cwd kwarg."""
        from bili.iris.mcp.cli_injectors import _GEMINI_CWD_KEY

        _, mock_popen, _ = self._run_node_with_mocks(cwd="/tmp/gemini_work")
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("cwd") == "/tmp/gemini_work"
        # Sentinel must NOT appear in the forwarded env.
        assert _GEMINI_CWD_KEY not in call_kwargs.get("env", {})

    def test_configured_llm_cwd_forwarded_to_subprocess(self):
        """A CliLLM.cwd configured on the model must reach the MCP subprocess.

        Regression test: build_mcp_node previously ignored llm_model.cwd
        entirely, so a configured working directory (e.g. for claude/codex,
        which have no cwd-sentinel injector) was silently dropped and the
        subprocess inherited the caller's cwd instead.
        """
        _, mock_popen, _ = self._run_node_with_mocks(llm_cwd="/fixed/workspace")
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("cwd") == "/fixed/workspace"

    def test_configured_llm_cwd_takes_precedence_over_gemini_sentinel(self):
        """An explicit CliLLM.cwd wins over the injector's own cwd sentinel.

        The Gemini injector requests a temp-dir cwd so it can pick up its
        generated project-scoped settings file, but a caller-configured
        CliLLM.cwd is an isolation boundary and must not be silently
        overridden by injector plumbing.
        """
        _, mock_popen, _ = self._run_node_with_mocks(
            cwd="/tmp/gemini_work", llm_cwd="/fixed/workspace"
        )
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("cwd") == "/fixed/workspace"

    def test_no_cwd_configured_leaves_subprocess_cwd_none(self):
        """With no configured cwd and no injector sentinel, cwd stays None.

        None means the subprocess inherits the calling process's cwd,
        matching subprocess.run's own default and the direct CLI execution
        path's default.
        """
        _, mock_popen, _ = self._run_node_with_mocks()
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("cwd") is None

    def test_configured_model_reaches_injector_command(self):
        """A CliLLM.model configured on the model must reach the base command
        handed to the injector, before MCP flags are appended.

        Regression coverage for the same dual-path gotcha the cwd fix (#236)
        had: model/reasoning-effort flags must be applied on the MCP
        tool-strategy path (build_mcp_node), not only on the direct
        _run_subprocess path.
        """
        _, _, injector = self._run_node_with_mocks(
            model="claude-sonnet-5",
            model_flag_template=["--model", "{value}"],
        )
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["command"] == ["claude", "-p", "--model", "claude-sonnet-5"]

    def test_configured_reasoning_effort_reaches_injector_command(self):
        """A CliLLM.reasoning_effort configured on the model must reach the
        base command handed to the injector, before MCP flags are appended."""
        _, _, injector = self._run_node_with_mocks(
            reasoning_effort="high",
            reasoning_effort_flag_template=["--effort", "{value}"],
        )
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["command"] == ["claude", "-p", "--effort", "high"]

    def test_configured_model_and_reasoning_effort_both_reach_injector_command(self):
        """Both model and reasoning_effort flags reach the injector's base
        command, model first, matching the direct-path ordering."""
        _, _, injector = self._run_node_with_mocks(
            model="claude-sonnet-5",
            model_flag_template=["--model", "{value}"],
            reasoning_effort="high",
            reasoning_effort_flag_template=["--effort", "{value}"],
        )
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["command"] == [
            "claude",
            "-p",
            "--model",
            "claude-sonnet-5",
            "--effort",
            "high",
        ]

    def test_no_model_or_reasoning_effort_configured_leaves_command_unchanged(self):
        """With neither model nor reasoning_effort set, the base command
        handed to the injector is unchanged -- today's behaviour, unchanged."""
        _, _, injector = self._run_node_with_mocks()
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["command"] == ["claude", "-p"]

    def test_model_set_with_no_template_omits_flag_on_mcp_path(self):
        """model set with model_flag_template=None is a documented no-op on
        the MCP path too: no extra flag is added to the injector's command."""
        _, _, injector = self._run_node_with_mocks(
            model="claude-sonnet-5", model_flag_template=None
        )
        call_kwargs = injector.inject.call_args.kwargs
        assert call_kwargs["command"] == ["claude", "-p"]

    def test_configured_model_flags_reach_final_subprocess_command_when_injector_passthrough(
        self,
    ):
        """End-to-end: when the injector passes the base command through
        unchanged (appending only its own flags), the final subprocess.run
        call carries the configured model flags too."""
        from bili.iris.mcp.cli_injectors import InjectionResult

        def _passthrough_inject(command, handle):  # pylint: disable=unused-argument
            return InjectionResult(
                augmented_command=list(command) + ["--mcp-config", "/tmp/x.json"],
                extra_env={},
                cleanup=None,
            )

        injector = MagicMock()
        injector.inject.side_effect = _passthrough_inject

        _, mock_popen, _ = self._run_node_with_mocks(
            injector=injector,
            model="claude-sonnet-5",
            model_flag_template=["--model", "{value}"],
        )
        called_cmd = mock_popen.call_args[0][0]
        assert called_cmd == [
            "claude",
            "-p",
            "--model",
            "claude-sonnet-5",
            "--mcp-config",
            "/tmp/x.json",
        ]

    def test_nonzero_exit_raises_cli_error(self):
        from bili.iris.providers.cli_provider import CliLLMError

        with pytest.raises(CliLLMError, match="exited with code 1"):
            self._run_node_with_mocks(returncode=1)

    def test_tool_preamble_prepended_to_prompt(self):
        """The stdin passed to subprocess must include the tool preamble."""
        _, mock_popen, _ = self._run_node_with_mocks()
        stdin_sent = mock_popen.proc.communicate.call_args[1].get("input", "")
        assert "bili_tools_abc" in stdin_sent  # preamble contains server name

    def test_subprocess_timeout_raises_cli_error(self):
        """A subprocess.TimeoutExpired must be re-raised as CliLLMError."""
        import subprocess as _subprocess

        from bili.iris.providers.cli_provider import CliLLMError

        tool = _make_plain_tool()
        llm = self._make_cli_llm()
        injector = self._make_injector()
        node = build_mcp_node(llm_model=llm, tools=[tool], injector=injector)

        mock_handle = EphemeralMcpHandle("http://127.0.0.1:9001/mcp", "tok", "srv")
        # The timeout surfaces from communicate(), not from the spawn.
        mock_proc = MagicMock()
        mock_proc.pid = _SPAWNED_PID
        mock_proc.communicate = MagicMock(
            side_effect=[
                _subprocess.TimeoutExpired(cmd="claude", timeout=30),
                ("", ""),  # the post-kill reap
            ]
        )
        with (
            patch("bili.iris.mcp.server.EphemeralMcpServer") as mock_srv,
            patch("bili.iris.mcp.server.subprocess.Popen", return_value=mock_proc),
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_handle)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_srv.return_value = mock_ctx

            with pytest.raises(CliLLMError, match="timed out"):
                node(self._make_state())

        # subprocess.run kills the child on timeout; Popen.communicate does
        # not, so a timed-out CLI would outlive the server serving it.
        mock_proc.kill.assert_called_once()

    def test_ansi_stripping_applied_when_enabled(self):
        """With strip_ansi_output=True the ANSI escape codes are removed."""
        tool = _make_plain_tool()
        llm = self._make_cli_llm()
        llm.strip_ansi_output = True
        injector = self._make_injector()
        node = build_mcp_node(llm_model=llm, tools=[tool], injector=injector)

        ansi_output = "\x1b[32mGreen text\x1b[0m"
        mock_handle = EphemeralMcpHandle("http://127.0.0.1:9001/mcp", "tok", "srv")
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = _SPAWNED_PID
        mock_proc.communicate = MagicMock(return_value=(ansi_output, ""))

        with (
            patch("bili.iris.mcp.server.EphemeralMcpServer") as mock_srv,
            patch("bili.iris.mcp.server.subprocess.Popen", return_value=mock_proc),
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_handle)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_srv.return_value = mock_ctx

            result = node(self._make_state())

        assert result["messages"][0].content == "Green text"

    def test_empty_messages_returns_error_ai_message(self):
        """Empty message list falls back gracefully."""
        from langchain_core.messages import AIMessage

        tool = _make_plain_tool()
        llm = self._make_cli_llm()
        injector = self._make_injector()
        node = build_mcp_node(llm_model=llm, tools=[tool], injector=injector)

        mock_handle = EphemeralMcpHandle("http://127.0.0.1:9001/mcp", "tok", "srv")
        with (
            patch("bili.iris.mcp.server.EphemeralMcpServer") as mock_srv,
            patch("bili.iris.mcp.server.subprocess.Popen"),
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_handle)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_srv.return_value = mock_ctx

            result = node({"messages": []})

        assert isinstance(result["messages"][0], AIMessage)
        assert "Error" in result["messages"][0].content


# ---------------------------------------------------------------------------
# resolve_mcp_injector
# ---------------------------------------------------------------------------


class TestResolveMcpInjector:
    def test_cli_llm_with_claude_command_resolves_injector(self):
        llm = MagicMock()
        llm.command = ["claude", "-p"]
        injector = resolve_mcp_injector(llm)
        from bili.iris.mcp.cli_injectors import ClaudeCodeInjector

        assert isinstance(injector, ClaudeCodeInjector)

    def test_cli_llm_with_codex_command_resolves_injector(self):
        llm = MagicMock()
        llm.command = ["codex", "exec"]
        injector = resolve_mcp_injector(llm)
        from bili.iris.mcp.cli_injectors import CodexInjector

        assert isinstance(injector, CodexInjector)

    def test_cli_llm_with_gemini_command_resolves_injector(self):
        llm = MagicMock()
        llm.command = ["gemini", "-p"]
        injector = resolve_mcp_injector(llm)
        from bili.iris.mcp.cli_injectors import GeminiCliInjector

        assert isinstance(injector, GeminiCliInjector)

    def test_unknown_cli_returns_none(self):
        llm = MagicMock()
        llm.command = ["unknown-llm-cli"]
        assert resolve_mcp_injector(llm) is None

    def test_model_without_command_attribute_returns_none(self):
        llm = MagicMock(spec=[])  # no 'command' attr
        assert resolve_mcp_injector(llm) is None

    def test_model_with_non_list_command_returns_none(self):
        llm = MagicMock()
        llm.command = "claude"  # string, not list
        assert resolve_mcp_injector(llm) is None

    def test_full_path_command_resolves_by_basename(self):
        llm = MagicMock()
        llm.command = ["/usr/local/bin/claude", "-p"]
        injector = resolve_mcp_injector(llm)
        from bili.iris.mcp.cli_injectors import ClaudeCodeInjector

        assert isinstance(injector, ClaudeCodeInjector)
