"""Ephemeral MCP server — serves an agent's LangChain tools as an MCP server.

This module is the *inverse* of the MCP client subsystem in
:mod:`bili.iris.mcp.adapters.tool_adapter` (which converts MCP tools into
LangChain tools for an IRIS agent to call).  Here the direction is reversed:
a set of LangChain :class:`~langchain_core.tools.BaseTool` objects registered
on an IRIS agent are exposed as an MCP server so that an MCP-capable CLI model
(Claude Code, Codex, Gemini CLI) can call them via its own native tool-calling
interface.

Architecture overview
---------------------
::

    bili-core process
    ┌─────────────────────────────────────────────────────┐
    │  IRIS agent node                                     │
    │    tools: [tool_a, tool_b, ...]  ───────────────┐   │
    │                                                  │   │
    │  EphemeralMcpServer (FastMCP + auth middleware)  │   │
    │    ─ registers each LangChain tool as MCP tool ◄─┘   │
    │    ─ listens on 127.0.0.1:<port> (Streamable HTTP)    │
    │    ─ enforces per-call Bearer-token auth              │
    │    ─ background thread (uvicorn + asyncio.run)        │
    │                                                  │   │
    │  CLI subprocess (claude / codex / gemini)        │   │
    │    ─ spawned with MCP config pointing at server  │   │
    │    ─ calls tools via MCP protocol                │   │
    │    ─ self-orchestrates; returns final answer     │   │
    └─────────────────────────────────────────────────────┘

Security model
--------------
Each :class:`EphemeralMcpServer` instance generates a cryptographically-random
per-call Bearer token (via :func:`secrets.token_urlsafe`).  Every HTTP request
to the server is validated against this token by :class:`_TokenAuthMiddleware`
before being forwarded to the FastMCP application.  Requests without the
correct ``Authorization: Bearer <token>`` header receive a ``401 Unauthorized``
response and the connection is dropped.

The token is handed to the :mod:`~bili.iris.mcp.cli_injectors` module, which
embeds it in the MCP configuration written for the CLI subprocess.  The
subprocess is therefore the **only** process that can authenticate to the
ephemeral server — no other local process knows the token.

The server binds to ``127.0.0.1`` (IPv4 loopback) only.  FastMCP
automatically enables DNS-rebinding protection for localhost hosts, blocking
requests with non-localhost ``Host:`` headers.  Together, these controls mean
the server is not reachable from any external host or via a DNS-rebinding
attack.

Lazy imports
------------
All ``mcp`` and ``uvicorn`` imports are deferred to the connection path so
this module can be imported without those packages installed.  The
``[mcp]`` extra installs both.

:class:`EphemeralMcpServer` is a **synchronous** context manager.  Its
internal server runs in a background daemon thread (uvicorn + asyncio.run).
The calling thread (IRIS node) remains free to run the CLI subprocess.

Typical usage
-------------
::

    from bili.iris.mcp.server import EphemeralMcpServer, build_mcp_node
    from bili.iris.mcp.cli_injectors import get_injector

    injector = get_injector("claude")
    node_callable = build_mcp_node(llm_model=cli_llm, tools=tools, injector=injector)
    # node_callable is a (state: dict) -> dict callable suitable for LangGraph.
"""

import inspect
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — deferred so importing this module does not
# crash when the [mcp] extra is not installed.  The names are assigned at
# module level so that unit tests can patch them via
# ``patch("bili.iris.mcp.server.FastMCP", ...)`` and
# ``patch("bili.iris.mcp.server.uvicorn", ...)``.
# ---------------------------------------------------------------------------

try:
    import uvicorn  # type: ignore[import-untyped]
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover — mcp/uvicorn not installed; gate is [mcp] extra
    uvicorn = None  # type: ignore[assignment]
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default readiness-poll timeout in seconds.
_DEFAULT_READY_TIMEOUT: float = 5.0

#: Readiness-poll interval in seconds.
_POLL_INTERVAL: float = 0.05

#: Graceful shutdown join timeout in seconds.
_SHUTDOWN_JOIN_TIMEOUT: float = 5.0

#: ANSI escape-sequence pattern (reused from cli_provider logic).
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


# ---------------------------------------------------------------------------
# ASGI auth middleware
# ---------------------------------------------------------------------------


class _TokenAuthMiddleware:  # pylint: disable=too-few-public-methods
    """ASGI middleware that enforces per-call ephemeral Bearer-token auth.

    Every HTTP request to the wrapped application must carry the header::

        Authorization: Bearer <token>

    where ``<token>`` is the cryptographically-random value generated by
    :class:`EphemeralMcpServer` for this call.  Requests without the header
    or with a wrong token receive a ``401 Unauthorized`` JSON response and are
    not forwarded to the inner app.

    Non-HTTP scope types (``"lifespan"``, ``"websocket"``, etc.) are passed
    through unchanged so uvicorn's startup/shutdown lifecycle works correctly.

    :param app: The inner ASGI application (the FastMCP Starlette app).
    :param token: The per-call secret Bearer token.
    """

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._token = token
        self._bearer_value: bytes = f"Bearer {token}".encode()

    async def __call__(self, scope: Dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            # Lifespan events and other non-HTTP scopes pass through.
            await self._app(scope, receive, send)
            return

        # Extract the Authorization header (bytes key/value pairs in ASGI).
        auth_value: bytes = b""
        for name, value in scope.get("headers", ()):
            if name.lower() == b"authorization":
                auth_value = value
                break

        if secrets.compare_digest(auth_value, self._bearer_value):
            await self._app(scope, receive, send)
            return

        # Reject: 401 Unauthorized.
        LOGGER.debug(
            "EphemeralMcpServer: rejected request without valid Bearer token "
            "(path=%s)",
            scope.get("path", "?"),
        )
        body = b'{"error":"Unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


# ---------------------------------------------------------------------------
# Port allocation helper
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to a random OS-assigned port on 127.0.0.1, read it, and release it.

    There is a small TOCTOU window between releasing the socket and uvicorn
    binding it, but this is negligible for local ephemeral servers — the port
    space is large and the caller does not expose the port to the network.

    :returns: An available TCP port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Readiness poll
# ---------------------------------------------------------------------------


def _wait_for_server_ready(
    host: str,
    port: int,
    timeout: float = _DEFAULT_READY_TIMEOUT,
    interval: float = _POLL_INTERVAL,
) -> None:
    """Poll until the server accepts TCP connections or *timeout* expires.

    Uses a raw TCP connect rather than an HTTP request so auth is not
    exercised during the readiness check — the middleware passes lifespan
    events before any request is validated.

    :param host: Server host (always ``"127.0.0.1"``).
    :param port: Server port.
    :param timeout: Maximum seconds to wait.
    :param interval: Seconds between poll attempts.
    :raises TimeoutError: If the server does not accept connections within
        *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(interval)
    raise TimeoutError(
        f"Ephemeral MCP server did not accept connections on {host}:{port} "
        f"within {timeout}s"
    )


# ---------------------------------------------------------------------------
# LangChain tool → FastMCP function bridge
# ---------------------------------------------------------------------------


def _build_mcp_fn(tool: Any) -> Callable:
    """Build a FastMCP-compatible function from a LangChain :class:`BaseTool`.

    FastMCP infers the MCP tool's JSON schema from the function's type
    annotations via :func:`inspect.signature`.  This function dynamically
    constructs a wrapper with the right signature so that FastMCP generates
    an accurate input schema for the tool.

    Two cases:

    1. **Structured tool** (has ``args_schema`` with Pydantic ``model_fields``):
       build a function whose :class:`~inspect.Parameter` list mirrors the
       schema's fields.  The body calls ``tool.invoke(kwargs)``.

    2. **Plain tool** (no ``args_schema``, or schema without ``model_fields``):
       fall back to a single ``tool_input: str`` parameter.  The body calls
       ``tool.invoke(tool_input)``.

    In both cases the function's ``__name__`` is set to ``tool.name`` so
    FastMCP registers it under the correct tool name.

    :param tool: A LangChain :class:`~langchain_core.tools.BaseTool`.
    :returns: A callable suitable for ``FastMCP.add_tool(fn, ...)``.
    """
    args_schema = getattr(tool, "args_schema", None)
    model_fields: Optional[Dict] = (
        getattr(args_schema, "model_fields", None) if args_schema is not None else None
    )

    if not model_fields:
        # Plain single-string-input tool.
        def _plain_fn(tool_input: str) -> str:
            return str(tool.invoke(tool_input))

        _plain_fn.__name__ = tool.name
        _plain_fn.__doc__ = tool.description or ""
        return _plain_fn

    # Build parameters from Pydantic model fields.
    params: List[inspect.Parameter] = []
    annotations: Dict[str, Any] = {"return": str}

    for field_name, field_info in model_fields.items():
        python_type = getattr(field_info, "annotation", None) or Any
        annotations[field_name] = python_type

        is_required = field_info.is_required()
        default = inspect.Parameter.empty if is_required else field_info.default
        params.append(
            inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=python_type,
            )
        )

    def _structured_fn(**kwargs: Any) -> str:
        return str(tool.invoke(kwargs))

    _structured_fn.__signature__ = inspect.Signature(
        parameters=params, return_annotation=str
    )
    _structured_fn.__annotations__ = annotations
    _structured_fn.__name__ = tool.name
    _structured_fn.__doc__ = tool.description or ""
    return _structured_fn


# ---------------------------------------------------------------------------
# EphemeralMcpHandle
# ---------------------------------------------------------------------------


@dataclass
class EphemeralMcpHandle:
    """Handle returned by :class:`EphemeralMcpServer.__enter__`.

    Carries the information that CLI injectors need to configure the CLI
    subprocess to connect to the ephemeral server.

    :param server_url: The MCP endpoint URL
        (``http://127.0.0.1:<port>/mcp`` for Streamable HTTP transport).
        Clients must include ``Authorization: Bearer <token>`` on every request.
    :param token: The cryptographically-random per-call Bearer token.
    :param server_name: The FastMCP server name used for this call.
    """

    server_url: str
    token: str
    server_name: str


# ---------------------------------------------------------------------------
# EphemeralMcpServer
# ---------------------------------------------------------------------------


class EphemeralMcpServer:  # pylint: disable=too-many-instance-attributes
    """Synchronous context manager that serves LangChain tools as an MCP server.

    On entry:

    1. Generates a cryptographically-random per-call Bearer token
       (:func:`secrets.token_urlsafe`).
    2. Allocates a free localhost port (:func:`_find_free_port`).
    3. Builds a FastMCP Streamable HTTP application and registers each tool via
       :func:`_build_mcp_fn`.  (Streamable HTTP — ``POST /mcp`` — is the
       current MCP transport standard; SSE is deprecated as of MCP spec
       2025-03-26.)
    4. Wraps the app in :class:`_TokenAuthMiddleware` to enforce per-call auth.
    5. Starts uvicorn in a background daemon thread (``asyncio.run`` in the
       thread creates a dedicated event loop — no interference with the caller's
       loop, if any).
    6. Polls until the server accepts TCP connections, then returns an
       :class:`EphemeralMcpHandle`.

    On exit (even on error):

    7. Sets ``uvicorn.Server.should_exit = True`` to signal shutdown.
    8. Joins the background thread with a :data:`_SHUTDOWN_JOIN_TIMEOUT` grace
       period.  The thread is a daemon, so the process does not hang if it
       fails to exit promptly.

    :param tools: LangChain tools to expose as MCP tools.
    :param server_name: Base name for the FastMCP server.  A short random
        suffix is appended per call to avoid collisions (e.g.
        ``bili_tools_3f7a``).  Defaults to ``"bili_tools"``.
    :param ready_timeout: Seconds to wait for the server to accept connections.
    :raises ImportError: If ``mcp`` or ``uvicorn`` is not installed
        (install with ``pip install bili-core[mcp]``).
    :raises TimeoutError: If the server does not accept connections within
        *ready_timeout* seconds.
    """

    def __init__(
        self,
        tools: List[Any],
        server_name: str = "bili_tools",
        ready_timeout: float = _DEFAULT_READY_TIMEOUT,
    ) -> None:
        self._tools = tools
        self._base_name = server_name
        self._ready_timeout = ready_timeout
        # Set in __enter__:
        self._uvicorn_server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._host = "127.0.0.1"
        self._port: Optional[int] = None
        self._token: Optional[str] = None
        self._call_name: Optional[str] = None

    def __enter__(self) -> EphemeralMcpHandle:
        if not _MCP_AVAILABLE:
            raise ImportError(
                "The 'mcp' and 'uvicorn' packages are required for the ephemeral "
                "MCP server.  Install with: pip install bili-core[mcp]"
            )

        # Per-call unique name (avoids collisions if two calls run concurrently).
        call_suffix = secrets.token_hex(4)
        self._call_name = f"{self._base_name}_{call_suffix}"

        # Cryptographically-random per-call auth token.
        self._token = secrets.token_urlsafe(32)

        # Allocate a free port.
        self._port = _find_free_port()

        LOGGER.debug(
            "EphemeralMcpServer '%s': starting on 127.0.0.1:%d",
            self._call_name,
            self._port,
        )

        # Build the FastMCP app and register tools.
        fmcp = FastMCP(
            self._call_name,
            host=self._host,
            port=self._port,
        )
        for tool in self._tools:
            mcp_fn = _build_mcp_fn(tool)
            fmcp.add_tool(mcp_fn, name=tool.name, description=tool.description or "")
            LOGGER.debug("EphemeralMcpServer: registered tool '%s'", tool.name)

        # Wrap the Streamable HTTP Starlette app with auth middleware.
        # Streamable HTTP (POST /mcp) is the current MCP transport standard;
        # it is supported by Claude Code ("type":"http"), Gemini CLI ("httpUrl"),
        # and Codex ("url").  SSE (/sse) is deprecated in the MCP 2025 spec.
        mcp_app = fmcp.streamable_http_app()
        auth_app = _TokenAuthMiddleware(mcp_app, self._token)

        # Build and start uvicorn server in a background daemon thread.
        config = uvicorn.Config(
            auth_app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)
        uv_server = self._uvicorn_server  # capture for thread closure

        def _run() -> None:
            import asyncio  # pylint: disable=import-outside-toplevel

            asyncio.run(uv_server.serve())

        self._thread = threading.Thread(
            target=_run, daemon=True, name=f"ephemeral-mcp-{call_suffix}"
        )
        self._thread.start()

        # Poll until the server accepts TCP connections.
        try:
            _wait_for_server_ready(self._host, self._port, timeout=self._ready_timeout)
        except TimeoutError:
            # Cleanup on startup failure.
            self._stop_server()
            raise

        LOGGER.info(
            "EphemeralMcpServer '%s': ready on 127.0.0.1:%d (%d tool(s))",
            self._call_name,
            self._port,
            len(self._tools),
        )
        return EphemeralMcpHandle(
            server_url=f"http://{self._host}:{self._port}/mcp",
            token=self._token,
            server_name=self._call_name,
        )

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        self._stop_server()
        return False  # Do not suppress exceptions.

    def _stop_server(self) -> None:
        """Signal the uvicorn server to stop and join the background thread."""
        if self._uvicorn_server is not None:
            LOGGER.debug(
                "EphemeralMcpServer '%s': signalling shutdown", self._call_name
            )
            self._uvicorn_server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_SHUTDOWN_JOIN_TIMEOUT)
            if (
                self._thread.is_alive()
            ):  # pragma: no cover — daemon exits on process end
                LOGGER.warning(
                    "EphemeralMcpServer '%s': background thread did not exit "
                    "within %ss; it is a daemon and will be cleaned up on process exit.",
                    self._call_name,
                    _SHUTDOWN_JOIN_TIMEOUT,
                )
        self._uvicorn_server = None
        self._thread = None


# ---------------------------------------------------------------------------
# Tool-description preamble builder
# ---------------------------------------------------------------------------


def _build_tool_preamble(tools: List[Any], server_name: str) -> str:
    """Return a textual description of available MCP tools for the prompt.

    Prepended to the rendered CLI prompt so the model is aware that tools
    are available via MCP and knows their names and descriptions.

    :param tools: The LangChain tools registered on the ephemeral server.
    :param server_name: The MCP server name (used in the preamble header).
    :returns: A formatted string block, or an empty string if there are no tools.
    """
    if not tools:
        return ""
    lines = [
        f"You have access to the following tools via MCP (server: {server_name}):",
        "",
    ]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description or '(no description)'}")
    lines.append("")
    lines.append("Use these tools as needed to complete the task.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# build_mcp_node — factory for the LangGraph node callable
# ---------------------------------------------------------------------------


def build_mcp_node(
    llm_model: Any,
    tools: List[Any],
    injector: Any,
) -> Callable[[Dict], Dict]:
    """Build a LangGraph node callable that serves *tools* to an MCP-capable CLI.

    The returned callable is suitable for use as a LangGraph node
    (``(state: dict) -> dict``).  On each invocation it:

    1. Starts an :class:`EphemeralMcpServer` with *tools* on a dynamically
       allocated localhost port.
    2. Calls *injector* to build the augmented CLI command and any temp
       resources (config files, env vars) needed to point the CLI at the server.
    3. Renders the conversation state into a prompt string using the LLM
       model's configured ``message_format``.
    4. Prepends a tool-description preamble to the prompt.
    5. Runs the CLI subprocess with the augmented command, the injector's
       extra environment, and the original timeout from the model config.
    6. Parses the CLI's stdout according to the model's ``output_format``.
    7. Returns a state dict update with the response as an ``AIMessage``.
    8. Cleans up the ephemeral server and any injector-created temp resources.

    Working-directory precedence
    -----------------------------
    The subprocess's working directory is resolved in this order:

    1. ``llm_model.cwd``, if set. This is a caller-controlled isolation
       boundary (e.g. pinning the CLI subprocess to a fixed sandbox
       directory rather than letting it inherit the calling process's cwd)
       and always takes precedence.
    2. The injector-provided cwd sentinel, if the CLI injector requires a
       specific working directory of its own (e.g. :class:`GeminiCliInjector`
       points the subprocess at a temp directory containing its
       project-scoped MCP settings file).
    3. ``None`` -- the subprocess inherits the calling process's cwd,
       matching ``subprocess.run``'s own default and the direct
       (:meth:`CliLLM._run_subprocess`) CLI execution path's default.

    :param llm_model: A :class:`~bili.iris.providers.cli_provider.CliLLM`
        instance (provides ``command``, ``message_format``, ``output_format``,
        ``json_path``, ``strip_ansi_output``, ``timeout_seconds``, and
        ``cwd``).
    :param tools: LangChain tools to expose as MCP tools.
    :param injector: A CLI injector from :mod:`bili.iris.mcp.cli_injectors`
        (or any object implementing ``inject(handle) -> InjectionResult``).
    :returns: A ``(state: dict) -> dict`` callable.
    :raises ImportError: If ``bili-core[mcp]`` is not installed.
    """
    # Capture the CliLLM configuration at construction time.
    command: List[str] = list(llm_model.command)
    message_format: str = getattr(llm_model, "message_format", "last")
    output_format: str = getattr(llm_model, "output_format", "text")
    json_path: str = getattr(llm_model, "json_path", "content")
    strip_ansi_flag: bool = getattr(llm_model, "strip_ansi_output", True)
    timeout_seconds: float = getattr(llm_model, "timeout_seconds", 120.0)
    configured_cwd: Optional[str] = getattr(llm_model, "cwd", None)

    # Import message rendering from cli_provider at construction time to fail fast
    # if the package is missing.
    from bili.iris.providers.cli_provider import (  # pylint: disable=import-outside-toplevel
        CliLLMError,
        render_messages,
    )

    def _node(state: Dict) -> Dict:  # pylint: disable=too-many-locals
        from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
            AIMessage,
        )

        # Render the conversation to a prompt string.
        messages = state.get("messages", [])
        try:
            prompt = render_messages(messages, message_format)
        except ValueError as exc:
            LOGGER.error("EphemeralMcpServer: failed to render messages: %s", exc)
            return {
                "messages": [AIMessage(content=f"[Error rendering messages: {exc}]")]
            }

        with EphemeralMcpServer(tools) as handle:
            # Build tool-description preamble and prepend to prompt.
            preamble = _build_tool_preamble(tools, handle.server_name)
            if preamble:
                prompt = preamble + "\n\n" + prompt

            # Ask the injector to augment the command and provide cleanup.
            injection = injector.inject(
                command=command,
                handle=handle,
            )
            augmented_cmd = injection.augmented_command
            extra_env = injection.extra_env or {}
            cleanup = injection.cleanup

            # Build the subprocess environment: base env + injector extras.
            # Extract the Gemini CWD sentinel before building the env dict.
            from bili.iris.mcp.cli_injectors import (  # pylint: disable=import-outside-toplevel
                _GEMINI_CWD_KEY,
            )

            run_env = os.environ.copy()
            sentinel_cwd: Optional[str] = extra_env.pop(_GEMINI_CWD_KEY, None)
            run_env.update(extra_env)

            # Resolve the subprocess working directory. An explicitly configured
            # CliLLM.cwd always wins -- it is a caller-controlled isolation
            # boundary (e.g. pinning the CLI subprocess to a sandbox directory)
            # and must not be silently overridden by an injector's own cwd
            # requirements. When no explicit cwd is configured, fall back to the
            # injector-provided sentinel (used by GeminiCliInjector to point the
            # subprocess at its generated project-scoped settings directory).
            # With neither set, cwd stays None and the subprocess inherits the
            # calling process's cwd, matching subprocess.run's own default and
            # the direct (_run_subprocess) CLI path's default.
            subprocess_cwd: Optional[str] = configured_cwd or sentinel_cwd

            LOGGER.debug(
                "EphemeralMcpServer: running CLI %s with MCP server %s",
                augmented_cmd[0],
                handle.server_url,
            )

            try:
                result = subprocess.run(  # pylint: disable=subprocess-run-check
                    augmented_cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=run_env,
                    cwd=subprocess_cwd,
                )
            except subprocess.TimeoutExpired as exc:
                raise CliLLMError(
                    f"CLI subprocess timed out after {timeout_seconds}s: "
                    f"{augmented_cmd[0]}"
                ) from exc
            finally:
                # Always clean up injector temp resources (files, dirs).
                if cleanup is not None:
                    try:
                        cleanup()
                    except (
                        OSError
                    ) as ce:  # pragma: no cover — transient OS cleanup error
                        LOGGER.debug("EphemeralMcpServer: cleanup error: %s", ce)

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise CliLLMError(
                f"CLI subprocess exited with code {result.returncode}: "
                f"{augmented_cmd[0]}\nstderr: {stderr_snippet}"
            )

        output = result.stdout
        if strip_ansi_flag:
            output = _ANSI_ESCAPE.sub("", output)

        # Parse output according to output_format.
        content = _parse_output(output, output_format, json_path, augmented_cmd[0])

        return {"messages": [AIMessage(content=content)]}

    return _node


def _parse_output(raw: str, output_format: str, json_path: str, cli_name: str) -> str:
    """Parse raw CLI stdout according to *output_format*.

    :param raw: The raw subprocess stdout (ANSI already stripped).
    :param output_format: ``"text"`` or ``"json"``.
    :param json_path: Dot-separated extraction path (only used for ``"json"``).
    :param cli_name: CLI executable name, used in error messages.
    :returns: The parsed output string.
    :raises CliLLMError: On JSON parse failure or path-not-found.
    """
    import json  # pylint: disable=import-outside-toplevel

    from bili.iris.providers.cli_provider import (  # pylint: disable=import-outside-toplevel
        CliLLMError,
        extract_json_path,
    )

    if output_format == "text":
        return raw.strip()

    # JSON output_format
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliLLMError(
            f"CLI output from {cli_name} is not valid JSON "
            f"(output_format='json'): {exc}"
        ) from exc
    try:
        return extract_json_path(parsed, json_path)
    except (KeyError, IndexError, TypeError) as exc:
        raise CliLLMError(
            f"json_path={json_path!r} not found in CLI output from {cli_name}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Helper: resolve injector from a CliLLM model
# ---------------------------------------------------------------------------


def resolve_mcp_injector(llm_model: Any) -> Optional[Any]:
    """Return the CLI injector for *llm_model*, or ``None`` if not resolvable.

    Checks that *llm_model* is a :class:`~bili.iris.providers.cli_provider.CliLLM`
    (has a ``command`` list attribute) and looks up its CLI basename in the
    injector registry.

    :param llm_model: Any LangChain-compatible model object.
    :returns: A :class:`~bili.iris.mcp.cli_injectors.McpCliInjector` subclass
        instance, or ``None`` if no injector is registered for this CLI.
    """
    command = getattr(llm_model, "command", None)
    if not command or not isinstance(command, list):
        return None

    from bili.iris.mcp.cli_injectors import (  # pylint: disable=import-outside-toplevel
        get_injector,
    )

    cli_name = Path(command[0]).name
    return get_injector(cli_name)


__all__ = [
    "EphemeralMcpHandle",
    "EphemeralMcpServer",
    "build_mcp_node",
    "resolve_mcp_injector",
]
