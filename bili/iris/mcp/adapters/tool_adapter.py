"""LangChain tool adapter for MCP tool definitions.

Wraps each MCP tool (name + description + JSON schema) as a LangChain
:class:`~langchain_core.tools.StructuredTool` so agents can call it via the
standard LangChain tool interface.

Each adapter exposes **both** a synchronous ``func`` and an asynchronous
``coroutine`` path on the ``StructuredTool``:

- The async path (``coroutine``) calls ``session.call_tool(...)`` directly --
  this is the preferred path for async agents.
- The sync path (``func``) bridges the async call to a synchronous context.
  See :func:`_run_async_sync` for the bridging strategy.

Sync/async bridge
-----------------
The ``mcp`` SDK is fully async.  AETHER and IRIS agents may invoke tools
synchronously (via LangChain's ``tool.invoke(...)``).  The bridge strategy:

1. If no loop is running in this thread, run via ``loop.run_until_complete(coro)``.
   This is the normal sync-agent case (e.g. LangChain ``AgentExecutor`` without
   ``arun``).  The MCP session is always invoked on the same loop it was created
   on because the session object is captured at construction time; the loop
   that created the session is the one that runs this call.

2. If a loop IS already running (e.g. inside ``asyncio.run`` or Jupyter),
   ``run_until_complete`` raises ``RuntimeError: This event loop is already
   running.``  In that case, submit to a fresh background thread via
   ``concurrent.futures.ThreadPoolExecutor`` and call ``asyncio.run(coro)``
   there.  **Important limitation:** the coroutine passed here must be
   self-contained and must NOT hold references to asyncio primitives
   (streams, locks, queues) that were created on the outer event loop.  MCP
   ``ClientSession`` objects carry such primitives internally, so calling a
   real MCP session through this path will fail.  The correct approach for an
   async caller is to use the ``coroutine`` path on the ``StructuredTool``
   directly, which avoids the bridge entirely.  The thread-executor branch is
   preserved for callers that genuinely have a running loop AND only pass
   self-contained coroutines (e.g. test helpers, simple async wrappers).

This approach requires no extra dependencies (stdlib only) and avoids
``nest_asyncio``, which patches the running loop globally and can cause
subtle interference in multi-agent scenarios.

Loop binding for MCP sessions
------------------------------
MCP ``ClientSession`` objects are bound to the event loop they were created
on (they hold internal asyncio streams and primitives).  The sync bridge
:func:`_run_async_sync` is safe for real MCP sessions ONLY when there is no
running loop in the calling thread (branch 1 above).  For the session-aware
sync invocation path, :func:`mcp_tool_to_langchain` captures the running
loop at adapter-creation time (via ``asyncio.get_event_loop()``) and uses
that specific loop for all sync calls.  This ensures the session is always
called on its own loop regardless of the caller's context.

Namespacing
-----------
Tools are registered under ``<server_name>__<tool_name>`` keys (e.g.
``my_server__edit_file``).  The separator is two underscores so it is
identifiable and unlikely to appear in either server or tool names naturally.

The ``StructuredTool.name`` is also set to the namespaced key so that
LangChain's tool routing can use it directly.

JSON schema mapping
-------------------
MCP tool definitions carry a JSON Schema for their input.  LangChain
``StructuredTool.args_schema`` accepts a Pydantic v2 model.  This module
dynamically builds a Pydantic model from the JSON schema so that LangChain
can validate inputs before passing them to the tool.

For complex schemas (nested objects, ``$ref``) the generated model may not
capture all constraints; in practice MCP tool schemas tend to be flat
property maps, so simple string/int/bool fields are the common case.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Optional, Type

LOGGER = logging.getLogger(__name__)

# Separator used to namespace MCP tools in the TOOL_REGISTRY.
MCP_TOOL_NAMESPACE_SEP = "__"


# ---------------------------------------------------------------------------
# Async -> sync bridge
# ---------------------------------------------------------------------------


def _run_async_sync(coro) -> Any:
    """Run *coro* to completion in a sync context without blocking an active loop.

    Strategy:

    1. If no loop is running in this thread, run via
       ``loop.run_until_complete(coro)``.  Safe for coroutines that reference
       asyncio objects (streams, locks) created on this loop.

    2. If a loop IS already running, submit to a fresh background thread and
       call ``asyncio.run(coro)`` there.  The coroutine must NOT reference
       asyncio objects from the outer loop -- only self-contained coroutines
       are safe through this path.  For real MCP sessions (which are bound
       to their creation loop), use the async ``coroutine`` path on the
       ``StructuredTool`` instead.

    :param coro: An awaitable coroutine.
    :returns: The coroutine's return value.
    :raises Exception: Propagates any exception raised by *coro*.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread at all -- create one.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Already inside a running loop (e.g. async agent, Jupyter).
        # Run in a fresh thread so we get a clean loop context.
        # NOTE: coroutine must be self-contained (no outer-loop asyncio refs).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    return loop.run_until_complete(coro)


def _run_on_loop(loop: asyncio.AbstractEventLoop, coro) -> Any:
    """Run *coro* on the specific *loop*, regardless of the caller's thread.

    Used by the sync invocation path for MCP tools so that session methods
    are always called on the event loop the session was created on.

    - If the loop is NOT running (caller is in a sync context): uses
      ``loop.run_until_complete(coro)``.
    - If the loop IS running (caller is inside an async context in another
      thread, or the loop was resumed): submits to the loop via
      ``asyncio.run_coroutine_threadsafe`` and blocks the calling thread
      until the result is ready.  The coroutine runs on the session's own
      loop, so session-internal asyncio objects remain on their home loop.

    :param loop: The event loop the session was created on.
    :param coro: An awaitable coroutine that uses objects bound to *loop*.
    :returns: The coroutine's return value.
    :raises Exception: Propagates any exception raised by *coro*.
    """
    if loop.is_running():
        # Submit to the running loop from this (presumably different) thread
        # and block until done.  This is the correct cross-thread call pattern
        # for asyncio objects that are bound to a specific loop.
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Pydantic schema generation from MCP JSON Schema
# ---------------------------------------------------------------------------

# JSON Schema type → Python type mapping for Pydantic field generation.
_JSON_SCHEMA_TO_PYTHON: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_schema(tool_name: str, input_schema: Dict[str, Any]) -> Optional[Type]:
    """Dynamically build a Pydantic model from a JSON Schema dict.

    Returns ``None`` if pydantic is not available or the schema is empty,
    in which case LangChain falls back to accepting ``**kwargs``.

    :param tool_name: Used as the generated model's class name.
    :param input_schema: The MCP tool's ``inputSchema`` dict.
    :returns: A Pydantic BaseModel subclass, or ``None``.
    """
    try:
        from pydantic import Field  # pylint: disable=import-outside-toplevel
        from pydantic import create_model  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    properties = input_schema.get("properties", {})
    required_fields: List[str] = input_schema.get("required", [])

    if not properties:
        return None

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        py_type = _JSON_SCHEMA_TO_PYTHON.get(json_type, Any)
        description = field_schema.get("description", "")
        if field_name in required_fields:
            field_definitions[field_name] = (py_type, Field(description=description))
        else:
            field_definitions[field_name] = (
                Optional[py_type],
                Field(default=None, description=description),
            )

    model_name = f"_{tool_name.title().replace('_', '')}Args"
    try:
        return create_model(model_name, **field_definitions)
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.debug(
            "Could not build Pydantic schema for MCP tool '%s'; "
            "falling back to untyped args",
            tool_name,
        )
        return None


# ---------------------------------------------------------------------------
# Tool adapter: MCP tool → LangChain StructuredTool
# ---------------------------------------------------------------------------


def mcp_tool_to_langchain(
    server_name: str,
    mcp_tool: Any,
    call_tool_fn: Callable,
) -> Any:
    """Wrap one MCP tool as a LangChain :class:`~langchain_core.tools.StructuredTool`.

    :param server_name: The MCP server's name (used for namespacing).
    :param mcp_tool: An ``mcp.types.Tool`` instance (name, description,
        inputSchema).
    :param call_tool_fn: An async callable ``(name, arguments) -> str`` that
        invokes the tool on the server.  Typically a closure over the live
        :class:`~mcp.ClientSession`.
    :returns: A LangChain ``StructuredTool`` registered under
        ``<server_name>__<tool_name>``.
    :raises ImportError: If ``langchain_core`` is not installed.
    """
    try:
        from langchain_core.tools import (  # pylint: disable=import-outside-toplevel
            StructuredTool,
        )
    except ImportError as exc:
        raise ImportError(
            "langchain_core is required for MCP tool adapters. "
            "Install it with: pip install langchain-core"
        ) from exc

    tool_name: str = mcp_tool.name
    namespaced_name = f"{server_name}{MCP_TOOL_NAMESPACE_SEP}{tool_name}"
    description: str = (
        mcp_tool.description or f"MCP tool '{tool_name}' from server '{server_name}'"
    )
    input_schema: Dict[str, Any] = (
        mcp_tool.inputSchema.model_dump()
        if hasattr(mcp_tool.inputSchema, "model_dump")
        else dict(mcp_tool.inputSchema)
    )

    args_schema = _build_args_schema(namespaced_name, input_schema)

    # Capture the event loop that is current at adapter-creation time.
    # The MCP session (and all its internal asyncio objects) was created on
    # this loop, so the sync invocation path MUST call back onto the same
    # loop to avoid cross-loop errors with real sessions.
    try:
        _session_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_event_loop()
    except RuntimeError:
        _session_loop = None

    async def _async_invoke(**kwargs: Any) -> str:
        """Async tool invocation path -- called by async LangChain agents."""
        LOGGER.debug(
            "MCP tool '%s' async call with args: %s",
            namespaced_name,
            list(kwargs.keys()),
        )
        return await call_tool_fn(tool_name, kwargs)

    def _sync_invoke(**kwargs: Any) -> str:
        """Sync tool invocation path -- calls back onto the session's own loop.

        Uses :func:`_run_on_loop` with the event loop captured at adapter-
        creation time so that the MCP session's internal asyncio objects
        (streams, read/write primitives) are always called on their home loop,
        regardless of which thread or loop the sync caller is on.
        """
        LOGGER.debug(
            "MCP tool '%s' sync call with args: %s",
            namespaced_name,
            list(kwargs.keys()),
        )

        async def _coro():
            return await call_tool_fn(tool_name, kwargs)

        if _session_loop is not None:
            return _run_on_loop(_session_loop, _coro())
        # No loop was running at adapter-creation time; fall back to
        # _run_async_sync which creates a fresh loop.
        return _run_async_sync(_coro())

    structured_tool = StructuredTool(
        name=namespaced_name,
        description=description,
        func=_sync_invoke,
        coroutine=_async_invoke,
        args_schema=args_schema,
    )

    LOGGER.debug(
        "Registered MCP tool '%s' (server='%s', original='%s')",
        namespaced_name,
        server_name,
        tool_name,
    )
    return structured_tool


# ---------------------------------------------------------------------------
# Batch adapter: list of MCP tools → list of LangChain tools
# ---------------------------------------------------------------------------


def mcp_tools_to_langchain(
    server_name: str,
    mcp_tools: List[Any],
    call_tool_fn: Callable,
) -> List[Any]:
    """Convert a list of MCP tools to LangChain tools.

    :param server_name: The MCP server's name.
    :param mcp_tools: A list of ``mcp.types.Tool`` objects.
    :param call_tool_fn: Async callable for invoking a single tool.
    :returns: List of LangChain ``StructuredTool`` objects.
    """
    tools = []
    for mcp_tool in mcp_tools:
        try:
            lc_tool = mcp_tool_to_langchain(server_name, mcp_tool, call_tool_fn)
            tools.append(lc_tool)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Failed to adapt MCP tool '%s' from server '%s': %s",
                getattr(mcp_tool, "name", "?"),
                server_name,
                exc,
            )
    return tools


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def extract_text_from_result(result: Any) -> str:
    """Extract a text string from a :class:`~mcp.types.CallToolResult`.

    MCP tool results carry a ``content`` list of typed content blocks
    (``TextContent``, ``ImageContent``, etc.).  This helper concatenates all
    ``TextContent`` blocks into a single string.  Non-text content is ignored
    with a debug log.

    :param result: A ``mcp.types.CallToolResult`` (or any object with a
        ``content`` attribute containing typed content blocks).
    :returns: The concatenated text, or an empty string if no text blocks.
    """
    if result is None:
        return ""

    # Handle isError flag
    if getattr(result, "isError", False):
        # Still extract text — the error message is usually in the text content
        LOGGER.warning("MCP tool returned an error result")

    content_blocks = getattr(result, "content", [])
    text_parts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        else:
            LOGGER.debug("Skipping non-text MCP content block: type=%s", block_type)

    return "\n".join(text_parts)
