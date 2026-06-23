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

1. Try ``asyncio.get_event_loop().run_until_complete(coro)``.  This works
   when there is no currently running event loop in the caller's thread
   (the typical case for sync LangChain agent execution).

2. If a loop IS already running (e.g. inside ``asyncio.run`` or Jupyter),
   ``run_until_complete`` raises ``RuntimeError: This event loop is already
   running.``  In that case, submit the coroutine to a fresh
   ``concurrent.futures.ThreadPoolExecutor`` thread, which has its own fresh
   event loop.  ``asyncio.run(coro)`` inside the thread handles it cleanly.

This approach requires no extra dependencies (stdlib only) and avoids
``nest_asyncio``, which patches the running loop globally and can cause
subtle interference in multi-agent scenarios.

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
# Async → sync bridge
# ---------------------------------------------------------------------------


def _run_async_sync(coro) -> Any:
    """Run *coro* to completion in a sync context without blocking an active loop.

    Strategy:
    1. If no loop is running in this thread, run in the thread's event loop.
    2. If a loop IS already running, submit to a fresh background thread
       (which has no running loop) and wait for the result.

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
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

    async def _async_invoke(**kwargs: Any) -> str:
        """Async tool invocation path — called by async LangChain agents."""
        LOGGER.debug(
            "MCP tool '%s' async call with args: %s",
            namespaced_name,
            list(kwargs.keys()),
        )
        return await call_tool_fn(tool_name, kwargs)

    def _sync_invoke(**kwargs: Any) -> str:
        """Sync tool invocation path — bridges async MCP call for sync agents."""
        LOGGER.debug(
            "MCP tool '%s' sync call with args: %s",
            namespaced_name,
            list(kwargs.keys()),
        )

        async def _coro():
            return await call_tool_fn(tool_name, kwargs)

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
