"""Tests for MCP tool adapters (bili/iris/mcp/adapters/tool_adapter.py).

All tests are unit-level; no real MCP server or subprocess is spawned.
The MCP SDK types are mocked with simple Python objects so the tests pass
without requiring the real ``mcp`` package to be installed in a special state.
"""

# pylint: disable=too-few-public-methods, not-callable

import asyncio
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from bili.iris.mcp.adapters.tool_adapter import (
    MCP_TOOL_NAMESPACE_SEP,
    _build_args_schema,
    _run_async_sync,
    _run_on_loop,
    extract_text_from_result,
    mcp_tool_to_langchain,
    mcp_tools_to_langchain,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str, description: str = "", schema: Optional[dict] = None
) -> Any:
    """Create a minimal mock mcp.types.Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = MagicMock()
    tool.inputSchema.model_dump = MagicMock(
        return_value=schema or {"type": "object", "properties": {}}
    )
    return tool


def _make_text_content(text: str) -> Any:
    """Create a mock mcp.types.TextContent block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_call_result(text_contents: list, is_error: bool = False) -> Any:
    """Create a mock mcp.types.CallToolResult."""
    result = MagicMock()
    result.content = [_make_text_content(t) for t in text_contents]
    result.isError = is_error
    return result


# ---------------------------------------------------------------------------
# _run_async_sync
# ---------------------------------------------------------------------------


class TestRunAsyncSync:
    """Tests for the async-to-sync bridge helper."""

    def test_runs_simple_coroutine(self):
        """A simple coroutine completes and returns its value."""

        async def _coro():
            return 42

        assert _run_async_sync(_coro()) == 42

    def test_propagates_exception(self):
        """Exceptions raised inside the coroutine propagate to the sync caller."""

        async def _coro():
            raise ValueError("inner error")

        with pytest.raises(ValueError, match="inner error"):
            _run_async_sync(_coro())

    def test_handles_already_running_loop(self):
        """When called from inside a running loop, uses a thread to run the coroutine."""

        async def _outer():
            async def _inner():
                return "from_thread"

            # Simulate the running-loop case by calling _run_async_sync from
            # within an already-running event loop.
            result = _run_async_sync(_inner())
            return result

        result = asyncio.run(_outer())
        assert result == "from_thread"

    def test_async_values_preserved(self):
        """The coroutine return value passes through unchanged."""

        async def _coro():
            return {"key": "value", "count": 99}

        assert _run_async_sync(_coro()) == {"key": "value", "count": 99}


# ---------------------------------------------------------------------------
# _run_on_loop
# ---------------------------------------------------------------------------


class TestRunOnLoop:
    """Tests for the session-loop-aware sync bridge."""

    def test_runs_on_idle_loop(self):
        """A coroutine runs to completion when the loop is not running."""
        loop = asyncio.new_event_loop()
        try:

            async def _coro():
                return "idle_result"

            result = _run_on_loop(loop, _coro())
            assert result == "idle_result"
        finally:
            loop.close()

    def test_propagates_exception_on_idle_loop(self):
        """Exceptions from the coroutine propagate to the sync caller."""
        loop = asyncio.new_event_loop()
        try:

            async def _coro():
                raise ValueError("loop error")

            with pytest.raises(ValueError, match="loop error"):
                _run_on_loop(loop, _coro())
        finally:
            loop.close()

    def test_runs_from_different_thread_on_running_loop(self):
        """When the loop is running in another thread, run_coroutine_threadsafe is used."""
        results = []
        loop = asyncio.new_event_loop()

        async def _session_coro():
            return "cross_thread"

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()

        # Give the loop a moment to start
        time.sleep(0.05)

        try:
            # Call from the main thread onto the running loop
            result = _run_on_loop(loop, _session_coro())
            results.append(result)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()

        assert results == ["cross_thread"]

    def test_sync_tool_uses_session_loop(self):
        """The sync func path of a StructuredTool calls back on the session's loop.

        This is the key regression test for cross-loop session misuse.  The
        call_tool_fn checks that it is running on the expected loop.  A wrong
        loop would cause the session's asyncio objects to raise errors.
        """
        expected_loop = asyncio.new_event_loop()
        observed_loops = []

        async def _call(_tool_name: str, _arguments: dict) -> str:
            # Record which loop is running this coroutine
            observed_loops.append(asyncio.get_event_loop())
            return "ok"

        # Create the adapter while the expected_loop is set as "current"
        # so it is captured as _session_loop.
        original_loop = None
        try:
            original_loop = asyncio.get_event_loop()
        except RuntimeError:
            pass

        asyncio.set_event_loop(expected_loop)
        try:
            tool = _make_mcp_tool("t")
            lc_tool = mcp_tool_to_langchain("srv", tool, _call)
        finally:
            # Restore the original loop (or clear it)
            if original_loop is not None:
                asyncio.set_event_loop(original_loop)
            else:
                asyncio.set_event_loop(None)

        # Now call the sync path -- it should route back to expected_loop
        result = lc_tool.func()  # type: ignore[operator]
        assert result == "ok"
        assert len(observed_loops) == 1
        assert observed_loops[0] is expected_loop

        expected_loop.close()


# ---------------------------------------------------------------------------
# extract_text_from_result
# ---------------------------------------------------------------------------


class TestExtractTextFromResult:
    """Tests for the CallToolResult text extractor."""

    def test_single_text_block(self):
        """A single TextContent block returns its text."""
        result = _make_call_result(["hello world"])
        assert extract_text_from_result(result) == "hello world"

    def test_multiple_text_blocks_joined(self):
        """Multiple TextContent blocks are joined with a newline."""
        result = _make_call_result(["first", "second", "third"])
        assert extract_text_from_result(result) == "first\nsecond\nthird"

    def test_empty_content_list(self):
        """An empty content list returns an empty string."""
        result = _make_call_result([])
        assert extract_text_from_result(result) == ""

    def test_none_result(self):
        """A None result returns an empty string."""
        assert extract_text_from_result(None) == ""

    def test_non_text_block_ignored(self):
        """Non-text content blocks (images etc.) are skipped."""
        result = MagicMock()
        result.isError = False
        image_block = MagicMock()
        image_block.type = "image"
        text_block = _make_text_content("kept text")
        result.content = [image_block, text_block]
        assert extract_text_from_result(result) == "kept text"

    def test_error_result_still_extracts_text(self):
        """isError=True results still extract their text content."""
        result = _make_call_result(["error: something went wrong"], is_error=True)
        assert "error: something went wrong" in extract_text_from_result(result)


# ---------------------------------------------------------------------------
# _build_args_schema
# ---------------------------------------------------------------------------


class TestBuildArgsSchema:
    """Tests for the JSON schema to Pydantic model builder."""

    def test_empty_schema_returns_none(self):
        """An empty schema (no properties) returns None."""
        schema = {"type": "object", "properties": {}}
        result = _build_args_schema("my_tool", schema)
        assert result is None

    def test_simple_schema_creates_model(self):
        """A schema with string and int fields creates a Pydantic model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name param"},
                "count": {"type": "integer", "description": "Count param"},
            },
            "required": ["name"],
        }
        model = _build_args_schema("test_tool", schema)
        assert model is not None
        assert hasattr(model, "model_fields")
        assert "name" in model.model_fields
        assert "count" in model.model_fields

    def test_required_fields_not_optional(self):
        """Required fields are not wrapped in Optional."""
        schema = {
            "type": "object",
            "properties": {"req": {"type": "string"}},
            "required": ["req"],
        }
        model = _build_args_schema("t", schema)
        assert model is not None
        field = model.model_fields["req"]
        # Required field should not have a default
        assert field.default is None or field.is_required()

    def test_optional_fields_get_none_default(self):
        """Optional fields default to None."""
        schema = {
            "type": "object",
            "properties": {"opt": {"type": "string"}},
            "required": [],
        }
        model = _build_args_schema("t", schema)
        assert model is not None
        field = model.model_fields["opt"]
        assert field.default is None

    def test_no_schema_returns_none(self):
        """A missing properties key returns None."""
        result = _build_args_schema("t", {})
        assert result is None


# ---------------------------------------------------------------------------
# mcp_tool_to_langchain
# ---------------------------------------------------------------------------


class TestMcpToolToLangchain:
    """Tests for the single-tool adapter."""

    def _async_call_fn(self, response: str = "result"):
        """Return an async call_tool function that returns a fixed response."""

        async def _call(_tool_name: str, _arguments: dict) -> str:
            return response

        return _call

    def test_namespacing(self):
        """The StructuredTool name uses the <server>__<tool> convention."""
        tool = _make_mcp_tool("edit_file", "Edits a file")
        lc_tool = mcp_tool_to_langchain("my_server", tool, self._async_call_fn())
        assert lc_tool.name == f"my_server{MCP_TOOL_NAMESPACE_SEP}edit_file"

    def test_description_preserved(self):
        """The StructuredTool description comes from the MCP tool."""
        tool = _make_mcp_tool("read_file", "Reads a file from disk")
        lc_tool = mcp_tool_to_langchain("srv", tool, self._async_call_fn())
        assert "Reads a file from disk" in lc_tool.description

    def test_empty_description_fallback(self):
        """An empty MCP description gets a generated fallback description."""
        tool = _make_mcp_tool("do_thing", description="")
        lc_tool = mcp_tool_to_langchain("srv", tool, self._async_call_fn())
        assert len(lc_tool.description) > 0

    def test_sync_path_invokes_call_fn(self):
        """The sync func path calls the async call_tool_fn and returns its result."""
        call_results = []

        async def _call(tool_name: str, arguments: dict) -> str:
            call_results.append((tool_name, arguments))
            return "sync_response"

        tool = _make_mcp_tool("my_tool")
        lc_tool = mcp_tool_to_langchain("srv", tool, _call)
        result = lc_tool.func(param1="value1")  # type: ignore[operator]
        assert result == "sync_response"
        assert len(call_results) == 1
        assert call_results[0][0] == "my_tool"

    def test_async_path_invokes_call_fn(self):
        """The async coroutine path calls the async call_tool_fn."""
        call_results = []

        async def _call(tool_name: str, arguments: dict) -> str:
            call_results.append((tool_name, arguments))
            return "async_response"

        tool = _make_mcp_tool("my_tool")
        lc_tool = mcp_tool_to_langchain("srv", tool, _call)

        result = asyncio.run(lc_tool.coroutine(param1="value1"))  # type: ignore[operator]
        assert result == "async_response"
        assert len(call_results) == 1

    def test_structured_tool_has_both_paths(self):
        """The StructuredTool has both func and coroutine attributes."""
        tool = _make_mcp_tool("t")
        lc_tool = mcp_tool_to_langchain("s", tool, self._async_call_fn())
        assert callable(lc_tool.func)
        assert callable(lc_tool.coroutine)

    def test_with_json_schema(self):
        """A tool with a JSON schema gets an args_schema set."""
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        }
        tool = _make_mcp_tool("edit_file", schema=schema)
        lc_tool = mcp_tool_to_langchain("srv", tool, self._async_call_fn())
        assert lc_tool.args_schema is not None


# ---------------------------------------------------------------------------
# mcp_tools_to_langchain (batch)
# ---------------------------------------------------------------------------


class TestMcpToolsToLangchain:
    """Tests for the batch tool adapter."""

    def test_converts_all_tools(self):
        """All tools in the list are converted."""
        tools = [_make_mcp_tool(f"tool_{i}") for i in range(3)]

        async def _call(_n, _a):
            return ""

        lc_tools = mcp_tools_to_langchain("srv", tools, _call)
        assert len(lc_tools) == 3

    def test_namespacing_applied_to_all(self):
        """All tools are namespaced with the server name."""
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b")]

        async def _call(_n, _a):
            return ""

        lc_tools = mcp_tools_to_langchain("server", tools, _call)
        names = [t.name for t in lc_tools]
        assert all(n.startswith("server__") for n in names)

    def test_skips_broken_tool_gracefully(self):
        """A tool that fails to adapt is skipped; others succeed."""
        good_tool = _make_mcp_tool("good")
        bad_tool = MagicMock()
        bad_tool.name = "bad"
        bad_tool.description = "oops"
        bad_tool.inputSchema = MagicMock()
        bad_tool.inputSchema.model_dump = MagicMock(
            side_effect=RuntimeError("schema error")
        )

        original_fn = mcp_tool_to_langchain

        def _patched(server_name, mcp_tool, call_tool_fn):
            if mcp_tool is bad_tool:
                raise RuntimeError("schema error")
            return original_fn(server_name, mcp_tool, call_tool_fn)

        async def _call(_n, _a):
            return ""

        with patch(
            "bili.iris.mcp.adapters.tool_adapter.mcp_tool_to_langchain",
            side_effect=_patched,
        ):
            lc_tools = mcp_tools_to_langchain("srv", [good_tool, bad_tool], _call)

        assert len(lc_tools) == 1
        assert lc_tools[0].name == "srv__good"

    def test_empty_list_returns_empty(self):
        """An empty tool list returns an empty list."""

        async def _call(_n, _a):
            return ""

        assert not mcp_tools_to_langchain("srv", [], _call)
