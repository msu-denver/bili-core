"""Real end-to-end integration test: ask_user through the ephemeral MCP server.

Unlike every other test in bili/iris/mcp/tests/ (which mock the server and
transport entirely -- see test_server.py's own module docstring: "All tests
run without a real MCP server or CLI binary. Network I/O is mocked."), this
test spins up a REAL EphemeralMcpServer (real uvicorn thread, real MCPServer
tool registration) and drives it with a REAL MCP client
(streamable-HTTP transport, matching what EphemeralMcpServer actually
serves) making a genuine HTTP tool call.

The fake CLI agent here is a real MCP client session standing in for what
claude -p / codex / gemini would do internally when they call an MCP tool --
the actual CLI subprocess spawn (build_mcp_node's subprocess.run) is
exercised separately and with mocks in test_server.py; this test's job is
proving the tool CALL itself, over the real transport, reaches a real
HitlResponder and blocks correctly.

Requires the [mcp] extra (mcp + uvicorn). Skipped, not failed, when it is
not installed -- mirrors bili.iris.mcp.server's own optional-dependency
handling rather than making this a hard requirement for the base install.
"""

# pylint: disable=import-outside-toplevel
# Every mcp/uvicorn/ask_user import in this module is deferred to inside
# test functions and async closures, matching the optional-[mcp]-extra
# pattern bili.iris.mcp.server itself uses (importing at module level would
# make collecting this file fail entirely when the extra is not installed,
# defeating the pytestmark skip above).

import asyncio
import contextlib
import os
import threading
import time

import pytest

from bili.iris.mcp.server import _MCP_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _MCP_AVAILABLE, reason="requires the [mcp] extra (mcp + uvicorn)"
)


def _register_and_get_tool(responder):
    """Register ask_user with *responder* and return the built LangChain tool."""
    from bili.iris.loaders.tools_loader import TOOL_REGISTRY
    from bili.iris.tools.ask_user import ASK_USER_TOOL_NAME, register_ask_user_tool

    register_ask_user_tool(responder)
    return TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})


@contextlib.asynccontextmanager
async def _client_session(handle):
    """Open an authenticated streamable-HTTP MCP client session to *handle*.

    Encapsulates the mcp client construction so each test states only the tool
    call it makes.  The Bearer token is carried on a pre-built ``httpx2``
    client because ``streamable_http_client`` takes an ``http_client`` rather
    than a ``headers`` argument.  Imports are deferred to match this module's
    optional-[mcp]-extra pattern.
    """
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {handle.token}"}
    async with httpx2.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(
            handle.server_url, http_client=http_client
        ) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class TestAskUserMcpIntegration:
    """Proves the ask_user MCP-path seam end-to-end over a real server."""

    def teardown_method(self):
        """Unregister ask_user after each test."""
        from bili.iris.tools.ask_user import unregister_ask_user_tool

        unregister_ask_user_tool()

    def test_real_mcp_call_blocks_on_responder_and_returns_answer(self):
        """A real MCP client calls ask_user; the call blocks on a real
        ScriptedHitlResponder and the answer comes back as the tool result
        -- the exact round trip a spawned CLI subprocess's own internal
        tool-calling loop would make against the ephemeral server.
        """
        from bili.iris.mcp.server import EphemeralMcpServer
        from bili.iris.tools.hitl import ScriptedHitlResponder

        responder = ScriptedHitlResponder(["staging"])
        tool = _register_and_get_tool(responder)

        async def _run():
            _server = EphemeralMcpServer([tool])
            with _server as handle:
                # This test process is the caller, so it is what the
                # server must be bound to; build_mcp_node does the same
                # for the CLI it spawns.
                _server.authorize_subprocess(os.getpid())
                async with _client_session(handle) as session:
                    tools_result = await session.list_tools()
                    assert [t.name for t in tools_result.tools] == ["ask_user"]

                    return await session.call_tool(
                        "ask_user",
                        arguments={"question": "Which environment should I deploy to?"},
                    )

        result = asyncio.run(_run())

        assert result.is_error is False
        text_blocks = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        assert text_blocks == ["staging"]

        # The responder genuinely received the call -- not a placeholder
        # short-circuit somewhere upstream of the real HTTP round trip.
        assert responder.calls == [
            {"question": "Which environment should I deploy to?", "options": None}
        ]

    def test_real_mcp_call_forwards_options(self):
        """The 'options' rendering hint reaches the responder over the wire."""
        from bili.iris.mcp.server import EphemeralMcpServer
        from bili.iris.tools.hitl import ScriptedHitlResponder

        responder = ScriptedHitlResponder(["staging"])
        tool = _register_and_get_tool(responder)

        async def _run():
            _server = EphemeralMcpServer([tool])
            with _server as handle:
                # This test process is the caller, so it is what the
                # server must be bound to; build_mcp_node does the same
                # for the CLI it spawns.
                _server.authorize_subprocess(os.getpid())
                async with _client_session(handle) as session:
                    return await session.call_tool(
                        "ask_user",
                        arguments={
                            "question": "Which environment?",
                            "options": ["staging", "production"],
                        },
                    )

        asyncio.run(_run())

        assert responder.calls == [
            {
                "question": "Which environment?",
                "options": ["staging", "production"],
            }
        ]

    def test_real_mcp_call_with_no_responder_returns_null_sentinel(self):
        """No responder registered (NullHitlResponder default): the real MCP
        round trip returns the no-response sentinel instead of hanging or
        raising -- an unconfigured CLI-path deployment degrades gracefully.
        """
        from bili.iris.mcp.server import EphemeralMcpServer
        from bili.iris.tools.hitl import NO_RESPONSE_PREFIX

        tool = _register_and_get_tool(None)

        async def _run():
            _server = EphemeralMcpServer([tool])
            with _server as handle:
                # This test process is the caller, so it is what the
                # server must be bound to; build_mcp_node does the same
                # for the CLI it spawns.
                _server.authorize_subprocess(os.getpid())
                async with _client_session(handle) as session:
                    return await session.call_tool(
                        "ask_user", arguments={"question": "Which environment?"}
                    )

        result = asyncio.run(_run())

        text_blocks = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        assert len(text_blocks) == 1
        assert text_blocks[0].startswith(NO_RESPONSE_PREFIX)

    def test_call_blocks_the_calling_thread_until_responder_returns(self):
        """The tool call genuinely BLOCKS for as long as the responder takes
        to answer -- proving the seam is a real blocking call, not something
        that returns immediately with a promise/future the CLI subprocess
        would then have to poll (which is not how MCP tool calls work; the
        subprocess's own request blocks on this call by construction, but
        this test pins the seam's own blocking behavior directly, independent
        of MCP transport semantics).
        """
        from bili.iris.mcp.server import EphemeralMcpServer

        # pylint: disable=too-few-public-methods
        # A single-method HitlResponder test double by design -- see hitl.py's
        # own module-level disable comment on the same shape.
        class _SlowResponder:
            """Sleeps briefly before answering, to make blocking observable."""

            def __init__(self, delay_seconds, answer):
                self._delay = delay_seconds
                self._answer = answer
                self.answered_at = None

            def ask(self, question, options=None):  # pylint: disable=unused-argument
                """Sleep, then return the scripted answer."""
                time.sleep(self._delay)
                self.answered_at = time.monotonic()
                return self._answer

        responder = _SlowResponder(delay_seconds=0.3, answer="staging")
        tool = _register_and_get_tool(responder)

        async def _run():
            _server = EphemeralMcpServer([tool])
            with _server as handle:
                # This test process is the caller, so it is what the
                # server must be bound to; build_mcp_node does the same
                # for the CLI it spawns.
                _server.authorize_subprocess(os.getpid())
                async with _client_session(handle) as session:
                    before = time.monotonic()
                    result = await session.call_tool(
                        "ask_user", arguments={"question": "Which environment?"}
                    )
                    after = time.monotonic()
                    return result, before, after

        _result, before, after = asyncio.run(_run())

        assert (after - before) >= 0.3, (
            "the call returned before the responder's delay elapsed -- "
            "the seam is not genuinely blocking"
        )
        assert responder.answered_at is not None
        assert before <= responder.answered_at <= after

    def test_two_concurrent_agent_runs_against_one_responder_do_not_cross_answers(self):
        """Two ask_user calls in flight at once against ONE shared responder
        (the fan-out-batch scenario HitlResponder's docstring names) get
        their own distinct, correctly-matched answers -- not a race where
        one call's answer leaks into the other's result.

        Uses two separate EphemeralMcpServer instances (each call is its own
        MCP server + subprocess in the real architecture) driven from two
        threads, both hitting the SAME ScriptedHitlResponder instance, which
        is exactly the shape a host's ScriptedHitlResponder must be safe
        under (see hitl.py's thread-safety contract).
        """
        from bili.iris.mcp.server import EphemeralMcpServer
        from bili.iris.tools.hitl import ScriptedHitlResponder

        responder = ScriptedHitlResponder(["staging", "production"])

        results = {}
        errors = []

        def _run_one(index, question):
            try:
                tool = _register_and_get_tool_isolated(responder)

                async def _run():
                    _server = EphemeralMcpServer([tool])
                    with _server as handle:
                        # This test process is the caller, so it is what the
                        # server must be bound to; build_mcp_node does the
                        # same for the CLI it spawns.
                        _server.authorize_subprocess(os.getpid())
                        async with _client_session(handle) as session:
                            return await session.call_tool(
                                "ask_user", arguments={"question": question}
                            )

                results[index] = asyncio.run(_run())
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append((index, exc))

        threads = [
            threading.Thread(target=_run_one, args=(0, "Which environment (a)?")),
            threading.Thread(target=_run_one, args=(1, "Which environment (b)?")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent calls raised: {errors}"
        assert set(results.keys()) == {0, 1}

        answers = set()
        for result in results.values():
            text_blocks = [
                block.text
                for block in result.content
                if getattr(block, "type", None) == "text"
            ]
            answers.add(text_blocks[0])

        # Both scripted answers were consumed exactly once each, with no
        # duplicate/lost/crossed answer -- the lock in ScriptedHitlResponder
        # (see hitl.py) is what makes this deterministic instead of racy.
        assert answers == {"staging", "production"}
        assert len(responder.calls) == 2


def _register_and_get_tool_isolated(responder):
    """Like _register_and_get_tool but does not mutate TOOL_REGISTRY.

    Used by the concurrency test, which needs two independent tool
    instances bound to the SAME responder without one registration
    clobbering the other via the shared TOOL_REGISTRY dict (register_ask_user_tool
    is a process-global registration, not meant to be called concurrently
    from two threads for two different tool instances).
    """
    from langchain_core.tools import StructuredTool

    from bili.iris.tools.ask_user import _ASK_USER_DESCRIPTION, _build_ask_user_func

    return StructuredTool.from_function(
        func=_build_ask_user_func(responder),
        name="ask_user",
        description=_ASK_USER_DESCRIPTION,
    )
