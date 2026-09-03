"""Tests that the ephemeral MCP server is bound to the process tree it serves.

The property under test has two directions, and only asserting one of them is
how this class of check ships broken:

* the spawned subprocess (and anything it spawns) **succeeds**;
* a process that is not in that tree, **holding the correct token**, is
  refused.

Direction two is the one that matters, and it is asserted against the correct
token on purpose.  Refusing a caller because its token is wrong proves nothing
about identity, so a test that lets the token be wrong passes vacuously with
the identity check deleted.

The end-to-end cases below drive a real :class:`EphemeralMcpServer` (real
uvicorn thread, real MCPServer app, real HTTP over loopback) and real spawned
processes.  Attribution is a property of actual sockets owned by actual
processes; a mocked peer would only prove that the mock was consulted.
"""

# pylint: disable=too-few-public-methods,protected-access,missing-function-docstring,missing-class-docstring,import-outside-toplevel

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from bili.iris.mcp.peer_identity import PeerAuthorization, ProcessIdentity
from bili.iris.mcp.server import _MCP_AVAILABLE, EphemeralMcpServer

pytestmark = pytest.mark.skipif(
    not _MCP_AVAILABLE, reason="requires the [mcp] extra (mcp + uvicorn)"
)

#: What the served tool returns. A test asserts on this exact string so a
#: refusal cannot be mistaken for a successful call that returned nothing.
_PRIVILEGED_RESULT = "privileged-tool-result"


class _EchoArgs(BaseModel):
    value: str = Field(description="Anything")


def _privileged_tool() -> StructuredTool:
    """A tool whose result is recognisable, standing in for a real capability."""

    def _run(value: str) -> str:
        return f"{_PRIVILEGED_RESULT}:{value}"

    return StructuredTool(
        name="privileged_capability",
        description="Returns privileged data",
        func=_run,
        args_schema=_EchoArgs,
    )


# The client half, run both in-process and inside spawned children. Kept as
# source so a child can execute it without importing the test module.
_CLIENT_SOURCE = textwrap.dedent(
    """
    import asyncio, json, sys

    async def _call(url, token):
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx2.AsyncClient(headers=headers, timeout=10) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read, write, *_,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.call_tool(
                        "privileged_capability", arguments={"value": "x"}
                    )
                    return res.content[0].text

    def main(url, token):
        try:
            return {"ok": True, "result": asyncio.run(_call(url, token))}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    """
)


def _call_in_process(url: str, token: str) -> dict:
    """Call the tool from *this* process."""
    namespace: dict = {}
    exec(compile(_CLIENT_SOURCE, "<client>", "exec"), namespace)  # nosec B102
    return namespace["main"](url, token)


#: A standalone program that calls the tool and prints the outcome as JSON.
#: Built by concatenation rather than by interpolating the client source into
#: an indented template, which produces source whose indentation does not
#: survive dedent.
_CALLER_PROGRAM = _CLIENT_SOURCE + textwrap.dedent(
    """
    sys.stdout.write(json.dumps(main(sys.argv[1], sys.argv[2])))
    """
)

#: A program that spawns _CALLER_PROGRAM (passed as argv[3]) and relays its
#: output, so the call originates one generation further down.
_RELAY_PROGRAM = textwrap.dedent(
    """
    import subprocess, sys
    child = subprocess.Popen(
        [sys.executable, "-c", sys.argv[3], sys.argv[1], sys.argv[2]],
        stdout=subprocess.PIPE, text=True,
    )
    out, _ = child.communicate(timeout=80)
    sys.stdout.write(out)
    """
)


def _read_outcome(stdout: str, stderr: str) -> dict:
    """Parse the JSON outcome a caller program printed."""
    if not stdout.strip():
        raise AssertionError(f"caller produced no output; stderr={stderr[:800]}")
    return json.loads(stdout.strip().splitlines()[-1])


def _call_in_child(url: str, token: str) -> dict:
    """Call the tool from a freshly spawned process."""
    proc = subprocess.run(
        [sys.executable, "-c", _CALLER_PROGRAM, url, token],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return _read_outcome(proc.stdout, proc.stderr)


class _ServedTool:
    """Runs a real ephemeral server and spawns a real child to authorize.

    The child is a process that sleeps: `build_mcp_node` authorizes the CLI it
    spawns, and the tests need the same shape without needing a CLI binary.
    """

    def __init__(self) -> None:
        self.server = EphemeralMcpServer([_privileged_tool()])
        self.handle = None
        self.child = None

    def __enter__(self) -> "_ServedTool":
        self.handle = self.server.__enter__()
        return self

    def spawn_and_authorize(self) -> int:
        """Spawn the process this server exists for and grant it access."""
        self.child = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", "import time; time.sleep(120)"],
        )
        self.server.authorize_subprocess(self.child.pid)
        return self.child.pid

    def __exit__(self, *exc) -> bool:
        if self.child is not None:
            self.child.kill()
            self.child.wait(timeout=10)
        self.server.__exit__(*exc)
        return False


class TestConnectionIsBoundToTheSpawnedProcessTree:
    """The end-to-end property, in both directions, over a real server."""

    def test_a_process_holding_the_token_but_outside_the_tree_is_refused(self):
        """DIRECTION 2: the correct token is not sufficient.

        This test process spawned the authorized child, so it is that child's
        parent and therefore not inside the authorized tree.  It holds the
        real token.  It must still be refused.
        """
        with _ServedTool() as served:
            served.spawn_and_authorize()
            outcome = _call_in_process(served.handle.server_url, served.handle.token)

        assert not outcome["ok"], (
            "a process outside the spawned tree completed a tool call using the "
            f"token: {outcome}"
        )
        assert _PRIVILEGED_RESULT not in json.dumps(outcome)

    def test_the_refusal_is_403_not_401(self):
        """The refusal must be the identity check, not an accidental token failure.

        Without this, a test whose token happened to be wrong would report the
        same pass with the identity check removed entirely.
        """
        import urllib.error
        import urllib.request

        with _ServedTool() as served:
            served.spawn_and_authorize()
            req = urllib.request.Request(
                served.handle.server_url,
                data=b"{}",
                headers={
                    "Authorization": f"Bearer {served.handle.token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=10)  # nosec B310

            assert exc_info.value.code == 403, (
                "a valid token from outside the tree must be refused as "
                f"Forbidden, got {exc_info.value.code}"
            )

            # And the token check still works, distinguishably.
            bad = urllib.request.Request(
                served.handle.server_url,
                data=b"{}",
                headers={
                    "Authorization": "Bearer not-the-token",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as bad_info:
                urllib.request.urlopen(bad, timeout=10)  # nosec B310
            assert bad_info.value.code == 401

    def test_the_spawned_process_succeeds(self):
        """DIRECTION 1: the process the server exists for can call its tools.

        A check that refuses everything satisfies direction 2 perfectly and is
        useless, so this leg is what stops the fix from being a denial of
        service.
        """
        with _ServedTool() as served:
            # Authorize the child that will make the call, which is exactly
            # build_mcp_node's shape.
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    sys.executable,
                    "-c",
                    _CALLER_PROGRAM,
                    served.handle.server_url,
                    served.handle.token,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            served.server.authorize_subprocess(proc.pid)
            stdout, stderr = proc.communicate(timeout=90)

        outcome = _read_outcome(stdout, stderr)
        assert outcome["ok"], f"the authorized subprocess was refused: {outcome}"
        assert _PRIVILEGED_RESULT in outcome["result"]

    def test_a_descendant_of_the_spawned_process_succeeds(self):
        """DIRECTION 1, one generation down.

        CLI agents dispatch tool calls from workers they spawn themselves, so
        an equality check on the authorized PID refuses exactly the traffic
        this path exists to serve.  Both other direction-1 legs stay green
        under that mistake; only this one turns red.
        """
        with _ServedTool() as served:
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    sys.executable,
                    "-c",
                    _RELAY_PROGRAM,
                    served.handle.server_url,
                    served.handle.token,
                    _CALLER_PROGRAM,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Authorize only the direct child; the CALL comes from its child.
            served.server.authorize_subprocess(proc.pid)
            stdout, stderr = proc.communicate(timeout=90)

        outcome = _read_outcome(stdout, stderr)
        assert outcome[
            "ok"
        ], f"a descendant of the authorized process was refused: {outcome}"
        assert _PRIVILEGED_RESULT in outcome["result"]

    def test_requests_before_authorization_are_refused(self):
        """The window between listening and spawned must be closed.

        The server has to be listening before the injector can write the URL
        the subprocess connects to, so the token is on disk while nothing is
        yet authorized.  Serving during that window would leave the whole
        exposure open to whoever asks first.
        """
        server = EphemeralMcpServer([_privileged_tool()])
        with server as handle:
            # Nothing authorized yet: even the eventual caller is refused.
            outcome = _call_in_child(handle.server_url, handle.token)
        assert not outcome["ok"], f"served a request before authorization: {outcome}"


#: A stand-in CLI: reads the MCP config the injector wrote, calls the tool the
#: way a real CLI agent would, and prints what it got. Substituting for the
#: CLI rather than for the server is what keeps this a test of the production
#: path: everything between build_mcp_node and the tool is real.
_FAKE_CLI_PROGRAM = _CLIENT_SOURCE + textwrap.dedent(
    """
    cfg_path = sys.argv[sys.argv.index("--mcp-config") + 1]
    with open(cfg_path, encoding="utf-8") as fh:
        entry = next(iter(json.load(fh)["mcpServers"].values()))
    outcome = main(entry["url"], entry["headers"]["Authorization"].split()[-1])
    sys.stdin.read()
    sys.stdout.write(json.dumps(outcome))
    """
)


class TestBuildMcpNodeEndToEnd:
    """The production path, with only the CLI binary substituted.

    Everything else is real: a real ephemeral server, the real Claude-Code
    injector writing a real config file, a real spawned process reading it,
    and a real MCP tool call back over loopback.  This is the only place the
    wiring between build_mcp_node and the server is observable; with the
    server mocked, a node that never authorizes the process it spawned
    produces byte-identical output.
    """

    def _node(self, tmp_path):
        from bili.iris.mcp.cli_injectors import ClaudeCodeInjector
        from bili.iris.mcp.server import build_mcp_node

        llm = SimpleNamespace(
            command=[sys.executable, "-c", _FAKE_CLI_PROGRAM],
            message_format="last",
            output_format="text",
            json_path="content",
            strip_ansi_output=False,
            timeout_seconds=120.0,
            cwd=str(tmp_path),
            model=None,
            reasoning_effort=None,
            model_flag_template=None,
            reasoning_effort_flag_template=None,
        )
        return build_mcp_node(
            llm_model=llm, tools=[_privileged_tool()], injector=ClaudeCodeInjector()
        )

    def test_the_spawned_cli_can_call_the_agents_tools(self, tmp_path):
        """DIRECTION 1, over the real path.

        A node that spawns the CLI and never authorizes it passes every other
        test in the suite and fails this one, because the CLI's tool call is
        refused and the privileged result never comes back.
        """
        from langchain_core.messages import HumanMessage

        state = {"messages": [HumanMessage(content="do the thing")]}
        result = self._node(tmp_path)(state)

        content = result["messages"][0].content
        outcome = json.loads(content.strip().splitlines()[-1])
        assert outcome["ok"], f"the spawned CLI was refused its own tools: {outcome}"
        assert _PRIVILEGED_RESULT in outcome["result"]


class TestPeerAuthorization:
    """Unit-level behaviour of the authorization object itself."""

    def test_denies_before_anything_is_authorized(self):
        auth = PeerAuthorization()
        assert auth.authorized is None
        assert auth.permits(peer_port=1, server_port=2) is False

    def test_revoke_returns_to_denying(self):
        auth = PeerAuthorization()
        auth.authorize_subprocess(os.getpid())
        assert auth.authorized is not None
        auth.revoke()
        assert auth.authorized is None
        assert auth.permits(peer_port=1, server_port=2) is False

    def test_identity_carries_creation_time(self):
        identity = ProcessIdentity.of(os.getpid())
        assert identity.pid == os.getpid()
        assert identity.create_time > 0

    def test_a_recycled_pid_does_not_inherit_the_grant(self):
        """Identity is (pid, create_time), so a PID alone cannot be trusted.

        A grant naming only a number would transfer to whatever process next
        holds it, and the window this server lives in is exactly the window in
        which the spawned process may exit.
        """
        auth = PeerAuthorization()
        auth.authorize_subprocess(os.getpid())
        # Same PID, different process: what PID reuse looks like from here.
        stale = ProcessIdentity(
            pid=os.getpid(), create_time=auth.authorized.create_time - 1000
        )
        assert stale.resolve() is None

    def test_a_grant_for_an_exited_process_permits_nothing(self):
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", "pass"]
        )
        proc.wait(timeout=30)
        auth = PeerAuthorization()
        try:
            auth.authorize_subprocess(proc.pid)
        except Exception:  # pylint: disable=broad-exception-caught
            # Already reaped before we could capture it; the grant never
            # existed, which is the same outcome the assertion below checks.
            return
        # Give the OS a moment to retire the entry, then confirm nothing is
        # permitted through a grant whose process is gone.
        for _ in range(20):
            if auth.authorized.resolve() is None:
                break
            time.sleep(0.05)
        assert auth.permits(peer_port=1, server_port=2) is False

    def test_attributes_a_real_connection_to_the_owning_subtree(self):
        """The positive half of the lookup, against a real socket.

        Paired with the negative below: a `permits` that always returned True
        would satisfy this alone.
        """
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        server_port = listener.getsockname()[1]

        child = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                "-c",
                "import socket,sys,time;"
                f"s=socket.create_connection(('127.0.0.1',{server_port}));"
                "print(s.getsockname()[1], flush=True); time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            child_peer_port = int(child.stdout.readline().strip())
            auth = PeerAuthorization()
            auth.authorize_subprocess(child.pid)

            assert auth.permits(child_peer_port, server_port) is True

            # This process's own connection to the same server is not in the
            # authorized subtree and must not be attributed to it.
            mine = socket.create_connection(("127.0.0.1", server_port))
            try:
                assert auth.permits(mine.getsockname()[1], server_port) is False
            finally:
                mine.close()
        finally:
            child.kill()
            child.wait(timeout=10)
            listener.close()
