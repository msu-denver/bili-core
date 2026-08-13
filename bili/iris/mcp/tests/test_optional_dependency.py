"""The optional-dependency guard must reflect reality, not merely not crash.

``bili.iris.mcp.server`` decides at import whether the MCP subsystem is usable
by trying its imports and catching :class:`ImportError`.  That guard cannot
tell "the extra is not installed" apart from "the extra is installed and its
API moved", and it answers the second case with the first case's message.  The
subsystem then turns itself off: the server raises "install
bili-core[mcp]" at people who did, and every test that skips on
``_MCP_AVAILABLE`` skips, so a whole subsystem stops being exercised with
nothing in the run going red.

That is not hypothetical.  The ephemeral server is built from
``mcp.server.MCPServer``, which exists only in ``mcp`` 2.x; the 1.x
``mcp.server.fastmcp`` it was built from before was removed in the same 2.0
release, so the two APIs never coexist.  The extra pins ``mcp>=2.0,<3`` for
exactly this reason: a 1.x SDK fails the import guard, and the pin turns that
into a resolver error the operator sees instead of a subsystem that quietly
switches itself off.  The next major could move the API again, and this test
is what would then catch ``_MCP_AVAILABLE`` disagreeing with the environment.

The assertion below is two-sided and has no skip: whichever way the
environment is set up, something real is being claimed.
"""

# pylint: disable=missing-function-docstring

from importlib.metadata import PackageNotFoundError, version

from bili.iris.mcp.server import _MCP_AVAILABLE


def _installed(distribution: str) -> bool:
    """Is *distribution* installed in this environment?"""
    try:
        version(distribution)
        return True
    except PackageNotFoundError:
        return False


class TestMcpAvailabilityMatchesTheEnvironment:
    """``_MCP_AVAILABLE`` must agree with what is actually installed."""

    def test_availability_agrees_with_the_installed_distributions(self):
        both_installed = _installed("mcp") and _installed("uvicorn")

        if both_installed:
            assert _MCP_AVAILABLE, (
                "mcp and uvicorn are installed, but the ephemeral MCP server "
                "reports itself unavailable. The import guard catches "
                "ImportError, so an installed-but-incompatible SDK is "
                "indistinguishable from an absent one and silently disables "
                "the subsystem. Installed: "
                f"mcp=={version('mcp')}, uvicorn=={version('uvicorn')}. The "
                "supported range is declared by the [mcp] extra in setup.py."
            )
        else:
            assert not _MCP_AVAILABLE, (
                "the ephemeral MCP server reports itself available with "
                "mcp or uvicorn missing; the base install must stay lean and "
                "the guard is what keeps importing this module from failing."
            )
