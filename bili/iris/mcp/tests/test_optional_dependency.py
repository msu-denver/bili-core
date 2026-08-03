"""The optional-dependency guard must reflect reality, not merely not crash.

``bili.iris.mcp.server`` decides at import whether the MCP subsystem is usable
by trying its imports and catching :class:`ImportError`.  That guard cannot
tell "the extra is not installed" apart from "the extra is installed and its
API moved", and it answers the second case with the first case's message.  The
subsystem then turns itself off: the server raises "install
bili-core[mcp]" at people who did, and every test that skips on
``_MCP_AVAILABLE`` skips, so a whole subsystem stops being exercised with
nothing in the run going red.

That is not hypothetical.  ``mcp`` 2.0 removed ``mcp.server.fastmcp``, the
module the ephemeral server is built from, and the extra's floor-only pin
(``mcp>=1.0``) resolved straight to it.

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
