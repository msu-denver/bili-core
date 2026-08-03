"""Bind an ephemeral-server connection to the process tree that was spawned.

A bearer token proves the caller holds a secret.  It cannot prove the caller
*is* the subprocess the server was started for, because the secret has to be
delivered through a file or an environment variable and every process running
as the same user can read both.  This module supplies the missing half: given
the peer address of an inbound loopback connection, decide whether that
connection belongs to the spawned subprocess or to one of its descendants.

Why the usual mechanism is not used
-----------------------------------
Peer credentials (``SO_PEERCRED`` on Linux, ``LOCAL_PEERCRED`` on macOS) are
the textbook answer and are not reachable here.  They are a property of unix
domain sockets, and the MCP specification defines two transports: stdio, and
Streamable HTTP, which is defined over an HTTP URL.  The ephemeral server
speaks Streamable HTTP over loopback TCP, and a TCP connection carries no
peer identity at all.

What is done instead
--------------------
The connection is attributed by asking the *authorized subtree* which
connections it owns, rather than asking the operating system who the caller
is.  That inversion matters twice: it is bounded by the size of the spawned
CLI's own process tree instead of by the number of processes on the host, and
a caller that cannot be attributed is refused without ever having to be
identified, so an unresolvable peer fails closed for free.

Descendants, not equality
-------------------------
The spawned CLI is not necessarily the process that opens the connection;
CLI agents routinely make their tool calls from workers they spawn
themselves.  Authorization therefore covers the spawned process **and its
descendants**, and an equality check on the spawned PID would refuse exactly
the traffic this path exists to serve.

Identity is ``(pid, create_time)``
----------------------------------
A PID alone is not an identity.  PIDs are reused, and the window this server
lives in is precisely the window in which the spawned process may exit, so a
recycled PID could inherit an authorization that was granted to something
else.  :class:`ProcessIdentity` pairs the PID with the process creation time,
which is what makes the grant refer to one process rather than to a number.

What this does and does not defend against
------------------------------------------
It refuses the opportunistic case: another process running as the same user
that has read the token out of the configuration file or the environment and
calls the server directly.  On a single-user workstation that is the case
that matters, and it covers editor plugins, shell helpers, and other agents.

It does **not** defend against a same-user attacker who injects into, or
attaches to, the spawned process tree.  Same-user isolation is inherently
weak: a process can generally debug another process running as the same user,
and anything reached from inside the authorized subtree is indistinguishable
from the subtree's own traffic by construction.  This raises the bar from
"holds the secret" to "is inside the spawned process tree"; it does not make
the boundary a security boundary between programs run by one user.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

try:
    import psutil  # type: ignore[import-untyped]

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover — gated by the [mcp] extra
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


@dataclass(frozen=True)
class ProcessIdentity:
    """A process, identified by PID *and* creation time.

    :param pid: The operating-system process id.
    :param create_time: The process's creation time, as reported by psutil.
        Present because a PID on its own is reused and would let a later,
        unrelated process inherit this grant.
    """

    pid: int
    create_time: float

    @classmethod
    def of(cls, pid: int) -> "ProcessIdentity":
        """Capture the identity of the currently running process *pid*.

        :param pid: The process id to capture.
        :returns: A :class:`ProcessIdentity` pinned to that process.
        :raises RuntimeError: If psutil is unavailable.
        :raises psutil.NoSuchProcess: If *pid* is not running.
        """
        _require_psutil()
        return cls(pid=pid, create_time=psutil.Process(pid).create_time())

    def resolve(self) -> Optional["psutil.Process"]:
        """Return the live process for this identity, or ``None``.

        ``None`` covers both "the process has exited" and "the PID exists but
        belongs to something else now", which are the same answer for
        authorization purposes: this grant no longer refers to anything.
        """
        try:
            proc = psutil.Process(self.pid)
            if proc.create_time() != self.create_time:
                return None
            return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


def _require_psutil() -> None:
    """Raise if psutil is missing, naming the extra that provides it."""
    if not _PSUTIL_AVAILABLE:  # pragma: no cover — gated by the [mcp] extra
        raise RuntimeError(
            "The 'psutil' package is required to bind the ephemeral MCP server "
            "to the process tree it was started for.  Install with: "
            "pip install bili-core[mcp]"
        )


def _subtree(root: "psutil.Process") -> List["psutil.Process"]:
    """Return *root* and every descendant of it that is still running."""
    try:
        return [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
        return [root]


class PeerAuthorization:
    """Decides whether an inbound connection belongs to the authorized subtree.

    Starts in a **deny-all** state and stays there until
    :meth:`authorize_subprocess` records the spawned process.  That window is
    real rather than theoretical: the server must be listening before the
    subprocess can be told where to connect, so the token is already on disk
    while nothing is yet authorized.  Answering "allow" during it would leave
    the whole exposure open to whoever asks first.

    Instances are shared between the caller's thread (which authorizes) and
    the server thread (which checks), so all state changes are taken under a
    lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._root: Optional[ProcessIdentity] = None

    @property
    def authorized(self) -> Optional[ProcessIdentity]:
        """The currently authorized process, or ``None`` if none is."""
        with self._lock:
            return self._root

    def authorize_subprocess(self, pid: int) -> ProcessIdentity:
        """Authorize *pid* and its descendants for the life of this server.

        Call immediately after spawning the subprocess.  The identity is
        captured at this moment, so a later process that reuses the PID does
        not inherit the grant.

        :param pid: The spawned subprocess's process id.
        :returns: The captured :class:`ProcessIdentity`.
        :raises RuntimeError: If psutil is unavailable.
        :raises psutil.NoSuchProcess: If the subprocess has already exited.
        """
        identity = ProcessIdentity.of(pid)
        with self._lock:
            self._root = identity
        LOGGER.debug("PeerAuthorization: authorized subprocess tree at pid %d", pid)
        return identity

    def revoke(self) -> None:
        """Return to the deny-all state."""
        with self._lock:
            self._root = None

    def permits(self, peer_port: int, server_port: int) -> bool:
        """Is the connection from *peer_port* owned by the authorized subtree?

        Enumerates the authorized process and its descendants and asks each
        which TCP connections it holds, looking for the one whose local port
        is *peer_port* and whose remote port is this server's.  A connection
        no member of the subtree claims is refused, which is also the answer
        for a caller belonging to another user, since their processes are not
        in the subtree and their connections are therefore never claimed.

        :param peer_port: The client port, from the ASGI scope's ``client``.
        :param server_port: The port this ephemeral server listens on.
        :returns: ``True`` only if a member of the authorized subtree owns
            that connection.
        """
        root_identity = self.authorized
        if root_identity is None:
            LOGGER.debug(
                "PeerAuthorization: refusing peer port %d; no subprocess is "
                "authorized yet",
                peer_port,
            )
            return False

        root = root_identity.resolve()
        if root is None:
            LOGGER.debug(
                "PeerAuthorization: refusing peer port %d; the authorized "
                "process (pid %d) is no longer running",
                peer_port,
                root_identity.pid,
            )
            return False

        for proc in _subtree(root):
            try:
                connections = proc.net_connections(kind="tcp")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # A subtree member that exited mid-scan, or one this process
                # may not inspect. Neither can vouch for the connection, so
                # neither contributes an "allow"; the scan continues so a
                # sibling can still claim it.
                continue
            for conn in connections:
                if (
                    conn.laddr
                    and conn.raddr
                    and conn.laddr.port == peer_port
                    and conn.raddr.port == server_port
                ):
                    return True

        LOGGER.warning(
            "PeerAuthorization: refusing a request carrying a valid token from "
            "peer port %d; the connection is not owned by the authorized "
            "subprocess tree (pid %d) or any of its descendants.  The token "
            "was readable by another process running as this user.",
            peer_port,
            root_identity.pid,
        )
        return False


__all__ = [
    "PeerAuthorization",
    "ProcessIdentity",
]
