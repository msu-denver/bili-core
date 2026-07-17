"""Human-in-the-loop responder seam for the ``ask_user`` tool.

:class:`HitlResponder` is the one surface-agnostic contract a host implements
to answer questions an agent raises mid-run via the ``ask_user`` tool
(:mod:`bili.iris.tools.ask_user`). bili-core never renders a question itself
and never picks a delivery surface (CLI prompt, HTTP endpoint, desktop modal,
a message queue) -- that is entirely the host's responsibility. bili-core
only defines the blocking call shape and the no-answer sentinel convention so
an agent can tell "the human answered" apart from "no answer was available."

This module carries no LLM-application concept beyond that seam: it does not
know what a question is *about*, does not validate or store content, and does
not impose a timeout policy. A host that needs a timeout implements it inside
its own :meth:`HitlResponder.ask` (see :class:`ScriptedHitlResponder` for the
shape of a minimal implementation).
"""

import threading
from typing import List, Optional, Protocol, runtime_checkable

from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Prefix every no-answer sentinel returned by a HitlResponder must carry.
#: Callers (agents, tests, and bili-core's own NullHitlResponder) check for
#: this prefix rather than parsing free-text, so any HitlResponder
#: implementation can signal "no answer" without bili-core needing to know
#: why (timeout, explicit skip, or -- for NullHitlResponder -- "nothing was
#: ever registered").
NO_RESPONSE_PREFIX = "[no response:"


@runtime_checkable
# pylint: disable=too-few-public-methods
# A single-method protocol by design: ask() is the entire seam, matching
# HitlResponder's own docstring ("the ONE surface-agnostic contract").
class HitlResponder(Protocol):
    """Host-implemented callback that blocks until a human answers a question.

    A single instance is registered once per process (or run) via
    :func:`bili.iris.tools.ask_user.register_ask_user_tool` and is shared by
    every ``ask_user`` call that registration produced, across BOTH pause
    mechanisms (the native ``interrupt()`` path calls :meth:`ask` indirectly
    via a graph resume, the CLI/MCP path calls it directly and blocks on it).

    Thread safety is a host-side contract this protocol does not enforce but
    every implementation must honor: more than one ``ask_user`` call can be
    in flight AT THE SAME TIME against the same responder instance whenever
    a host runs multiple agent RUNS concurrently (e.g. a fan-out batch of N
    MAS runs, each potentially raising its own question). Each in-flight
    call blocks the thread that made it, so :meth:`ask` -- and any shared
    state it reads or writes -- must be safe to call from multiple threads
    at once. A responder that renders questions into a single-consumer UI
    (one prompt at a time) must serialize or queue internally rather than
    silently interleaving or corrupting concurrent calls.
    """

    def ask(self, question: str, options: Optional[List[str]] = None) -> str:
        """Block until a human answers *question*, then return the answer.

        :param question: The question to surface to the human, already in
            human-readable form -- bili-core does no further formatting.
        :param options: Optional short list of suggested answers. Render as
            quick-pick choices if the surface supports it; free-text must
            always remain a valid answer regardless of *options*.
        :returns: The human's answer as plain text. On timeout or an
            explicit skip, return a string starting with
            :data:`NO_RESPONSE_PREFIX` instead of raising, so the calling
            agent can branch on "no answer" the same way it would on any
            other tool observation.
        """
        # pylint: disable-next=unnecessary-ellipsis
        ...  # pragma: no cover -- Protocol method body, never actually called


# pylint: disable=too-few-public-methods
# A single-method HitlResponder implementation by design -- see the
# module-level disable comment on HitlResponder itself.
class NullHitlResponder:
    """Default responder when no host has registered one.

    Exists so an agent whose config declares ``ask_user`` but runs in a host
    that never wired up a real responder degrades to a clear sentinel answer
    rather than crashing the whole run on a missing dependency. This is the
    deliberate default of :func:`~bili.iris.tools.ask_user.register_ask_user_tool`
    when called without an explicit *responder* -- it makes an unconfigured
    ``ask_user`` visibly inert instead of silently unavailable.
    """

    def ask(  # pylint: disable=unused-argument
        self, question: str, options: Optional[List[str]] = None
    ) -> str:
        """Return the no-responder sentinel without blocking."""
        LOGGER.warning(
            "ask_user called but no HitlResponder is registered; "
            "returning the no-response sentinel. Call "
            "register_ask_user_tool(responder=...) with a real HitlResponder "
            "to enable ask_user for this process."
        )
        return f"{NO_RESPONSE_PREFIX} ask_user not configured]"


class ScriptedHitlResponder:
    """Test double that returns pre-scripted answers in order.

    Not a production responder. Used by bili-core's own tests, and usable by
    a host's tests, to exercise the ``ask_user`` pause/resume seam without a
    real human or a real event loop.

    Thread-safe: a lock guards the append-then-index sequence in :meth:`ask`
    so concurrent calls (e.g. a test that fans out several agent runs against
    one shared responder) get distinct, correctly-ordered answers rather than
    racing on the shared call count.
    """

    def __init__(self, answers: List[str]) -> None:
        """:param answers: Answers returned in order, one per :meth:`ask` call."""
        self._answers = list(answers)
        self._calls: List[dict] = []
        self._lock = threading.Lock()

    def ask(self, question: str, options: Optional[List[str]] = None) -> str:
        """Record the call and return the next scripted answer.

        :raises IndexError: If called more times than there are scripted
            answers -- a test bug (the script under-provisioned answers),
            not a runtime condition to degrade gracefully from.
        """
        with self._lock:
            self._calls.append({"question": question, "options": options})
            return self._answers[len(self._calls) - 1]

    @property
    def calls(self) -> List[dict]:
        """The ``{"question", "options"}`` dicts recorded for each :meth:`ask` call."""
        with self._lock:
            return list(self._calls)


__all__ = [
    "NO_RESPONSE_PREFIX",
    "HitlResponder",
    "NullHitlResponder",
    "ScriptedHitlResponder",
]
