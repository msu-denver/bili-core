"""``ask_user`` -- a generic agent-callable human-in-the-loop tool.

Lets an agent pause its own turn to ask the human operating it a question it
cannot answer from the conversation or its other tools, then continue with
the answer folded back into context.

One ``TOOL_REGISTRY`` entry serves TWO pause mechanisms, because
:func:`~bili.aether.compiler.llm_resolver.resolve_tools` resolves an agent's
tool list before ``tool_strategy`` is known (a compile-time ordering this
module does not control), so the same tool object must work correctly
whichever path ends up calling it:

- **Native tool-calling** (``tool_strategy="native"``, i.e.
  ``langchain.agents.create_agent`` / ``bind_tools``): the tool's ``func``
  calls :func:`langgraph.types.interrupt`, which requires a LangGraph node to
  call it from and pauses the whole graph until
  :meth:`~bili.aether.runtime.executor.MASExecutor.resume_with_value` supplies
  the answer.
- **CLI/MCP** (``tool_strategy="mcp"``): the tool is served over the
  ephemeral authenticated MCP server (:mod:`bili.iris.mcp.server`) to a
  spawned CLI agent that self-orchestrates. There is no LangGraph node to
  interrupt there, so the tool's ``func`` instead blocks by calling the
  registered :class:`~bili.iris.tools.hitl.HitlResponder` directly.

``func`` therefore dispatches to exactly one of two separate, single-purpose
implementations (:func:`_ask_user_native_impl`, :func:`_ask_user_mcp_impl`);
the dispatcher itself carries no HITL logic of its own. The prompted
(facilitated) ReAct loop needs a third mechanism and is not implemented by
this module.

Dispatch signal
----------------
Calling ``langchain_core.runnables.Runnable.invoke()`` at all -- which both
``langgraph.prebuilt.tool_node.ToolNode`` (native path) and
:func:`bili.iris.mcp.server._build_mcp_fn` (MCP path) do -- sets LangChain's
own ambient ``RunnableConfig`` propagation contextvar regardless of whether a
LangGraph graph is actually driving the call. That makes
``langgraph.config.get_config()`` indistinguishable between the two paths: it
succeeds in BOTH, so it cannot be used as the dispatch signal (confirmed by a
real MCP round trip: a bare ``tool.invoke()`` call already populates
``RunnableConfig``, so ``get_config()`` returns cleanly outside any graph).
The dispatcher instead reads :data:`bili.iris.mcp.server.IN_MCP_TOOL_CALL`,
a contextvar bili-core's own MCP bridge sets around its ``tool.invoke(...)``
call -- true only for the duration of a call the ephemeral MCP server itself
made, which is the actual distinction needed.

No ``ask_user``-specific server code was added for this: ``IN_MCP_TOOL_CALL``
is generic (any tool wanting to detect "am I being called through the MCP
bridge" can read it), and ``EphemeralMcpServer`` still registers whatever
LangChain tool it is given without knowing anything about ``ask_user``
itself -- ``ask_user`` reaches it through the exact same
``resolve_tools -> build_mcp_node`` route as every other tool.

Usage
-----
::

    from bili.iris.tools.ask_user import register_ask_user_tool, unregister_ask_user_tool
    from bili.iris.tools.hitl import ScriptedHitlResponder

    register_ask_user_tool(ScriptedHitlResponder(["staging"]))
    try:
        tools = initialize_tools(active_tools=["ask_user"], tool_prompts={})
        # ... build and run a graph whose agent has these tools ...
    finally:
        unregister_ask_user_tool()

A host that never calls :func:`register_ask_user_tool` still gets a working
``ask_user`` entry once it is registered with the default
:class:`~bili.iris.tools.hitl.NullHitlResponder` (see that function's
default), so a MAS config that lists ``ask_user`` does not fail to compile
in an environment where no real responder is wired up -- it degrades to the
no-response sentinel instead.
"""

from typing import List, Optional

from langchain_core.tools import StructuredTool

from bili.iris.loaders.tools_loader import TOOL_REGISTRY
from bili.iris.tools.hitl import HitlResponder, NullHitlResponder
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

#: The fixed TOOL_REGISTRY / tool name. Not user-configurable -- unlike the
#: domain tools in tools_loader.py (weather, SERP, ...), ask_user has no
#: per-deployment parameters, so there is nothing for a caller to customize
#: by choosing a different registry key.
ASK_USER_TOOL_NAME = "ask_user"

_ASK_USER_DESCRIPTION = (
    "Ask the human operating this agent a question and wait for their answer. "
    "Use this when a decision requires information or judgment only that "
    "person has -- not something inferable from the conversation or your "
    "other tools. Do not use it for routine clarification the task "
    "instructions already answer. The optional 'options' list is a "
    "rendering hint for short suggested answers; the human can always "
    "reply with free text instead. The response may be a sentinel "
    "starting with '[no response:' if no human answer was available -- "
    "treat that as 'no answer', not as a real answer, and proceed "
    "accordingly (e.g. state your own best judgment and note that no "
    "human input was available, rather than inventing an answer)."
)


def _ask_user_native_impl(question: str, options: Optional[List[str]]) -> str:
    """Native tool-calling pause: block via ``langgraph.types.interrupt``.

    ``interrupt()`` raises ``GraphInterrupt`` on first invocation within a
    task, which ``langgraph.prebuilt.tool_node.ToolNode`` re-raises rather
    than converting into a ``ToolMessage(status="error")`` -- so the pause
    propagates cleanly out of native tool execution instead of being
    swallowed as a tool failure. On resume within the same task, the second
    ``interrupt()`` call returns the previously supplied resume value instead
    of pausing again.

    Does not call a :class:`~bili.iris.tools.hitl.HitlResponder` -- on this
    path the pause/resume IS the answer-delivery mechanism (the caller
    resumes the graph with the answer via ``Command(resume=...)``, see
    :meth:`~bili.aether.runtime.executor.MASExecutor.resume_with_value`).
    """
    from langgraph.types import interrupt  # pylint: disable=import-outside-toplevel

    payload = {"type": ASK_USER_TOOL_NAME, "question": question, "options": options}
    answer = interrupt(payload)
    return str(answer)


def _ask_user_mcp_impl(
    question: str, options: Optional[List[str]], responder: HitlResponder
) -> str:
    """CLI/MCP pause: block by calling *responder* directly.

    There is no LangGraph node to interrupt here -- the MCP tool handler
    runs as a plain function on the ephemeral server's own event loop
    (:mod:`bili.iris.mcp.server`), invoked by a spawned CLI subprocess that
    self-orchestrates outside any graph bili-core drives. The block IS the
    delivery mechanism: this call does not return until *responder* does,
    which is exactly what leaves the CLI subprocess's own outstanding tool
    call pending for as long as the human takes to answer.
    """
    return responder.ask(question, options)


def _called_via_mcp_bridge() -> bool:
    """True if the current call originated from the ephemeral MCP server.

    Reads :data:`bili.iris.mcp.server.IN_MCP_TOOL_CALL`, set by
    :func:`bili.iris.mcp.server._build_mcp_fn` around its ``tool.invoke(...)``
    call. This is NOT the same question as "is a LangGraph graph driving this
    call" -- ``langgraph.config.get_config()`` cannot answer that reliably
    here, because calling ``Runnable.invoke()`` at all (which both the native
    ``ToolNode`` path and the MCP bridge do) populates LangChain's ambient
    ``RunnableConfig`` regardless of LangGraph involvement (see the module
    docstring's "Dispatch signal" section for the concrete finding). Lazily
    imports :mod:`bili.iris.mcp.server` so that importing this module does
    not require the ``[mcp]`` extra when the MCP path is never used.
    """
    from bili.iris.mcp.server import (  # pylint: disable=import-outside-toplevel
        IN_MCP_TOOL_CALL,
    )

    return IN_MCP_TOOL_CALL.get()


def _build_ask_user_func(responder: HitlResponder):
    """Return the tool ``func``, dispatching to the native or MCP pause path.

    A thin context check, not business logic: the two pause mechanisms
    (:func:`_ask_user_native_impl`, :func:`_ask_user_mcp_impl`) are separate
    functions because they are genuinely different operations (one raises to
    pause a graph, one blocks synchronously calling out to a host callback),
    not two branches of one algorithm.
    """

    def _ask_user(question: str, options: Optional[List[str]] = None) -> str:
        if _called_via_mcp_bridge():
            return _ask_user_mcp_impl(question, options, responder)
        return _ask_user_native_impl(question, options)

    return _ask_user


def register_ask_user_tool(responder: Optional[HitlResponder] = None) -> None:
    """Insert ``ask_user`` into ``TOOL_REGISTRY``.

    Mirrors :func:`bili.iris.mcp.loader.register_mcp_tools`'s pattern of
    inserting a pre-built LangChain tool into
    :data:`~bili.iris.loaders.tools_loader.TOOL_REGISTRY` under a
    ``(name, prompt, params) -> Tool`` factory that ignores its arguments,
    since (unlike the domain tools in ``tools_loader.py``) ``ask_user`` has
    no per-call configuration.

    Calling this twice without an intervening :func:`unregister_ask_user_tool`
    replaces the previously registered tool (and its responder); a warning is
    logged so a host does not silently lose track of a prior registration.

    :param responder: The host's :class:`~bili.iris.tools.hitl.HitlResponder`,
        consulted on the CLI/MCP tool-strategy path (there is no LangGraph
        node to interrupt there). The native (``create_agent``) path does
        not call *responder* -- ``interrupt()`` / ``Command(resume=...)`` IS
        the answer-delivery mechanism on that path; *responder* is inert for
        agents that never resolve to ``tool_strategy="mcp"``.
        Defaults to :class:`~bili.iris.tools.hitl.NullHitlResponder` when
        omitted, so an unconfigured CLI-path ``ask_user`` call degrades to
        the no-response sentinel instead of raising.
    """
    if ASK_USER_TOOL_NAME in TOOL_REGISTRY:
        LOGGER.warning(
            "register_ask_user_tool: '%s' is already registered; replacing "
            "the existing tool.",
            ASK_USER_TOOL_NAME,
        )

    effective_responder = responder if responder is not None else NullHitlResponder()

    tool = StructuredTool.from_function(
        func=_build_ask_user_func(effective_responder),
        name=ASK_USER_TOOL_NAME,
        description=_ASK_USER_DESCRIPTION,
    )

    # Ignores (name, prompt, params) -- ask_user has no per-registration
    # config for a TOOL_REGISTRY caller to pass through initialize_tools().
    TOOL_REGISTRY[ASK_USER_TOOL_NAME] = (
        lambda _n, _p, _params, _t=tool: _t  # noqa: E731
    )
    LOGGER.info(
        "TOOL_REGISTRY: registered '%s' (responder=%s)",
        ASK_USER_TOOL_NAME,
        type(effective_responder).__name__,
    )


def unregister_ask_user_tool() -> None:
    """Remove ``ask_user`` from ``TOOL_REGISTRY``, if present.

    No-op (not an error) when ``ask_user`` is not currently registered, so
    callers can call this unconditionally in teardown/``finally`` blocks.
    """
    if ASK_USER_TOOL_NAME in TOOL_REGISTRY:
        del TOOL_REGISTRY[ASK_USER_TOOL_NAME]
        LOGGER.debug("TOOL_REGISTRY: removed '%s'", ASK_USER_TOOL_NAME)


__all__ = [
    "ASK_USER_TOOL_NAME",
    "register_ask_user_tool",
    "unregister_ask_user_tool",
]
