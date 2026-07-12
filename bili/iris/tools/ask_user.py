"""``ask_user`` -- a generic agent-callable human-in-the-loop tool.

Lets an agent pause its own turn to ask the human operating it a question it
cannot answer from the conversation or its other tools, then continue with
the answer folded back into context. This is the native tool-calling path
only (``tool_strategy="native"``, i.e. ``langchain.agents.create_agent`` /
``bind_tools``): the tool's function calls :func:`langgraph.types.interrupt`,
which requires a LangGraph node to call it from. The CLI/MCP tool-strategy
path and the prompted (facilitated) ReAct loop need a different pause
mechanism each (no LangGraph node to interrupt on either) and are not
implemented by this module.

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


def _build_ask_user_func():
    """Return the tool ``func`` for the native tool-calling pause path.

    Calls :func:`langgraph.types.interrupt` so the pause happens inside the
    calling LangGraph node (native tool-calling / ``create_agent`` path
    only). ``interrupt()`` raises ``GraphInterrupt`` on first invocation
    within a task, which ``langgraph.prebuilt.tool_node.ToolNode`` re-raises
    rather than converting into a ``ToolMessage(status="error")`` -- so the
    pause propagates cleanly out of native tool execution instead of being
    swallowed as a tool failure. On resume within the same task, the second
    ``interrupt()`` call returns the previously supplied resume value instead
    of pausing again.

    Does not call a :class:`~bili.iris.tools.hitl.HitlResponder` -- on this
    path the pause/resume IS the answer-delivery mechanism (the caller
    resumes the graph with the answer via ``Command(resume=...)``, see
    :meth:`~bili.aether.runtime.executor.MASExecutor.resume_with_value`).
    A ``HitlResponder`` is only consulted on the CLI/MCP tool-strategy path,
    where there is no LangGraph node to interrupt (tracked separately, not
    built here).
    """

    def _ask_user(question: str, options: Optional[List[str]] = None) -> str:
        # pylint: disable=import-outside-toplevel
        from langgraph.types import interrupt

        payload = {"type": ASK_USER_TOOL_NAME, "question": question, "options": options}
        answer = interrupt(payload)
        return str(answer)

    return _ask_user


def register_ask_user_tool(responder: Optional[HitlResponder] = None) -> None:
    """Insert ``ask_user`` into ``TOOL_REGISTRY``.

    Mirrors :func:`bili.iris.mcp.loader.register_mcp_tools`'s pattern of
    inserting a pre-built LangChain tool into
    :data:`~bili.iris.loaders.tools_loader.TOOL_REGISTRY` under a
    ``(name, prompt, params) -> Tool`` factory that ignores its arguments,
    since (unlike the domain tools in ``tools_loader.py``) ``ask_user`` has
    no per-call configuration on the native tool-calling path.

    Calling this twice without an intervening :func:`unregister_ask_user_tool`
    replaces the previously registered tool; a warning is logged so a host
    does not silently lose track of a prior registration.

    :param responder: Reserved for the CLI/MCP tool-strategy path, where
        there is no LangGraph node to interrupt and a
        :class:`~bili.iris.tools.hitl.HitlResponder` is the only pause
        mechanism available. The native (``create_agent``) tool built here
        does not call *responder* -- on that path ``interrupt()`` /
        ``Command(resume=...)`` IS the answer-delivery mechanism. Passing
        *responder* now is forward-compatible with the CLI/MCP path landing
        later; it has no effect on native tool-calling behavior today.
        Defaults to :class:`~bili.iris.tools.hitl.NullHitlResponder` when
        omitted.
    """
    if ASK_USER_TOOL_NAME in TOOL_REGISTRY:
        LOGGER.warning(
            "register_ask_user_tool: '%s' is already registered; replacing "
            "the existing tool.",
            ASK_USER_TOOL_NAME,
        )

    # Constructed for parity with the eventual CLI/MCP-path registration
    # (which will actually call it) and to validate *responder* satisfies
    # the HitlResponder protocol now rather than failing later, silently,
    # the first time a CLI-path ask_user call needs it.
    effective_responder = responder if responder is not None else NullHitlResponder()

    tool = StructuredTool.from_function(
        func=_build_ask_user_func(),
        name=ASK_USER_TOOL_NAME,
        description=_ASK_USER_DESCRIPTION,
    )

    # Ignores (name, prompt, params) -- ask_user has no per-registration
    # config for a TOOL_REGISTRY caller to pass through initialize_tools().
    TOOL_REGISTRY[ASK_USER_TOOL_NAME] = (
        lambda _n, _p, _params, _t=tool: _t  # noqa: E731
    )
    LOGGER.info(
        "TOOL_REGISTRY: registered '%s' (responder=%s reserved for the "
        "CLI/MCP tool-strategy path)",
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
