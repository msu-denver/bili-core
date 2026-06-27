"""
react_agent_node.py
-------------------

This module provides a utility for constructing a REACT agent node for conversational
workflows.  It supports three operating modes, selected automatically based on the tools
provided and the ``supports_tools`` flag:

1. **Native tool-calling** (default when ``tools`` is provided and the model supports
   ``bind_tools``): builds a compiled LangGraph sub-graph via ``create_agent``.  The
   model must implement ``BaseChatModel.bind_tools``; all major API providers do.

2. **Prompted tool-calling** (when ``tools`` is provided but ``supports_tools=False``
   is passed via ``node_kwargs``): runs a hand-rolled ReAct loop entirely within the
   node.  Tools are injected into a system-message preamble, the model's text responses
   are parsed for ``Action:`` / ``Final Answer:`` markers, matched tools are executed,
   and observations are fed back as ``HumanMessage`` turns.  This path works with any
   model that can follow text instructions -- including CLI-backed models
   (e.g. ``CliLLM``) that do not implement ``bind_tools``.

3. **Tool-less fallback** (when ``tools`` is ``None``): calls the model directly,
   filtering ToolMessages from history for compatibility.

Capability detection
--------------------
Pass ``supports_tools=False`` in ``node_kwargs`` alongside a non-``None`` ``tools``
list to engage the prompted path.  The flag defaults to ``True`` so existing callers
that pass tools without the flag continue to use the native path unchanged.

``hasattr`` / ``try-bind_tools`` probing is intentionally avoided: LangChain's
``BaseChatModel.bind_tools`` is always present (it raises ``NotImplementedError`` for
models that do not override it), so attribute presence is not a reliable signal.

Prompted ReAct protocol
-----------------------
The system preamble instructs the model to respond using one of two formats::

    Thought: <reasoning>
    Action: <tool_name>
    Action Input: {"arg": value}

or::

    Thought: <reasoning>
    Final Answer: <response>

The parser scans the full response text with a regex, tolerating preamble, varying
capitalisation, and markdown code fences.  Parse failures and tool errors are fed back
as ``Observation: Error - <reason>`` turns so the model can self-correct.  Three
consecutive unparseable responses terminate the loop early.  A configurable
``max_react_iterations`` cap (default 10) prevents runaway loops.

Observations are returned as ``HumanMessage`` objects to satisfy strict-alternating-turn
constraints (Bedrock Converse API, some Mistral deployments).

Functions
---------
- ``build_react_agent_node(tools, state, llm_model, middleware, **kwargs)``:
  Entry point; selects the operating mode and returns the appropriate callable or
  compiled graph.

- ``_build_prompted_react_loop(llm_model, tools, max_react_iterations)``:
  Internal factory; returns the ``prompted_react_loop(state) -> dict`` callable that
  runs the ReAct loop for a single node invocation.

- ``_build_tool_preamble(tools)``:
  Internal helper; renders the tool-description block injected into the system message.

- ``_extract_json_object(text)``:
  Internal helper; returns the first brace-balanced JSON object or array found in
  *text*.  Used by ``_parse_react_response`` to correctly handle nested object args
  (e.g. ``{"filters": {"type": "exact"}}``).  A non-greedy regex alternative would
  truncate at the first inner ``}``.

Dependencies
------------
- ``langchain_core.messages``: ``AIMessage``, ``HumanMessage``, ``SystemMessage``
- ``langchain_core.tools``: ``Tool`` (type hint)
- ``langchain.agents``: ``AgentState``, ``create_agent``
- ``bili.utils.langgraph_utils.State``: conversation state schema
- ``bili.utils.logging_utils.get_logger``: structured logger

Usage
-----
Import and use ``build_react_agent_node`` to create a REACT agent node for
conversational agent workflows.

Example (native path -- model supports bind_tools)::

    agent_node = build_react_agent_node(
        tools=my_tools,
        state=MyStateSchema,
        llm_model=my_llm,
    )

Example (prompted path -- CLI or other text-only model)::

    agent_node = build_react_agent_node(
        tools=my_tools,
        llm_model=cli_llm,
        supports_tools=False,      # passed through **kwargs
        max_react_iterations=8,    # optional; default 10
    )
"""

import json
import re
from functools import partial
from typing import Type

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool

from bili.iris.graph_builder.classes.node import Node
from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default guard values for the prompted ReAct loop
# ──────────────────────────────────────────────────────────────────────────────

#: Maximum number of Thought/Action/Observation iterations before the loop
#: gives up and returns whatever the model last said.
_DEFAULT_MAX_REACT_ITERATIONS = 10

#: Number of consecutive responses that cannot be parsed as either an Action
#: or a Final Answer before the loop exits with the last response as-is.
_MAX_CONSECUTIVE_PARSE_FAILURES = 3


# ──────────────────────────────────────────────────────────────────────────────
# Prompted ReAct helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_tool_preamble(tools: list) -> str:
    """Render the tool-description block for the system-message preamble.

    Produces a plain-text listing of each tool's name, argument schema, and
    description, followed by the Action/Observation protocol the model must
    follow to invoke a tool or signal a final answer.

    :param tools: List of LangChain ``BaseTool`` instances to describe.
    :type tools: list
    :return: Multi-line string to prepend to (or substitute for) the system message.
    :rtype: str
    """
    lines = [
        "You have access to the following tools.  Use them whenever they help you",
        "answer the question accurately.\n",
        "TOOLS:",
    ]
    for tool in tools:
        # Extract argument names from the tool's args_schema when available
        try:
            schema = tool.args_schema.schema() if tool.args_schema else {}
            props = schema.get("properties", {})
            arg_parts = []
            for arg_name, arg_info in props.items():
                arg_type = arg_info.get("type", "any")
                arg_parts.append(f"{arg_name}: {arg_type}")
            sig = f"({', '.join(arg_parts)})" if arg_parts else "()"
        except Exception:  # pylint: disable=broad-except
            sig = "(...)"
        description = getattr(tool, "description", "") or ""
        lines.append(f"  - {tool.name}{sig}: {description}")

    lines += [
        "",
        "RESPONSE FORMAT",
        "To call a tool, your response must contain EXACTLY:",
        "  Thought: <your reasoning>",
        "  Action: <tool_name>",
        "  Action Input: <JSON object of arguments>",
        "",
        "When you have the final answer (no more tools needed):",
        "  Thought: <your reasoning>",
        "  Final Answer: <your complete response to the user>",
        "",
        "Do not include any text after 'Final Answer:' on the same or later lines.",
        "Do not mix Action and Final Answer in the same response.",
    ]
    return "\n".join(lines)


# Regex patterns used by the response parser.  Compiled once at module load.
_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Anchors on the "Action Input:" label and captures everything that follows to
# end-of-string.  The actual JSON boundary is determined by _extract_json_object
# so that brace-balanced nested objects parse correctly.  A non-greedy \{.*?\}
# regex would truncate at the first "}" inside a nested object.
_ACTION_INPUT_ANCHOR_RE = re.compile(
    r"Action\s+Input\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    """Return the first brace-balanced JSON object (or array) from *text*.

    Scans forward from the first ``{`` (or ``[``), counting opening and closing
    brackets until the nesting depth returns to zero.  Returns the full balanced
    substring, which correctly handles nested objects such as::

        {"filters": {"type": "exact", "value": "foo"}, "limit": 5}

    Returns ``None`` if no ``{`` or ``[`` is found, or if the brackets are
    unbalanced (unclosed).

    :param text: String that may contain a JSON object or array.
    :type text: str
    :return: The balanced JSON substring, or ``None``.
    :rtype: str or None
    """
    open_chars = {"{": "}", "[": "]"}
    for start_idx, ch in enumerate(text):
        if ch not in open_chars:
            continue
        close_ch = open_chars[ch]
        depth = 0
        in_string = False
        escape_next = False
        for end_idx in range(start_idx, len(text)):
            c = text[end_idx]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start_idx : end_idx + 1]
        # Unbalanced — stop looking
        return None
    return None


def _parse_react_response(text: str):
    """Parse a model response for Action/Input or Final Answer markers.

    Scans the full response text (not line by line) to tolerate preamble
    and reasoning text that precedes the structured markers.  Returns a
    3-tuple:

    - ``("action", tool_name, args_dict)`` when an Action block is found and
      the Action Input JSON parses cleanly.
    - ``("final", answer_text, None)`` when a Final Answer marker is found.
    - ``("unknown", None, None)`` when neither marker is detected.
    - ``("parse_error", raw_action_input, error_message)`` when an Action block
      is found but the Action Input cannot be decoded as JSON.

    :param text: Raw text content from the model's response.
    :type text: str
    :return: A 3-tuple ``(kind, value, extra)`` as described above.
    :rtype: tuple
    """
    # Strip markdown code fences that some models add around structured output
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)

    # Check for Final Answer first -- if present, we're done regardless of Actions
    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        return ("final", final_match.group(1).strip(), None)

    # Check for Action block
    action_match = _ACTION_RE.search(text)
    if action_match:
        tool_name = action_match.group(1).strip()
        input_match = _ACTION_INPUT_ANCHOR_RE.search(text)
        if input_match:
            # Extract the brace-balanced JSON object from the remainder.
            # _extract_json_object handles nested structures correctly; the
            # old non-greedy \{.*?\} approach truncated at the first inner "}"
            # and broke tools that accept object-valued arguments.
            remainder = input_match.group(1)
            raw = _extract_json_object(remainder)
            if raw is None:
                # No { or [ found — fall through to the parse_error path
                raw = remainder.strip()
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as exc:
                return ("parse_error", raw, str(exc))
            return ("action", tool_name, args)
        # Action with no parseable Input -- treat the remainder after Action as input
        return ("parse_error", "", "Action Input block not found or not valid JSON")

    return ("unknown", None, None)


def _build_prompted_react_loop(
    llm_model,
    tools: list,
    max_react_iterations: int = _DEFAULT_MAX_REACT_ITERATIONS,
):
    """Factory that returns the prompted ReAct node callable.

    The returned callable runs entirely within a single LangGraph node invocation:
    it injects the tool descriptions into the system message, then iterates --
    invoking the model, parsing the response, executing any requested tool, and
    feeding back the observation -- until a Final Answer is found, the iteration
    cap is reached, or too many consecutive parse failures occur.

    The callable always returns ``{"messages": [AIMessage(content=...)]}`` with
    only the final answer, matching the shape produced by the tool-less
    ``call_model`` fallback and compatible with the ``wrap_node`` wrapper in
    ``langchain_loader.py``.

    :param llm_model: LangChain-compatible chat model; must implement ``.invoke()``.
    :param tools: List of LangChain ``BaseTool`` instances available to the model.
    :param max_react_iterations: Maximum Thought/Action/Observation iterations before
        the loop terminates with the last model response.
    :type max_react_iterations: int
    :return: A ``(state: dict) -> dict`` callable suitable for LangGraph.
    :rtype: Callable
    """
    # Build the tool lookup dict and preamble once (at node-construction time)
    tool_map = {t.name: t for t in tools}
    tool_preamble = _build_tool_preamble(tools)

    def prompted_react_loop(state: State) -> dict:  # pylint: disable=too-many-branches
        """Execute the prompted ReAct loop for one node invocation.

        :param state: LangGraph state dict; must contain ``"messages"``.
        :return: Dict with ``"messages"`` key holding a single ``AIMessage``
            containing the final answer.
        :rtype: dict
        """
        # ── Inject the tool preamble into the system message ──────────────────
        messages = list(state["messages"])

        # Find an existing SystemMessage and augment it, or prepend a new one
        sys_idx = next(
            (i for i, m in enumerate(messages) if isinstance(m, SystemMessage)), None
        )
        if sys_idx is not None:
            existing = messages[sys_idx].content
            messages[sys_idx] = SystemMessage(content=f"{existing}\n\n{tool_preamble}")
        else:
            messages.insert(0, SystemMessage(content=tool_preamble))

        llm_config = state.get("llm_config", {})
        consecutive_parse_failures = 0
        last_response_text = ""

        for iteration in range(max_react_iterations):
            # ── Model call ────────────────────────────────────────────────────
            response = llm_model.invoke(messages, config=llm_config)
            response_text = str(response.content) if response.content else ""
            last_response_text = response_text

            LOGGER.debug(
                "Prompted ReAct iteration %d/%d | response length=%d chars",
                iteration + 1,
                max_react_iterations,
                len(response_text),
            )

            # ── Parse ─────────────────────────────────────────────────────────
            kind, value, extra = _parse_react_response(response_text)

            if kind == "final":
                LOGGER.debug(
                    "Prompted ReAct: Final Answer after %d iteration(s).", iteration + 1
                )
                return {"messages": [AIMessage(content=value)]}

            if kind == "action":
                consecutive_parse_failures = 0
                tool_name = value
                args = extra

                # Append the model's action turn to the conversation
                messages.append(AIMessage(content=response_text))

                # ── Tool execution ────────────────────────────────────────────
                if tool_name not in tool_map:
                    available = ", ".join(sorted(tool_map.keys()))
                    observation = (
                        f"Error - unknown tool '{tool_name}'.  "
                        f"Available tools: {available}"
                    )
                    LOGGER.warning(
                        "Prompted ReAct: model requested unknown tool '%s'.", tool_name
                    )
                else:
                    try:
                        observation = str(tool_map[tool_name].invoke(args))
                        LOGGER.debug(
                            "Prompted ReAct: tool '%s' returned observation of %d chars.",
                            tool_name,
                            len(observation),
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        observation = f"Error - tool '{tool_name}' raised: {exc}"
                        LOGGER.warning(
                            "Prompted ReAct: tool '%s' raised an exception: %s",
                            tool_name,
                            exc,
                        )

                # Feed the observation back as a HumanMessage so strict-alternating-turn
                # models (e.g. Bedrock Converse) accept the message sequence.
                messages.append(HumanMessage(content=f"Observation: {observation}"))
                continue

            # kind is "parse_error" or "unknown"
            consecutive_parse_failures += 1
            LOGGER.warning(
                "Prompted ReAct parse failure (%d/%d): kind=%s value=%r extra=%r",
                consecutive_parse_failures,
                _MAX_CONSECUTIVE_PARSE_FAILURES,
                kind,
                value,
                extra,
            )

            if consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                LOGGER.warning(
                    "Prompted ReAct: %d consecutive parse failures; returning last response.",
                    consecutive_parse_failures,
                )
                return {"messages": [AIMessage(content=last_response_text)]}

            # Feed the error back so the model can self-correct
            messages.append(AIMessage(content=response_text))
            error_detail = (
                extra or "response matched neither Action nor Final Answer format"
            )
            messages.append(
                HumanMessage(
                    content=(
                        f"Observation: Error - could not parse your response. "
                        f"{error_detail}.  "
                        "Please respond using the exact Action/Final Answer format."
                    )
                )
            )

        # ── Max iterations reached ─────────────────────────────────────────────
        LOGGER.warning(
            "Prompted ReAct: reached max_react_iterations=%d without a Final Answer; "
            "returning last model response.",
            max_react_iterations,
        )
        return {"messages": [AIMessage(content=last_response_text)]}

    return prompted_react_loop


# ──────────────────────────────────────────────────────────────────────────────
# Public node builder
# ──────────────────────────────────────────────────────────────────────────────


def build_react_agent_node(
    tools: list[Tool] = None,
    state: Type[AgentState] = State,
    llm_model=None,
    middleware: list = None,
    **kwargs,
):
    """Construct a REACT agent node, selecting the appropriate execution mode.

    Four modes are available, selected by ``tool_strategy`` (or the legacy
    ``supports_tools`` kwarg):

    **Native tool-calling** (``tool_strategy="native"``, the default):
        Builds a compiled LangGraph sub-graph via ``create_agent`` + ``bind_tools``.
        Requires the model to implement ``BaseChatModel.bind_tools`` (all major API
        providers do).

    **Prompted tool-calling** (``tool_strategy="facilitated"``):
        Runs a hand-rolled ReAct loop inside the node.  Tools are described in a
        system-message preamble; the model's text output is parsed for
        ``Action:`` / ``Final Answer:`` markers.  Compatible with any model that
        can follow text instructions.  Does not call ``create_agent`` and never
        invokes ``bind_tools``.

    **MCP / agentic CLI** (``tool_strategy="mcp"``):
        Interim behaviour until the MCP mechanism is implemented: tools are dropped
        and the model runs on the plain tool-less path so it can self-orchestrate.
        Passing ``bind_tools`` kwargs to an agentic CLI raises; dropping tools is
        the safe interim behaviour.

    **No-tool model** (``tool_strategy="none"``):
        The model has no tool support (e.g. some reasoning models).  Tools are
        dropped and the model runs on the plain tool-less path.

    **Tool-less fallback** (``tools`` is ``None``):
        Calls the model directly, filtering ToolMessages from history for
        compatibility with models that reject them.

    Backward compatibility: the legacy ``supports_tools`` (``bool``) kwarg is
    still accepted.  When ``tool_strategy`` is absent, the router infers the
    strategy: ``supports_tools=True`` (or absent) maps to ``"native"``; ``False``
    maps to ``"facilitated"``.  Prefer passing ``tool_strategy`` explicitly.

    Extra ``**kwargs`` recognised:

    - ``tool_strategy`` (``str``): ``"native"`` | ``"facilitated"`` | ``"mcp"`` |
      ``"none"`` -- the authoritative routing key.
    - ``supports_tools`` (``bool``): legacy convenience; ignored when
      ``tool_strategy`` is provided.
    - ``max_react_iterations`` (``int``, default 10): iteration cap for the
      prompted (``"facilitated"``) path.

    :param tools: List of tools for the agent.  ``None`` (or a strategy of
        ``"mcp"``/``"none"``) engages the tool-less path.
    :type tools: list[Tool] or None

    :param state: State schema (must subclass ``AgentState``).  Used only by the
        native path (``create_agent``).
    :type state: Type[AgentState]

    :param llm_model: LangChain-compatible chat model.

    :param middleware: Middleware list forwarded to ``create_agent`` on the native
        path.  Ignored on the prompted and fallback paths.
    :type middleware: list or None

    :param kwargs: Additional node configuration; see recognised keys above.

    :return: A compiled ``CompiledStateGraph`` (native path) or a plain callable
        ``(state: dict) -> dict`` (prompted and fallback paths).
    :rtype: CompiledStateGraph or Callable
    """
    # Resolve tool_strategy, falling back to the legacy supports_tools bool.
    tool_strategy = kwargs.get("tool_strategy")
    if tool_strategy is None:
        tool_strategy = (
            "native" if kwargs.get("supports_tools", True) else "facilitated"
        )

    max_react_iterations = kwargs.get(
        "max_react_iterations", _DEFAULT_MAX_REACT_ITERATIONS
    )

    LOGGER.debug(
        "Using model: %s | Tools provided: %s | tool_strategy: %s | Middleware count: %s",
        getattr(llm_model, "__class__", None),
        tools is not None,
        tool_strategy,
        len(middleware) if middleware else 0,
    )

    if tools is not None and tool_strategy == "native":
        # Native path: the model implements bind_tools; delegate to create_agent.
        LOGGER.debug(
            "Native tool-calling path: creating REACT agent with tools: %s", tools
        )
        agent = create_agent(
            model=llm_model,
            state_schema=state,
            tools=tools,
            middleware=middleware or (),
        )

    elif tools is not None and tool_strategy == "facilitated":
        # Prompted path: the model cannot bind_tools; run the ReAct loop ourselves.
        LOGGER.debug(
            "Prompted tool-calling path: model does not support bind_tools; "
            "using prompted ReAct loop with %d tool(s) and max_react_iterations=%d.",
            len(tools),
            max_react_iterations,
        )
        agent = _build_prompted_react_loop(
            llm_model=llm_model,
            tools=tools,
            max_react_iterations=max_react_iterations,
        )

    elif tools is not None and tool_strategy in ("mcp", "none"):
        # MCP/none interim path: drop tools; let the model run plain.
        # "mcp" models are agentic CLIs that self-orchestrate; "none" models
        # reject tool kwargs entirely.  Both run tool-less until the MCP
        # mechanism is in place.
        LOGGER.debug(
            "Tool-less interim path for strategy '%s': dropping %d tool(s); "
            "model runs plain.",
            tool_strategy,
            len(tools),
        )
        tools = None
        # Fall through to the tool-less branch below.

    if tools is None:
        # Tool-less fallback: no tools provided (or dropped above); call the model directly.
        LOGGER.debug(
            "Tool-less fallback path: no tools provided; invoking the LLM directly "
            "and converting the state to a format the LLM can understand."
        )

        def call_model(state: State):
            # Filter out any tool messages to prevent errors in the LLM during the call
            # This is to allow for the scenario where the state was established with a model
            # that used tools, and then later a model that does not support tools was selected.
            # This allows the conversation to continue without the history of tool messages being
            # passed to the model.
            messages = [
                message
                for message in state["messages"]
                if isinstance(message, (AIMessage, HumanMessage, SystemMessage))
            ]
            # Repackage every message as a HumanMessage with only the text content
            # There are some bugs out there right now with limitations of the kinds
            # of messages that can be sent to certain models, and this is a way to work around
            # that for now.
            messages_to_send = [
                HumanMessage(content=str(message.content)) for message in messages
            ]

            # Load the llm config if one exists, otherwise pass an empty dict
            llm_config = state.get("llm_config", {})

            response = llm_model.invoke(messages_to_send, config=llm_config)
            return {"messages": [response]}

        agent = call_model

    return agent


react_agent_node = partial(Node, "react_agent", build_react_agent_node)
