"""Agent node generation — creates LLM-backed or stub callables from AgentSpec definitions.

When an ``AgentSpec`` has ``model_name`` set, the generated node makes
real LLM calls using bili-core's ``llm_loader``.  If the agent also has
``tools`` configured, the node execution path is selected based on the
model's ``tool_strategy`` (sourced from ``LLM_MODELS`` via
:func:`~bili.aether.compiler.llm_resolver.resolve_tool_strategy`),
mirroring the routing in ``bili/iris/nodes/react_agent_node.py``:

1. **Native tool-calling** (``tool_strategy="native"``, the default for API
   providers): uses ``langchain.agents.create_agent`` + ``bind_tools``.

2. **Prompted tool-calling** (``tool_strategy="facilitated"``): uses the
   shared :func:`~bili.iris.nodes.react_agent_node._build_prompted_react_loop`
   factory imported from IRIS.  Tools are described in a system-message
   preamble; the model's text output is parsed for ``Action:`` /
   ``Final Answer:`` markers.  Works with any model that can follow text
   instructions.

3. **MCP / agentic CLI** (``tool_strategy="mcp"``): the agent's tools are
   exposed as an ephemeral, per-call authenticated MCP server
   (:class:`~bili.iris.mcp.server.EphemeralMcpServer`) on a dynamic
   localhost port.  The CLI model connects to this server and calls tools
   via its own native tool-calling interface.  This path routes through the
   shared ``_agent_node`` closure so system_prompt injection, comm_context,
   human-message ordering, timing, and provenance are all inherited.  For
   CLIs that use ``message_format="last"`` (the default), the executor
   embeds the per-agent system instruction into the last HumanMessage
   before the MCP node renders it, because ``render_messages("last")``
   discards SystemMessage objects.  If no injector is registered for the
   CLI, the agent falls back to the direct-LLM path.

4. **No-tool model** (``tool_strategy="none"``): the model has no tool
   support; tools are dropped and the model runs tool-less.

5. **No tools configured**: calls ``llm.invoke()`` directly (unchanged).

The prompted-loop implementation is shared with IRIS via a direct import
from ``bili.iris.nodes.react_agent_node`` — no code is duplicated.
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from bili.aether.schema import AgentSpec, OutputFormat
from bili.iris.multimodal import content_has_non_text_parts

LOGGER = logging.getLogger(__name__)


def _ensure_human_last(messages: list, agent: "AgentSpec") -> None:
    """Ensure the last message is a HumanMessage for Bedrock turn constraints.

    Some providers (e.g. Mistral on Bedrock) require the conversation to
    end with a HumanMessage. If the last message is an AIMessage (common
    when context flows from a prior agent), append a synthetic
    HumanMessage with the agent's objective.

    Mutates *messages* in place.
    """
    from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
        AIMessage,
        HumanMessage,
    )

    if messages and isinstance(messages[-1], AIMessage):
        task_cue = (
            agent.objective or "Please complete your task based on the context above."
        )
        messages.append(HumanMessage(content=task_cue))


def _normalise_content_value(content: Any) -> str:
    """Coerce a raw LLM content value to ``str``.

    Some providers (e.g. Google Vertex / Gemini) return content as a list of
    part dicts such as ``[{"type": "text", "text": "..."}]``.  Joins text
    parts into a single string; falls back to ``str()`` for unrecognised
    types.  Returns an empty string for falsy input.
    """
    if isinstance(content, list):
        return " ".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


def _normalise_message_content(msg: Any) -> Any:
    """Return *msg* with text-only list ``content`` coerced to ``str``.

    Some LLM providers (e.g. Google Vertex / Gemini) return ``content`` as a
    list of content-part dicts.  Forwarding such messages as conversation
    history to the same or a different provider can cause serialisation errors
    (e.g. Vertex rejects a message with an unrecognised parts structure).
    Delegates to :func:`_normalise_content_value` for the join logic.

    A message carrying a recognised NON-TEXT part (an image) is returned
    unchanged instead.  Coercing it would join ``part.get("text", str(part))``
    over parts that have no ``"text"`` key, so the image is stringified into
    the prompt and effectively dropped -- silently, since the join succeeds.
    The safety this coercion exists for is unaffected: a text-only list, which
    is what a provider returning parts produces, still collapses to a string
    exactly as before.
    """
    if not hasattr(msg, "content") or not isinstance(msg.content, list):
        return msg
    if content_has_non_text_parts(msg.content):
        return msg
    return msg.model_copy(update={"content": _normalise_content_value(msg.content)})


def generate_agent_node(agent: AgentSpec) -> Callable[[dict], dict]:
    """Create a node callable for the given agent.

    If ``agent.model_name`` is set, returns a node that makes real LLM
    calls via ``bili.loaders.llm_loader``.  If the agent also has
    ``tools``, uses ``create_agent()`` for tool-enabled execution.
    Otherwise returns a stub node that emits a placeholder ``AIMessage``
    so the graph can execute end-to-end without API keys.

    Args:
        agent: The ``AgentSpec`` for this agent.

    Returns:
        A callable ``(state: dict) -> dict`` suitable for
        ``StateGraph.add_node``.
    """
    if agent.model_name:
        return _generate_llm_agent_node(agent)
    return _generate_stub_agent_node(agent)


# =========================================================================
# Real LLM node
# =========================================================================


def _generate_llm_agent_node(agent: AgentSpec) -> Callable[[dict], dict]:
    """Create a node callable that invokes a real LLM.

    The LLM instance is created eagerly (at compile time) via
    :func:`~bili.aether.compiler.llm_resolver.create_llm` so that
    provider-resolution errors surface immediately rather than at
    graph-execution time.

    If the agent has ``tools`` configured, resolves them via
    :func:`~bili.aether.compiler.llm_resolver.resolve_tools` and uses
    ``langchain.agents.create_agent`` — the same pattern as
    ``bili/iris/nodes/react_agent_node.py``.  Middleware (if configured) is
    resolved and passed to ``create_agent`` for tool-enabled agents.
    """
    # pylint: disable=import-outside-toplevel
    from bili.aether.compiler.llm_resolver import (
        create_llm,
        resolve_tool_strategy,
        resolve_tools,
    )

    llm = create_llm(agent)
    tools = resolve_tools(agent)
    middleware = _resolve_middleware(agent)
    tool_strategy = (
        resolve_tool_strategy(agent.model_name) if agent.model_name else "native"
    )

    if tools:
        return _generate_tool_agent_node(
            agent, llm, tools, middleware, tool_strategy=tool_strategy
        )

    if middleware:
        LOGGER.warning(
            "Agent '%s' has middleware configured but no tools; "
            "middleware requires tool-enabled agents (via create_agent). "
            "Middleware will be ignored.",
            agent.agent_id,
        )
    return _generate_direct_llm_node(agent, llm)


def _generate_tool_agent_node(
    agent: AgentSpec,
    llm: object,
    tools: List,
    middleware: Optional[List] = None,
    tool_strategy: str = "native",
) -> Callable[[dict], dict]:
    """Create a node for tool-enabled agents, selecting the execution path by strategy.

    ``tool_strategy`` is the authoritative routing key:

    - ``"native"`` (default for API-backed models): delegates to
      ``langchain.agents.create_agent`` + ``bind_tools``.  Middleware is
      forwarded to ``create_agent()``.
    - ``"facilitated"``: uses the shared prompted ReAct loop imported from
      ``bili/iris/nodes/react_agent_node.py``.  Middleware is not applicable
      on this path and is silently ignored.
    - ``"mcp"``: the agent's tools are exposed as an ephemeral, per-call
      authenticated MCP server on a dynamic localhost port.  The CLI model
      (Claude Code, Codex, Gemini CLI) connects to the server and calls
      tools via its own native tool-calling interface.  If no injector is
      registered for the CLI, falls back to the direct-LLM path with a
      warning.
    - ``"none"``: the model has no tool support (e.g. some reasoning models
      reject tool kwargs entirely); tools are dropped and the agent runs on
      the direct-LLM path.

    ``max_react_iterations`` for the ``"facilitated"`` path can be tuned via
    ``agent.metadata["max_react_iterations"]``; the default is 10.
    """
    # ── Build the executor at node-construction time ──────────────────────────
    if tool_strategy == "native":
        from langchain.agents import (  # pylint: disable=import-error,import-outside-toplevel
            create_agent,
        )
        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
        )
        from langgraph.errors import (  # pylint: disable=import-error,import-outside-toplevel
            GraphRecursionError,
        )

        from bili.iris.nodes.react_agent_node import (  # pylint: disable=import-outside-toplevel
            _DEFAULT_MAX_REACT_ITERATIONS,
            _with_prompt_caching,
        )

        # Append provider prompt caching (Anthropic direct-API, Claude/Nova via
        # Bedrock) the same way the IRIS single-agent native path does, so a
        # multi-agent run re-reads its stable prefix from the provider cache
        # instead of re-billing it on every model call.  A no-op for every
        # non-target provider (the helper returns the middleware unchanged), so
        # behaviour is byte-for-byte identical off the Anthropic/Bedrock path.
        react_agent = create_agent(
            model=llm,
            tools=tools,
            middleware=_with_prompt_caching(middleware or (), llm),
        )
        executor_mode = "tool-agent (native)"

        # Bound the inner react loop the same way the facilitated path is bounded.
        # create_agent produces a compiled LangGraph whose invoke() defaults to
        # recursion_limit=5000; that is a per-agent budget of thousands of billed
        # model calls, and the outer MAS graph's recursion_limit does NOT bound
        # this inner loop.  A native agent that never emits a final answer would
        # otherwise spin unbounded.  LangGraph counts SUPERSTEPS: N tool cycles
        # (model + tools = 2 steps each) plus the terminal model turn is 2N + 1,
        # so an N-iteration cap maps to recursion_limit 2N + 1.  The cap is
        # tunable per agent via metadata, mirroring the facilitated path.
        native_max_iterations = int(
            agent.metadata.get("max_react_iterations", _DEFAULT_MAX_REACT_ITERATIONS)
        )
        native_recursion_limit = 2 * native_max_iterations + 1

        def _invoke_executor(messages: list) -> list:
            try:
                result = react_agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": native_recursion_limit},
                )
            except GraphRecursionError:
                # Fail clean and loud, not silently: the agent hit its cap without
                # producing a final answer.  Return a bounded result so one stuck
                # agent does not abort the whole MAS run, matching the facilitated
                # path's warn-and-return behaviour.  Raise
                # agent.metadata["max_react_iterations"] if a longer tool loop is
                # genuinely expected.
                LOGGER.warning(
                    "Agent '%s': native react loop hit the %d-iteration cap "
                    "(recursion_limit=%d) without producing a final answer; "
                    "returning a bounded result.",
                    agent.agent_id,
                    native_max_iterations,
                    native_recursion_limit,
                )
                return [
                    AIMessage(
                        content=(
                            f"[Agent stopped: reached the {native_max_iterations}-"
                            "iteration tool-use limit without a final answer.]"
                        )
                    )
                ]
            return result.get("messages", [])

    elif tool_strategy == "facilitated":
        # Prompted path: model cannot bind_tools; run the hand-rolled ReAct loop.
        # Imported from IRIS so the implementation is shared, not duplicated.
        from bili.iris.nodes.react_agent_node import (  # pylint: disable=import-outside-toplevel
            _DEFAULT_MAX_REACT_ITERATIONS,
            _build_prompted_react_loop,
        )

        if middleware:
            LOGGER.warning(
                "Agent '%s' has middleware configured but tool_strategy='facilitated'; "
                "middleware is only applicable on the native create_agent path "
                "and will be ignored for the prompted ReAct path.",
                agent.agent_id,
            )

        max_react_iterations = int(
            agent.metadata.get("max_react_iterations", _DEFAULT_MAX_REACT_ITERATIONS)
        )
        prompted_loop = _build_prompted_react_loop(
            llm_model=llm,
            tools=tools,
            max_react_iterations=max_react_iterations,
        )
        executor_mode = "tool-agent (prompted)"

        def _invoke_executor(messages: list) -> list:
            result = prompted_loop({"messages": messages})
            return result.get("messages", [])

    elif tool_strategy == "mcp":
        # MCP path: expose tools as an ephemeral authenticated MCP server.
        # The CLI self-orchestrates; bili-core takes its final stdout as the
        # agent response.  This branch routes through the shared _agent_node
        # so every cross-cutting concern (system_prompt injection, comm_context,
        # human-message ordering, timing, provenance) is inherited in one place.
        from bili.iris.mcp.server import (  # pylint: disable=import-outside-toplevel
            build_mcp_node,
            resolve_mcp_injector,
        )

        injector = resolve_mcp_injector(llm)
        if injector is None:
            LOGGER.warning(
                "Agent '%s': tool_strategy='mcp' but no injector found for CLI '%s'; "
                "falling back to direct-LLM node.  Register an injector via "
                "bili.iris.mcp.cli_injectors.register_cli_mcp_injector() to enable "
                "MCP tool-calling for this CLI.",
                agent.agent_id,
                getattr(llm, "command", ["?"])[0],
            )
            return _generate_direct_llm_node(agent, llm)

        LOGGER.debug(
            "Agent '%s': tool_strategy='mcp' -- serving %d tool(s) via "
            "ephemeral MCP server for CLI '%s'.",
            agent.agent_id,
            len(tools),
            getattr(llm, "command", ["?"])[0],
        )
        mcp_raw_node = build_mcp_node(llm_model=llm, tools=tools, injector=injector)
        # Capture the CLI's message_format at construction time so the
        # executor can handle it without referencing the outer llm object.
        cli_message_format: str = str(getattr(llm, "message_format", "last"))
        executor_mode = "tool-agent (mcp)"

        def _invoke_executor(messages: list) -> list:
            # _agent_node injects the per-agent system_prompt as a
            # SystemMessage.  For "last"-format CLIs, render_messages()
            # ignores SystemMessage and returns only the last HumanMessage
            # content.  Embed the system instruction into the last
            # HumanMessage before passing to the MCP node so the CLI
            # receives its per-role instructions.
            if cli_message_format == "last":
                from langchain_core.messages import (
                    HumanMessage as _HM,  # pylint: disable=import-error,import-outside-toplevel
                )
                from langchain_core.messages import SystemMessage as _SM

                sys_parts = [m.content for m in messages if isinstance(m, _SM)]
                if sys_parts:
                    sys_instruction = str(sys_parts[0])
                    stripped = [m for m in messages if not isinstance(m, _SM)]
                    for i in range(len(stripped) - 1, -1, -1):
                        if isinstance(stripped[i], _HM):
                            orig = str(stripped[i].content)
                            stripped[i] = _HM(content=f"{sys_instruction}\n\n{orig}")
                            break
                    messages = stripped

            result = mcp_raw_node({"messages": messages})
            return result.get("messages", [])

    else:
        # "none": model has no tool support (e.g. reasoning models that reject
        # tool kwargs entirely).  Drop tools; delegate to the direct-LLM node.
        LOGGER.debug(
            "Agent '%s': tool_strategy='none' -- dropping tools; routing to direct-LLM node.",
            agent.agent_id,
        )
        return _generate_direct_llm_node(agent, llm)

    # ── Shared node callable ──────────────────────────────────────────────────

    def _agent_node(state: dict) -> dict:  # pylint: disable=too-many-locals
        start_time = time.time()

        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        # Inject system prompt into messages if not already present
        system_prompt = agent.system_prompt or agent.objective

        # Append pending inter-agent messages to system prompt
        comm_context = _get_communication_context(state, agent.agent_id)
        if comm_context:
            system_prompt += "\n\n--- Messages from other agents ---\n" + comm_context

        messages = list(state.get("messages", []))

        # Normalise any list-content messages from previous agents (e.g. Gemini
        # returns content as a list of parts).  Providers like Google Vertex reject
        # messages where content is a list they don't recognise when those messages
        # are forwarded as conversation history to a subsequent agent.
        messages = [_normalise_message_content(m) for m in messages]

        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages.insert(0, SystemMessage(content=system_prompt))

        if not any(isinstance(m, HumanMessage) for m in messages):
            messages.append(HumanMessage(content="Begin your task."))

        # Bedrock Converse API requires the first non-system message to be a
        # HumanMessage.  In a sequential chain the first message in accumulated
        # context is often an AIMessage from a prior agent — insert a synthetic
        # task cue before it so the turn order constraint is satisfied.
        first_non_sys = next(
            (i for i, m in enumerate(messages) if not isinstance(m, SystemMessage)),
            None,
        )
        if first_non_sys is not None and isinstance(messages[first_non_sys], AIMessage):
            messages.insert(
                first_non_sys,
                HumanMessage(
                    content="Please review the following context and complete your task."
                ),
            )

        _ensure_human_last(messages, agent)

        # Invoke via the pre-selected executor (native react_agent or prompted loop)
        response_messages = _invoke_executor(messages)

        execution_ms = (time.time() - start_time) * 1000
        LOGGER.info(
            "Agent node '%s' executed in %.2f ms (%s)",
            agent.agent_id,
            execution_ms,
            executor_mode,
        )

        # Extract the final response content
        content = ""
        if response_messages:
            content = _normalise_content_value(response_messages[-1].content)

        output = _build_output(agent, content)
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs[agent.agent_id] = output

        state_update: Dict[str, Any] = {
            "messages": [AIMessage(content=content, name=agent.agent_id)],
            "current_agent": agent.agent_id,
            "agent_outputs": agent_outputs,
        }

        # For supervisor agents, extract routing decision
        if agent.is_supervisor:
            state_update["next_agent"] = _extract_next_agent(content, agent)

        state_update.update(_build_communication_update(state, agent.agent_id, content))
        return state_update

    _agent_node.agent_spec = agent  # type: ignore[attr-defined]
    _agent_node.__name__ = f"agent_{agent.agent_id}"
    _agent_node.__qualname__ = f"agent_{agent.agent_id}"

    return _agent_node


def _wrap_mcp_node_with_provenance(
    inner_node: Callable[[dict], dict], agent: AgentSpec
) -> Callable[[dict], dict]:
    """Wrap an IRIS MCP node so it synthesises AETHER per-agent provenance.

    .. note::
        The main ``tool_strategy="mcp"`` code path in
        :func:`_generate_tool_agent_node` now routes through the shared
        ``_agent_node`` closure, which handles provenance (and system_prompt
        injection, comm_context, human-message ordering) in one place.  This
        function is retained as a standalone utility for direct use or testing.

    The MCP node returned by :func:`~bili.iris.mcp.server.build_mcp_node` is
    intentionally generic: it returns only
    ``{"messages": [AIMessage(content=...)]}`` with no ``name``, no
    ``current_agent``, no ``agent_outputs``, and no ``communication_log``.
    This wrapper synthesises the full AETHER provenance payload from the agent
    closure, matching what the shared ``_agent_node`` and
    :func:`~bili.aether.compiler.graph_builder.GraphBuilder._wrap_pipeline_as_agent_node`
    emit: an ``AIMessage`` tagged ``name=agent_id``, ``current_agent``,
    ``agent_outputs[agent_id]``, and a ``communication_log`` broadcast entry.

    .. warning::
        This wrapper does **not** inject the agent's ``system_prompt`` or
        comm_context into the prompt the CLI receives.  The shared ``_agent_node``
        path is preferred because it applies all cross-cutting per-turn setup.
    """
    from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
        AIMessage,
    )

    agent_id = agent.agent_id

    def _agent_node(state: dict) -> dict:
        result = inner_node(state)
        inner_messages = result.get("messages", []) if isinstance(result, dict) else []

        content = ""
        for msg in reversed(inner_messages):
            if getattr(msg, "content", None):
                content = _normalise_content_value(msg.content)
                break

        output = _build_output(agent, content)
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs[agent_id] = output

        state_update: Dict[str, Any] = {
            "messages": [AIMessage(content=content, name=agent_id)],
            "current_agent": agent_id,
            "agent_outputs": agent_outputs,
        }

        if agent.is_supervisor:
            state_update["next_agent"] = _extract_next_agent(content, agent)

        state_update.update(_build_communication_update(state, agent_id, content))
        return state_update

    _agent_node.agent_spec = agent  # type: ignore[attr-defined]
    _agent_node.__name__ = f"agent_{agent_id}"
    _agent_node.__qualname__ = f"agent_{agent_id}"

    return _agent_node


def _generate_direct_llm_node(agent: AgentSpec, llm: object) -> Callable[[dict], dict]:
    """Create a node that calls ``llm.invoke()`` directly (no tools).

    Mirrors the fallback path in ``bili/iris/nodes/react_agent_node.py``.
    """

    def _agent_node(state: dict) -> dict:  # pylint: disable=too-many-locals
        start_time = time.time()

        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        # Build message list
        system_prompt = agent.system_prompt or agent.objective

        # Append pending inter-agent messages to system prompt
        comm_context = _get_communication_context(state, agent.agent_id)
        if comm_context:
            system_prompt += "\n\n--- Messages from other agents ---\n" + comm_context

        # Filter state messages to compatible types and normalise list content
        state_messages = state.get("messages", [])
        compatible = [
            _normalise_message_content(m)
            for m in state_messages
            if isinstance(m, (AIMessage, HumanMessage, SystemMessage))
        ]

        # Check if there's already a SystemMessage (avoid duplicates)
        has_system = any(isinstance(m, SystemMessage) for m in compatible)
        messages = []
        if not has_system:
            messages.append(SystemMessage(content=system_prompt))

        if compatible:
            messages.extend(compatible)
        else:
            messages.append(HumanMessage(content="Begin your task."))

        # Bedrock Converse API requires the first non-system message to be a
        # HumanMessage.  In a sequential chain the first message in accumulated
        # context is often an AIMessage from a prior agent — insert a synthetic
        # task cue before it so the turn order constraint is satisfied.
        first_non_sys = next(
            (i for i, m in enumerate(messages) if not isinstance(m, SystemMessage)),
            None,
        )
        if first_non_sys is not None and isinstance(messages[first_non_sys], AIMessage):
            messages.insert(
                first_non_sys,
                HumanMessage(
                    content="Please review the following context and complete your task."
                ),
            )

        _ensure_human_last(messages, agent)

        # Invoke the LLM directly
        response = llm.invoke(messages)
        content = _normalise_content_value(response.content)

        execution_ms = (time.time() - start_time) * 1000
        LOGGER.info(
            "Agent node '%s' executed in %.2f ms (LLM)",
            agent.agent_id,
            execution_ms,
        )

        output = _build_output(agent, content)
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs[agent.agent_id] = output

        state_update: Dict[str, Any] = {
            "messages": [AIMessage(content=content, name=agent.agent_id)],
            "current_agent": agent.agent_id,
            "agent_outputs": agent_outputs,
        }

        # For supervisor agents, extract routing decision
        if agent.is_supervisor:
            state_update["next_agent"] = _extract_next_agent(content, agent)

        state_update.update(_build_communication_update(state, agent.agent_id, content))
        return state_update

    _agent_node.agent_spec = agent  # type: ignore[attr-defined]
    _agent_node.__name__ = f"agent_{agent.agent_id}"
    _agent_node.__qualname__ = f"agent_{agent.agent_id}"

    return _agent_node


# =========================================================================
# Stub node (no LLM — used when model_name is not set)
# =========================================================================


def _generate_stub_agent_node(agent: AgentSpec) -> Callable[[dict], dict]:
    """Create a stub node callable (no LLM calls).

    The stub records itself in the state and emits an ``AIMessage`` so
    the graph can execute end-to-end without real LLM calls.  The
    ``AgentSpec`` is captured in the closure and attached as an attribute
    for introspection.
    """

    def _agent_node(state: dict) -> dict:
        start_time = time.time()

        stub_output = {
            "agent_id": agent.agent_id,
            "role": agent.role,
            "status": "stub",
            "message": f"[STUB] Agent '{agent.agent_id}' ({agent.role}) executed.",
        }

        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs[agent.agent_id] = stub_output

        execution_ms = (time.time() - start_time) * 1000
        LOGGER.info(
            "Agent node '%s' executed in %.2f ms (stub)",
            agent.agent_id,
            execution_ms,
        )

        # Consume any pending inter-agent messages (for state bookkeeping)
        _get_communication_context(state, agent.agent_id)

        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
        )

        state_update: Dict[str, Any] = {
            "messages": [
                AIMessage(
                    content=stub_output["message"],
                    name=agent.agent_id,
                )
            ],
            "current_agent": agent.agent_id,
            "agent_outputs": agent_outputs,
        }

        # For supervisor agents, extract routing decision
        if agent.is_supervisor:
            state_update["next_agent"] = _extract_next_agent(
                stub_output["message"], agent
            )

        state_update.update(
            _build_communication_update(state, agent.agent_id, stub_output["message"])
        )
        return state_update

    # Attach metadata for introspection
    _agent_node.agent_spec = agent  # type: ignore[attr-defined]
    _agent_node.__name__ = f"agent_{agent.agent_id}"
    _agent_node.__qualname__ = f"agent_{agent.agent_id}"

    return _agent_node


# =========================================================================
# Performance wrapper
# =========================================================================


def wrap_agent_node(node_func: Callable, agent_id: str) -> Callable:
    """Return unwrapped agent node (no-op wrapper).

    Individual agent nodes already log their own execution time with
    detailed context (tool-agent vs LLM vs stub), so wrapping would
    create duplicate timing logs. This function is kept for backward
    compatibility but returns the node unmodified.
    """
    # pylint: disable=unused-argument
    # Return unwrapped - agent nodes already have detailed timing
    return node_func


# =========================================================================
# Middleware resolution
# =========================================================================


def _resolve_middleware(agent: AgentSpec) -> list:
    """Resolve middleware names to instances via bili-core's middleware loader.

    Args:
        agent: The ``AgentSpec`` with ``middleware`` and ``middleware_params``.

    Returns:
        A list of initialised middleware instances, or an empty list
        if no middleware is configured or the loader is unavailable.
    """
    if not agent.middleware:
        return []

    try:
        from bili.iris.loaders.middleware_loader import (  # pylint: disable=import-outside-toplevel
            initialize_middleware,
        )
    except ImportError:
        LOGGER.warning(
            "bili.iris.loaders.middleware_loader not available; "
            "skipping middleware for agent '%s'",
            agent.agent_id,
        )
        return []

    try:
        instances = initialize_middleware(
            active_middleware=agent.middleware,
            middleware_params=agent.middleware_params,
        )
        if instances:
            LOGGER.info(
                "Resolved %d middleware for agent '%s': %s",
                len(instances),
                agent.agent_id,
                agent.middleware,
            )
        return instances
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.warning(
            "Failed to resolve middleware %s for agent '%s'; "
            "agent will run without middleware",
            agent.middleware,
            agent.agent_id,
            exc_info=True,
        )
        return []


# =========================================================================
# Shared helpers
# =========================================================================


def _build_output(agent: AgentSpec, content: str) -> dict:
    """Build the agent output dict, parsing JSON/structured output if configured.

    For ``output_format="json"`` the content is best-effort parsed (legacy
    behaviour, no schema).  For ``output_format="structured"`` the content is
    parsed *and validated* against the agent's ``output_schema``; on success
    ``output["parsed"]`` is set (which is also what consensus vote extraction
    reads), on failure ``output["raw"]`` and ``output["schema_error"]`` are
    set.  Validation runs regardless of whether the model was decode-time
    constrained, so post-hoc validation covers providers without constrained
    decoding.
    """
    output = {
        "agent_id": agent.agent_id,
        "role": agent.role,
        "status": "completed",
        "message": content,
    }

    if agent.output_format == OutputFormat.JSON:
        try:
            output["parsed"] = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            output["raw"] = content
    elif agent.output_format == OutputFormat.STRUCTURED:
        from bili.iris.providers.structured_output import (  # pylint: disable=import-outside-toplevel
            StructuredOutputError,
            parse_structured_content,
        )

        try:
            output["parsed"] = parse_structured_content(
                content, schema=agent.output_schema
            )
        except StructuredOutputError as exc:
            output["raw"] = content
            output["schema_error"] = str(exc)
            LOGGER.warning(
                "Agent '%s': structured output failed schema validation: %s",
                agent.agent_id,
                exc,
            )
    else:
        output["raw"] = content

    return output


def _extract_next_agent(
    content: str, agent: AgentSpec
) -> str:  # pylint: disable=unused-argument
    """Extract next_agent routing decision from supervisor output.

    Looks for routing decisions in these formats:
    - JSON object with a "next_agent" field (plain or inside a markdown code fence)
    - JSON-quoted key pattern: ``"next_agent": "agent_id"``
    - Lines like "ROUTE_TO: agent_id" or "NEXT_AGENT: agent_id"
    - Returns "END" if no routing decision found

    Args:
        content: Agent output message
        agent: AgentSpec for the agent (unused, kept for API compatibility)

    Returns:
        Agent ID to route to, or "END" to finish workflow
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```) then try JSON
    stripped = re.sub(r"```(?:json)?\s*", "", content).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "next_agent" in parsed:
            return parsed["next_agent"]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try text patterns — JSON-quoted key first, then plain-text directives
    patterns = [
        r'"next_agent"\s*:\s*"([^"]+)"',
        r"ROUTE_TO:\s*(\w+)",
        r"NEXT_AGENT:\s*(\w+)",
        r"next_agent:\s*(\w+)",
        r"route\s+to:\s*(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

    # Default to END if no routing decision found
    return "END"


def _get_communication_context(state: dict, agent_id: str) -> str:
    """Build a text block from pending messages for LLM context injection.

    Returns an empty string when no communication fields are present.
    """
    # pylint: disable=import-outside-toplevel
    from bili.aether.runtime.communication_state import (
        format_messages_for_context,
        get_pending_messages,
    )

    pending = get_pending_messages(state, agent_id)
    if not pending:
        return ""
    return format_messages_for_context(pending)


def _build_communication_update(
    state: dict, agent_id: str, content: str
) -> Dict[str, Any]:
    """Record agent output in the communication log for provenance.

    ``communication_log`` is always present in the state schema so this
    function always records the agent's output as a broadcast event on the
    ``__agent_output__`` channel.  This produces one entry per agent per
    superstep, giving a durable per-agent provenance trail regardless of
    whether the MAS declares explicit inter-agent channels.

    The returned dict is merged into the LangGraph state update by the
    caller.  It contains:

    - ``communication_log``: a single-element list with the new entry
      (the ``operator.add`` reducer appends it to the accumulated log).
    - ``channel_messages`` / ``pending_messages``: routing auxiliaries;
      present in the update but ignored by LangGraph when those channels
      are not in the state schema (MAS without explicit channels).

    Args:
        state: Current LangGraph state dict.
        agent_id: ID of the agent that just completed.
        content: Agent output content to record.

    Returns:
        State-update dict to merge into the node's return value.
    """
    # pylint: disable=import-outside-toplevel
    from bili.aether.runtime.communication_state import send_message_in_state
    from bili.aether.runtime.messages import MessageType

    return send_message_in_state(
        state=state,
        channel_id="__agent_output__",
        sender=agent_id,
        content=content,
        receiver="__all__",
        message_type=MessageType.BROADCAST,
    )
