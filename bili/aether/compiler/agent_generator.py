"""Agent node generation — creates LLM-backed or stub callables from AgentSpec definitions.

When an ``AgentSpec`` has ``model_name`` set, the generated node makes
real LLM calls using bili-core's ``llm_loader``.  If the agent also has
``tools`` configured, the node is built with ``langchain.agents.create_agent``
— the same pattern used by ``bili/nodes/react_agent_node.py``.
"""

import json
import logging
import time
from typing import Any, Callable, Dict

from bili.aether.schema import AgentSpec, OutputFormat

LOGGER = logging.getLogger(__name__)


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
    ``bili/nodes/react_agent_node.py``.
    """
    # pylint: disable=import-outside-toplevel
    from bili.aether.compiler.llm_resolver import create_llm, resolve_tools

    llm = create_llm(agent)
    tools = resolve_tools(agent)

    if tools:
        return _generate_tool_agent_node(agent, llm, tools)
    return _generate_direct_llm_node(agent, llm)


def _generate_tool_agent_node(
    agent: AgentSpec, llm: object, tools: list
) -> Callable[[dict], dict]:
    """Create a node using ``create_agent()`` for tool-enabled agents.

    Mirrors the tool-enabled path in ``bili/nodes/react_agent_node.py``.
    """
    from langchain.agents import (  # pylint: disable=import-error,import-outside-toplevel
        create_agent,
    )

    react_agent = create_agent(model=llm, tools=tools)

    def _agent_node(state: dict) -> dict:  # pylint: disable=too-many-locals
        start_time = time.time()

        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
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

        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages.insert(0, SystemMessage(content=system_prompt))

        if not any(isinstance(m, HumanMessage) for m in messages):
            messages.append(HumanMessage(content="Begin your task."))

        # Invoke the react agent — it handles tool calls internally
        result = react_agent.invoke({"messages": messages})

        execution_ms = (time.time() - start_time) * 1000
        LOGGER.info(
            "Agent node '%s' executed in %.2f ms (tool-agent)",
            agent.agent_id,
            execution_ms,
        )

        # Extract the final response content
        response_messages = result.get("messages", [])
        content = ""
        if response_messages:
            content = response_messages[-1].content or ""

        output = _build_output(agent, content)
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs[agent.agent_id] = output

        from langchain_core.messages import (  # pylint: disable=import-error,import-outside-toplevel
            AIMessage,
        )

        state_update: Dict[str, Any] = {
            "messages": [AIMessage(content=content, name=agent.agent_id)],
            "current_agent": agent.agent_id,
            "agent_outputs": agent_outputs,
        }
        state_update.update(_build_communication_update(state, agent.agent_id, content))
        return state_update

    _agent_node.agent_spec = agent  # type: ignore[attr-defined]
    _agent_node.__name__ = f"agent_{agent.agent_id}"
    _agent_node.__qualname__ = f"agent_{agent.agent_id}"

    return _agent_node


def _generate_direct_llm_node(agent: AgentSpec, llm: object) -> Callable[[dict], dict]:
    """Create a node that calls ``llm.invoke()`` directly (no tools).

    Mirrors the fallback path in ``bili/nodes/react_agent_node.py``.
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

        messages = [SystemMessage(content=system_prompt)]

        # Filter state messages to compatible types, matching
        # react_agent_node.py's fallback path
        state_messages = state.get("messages", [])
        if state_messages:
            compatible = [
                m
                for m in state_messages
                if isinstance(m, (AIMessage, HumanMessage, SystemMessage))
            ]
            messages.extend(compatible)
        else:
            messages.append(HumanMessage(content="Begin your task."))

        # Invoke the LLM directly
        response = llm.invoke(messages)
        content = response.content

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
    """Wrap an agent node with performance logging.

    Mirrors the ``wrap_node`` pattern from
    ``bili/loaders/langchain_loader.py``.
    """

    def wrapper(state: dict) -> dict:
        start_time = time.time()
        result = node_func(state)
        execution_ms = (time.time() - start_time) * 1000
        LOGGER.info("Agent '%s' executed in %.2f ms", agent_id, execution_ms)
        return result

    wrapper.__name__ = node_func.__name__
    wrapper.__qualname__ = node_func.__qualname__
    if hasattr(node_func, "agent_spec"):
        wrapper.agent_spec = node_func.agent_spec  # type: ignore[attr-defined]

    return wrapper


# =========================================================================
# Shared helpers
# =========================================================================


def _build_output(agent: AgentSpec, content: str) -> dict:
    """Build the agent output dict, parsing JSON if configured."""
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
    else:
        output["raw"] = content

    return output


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
    """Record agent output in communication_log if communication is active.

    Returns a dict of state fields to merge (empty if communication is
    not configured).  Only **new** entries are returned — the state
    schema uses ``operator.add`` / ``_merge_dicts`` reducers to combine
    them with existing values, which supports concurrent agent execution.
    """
    if "communication_log" not in state:
        return {}

    new_entry = {
        "sender": agent_id,
        "receiver": "__all__",
        "channel": "__agent_output__",
        "content": content,
    }

    return {
        "communication_log": [new_entry],
        "pending_messages": {agent_id: []},
    }
