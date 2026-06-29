"""Regression tests: per-agent system_prompt injection on the MCP/CLI execution path.

When a MAS agent has ``tool_strategy="mcp"``, its execution is delegated to a
CLI model via an ephemeral MCP server.  The CLI receives a text prompt rendered
from the state's message list.  For CLI models with ``message_format="last"``
(the default for all built-in presets), :func:`render_messages` returns only
the content of the last ``HumanMessage`` — the ``SystemMessage`` that the
shared ``_agent_node`` uses to carry the per-agent system instruction is
silently discarded.

Before this fix, the MCP branch in ``_generate_tool_agent_node`` returned
``build_mcp_node(...)`` wrapped in a provenance wrapper but never routed
through the shared ``_agent_node``.  As a result:

1. The per-agent ``system_prompt`` was never injected at all.
2. Even if it had been, the ``"last"`` format renderer would have dropped it.

Fix (``agent_generator.py``): route the MCP path through the shared
``_agent_node``.  The ``_invoke_executor`` for MCP is format-aware: for
``message_format="last"`` CLIs it extracts the ``SystemMessage`` that
``_agent_node`` injected and embeds its content at the front of the last
``HumanMessage`` before the MCP node renders the prompt.  For
``"roles"``/``"chatml"`` formats the ``SystemMessage`` is forwarded as-is.

These tests confirm the injection behaviour and are designed to FAIL against
the pre-fix codebase (where the MCP path returned the raw node and never
entered ``_agent_node``) and PASS after.
"""

# pylint: disable=missing-function-docstring

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bili.aether.compiler import compile_mas
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType

# ---------------------------------------------------------------------------
# Helpers: capture the prompt the MCP node actually receives
# ---------------------------------------------------------------------------

# Sentinel system_prompts that are unique per agent and easy to assert.
_AGENT_A_SYSTEM = "AGENT_A_UNIQUE_INSTRUCTION: perform task A only"
_AGENT_B_SYSTEM = "AGENT_B_UNIQUE_INSTRUCTION: perform task B only"


def _make_two_agent_config(mas_id: str = "sp_test") -> MASConfig:
    """Sequential MAS: two MCP agents with distinct system_prompts."""
    return MASConfig(
        mas_id=mas_id,
        name="System Prompt Injection Test",
        workflow_type=WorkflowType.SEQUENTIAL,
        agents=[
            AgentSpec(
                agent_id="agent_a",
                role="worker_a",
                objective="Perform task A only",
                system_prompt=_AGENT_A_SYSTEM,
                model_name="cli_claude_code",
                tools=["tool_a"],
            ),
            AgentSpec(
                agent_id="agent_b",
                role="worker_b",
                objective="Perform task B only",
                system_prompt=_AGENT_B_SYSTEM,
                model_name="cli_claude_code",
                tools=["tool_b"],
            ),
        ],
        checkpoint_enabled=False,
    )


def _run_capturing_prompts(
    mas_id: str = "sp_test",
    message_format: str = "last",
):
    """Run a two-MCP-agent MAS and capture the prompt each CLI invocation receives.

    Patches five call sites so no real subprocess is spawned.  Crucially,
    ``build_mcp_node`` is replaced with a factory that returns a capturing
    inner node — each invocation records the ``state["messages"]`` it received
    AND the string that ``render_messages`` would produce (the actual CLI prompt).

    Returns ``(captured_prompts, final_result)`` where ``captured_prompts`` is
    a list of ``{"agent_id": ..., "messages": ..., "rendered_prompt": ...}``
    dicts in invocation order.
    """
    from bili.iris.providers.cli_provider import (  # pylint: disable=import-outside-toplevel
        render_messages,
    )

    captured: list = []

    def _make_capturing_mcp_node(agent_id_label: str):
        """Return a fake MCP inner node that captures what it receives."""

        def _node(state: dict) -> dict:
            messages = state.get("messages", [])
            try:
                rendered = render_messages(messages, message_format)
            except ValueError:
                rendered = ""
            captured.append(
                {
                    "agent_id": agent_id_label,
                    "messages": list(messages),
                    "rendered_prompt": rendered,
                }
            )
            # Return content that identifies the responding agent
            return {"messages": [AIMessage(content=f"response from {agent_id_label}")]}

        return _node

    # build_mcp_node is a factory; we need DIFFERENT nodes for each call so
    # we can correlate invocations with agent_ids.  Use a side_effect list.
    mock_build = MagicMock(
        side_effect=[
            _make_capturing_mcp_node("agent_a"),
            _make_capturing_mcp_node("agent_b"),
        ]
    )

    fake_llm = MagicMock()
    fake_llm.command = ["claude"]
    fake_llm.message_format = message_format
    fake_tool = MagicMock()
    fake_tool.name = "tool_x"
    fake_injector = MagicMock()

    with (
        patch(
            "bili.aether.compiler.llm_resolver.create_llm",
            return_value=fake_llm,
        ),
        patch(
            "bili.aether.compiler.llm_resolver.resolve_tools",
            return_value=[fake_tool],
        ),
        patch(
            "bili.aether.compiler.llm_resolver.resolve_tool_strategy",
            return_value="mcp",
        ),
        patch(
            "bili.iris.mcp.server.resolve_mcp_injector",
            return_value=fake_injector,
        ),
        patch(
            "bili.iris.mcp.server.build_mcp_node",
            mock_build,
        ),
    ):
        config = _make_two_agent_config(mas_id)
        compiled = compile_mas(config)
        graph = compiled.compile_graph(checkpointer=None)
        result = graph.invoke(
            {"messages": [HumanMessage(content="begin workflow")]},
            config={"configurable": {"thread_id": f"{mas_id}-001"}},
        )

    return captured, result


# ---------------------------------------------------------------------------
# "last" format: system_prompt embedded in the last HumanMessage
# ---------------------------------------------------------------------------


class TestSystemPromptInjectionLastFormat:
    """Per-agent system_prompts reach the CLI prompt for message_format="last".

    With ``"last"`` format, ``render_messages`` returns only the content of the
    final ``HumanMessage``.  The fix embeds the ``SystemMessage`` content into
    that ``HumanMessage`` so the agent's role instruction is always present.

    These tests fail against the pre-fix codebase and pass after.
    """

    def test_agent_a_prompt_contains_its_own_system_prompt(self):
        captured, _ = _run_capturing_prompts(message_format="last")
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries, "No invocation captured for agent_a"
        prompt = a_entries[0]["rendered_prompt"]
        assert _AGENT_A_SYSTEM in prompt, (
            f"agent_a prompt missing its system_prompt.\n"
            f"Expected to find: {_AGENT_A_SYSTEM!r}\n"
            f"Actual prompt:    {prompt!r}"
        )

    def test_agent_b_prompt_contains_its_own_system_prompt(self):
        captured, _ = _run_capturing_prompts(mas_id="sp_b_test", message_format="last")
        b_entries = [c for c in captured if c["agent_id"] == "agent_b"]
        assert b_entries, "No invocation captured for agent_b"
        prompt = b_entries[0]["rendered_prompt"]
        assert _AGENT_B_SYSTEM in prompt, (
            f"agent_b prompt missing its system_prompt.\n"
            f"Expected to find: {_AGENT_B_SYSTEM!r}\n"
            f"Actual prompt:    {prompt!r}"
        )

    def test_agents_receive_different_system_prompts(self):
        """Each agent's CLI prompt is distinct — they carry different role instructions."""
        captured, _ = _run_capturing_prompts(
            mas_id="sp_distinct_test", message_format="last"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        b_entries = [c for c in captured if c["agent_id"] == "agent_b"]
        assert a_entries and b_entries
        prompt_a = a_entries[0]["rendered_prompt"]
        prompt_b = b_entries[0]["rendered_prompt"]
        assert prompt_a != prompt_b, (
            "agent_a and agent_b received identical prompts — "
            "per-agent system_prompts are not being injected"
        )

    def test_agent_a_prompt_does_not_contain_agent_b_system_prompt(self):
        """Cross-contamination check: agent_a must not receive agent_b's instructions."""
        captured, _ = _run_capturing_prompts(
            mas_id="sp_cross_test", message_format="last"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        prompt_a = a_entries[0]["rendered_prompt"]
        assert _AGENT_B_SYSTEM not in prompt_a, (
            f"agent_a prompt contains agent_b's system_prompt (cross-contamination).\n"
            f"Prompt: {prompt_a!r}"
        )

    def test_system_message_not_forwarded_raw_to_last_format_node(self):
        """For "last" format, the messages list passed to the MCP node has no SystemMessage.

        The system instruction is embedded into the HumanMessage instead.
        A raw SystemMessage in the list would be ignored by render_messages("last"),
        which is the original bug — this confirms the workaround is applied.
        """
        captured, _ = _run_capturing_prompts(
            mas_id="sp_nosys_test", message_format="last"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        messages = a_entries[0]["messages"]
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert not sys_msgs, (
            f"For 'last' format, SystemMessage should be removed from the "
            f"messages list before the MCP node renders them. "
            f"Found SystemMessages: {[m.content for m in sys_msgs]}"
        )

    def test_last_human_message_contains_system_instruction(self):
        """The last HumanMessage in the prepared list carries the system instruction."""
        captured, _ = _run_capturing_prompts(mas_id="sp_hm_test", message_format="last")
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        messages = a_entries[0]["messages"]
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert human_msgs, "No HumanMessage found in messages passed to MCP node"
        last_hm = human_msgs[-1]
        assert _AGENT_A_SYSTEM in last_hm.content, (
            f"Last HumanMessage does not contain the system instruction.\n"
            f"Expected to find: {_AGENT_A_SYSTEM!r}\n"
            f"HumanMessage content: {last_hm.content!r}"
        )

    def test_original_task_content_preserved_in_prompt(self):
        """The original task text is preserved after system_prompt prepend."""
        captured, _ = _run_capturing_prompts(
            mas_id="sp_task_test", message_format="last"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        prompt = a_entries[0]["rendered_prompt"]
        # The original HumanMessage content should appear after the system instruction
        assert (
            "begin workflow" in prompt or "Begin your task" in prompt
        ), f"Original task content missing from prompt.\nPrompt: {prompt!r}"


# ---------------------------------------------------------------------------
# "roles" format: SystemMessage forwarded as-is (no embedding needed)
# ---------------------------------------------------------------------------


class TestSystemPromptInjectionRolesFormat:
    """For message_format="roles", SystemMessage is rendered directly — no embedding.

    The "roles" format prefixes each message with its role label, so
    ``SystemMessage(content=...)`` becomes ``System: <content>``.  The fix
    should NOT embed the system instruction into the HumanMessage for this
    format.
    """

    def test_agent_a_roles_prompt_contains_system_prompt(self):
        captured, _ = _run_capturing_prompts(
            mas_id="sp_roles_a", message_format="roles"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries, "No invocation captured for agent_a (roles format)"
        prompt = a_entries[0]["rendered_prompt"]
        assert _AGENT_A_SYSTEM in prompt, (
            f"agent_a prompt (roles format) missing system_prompt.\n"
            f"Prompt: {prompt!r}"
        )

    def test_agent_b_roles_prompt_contains_system_prompt(self):
        captured, _ = _run_capturing_prompts(
            mas_id="sp_roles_b", message_format="roles"
        )
        b_entries = [c for c in captured if c["agent_id"] == "agent_b"]
        assert b_entries, "No invocation captured for agent_b (roles format)"
        prompt = b_entries[0]["rendered_prompt"]
        assert _AGENT_B_SYSTEM in prompt, (
            f"agent_b prompt (roles format) missing system_prompt.\n"
            f"Prompt: {prompt!r}"
        )

    def test_roles_format_preserves_system_message_type(self):
        """For "roles" format, the SystemMessage is NOT embedded into HumanMessage."""
        captured, _ = _run_capturing_prompts(
            mas_id="sp_roles_sys", message_format="roles"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        messages = a_entries[0]["messages"]
        # SystemMessage must be in the list (it is rendered by roles format)
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert sys_msgs, (
            "For 'roles' format, SystemMessage should be forwarded directly "
            "to the MCP node (not embedded into HumanMessage)"
        )


# ---------------------------------------------------------------------------
# Prompt-content unit tests: _invoke_executor embedding logic in isolation
# ---------------------------------------------------------------------------


class TestMcpInvokeExecutorEmbedding:
    """Unit tests that drive the embedding logic via the compiled graph.

    These tests confirm the prompt the MCP inner node receives has the right
    structure, without asserting on higher-level state like agent_outputs.
    """

    def test_system_prompt_appears_before_task_content_in_last_format(self):
        captured, _ = _run_capturing_prompts(
            mas_id="sp_order_test", message_format="last"
        )
        a_entries = [c for c in captured if c["agent_id"] == "agent_a"]
        assert a_entries
        prompt = a_entries[0]["rendered_prompt"]
        sys_pos = prompt.find(_AGENT_A_SYSTEM)
        # Find any task content that appears after the system instruction
        task_pos = prompt.find("begin workflow", sys_pos)
        if task_pos == -1:
            task_pos = prompt.find("Begin your task", sys_pos)
        assert sys_pos >= 0, "System prompt not found in rendered prompt"
        assert (
            task_pos > sys_pos
        ), "Task content appears BEFORE the system instruction in the prompt"

    def test_no_system_message_in_last_format_mcp_node_input(self):
        """Verifies that embedding removes the raw SystemMessage for "last" format."""
        captured, _ = _run_capturing_prompts(
            mas_id="sp_noraw_test", message_format="last"
        )
        for entry in captured:
            sys_in_list = [m for m in entry["messages"] if isinstance(m, SystemMessage)]
            assert not sys_in_list, (
                f"Agent {entry['agent_id']!r} has raw SystemMessage in MCP node input "
                f"(would be silently discarded by render_messages('last'))"
            )
