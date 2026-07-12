"""Unit tests for bili.iris.tools.ask_user registration/unregistration.

End-to-end pause/resume behavior is covered by the integration tests
(test_ask_user_iris_integration.py, and AETHER's
test_ask_user_aether_integration.py) -- this module covers the registration
lifecycle and edge cases those don't exercise: double registration,
unregister when absent, and the tool's shape (name/description/schema)
independent of a running graph.
"""

# pylint: disable=missing-function-docstring

import logging

import pytest

from bili.iris.loaders.tools_loader import TOOL_REGISTRY
from bili.iris.tools.ask_user import (
    ASK_USER_TOOL_NAME,
    register_ask_user_tool,
    unregister_ask_user_tool,
)
from bili.iris.tools.hitl import ScriptedHitlResponder


class TestRegisterAskUserTool:
    """Tests for register_ask_user_tool()."""

    def teardown_method(self):
        unregister_ask_user_tool()

    def test_registers_into_tool_registry(self):
        register_ask_user_tool()
        assert ASK_USER_TOOL_NAME in TOOL_REGISTRY

    def test_registered_tool_has_expected_name_and_description(self):
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        assert tool.name == ASK_USER_TOOL_NAME
        assert "human" in tool.description.lower()

    def test_registered_tool_accepts_question_and_optional_options(self):
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        schema_fields = tool.args_schema.model_fields
        assert "question" in schema_fields
        assert "options" in schema_fields
        assert schema_fields["question"].is_required()
        assert not schema_fields["options"].is_required()

    def test_tool_func_reaches_interrupt_outside_a_graph(self):
        """Confirms the native tool's func calls langgraph.types.interrupt()
        rather than silently returning a placeholder, by observing the
        specific failure interrupt() itself raises when called with no
        active LangGraph runnable context: RuntimeError from
        langgraph.config.get_config(), not GraphInterrupt (GraphInterrupt
        requires the runnable-context machinery interrupt() failed to find
        here). The pause/resume-with-an-answer path is exercised for real
        inside a compiled graph by the integration tests
        (test_ask_user_iris_integration.py, test_ask_user_aether_integration.py).
        """
        register_ask_user_tool()
        tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        with pytest.raises(RuntimeError, match="runnable context"):
            tool.func(question="Which environment?")

    def test_double_registration_replaces_and_warns(self, caplog):
        register_ask_user_tool()
        first_tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})

        with caplog.at_level(logging.WARNING):
            register_ask_user_tool()

        assert any("already registered" in record.message for record in caplog.records)
        second_tool = TOOL_REGISTRY[ASK_USER_TOOL_NAME](None, None, {})
        assert first_tool is not second_tool

    def test_accepts_explicit_responder_without_error(self):
        # Chunk 1: the native tool does not call *responder* (see
        # _build_ask_user_func's docstring) -- this test only verifies
        # register_ask_user_tool() accepts one without raising, ahead of the
        # CLI/MCP-path wiring that will consume it.
        responder = ScriptedHitlResponder(["staging"])
        register_ask_user_tool(responder)
        assert ASK_USER_TOOL_NAME in TOOL_REGISTRY


class TestUnregisterAskUserTool:
    """Tests for unregister_ask_user_tool()."""

    def test_removes_from_tool_registry(self):
        register_ask_user_tool()
        unregister_ask_user_tool()
        assert ASK_USER_TOOL_NAME not in TOOL_REGISTRY

    def test_noop_when_not_registered(self):
        unregister_ask_user_tool()  # already absent; must not raise
        unregister_ask_user_tool()  # calling twice must also not raise
        assert ASK_USER_TOOL_NAME not in TOOL_REGISTRY
