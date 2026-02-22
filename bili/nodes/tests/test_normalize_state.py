"""Tests for normalize_state node.

Tests the state normalization functionality:
- Removing function_call from additional_kwargs
- Marking invalid tool call messages for removal
- Marking empty AI messages for removal
- Handling messages with no normalization needed
"""

# pylint: disable=missing-function-docstring

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from bili.nodes.normalize_state import build_normalize_state_node


class TestNormalizeState:
    """Tests for build_normalize_state_node function."""

    def test_removes_function_call_from_additional_kwargs(self):
        """Test that function_call is removed from additional_kwargs."""
        # Create message with function_call in additional_kwargs
        ai_msg = AIMessage(
            content="Response",
            additional_kwargs={"function_call": {"name": "test_func"}, "other": "data"},
        )
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # function_call should be removed (modified in place)
        assert "function_call" not in ai_msg.additional_kwargs
        assert "other" in ai_msg.additional_kwargs  # Other data preserved

        # Should return empty dict (no messages to remove)
        assert result == {}

    def test_marks_invalid_tool_calls_for_removal(self):
        """Test that AI messages with invalid_tool_calls are marked for removal."""
        # Create message with invalid_tool_calls attribute
        ai_msg = AIMessage(content="Response")
        ai_msg.invalid_tool_calls = [{"error": "Invalid"}]

        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should return RemoveMessage
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], RemoveMessage)
        assert result["messages"][0].id == ai_msg.id

    def test_marks_empty_ai_messages_for_removal(self):
        """Test that AI messages with empty content are marked for removal."""
        ai_msg = AIMessage(content="")
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should return RemoveMessage
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], RemoveMessage)
        assert result["messages"][0].id == ai_msg.id

    def test_non_empty_ai_messages_not_removed(self):
        """Test that AI messages with content are not removed."""
        ai_msg = AIMessage(content="Valid response")
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should return empty dict (no removals)
        assert result == {}

    def test_handles_multiple_issues_in_single_message(self):
        """Test handling message with both function_call and empty content."""
        ai_msg = AIMessage(
            content="", additional_kwargs={"function_call": {"name": "test"}}
        )
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # function_call should be removed
        assert "function_call" not in ai_msg.additional_kwargs

        # Message should be marked for removal (empty content)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], RemoveMessage)

    def test_processes_multiple_messages(self):
        """Test processing multiple messages with various normalization needs."""
        msg1 = AIMessage(
            content="Valid", additional_kwargs={"function_call": {"name": "func1"}}
        )
        msg2 = AIMessage(content="")  # Empty - should be removed
        msg3 = HumanMessage(content="User input")
        msg4 = AIMessage(content="Another response")

        state = {"messages": [msg1, msg2, msg3, msg4]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # msg1's function_call should be removed
        assert "function_call" not in msg1.additional_kwargs

        # Only msg2 should be marked for removal
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].id == msg2.id

    def test_no_changes_returns_empty_dict(self):
        """Test that messages requiring no normalization return empty dict."""
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi there")
        msg3 = SystemMessage(content="System prompt")

        state = {"messages": [msg1, msg2, msg3]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # No changes needed
        assert result == {}

    def test_human_messages_not_affected(self):
        """Test that HumanMessage types are not affected by normalization."""
        human_msg = HumanMessage(content="")  # Empty content
        state = {"messages": [human_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # HumanMessage should not be removed even if empty
        assert result == {}

    def test_system_messages_not_affected(self):
        """Test that SystemMessage types are not affected by normalization."""
        system_msg = SystemMessage(content="")
        state = {"messages": [system_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # SystemMessage should not be removed
        assert result == {}

    def test_message_without_additional_kwargs(self):
        """Test handling message without additional_kwargs attribute."""
        ai_msg = AIMessage(content="Response")
        # Ensure no additional_kwargs
        if hasattr(ai_msg, "additional_kwargs"):
            ai_msg.additional_kwargs = {}

        state = {"messages": [ai_msg]}

        # Execute node (should not crash)
        node_func = build_normalize_state_node()
        result = node_func(state)

        assert result == {}

    def test_message_with_empty_additional_kwargs(self):
        """Test handling message with empty additional_kwargs dict."""
        ai_msg = AIMessage(content="Response", additional_kwargs={})
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        assert result == {}

    def test_preserves_other_additional_kwargs(self):
        """Test that other additional_kwargs fields are preserved."""
        ai_msg = AIMessage(
            content="Response",
            additional_kwargs={
                "function_call": {"name": "test"},
                "model": "gpt-4",
                "tokens": 100,
            },
        )
        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        node_func(state)

        # function_call removed, others preserved
        assert "function_call" not in ai_msg.additional_kwargs
        assert ai_msg.additional_kwargs["model"] == "gpt-4"
        assert ai_msg.additional_kwargs["tokens"] == 100

    def test_invalid_tool_calls_false_not_removed(self):
        """Test that messages with invalid_tool_calls=False are not removed."""
        ai_msg = AIMessage(content="Response")
        ai_msg.invalid_tool_calls = False

        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should not be removed (invalid_tool_calls is falsy)
        assert result == {}

    def test_invalid_tool_calls_empty_list_not_removed(self):
        """Test that messages with empty invalid_tool_calls list are not removed."""
        ai_msg = AIMessage(content="Response")
        ai_msg.invalid_tool_calls = []

        state = {"messages": [ai_msg]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should not be removed (empty list is falsy)
        assert result == {}

    def test_node_function_is_callable(self):
        """Test that build_normalize_state_node returns a callable."""
        node_func = build_normalize_state_node()
        assert callable(node_func)

    def test_kwargs_are_accepted(self):
        """Test that function accepts arbitrary kwargs for extensibility."""
        node_func = build_normalize_state_node(some_param="value")
        assert callable(node_func)

    def test_multiple_removals(self):
        """Test removing multiple messages in one pass."""
        msg1 = AIMessage(content="", id="msg1")  # Empty
        msg2 = AIMessage(content="Valid", id="msg2")
        msg3 = AIMessage(content="", id="msg3")  # Empty
        msg4 = AIMessage(content="Also valid", id="msg4")

        msg2_invalid = AIMessage(content="Has invalid", id="msg_invalid")
        msg2_invalid.invalid_tool_calls = [{"error": "bad"}]

        state = {"messages": [msg1, msg2, msg3, msg4, msg2_invalid]}

        # Execute node
        node_func = build_normalize_state_node()
        result = node_func(state)

        # Should have 3 RemoveMessage objects (msg1, msg3, msg2_invalid)
        assert "messages" in result
        assert len(result["messages"]) == 3

        removal_ids = {rm.id for rm in result["messages"]}
        assert msg1.id in removal_ids
        assert msg3.id in removal_ids
        assert msg2_invalid.id in removal_ids
        assert msg2.id not in removal_ids
        assert msg4.id not in removal_ids
