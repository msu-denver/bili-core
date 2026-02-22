"""Tests for inject_current_datetime node.

Tests the datetime injection functionality:
- Injecting datetime into existing SystemMessage
- Handling empty message list
- Handling non-SystemMessage first message
- Removing old SystemMessage and inserting new one
"""

# pylint: disable=missing-function-docstring

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bili.nodes.inject_current_datetime import build_inject_current_date_time


class TestInjectCurrentDateTime:
    """Tests for build_inject_current_date_time function."""

    def test_injects_datetime_into_system_message(self):
        """Test that datetime is appended to existing SystemMessage."""
        # Create initial state with SystemMessage
        system_msg = SystemMessage(content="You are a helpful assistant.")
        state = {"messages": [system_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Verify result
        assert "messages" in result
        assert len(result["messages"]) > 0

        # First message should be SystemMessage with datetime appended
        first_message = result["messages"][0]
        assert isinstance(first_message, SystemMessage)
        assert "You are a helpful assistant." in first_message.content
        assert "The current time in UTC is" in first_message.content

    def test_datetime_format_contains_utc_datetime(self):
        """Test that injected datetime contains valid UTC datetime string."""
        system_msg = SystemMessage(content="System prompt")
        state = {"messages": [system_msg]}

        # Get current time before execution
        time_before = datetime.now(timezone.utc)

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Get current time after execution
        datetime.now(timezone.utc)

        # Verify datetime is in the message
        first_message = result["messages"][0]
        content = first_message.content

        # Check format and approximate time match
        assert "The current time in UTC is" in content
        # Extract the year to verify it's a real datetime
        assert str(time_before.year) in content

    def test_empty_message_list_returns_unchanged(self):
        """Test that empty messages list returns state unchanged."""
        state = {"messages": []}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Should return original state
        assert result == state

    def test_no_messages_key_returns_unchanged(self):
        """Test that state without messages key returns unchanged."""
        state = {"other_key": "value"}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Should return original state
        assert result == state

    def test_first_message_not_system_message_returns_unchanged(self):
        """Test that non-SystemMessage first message returns state unchanged."""
        human_msg = HumanMessage(content="Hello")
        state = {"messages": [human_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Should return original state
        assert result == state

    def test_preserves_other_messages(self):
        """Test that other messages in the list are preserved."""
        system_msg = SystemMessage(content="System prompt")
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")

        state = {"messages": [system_msg, human_msg, ai_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Check that result has messages
        assert "messages" in result
        messages = result["messages"]

        # Should have additional RemoveMessage plus original messages
        # First should be updated SystemMessage
        assert isinstance(messages[0], SystemMessage)
        assert "The current time in UTC is" in messages[0].content

        # Other messages should be preserved somewhere in the list
        # (The exact structure includes RemoveMessage for old system message)
        assert any(isinstance(m, HumanMessage) for m in messages)
        assert any(isinstance(m, AIMessage) for m in messages)

    def test_removes_old_system_message(self):
        """Test that old SystemMessage is marked for removal."""
        from langchain_core.messages import RemoveMessage

        system_msg = SystemMessage(content="Original system message")
        state = {"messages": [system_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        messages = result["messages"]

        # Should contain a RemoveMessage for the old system message
        remove_messages = [m for m in messages if isinstance(m, RemoveMessage)]
        assert len(remove_messages) == 1
        assert remove_messages[0].id == system_msg.id

    def test_node_function_is_callable(self):
        """Test that build_inject_current_date_time returns a callable."""
        node_func = build_inject_current_date_time()
        assert callable(node_func)

    def test_kwargs_are_accepted(self):
        """Test that function accepts arbitrary kwargs for extensibility."""
        # Should not raise even with extra kwargs
        node_func = build_inject_current_date_time(some_param="value", another=123)
        assert callable(node_func)

    def test_system_message_with_existing_datetime(self):
        """Test updating system message that already contains datetime info."""
        system_msg = SystemMessage(
            content="You are a helpful assistant. Previous time: 2024-01-01"
        )
        state = {"messages": [system_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Should still append new datetime
        first_message = result["messages"][0]
        assert isinstance(first_message, SystemMessage)
        assert "Previous time: 2024-01-01" in first_message.content
        assert "The current time in UTC is" in first_message.content

    def test_multiple_system_messages_only_updates_first(self):
        """Test that only the first message is checked and updated."""
        system_msg1 = SystemMessage(content="First system message")
        human_msg = HumanMessage(content="Hello")
        system_msg2 = SystemMessage(content="Second system message")

        state = {"messages": [system_msg1, human_msg, system_msg2]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        messages = result["messages"]

        # First message should be updated SystemMessage
        assert isinstance(messages[0], SystemMessage)
        assert "The current time in UTC is" in messages[0].content

        # Second SystemMessage should be unchanged
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        # Should have 2 SystemMessages (updated first + unchanged second)
        second_system = [m for m in system_msgs if "Second system message" in m.content]
        assert len(second_system) == 1
        assert "The current time in UTC is" not in second_system[0].content

    def test_empty_system_message_content(self):
        """Test handling of SystemMessage with empty content."""
        system_msg = SystemMessage(content="")
        state = {"messages": [system_msg]}

        # Execute node
        node_func = build_inject_current_date_time()
        result = node_func(state)

        # Should still inject datetime
        first_message = result["messages"][0]
        assert isinstance(first_message, SystemMessage)
        assert "The current time in UTC is" in first_message.content
