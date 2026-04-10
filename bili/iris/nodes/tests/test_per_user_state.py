"""Tests for per_user_state node.

Tests the per-user state injection functionality:
- Builder returns a callable
- Injects user profile as HumanMessage after SystemMessage
- Returns state unmodified when no current_user
- Replaces existing profile message at position 1
- Inserts at position 0 when no SystemMessage is first
- Works with empty message list
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from bili.iris.nodes.per_user_state import buld_per_user_state_node, per_user_state_node


class TestBuildPerUserStateNode:
    """Tests for buld_per_user_state_node function."""

    def test_returns_callable(self):
        """Builder should return a callable function."""
        node_func = buld_per_user_state_node(current_user={"uid": "u1"})
        assert callable(node_func)

    def test_returns_callable_without_user(self):
        """Builder should return a callable even with no user."""
        node_func = buld_per_user_state_node(current_user=None)
        assert callable(node_func)

    def test_no_user_returns_messages_unchanged(self):
        """With no current_user, messages should pass through."""
        node_func = buld_per_user_state_node(current_user=None)
        human = HumanMessage(content="Hello")
        state = {"messages": [human]}

        result = node_func(state)

        assert result["messages"] == [human]

    def test_injects_profile_after_system_message(self):
        """Profile should be inserted at index 1 after SystemMessage."""
        user = {"uid": "u1", "name": "Alice"}
        node_func = buld_per_user_state_node(current_user=user)
        sys_msg = SystemMessage(content="System prompt")
        human = HumanMessage(content="Hi")
        state = {"messages": [sys_msg, human]}

        result = node_func(state)

        messages = result["messages"]
        # Index 0: SystemMessage, Index 1: profile HumanMessage
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert "USER PROFILE:" in messages[1].content
        assert "Alice" in messages[1].content

    def test_injects_profile_at_start_without_system_message(self):
        """Profile should be at index 0 when no SystemMessage leads."""
        user = {"uid": "u2", "name": "Bob"}
        node_func = buld_per_user_state_node(current_user=user)
        human = HumanMessage(content="Hi")
        state = {"messages": [human]}

        result = node_func(state)

        first_msg = result["messages"][0]
        assert isinstance(first_msg, HumanMessage)
        assert "USER PROFILE:" in first_msg.content
        assert "Bob" in first_msg.content

    def test_profile_contains_user_json(self):
        """Profile message should contain JSON of the user dict."""
        user = {"uid": "u3", "role": "researcher"}
        node_func = buld_per_user_state_node(current_user=user)
        state = {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="hi"),
            ]
        }

        result = node_func(state)

        profile_msg = result["messages"][1]
        assert '"uid": "u3"' in profile_msg.content
        assert '"role": "researcher"' in profile_msg.content

    def test_replaces_existing_profile_message(self):
        """Existing profile at position 1 should be marked for removal."""
        user = {"uid": "u4", "name": "Carol"}
        node_func = buld_per_user_state_node(current_user=user)

        old_profile = HumanMessage(content="USER PROFILE: old data")
        state = {
            "messages": [
                SystemMessage(content="sys"),
                old_profile,
                HumanMessage(content="question"),
            ]
        }

        result = node_func(state)

        messages = result["messages"]
        # Should have a RemoveMessage for the old profile
        remove_msgs = [m for m in messages if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 1
        assert remove_msgs[0].id == old_profile.id

        # New profile should be at index 1
        assert "USER PROFILE:" in messages[1].content
        assert "Carol" in messages[1].content

    def test_does_not_replace_non_profile_human_message(self):
        """Regular HumanMessage at position 1 should not be removed."""
        user = {"uid": "u5"}
        node_func = buld_per_user_state_node(current_user=user)
        state = {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="Just a question"),
            ]
        }

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 0

    def test_empty_messages_with_user(self):
        """Empty message list with a user should insert profile."""
        user = {"uid": "u6"}
        node_func = buld_per_user_state_node(current_user=user)
        state = {"messages": []}

        result = node_func(state)

        assert len(result["messages"]) == 1
        assert "USER PROFILE:" in result["messages"][0].content

    def test_empty_messages_without_user(self):
        """Empty message list with no user should stay empty."""
        node_func = buld_per_user_state_node(current_user=None)
        state = {"messages": []}

        result = node_func(state)

        assert result["messages"] == []

    def test_preserves_existing_messages(self):
        """Existing messages should be preserved in the output."""
        user = {"uid": "u7"}
        node_func = buld_per_user_state_node(current_user=user)
        sys_msg = SystemMessage(content="sys")
        human = HumanMessage(content="question")
        ai = AIMessage(content="answer")
        state = {"messages": [sys_msg, human, ai]}

        result = node_func(state)

        contents = [
            m.content for m in result["messages"] if not isinstance(m, RemoveMessage)
        ]
        assert "sys" in contents
        assert "question" in contents
        assert "answer" in contents

    def test_accepts_extra_kwargs(self):
        """Builder should accept extra kwargs without error."""
        node_func = buld_per_user_state_node(current_user=None, extra="val")
        assert callable(node_func)


class TestPerUserStateNodePartial:
    """Tests for the per_user_state_node partial."""

    def test_partial_creates_node_with_correct_name(self):
        """The partial should produce a Node named 'per_user_state'."""
        node = per_user_state_node()
        assert node.name == "per_user_state"

    def test_partial_creates_callable_node(self):
        """The Node created by the partial should be callable."""
        node = per_user_state_node()
        assert callable(node)

    def test_partial_call_invokes_builder(self):
        """Calling the Node should invoke the builder function."""
        node = per_user_state_node()
        result = node(current_user=None)
        assert callable(result)
