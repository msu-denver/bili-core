"""Tests for add_persona_and_summary node.

Tests the persona and summary injection functionality:
- Builder returns a callable
- Injects persona as SystemMessage at the start
- Appends summary when present in state
- Appends historical context when present
- Handles template substitution via template_dict in state
- Replaces existing SystemMessage at position 0
- Works with empty message list
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from bili.iris.nodes.add_persona_and_summary import (
    build_add_persona_and_summary_node,
    persona_and_summary_node,
)


class TestBuildAddPersonaAndSummaryNode:
    """Tests for build_add_persona_and_summary_node function."""

    def test_returns_callable(self):
        """Builder should return a callable function."""
        node_func = build_add_persona_and_summary_node(persona="You are helpful.")
        assert callable(node_func)

    def test_injects_persona_as_system_message(self):
        """Persona should be injected as SystemMessage at index 0."""
        node_func = build_add_persona_and_summary_node(persona="You are a scientist.")
        state = {
            "messages": [HumanMessage(content="Hello")],
        }

        result = node_func(state)

        first_msg = result["messages"][0]
        assert isinstance(first_msg, SystemMessage)
        assert "You are a scientist." in first_msg.content

    def test_appends_summary_when_present(self):
        """Summary from state should be appended to the persona."""
        node_func = build_add_persona_and_summary_node(persona="You are helpful.")
        state = {
            "messages": [HumanMessage(content="Hi")],
            "summary": "User asked about weather.",
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "You are helpful." in content
        assert "User asked about weather." in content
        assert "Summary of the current conversation" in content

    def test_no_summary_when_empty(self):
        """Empty summary should not be appended."""
        node_func = build_add_persona_and_summary_node(persona="You are helpful.")
        state = {
            "messages": [HumanMessage(content="Hi")],
            "summary": "",
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "Summary of the current conversation" not in content

    def test_no_summary_key_in_state(self):
        """Missing summary key should not cause an error."""
        node_func = build_add_persona_and_summary_node(persona="You are helpful.")
        state = {"messages": [HumanMessage(content="Hi")]}

        result = node_func(state)

        content = result["messages"][0].content
        assert content == "You are helpful."

    def test_appends_historical_context(self):
        """Historical context should be appended to persona."""
        node_func = build_add_persona_and_summary_node(persona="You are helpful.")
        state = {
            "messages": [HumanMessage(content="Hi")],
            "historical_context": "Previous session discussed AI.",
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "Previous session context" in content
        assert "Previous session discussed AI." in content

    def test_both_historical_context_and_summary(self):
        """Both historical context and summary should appear."""
        node_func = build_add_persona_and_summary_node(persona="Base persona.")
        state = {
            "messages": [HumanMessage(content="Hi")],
            "historical_context": "Old context",
            "summary": "Current summary",
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "Old context" in content
        assert "Current summary" in content

    def test_replaces_existing_system_message(self):
        """Existing SystemMessage at index 0 should be marked for removal."""
        old_sys = SystemMessage(content="Old persona")
        node_func = build_add_persona_and_summary_node(persona="New persona.")
        state = {
            "messages": [
                old_sys,
                HumanMessage(content="Hello"),
            ],
        }

        result = node_func(state)

        messages = result["messages"]
        # First message should be new SystemMessage
        assert isinstance(messages[0], SystemMessage)
        assert "New persona." in messages[0].content

        # Should contain a RemoveMessage for the old SystemMessage
        remove_msgs = [m for m in messages if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 1
        assert remove_msgs[0].id == old_sys.id

    def test_no_replace_when_first_is_not_system(self):
        """No RemoveMessage when first message is not SystemMessage."""
        node_func = build_add_persona_and_summary_node(persona="Persona.")
        state = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content="Hello"),
            ],
        }

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 0

    def test_empty_message_list(self):
        """Should work with an empty message list."""
        node_func = build_add_persona_and_summary_node(persona="Empty test.")
        state = {"messages": []}

        result = node_func(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Empty test." in result["messages"][0].content

    def test_template_dict_substitution(self):
        """Template placeholders should be replaced from state."""
        node_func = build_add_persona_and_summary_node(
            persona="You are a {role} in {dept}."
        )
        state = {
            "messages": [HumanMessage(content="Hi")],
            "template_dict": {
                "role": "manager",
                "dept": "engineering",
            },
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "manager" in content
        assert "engineering" in content
        assert "{role}" not in content

    def test_template_dict_ignores_extra_keys(self):
        """Extra keys in template_dict should be ignored."""
        node_func = build_add_persona_and_summary_node(persona="Hello {name}.")
        state = {
            "messages": [],
            "template_dict": {
                "name": "Alice",
                "unused": "value",
            },
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "Hello Alice." in content

    def test_template_dict_leaves_unknown_placeholders(self):
        """Placeholders not in template_dict stay unchanged."""
        node_func = build_add_persona_and_summary_node(
            persona="Hi {name}, welcome to {place}."
        )
        state = {
            "messages": [],
            "template_dict": {"name": "Bob"},
        }

        result = node_func(state)

        content = result["messages"][0].content
        assert "Bob" in content
        assert "{place}" in content

    def test_no_template_dict_in_state(self):
        """No template_dict should leave persona unchanged."""
        node_func = build_add_persona_and_summary_node(persona="Hello {name}.")
        state = {"messages": []}

        result = node_func(state)

        content = result["messages"][0].content
        assert "{name}" in content

    def test_preserves_non_system_messages(self):
        """Other messages should be preserved in the output."""
        node_func = build_add_persona_and_summary_node(persona="Persona.")
        human = HumanMessage(content="user msg")
        ai = AIMessage(content="ai msg")
        state = {"messages": [human, ai]}

        result = node_func(state)

        contents = [
            m.content for m in result["messages"] if not isinstance(m, RemoveMessage)
        ]
        assert "user msg" in contents
        assert "ai msg" in contents

    def test_accepts_extra_kwargs(self):
        """Builder should accept extra kwargs without error."""
        node_func = build_add_persona_and_summary_node(persona="test", extra="ignored")
        assert callable(node_func)


class TestPersonaAndSummaryNodePartial:
    """Tests for the persona_and_summary_node partial."""

    def test_partial_creates_node_with_correct_name(self):
        """The partial should produce a Node named correctly."""
        node = persona_and_summary_node()
        assert node.name == "add_persona_and_summary"

    def test_partial_creates_callable_node(self):
        """The Node created by the partial should be callable."""
        node = persona_and_summary_node()
        assert callable(node)

    def test_partial_call_invokes_builder(self):
        """Calling the Node should invoke the builder function."""
        node = persona_and_summary_node()
        result = node(persona="You are helpful.")
        assert callable(result)
