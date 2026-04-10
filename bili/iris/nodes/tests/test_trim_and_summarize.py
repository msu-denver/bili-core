"""Tests for trim_and_summarize node.

Tests the memory management functionality:
- Builder returns a callable
- Pass-through when below message threshold
- Trim logic when message count exceeds k
- Summarization trigger and LLM invocation
- Preserves last human and AI messages
- Preserves user profile message
- Respects disable_summarization flag
- Custom trim_k threshold
- Handles summarization failure gracefully
"""

from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from bili.iris.nodes.trim_and_summarize import (
    build_trim_and_summarize_node,
    trim_summarize_node,
)


def _make_conversation(num_pairs, include_system=True):
    """Build a conversation with num_pairs of human/AI exchanges."""
    messages = []
    if include_system:
        messages.append(SystemMessage(content="System prompt", id="sys-0"))
    for i in range(num_pairs):
        messages.append(HumanMessage(content=f"Question {i}", id=f"h-{i}"))
        messages.append(AIMessage(content=f"Answer {i}", id=f"a-{i}"))
    return messages


class TestBuildTrimAndSummarizeNode:
    """Tests for build_trim_and_summarize_node function."""

    def test_returns_callable(self):
        """Builder should return a callable function."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(llm_model=mock_llm)
        assert callable(node_func)

    def test_passthrough_when_below_threshold(self):
        """No messages removed when count is below k."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(llm_model=mock_llm, k=20)
        messages = _make_conversation(3)
        state = {"messages": messages}

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 0

    def test_trims_when_above_threshold(self):
        """Messages should be trimmed when count exceeds k."""
        mock_llm = MagicMock()
        # k=5 means keep 5 messages max; conversation has system + 10
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=5,
            memory_strategy="trim",
        )
        messages = _make_conversation(5)
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) > 0

    def test_preserves_last_human_and_ai_messages(self):
        """Last human and AI messages should never be removed."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="trim",
        )
        messages = _make_conversation(5)
        last_human = messages[-2]
        last_ai = messages[-1]
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        removed_ids = [m.id for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert last_human.id not in removed_ids
        assert last_ai.id not in removed_ids

    def test_preserves_user_profile_message(self):
        """USER PROFILE message should not be trimmed."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="trim",
        )
        profile = HumanMessage(
            content="USER PROFILE: some data",
            id="profile-0",
        )
        messages = [
            SystemMessage(content="sys", id="sys-0"),
            profile,
        ]
        for i in range(5):
            messages.append(HumanMessage(content=f"Q{i}", id=f"h-{i}"))
            messages.append(AIMessage(content=f"A{i}", id=f"a-{i}"))
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        removed_ids = [m.id for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert profile.id not in removed_ids

    def test_summarize_strategy_invokes_llm(self):
        """Summarize strategy should invoke the LLM on removed messages."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summarized conversation."
        mock_llm.invoke.return_value = mock_response

        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="summarize",
        )
        messages = _make_conversation(5)
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        mock_llm.invoke.assert_called_once()
        assert result["summary"] == "Summarized conversation."

    def test_summarize_with_existing_summary(self):
        """Existing summary should be included in summarization prompt."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Updated summary."
        mock_llm.invoke.return_value = mock_response

        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="summarize",
        )
        messages = _make_conversation(5)
        state = {
            "messages": messages,
            "summary": "Old summary content",
        }

        node_func(state)

        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "Old summary content" in prompt_arg

    def test_trim_strategy_does_not_invoke_llm(self):
        """Trim strategy should not call the LLM for summarization."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="trim",
        )
        messages = _make_conversation(5)
        state = {"messages": messages, "summary": ""}

        node_func(state)

        mock_llm.invoke.assert_not_called()

    def test_disable_summarization_flag(self):
        """When disable_summarization is True, return state as-is."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(llm_model=mock_llm, k=3)
        messages = _make_conversation(5)
        state = {
            "messages": messages,
            "summary": "",
            "disable_summarization": True,
        }

        result = node_func(state)

        assert result is state

    def test_disable_summarization_false_still_trims(self):
        """Explicit False for disable_summarization should trim."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="trim",
        )
        messages = _make_conversation(5)
        state = {
            "messages": messages,
            "summary": "",
            "disable_summarization": False,
        }

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) > 0

    def test_custom_trim_k(self):
        """Custom trim_k should apply a secondary trim threshold."""
        mock_llm = MagicMock()
        # k=10 triggers, trim_k=3 trims further
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=10,
            trim_k=3,
            memory_strategy="trim",
        )
        messages = _make_conversation(8)
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) > 0

    def test_summarization_failure_keeps_old_summary(self):
        """Failed summarization should preserve the existing summary."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM error")

        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="summarize",
        )
        messages = _make_conversation(5)
        state = {
            "messages": messages,
            "summary": "Existing summary",
        }

        result = node_func(state)

        assert result["summary"] == "Existing summary"

    def test_separate_summarize_llm_model(self):
        """A separate summarize_llm_model should be used if provided."""
        primary_llm = MagicMock()
        summarize_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary from secondary."
        summarize_llm.invoke.return_value = mock_response

        node_func = build_trim_and_summarize_node(
            llm_model=primary_llm,
            summarize_llm_model=summarize_llm,
            k=3,
            memory_strategy="summarize",
        )
        messages = _make_conversation(5)
        state = {"messages": messages, "summary": ""}

        result = node_func(state)

        summarize_llm.invoke.assert_called_once()
        primary_llm.invoke.assert_not_called()
        assert result["summary"] == "Summary from secondary."

    def test_no_summary_change_when_nothing_trimmed(self):
        """Summary should remain unchanged when no messages are trimmed."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=50,
            memory_strategy="summarize",
        )
        messages = _make_conversation(3)
        state = {
            "messages": messages,
            "summary": "Old summary",
        }

        result = node_func(state)

        assert result["summary"] == "Old summary"
        mock_llm.invoke.assert_not_called()

    def test_custom_prompt_template(self):
        """Custom prompt_template should be used for summarization."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Custom summary."
        mock_llm.invoke.return_value = mock_response

        template = "Custom: {existing_summary_content} " "and {conversation_text}"
        node_func = build_trim_and_summarize_node(
            llm_model=mock_llm,
            k=3,
            memory_strategy="summarize",
            prompt_template=template,
        )
        messages = _make_conversation(5)
        state = {
            "messages": messages,
            "summary": "prev",
        }

        node_func(state)

        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert prompt_arg.startswith("Custom: prev and ")

    def test_accepts_extra_kwargs(self):
        """Builder should accept extra kwargs without error."""
        mock_llm = MagicMock()
        node_func = build_trim_and_summarize_node(llm_model=mock_llm, extra="val")
        assert callable(node_func)


class TestTrimSummarizeNodePartial:
    """Tests for the trim_summarize_node partial."""

    def test_partial_creates_node_with_correct_name(self):
        """The partial should produce a Node named 'trim_summarize'."""
        node = trim_summarize_node()
        assert node.name == "trim_summarize"

    def test_partial_creates_callable_node(self):
        """The Node created by the partial should be callable."""
        node = trim_summarize_node()
        assert callable(node)

    def test_partial_call_invokes_builder(self):
        """Calling the Node should invoke the builder function."""
        mock_llm = MagicMock()
        node = trim_summarize_node()
        result = node(llm_model=mock_llm)
        assert callable(result)
