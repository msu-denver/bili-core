"""Tests for bili.utils.langgraph_utils.

Covers format_message_with_citations, clear_state, the State
TypedDict schema, the UntrackedValue re-export, and the
persisted-vs-ephemeral field semantics the State docstring documents.
"""

from typing import Annotated, Optional

import langgraph.channels
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bili.utils import langgraph_utils
from bili.utils.langgraph_utils import (
    State,
    UntrackedValue,
    clear_state,
    format_message_with_citations,
)

# ------------------------------------------------------------------
# format_message_with_citations
# ------------------------------------------------------------------


class TestFormatMessageNoCitations:
    """AIMessage with no citation metadata."""

    def test_returns_content_only(self):
        """Plain AIMessage content is returned unchanged."""
        msg = AIMessage(content="Hello world")
        result = format_message_with_citations(msg)
        assert result == "Hello world"


class TestFormatMessageWithTitleAndUri:
    """AIMessage with citations containing both title and uri."""

    def test_appends_markdown_link(self):
        """Citations with title+uri produce markdown links."""
        msg = AIMessage(
            content="Some answer",
            response_metadata={
                "citation_metadata": {
                    "citations": [
                        {
                            "title": "My Source",
                            "uri": "https://example.com",
                        }
                    ]
                }
            },
        )
        result = format_message_with_citations(msg)
        assert "**Citations:**" in result
        assert "- [My Source](https://example.com)" in result


class TestFormatMessageUriOnly:
    """AIMessage with citations that have uri but no title."""

    def test_uses_uri_as_link_text(self):
        """URI-only citations use the uri as display text."""
        uri = "https://example.com/doc"
        msg = AIMessage(
            content="Answer",
            response_metadata={"citation_metadata": {"citations": [{"uri": uri}]}},
        )
        result = format_message_with_citations(msg)
        expected_link = f"- [{uri}]({uri})"
        assert expected_link in result


class TestFormatMessageEmptyCitations:
    """AIMessage with an empty citations list."""

    def test_no_citations_header_appended(self):
        """Empty citations list does not add a Citations header."""
        msg = AIMessage(
            content="No refs",
            response_metadata={"citation_metadata": {"citations": []}},
        )
        result = format_message_with_citations(msg)
        assert result == "No refs"


class TestFormatNonAIMessage:
    """Non-AIMessage input (e.g. HumanMessage)."""

    def test_returns_content_string(self):
        """HumanMessage returns its content directly."""
        msg = HumanMessage(content="user question")
        result = format_message_with_citations(msg)
        assert result == "user question"


class TestFormatMessageNoContentAttr:
    """Object with no content attribute at all."""

    def test_falls_back_to_str(self):
        """Objects lacking .content fall back to str()."""
        obj = 42
        result = format_message_with_citations(obj)
        assert result == "42"

    def test_plain_object_without_content(self):
        """Plain object without content attribute returns str."""

        class NoContent:
            """Object with no content attribute."""

            def __str__(self):
                """Return a fixed string."""
                return "no-content-obj"

        result = format_message_with_citations(NoContent())
        assert result == "no-content-obj"


class TestFormatMultipleCitations:
    """AIMessage with several citations."""

    def test_all_citations_appear(self):
        """Multiple citations are each rendered."""
        msg = AIMessage(
            content="Multi-ref answer",
            response_metadata={
                "citation_metadata": {
                    "citations": [
                        {
                            "title": "A",
                            "uri": "https://a.com",
                        },
                        {
                            "title": "B",
                            "uri": "https://b.com",
                        },
                    ]
                }
            },
        )
        result = format_message_with_citations(msg)
        assert "- [A](https://a.com)" in result
        assert "- [B](https://b.com)" in result


# ------------------------------------------------------------------
# clear_state
# ------------------------------------------------------------------


class TestClearStateWithMessages:
    """clear_state when messages are present at top level."""

    def test_returns_remove_messages(self):
        """Each message produces a RemoveMessage."""
        msg1 = HumanMessage(content="hi", id="m1")
        msg2 = AIMessage(content="hello", id="m2")
        state = {"messages": [msg1, msg2]}

        result = clear_state(state)

        assert len(result["messages"]) == 2
        assert result["summary"] == ""
        for rm in result["messages"]:
            assert isinstance(rm, RemoveMessage)

    def test_remove_message_ids_match(self):
        """RemoveMessage ids match the original message ids."""
        msg = HumanMessage(content="hi", id="abc")
        state = {"messages": [msg]}
        result = clear_state(state)
        assert result["messages"][0].id == "abc"


class TestClearStateEmpty:
    """clear_state when messages list is empty."""

    def test_returns_empty_list(self):
        """Empty messages yields empty removal list."""
        state = {"messages": []}
        result = clear_state(state)
        assert result["messages"] == []
        assert result["summary"] == ""


# ------------------------------------------------------------------
# State class schema
# ------------------------------------------------------------------


class TestStateSchema:
    """Verify the State TypedDict has all expected fields."""

    def test_has_summary_field(self):
        """State annotations include summary."""
        assert "summary" in State.__annotations__

    def test_has_owner_field(self):
        """State annotations include owner."""
        assert "owner" in State.__annotations__

    def test_has_title_field(self):
        """State annotations include title."""
        assert "title" in State.__annotations__

    def test_has_tags_field(self):
        """State annotations include tags."""
        assert "tags" in State.__annotations__

    def test_has_previous_message_time(self):
        """State annotations include previous_message_time."""
        assert "previous_message_time" in State.__annotations__

    def test_has_current_message_time(self):
        """State annotations include current_message_time."""
        assert "current_message_time" in State.__annotations__

    def test_has_delta_time(self):
        """State annotations include delta_time."""
        assert "delta_time" in State.__annotations__

    def test_has_disable_summarization(self):
        """State annotations include disable_summarization."""
        assert "disable_summarization" in State.__annotations__

    def test_has_template_dict(self):
        """State annotations include template_dict."""
        assert "template_dict" in State.__annotations__

    def test_has_llm_config(self):
        """State annotations include llm_config."""
        assert "llm_config" in State.__annotations__


# ------------------------------------------------------------------
# UntrackedValue re-export
# ------------------------------------------------------------------


class TestUntrackedValueReExport:
    """Consumers declare ephemeral state fields against this module."""

    def test_is_the_langgraph_channel(self):
        """The re-export is LangGraph's channel, not a local stand-in."""
        assert UntrackedValue is langgraph.channels.UntrackedValue

    def test_listed_in_module_all(self):
        """__all__ carries UntrackedValue.

        The import is unused inside the module, so dropping it from
        __all__ lets autoflake delete it and breaks every consumer.
        """
        assert "UntrackedValue" in langgraph_utils.__all__


# ------------------------------------------------------------------
# Persisted vs. ephemeral State fields
# ------------------------------------------------------------------


class _ExtendedState(State):
    """A downstream State subclass declaring one field of each kind."""

    persisted: Optional[dict]
    ephemeral: Annotated[Optional[dict], UntrackedValue]


def _build_two_turn_graph(saver, first_turn_writes, observations):
    """Compile a graph that writes `first_turn_writes` on its first turn only.

    The second node records whether each field reached it, which is what
    distinguishes "crosses node boundaries" from "is checkpointed".
    """
    pending = [first_turn_writes]

    def emit(_state):
        """Return the queued write on the first turn, nothing after."""
        return pending.pop(0) if pending else {}

    def observe(state):
        """Record which of the two fields is visible mid-turn."""
        observations.append(
            {
                "persisted": state.get("persisted"),
                "ephemeral": state.get("ephemeral"),
            }
        )
        return {}

    graph = StateGraph(_ExtendedState)
    graph.add_node("emit", emit)
    graph.add_node("observe", observe)
    graph.add_edge(START, "emit")
    graph.add_edge("emit", "observe")
    graph.add_edge("observe", END)
    return graph.compile(checkpointer=saver)


class TestStateFieldPersistence:
    """Pins the four behaviours the State docstring documents.

    Documentation asserts these as facts about the pinned LangGraph, so a
    LangGraph upgrade that changes any of them has to make the docs fail
    rather than quietly go stale.
    """

    WRITES = {
        "persisted": {"turn": "first"},
        "ephemeral": {"turn": "first"},
    }
    CONFIG = {"configurable": {"thread_id": "state-kinds"}}

    def _run_two_turns(self):
        """Invoke the graph twice on one thread; return saver and results."""
        saver = MemorySaver()
        observations = []
        agent = _build_two_turn_graph(saver, dict(self.WRITES), observations)
        first = agent.invoke({"messages": []}, self.CONFIG)
        second = agent.invoke({"messages": []}, self.CONFIG)
        return saver, observations, first, second

    def test_both_kinds_cross_node_boundaries_and_reach_the_result(self):
        """Within a turn, both kinds reach later nodes and the result."""
        _, observations, first, _ = self._run_two_turns()
        assert observations[0]["persisted"] == {"turn": "first"}
        assert observations[0]["ephemeral"] == {"turn": "first"}
        assert first["persisted"] == {"turn": "first"}
        assert first["ephemeral"] == {"turn": "first"}

    def test_only_the_plain_field_is_checkpointed(self):
        """channel_values carries the plain field and not the untracked one."""
        saver, _, _, _ = self._run_two_turns()
        channel_values = saver.get(self.CONFIG)["channel_values"]
        assert channel_values["persisted"] == {"turn": "first"}
        assert "ephemeral" not in channel_values

    def test_plain_field_survives_a_turn_that_does_not_write_it(self):
        """A persisted channel retains the prior turn's value."""
        _, observations, _, second = self._run_two_turns()
        assert second["persisted"] == {"turn": "first"}
        assert observations[1]["persisted"] == {"turn": "first"}

    def test_untracked_field_does_not_carry_into_the_next_turn(self):
        """An untracked channel starts the next turn empty."""
        _, observations, _, second = self._run_two_turns()
        assert second.get("ephemeral") is None
        assert observations[1]["ephemeral"] is None
