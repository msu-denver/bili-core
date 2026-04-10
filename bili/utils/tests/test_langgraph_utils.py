"""Tests for bili.utils.langgraph_utils.

Covers format_message_with_citations, clear_state, and the State
TypedDict schema.
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from bili.utils.langgraph_utils import State, clear_state, format_message_with_citations

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
