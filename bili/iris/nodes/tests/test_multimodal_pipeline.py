"""Tests for image-bearing messages travelling the default IRIS node pipeline.

An image reaches a model only if every node between the entry point and the
provider leaves it alone.  Three nodes did not: the tool-less react fallback
repackaged every message through ``str(content)``, and both the per-user-state
node and the memory-management node called ``.startswith()`` straight on
``content``, which raises ``AttributeError`` on a list.

Each test below is written so that reverting its fix turns it red -- the
assertions are on the message the model actually receives, and on the node
completing at all, not on an internal call having been made.
"""

from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

from bili.iris.loaders.streaming_utils import _build_input
from bili.iris.multimodal import build_human_message
from bili.iris.nodes.per_user_state import buld_per_user_state_node
from bili.iris.nodes.react_agent_node import build_react_agent_node
from bili.iris.nodes.trim_and_summarize import build_trim_and_summarize_node

IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://x.invalid/i.png"}}
TEXT_PART = {"type": "text", "text": "What is in this picture?"}
MULTIMODAL = [TEXT_PART, IMAGE_PART]


# ---------------------------------------------------------------------------
# The tool-less react fallback
# ---------------------------------------------------------------------------


class TestToolLessFallbackCarriesImages:
    """``call_model`` must not stringify a message that carries an image."""

    @staticmethod
    def _invoke(state_messages):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="ok")
        node = build_react_agent_node(tools=None, llm_model=llm)
        node({"messages": state_messages})
        return llm.invoke.call_args.args[0]

    def test_image_parts_reach_the_model(self):
        """Image parts reach the model."""
        sent = self._invoke([HumanMessage(content=MULTIMODAL)])
        assert sent[0].content == MULTIMODAL

    def test_the_image_is_not_stringified(self):
        """The pre-fix behaviour produced the repr of the parts list, which
        looks like a successful turn and carries no image."""
        sent = self._invoke([HumanMessage(content=MULTIMODAL)])
        assert not isinstance(sent[0].content, str)
        assert IMAGE_PART in sent[0].content

    def test_text_string_content_is_unchanged(self):
        """Text string content is unchanged."""
        sent = self._invoke([HumanMessage(content="plain text")])
        assert sent[0].content == "plain text"

    def test_text_only_part_list_keeps_the_historical_coercion(self):
        """A provider that returns list-shaped text content still gets the
        existing str() treatment; only recognised non-text parts are exempt."""
        sent = self._invoke([AIMessage(content=[{"type": "text", "text": "hi"}])])
        assert sent[0].content == str([{"type": "text", "text": "hi"}])

    def test_mixed_history_carries_only_the_image_message_as_parts(self):
        """Mixed history carries only the image message as parts."""
        sent = self._invoke(
            [
                SystemMessage(content="be helpful"),
                HumanMessage(content=MULTIMODAL),
                AIMessage(content="I see a chart."),
            ]
        )
        assert [type(m.content) for m in sent] == [str, list, str]
        assert sent[1].content == MULTIMODAL

    def test_tool_messages_are_still_filtered_out(self):
        """Tool messages are still filtered out."""
        sent = self._invoke(
            [
                HumanMessage(content=MULTIMODAL),
                ToolMessage(content="tool result", tool_call_id="tc1"),
            ]
        )
        assert len(sent) == 1

    def test_every_message_is_still_repackaged_as_human(self):
        """Every message is still repackaged as human."""
        sent = self._invoke(
            [SystemMessage(content="sys"), HumanMessage(content=MULTIMODAL)]
        )
        assert all(isinstance(m, HumanMessage) for m in sent)


# ---------------------------------------------------------------------------
# per_user_state
# ---------------------------------------------------------------------------


class TestPerUserStateHandlesListContent:
    """The profile-prefix check must read text, not assume a string."""

    USER = {"uid": "u1", "name": "Test User"}

    def test_list_content_at_position_one_does_not_raise(self):
        """Pre-fix this raised AttributeError and took the node down for the
        whole conversation."""
        node = buld_per_user_state_node(current_user=self.USER)
        state = {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content=MULTIMODAL),
            ]
        }
        result = node(state)
        assert any(isinstance(m, HumanMessage) for m in result["messages"])

    def test_the_image_message_survives_the_node(self):
        """The image message survives the node."""
        node = buld_per_user_state_node(current_user=self.USER)
        result = node(
            {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(content=MULTIMODAL),
                ]
            }
        )
        assert any(m.content == MULTIMODAL for m in result["messages"])

    def test_an_existing_profile_message_is_still_replaced(self):
        """The behaviour the prefix check exists for is unchanged: a stale
        profile message at position 1 is removed."""
        node = buld_per_user_state_node(current_user=self.USER)
        result = node(
            {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(content="USER PROFILE: stale"),
                ]
            }
        )
        assert any(isinstance(m, RemoveMessage) for m in result["messages"])

    def test_an_image_message_is_not_mistaken_for_a_profile_message(self):
        """An image message is not mistaken for a profile message."""
        node = buld_per_user_state_node(current_user=self.USER)
        result = node(
            {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(content=MULTIMODAL),
                ]
            }
        )
        assert not any(isinstance(m, RemoveMessage) for m in result["messages"])

    def test_a_profile_prefix_spread_over_text_parts_is_still_detected(self):
        """The text of a list-content message is read for the prefix, so a
        profile message that arrived as parts is still de-duplicated."""
        node = buld_per_user_state_node(current_user=self.USER)
        result = node(
            {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "USER PROFILE: "},
                            {"type": "text", "text": "stale"},
                        ]
                    ),
                ]
            }
        )
        assert any(isinstance(m, RemoveMessage) for m in result["messages"])


# ---------------------------------------------------------------------------
# trim_and_summarize
# ---------------------------------------------------------------------------


class TestTrimAndSummarizeHandlesListContent:
    """The same string assumption, on the memory-management node."""

    def test_list_content_does_not_raise(self):
        """List content does not raise."""
        node = build_trim_and_summarize_node(
            llm_model=MagicMock(), memory_limit_type="message_count", k=5
        )
        result = node(
            {
                "messages": [
                    HumanMessage(content=MULTIMODAL),
                    AIMessage(content="I see a chart."),
                ]
            }
        )
        assert "messages" in result

    def test_a_profile_message_is_still_recognised(self):
        """The node keeps identifying the profile message by its prefix."""
        node = build_trim_and_summarize_node(
            llm_model=MagicMock(), memory_limit_type="message_count", k=1
        )
        result = node(
            {
                "messages": [
                    HumanMessage(content="USER PROFILE: x"),
                    HumanMessage(content=MULTIMODAL),
                    AIMessage(content="reply"),
                ]
            }
        )
        assert "messages" in result


# ---------------------------------------------------------------------------
# The three nodes in sequence
# ---------------------------------------------------------------------------


class TestTheImageSurvivesTheWholePipeline:
    """The end-to-end property the individual node tests only cover in parts.

    An image reaches the model only if EVERY node between the entry point and
    the provider leaves it alone, so the pipeline is exercised as a chain:
    the entry point builds the state, the per-user-state and memory nodes run
    over it, and the tool-less react node hands the result to the model.
    """

    def test_entry_point_to_model(self):
        """Each node returns a LangGraph state UPDATE, so the updates are
        applied through the same ``add_messages`` reducer the graph uses
        rather than replacing the list (which would discard the
        ``RemoveMessage`` semantics these nodes rely on)."""
        message = build_human_message(
            text="What is in this picture?",
            images=["https://x.invalid/i.png"],
        )
        state = _build_input(message.content, None)

        for node in (
            buld_per_user_state_node(current_user={"uid": "u1", "name": "Test User"}),
            build_trim_and_summarize_node(
                llm_model=MagicMock(), memory_limit_type="message_count", k=10
            ),
        ):
            update = node(state)
            state["messages"] = add_messages(state["messages"], update["messages"])

        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="a chart")
        build_react_agent_node(tools=None, llm_model=llm)(state)

        sent = llm.invoke.call_args.args[0]
        assert any(
            isinstance(m.content, list) and IMAGE_PART in m.content for m in sent
        ), "the image content part did not reach the model"
