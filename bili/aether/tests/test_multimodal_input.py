"""Tests for image-bearing messages travelling an AETHER run.

AETHER's message flattener exists to keep list-shaped content (which some
providers return for ordinary text) from breaking a downstream provider that
rejects an unrecognised parts structure.  It did that by joining
``part.get("text", str(part))`` over every part, which for an image part has no
``"text"`` key: the image was stringified into the prompt and effectively
dropped, silently, because the join always succeeds.

The flattener is now image-preserving, and these tests pin both halves: an
image survives, and a text-only parts list still collapses exactly as before,
which is the safety the coercion was written for.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Imported for its import side effect: the native tool-calling test below
# stubs ``langchain.agents`` in sys.modules, and the node generator imports
# from this module inside that patch.  Importing it here caches the real
# module first, so the stub cannot break an unrelated import.
import bili.iris.nodes.react_agent_node  # noqa: F401  pylint: disable=unused-import
from bili.aether.compiler.agent_generator import (
    _normalise_content_value,
    _normalise_message_content,
    generate_agent_node,
)
from bili.aether.runtime.cli import _build_input_data, _build_parser
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.iris.multimodal import MultimodalContentError

_MOCK_CREATE = "bili.aether.compiler.llm_resolver.create_llm"
_MOCK_TOOLS = "bili.aether.compiler.llm_resolver.resolve_tools"

IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://x.invalid/i.png"}}
TEXT_PART = {"type": "text", "text": "What is in this picture?"}
MULTIMODAL = [TEXT_PART, IMAGE_PART]
TEXT_ONLY_PARTS = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    defaults = {"role": "test_role", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


# ---------------------------------------------------------------------------
# The flattener
# ---------------------------------------------------------------------------


class TestNormaliseMessageContent:
    """The one predicate that decides whether content is coerced."""

    def test_text_only_parts_still_collapse_to_a_string(self):
        """The behaviour the coercion was written for, unchanged."""
        result = _normalise_message_content(AIMessage(content=TEXT_ONLY_PARTS))
        assert result.content == "a b"

    def test_string_content_is_returned_unchanged(self):
        """String content is returned unchanged."""
        message = AIMessage(content="already a string")
        assert _normalise_message_content(message) is message

    def test_an_image_bearing_message_is_returned_unchanged(self):
        """An image bearing message is returned unchanged."""
        message = HumanMessage(content=MULTIMODAL)
        assert _normalise_message_content(message) is message
        assert message.content == MULTIMODAL

    def test_the_image_url_is_not_stringified_into_the_text(self):
        """The pre-fix failure mode, stated as the artefact it produced."""
        result = _normalise_message_content(HumanMessage(content=MULTIMODAL))
        assert not isinstance(result.content, str)

    def test_a_message_without_content_is_returned_unchanged(self):
        """A message without content is returned unchanged."""
        sentinel = object()
        assert _normalise_message_content(sentinel) is sentinel

    def test_an_unrecognised_part_still_takes_the_coercion_path(self):
        """Only *recognised* non-text parts are exempt; an unknown part is not
        claimed to be multimodal, so history keeps its provider-safe form."""
        content = [{"type": "some_future_part", "value": 1}]
        result = _normalise_message_content(AIMessage(content=content))
        assert isinstance(result.content, str)

    def test_normalise_content_value_is_unchanged(self):
        """The response-side helper is deliberately untouched: an agent's
        OUTPUT is still reduced to text."""
        assert _normalise_content_value(TEXT_ONLY_PARTS) == "a b"
        assert _normalise_content_value("plain") == "plain"
        assert _normalise_content_value(None) == ""


# ---------------------------------------------------------------------------
# End to end through an agent node
# ---------------------------------------------------------------------------


class TestAgentNodeCarriesImages:
    """Both node paths hand the image to the model."""

    @staticmethod
    def _run(agent, state_messages):
        with patch(_MOCK_CREATE) as mock_create, patch(_MOCK_TOOLS, return_value=[]):
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="a chart")
            mock_create.return_value = llm
            node_fn = generate_agent_node(agent)
            node_fn({"messages": state_messages, "agent_outputs": {}})
            return llm.invoke.call_args[0][0]

    def test_direct_llm_path_carries_the_image(self):
        """The tool-less path: no tools configured, llm.invoke() called directly."""
        sent = self._run(
            _agent("vision_agent", model_name="gpt-4o"),
            [HumanMessage(content=MULTIMODAL)],
        )
        assert any(m.content == MULTIMODAL for m in sent)

    def test_direct_llm_path_still_collapses_text_only_parts(self):
        """Direct llm path still collapses text only parts."""
        sent = self._run(
            _agent("text_agent", model_name="gpt-4o"),
            [AIMessage(content=TEXT_ONLY_PARTS)],
        )
        assert any(m.content == "a b" for m in sent)

    def test_direct_llm_path_preserves_surrounding_history(self):
        """Direct llm path preserves surrounding history."""
        sent = self._run(
            _agent("vision_agent", model_name="gpt-4o"),
            [
                HumanMessage(content=MULTIMODAL),
                AIMessage(content="a prior answer"),
            ],
        )
        contents = [m.content for m in sent]
        assert MULTIMODAL in contents
        assert "a prior answer" in contents

    def test_tool_calling_path_carries_the_image(self):
        """The native tool-calling path applies the same flattener to history."""
        agent = _agent("vision_tools", model_name="gpt-4o", tools=["some_tool"])

        compiled = MagicMock()
        compiled.invoke.return_value = {"messages": [AIMessage(content="done")]}
        agents_stub = types.ModuleType("langchain.agents")
        agents_stub.create_agent = MagicMock(return_value=compiled)
        langchain_stub = types.ModuleType("langchain")
        langchain_stub.agents = agents_stub

        with (
            patch(_MOCK_CREATE) as mock_create,
            patch(_MOCK_TOOLS, return_value=[MagicMock()]),
            patch.dict(
                sys.modules,
                {"langchain": langchain_stub, "langchain.agents": agents_stub},
            ),
        ):
            mock_create.return_value = MagicMock()
            node_fn = generate_agent_node(agent)
            node_fn(
                {"messages": [HumanMessage(content=MULTIMODAL)], "agent_outputs": {}}
            )

        sent = compiled.invoke.call_args[0][0]["messages"]
        assert any(m.content == MULTIMODAL for m in sent)

    def test_a_system_message_is_still_injected_alongside_an_image(self):
        """A system message is still injected alongside an image."""
        sent = self._run(
            _agent("vision_agent", model_name="gpt-4o", system_prompt="Be helpful."),
            [HumanMessage(content=MULTIMODAL)],
        )
        assert isinstance(sent[0], SystemMessage)
        assert sent[0].content == "Be helpful."


# ---------------------------------------------------------------------------
# The runtime CLI affordance
# ---------------------------------------------------------------------------


class TestRuntimeCliInputImage:
    """``--input-image`` builds the multimodal turn for a CLI-driven run."""

    @staticmethod
    def _parse(argv):
        return _build_parser().parse_args(argv)

    def test_text_only_invocation_is_unchanged(self):
        """The backwards-compatibility claim for the CLI entry point."""
        data = _build_input_data(self._parse(["c.yaml", "--input", "hello"]))
        assert data["messages"][0].content == "hello"

    def test_no_input_still_returns_an_empty_dict(self):
        """No input still returns an empty dict."""
        assert _build_input_data(self._parse(["c.yaml"])) == {}

    def test_a_url_becomes_an_image_part(self):
        """A URL becomes an image part."""
        data = _build_input_data(
            self._parse(
                [
                    "c.yaml",
                    "--input",
                    "what is this",
                    "--input-image",
                    "https://x/i.png",
                ]
            )
        )
        assert data["messages"][0].content == [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
        ]

    def test_a_data_uri_is_passed_through(self):
        """A data URI is passed through."""
        uri = "data:image/png;base64,QUJD"
        data = _build_input_data(
            self._parse(["c.yaml", "--input", "x", "--input-image", uri])
        )
        assert data["messages"][0].content[1]["image_url"]["url"] == uri

    def test_a_local_path_is_inlined(self, tmp_path):
        """A local path is inlined."""
        image = tmp_path / "shot.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = _build_input_data(
            self._parse(["c.yaml", "--input", "x", "--input-image", str(image)])
        )
        url = data["messages"][0].content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_the_flag_is_repeatable(self):
        """The flag is repeatable."""
        data = _build_input_data(
            self._parse(
                [
                    "c.yaml",
                    "--input",
                    "x",
                    "--input-image",
                    "https://x/a.png",
                    "--input-image",
                    "https://x/b.png",
                ]
            )
        )
        assert len(data["messages"][0].content) == 3

    def test_an_image_with_no_text_still_builds_a_turn(self):
        """An image with no text still builds a turn."""
        data = _build_input_data(
            self._parse(["c.yaml", "--input-image", "https://x/a.png"])
        )
        assert len(data["messages"][0].content) == 1

    def test_input_file_still_works_with_an_image(self, tmp_path):
        """Input file still works with an image."""
        text_file = tmp_path / "prompt.txt"
        text_file.write_text("from a file")
        data = _build_input_data(
            self._parse(
                [
                    "c.yaml",
                    "--input-file",
                    str(text_file),
                    "--input-image",
                    "https://x/a.png",
                ]
            )
        )
        assert data["messages"][0].content[0] == {
            "type": "text",
            "text": "from a file",
        }


# ---------------------------------------------------------------------------
# The executor's human-input seams
# ---------------------------------------------------------------------------


class TestExecutorAcceptsContentParts:  # pylint: disable=protected-access
    """HITL resume and operator steering both take content parts.

    Reaches into ``_compiled_graph`` deliberately: the property under test is
    what these two seams WRITE into graph state, which is otherwise only
    observable by running a real compiled graph against a real model.
    """

    @staticmethod
    def _executor():
        config = MASConfig(
            mas_id="test_mas",
            name="Test MAS",
            objective="test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[_agent("a")],
        )
        executor = MASExecutor(config)
        executor._compiled_graph = MagicMock()  # pylint: disable=protected-access
        return executor

    def test_resume_streaming_accepts_a_parts_list(self):
        """Resume streaming accepts a parts list."""
        executor = self._executor()
        executor._compiled_graph.stream.return_value = iter([])
        list(executor.resume_streaming(MULTIMODAL, thread_id="t1"))
        update = executor._compiled_graph.update_state.call_args[0][1]
        assert update["messages"][0].content == MULTIMODAL

    def test_resume_streaming_string_path_is_unchanged(self):
        """Resume streaming string path is unchanged."""
        executor = self._executor()
        executor._compiled_graph.stream.return_value = iter([])
        list(executor.resume_streaming("approved", thread_id="t1"))
        update = executor._compiled_graph.update_state.call_args[0][1]
        assert update["messages"][0].content == "approved"

    def test_steer_accepts_a_parts_list(self):
        """An operator directive may be a list of content parts."""
        executor = self._executor()
        executor._compiled_graph.stream.return_value = iter([])
        list(executor.steer(MULTIMODAL, thread_id="t1"))
        update = executor._compiled_graph.update_state.call_args[0][1]
        assert update["messages"][0].content == MULTIMODAL

    def test_apply_steer_directives_rejects_an_unsupported_type(self):
        """Apply steer directives rejects an unsupported type."""
        executor = self._executor()
        with pytest.raises(MultimodalContentError):
            executor._apply_steer_directives(  # pylint: disable=protected-access
                {"configurable": {"thread_id": "t1"}}, [42]
            )
