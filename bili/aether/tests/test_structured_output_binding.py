"""Tests for AETHER's decode-time structured-output binding.

An ``AgentSpec`` with ``output_format="structured"`` and an ``output_schema``
previously carried the schema as declaration only: nothing constrained the
model and ``_build_output`` never parsed or validated the content.  These
tests pin the end-to-end enforcement added on top of the provider-level
``structured_output_schema`` capability:

- ``create_llm`` threads ``output_schema`` into ``load_model`` as
  ``structured_output_schema`` when the resolved provider supports
  decode-time enforcement and the agent has no tools.
- Tool-bearing agents and unsupported providers degrade gracefully (schema
  not bound, warning logged, agent still runs).
- Fallback chains evaluate support per fallback provider.
- ``_build_output`` parses and validates structured content, populating
  ``output["parsed"]`` (which consensus vote extraction reads) on success
  and ``output["raw"]`` + ``output["schema_error"]`` on failure.
"""

# pylint: disable=too-few-public-methods,duplicate-code

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from bili.aether.compiler.agent_generator import _build_output
from bili.aether.compiler.llm_resolver import create_llm
from bili.aether.schema import AgentSpec

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["label", "score"],
            },
        },
    },
    "required": ["verdict", "findings"],
}


def _structured_agent(**overrides) -> AgentSpec:
    """Build a minimal structured-output AgentSpec for these tests."""
    spec = {
        "agent_id": "structured-writer",
        "role": "writer",
        "objective": "Produce a schema-valid document.",
        "model_name": "ollama:qwen3:8b",
        "output_format": "structured",
        "output_schema": SCHEMA,
    }
    spec.update(overrides)
    return AgentSpec(**spec)


@contextmanager
def _mock_module(module_name: str, **attrs):
    """Temporarily inject a fake module into sys.modules for lazy-import tests."""
    mod = ModuleType(module_name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    already_present = sys.modules.get(module_name)
    sys.modules[module_name] = mod
    try:
        yield mod
    finally:
        if already_present is not None:
            sys.modules[module_name] = already_present
        else:
            sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# create_llm — schema threading
# ---------------------------------------------------------------------------


class TestCreateLLMStructuredBinding:
    """Verify create_llm binds output_schema for decode-time enforcement."""

    def test_structured_agent_binds_schema_end_to_end(self):
        """Verify the full chain: AgentSpec -> load_model -> ChatOllama format."""
        mock_cls = MagicMock()
        with _mock_module("langchain_ollama", ChatOllama=mock_cls):
            create_llm(_structured_agent())
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "qwen3:8b"
        # Equality, not identity: Pydantic deep-copies dict fields on
        # AgentSpec construction.
        assert kwargs["format"] == SCHEMA

    def test_text_agent_does_not_bind_schema(self):
        """Verify a text-format agent never receives the kwarg."""
        agent = AgentSpec(
            agent_id="plain",
            role="writer",
            objective="Write text.",
            model_name="ollama:qwen3:8b",
        )
        with patch("bili.iris.loaders.llm_loader.load_model") as mock_load:
            create_llm(agent)
        assert "structured_output_schema" not in mock_load.call_args[1]

    def test_json_agent_does_not_bind_schema(self):
        """Verify output_format='json' (schema-less) never binds."""
        agent = AgentSpec(
            agent_id="jsonish",
            role="writer",
            objective="Write JSON.",
            model_name="ollama:qwen3:8b",
            output_format="json",
        )
        with patch("bili.iris.loaders.llm_loader.load_model") as mock_load:
            create_llm(agent)
        assert "structured_output_schema" not in mock_load.call_args[1]

    def test_tool_bearing_agent_skips_binding(self, caplog):
        """Verify agents with tools degrade to post-hoc validation."""
        agent = _structured_agent(tools=["weather_api_tool"])
        with patch("bili.iris.loaders.llm_loader.load_model") as mock_load:
            with caplog.at_level("WARNING"):
                create_llm(agent)
        assert "structured_output_schema" not in mock_load.call_args[1]
        assert "post-hoc" in caplog.text

    def test_unsupported_provider_skips_binding(self, caplog):
        """Verify a provider without decode-time enforcement degrades."""
        agent = _structured_agent(model_name="cli:some-cli-tool")
        with patch("bili.iris.loaders.llm_loader.load_model") as mock_load:
            with caplog.at_level("WARNING"):
                create_llm(agent)
        assert "structured_output_schema" not in mock_load.call_args[1]
        assert "no decode-time structured-output" in caplog.text

    def test_fallback_chain_evaluates_support_per_provider(self):
        """Verify mixed chains bind for supported entries only.

        Primary (Ollama) supports decode-time enforcement; the CLI fallback
        does not.  The unsupported fallback must not receive the kwarg, or
        load_model's fail-fast gate would reject it at failover time.
        """
        agent = _structured_agent(fallback_models=["cli:some-cli-tool"])
        captured = {}

        def _capture_fallback(primary_llm, fallback_chain):  # noqa: ANN001
            captured["chain"] = fallback_chain
            return primary_llm

        with patch("bili.iris.loaders.llm_loader.load_model") as mock_load:
            with patch(
                "bili.iris.providers.fallback.build_fallback_llm",
                side_effect=_capture_fallback,
            ):
                create_llm(agent)

        primary_kwargs = mock_load.call_args[1]
        assert primary_kwargs["structured_output_schema"] == SCHEMA

        (fb_provider, fb_kwargs) = captured["chain"][0]
        assert fb_provider == "cli"
        assert "structured_output_schema" not in fb_kwargs


# ---------------------------------------------------------------------------
# _build_output — parse + validate
# ---------------------------------------------------------------------------


class TestBuildOutputStructured:
    """Verify _build_output enforces structured output post-hoc."""

    VALID = '{"verdict": "pass", "findings": [{"label": "a", "score": 0.9}]}'

    def test_valid_structured_content_is_parsed(self):
        """Verify schema-valid content lands in output['parsed']."""
        output = _build_output(_structured_agent(), self.VALID)
        assert output["parsed"]["verdict"] == "pass"
        assert "raw" not in output
        assert "schema_error" not in output

    def test_parsed_feeds_consensus_vote_extraction(self):
        """Verify the parsed dict exposes vote fields for consensus workflows.

        Consensus vote extraction reads ``output["parsed"][vote_field]``;
        before this change structured agents only ever populated ``raw``.
        """
        output = _build_output(
            _structured_agent(consensus_vote_field="verdict"), self.VALID
        )
        assert output["parsed"]["verdict"] == "pass"

    def test_fenced_structured_content_is_parsed(self):
        """Verify fenced JSON (unconstrained-provider mode) still parses."""
        output = _build_output(_structured_agent(), f"```json\n{self.VALID}\n```")
        assert output["parsed"]["verdict"] == "pass"

    def test_invalid_json_sets_raw_and_schema_error(self):
        """Verify unparseable content degrades with a schema_error."""
        content = "verdict: pass\n---\nnot json"
        output = _build_output(_structured_agent(), content)
        assert "parsed" not in output
        assert output["raw"] == content
        assert "not valid JSON" in output["schema_error"]

    def test_schema_violation_sets_raw_and_schema_error(self):
        """Verify schema-invalid JSON degrades with a schema_error.

        A string where the schema requires a list of objects: the exact
        shape of the motivating failure class.
        """
        content = '{"verdict": "pass", "findings": "just a string"}'
        output = _build_output(_structured_agent(), content)
        assert "parsed" not in output
        assert output["raw"] == content
        assert "conform" in output["schema_error"]

    def test_json_format_behaviour_unchanged(self):
        """Verify output_format='json' keeps its schema-less best-effort parse."""
        agent = AgentSpec(
            agent_id="jsonish",
            role="writer",
            objective="Write JSON.",
            model_name="ollama:qwen3:8b",
            output_format="json",
        )
        output = _build_output(agent, '{"free": "form"}')
        assert output["parsed"] == {"free": "form"}
        assert "schema_error" not in output

    def test_text_format_behaviour_unchanged(self):
        """Verify output_format='text' still passes content through as raw."""
        agent = AgentSpec(
            agent_id="plain",
            role="writer",
            objective="Write text.",
            model_name="ollama:qwen3:8b",
        )
        output = _build_output(agent, "hello")
        assert output["raw"] == "hello"
        assert "parsed" not in output


# ---------------------------------------------------------------------------
# AgentSpec validation (pre-existing contract this feature relies on)
# ---------------------------------------------------------------------------


class TestAgentSpecStructuredContract:
    """Verify the schema-presence contract create_llm relies on."""

    def test_structured_without_schema_rejected(self):
        """Verify AgentSpec enforces output_schema for structured format."""
        with pytest.raises(ValueError, match="output_schema"):
            AgentSpec(
                agent_id="broken",
                role="writer",
                objective="Produce a schema-valid document.",
                model_name="ollama:qwen3:8b",
                output_format="structured",
            )
