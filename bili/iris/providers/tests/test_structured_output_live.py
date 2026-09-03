"""Live structured-output test against a local Ollama daemon.

Skipped by default: requires a running Ollama daemon with a pulled model and
is therefore not suitable for CI.  Enable it locally with::

    BILI_LIVE_OLLAMA_TESTS=1 pytest bili/iris/providers/tests/test_structured_output_live.py

Environment knobs:

- ``BILI_LIVE_OLLAMA_TESTS`` — set to ``1`` to run.
- ``BILI_LIVE_OLLAMA_MODEL`` — model tag to use (default ``qwen3:8b``).
- ``OLLAMA_BASE_URL`` — daemon endpoint (default ``http://localhost:11434``).

This is the proof the capability exists for: a small local model asked for a
nested document (an array of objects, the shape models routinely get wrong as
a bare string) must produce schema-valid JSON on the FIRST attempt, because
decoding is grammar-constrained.
"""

import importlib.util
import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BILI_LIVE_OLLAMA_TESTS") != "1",
        reason="live Ollama test; set BILI_LIVE_OLLAMA_TESTS=1 to run",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("langchain_ollama") is None,
        reason="requires the [ollama] extra (langchain-ollama)",
    ),
]

NESTED_SCHEMA = {
    "title": "Outline Document",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["heading", "bullets"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "sections"],
    "additionalProperties": False,
}


def test_constrained_generation_is_schema_valid_first_try():
    """Verify a live local model produces schema-valid output when constrained."""
    from bili.iris.loaders.llm_loader import load_model
    from bili.iris.providers.structured_output import parse_structured_content

    llm = load_model(
        "local_ollama",
        model_name=os.environ.get("BILI_LIVE_OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.environ.get("OLLAMA_BASE_URL"),
        temperature=0.2,
        max_tokens=2048,
        structured_output_schema=NESTED_SCHEMA,
    )
    response = llm.invoke(
        "Produce a short two-section outline about renewable energy. "
        "Respond only with the structured object."
    )

    # parse_structured_content raises on parse or validation failure; a clean
    # return IS the assertion that generation was schema-constrained.
    document = parse_structured_content(response.content, schema=NESTED_SCHEMA)
    assert len(document["sections"]) >= 2
    assert all(len(s["bullets"]) >= 2 for s in document["sections"])
