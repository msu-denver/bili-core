"""Schema-constrained structured output for bili-core LLM providers.

This module is the single seam through which a caller binds a JSON schema to
a model so that generation is constrained to schema-valid output at decode
time.  It solves a whole failure class: a model asked to produce a large,
strictly-validated nested document emits plausible-but-invalid output, gets a
validation error back, retries, and never converges.  With decode-time
constraint the model is only *allowed* to emit schema-conforming tokens, so
the first response is valid by construction.

How to request structured output
--------------------------------
Pass a JSON schema (a ``dict``, or a Pydantic ``BaseModel`` subclass) as the
``structured_output_schema`` kwarg to
:func:`bili.iris.loaders.llm_loader.load_model` or to a provider's
``load()``::

    from bili.iris.loaders.llm_loader import load_model
    from bili.iris.providers.structured_output import parse_structured_content

    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    llm = load_model("local_ollama", model_name="qwen3:8b",
                     structured_output_schema=schema)
    response = llm.invoke("Produce the document.")
    document = parse_structured_content(response.content, schema=schema)

The returned model object keeps bili-core's duck-typed contract: ``.invoke()``
still returns a message whose ``.content`` is a *string*.  The constraint
guarantees that string is schema-valid JSON; :func:`parse_structured_content`
turns it into a validated Python object.  This deliberately does NOT use
LangChain's ``with_structured_output()``, which changes the return type to a
dict/Pydantic object and would break every consumer of the ``.content``
contract (IRIS node pipelines, AETHER agent nodes, checkpointers, streaming).

Per-provider enforcement mechanisms
-----------------------------------
================================  =============================================
Provider type                      Mechanism
================================  =============================================
``local_ollama``                   ``ChatOllama(format=<schema>)`` — Ollama
                                   compiles the schema to a llama.cpp GBNF
                                   grammar; decoding is token-level constrained.
``remote_openai``                  ``.bind(response_format={"type":
                                   "json_schema", ...})`` — OpenAI structured
                                   outputs with ``strict: true``; server-side
                                   constrained decoding.
``remote_azure_openai``            Same ``response_format`` binding as OpenAI.
``remote_google_vertex``           ``response_schema`` + ``response_mime_type=
                                   "application/json"`` — Gemini controlled
                                   generation (constrained decoding).
``remote_google_genai``            Same ``response_schema`` mechanism via the
                                   Google AI Developer API.
================================  =============================================

Providers not listed have no decode-time schema enforcement wired yet:

- ``remote_anthropic`` — the pinned ``langchain-anthropic`` release exposes
  structured output only via forced tool-use (``with_structured_output``),
  which validates the schema server-side but changes the return type; a
  ``response_format``-style content constraint is not yet exposed.  Follow-up.
- ``remote_aws_bedrock`` — the Converse API has no response-format parameter;
  the only schema channel is tool input.  Follow-up (tool-use path).
- ``remote_mistral`` / ``remote_cohere`` / ``remote_groq`` / ``remote_xai`` /
  ``remote_deepseek`` — the underlying APIs expose JSON modes of varying
  strictness; not yet wired.  Follow-up.
- ``local_llamacpp`` — llama.cpp supports GBNF grammars directly
  (``ChatLlamaCpp(grammar=...)``) but not a JSON-schema parameter through the
  pinned LangChain wrapper; a schema→GBNF conversion is a follow-up.
- ``local_huggingface``, ``cli``, ``cli_*`` presets — free-form text
  transports with no decode-time constraint channel.

Requesting ``structured_output_schema`` for an unsupported provider raises
``ValueError`` at load time (fail-fast) rather than silently returning an
unconstrained model, because a caller that believes output is constrained
when it is not would ship the exact failure this module exists to prevent.

Structured output vs. tool calling
----------------------------------
Constrained generation and *native tool calling* are mutually exclusive on
this seam: a schema-constrained generation cannot also emit the provider's
tool-call format, and for the OpenAI-family providers the bound object is a
``RunnableBinding`` that does not expose ``bind_tools``.  Agents that must
both call tools and produce a large schema-valid document should produce the
document via a dedicated schema-constrained generation (this seam) and apply
any side effects deterministically with the validated object, rather than
passing the document as a free-form tool argument.  Tool *arguments* are only
decode-time constrained where the provider API does it natively (OpenAI
``strict`` function calling); notably Ollama applies ``format`` to assistant
content only, never to tool-call arguments.
"""

import copy
import json
import logging
import re
from typing import Any, List, Optional

LOGGER = logging.getLogger(__name__)

#: Fallback name for the OpenAI ``response_format`` payload when the schema
#: has no usable ``title``.  OpenAI requires a name matching
#: ``^[a-zA-Z0-9_-]{1,64}$``.
_DEFAULT_SCHEMA_NAME = "structured_output"

#: Characters OpenAI accepts in a ``json_schema.name``; everything else is
#: replaced with ``_`` by :func:`_sanitize_schema_name`.
_SCHEMA_NAME_INVALID_RE = re.compile(r"[^a-zA-Z0-9_-]")

#: Matches a fenced code block so :func:`parse_structured_content` can accept
#: output from providers running in degraded (unconstrained) mode, where
#: models commonly wrap JSON in markdown fences.
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)```", re.DOTALL)

#: Matches a ``<think>...</think>`` reasoning preamble so
#: :func:`parse_structured_content` can accept output from a reasoning model
#: running unconstrained, which emits its chain-of-thought in such a block
#: before the JSON.  Case-insensitive and DOTALL so a multi-line block in any
#: casing is stripped; only the first block is removed (the JSON payload does
#: not contain the tag).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Provider types with decode-time schema enforcement wired.  Mutable so that
# external provider authors can declare support at application startup via
# register_structured_output_provider(), mirroring the provider registry.
_STRUCTURED_OUTPUT_PROVIDERS: set = {
    "local_ollama",
    "remote_openai",
    "remote_azure_openai",
    "remote_google_vertex",
    "remote_google_genai",
}


class StructuredOutputError(ValueError):
    """Base error for structured-output parsing and validation failures."""


class StructuredOutputParseError(StructuredOutputError):
    """The model's content could not be parsed as JSON."""


class StructuredOutputValidationError(StructuredOutputError):
    """The parsed content does not conform to the bound JSON schema."""


def supports_structured_output(provider_type: str) -> bool:
    """Return whether *provider_type* has decode-time schema enforcement wired.

    :param provider_type: A provider-type string (e.g. ``"local_ollama"``).
    :returns: ``True`` when ``load_model(provider_type,
        structured_output_schema=...)`` will produce a schema-constrained
        model; ``False`` otherwise.
    """
    return provider_type in _STRUCTURED_OUTPUT_PROVIDERS


def structured_output_providers() -> frozenset:
    """Return the provider types with decode-time schema enforcement wired."""
    return frozenset(_STRUCTURED_OUTPUT_PROVIDERS)


def register_structured_output_provider(provider_type: str) -> None:
    """Declare that an externally-registered provider enforces schemas.

    External provider authors whose ``load()`` honours the
    ``structured_output_schema`` kwarg call this at application startup so
    that :func:`require_structured_output_support` (and therefore
    ``load_model``) accepts schema requests for their provider type.

    :param provider_type: The provider-type string registered via
        :func:`bili.iris.providers.register_provider`.
    """
    _STRUCTURED_OUTPUT_PROVIDERS.add(provider_type)
    LOGGER.debug("Structured-output support registered for '%s'", provider_type)


def require_structured_output_support(provider_type: str) -> None:
    """Raise ``ValueError`` when *provider_type* cannot enforce a schema.

    Called by ``load_model`` before dispatch so a schema request against an
    unsupported provider fails fast instead of silently returning an
    unconstrained model.

    :param provider_type: The provider-type string being loaded.
    :raises ValueError: When the provider has no decode-time enforcement.
    """
    if provider_type not in _STRUCTURED_OUTPUT_PROVIDERS:
        supported = ", ".join(sorted(_STRUCTURED_OUTPUT_PROVIDERS))
        raise ValueError(
            f"Provider '{provider_type}' does not support decode-time "
            f"structured output; structured_output_schema cannot be honoured. "
            f"Supported provider types: {supported}. "
            "External providers can declare support via "
            "register_structured_output_provider()."
        )


def normalize_schema(schema: Any) -> dict:
    """Coerce *schema* to a plain JSON-schema ``dict``.

    Accepts either a JSON schema dictionary (returned as-is) or a Pydantic
    ``BaseModel`` *class* (converted via ``model_json_schema()``).  Detection
    is duck-typed on ``model_json_schema`` so this module never imports
    Pydantic itself.

    :param schema: A ``dict`` JSON schema or a Pydantic model class.
    :returns: A JSON schema ``dict``.
    :raises TypeError: For any other input, including Pydantic *instances*
        (pass the class, not an instance).
    """
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    raise TypeError(
        "structured_output_schema must be a JSON schema dict or a Pydantic "
        f"BaseModel subclass; got {type(schema).__name__!r}."
    )


def _sanitize_schema_name(name: str) -> str:
    """Reduce *name* to the ``^[a-zA-Z0-9_-]{1,64}$`` form OpenAI requires."""
    sanitized = _SCHEMA_NAME_INVALID_RE.sub("_", name)[:64]
    return sanitized or _DEFAULT_SCHEMA_NAME


def adapt_schema_for_openai_strict(schema: dict) -> dict:
    """Adapt a JSON schema to OpenAI's strict structured-output subset.

    OpenAI ``strict`` mode requires every object to declare
    ``additionalProperties: false`` and to list *every* property in
    ``required``.  A schema with optional properties (the ordinary Pydantic
    shape) is otherwise rejected.  This recursively, on every object node
    (including inside ``$defs``):

    - sets ``required`` to all property names, and
    - makes each previously-optional property nullable
      (``{"anyOf": [<original>, {"type": "null"}]}``) so it stays semantically
      optional (the model emits ``null`` when it has nothing), and
    - sets ``additionalProperties: false``.

    ``$defs`` / ``$ref`` are preserved (OpenAI strict supports them).  The input
    is not mutated.

    :param schema: A normalized JSON schema ``dict``.
    :returns: A new schema conforming to the OpenAI strict subset.
    """

    def _adapt(node: Any) -> Any:
        if isinstance(node, list):
            return [_adapt(item) for item in node]
        if not isinstance(node, dict):
            return node
        adapted = {key: _adapt(value) for key, value in node.items()}
        properties = adapted.get("properties")
        if adapted.get("type") == "object" and isinstance(properties, dict):
            originally_required = set(adapted.get("required", []) or [])
            for prop_name, prop_schema in list(properties.items()):
                if prop_name not in originally_required:
                    # Keep the property optional in spirit by allowing null,
                    # while satisfying strict's "every property required" rule.
                    properties[prop_name] = {"anyOf": [prop_schema, {"type": "null"}]}
            adapted["required"] = list(properties.keys())
            adapted["additionalProperties"] = False
        return adapted

    return _adapt(copy.deepcopy(schema))


def adapt_schema_for_gemini(schema: dict) -> dict:
    """Adapt a JSON schema to Gemini's ``response_schema`` subset.

    Gemini rejects ``$defs`` / ``$ref`` and ``additionalProperties``; langchain
    strips them, which leaves dangling ``$ref``s (a broken schema).  This inlines
    every ``$ref`` from ``$defs`` (dereference), then drops ``$defs`` and
    ``additionalProperties``, producing a self-contained schema within the
    subset.  A recursive schema (a definition that references itself) cannot be
    inlined; on a cycle the offending reference is replaced with a permissive
    ``{"type": "object"}`` so the request still dispatches.  The input is not
    mutated.

    :param schema: A normalized JSON schema ``dict``.
    :returns: A new self-contained schema within the Gemini subset.
    """
    defs = schema.get("$defs", {}) or {}

    def _inline(node: Any, expanding: List[str]) -> Any:
        if isinstance(node, list):
            return [_inline(item, expanding) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_name = str(node["$ref"]).rsplit("/", maxsplit=1)[-1]
            if ref_name in expanding or ref_name not in defs:
                # Cycle, or a ref we cannot resolve: fall back to a permissive
                # object so the schema stays valid within the subset.
                return {"type": "object"}
            target = copy.deepcopy(defs[ref_name])
            # Sibling keys alongside a $ref override the target (JSON Schema 2020-12).
            for key, value in node.items():
                if key != "$ref":
                    target[key] = value
            return _inline(target, expanding + [ref_name])
        return {
            key: _inline(value, expanding)
            for key, value in node.items()
            if key not in ("$defs", "additionalProperties")
        }

    return _inline(copy.deepcopy(schema), [])


def openai_response_format(schema: dict, *, name: Optional[str] = None) -> dict:
    """Build the OpenAI ``response_format`` payload for a JSON schema.

    Produces the Chat Completions ``response_format`` structure that both
    ``ChatOpenAI`` and ``AzureChatOpenAI`` forward verbatim::

        {"type": "json_schema",
         "json_schema": {"name": ..., "schema": ..., "strict": True}}

    ``strict`` is always ``True`` because decode-time enforcement is the point of
    this seam.  The schema is adapted to OpenAI's strict subset first (see
    :func:`adapt_schema_for_openai_strict`), so a schema with optional properties
    is accepted rather than rejected.

    :param schema: A JSON schema ``dict`` (already normalized).
    :param name: Optional payload name; defaults to the schema's ``title``
        (sanitized) or ``"structured_output"``.
    :returns: The ``response_format`` dict to bind onto the model.
    """
    resolved_name = _sanitize_schema_name(name or schema.get("title", ""))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": resolved_name,
            "schema": adapt_schema_for_openai_strict(schema),
            "strict": True,
        },
    }


def gemini_response_schema(schema: Any) -> dict:
    """Normalize *schema* and adapt it to Gemini's ``response_schema`` subset.

    The Gemini-side complement of :func:`openai_response_format`: the one call
    the Vertex and Google GenAI providers use so schema adaptation lives in the
    seam rather than in each provider.

    :param schema: A JSON schema ``dict`` or Pydantic model class.
    :returns: A self-contained schema within the Gemini subset.
    """
    return adapt_schema_for_gemini(normalize_schema(schema))


def parse_structured_content(content: str, schema: Optional[dict] = None) -> Any:
    """Parse (and optionally validate) a model's structured-output content.

    The complement of the load-time binding: turns the schema-constrained
    JSON string in ``response.content`` into a Python object, validating it
    against *schema* when one is given.  Also tolerates a markdown code fence
    and a ``<think>...</think>`` reasoning preamble, so it can post-hoc
    validate output from providers running unconstrained (including reasoning
    models that emit chain-of-thought before the JSON).

    :param content: The raw ``.content`` string from the model response.
    :param schema: Optional JSON schema ``dict`` to validate against.
    :returns: The parsed Python object (typically a ``dict``).
    :raises StructuredOutputParseError: When no candidate parses as JSON.
    :raises StructuredOutputValidationError: When the parsed object does not
        conform to *schema*.
    """
    text = (content or "").strip()

    candidates = [text]

    # A reasoning model running unconstrained emits its chain-of-thought in a
    # <think>...</think> block before the JSON; strip it and offer the
    # remainder.  Additive: the raw text is tried first, so a payload that
    # somehow contained the tag still parses from candidates[0].
    if _THINK_BLOCK_RE.search(text):
        think_stripped = _THINK_BLOCK_RE.sub("", text, count=1).strip()
        if think_stripped and think_stripped != text:
            candidates.append(think_stripped)

    # Providers running unconstrained commonly wrap JSON in a markdown fence;
    # accept the fence contents from each candidate produced so far (the raw
    # text and, when present, the think-stripped text).
    for base in list(candidates):
        fence_match = _CODE_FENCE_RE.search(base)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

    parsed: Any = None
    parse_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            parse_error = None
            break
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = exc

    if parse_error is not None:
        raise StructuredOutputParseError(
            f"Structured output is not valid JSON: {parse_error}"
        ) from parse_error

    if schema is not None:
        # Lazy import keeps this module import-light on paths that never
        # validate (the binding helpers above need only the stdlib).
        import jsonschema  # pylint: disable=import-outside-toplevel

        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            raise StructuredOutputValidationError(
                f"Structured output does not conform to the schema: {exc.message}"
            ) from exc

    return parsed
