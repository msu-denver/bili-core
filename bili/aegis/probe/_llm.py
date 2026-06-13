"""LLM Protocol + fake/real adapters for PROBE.

PROBE's nodes and policies depend on the ``ProbeLLM`` Protocol rather than any
concrete provider. This lets unit tests use a deterministic in-process fake
(``_FakeLLM``) and production code use a real LangChain ChatModel (via
``resolve_real_llm``) through the same interface.

Token accounting is part of the Protocol: every ``invoke`` returns the
response text plus its ``(tokens_in, tokens_out)`` cost so the runner can
record per-turn consumption against the BudgetState without inspecting
provider-specific response objects.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage

LOGGER = logging.getLogger(__name__)


# Single-method Protocol is the entire type contract; adding another
# method would change the contract. pylint min-public-methods=2 fires
# on any 1-method Protocol regardless.
@runtime_checkable
class ProbeLLM(Protocol):  # pylint: disable=too-few-public-methods
    """Minimal LLM interface used by every PROBE node and policy.

    Implementations return a 3-tuple of ``(response_text, tokens_in,
    tokens_out)``. Token counts of ``(0, 0)`` are valid (e.g. for the fake
    in unit tests where budget enforcement is not being exercised).
    """

    def invoke(self, prompt: str) -> tuple[str, int, int]:
        """Send ``prompt`` to the underlying model and return its response."""


class _FakeLLM:
    """Deterministic in-process LLM for unit tests.

    Two mutually-exclusive modes:

    - **Script mode**: pass ``script={"label": ["resp1", "resp2", ...]}``.
      Each ``invoke()`` returns the next response from the current label's
      bucket; switch labels via ``set_label()``. Raises ``AssertionError``
      when a bucket is exhausted.

    - **Responder mode**: pass ``responder=callable``. The callable receives
      the full prompt and returns ``(response, tokens_in, tokens_out)``.
      Use this for prompt-content assertions where the responder inspects
      what the node actually built.

    Exactly one mode must be set; rejecting both empty and both populated.
    """

    def __init__(
        self,
        script: Optional[dict[str, list[str]]] = None,
        responder: Optional[Callable[[str], tuple[str, int, int]]] = None,
        tokens_per_call: tuple[int, int] = (0, 0),
        label_when_unscripted: str = "default",
    ) -> None:
        if script is None and responder is None:
            raise ValueError(
                "_FakeLLM requires exactly one of `script` or `responder`."
            )
        if script is not None and responder is not None:
            raise ValueError(
                "_FakeLLM rejects both `script` and `responder` set "
                "simultaneously; pass exactly one."
            )
        self._script = script
        self._responder = responder
        self._tokens_per_call = tokens_per_call
        self._current_label = label_when_unscripted
        self._cursors: dict[str, int] = {}

    def invoke(self, prompt: str) -> tuple[str, int, int]:
        """See :class:`ProbeLLM`."""
        if self._responder is not None:
            return self._responder(prompt)
        # Script mode
        label = self._current_label
        if self._script is None or label not in self._script:
            raise AssertionError(
                f"_FakeLLM script has no bucket for label {label!r}; "
                f"available labels: {sorted(self._script or {})}"
            )
        bucket = self._script[label]
        cursor = self._cursors.get(label, 0)
        if cursor >= len(bucket):
            raise AssertionError(
                f"_FakeLLM script exhausted for label {label!r} "
                f"({cursor} responses requested, only {len(bucket)} scripted)"
            )
        response = bucket[cursor]
        self._cursors[label] = cursor + 1
        tokens_in, tokens_out = self._tokens_per_call
        return response, tokens_in, tokens_out

    def set_label(self, label: str) -> None:
        """Switch which script bucket subsequent ``invoke()`` calls drain from."""
        self._current_label = label


# ---------------------------------------------------------------------------
# Stub helpers for ``--stub`` mode (offline pipeline testing / CI).
#
# ``_stub_responder`` drives a :class:`_FakeLLM` with deterministic,
# plausibly-shaped JSON for every PROBE node type; ``_StubVictimExecutor``
# stands in for a real AETHER ``MASExecutor`` so the whole runner pipeline
# executes without the network. Both are imported by the suite runner and
# by the attacker-MAS integration tests, so they live here next to
# ``_FakeLLM`` rather than being duplicated at each call site.
# ---------------------------------------------------------------------------

_STUB_PLANNER_JSON = (
    '{"improvement": "try roleplay since direct ask failed", '
    '"attack_angle": "stub_roleplay", '
    '"rationale": "stub rationale for offline testing"}'
)

_STUB_LADDER_JSON = (
    '{"ladder": ["benign opener", "ask about topic", "probe deeper", '
    '"recontextualize", "indirect framing", "specific request", '
    '"persuasive framing", "final ask"]}'
)

_STUB_REFINEMENTS_JSON = (
    '{"refinements": ["refinement A", "refinement B", "refinement C"]}'
)

_STUB_JUDGE_JSON = (
    '{"score": 1, "reasoning": "stub partial progress", "confidence": "low"}'
)

_STUB_CRAFTER_OUTPUT = (
    "Stub crafted payload — used in --stub mode for offline pipeline testing."
)


def _stub_responder(prompt: str) -> tuple[str, int, int]:
    """Return a deterministic plausibly-shaped response for any PROBE prompt."""
    lowered = prompt.lower()
    if "ladder" in lowered and "json object with one key" in lowered:
        return _STUB_LADDER_JSON, 80, 40
    if "refinements" in lowered:
        return _STUB_REFINEMENTS_JSON, 80, 40
    if "score guide" in lowered or "Score this turn" in prompt:
        return _STUB_JUDGE_JSON, 30, 15
    if "FRAMEWORK" in prompt and "SEPARATOR" in prompt:
        return _STUB_CRAFTER_OUTPUT, 60, 30
    if "improvement" in lowered and "attack_angle" in lowered:
        return _STUB_PLANNER_JSON, 50, 25
    # Fallback: return planner JSON shape (least harmful default)
    return _STUB_PLANNER_JSON, 50, 25


class _StubVictimExecutor:  # pylint: disable=too-few-public-methods  # callable stub records calls/received for test assertions
    """Stub AETHER ``MASExecutor`` for ``--stub`` mode and integration tests.

    Records each ``run`` input and returns a canned result dict shaped like
    a real ``MASExecutionResult`` so the observer's defensive code paths
    still get exercised. Set ``raises`` to make ``run`` raise instead (used
    by the victim-crash termination test).
    """

    def __init__(self, raises: Optional[Exception] = None) -> None:
        self.raises = raises
        self.calls = 0
        self.received: list[str] = []

    def run(self, input_data: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        """Mimic ``MASExecutor.run`` — record the input and return canned output.

        ``**_kwargs`` absorbs interface-required keyword args (``save_results``
        etc.) that this stub ignores.
        """
        self.calls += 1
        if input_data and "messages" in input_data:
            messages = input_data["messages"]
            if messages:
                content = getattr(messages[0], "content", "")
                self.received.append(content)
        if self.raises is not None:
            raise self.raises
        return {
            "messages": [],
            "agent_results": [
                {
                    "agent_id": "agent_a",
                    "role": "reviewer",
                    "input_state": {
                        "prompt": self.received[-1] if self.received else ""
                    },
                    "output_state": {"text": "victim said X"},
                }
            ],
        }


class _LangChainLLMAdapter:  # pylint: disable=too-few-public-methods  # thin adapter implementing ProbeLLM.invoke
    """Wraps a LangChain ChatModel to satisfy the :class:`ProbeLLM` Protocol.

    The wrapped model is stored as ``chat_model``; tests that previously
    read ``adapter._chat`` should use ``adapter.chat_model``.
    """

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model

    def invoke(self, prompt: str) -> tuple[str, int, int]:
        """Invoke the underlying ChatModel with ``prompt`` as a HumanMessage."""
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            LOGGER.warning(
                "LangChain response has no usage_metadata; token counts "
                "default to (0, 0). Model: %s",
                type(self.chat_model).__name__,
            )
            return text, 0, 0
        return (
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )


def resolve_real_llm(model_config: dict[str, Any]) -> ProbeLLM:
    """Wrap an IRIS-loaded LangChain ChatModel into a :class:`ProbeLLM`.

    ``model_config`` is passed through to
    :func:`bili.iris.loaders.llm_loader.load_model` as keyword arguments.
    The dict MUST contain a ``model_type`` key (one of
    ``remote_aws_bedrock``, ``remote_google_vertex``, ``remote_azure_openai``,
    ``local_llamacpp``, ``local_huggingface``) plus the appropriate
    type-specific kwargs (e.g. ``model_name``, ``temperature``).

    Returns an adapter whose ``invoke`` method translates the LangChain
    ``AIMessage`` return into the PROBE 3-tuple shape, reading token counts
    from ``response.usage_metadata``; when absent, falls back to ``(0, 0)``
    with a ``LOGGER.warning``.
    """
    # IRIS llm_loader transitively imports torch, transformers, and
    # Streamlit — multi-second import cost. Defer until a real (non-stub)
    # LLM is actually needed so stub-only PROBE tests stay fast.
    from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
        load_model,
    )

    chat_model = load_model(**model_config)
    return _LangChainLLMAdapter(chat_model)
