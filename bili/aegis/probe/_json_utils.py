"""Robust LLM-call helpers: invoke, parse JSON, retry once, fall back.

Consolidates the "invoke LLM → parse JSON → retry on parse failure → fall
back to stub" pattern that's repeated across PROBE's success_evaluator,
PAIR planner, Crescendo ladder generation, and TAP branching. A single
implementation gives us one place to test the retry/fallback semantics
aggressively and one place to fix bugs.

Token counts from BOTH attempts (when a retry happens) are accumulated so
that BudgetState accounting is honest even when we paid for a failed parse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

from bili.aegis.probe._llm import ProbeLLM

LOGGER = logging.getLogger(__name__)


_RETRY_PREFIX = (
    "Your previous response was not valid JSON in the required format. "
    "Retry now, returning ONLY the JSON object specified by the original "
    "instructions.\n\n"
    "Original instructions follow.\n\n"
)


def _extract_json_object(raw: str) -> dict:
    """Extract a single JSON object from an LLM response string.

    Strips Markdown code fences (``` ... ```), then tries
    :func:`json.loads` on the full text. If that fails, locates the first
    ``{`` and last ``}`` and parses the slice. The result must be a JSON
    object (``dict``); a JSON array or scalar raises.

    Raises:
        ValueError: if no JSON object can be extracted.
    """
    cleaned = raw.strip()

    # Strip markdown code fences (handles single-line and multi-line)
    cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {raw!r}") from None
        try:
            data = json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSON from response: {exc}; raw={raw!r}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}: {data!r}")

    return data


def invoke_with_json_retry(
    llm: ProbeLLM,
    prompt: str,
    required_keys: set[str],
    fallback_factory: Callable[[], dict],
    label: str = "",
) -> tuple[dict, int, int]:
    """Invoke an LLM, parse its JSON response, retry on failure, fall back.

    Pipeline:

    1. Call ``llm.invoke(prompt)`` and try to parse the response as a JSON
       object containing every key in ``required_keys``.
    2. If parsing fails OR required keys are missing, log a warning and
       retry once with the prompt prefixed by an error-signal preamble.
    3. If the retry also fails, log a warning and return
       ``fallback_factory()`` instead.

    Token counts from BOTH attempts are accumulated into the returned tuple
    so the runner's BudgetState reflects what was actually paid for. The
    ``fallback_factory`` is called at most once.

    Args:
        llm: any :class:`ProbeLLM` implementation.
        prompt: the first-attempt prompt.
        required_keys: set of keys the parsed dict MUST contain. Pass
            ``set()`` to skip the key-presence check (only parse must succeed).
        fallback_factory: zero-arg callable returning the fallback dict.
            Called once if both attempts fail.
        label: short identifier (e.g. ``"pair_planner"``) used in log records.

    Returns:
        ``(parsed_dict, total_tokens_in, total_tokens_out)`` where
        ``parsed_dict`` is the LLM-produced dict on success or
        ``fallback_factory()``'s return on double failure.
    """
    total_in = 0
    total_out = 0

    # First attempt
    response_text, tokens_in, tokens_out = llm.invoke(prompt)
    total_in += tokens_in
    total_out += tokens_out

    parsed = _try_parse(response_text, required_keys, label, attempt="first")
    if parsed is not None:
        return parsed, total_in, total_out

    # Retry once with an explicit error signal in the prompt
    retry_prompt = _RETRY_PREFIX + prompt
    response_text_2, tokens_in_2, tokens_out_2 = llm.invoke(retry_prompt)
    total_in += tokens_in_2
    total_out += tokens_out_2

    parsed = _try_parse(response_text_2, required_keys, label, attempt="retry")
    if parsed is not None:
        return parsed, total_in, total_out

    LOGGER.warning(
        "invoke_with_json_retry: falling back to stub for label=%r "
        "after two failed attempts",
        label,
    )
    return fallback_factory(), total_in, total_out


def _try_parse(
    response_text: str,
    required_keys: set[str],
    label: str,
    attempt: str,
) -> dict | None:
    """Parse JSON, validate required keys, log on failure, return None on miss."""
    try:
        parsed = _extract_json_object(response_text)
    except ValueError as exc:
        LOGGER.warning(
            "invoke_with_json_retry: %s parse failed for label=%r: %s",
            attempt,
            label,
            exc,
        )
        return None

    missing = required_keys - set(parsed.keys())
    if missing:
        LOGGER.warning(
            "invoke_with_json_retry: %s response missing required keys for "
            "label=%r; got %s, missing %s",
            attempt,
            label,
            sorted(parsed.keys()),
            sorted(missing),
        )
        return None

    return parsed
