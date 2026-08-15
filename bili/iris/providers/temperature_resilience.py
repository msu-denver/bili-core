"""Temperature resilience for chat-model providers.

Current frontier "reasoning" models reject a ``temperature`` parameter: the
request fails with a ``400`` such as ``temperature is deprecated for this
model`` (Anthropic) or ``... does not support ... Only the default ... value is
supported`` (OpenAI o-series / GPT-5.x).  bili-core sends a temperature on
essentially every agent call (from role presets and defaults), so without this
module it simply cannot drive those models.

:func:`apply_temperature_resilience` makes a loaded chat model self-adapt: when a
generation call fails with a temperature-rejection ``400``, it retries the same
request once with ``temperature`` removed, and remembers (for the rest of the
process) that the model rejects temperature so subsequent calls strip it up
front.  That is one wasted round-trip per model per process, not per call, and it
needs no per-model catalog metadata, so it works for any new model automatically.

Design
------
- **Type identity is preserved.**  The model's generation methods are wrapped in
  place on the instance rather than the object being wrapped in a proxy, so
  provider-specific ``isinstance`` checks elsewhere (prompt-caching gates,
  ``bind_tools`` detection, structured-output detection) are unaffected.
- **Every invocation path is covered.**  Wrapping ``_generate`` / ``_agenerate``
  / ``_stream`` / ``_astream`` (the methods every LangChain invocation funnels
  through, including ``bind_tools(...).invoke`` and agent graphs) means callers
  that invoke the loaded model directly are covered too.
- **Non-temperature errors are never swallowed.**  The rejection is recognised by
  the error message (``temperature`` plus a rejection marker); anything else is
  re-raised unchanged.
- **Temperature-accepting models are unchanged.**  The retry branch never fires,
  so the request still carries the configured temperature.
"""

# This module's whole job is to wrap a chat model's protected generation
# methods (_generate/_agenerate/_stream/_astream) in place, so protected-member
# access is intrinsic rather than a smell.
# pylint: disable=protected-access
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

#: Substrings that, together with the word "temperature" in an error message,
#: mark a provider rejecting the parameter (rather than an unrelated 400).  Kept
#: message-based and provider-agnostic so no per-SDK exception import is needed;
#: an unrecognised error is re-raised, so a miss is safe (behaviour as if this
#: module were absent).
_TEMPERATURE_REJECTION_MARKERS = (
    "deprecated",
    "not supported",
    "does not support",
    "unsupported",
    "only the default",
    "only default",
    "must be the default",
    "cannot be set",
)


def _is_temperature_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* looks like a provider rejecting ``temperature``.

    Requires the word ``temperature`` and one of
    :data:`_TEMPERATURE_REJECTION_MARKERS`, so an unrelated ``400`` (bad key,
    context length, content policy) is not mistaken for a temperature rejection.
    """
    text = str(exc).lower()
    if "temperature" not in text:
        return False
    return any(marker in text for marker in _TEMPERATURE_REJECTION_MARKERS)


def apply_temperature_resilience(model: Any) -> Any:
    """Wrap *model*'s generation methods so a temperature rejection self-heals.

    Returns *model* (the same object) so callers can assign the result.  A model
    that sets no temperature, is not a standard chat model, or has already been
    wrapped is returned unchanged.

    :param model: A loaded chat model (or any object; non-chat-models are a
        no-op).
    :returns: The same object, with temperature-resilient generation methods.
    :rtype: Any
    """
    # One cohesive routine that installs the four generation-method wrappers.
    # pylint: disable=too-many-statements
    # Nothing to protect: no temperature is ever sent.
    if getattr(model, "temperature", None) is None:
        return model
    cls = type(model)
    # Not a standard chat model (e.g. a CLI/subprocess provider): nothing to wrap.
    if not hasattr(cls, "_generate"):
        return model
    # Idempotent: the wrapped method carries a marker attribute.
    if getattr(getattr(model, "_generate", None), "_bili_temperature_resilient", False):
        return model

    # Shared per-model state: once a rejection is seen, ``stripped`` holds a
    # temperature-free copy reused for every subsequent call (proactive strip).
    state: dict = {"stripped": None}

    def _stripped() -> Any:
        if state["stripped"] is None:
            state["stripped"] = model.model_copy(update={"temperature": None})
        return state["stripped"]

    def _note_rejection() -> None:
        LOGGER.warning(
            "Model %s rejected 'temperature'; retrying without it and dropping it "
            "for the rest of this process.",
            getattr(model, "model", None) or getattr(model, "model_id", cls.__name__),
        )

    real_generate = cls._generate

    def _generate(*args: Any, **kwargs: Any) -> Any:
        if state["stripped"] is not None:
            return real_generate(_stripped(), *args, **kwargs)
        try:
            return real_generate(model, *args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not _is_temperature_error(exc):
                raise
            _note_rejection()
            return real_generate(_stripped(), *args, **kwargs)

    _generate._bili_temperature_resilient = True  # type: ignore[attr-defined]
    model._generate = _generate

    if hasattr(cls, "_agenerate"):
        real_agenerate = cls._agenerate

        async def _agenerate(*args: Any, **kwargs: Any) -> Any:
            if state["stripped"] is not None:
                return await real_agenerate(_stripped(), *args, **kwargs)
            try:
                return await real_agenerate(model, *args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not _is_temperature_error(exc):
                    raise
                _note_rejection()
                return await real_agenerate(_stripped(), *args, **kwargs)

        model._agenerate = _agenerate

    if hasattr(cls, "_stream"):
        real_stream = cls._stream

        def _stream(*args: Any, **kwargs: Any) -> Any:
            if state["stripped"] is not None:
                yield from real_stream(_stripped(), *args, **kwargs)
                return
            iterator = real_stream(model, *args, **kwargs)
            try:
                first = next(iterator)
            except StopIteration:
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Retry only on a clean start-failure (temperature is rejected
                # before any token); a mid-stream error is not re-attempted.
                if not _is_temperature_error(exc):
                    raise
                _note_rejection()
                yield from real_stream(_stripped(), *args, **kwargs)
                return
            yield first
            yield from iterator

        model._stream = _stream

    if hasattr(cls, "_astream"):
        real_astream = cls._astream

        async def _astream(*args: Any, **kwargs: Any) -> Any:
            if state["stripped"] is not None:
                async for chunk in real_astream(_stripped(), *args, **kwargs):
                    yield chunk
                return
            iterator = real_astream(model, *args, **kwargs)
            try:
                first = await anext(iterator)
            except StopAsyncIteration:
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not _is_temperature_error(exc):
                    raise
                _note_rejection()
                async for chunk in real_astream(_stripped(), *args, **kwargs):
                    yield chunk
                return
            yield first
            async for chunk in iterator:
                yield chunk

        model._astream = _astream

    return model
