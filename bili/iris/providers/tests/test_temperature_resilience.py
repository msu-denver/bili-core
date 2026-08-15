"""Tests for temperature-rejection resilience.

Current reasoning models reject a ``temperature`` parameter with a ``400``.
:func:`apply_temperature_resilience` makes a loaded model retry the request once
without temperature and remember the rejection, so bili-core can drive those
models.  No network is used: a fake chat model raises a temperature-style error.
"""

# Tests deliberately inspect the wrapped protected generation method.
# pylint: disable=protected-access
import asyncio
import sys
from typing import List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from bili.iris.loaders.llm_loader import load_model
from bili.iris.providers import LLMProvider, register_provider
from bili.iris.providers.temperature_resilience import (
    _is_temperature_error,
    apply_temperature_resilience,
)


class _TemperatureError(Exception):
    """Stand-in for a provider's 400 rejecting temperature."""


class FakeChat(BaseChatModel):
    """A chat model that records the temperature in effect on each call and,
    when configured to, rejects a non-None temperature the way a reasoning model
    does.  Implements all four generation methods so each wrapper is exercised.
    """

    temperature: Optional[float] = 0.7
    reject_temperature: bool = True
    error_text: str = (
        "400 invalid_request_error: temperature is deprecated for this model"
    )
    temps_seen: Optional[List[Optional[float]]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A per-instance record (not a pydantic field) of temperatures used.
        object.__setattr__(self, "temps_seen", [])

    def _reject_if_needed(self) -> None:
        self.temps_seen.append(self.temperature)
        if self.reject_temperature and self.temperature is not None:
            raise _TemperatureError(self.error_text)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._reject_if_needed()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        self._reject_if_needed()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._reject_if_needed()
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._reject_if_needed()
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

    @property
    def _llm_type(self) -> str:
        return "fake-chat"


# ---------------------------------------------------------------------------
# _is_temperature_error
# ---------------------------------------------------------------------------


class TestIsTemperatureError:
    """The predicate recognises a temperature rejection and nothing else."""

    @pytest.mark.parametrize(
        "text",
        [
            "temperature is deprecated for this model",
            (
                "Unsupported value: 'temperature' does not support 0.7. "
                "Only the default (1) value is supported."
            ),
            "temperature is not supported with this model",
            "temperature must be the default",
        ],
    )
    def test_true_for_temperature_rejections(self, text):
        """A message naming temperature plus a rejection marker is recognised."""
        assert _is_temperature_error(Exception(text)) is True

    @pytest.mark.parametrize(
        "text",
        [
            "rate limit exceeded",
            "context_length_exceeded: too many tokens",
            "invalid api key",
            "temperature was set to 0.5",  # names temperature but no rejection marker
        ],
    )
    def test_false_for_other_errors(self, text):
        """An unrelated error, or temperature without a rejection marker, is not."""
        assert _is_temperature_error(Exception(text)) is False


# ---------------------------------------------------------------------------
# apply_temperature_resilience
# ---------------------------------------------------------------------------


class TestApplyTemperatureResilience:
    """The wrapper self-heals a temperature rejection and is otherwise inert."""

    def test_rejecting_model_retries_without_temperature(self):
        """The first call tries the temperature, is rejected, and retries without."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=True)
        )
        result = model.invoke("hi")
        assert result.content == "ok"
        # First the configured temperature (rejected), then None (retry succeeds).
        assert model.temps_seen == [0.7, None]

    def test_rejection_is_remembered_for_later_calls(self):
        """After one rejection, later calls strip temperature up front."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=True)
        )
        model.invoke("one")
        model.invoke("two")
        # Only the very first call pays the rejected round-trip; then all None.
        assert model.temps_seen == [0.7, None, None]

    def test_accepting_model_still_sends_temperature(self):
        """A model that accepts temperature is unchanged: temperature still sent."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=False)
        )
        result = model.invoke("hi")
        assert result.content == "ok"
        assert model.temps_seen == [0.7]

    def test_non_temperature_error_is_reraised(self):
        """A non-temperature 400 is re-raised, not retried."""
        model = apply_temperature_resilience(
            FakeChat(
                temperature=0.7,
                reject_temperature=True,
                error_text="400 invalid_request_error: context_length_exceeded",
            )
        )
        with pytest.raises(_TemperatureError, match="context_length_exceeded"):
            model.invoke("hi")
        # No retry: exactly one underlying call.
        assert model.temps_seen == [0.7]

    def test_no_temperature_model_is_untouched(self):
        """A model that sets no temperature is returned unwrapped."""
        model = FakeChat(temperature=None)
        before = model._generate
        result = apply_temperature_resilience(model)
        assert result is model
        assert model._generate == before

    def test_non_chat_model_is_untouched(self):
        """An object without a chat-model generate method is a no-op."""

        class NotAChatModel:  # pylint: disable=too-few-public-methods
            """A plain object with a temperature but no chat-model interface."""

            temperature = 0.7

        obj = NotAChatModel()
        assert apply_temperature_resilience(obj) is obj

    def test_idempotent(self):
        """Applying twice does not double-wrap."""
        model = apply_temperature_resilience(FakeChat(temperature=0.7))
        wrapped = model._generate
        apply_temperature_resilience(model)
        assert model._generate is wrapped

    def test_returns_same_object(self):
        """The model object (and its type) is preserved, not proxied."""
        model = FakeChat(temperature=0.7)
        assert apply_temperature_resilience(model) is model
        assert isinstance(model, FakeChat)

    def test_async_retries_without_temperature(self):
        """The async path self-heals, then strips proactively on reuse."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=True)
        )

        async def _drive():
            first = await model.ainvoke("one")
            second = await model.ainvoke("two")
            return first, second

        first, second = asyncio.run(_drive())
        assert first.content == "ok"
        assert second.content == "ok"
        # First call: rejected temperature then retry; second: proactive strip.
        assert model.temps_seen == [0.7, None, None]

    def test_sync_stream_retries_without_temperature(self):
        """Streaming self-heals on a clean start-failure, then strips proactively."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=True)
        )
        first = [c.content for c in model.stream("one")]
        second = [c.content for c in model.stream("two")]
        # LangChain may append a trailing empty chunk; join to compare content.
        assert "".join(first) == "ok"
        assert "".join(second) == "ok"
        assert model.temps_seen == [0.7, None, None]

    def test_async_stream_retries_without_temperature(self):
        """Async streaming self-heals, then strips proactively on reuse."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=True)
        )

        async def _collect(prompt):
            return [chunk.content async for chunk in model.astream(prompt)]

        first = asyncio.run(_collect("one"))
        second = asyncio.run(_collect("two"))
        assert "".join(first) == "ok"
        assert "".join(second) == "ok"
        assert model.temps_seen == [0.7, None, None]

    def test_accepting_model_streams_unchanged(self):
        """A temperature-accepting model streams with temperature still sent."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=False)
        )
        chunks = [c.content for c in model.stream("hi")]
        assert "".join(chunks) == "ok"
        assert model.temps_seen == [0.7]

    def test_accepting_model_astreams_unchanged(self):
        """A temperature-accepting model async-streams with temperature sent."""
        model = apply_temperature_resilience(
            FakeChat(temperature=0.7, reject_temperature=False)
        )

        async def _collect():
            return [chunk.content async for chunk in model.astream("hi")]

        chunks = asyncio.run(_collect())
        assert "".join(chunks) == "ok"
        assert model.temps_seen == [0.7]

    def test_stream_non_temperature_error_reraised(self):
        """A non-temperature error during streaming is re-raised, not retried."""
        model = apply_temperature_resilience(
            FakeChat(
                temperature=0.7,
                reject_temperature=True,
                error_text="400 invalid_request_error: content policy violation",
            )
        )
        with pytest.raises(_TemperatureError, match="content policy"):
            list(model.stream("hi"))
        assert model.temps_seen == [0.7]

    def test_async_non_temperature_error_reraised(self):
        """A non-temperature error on the async path is re-raised, not retried."""
        model = apply_temperature_resilience(
            FakeChat(
                temperature=0.7,
                reject_temperature=True,
                error_text="400 invalid_request_error: context_length_exceeded",
            )
        )
        with pytest.raises(_TemperatureError, match="context_length_exceeded"):
            asyncio.run(model.ainvoke("hi"))
        assert model.temps_seen == [0.7]

    def test_async_stream_non_temperature_error_reraised(self):
        """A non-temperature error during async streaming is re-raised."""
        model = apply_temperature_resilience(
            FakeChat(
                temperature=0.7,
                reject_temperature=True,
                error_text="400 invalid_request_error: content policy violation",
            )
        )

        async def _collect():
            return [chunk.content async for chunk in model.astream("hi")]

        with pytest.raises(_TemperatureError, match="content policy"):
            asyncio.run(_collect())
        assert model.temps_seen == [0.7]


# ---------------------------------------------------------------------------
# Wiring: load_model applies resilience to every loaded model.
# ---------------------------------------------------------------------------


def test_load_model_applies_resilience():
    """A model obtained through load_model self-heals a temperature rejection."""

    class _FakeTempProvider(LLMProvider):  # pylint: disable=too-few-public-methods
        def load(self, **kwargs):  # pylint: disable=arguments-differ
            return FakeChat(temperature=0.7, reject_temperature=True)

    register_provider("remote_fake_temp_resilience", _FakeTempProvider)

    model = load_model("remote_fake_temp_resilience", model_name="x")
    result = model.invoke("hi")
    assert result.content == "ok"
    # The rejected-then-retried pattern proves resilience was applied by the loader.
    assert model.temps_seen == [0.7, None]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
