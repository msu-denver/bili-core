"""Tests for the bili-core provider fallback engine.

Covers:
- ``FallbackPolicy`` exception classification (retryable vs fatal, all three
  modes: type-based, callable, name-based default)
- ``ProviderChain`` construction, iteration, and ``load_all()``
- ``FallbackLLM.invoke()`` — first-fails-falls-to-second, all-fail-raises,
  no-fallback-configured (plain passthrough), fatal-error-does-not-fallback,
  max_attempts cap
- ``FallbackLLM.stream()`` — same retryable vs fatal split
- ``FallbackLLM.from_chain()`` — end-to-end construction via ProviderChain
- ``build_fallback_llm()`` — convenience constructor
- ``AgentSpec.fallback_models`` field — default empty, round-trips, accepted
- ``create_llm()`` integration — no fallback → plain LLM; with fallback →
  FallbackLLM; resolver propagation
"""

# pylint: disable=too-few-public-methods,duplicate-code,no-member,too-many-lines

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from bili.aether.compiler.llm_resolver import _load_fallback_member, create_llm
from bili.aether.schema import AgentSpec
from bili.iris.providers.base import LLMProvider
from bili.iris.providers.fallback import (
    DEFAULT_POLICY,
    FallbackLLM,
    FallbackPolicy,
    ProviderChain,
    _is_retryable_by_name,
    _registry_load,
    build_fallback_llm,
)
from bili.iris.providers.registry import PROVIDER_REGISTRY

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_llm(response="ok", side_effect=None):
    """Return a minimal mock LLM with .invoke(), .stream(), .astream()."""
    llm = MagicMock()
    if side_effect is not None:
        llm.invoke.side_effect = side_effect
        llm.stream.side_effect = side_effect
    else:
        llm.invoke.return_value = MagicMock(content=response)
        llm.stream.return_value = iter([MagicMock(content=response)])
    return llm


class _RetryableError(Exception):
    """An exception whose name is in _DEFAULT_RETRYABLE_NAMES."""


class _FatalError(Exception):
    """An exception whose name is NOT in _DEFAULT_RETRYABLE_NAMES."""


# Patch the name so the default policy sees it correctly.
_RetryableError.__name__ = "RateLimitError"


# ---------------------------------------------------------------------------
# FallbackPolicy
# ---------------------------------------------------------------------------


class TestFallbackPolicy:
    """Verify FallbackPolicy exception classification in all three modes."""

    def test_default_policy_retryable_by_name(self):
        """Default policy classifies known retryable names as retryable."""
        exc = _RetryableError("too many requests")
        assert DEFAULT_POLICY.should_fallback(exc) is True

    def test_default_policy_fatal_by_name(self):
        """Default policy classifies unknown exception names as fatal."""
        exc = _FatalError("auth error")
        assert DEFAULT_POLICY.should_fallback(exc) is False

    def test_type_based_policy_retryable(self):
        """Type-based policy classifies instance-of as retryable."""
        policy = FallbackPolicy(retryable_exceptions=(ValueError, TimeoutError))
        assert policy.should_fallback(ValueError("bad")) is True
        assert policy.should_fallback(TimeoutError()) is True

    def test_type_based_policy_fatal(self):
        """Type-based policy classifies non-matching types as fatal."""
        policy = FallbackPolicy(retryable_exceptions=(ValueError,))
        assert policy.should_fallback(TypeError("wrong type")) is False

    def test_callable_policy_overrides_type(self):
        """Callable policy takes precedence over retryable_exceptions."""

        def my_fn(exc):
            return "retry" in str(exc)

        policy = FallbackPolicy(
            retryable_exceptions=(TypeError,),  # would say TypeError is retryable
            is_retryable=my_fn,  # callable overrides
        )
        # callable says yes for ValueError if message contains "retry"
        assert policy.should_fallback(ValueError("please retry")) is True
        # callable says no for TypeError even though it's in retryable_exceptions
        assert policy.should_fallback(TypeError("wrong")) is False

    def test_callable_policy_fatal(self):
        """Callable policy returning False means fatal."""
        policy = FallbackPolicy(is_retryable=lambda _: False)
        assert policy.should_fallback(Exception("anything")) is False

    def test_max_attempts_stored(self):
        """max_attempts is stored on the policy."""
        policy = FallbackPolicy(max_attempts=3)
        assert policy.max_attempts == 3

    def test_default_max_attempts_is_zero(self):
        """Default max_attempts is 0 (unlimited)."""
        policy = FallbackPolicy()
        assert policy.max_attempts == 0


class TestIsRetryableByName:
    """Unit tests for the name-based helper used by the default policy."""

    def test_known_name_is_retryable(self):
        """Known retryable name resolves to True."""

        class RateLimitError(Exception):
            """Fake rate limit error."""

        assert _is_retryable_by_name(RateLimitError()) is True

    def test_unknown_name_is_not_retryable(self):
        """Unknown name resolves to False."""

        class FooBarBazError(Exception):
            """Completely unknown error."""

        assert _is_retryable_by_name(FooBarBazError()) is False

    def test_subclass_of_known_name(self):
        """Subclass of a known name is retryable via MRO check."""

        class TimeoutError(Exception):  # pylint: disable=redefined-builtin
            """Timeout."""

        class SpecificTimeoutError(TimeoutError):
            """More specific timeout."""

        assert _is_retryable_by_name(SpecificTimeoutError()) is True

    def test_api_status_error_base_class_is_fatal(self):
        """APIStatusError itself is NOT in the retryable set.

        APIStatusError is the base class for all openai/anthropic status-coded
        errors including auth (401), bad request (400), and permission (403).
        It must NOT be treated as retryable — doing so would silently fall
        through to other providers on auth failures, masking misconfiguration.
        """

        class APIStatusError(Exception):
            """Simulated openai/anthropic base status error."""

        assert _is_retryable_by_name(APIStatusError()) is False

    def test_api_status_error_subclass_auth_failure_is_fatal(self):
        """AuthenticationError (subclass of APIStatusError) is classified fatal.

        Because APIStatusError is not in the retryable set, any subclass
        (AuthenticationError, BadRequestError, PermissionDeniedError, etc.)
        is also fatal unless its own class name is explicitly listed.
        """

        class APIStatusError(Exception):
            """Simulated base."""

        class AuthenticationError(APIStatusError):
            """Simulated 401 auth failure."""

        assert _is_retryable_by_name(AuthenticationError()) is False

    def test_rate_limit_error_subclass_of_api_status_is_retryable(self):
        """RateLimitError is retryable even though it inherits APIStatusError.

        RateLimitError is listed explicitly in the retryable set.  The MRO
        check finds it by the direct class name.
        """

        class APIStatusError(Exception):
            """Simulated base."""

        class RateLimitError(APIStatusError):
            """Simulated 429 rate limit."""

        assert _is_retryable_by_name(RateLimitError()) is True


# ---------------------------------------------------------------------------
# ProviderChain
# ---------------------------------------------------------------------------


class TestProviderChain:
    """Verify ProviderChain construction and load_all()."""

    def test_empty_raises(self):
        """Empty entries list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            ProviderChain([])

    def test_len(self):
        """len() returns number of entries."""
        chain = ProviderChain([("a", {}), ("b", {})])
        assert len(chain) == 2

    def test_iter(self):
        """Iteration yields (provider_type, kwargs) tuples."""
        entries = [("a", {"x": 1}), ("b", {"y": 2})]
        chain = ProviderChain(entries)
        assert list(chain) == entries

    def test_repr(self):
        """repr includes provider type strings."""
        chain = ProviderChain([("remote_openai", {})])
        assert "remote_openai" in repr(chain)

    def test_load_all_calls_registry(self):
        """load_all() calls get_or_raise and each provider's load()."""
        mock_llm = MagicMock()

        type_key = "_test_chain_load_type"
        PROVIDER_REGISTRY.register(
            type_key,
            type("P", (LLMProvider,), {"load": lambda self, **kw: mock_llm}),
        )
        try:
            chain = ProviderChain([(type_key, {"model_name": "m"})])
            llms = chain.load_all()
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert len(llms) == 1
        assert llms[0] is mock_llm

    def test_load_all_raises_for_unknown_type(self):
        """load_all() raises ValueError for unregistered provider type."""
        chain = ProviderChain([("__nonexistent_provider__", {})])
        with pytest.raises(ValueError, match="Invalid model type"):
            chain.load_all()


# ---------------------------------------------------------------------------
# FallbackLLM.invoke()
# ---------------------------------------------------------------------------


class TestFallbackLLMInvoke:
    """Verify FallbackLLM.invoke() fallback semantics."""

    def test_primary_success_no_fallback_called(self):
        """When primary succeeds, fallback is never called."""
        primary = _mock_llm("primary-response")
        fallback = _mock_llm("fallback-response")
        llm = FallbackLLM(primary=primary, fallbacks=[fallback])

        result = llm.invoke(["msg"])

        assert result.content == "primary-response"
        fallback.invoke.assert_not_called()

    def test_first_fails_falls_to_second(self):
        """Retryable failure on primary → fallback is tried and succeeds."""
        primary = _mock_llm(side_effect=_RetryableError("rate limited"))
        fallback = _mock_llm("fallback-ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        result = llm.invoke(["msg"])

        assert result.content == "fallback-ok"
        primary.invoke.assert_called_once()
        fallback.invoke.assert_called_once()

    def test_all_fail_raises_last_exception(self):
        """When all providers fail, the last exception is re-raised."""
        err1 = _RetryableError("provider 1 down")
        err2 = _RetryableError("provider 2 down")
        primary = _mock_llm(side_effect=err1)
        fallback = _mock_llm(side_effect=err2)
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        with pytest.raises(_RetryableError) as exc_info:
            llm.invoke(["msg"])

        assert exc_info.value is err2

    def test_no_fallback_configured_passes_through(self):
        """FallbackLLM with no fallbacks behaves identically to a plain LLM."""
        primary = _mock_llm("sole-response")
        llm = FallbackLLM(primary=primary, fallbacks=[])

        result = llm.invoke(["msg"])

        assert result.content == "sole-response"

    def test_fatal_error_does_not_fallback(self):
        """Fatal (non-retryable) exceptions are re-raised immediately."""
        fatal_err = _FatalError("auth failed")
        primary = _mock_llm(side_effect=fatal_err)
        fallback = _mock_llm("fallback-ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        with pytest.raises(_FatalError):
            llm.invoke(["msg"])

        fallback.invoke.assert_not_called()

    def test_max_attempts_caps_chain(self):
        """Policy max_attempts limits how many providers are tried."""
        err = _RetryableError("down")
        p1 = _mock_llm(side_effect=err)
        p2 = _mock_llm(side_effect=err)
        p3 = _mock_llm("p3-ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,), max_attempts=2)
        llm = FallbackLLM(primary=p1, fallbacks=[p2, p3], policy=policy)

        with pytest.raises(_RetryableError):
            llm.invoke(["msg"])

        # Only p1 and p2 were attempted; p3 was NOT reached.
        p1.invoke.assert_called_once()
        p2.invoke.assert_called_once()
        p3.invoke.assert_not_called()

    def test_kwargs_forwarded_to_provider(self):
        """Extra kwargs are forwarded to each provider's .invoke()."""
        primary = _mock_llm("ok")
        llm = FallbackLLM(primary=primary)

        llm.invoke(["msg"], config={"run_name": "test"})

        primary.invoke.assert_called_once_with(["msg"], config={"run_name": "test"})

    def test_second_fallback_tried_after_first_fallback_fails(self):
        """Chain of 3: primary fails, first fallback fails, second succeeds."""
        err = _RetryableError("down")
        p1 = _mock_llm(side_effect=err)
        p2 = _mock_llm(side_effect=err)
        p3 = _mock_llm("p3-ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=p1, fallbacks=[p2, p3], policy=policy)

        result = llm.invoke(["msg"])

        assert result.content == "p3-ok"
        p1.invoke.assert_called_once()
        p2.invoke.assert_called_once()
        p3.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# FallbackLLM.stream()
# ---------------------------------------------------------------------------


class TestFallbackLLMStream:
    """Verify FallbackLLM.stream() fallback semantics."""

    def test_primary_stream_success(self):
        """When primary stream succeeds, fallback is not tried."""
        chunk = MagicMock(content="chunk1")
        primary = MagicMock()
        primary.stream.return_value = iter([chunk])
        fallback = _mock_llm("fallback")
        llm = FallbackLLM(primary=primary, fallbacks=[fallback])

        chunks = list(llm.stream(["msg"]))

        assert len(chunks) == 1
        assert chunks[0].content == "chunk1"
        fallback.stream.assert_not_called()

    def test_stream_retryable_falls_to_fallback(self):
        """Retryable error on stream open triggers fallback."""
        chunk = MagicMock(content="fb-chunk")
        primary = MagicMock()
        primary.stream.side_effect = _RetryableError("stream down")
        fallback = MagicMock()
        fallback.stream.return_value = iter([chunk])
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        chunks = list(llm.stream(["msg"]))

        assert len(chunks) == 1
        assert chunks[0].content == "fb-chunk"

    def test_stream_fatal_does_not_fallback(self):
        """Fatal error on stream propagates immediately."""
        primary = MagicMock()
        primary.stream.side_effect = _FatalError("auth")
        fallback = _mock_llm("ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        with pytest.raises(_FatalError):
            list(llm.stream(["msg"]))

        fallback.stream.assert_not_called()

    def test_all_stream_fail_raises(self):
        """All stream providers fail → last exception raised."""
        err = _RetryableError("all down")
        primary = MagicMock()
        primary.stream.side_effect = err
        fallback = MagicMock()
        fallback.stream.side_effect = err
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        with pytest.raises(_RetryableError):
            list(llm.stream(["msg"]))

    def test_stream_mid_stream_failure_propagates_no_fallback(self):
        """A failure after >= 1 token propagates immediately; fallback NOT tried.

        Once the primary has yielded tokens, falling over to a fallback would
        produce the primary's partial output followed by the fallback's full
        output — corrupted from the caller's perspective.  The exception must
        propagate instead.
        """
        mid_err = _RetryableError("connection reset mid-stream")

        def _partial_stream(*_args, **_kwargs):
            yield MagicMock(content="tok1")
            raise mid_err

        primary = MagicMock()
        primary.stream = _partial_stream
        fallback = _mock_llm("fallback-ok")
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        collected = []
        with pytest.raises(_RetryableError) as exc_info:
            for chunk in llm.stream(["msg"]):
                collected.append(chunk)

        # The one token before the failure was already yielded to the caller.
        assert len(collected) == 1
        assert collected[0].content == "tok1"
        # The raised exception is the original mid-stream error.
        assert exc_info.value is mid_err
        # Fallback was never called.
        fallback.stream.assert_not_called()

    def test_stream_pre_token_retryable_falls_to_fallback(self):
        """A retryable failure before the first token triggers fallback.

        This is the clean start-failure path: the primary raises before yielding
        anything, so the fallback can safely re-stream from the beginning.
        """

        def _fail_before_first_token(*_args, **_kwargs):
            raise _RetryableError("connect timeout")
            yield  # make it a generator  # pylint: disable=unreachable

        primary = MagicMock()
        primary.stream = _fail_before_first_token
        fb_chunk = MagicMock(content="fb-tok")
        fallback = MagicMock()
        fallback.stream.return_value = iter([fb_chunk])
        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
        llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

        chunks = list(llm.stream(["msg"]))

        assert len(chunks) == 1
        assert chunks[0].content == "fb-tok"
        fallback.stream.assert_called_once()


# ---------------------------------------------------------------------------
# FallbackLLM.from_chain()
# ---------------------------------------------------------------------------


class TestFallbackLLMFromChain:
    """Verify FallbackLLM.from_chain() end-to-end construction."""

    def test_from_chain_loads_providers(self):
        """from_chain() loads the primary eagerly and stores fallbacks lazily.

        The primary LLM is loaded at construction time.  Fallback providers
        are stored as specs (chain_length reflects them) but not loaded until
        actually needed.  Invoking the FallbackLLM triggers lazy loading of
        the fallback when the primary fails.
        """
        mock_llm_a = MagicMock(name="llm_a")
        mock_llm_b = MagicMock(name="llm_b")

        class _ProvA(LLMProvider):
            def load(self, **kwargs):
                return mock_llm_a

        class _ProvB(LLMProvider):
            def load(self, **kwargs):
                return mock_llm_b

        type_a = "_test_from_chain_a"
        type_b = "_test_from_chain_b"
        PROVIDER_REGISTRY.register(type_a, _ProvA)
        PROVIDER_REGISTRY.register(type_b, _ProvB)
        try:
            chain = ProviderChain(
                [
                    (type_a, {"model_name": "m1"}),
                    (type_b, {"model_name": "m2"}),
                ]
            )
            fb_llm = FallbackLLM.from_chain(chain)
        finally:
            PROVIDER_REGISTRY.unregister(type_a)
            PROVIDER_REGISTRY.unregister(type_b)

        # Primary was loaded eagerly.
        assert fb_llm.primary is mock_llm_a
        # chain_length counts both primary and the lazy fallback spec.
        assert fb_llm.chain_length == 2
        # fallbacks property returns only pre-loaded LLMs, not lazy specs.
        assert not fb_llm.fallbacks


# ---------------------------------------------------------------------------
# build_fallback_llm()
# ---------------------------------------------------------------------------


class TestBuildFallbackLLM:
    """Verify the build_fallback_llm() convenience function."""

    def test_wraps_primary_with_fallback(self):
        """build_fallback_llm() wraps primary with a lazy fallback spec.

        The returned FallbackLLM has the correct chain_length and is
        functional: when the primary fails, the fallback is loaded and used.
        """
        primary = _mock_llm(side_effect=_RetryableError("rate limited"))
        fallback_llm_obj = _mock_llm("fallback-result")

        class _FallbackProv(LLMProvider):
            def load(self, **kwargs):
                return fallback_llm_obj

        type_key = "_test_build_fb_type"
        PROVIDER_REGISTRY.register(type_key, _FallbackProv)
        try:
            result = build_fallback_llm(
                primary_llm=primary,
                fallback_chain=[(type_key, {"model_name": "m"})],
                policy=FallbackPolicy(retryable_exceptions=(_RetryableError,)),
            )
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert isinstance(result, FallbackLLM)
        assert result.primary is primary
        # chain_length = 1 (primary) + 1 (lazy spec) = 2
        assert result.chain_length == 2
        # fallbacks property returns only pre-loaded LLMs; lazy specs are not listed.
        assert not result.fallbacks
        # Re-register so invoke() can load the lazy spec.
        PROVIDER_REGISTRY.register(type_key, _FallbackProv)
        try:
            response = result.invoke(["hello"])
        finally:
            PROVIDER_REGISTRY.unregister(type_key)
        assert response.content == "fallback-result"

    def test_unknown_fallback_type_raises(self):
        """Unknown provider type in fallback_chain raises ValueError at build time."""
        primary = _mock_llm("primary")
        with pytest.raises(ValueError, match="Invalid model type"):
            build_fallback_llm(
                primary_llm=primary,
                fallback_chain=[("__nonexistent__", {"model_name": "x"})],
            )

    def test_empty_fallback_chain_returns_no_fallback_llm(self):
        """Empty fallback_chain produces a FallbackLLM with no fallbacks."""
        primary = _mock_llm("primary")
        result = build_fallback_llm(primary_llm=primary, fallback_chain=[])
        assert isinstance(result, FallbackLLM)
        assert not result.fallbacks

    def test_misconfigured_fallback_does_not_break_construction(self):
        """A fallback whose load() would fail does not break agent construction.

        The fallback spec is validated (type must be registered) at build time,
        but load() is deferred.  When the primary succeeds, the fallback is
        never loaded and the broken configuration is never surfaced.
        """

        class _BrokenProv(LLMProvider):
            def load(self, **kwargs):
                raise RuntimeError("missing credentials — misconfigured fallback")

        type_key = "_test_broken_fb_type"
        PROVIDER_REGISTRY.register(type_key, _BrokenProv)
        try:
            primary = _mock_llm("primary-ok")
            # Construction must succeed even though the fallback's load() would fail.
            result = build_fallback_llm(
                primary_llm=primary,
                fallback_chain=[(type_key, {"model_name": "broken"})],
            )
            assert isinstance(result, FallbackLLM)
            assert result.chain_length == 2

            # Invoking with a healthy primary must succeed without touching the fallback.
            response = result.invoke(["hello"])
            assert response.content == "primary-ok"
        finally:
            PROVIDER_REGISTRY.unregister(type_key)


# ---------------------------------------------------------------------------
# FallbackLLM introspection
# ---------------------------------------------------------------------------


class TestFallbackLLMIntrospection:
    """Verify FallbackLLM property accessors and repr."""

    def test_primary_property(self):
        """primary property returns the primary LLM."""
        p = _mock_llm()
        llm = FallbackLLM(primary=p)
        assert llm.primary is p

    def test_fallbacks_property_copy(self):
        """fallbacks property returns a copy; mutating it does not affect chain."""
        p = _mock_llm()
        f1 = _mock_llm()
        llm = FallbackLLM(primary=p, fallbacks=[f1])
        copy = llm.fallbacks
        copy.append(_mock_llm())
        assert len(llm.fallbacks) == 1

    def test_chain_length(self):
        """chain_length counts primary + fallbacks."""
        llm = FallbackLLM(primary=_mock_llm(), fallbacks=[_mock_llm(), _mock_llm()])
        assert llm.chain_length == 3

    def test_repr_includes_chain_length(self):
        """repr mentions chain_length."""
        llm = FallbackLLM(primary=_mock_llm(), fallbacks=[_mock_llm()])
        assert "chain_length=2" in repr(llm)


# ---------------------------------------------------------------------------
# AgentSpec.fallback_models field
# ---------------------------------------------------------------------------


class TestAgentSpecFallbackModels:
    """Verify the new fallback_models field on AgentSpec."""

    def _make_spec(self, **kwargs) -> AgentSpec:
        return AgentSpec(
            agent_id="test-agent",
            role="tester",
            objective="Test the fallback field behaviour end to end.",
            **kwargs,
        )

    def test_default_is_empty_list(self):
        """fallback_models defaults to an empty list."""
        spec = self._make_spec()
        assert spec.fallback_models == []

    def test_fallback_models_round_trips(self):
        """fallback_models survives Pydantic round-trip."""
        spec = self._make_spec(
            model_name="gpt-4o",
            fallback_models=["gpt-4-turbo", "gemini-pro"],
        )
        assert spec.fallback_models == ["gpt-4-turbo", "gemini-pro"]

    def test_fallback_models_accepted_in_yaml_dict(self):
        """AgentSpec can be constructed from a dict (as AETHER YAML does)."""
        spec = AgentSpec(
            **{
                "agent_id": "yaml-agent",
                "role": "tester",
                "objective": "Test YAML-dict construction with fallback_models.",
                "model_name": "gpt-4o",
                "fallback_models": ["gpt-4-turbo"],
            }
        )
        assert spec.fallback_models == ["gpt-4-turbo"]

    def test_existing_agentspec_fields_unchanged(self):
        """Adding fallback_models does not change any other field default."""
        spec = self._make_spec(model_name="gpt-4o")
        # These fields must still carry their original defaults.
        assert spec.temperature is None
        assert spec.max_tokens is None
        assert spec.tools == []
        assert spec.capabilities == []


# ---------------------------------------------------------------------------
# create_llm() integration
# ---------------------------------------------------------------------------


class TestCreateLLMIntegration:
    """Verify create_llm() wires the fallback engine correctly."""

    def _make_spec(self, model_name="gpt-4o", fallback_models=None) -> AgentSpec:
        return AgentSpec(
            agent_id="integration-agent",
            role="tester",
            objective="Integration test for create_llm fallback wiring.",
            model_name=model_name,
            fallback_models=fallback_models or [],
        )

    def test_no_fallback_returns_plain_llm(self):
        """create_llm() without fallbacks returns the raw LLM, not FallbackLLM."""
        mock_llm = MagicMock()
        with patch(
            "bili.iris.loaders.llm_loader.load_model", return_value=mock_llm
        ) as mock_load:
            spec = self._make_spec()
            result = create_llm(spec)

        mock_load.assert_called_once()
        assert result is mock_llm
        assert not isinstance(result, FallbackLLM)

    def test_with_fallback_returns_fallback_llm(self):
        """create_llm() with fallback_models returns a FallbackLLM.

        Registers a test provider type in the global PROVIDER_REGISTRY and
        patches _resolve_model_full to route the fallback model name to it,
        then cleans up the global registry on exit.
        """
        primary_llm = MagicMock(name="primary")
        fallback_llm_obj = MagicMock(name="fallback")

        class _FbProv(LLMProvider):
            def load(self, **kwargs):
                return fallback_llm_obj

        type_key = "_test_cllm_fb_type"
        PROVIDER_REGISTRY.register(type_key, _FbProv)
        try:
            with patch(
                "bili.iris.loaders.llm_loader.load_model", return_value=primary_llm
            ):
                with patch(
                    "bili.aether.compiler.llm_resolver._resolve_model_full",
                    side_effect=[
                        ("remote_aws_bedrock", "test-primary-id", {}),
                        (type_key, "test-fallback-id", {}),
                    ],
                ):
                    spec = self._make_spec(
                        model_name="test-primary-id",
                        fallback_models=["test-fallback-id"],
                    )
                    result = create_llm(spec)
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert isinstance(result, FallbackLLM)
        assert result.primary is primary_llm
        # Fallbacks are now lazy specs; chain_length counts them.
        assert result.chain_length == 2

    def test_fallback_resolution_propagates_temperature(self):
        """Temperature from AgentSpec reaches the fallback member's load.

        A fallback member is loaded lazily through the ``load_model`` choke
        point (so it gets the same catalog-derived defaults as the primary),
        not by the bare provider.  The declared temperature must therefore
        reach that ``load_model`` call.  The test triggers a primary failure so
        the lazy fallback is loaded, then inspects the second ``load_model``
        call (the first is the eager primary load).
        """
        primary_llm = MagicMock(name="primary")
        primary_llm.invoke.side_effect = _RetryableError("rate limited")
        fallback_llm_obj = MagicMock(name="fallback")

        class _TrivialProv(LLMProvider):
            # Registered only so build_fallback_llm's registration check passes;
            # the member itself is loaded through the patched load_model.
            def load(self, **kwargs):
                return fallback_llm_obj

        type_key = "_test_cllm_temp_type"
        PROVIDER_REGISTRY.register(type_key, _TrivialProv)
        try:
            with patch(
                "bili.iris.loaders.llm_loader.load_model",
                side_effect=[primary_llm, fallback_llm_obj],
            ) as mock_load:
                with patch(
                    "bili.aether.compiler.llm_resolver._resolve_model_full",
                    side_effect=[
                        ("remote_aws_bedrock", "test-primary-id", {}),
                        (type_key, "test-fallback-id", {}),
                    ],
                ):
                    spec = AgentSpec(
                        agent_id="temp-agent",
                        role="tester",
                        objective="Verify temperature propagates to fallback providers.",
                        model_name="test-primary-id",
                        temperature=0.3,
                        fallback_models=["test-fallback-id"],
                    )
                    result_llm = create_llm(spec)
                    # Trigger a primary failure so the lazy fallback is loaded.
                    policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
                    result_llm._policy = policy  # pylint: disable=protected-access
                    try:
                        result_llm.invoke(["msg"])
                    except _RetryableError:
                        pass  # all-fail path is fine; we just need the load call

        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        # Two load_model calls: [0] eager primary, [1] the lazy fallback member.
        assert mock_load.call_count == 2
        fb_call = mock_load.call_args_list[1]
        assert fb_call.args[0] == type_key
        assert fb_call.kwargs.get("model_name") == "test-fallback-id"
        assert fb_call.kwargs.get("temperature") == 0.3

    def test_create_llm_fallback_invokes_correctly(self):
        """FallbackLLM returned by create_llm() correctly falls back on failure.

        The provider type must remain registered through the invoke() call
        so the lazy fallback spec can be loaded when the primary fails.
        """
        primary_llm = MagicMock(name="primary")
        primary_llm.invoke.side_effect = _RetryableError("rate limited")
        fallback_llm_obj = MagicMock(name="fallback")
        fallback_llm_obj.invoke.return_value = MagicMock(content="fallback-content")

        class _FbProv(LLMProvider):
            def load(self, **kwargs):
                return fallback_llm_obj

        policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))

        import bili.iris.providers.fallback as _fallback_mod  # pylint: disable=import-outside-toplevel

        type_key = "_test_cllm_invoke_type"
        # Keep the type registered through the full test (including invoke).
        PROVIDER_REGISTRY.register(type_key, _FbProv)
        try:
            with patch(
                "bili.iris.loaders.llm_loader.load_model", return_value=primary_llm
            ):
                with patch(
                    "bili.aether.compiler.llm_resolver._resolve_model_full",
                    side_effect=[
                        ("remote_aws_bedrock", "test-primary-id", {}),
                        (type_key, "test-fallback-id", {}),
                    ],
                ):
                    with patch.object(_fallback_mod, "DEFAULT_POLICY", policy):
                        spec = self._make_spec(
                            model_name="test-primary-id",
                            fallback_models=["test-fallback-id"],
                        )
                        result_llm = create_llm(spec)

            # invoke() is outside the load_model/resolve patches but
            # inside the PROVIDER_REGISTRY registration.
            response = result_llm.invoke(["hello"])
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert response.content == "fallback-content"


# ---------------------------------------------------------------------------
# astream (async) smoke test
# ---------------------------------------------------------------------------


class TestFallbackLLMAstream:
    """Verify astream() basic retryable fallback."""

    def test_astream_primary_success(self):
        """astream() yields chunks from primary when primary succeeds."""

        async def _run():
            chunk = MagicMock(content="async-chunk")
            primary = MagicMock()

            async def _astream_ok(*_args, **_kwargs):
                yield chunk

            primary.astream = _astream_ok
            fallback = _mock_llm("fb")
            llm = FallbackLLM(primary=primary, fallbacks=[fallback])
            chunks = [c async for c in llm.astream(["msg"])]
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 1
        assert chunks[0].content == "async-chunk"

    def test_astream_retryable_falls_to_fallback(self):
        """astream() falls back when primary raises a retryable error."""

        async def _run():
            primary = MagicMock()

            async def _astream_fail(*_args, **_kwargs):
                raise _RetryableError("async down")
                yield  # make it a generator  # pylint: disable=unreachable

            chunk = MagicMock(content="fb-async")
            fallback = MagicMock()

            async def _astream_ok(*_args, **_kwargs):
                yield chunk

            primary.astream = _astream_fail
            fallback.astream = _astream_ok
            policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
            llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)
            chunks = [c async for c in llm.astream(["msg"])]
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 1
        assert chunks[0].content == "fb-async"

    def test_astream_mid_stream_failure_propagates_no_fallback(self):
        """astream() mid-stream failure propagates; fallback NOT tried.

        Once the primary has yielded >= 1 token via astream(), any subsequent
        failure must propagate immediately.  Falling over would concatenate the
        primary's partial output with the fallback's full output.
        """
        mid_err = _RetryableError("async connection reset mid-stream")

        async def _run():
            async def _astream_partial(*_args, **_kwargs):
                yield MagicMock(content="async-tok1")
                raise mid_err

            fallback_call_count = [0]

            async def _astream_fb(*_args, **_kwargs):
                fallback_call_count[0] += 1
                yield MagicMock(content="should-not-appear")

            primary = MagicMock()
            primary.astream = _astream_partial
            fallback = MagicMock()
            fallback.astream = _astream_fb
            policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
            llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)

            collected = []
            raised = None
            try:
                async for chunk in llm.astream(["msg"]):
                    collected.append(chunk)
            except _RetryableError as exc:
                raised = exc
            return collected, raised, fallback_call_count[0]

        collected, raised, fb_calls = asyncio.run(_run())

        assert len(collected) == 1
        assert collected[0].content == "async-tok1"
        assert raised is mid_err
        assert fb_calls == 0  # fallback was never entered

    def test_astream_pre_token_retryable_falls_to_fallback(self):
        """astream() pre-token retryable failure falls back cleanly.

        The primary raises before yielding any token, so the fallback can
        safely re-stream from the beginning with no output duplication.
        """

        async def _run():
            async def _astream_fail_early(*_args, **_kwargs):
                raise _RetryableError("async connect timeout")
                yield  # make it a generator  # pylint: disable=unreachable

            fb_chunk = MagicMock(content="async-fb-tok")
            fallback_call_count = [0]

            async def _astream_fb_ok(*_args, **_kwargs):
                fallback_call_count[0] += 1
                yield fb_chunk

            primary = MagicMock()
            primary.astream = _astream_fail_early
            fallback = MagicMock()
            fallback.astream = _astream_fb_ok
            policy = FallbackPolicy(retryable_exceptions=(_RetryableError,))
            llm = FallbackLLM(primary=primary, fallbacks=[fallback], policy=policy)
            chunks = [c async for c in llm.astream(["msg"])]
            return chunks, fallback_call_count[0]

        chunks, fb_calls = asyncio.run(_run())

        assert len(chunks) == 1
        assert chunks[0].content == "async-fb-tok"
        assert fb_calls == 1  # fallback was called exactly once


# ---------------------------------------------------------------------------
# Fallback-member load resilience: members load through the same path as the
# primary (the load_model choke point), not the bare provider.
# ---------------------------------------------------------------------------


class TestFallbackMemberLoaderInjection:
    """A FallbackLLM loads lazy members through an injectable loader."""

    def test_injected_loader_used_for_lazy_member(self):
        """The injected loader (not the registry) loads a lazy fallback spec."""
        primary = MagicMock(name="primary")
        primary.invoke.side_effect = _RetryableError("down")
        loaded = MagicMock(name="loaded-by-injected-loader")
        seen: list = []

        def _loader(provider_type, kwargs):
            seen.append((provider_type, dict(kwargs)))
            return loaded

        llm = FallbackLLM(
            primary=primary,
            fallback_specs=[("some_type", {"model_name": "m", "temperature": 0.2})],
            policy=FallbackPolicy(retryable_exceptions=(_RetryableError,)),
            loader=_loader,
        )
        # The injected loader must not touch the registry; use a type that is
        # not registered, so a registry path would raise instead of returning.
        loaded.invoke.return_value = MagicMock(content="ok")
        result = llm.invoke(["msg"])

        assert result.content == "ok"
        assert seen == [("some_type", {"model_name": "m", "temperature": 0.2})]

    def test_default_loader_is_registry(self):
        """With no loader, a lazy member loads via the registry (_registry_load)."""
        primary = MagicMock(name="primary")
        primary.invoke.side_effect = _RetryableError("down")
        fb_obj = MagicMock(name="fallback")
        fb_obj.invoke.return_value = MagicMock(content="fb")

        class _Prov(LLMProvider):
            def load(self, **kwargs):
                return fb_obj

        type_key = "_test_default_loader_type"
        PROVIDER_REGISTRY.register(type_key, _Prov)
        try:
            llm = FallbackLLM(
                primary=primary,
                fallback_specs=[(type_key, {"model_name": "m"})],
                policy=FallbackPolicy(retryable_exceptions=(_RetryableError,)),
            )
            result = llm.invoke(["msg"])
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert result.content == "fb"

    def test_registry_load_helper(self):
        """_registry_load looks the type up and calls its provider.load."""
        obj = MagicMock(name="obj")

        class _Prov(LLMProvider):
            def load(self, **kwargs):
                return obj

        type_key = "_test_registry_load_helper"
        PROVIDER_REGISTRY.register(type_key, _Prov)
        try:
            assert _registry_load(type_key, {"model_name": "m"}) is obj
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

    def test_build_fallback_llm_forwards_loader(self):
        """build_fallback_llm forwards the loader to the FallbackLLM."""
        primary = MagicMock(name="primary")
        primary.invoke.side_effect = _RetryableError("down")
        loaded = MagicMock(name="loaded")
        loaded.invoke.return_value = MagicMock(content="ok")

        def _loader(provider_type, kwargs):  # noqa: ARG001
            return loaded

        # An unregistered type would fail build_fallback_llm's registration
        # check, so register a trivial provider for the build-time validation.
        class _Prov(LLMProvider):
            def load(self, **kwargs):
                return MagicMock()

        type_key = "_test_build_forwards_loader"
        PROVIDER_REGISTRY.register(type_key, _Prov)
        try:
            llm = build_fallback_llm(
                primary_llm=primary,
                fallback_chain=[(type_key, {"model_name": "m"})],
                policy=FallbackPolicy(retryable_exceptions=(_RetryableError,)),
                loader=_loader,
            )
            result = llm.invoke(["msg"])
        finally:
            PROVIDER_REGISTRY.unregister(type_key)

        assert result.content == "ok"


class TestFallbackMemberResilience:
    """A fallback member receives the same catalog-derived defaults the primary
    does, because it is loaded through the load_model choke point."""

    def test_load_fallback_member_routes_through_load_model(self):
        """_load_fallback_member calls load_model with the member's kwargs."""
        sentinel = MagicMock(name="loaded")
        with patch(
            "bili.iris.loaders.llm_loader.load_model", return_value=sentinel
        ) as mock_load:
            result = _load_fallback_member(
                "remote_anthropic", {"model_name": "m", "temperature": 0.1}
            )
        assert result is sentinel
        mock_load.assert_called_once_with(
            "remote_anthropic", model_name="m", temperature=0.1
        )

    def test_cataloged_anthropic_member_gets_catalog_max_tokens(self):
        """A cataloged Anthropic fallback member with no explicit max_tokens gets
        the catalog budget, not the bare provider's 1024 default.  This is the
        resilience a fallback member missed when loaded by the bare provider."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            _load_fallback_member("remote_anthropic", {"model_name": "claude-sonnet-5"})
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["max_tokens"] == 16000

    def test_uncataloged_anthropic_member_gets_the_floor_not_1024(self):
        """An uncataloged Anthropic fallback member floors at 4096, not 1024."""
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            _load_fallback_member(
                "remote_anthropic", {"model_name": "claude-unlisted-fallback"}
            )
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
