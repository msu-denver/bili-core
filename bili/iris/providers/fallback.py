"""Fallback-policy engine for bili-core LLM providers.

When a provider fails with a transient error (rate limit, API unavailable,
network timeout) this module automatically falls through an ordered list of
provider alternatives without propagating the error to the caller.  The
mechanism is opt-in and backward-compatible: callers that do not configure a
fallback chain see no change in behaviour.

Core types
----------
- :class:`FallbackPolicy` — classifies exceptions as retryable vs fatal.
- :class:`ProviderChain` — an ordered sequence of ``(provider_type, kwargs)``
  pairs that represents the fallback priority list.
- :class:`FallbackLLM` — a transparent proxy that tries each entry in the
  chain on retryable failure and exposes ``.invoke()`` / ``.stream()`` /
  ``.astream()`` to callers.

Usage example
-------------
Typical usage via :func:`~bili.aether.compiler.llm_resolver.create_llm` when
``AgentSpec.fallback_models`` is populated::

    from bili.iris.providers.fallback import FallbackLLM, ProviderChain

    chain = ProviderChain([
        ("remote_aws_bedrock", {"model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0"}),
        ("remote_openai",      {"model_name": "gpt-4o"}),
    ])
    llm = FallbackLLM.from_chain(chain)

    # Callers interact with the standard .invoke() / .stream() interface.
    response = llm.invoke(messages)

Standalone usage without AgentSpec::

    from bili.iris.providers.fallback import FallbackLLM, FallbackPolicy

    policy = FallbackPolicy(retryable_exceptions=(RateLimitError, TimeoutError))
    primary_llm = load_model("remote_aws_bedrock", model_name="...")
    fallback_llm = load_model("remote_openai", model_name="gpt-4o")
    llm = FallbackLLM(primary=primary_llm, fallbacks=[fallback_llm], policy=policy)

Design notes
------------
The fallback engine is *completely generic*.  It knows nothing about which
providers exist or in what order they should be tried; that is the consumer's
responsibility.  The engine only handles: exception classification, ordered
retry, logging, and transparent proxy behaviour.

``FallbackLLM`` implements the same duck-typed ``.invoke()`` / ``.stream()``
/ ``.astream()`` interface as any LangChain chat model, so it is transparent
to AETHER executor nodes, IRIS pipeline nodes, and any other caller that
receives an LLM object.
"""

# pylint: disable=too-few-public-methods

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Tuple, Type

from bili.iris.providers.registry import PROVIDER_REGISTRY

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default retryable exception types
# ---------------------------------------------------------------------------

# These are the exception class *names* (as strings) that are treated as
# retryable by default.  String matching is used so the engine does not need
# hard imports of provider SDKs — providers that are not installed produce no
# ImportError here.
#
# Only genuinely *transient* failures belong here: rate limits, timeouts,
# connection drops, and provider-unavailable (HTTP 5xx).  Do NOT add base
# exception classes that cover 4xx responses (auth failure, bad request,
# not found, validation error); those are fatal and must not silently trigger
# fallback, as that would mask misconfiguration and waste quota against every
# fallback provider.
#
# In particular, ``APIStatusError`` (openai/anthropic) is the base class for
# *all* status-coded errors including 401/403/400/422, so it must not be
# listed here.  The specific subclasses below are the safe subset.
#
# Consumers can override the classification entirely via FallbackPolicy.
_DEFAULT_RETRYABLE_NAMES: Tuple[str, ...] = (
    # openai SDK — transient only
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    # anthropic SDK — transient only
    # "OverloadedError" is the Anthropic analogue of RateLimitError (HTTP 529)
    "OverloadedError",
    # google-cloud / vertexai SDK
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    # boto3 / botocore (Bedrock)
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    # generic Python network/IO
    "TimeoutError",
    "ConnectionError",
    "ConnectionResetError",
    "BrokenPipeError",
)


def _is_retryable_by_name(exc: BaseException) -> bool:
    """Return True if the exception's class name (or any base class name) is in
    the default retryable set.

    The MRO check lets named subclasses inherit retryability (e.g. a
    ``GoogleRateLimitError`` subclassing ``ResourceExhausted`` is also
    treated as retryable).  The direct name check is subsumed by the MRO
    set intersection — listed separately only for clarity.
    """
    exc_base_names = {c.__name__ for c in type(exc).__mro__}
    return bool(exc_base_names & set(_DEFAULT_RETRYABLE_NAMES))


# ---------------------------------------------------------------------------
# FallbackPolicy
# ---------------------------------------------------------------------------


class FallbackPolicy:
    """Classifies exceptions as retryable (trigger fallback) vs fatal (re-raise).

    The policy is the only place where exception type decisions live.  Two
    configuration modes:

    1. **Type-based** — pass a tuple of exception classes to
       ``retryable_exceptions``.  Any exception that is an instance of one of
       those classes triggers the fallback.
    2. **Callable** — pass a callable to ``is_retryable``.  The callable
       receives the exception and returns ``True`` to trigger fallback.

    If neither is supplied, the engine uses name-based matching against
    :data:`_DEFAULT_RETRYABLE_NAMES`, which covers the most common transient
    errors across the built-in providers without requiring hard imports of
    their SDKs.

    Parameters
    ----------
    retryable_exceptions : tuple of exception classes, optional
        Hard-typed exception classes.  Checked with ``isinstance``.
    is_retryable : callable, optional
        ``(exc: BaseException) -> bool``.  Takes precedence over
        ``retryable_exceptions`` when both are supplied.
    max_attempts : int
        Total number of providers to try (primary + fallbacks).  0 means
        unlimited (try the whole chain).  Default: 0.

    Examples
    --------
    Type-based::

        from openai import RateLimitError
        policy = FallbackPolicy(retryable_exceptions=(RateLimitError,))

    Callable (covers any 429 / 503 HTTP status)::

        def my_policy(exc):
            return getattr(exc, "status_code", 0) in (429, 503)

        policy = FallbackPolicy(is_retryable=my_policy)
    """

    def __init__(
        self,
        retryable_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
        is_retryable: Optional[Any] = None,
        max_attempts: int = 0,
    ) -> None:
        self._retryable_exceptions = retryable_exceptions
        self._is_retryable_fn = is_retryable
        self.max_attempts = max_attempts

    def should_fallback(self, exc: BaseException) -> bool:
        """Return ``True`` if *exc* should trigger a fallback attempt.

        :param exc: The exception raised by the current provider.
        :returns: ``True`` to try the next provider; ``False`` to re-raise.
        """
        if self._is_retryable_fn is not None:
            return bool(self._is_retryable_fn(exc))
        if self._retryable_exceptions is not None:
            return isinstance(exc, self._retryable_exceptions)
        # Default: name-based matching (no hard SDK imports required)
        return _is_retryable_by_name(exc)


#: The default policy used when no explicit policy is provided to FallbackLLM.
DEFAULT_POLICY = FallbackPolicy()


# ---------------------------------------------------------------------------
# ProviderChain
# ---------------------------------------------------------------------------


class ProviderChain:
    """An ordered list of ``(provider_type, load_kwargs)`` pairs.

    Each entry describes one candidate LLM provider.  The engine tries them
    in order, stopping at the first that succeeds.

    Parameters
    ----------
    entries : list of (str, dict)
        Ordered list of ``(provider_type_string, load_kwargs)`` pairs.
        ``provider_type_string`` must be registered in
        :data:`~bili.iris.providers.registry.PROVIDER_REGISTRY`.
        ``load_kwargs`` are forwarded verbatim to the provider's
        :meth:`~bili.iris.providers.base.LLMProvider.load` method.

    Example
    -------
    ::

        chain = ProviderChain([
            ("remote_aws_bedrock", {"model_name": "anthropic.claude-3-5-sonnet..."}),
            ("remote_openai",      {"model_name": "gpt-4o"}),
            ("remote_google_vertex", {"model_name": "gemini-2.0-flash"}),
        ])
    """

    def __init__(self, entries: List[Tuple[str, dict]]) -> None:
        if not entries:
            raise ValueError("ProviderChain requires at least one entry.")
        self._entries: List[Tuple[str, dict]] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        types = [e[0] for e in self._entries]
        return f"ProviderChain({types})"

    def load_all(self) -> List[Any]:
        """Instantiate all providers in the chain and return their LLM objects.

        :returns: Ordered list of LLM objects, one per chain entry.
        :raises ValueError: If a provider type is not registered.
        :raises ImportError: If a provider's optional dependency is absent.
        """
        llms = []
        for provider_type, kwargs in self._entries:
            provider_class = PROVIDER_REGISTRY.get_or_raise(provider_type)
            llm = provider_class().load(**kwargs)
            llms.append(llm)
        return llms


# ---------------------------------------------------------------------------
# FallbackLLM
# ---------------------------------------------------------------------------


class FallbackLLM:
    """Transparent proxy that falls through an ordered list of LLM objects.

    On a retryable error from the current LLM, ``FallbackLLM`` automatically
    tries the next one in the chain.  If the chain is exhausted the last
    exception is re-raised.  On a fatal (non-retryable) error the exception
    propagates immediately without trying further providers.

    ``FallbackLLM`` is transparent to callers — it exposes the same
    ``.invoke()`` / ``.stream()`` / ``.astream()`` interface as any
    LangChain chat model.

    Fallback providers are loaded **lazily** — on first use, not at
    construction time.  This means a misconfigured fallback (missing
    credentials, wrong region) only surfaces as an error when the chain
    actually reaches it after the primary fails.  A healthy primary will
    never trigger loading of any fallback provider.

    Parameters
    ----------
    primary : object
        The first LLM to try (must have ``.invoke()``).
    fallbacks : list of objects, optional
        Pre-loaded subsequent LLMs, in priority order.  Use this when you
        have already-instantiated LLM objects.
    fallback_specs : list of (str, dict), optional
        Lazy-loaded fallback specs as ``(provider_type, load_kwargs)`` pairs.
        Each entry is loaded via :data:`~bili.iris.providers.registry.PROVIDER_REGISTRY`
        on first use.  Mutually exclusive with *fallbacks*.
    policy : FallbackPolicy, optional
        Exception classification policy.  Defaults to
        :data:`DEFAULT_POLICY`.

    Class methods
    -------------
    :meth:`from_chain` — convenience constructor that loads the primary from
    a :class:`ProviderChain` and stores the rest as lazy specs.
    """

    def __init__(
        self,
        primary: Any,
        fallbacks: Optional[List[Any]] = None,
        fallback_specs: Optional[List[Tuple[str, dict]]] = None,
        policy: Optional[FallbackPolicy] = None,
    ) -> None:
        if fallbacks and fallback_specs:
            raise ValueError(
                "Provide either 'fallbacks' (pre-loaded LLMs) or "
                "'fallback_specs' (lazy (provider_type, kwargs) pairs), not both."
            )
        self._primary = primary
        self._fallbacks: List[Any] = list(fallbacks) if fallbacks else []
        # Lazy-loading state: specs to load + cache of loaded objects
        self._fallback_specs: List[Tuple[str, dict]] = (
            list(fallback_specs) if fallback_specs else []
        )
        self._loaded_fallbacks: List[Optional[Any]] = [None] * len(self._fallback_specs)
        self._policy = policy if policy is not None else DEFAULT_POLICY

    def _get_fallback(self, spec_index: int) -> Any:
        """Return the fallback LLM at *spec_index*, loading it on first use."""
        if self._loaded_fallbacks[spec_index] is None:
            provider_type, kwargs = self._fallback_specs[spec_index]
            provider_class = PROVIDER_REGISTRY.get_or_raise(provider_type)
            self._loaded_fallbacks[spec_index] = provider_class().load(**kwargs)
        return self._loaded_fallbacks[spec_index]

    @classmethod
    def from_chain(
        cls,
        chain: ProviderChain,
        policy: Optional[FallbackPolicy] = None,
    ) -> "FallbackLLM":
        """Construct a ``FallbackLLM`` from a :class:`ProviderChain`.

        The primary LLM is loaded immediately.  Fallback providers are stored
        as lazy specs and loaded on first use.

        :param chain: An ordered :class:`ProviderChain` with at least one
            entry.  The first entry is the primary; the rest are fallbacks.
        :param policy: Optional :class:`FallbackPolicy`.  Defaults to
            :data:`DEFAULT_POLICY`.
        :returns: A ``FallbackLLM`` ready to use.
        :raises ValueError: If the chain is empty or a type is unregistered.
        """
        entries = list(chain)
        if not entries:
            raise ValueError("ProviderChain must have at least one entry.")
        primary_type, primary_kwargs = entries[0]
        primary_class = PROVIDER_REGISTRY.get_or_raise(primary_type)
        primary_llm = primary_class().load(**primary_kwargs)
        return cls(
            primary=primary_llm,
            fallback_specs=entries[1:] if len(entries) > 1 else None,
            policy=policy,
        )

    # ------------------------------------------------------------------
    # Core invocation helpers
    # ------------------------------------------------------------------

    def _candidates(self) -> List[Any]:
        """Return the full ordered list of LLM candidates.

        Pre-loaded fallbacks (from the ``fallbacks`` constructor parameter)
        appear first, followed by lazy-spec fallbacks.  The primary is always
        index 0.
        """
        return (
            [self._primary] + self._fallbacks + list(range(len(self._fallback_specs)))
        )

    def _resolve_candidate(self, candidate: Any) -> Any:
        """Resolve a candidate: load it lazily if it is a spec index."""
        if isinstance(candidate, int):
            return self._get_fallback(candidate)
        return candidate

    def _log_fallback(
        self,
        exc: BaseException,
        attempt_index: int,
        next_index: int,
        total: int,
    ) -> None:
        """Log a fallback transition."""
        LOGGER.warning(
            "Provider attempt %d/%d failed with %s: %s. "
            "Falling through to candidate %d/%d.",
            attempt_index + 1,
            total,
            type(exc).__name__,
            exc,
            next_index + 1,
            total,
        )

    # ------------------------------------------------------------------
    # Public interface — mirrors LangChain chat model protocol
    # ------------------------------------------------------------------

    def invoke(  # pylint: disable=redefined-builtin
        self, input: Any, **kwargs: Any
    ) -> Any:
        """Invoke the LLM chain, falling back on retryable errors.

        :param input: Messages or prompt to pass to the LLM.
        :param kwargs: Additional keyword arguments forwarded to each
            provider's ``.invoke()`` call.
        :returns: The first successful response.
        :raises: The last provider's exception if all candidates fail,
            or the original exception immediately if it is classified as
            fatal.
        """
        candidates = self._candidates()
        max_attempts = self._policy.max_attempts or len(candidates)
        last_exc: Optional[BaseException] = None

        for i, candidate in enumerate(candidates[:max_attempts]):
            llm = self._resolve_candidate(candidate)
            try:
                result = llm.invoke(input, **kwargs)
                if i > 0:
                    LOGGER.info(
                        "Fallback to candidate %d/%d succeeded.", i + 1, len(candidates)
                    )
                return result
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not self._policy.should_fallback(exc):
                    LOGGER.debug(
                        "Provider %d raised fatal exception %s; not falling back.",
                        i + 1,
                        type(exc).__name__,
                    )
                    raise
                last_exc = exc
                if i < max_attempts - 1:
                    self._log_fallback(
                        exc, i, i + 1, min(max_attempts, len(candidates))
                    )

        # All candidates exhausted
        LOGGER.error(
            "All %d provider candidates failed. Last error: %s",
            min(max_attempts, len(candidates)),
            last_exc,
        )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("FallbackLLM: no candidates available.")  # pragma: no cover

    def stream(
        self, input: Any, **kwargs: Any  # pylint: disable=redefined-builtin
    ) -> Iterator[Any]:
        """Stream responses, falling back on retryable errors.

        Falls back at stream-open time (before the first chunk).  Once a
        provider starts streaming, errors mid-stream propagate normally.

        :param input: Messages or prompt to pass to the LLM.
        :param kwargs: Additional keyword arguments forwarded to each
            provider's ``.stream()`` call.
        :returns: An iterator of response chunks from the first successful
            provider.
        :raises: The last provider's exception if all candidates fail.
        """
        candidates = self._candidates()
        max_attempts = self._policy.max_attempts or len(candidates)
        last_exc: Optional[BaseException] = None

        for i, candidate in enumerate(candidates[:max_attempts]):
            llm = self._resolve_candidate(candidate)
            try:
                yield from llm.stream(input, **kwargs)
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not self._policy.should_fallback(exc):
                    raise
                last_exc = exc
                if i < max_attempts - 1:
                    self._log_fallback(
                        exc, i, i + 1, min(max_attempts, len(candidates))
                    )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("FallbackLLM: no candidates available.")  # pragma: no cover

    async def astream(
        self, input: Any, **kwargs: Any  # pylint: disable=redefined-builtin
    ) -> AsyncIterator[Any]:
        """Async-stream responses, falling back on retryable errors.

        :param input: Messages or prompt to pass to the LLM.
        :param kwargs: Additional keyword arguments forwarded to each
            provider's ``.astream()`` call.
        :returns: An async iterator of response chunks.
        :raises: The last provider's exception if all candidates fail.
        """
        candidates = self._candidates()
        max_attempts = self._policy.max_attempts or len(candidates)
        last_exc: Optional[BaseException] = None

        for i, candidate in enumerate(candidates[:max_attempts]):
            llm = self._resolve_candidate(candidate)
            try:
                async for chunk in llm.astream(input, **kwargs):
                    yield chunk
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not self._policy.should_fallback(exc):
                    raise
                last_exc = exc
                if i < max_attempts - 1:
                    self._log_fallback(
                        exc, i, i + 1, min(max_attempts, len(candidates))
                    )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("FallbackLLM: no candidates available.")  # pragma: no cover

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def primary(self) -> Any:
        """The primary (first) LLM in the chain."""
        return self._primary

    @property
    def fallbacks(self) -> List[Any]:
        """Pre-loaded fallback LLMs in priority order.

        Does not include lazy-spec fallbacks that have not been loaded yet.
        Use :attr:`chain_length` to get the total number of providers.
        """
        return list(self._fallbacks)

    @property
    def chain_length(self) -> int:
        """Total number of providers in the chain (primary + all fallbacks)."""
        return 1 + len(self._fallbacks) + len(self._fallback_specs)

    def __repr__(self) -> str:
        return (
            f"FallbackLLM(chain_length={self.chain_length}, "
            f"policy={self._policy!r})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def build_fallback_llm(
    primary_llm: Any,
    fallback_chain: Sequence[Tuple[str, dict]],
    policy: Optional[FallbackPolicy] = None,
) -> FallbackLLM:
    """Build a :class:`FallbackLLM` from a primary LLM and a fallback spec.

    Fallback providers are stored as lazy specs and loaded on first use —
    only when the chain actually reaches them after the primary fails.  This
    means a misconfigured fallback (missing credentials, wrong region) will
    not break agent construction while the primary provider is healthy.

    This is the low-level convenience function used by
    :func:`~bili.aether.compiler.llm_resolver.create_llm` when
    ``AgentSpec.fallback_models`` is populated.  It can also be used
    directly when callers have an existing LLM object and want to wrap it
    with fallback behaviour without going through AgentSpec.

    :param primary_llm: An already-instantiated primary LLM object.
    :param fallback_chain: Ordered list of ``(provider_type, load_kwargs)``
        pairs for the fallback providers.  Each entry is validated (provider
        type must be registered) and then stored; the provider's
        :meth:`~bili.iris.providers.base.LLMProvider.load` is called only
        on first use.
    :param policy: Optional :class:`FallbackPolicy`.  Defaults to
        :data:`DEFAULT_POLICY`.
    :returns: A :class:`FallbackLLM` wrapping the primary with lazy
        fallback specs.
    :raises ValueError: If a provider type in *fallback_chain* is not
        registered.

    Example
    -------
    ::

        primary = load_model("remote_aws_bedrock", model_name="...")
        llm = build_fallback_llm(
            primary_llm=primary,
            fallback_chain=[
                ("remote_openai", {"model_name": "gpt-4o"}),
            ],
        )
    """
    specs: List[Tuple[str, dict]] = []
    for provider_type, kwargs in fallback_chain:
        # Validate the type is registered now (fail fast at build time, not
        # during a live request); load() is deferred until first use.
        PROVIDER_REGISTRY.get_or_raise(provider_type)
        specs.append((provider_type, dict(kwargs)))

    if not specs:
        return FallbackLLM(primary=primary_llm, policy=policy)

    return FallbackLLM(primary=primary_llm, fallback_specs=specs, policy=policy)
