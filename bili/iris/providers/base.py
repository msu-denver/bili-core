"""Abstract base class for bili-core LLM providers.

All provider implementations — LangChain-native API chat models,
subprocess/CLI providers, and MCP-backed providers — inherit from
``LLMProvider`` and implement :meth:`load`.

The returned object from :meth:`load` must expose an ``.invoke(messages)``
method that accepts a list of ``langchain_core.messages.BaseMessage`` objects
and returns a response with a ``.content`` attribute.  Concrete providers
need not subclass any LangChain base class; the duck-typed ``.invoke()``
contract is sufficient for compatibility with both IRIS node pipelines and
AETHER agent nodes.

Provider type strings
---------------------
Each concrete subclass is identified by a *provider type string* — a plain
str used as the key in :class:`~bili.iris.providers.registry.ProviderRegistry`
and in ``LLM_MODELS`` catalog entries.  Established values are listed in
:data:`KNOWN_PROVIDER_TYPES` for reference; implementations are not
constrained to that list.
"""

from abc import ABC, abstractmethod
from typing import Any

# ProviderRegistry uses these strings as keys.  The list is informational;
# the registry itself is the authoritative source of which types are active.
KNOWN_PROVIDER_TYPES = frozenset(
    {
        # LangChain-native remote API providers (original)
        "remote_aws_bedrock",
        "remote_google_vertex",
        "remote_azure_openai",
        "remote_openai",
        # LangChain-native remote API providers (expanded catalog)
        "remote_anthropic",
        "remote_mistral",
        "remote_cohere",
        "remote_google_genai",
        "remote_deepseek",
        "remote_xai",
        "remote_groq",
        # Local in-process providers
        "local_llamacpp",
        "local_huggingface",
        # Local server provider (Ollama daemon over HTTP)
        "local_ollama",
        # Subprocess / transport-level providers
        "cli",
        # Named CLI presets (each preset is a CliPresetProvider subclass)
        "cli_claude_code",
        "cli_codex",
        "cli_gemini_cli",
        # Future provider shapes (not yet implemented)
        # "mcp"   -- Model Context Protocol server transport
    }
)


# pylint: disable=too-few-public-methods
class LLMProvider(ABC):
    """Abstract base class for all bili-core LLM providers.

    Subclasses represent a single *provider shape* — a transport mechanism
    and credential strategy for obtaining LLM completions.  Concrete
    examples: an AWS Bedrock chat model, a local LlamaCpp model, a future
    subprocess-based CLI provider, or a future MCP-server-backed provider.

    Each concrete provider intentionally exposes only the single ``load()``
    method; providers are lightweight factory objects, not service classes.

    Usage
    -----
    Subclasses are not instantiated per-call.  Instead the registry maps a
    provider-type string to the *class*, and :meth:`load` is called via
    ``ProviderClass().load(**kwargs)``::

        from bili.iris.providers import get_provider

        ProviderClass = get_provider("remote_aws_bedrock")
        llm = ProviderClass().load(model_name="...", max_tokens=4096)

    For typical usage prefer :func:`bili.iris.loaders.llm_loader.load_model`,
    which handles provider lookup and delegation automatically.

    Implementing a new provider
    ---------------------------
    1. Create a module under ``bili/iris/providers/`` (e.g.
       ``anthropic_provider.py``).
    2. Subclass ``LLMProvider`` and implement :meth:`load`.
    3. Import any heavy SDK *inside* :meth:`load` (never at module scope).
    4. Register the provider at startup::

           from bili.iris.providers import register_provider
           from bili.iris.providers.my_provider import MyProvider

           register_provider("remote_my_api", MyProvider)

    Parameter signature
    -------------------
    Concrete overrides of :meth:`load` SHOULD declare explicit keyword
    parameters rather than relying solely on ``**kwargs``.  This makes
    accepted parameters discoverable via introspection and IDE tooling.
    Pylint ``arguments-differ`` warnings are suppressed on concrete
    subclasses because the abstract signature intentionally uses only
    ``**kwargs`` as the lowest-common-denominator contract.
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> Any:
        """Instantiate and return an LLM client object.

        Implementations MUST import any heavy dependencies (cloud SDKs,
        ``torch``, etc.) inside this method body rather than at module scope.

        :param kwargs: Provider-specific keyword arguments (e.g.
            ``model_name``, ``max_tokens``, ``temperature``).  Each provider
            documents its accepted kwargs in its own docstring.
        :returns: An object with an ``.invoke(messages)`` method that accepts
            a list of ``langchain_core.messages.BaseMessage`` objects and
            returns a response with a ``.content`` attribute.  The object
            may additionally expose ``.stream()`` and ``.astream()`` for
            streaming support.
        :raises ValueError: For invalid or missing required kwargs.
        :raises ImportError: If a required optional dependency is not
            installed.
        """
