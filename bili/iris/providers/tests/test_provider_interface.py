"""Tests for the bili-core provider abstraction layer.

Covers:
- ``LLMProvider`` abstract base class contract
- ``ProviderRegistry`` registration, lookup, error paths
- ``builtin`` registration (idempotency and completeness)
- Each concrete provider's ``load()`` method with mocked heavy dependencies
- ``load_model()`` routing through the registry (backward-compat layer)
- Module-level convenience functions (``register_provider``, ``get_provider``)
"""

# pylint: disable=too-few-public-methods,duplicate-code

from unittest.mock import MagicMock, patch

import pytest

import bili.iris.providers.builtin  # noqa: F401  pylint: disable=unused-import
from bili.iris.providers.azure_openai_provider import AzureOpenAIProvider
from bili.iris.providers.base import KNOWN_PROVIDER_TYPES, LLMProvider
from bili.iris.providers.bedrock_provider import BedrockProvider
from bili.iris.providers.huggingface_provider import HuggingFaceProvider
from bili.iris.providers.llamacpp_provider import LlamaCppProvider
from bili.iris.providers.openai_provider import OpenAIProvider
from bili.iris.providers.registry import (
    PROVIDER_REGISTRY,
    ProviderRegistry,
    get_provider,
    register_provider,
)
from bili.iris.providers.vertex_provider import VertexAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_provider(name: str = "TestProvider") -> type:
    """Return a minimal concrete LLMProvider subclass for test use."""

    class _P(LLMProvider):
        def load(self, **kwargs):
            return MagicMock()

    _P.__name__ = name
    _P.__qualname__ = name
    return _P


# ---------------------------------------------------------------------------
# LLMProvider — abstract base contract
# ---------------------------------------------------------------------------


class TestLLMProviderAbstractBase:
    """Verify the abstract base cannot be instantiated and enforces load()."""

    def test_cannot_instantiate_abstract_base(self):
        """Verify direct instantiation of LLMProvider raises TypeError."""
        with pytest.raises(TypeError):
            LLMProvider()  # pylint: disable=abstract-class-instantiated

    def test_concrete_subclass_without_load_raises(self):
        """Verify a subclass missing load() cannot be instantiated."""

        class IncompleteProvider(LLMProvider):
            """Incomplete provider missing load()."""

        with pytest.raises(TypeError):
            IncompleteProvider()  # pylint: disable=abstract-class-instantiated

    def test_concrete_subclass_with_load_instantiates(self):
        """Verify a complete concrete subclass can be instantiated."""
        provider = _minimal_provider()()
        assert isinstance(provider, LLMProvider)

    def test_load_returns_invokeable_object(self):
        """Verify load() returns an object with .invoke()."""

        class InvokeProvider(LLMProvider):
            """Minimal provider whose load() returns an invokeable mock."""

            def load(self, **kwargs):
                mock_llm = MagicMock()
                mock_llm.invoke = MagicMock(return_value=MagicMock(content="ok"))
                return mock_llm

        llm = InvokeProvider().load(model_name="test")
        result = llm.invoke([])
        assert result.content == "ok"

    def test_known_provider_types_is_frozenset(self):
        """Verify KNOWN_PROVIDER_TYPES is an immutable frozenset."""
        assert isinstance(KNOWN_PROVIDER_TYPES, frozenset)
        assert "remote_aws_bedrock" in KNOWN_PROVIDER_TYPES
        assert "remote_google_vertex" in KNOWN_PROVIDER_TYPES
        assert "remote_openai" in KNOWN_PROVIDER_TYPES
        assert "local_llamacpp" in KNOWN_PROVIDER_TYPES
        assert "local_huggingface" in KNOWN_PROVIDER_TYPES


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    """Verify ProviderRegistry registration, lookup, and error paths."""

    def test_register_and_get(self):
        """Verify register() + get() round-trips correctly."""
        registry = ProviderRegistry()
        cls = _minimal_provider()
        registry.register("test_type", cls)
        assert registry.get("test_type") is cls

    def test_get_unknown_returns_none(self):
        """Verify get() returns None for unregistered types."""
        registry = ProviderRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_raise_raises_for_unknown(self):
        """Verify get_or_raise() raises ValueError for unknown types."""
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="Invalid model type"):
            registry.get_or_raise("nonexistent")

    def test_get_or_raise_returns_class_for_known(self):
        """Verify get_or_raise() returns the class for known types."""
        registry = ProviderRegistry()
        cls = _minimal_provider()
        registry.register("known_type", cls)
        assert registry.get_or_raise("known_type") is cls

    def test_duplicate_registration_raises(self):
        """Verify registering the same type twice raises ValueError."""
        registry = ProviderRegistry()
        cls = _minimal_provider()
        registry.register("dupe_type", cls)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dupe_type", cls)

    def test_unregister_removes_entry(self):
        """Verify unregister() removes the provider."""
        registry = ProviderRegistry()
        cls = _minimal_provider()
        registry.register("to_remove", cls)
        registry.unregister("to_remove")
        assert registry.get("to_remove") is None

    def test_unregister_unknown_raises_key_error(self):
        """Verify unregistering a non-existent type raises KeyError."""
        registry = ProviderRegistry()
        with pytest.raises(KeyError):
            registry.unregister("not_there")

    def test_register_non_provider_raises_type_error(self):
        """Verify registering a non-LLMProvider class raises TypeError."""
        registry = ProviderRegistry()
        with pytest.raises(TypeError, match="subclass of LLMProvider"):
            registry.register("bad", str)

    def test_contains_operator(self):
        """Verify 'in' operator works on registry."""
        registry = ProviderRegistry()
        cls = _minimal_provider()
        registry.register("check_type", cls)
        assert "check_type" in registry
        assert "other_type" not in registry

    def test_len(self):
        """Verify len() reflects number of registered providers."""
        registry = ProviderRegistry()
        assert len(registry) == 0
        cls = _minimal_provider()
        registry.register("t1", cls)
        assert len(registry) == 1

    def test_list_types_sorted(self):
        """Verify list_types() returns sorted provider type strings."""
        registry = ProviderRegistry()
        registry.register("b_type", _minimal_provider("B"))
        registry.register("a_type", _minimal_provider("A"))
        assert registry.list_types() == ["a_type", "b_type"]

    def test_repr(self):
        """Verify repr includes the sorted type list."""
        registry = ProviderRegistry()
        registry.register("repr_type", _minimal_provider())
        assert "repr_type" in repr(registry)


# ---------------------------------------------------------------------------
# Builtin registration
# ---------------------------------------------------------------------------

_EXPECTED_BUILTIN_TYPES = {
    "remote_aws_bedrock",
    "remote_google_vertex",
    "remote_azure_openai",
    "remote_openai",
    "local_llamacpp",
    "local_huggingface",
}


class TestBuiltinRegistration:
    """Verify all six built-in providers are registered in PROVIDER_REGISTRY."""

    def test_all_builtins_registered(self):
        """Verify all six built-in provider types are in PROVIDER_REGISTRY."""
        for provider_type in _EXPECTED_BUILTIN_TYPES:
            assert (
                provider_type in PROVIDER_REGISTRY
            ), f"Expected '{provider_type}' in PROVIDER_REGISTRY"

    def test_builtin_providers_are_llmprovider_subclasses(self):
        """Verify every registered built-in is an LLMProvider subclass."""
        for provider_type in _EXPECTED_BUILTIN_TYPES:
            cls = PROVIDER_REGISTRY.get(provider_type)
            assert cls is not None
            assert issubclass(
                cls, LLMProvider
            ), f"Provider '{provider_type}' → {cls} is not an LLMProvider subclass"

    def test_double_import_does_not_raise(self):
        """Verify importing builtin twice does not raise ValueError."""
        # The import at the top of this file already triggered registration.
        # A second import must be idempotent.
        import importlib  # pylint: disable=import-outside-toplevel

        importlib.import_module("bili.iris.providers.builtin")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Verify module-level register_provider() and get_provider()."""

    def test_get_provider_returns_none_for_unknown(self):
        """Verify get_provider() returns None for unregistered type."""
        result = get_provider("__definitely_not_registered__")
        assert result is None

    def test_get_provider_returns_class_for_registered(self):
        """Verify get_provider() returns the class for a registered type."""
        cls = get_provider("remote_openai")
        assert cls is not None
        assert issubclass(cls, LLMProvider)

    def test_register_provider_and_retrieve(self):
        """Verify register_provider() adds to PROVIDER_REGISTRY."""
        cls = _minimal_provider("ConvenienceProvider")
        register_provider("test_convenience_type", cls)
        try:
            retrieved = get_provider("test_convenience_type")
            assert retrieved is cls
        finally:
            PROVIDER_REGISTRY.unregister("test_convenience_type")


# ---------------------------------------------------------------------------
# Concrete providers — mocked load() calls
# ---------------------------------------------------------------------------


class TestBedrockProvider:
    """Verify BedrockProvider.load() constructs ChatBedrockConverse correctly."""

    def test_minimal_load(self):
        """Verify minimal config calls ChatBedrockConverse with model_id only."""
        mock_cls = MagicMock()
        with patch("langchain_aws.ChatBedrockConverse", mock_cls):
            BedrockProvider().load(model_name="test-model")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_id"] == "test-model"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatBedrockConverse."""
        mock_cls = MagicMock()
        with patch("langchain_aws.ChatBedrockConverse", mock_cls):
            BedrockProvider().load(
                model_name="claude-v2",
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                seed=42,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_id"] == "claude-v2"
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40
        assert kwargs["seed"] == 42

    def test_extra_kwargs_ignored(self):
        """Verify unexpected kwargs do not raise errors."""
        mock_cls = MagicMock()
        with patch("langchain_aws.ChatBedrockConverse", mock_cls):
            BedrockProvider().load(model_name="m", unknown_extra="ignored")
        assert mock_cls.called


class TestVertexAIProvider:
    """Verify VertexAIProvider.load() constructs ChatVertexAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model_name to ChatVertexAI."""
        mock_cls = MagicMock()
        with patch("langchain_google_vertexai.ChatVertexAI", mock_cls):
            VertexAIProvider().load(model_name="gemini-pro")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_name"] == "gemini-pro"

    def test_all_optional_params(self):
        """Verify every optional param is forwarded to ChatVertexAI."""
        mock_cls = MagicMock()
        schema = {"type": "object"}
        with patch("langchain_google_vertexai.ChatVertexAI", mock_cls):
            VertexAIProvider().load(
                model_name="gemini-pro",
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
                top_k=20,
                seed=3,
                response_mime_type="application/json",
                response_schema=schema,
                additional_headers={"X-Test": "1"},
                location="global",
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["max_output_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["response_mime_type"] == "application/json"
        assert kwargs["response_schema"] == schema
        assert kwargs["additional_headers"] == {"X-Test": "1"}
        assert kwargs["location"] == "global"


class TestAzureOpenAIProvider:
    """Verify AzureOpenAIProvider.load() constructs AzureChatOpenAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes azure_deployment and api_version."""
        mock_cls = MagicMock()
        with patch("langchain_openai.AzureChatOpenAI", mock_cls):
            AzureOpenAIProvider().load(model_name="gpt-4", api_version="2024-01")
        kwargs = mock_cls.call_args[1]
        assert kwargs["azure_deployment"] == "gpt-4"
        assert kwargs["api_version"] == "2024-01"

    def test_full_config(self):
        """Verify all optional params are forwarded to AzureChatOpenAI."""
        mock_cls = MagicMock()
        with patch("langchain_openai.AzureChatOpenAI", mock_cls):
            AzureOpenAIProvider().load(
                model_name="gpt-4",
                api_version="2024-01",
                max_tokens=200,
                temperature=0.7,
                top_p=0.95,
                seed=123,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["max_completion_tokens"] == 200
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95
        assert kwargs["seed"] == 123


class TestOpenAIProvider:
    """Verify OpenAIProvider.load() constructs ChatOpenAI correctly."""

    def test_minimal_load(self):
        """Verify minimal config passes model to ChatOpenAI."""
        mock_cls = MagicMock()
        with patch("langchain_openai.ChatOpenAI", mock_cls):
            OpenAIProvider().load(model_name="gpt-4o")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gpt-4o"

    def test_full_config(self):
        """Verify all optional params are forwarded to ChatOpenAI."""
        mock_cls = MagicMock()
        with patch("langchain_openai.ChatOpenAI", mock_cls):
            OpenAIProvider().load(
                model_name="gpt-4o",
                max_tokens=500,
                temperature=0.3,
                top_p=0.8,
                seed=7,
                max_retries=3,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["max_completion_tokens"] == 500
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_retries"] == 3


class TestLlamaCppProvider:
    """Verify LlamaCppProvider.load() constructs ChatLlamaCpp correctly."""

    def test_full_config(self):
        """Verify all optional params are merged into the LlamaCpp config."""
        mock_cls = MagicMock()
        with patch("langchain_community.chat_models.ChatLlamaCpp", mock_cls):
            LlamaCppProvider().load(
                model_name="model.gguf",
                max_tokens=256,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                seed=11,
            )
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_path"] == "model.gguf"
        assert kwargs["n_ctx"] == 4096
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.95
        assert kwargs["top_k"] == 50
        assert kwargs["seed"] == 11

    def test_minimal_config_omits_optional(self):
        """Verify optional params are absent when not provided."""
        mock_cls = MagicMock()
        with patch("langchain_community.chat_models.ChatLlamaCpp", mock_cls):
            LlamaCppProvider().load(model_name="model.gguf")
        kwargs = mock_cls.call_args[1]
        assert kwargs["model_path"] == "model.gguf"
        for absent in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
            assert absent not in kwargs


class TestHuggingFaceProvider:
    """Verify HuggingFaceProvider.load() assembles the pipeline correctly."""

    @patch("bili.iris.loaders.tokenizer_loader.load_huggingface_tokenizer")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.pipeline")
    @patch(
        "langchain_huggingface.chat_models.huggingface.HuggingFacePipeline",
        create=True,
    )
    @patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace",
        create=True,
    )
    def test_full_config(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_chat,
        mock_hf_pipe,  # pylint: disable=unused-argument
        mock_pipe,
        mock_auto,
        mock_empty_cache,  # pylint: disable=unused-argument
        mock_cuda_avail,  # pylint: disable=unused-argument
        mock_tok,
    ):
        """Verify all optional params flow into the HuggingFace pipeline."""
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        mock_tok.return_value = tokenizer
        mock_auto.from_pretrained.return_value = MagicMock()
        mock_chat.return_value = "chat_model"

        result = HuggingFaceProvider().load(
            model_name="gpt2",
            max_tokens=128,
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            seed=7,
        )

        assert result == "chat_model"
        assert tokenizer.pad_token == "</s>"
        pipe_kwargs = mock_pipe.call_args[1]
        assert pipe_kwargs["task"] == "text-generation"
        assert pipe_kwargs["max_new_tokens"] == 128
        assert pipe_kwargs["temperature"] == 0.6


# ---------------------------------------------------------------------------
# load_model() backward-compat integration
# ---------------------------------------------------------------------------


class TestLoadModelBackwardCompat:
    """Verify load_model() still works correctly after the refactor.

    These tests confirm the existing public API is unchanged for all callers.
    """

    @patch("bili.iris.loaders.llm_loader.load_llamacpp_model")
    def test_routes_llamacpp_unchanged(self, mock_loader):
        """load_model('local_llamacpp') still routes to load_llamacpp_model."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_loader.return_value = MagicMock()
        result = load_model("local_llamacpp", model_name="m.gguf", max_tokens=50)
        mock_loader.assert_called_once_with(model_name="m.gguf", max_tokens=50)
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_bedrock_model")
    def test_routes_bedrock_unchanged(self, mock_loader):
        """load_model('remote_aws_bedrock') still routes to load_remote_bedrock_model."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_loader.return_value = MagicMock()
        result = load_model("remote_aws_bedrock", model_name="anthropic.claude-v2")
        mock_loader.assert_called_once_with(model_name="anthropic.claude-v2")
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_routes_openai_unchanged(self, mock_loader):
        """load_model('remote_openai') still routes to load_remote_openai."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_loader.return_value = MagicMock()
        result = load_model("remote_openai", model_name="gpt-4o")
        mock_loader.assert_called_once_with(model_name="gpt-4o")
        assert result is mock_loader.return_value

    def test_invalid_type_still_raises_value_error(self):
        """load_model() with unknown type still raises ValueError."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        with pytest.raises(ValueError, match="Invalid model type"):
            load_model("__unknown_provider__", model_name="x")

    def test_registered_provider_routes_through_registry(self):
        """load_model() with a registered type delegates to the provider class."""
        from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
            load_model,
        )

        mock_llm = MagicMock()

        class _RegistryProvider(LLMProvider):
            def load(self, **kwargs):
                return mock_llm

        register_provider("test_registry_route", _RegistryProvider)
        try:
            result = load_model("test_registry_route", model_name="test")
            assert result is mock_llm
        finally:
            PROVIDER_REGISTRY.unregister("test_registry_route")
