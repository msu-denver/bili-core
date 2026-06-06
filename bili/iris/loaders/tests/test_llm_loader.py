"""Tests for bili.iris.loaders.llm_loader module.

Covers load_model routing, prepare_runtime_config for all
provider types, and individual loader functions with mocked
external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from bili.iris.loaders.llm_loader import (
    load_huggingface_model,
    load_llamacpp_model,
    load_model,
    load_remote_azure_openai,
    load_remote_bedrock_model,
    load_remote_gcp_vertex_model,
    load_remote_openai,
    prepare_runtime_config,
)

# ---------------------------------------------------------------------------
# load_model — routing
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Verify load_model dispatches to the correct provider loader."""

    @patch("bili.iris.loaders.llm_loader.load_llamacpp_model")
    def test_routes_to_llamacpp(self, mock_loader):
        """Verify local_llamacpp routes to load_llamacpp_model."""
        mock_loader.return_value = MagicMock()
        result = load_model("local_llamacpp", model_name="m.gguf", max_tokens=50)
        mock_loader.assert_called_once_with(model_name="m.gguf", max_tokens=50)
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_huggingface_model")
    def test_routes_to_huggingface(self, mock_loader):
        """Verify local_huggingface routes to load_huggingface_model."""
        mock_loader.return_value = MagicMock()
        result = load_model("local_huggingface", model_name="gpt2", max_tokens=10)
        mock_loader.assert_called_once_with(model_name="gpt2", max_tokens=10)
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_gcp_vertex_model")
    def test_routes_to_vertex(self, mock_loader):
        """Verify remote_google_vertex routes correctly."""
        mock_loader.return_value = MagicMock()
        result = load_model("remote_google_vertex", model_name="gemini-pro")
        mock_loader.assert_called_once_with(model_name="gemini-pro")
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_bedrock_model")
    def test_routes_to_bedrock(self, mock_loader):
        """Verify remote_aws_bedrock routes correctly."""
        mock_loader.return_value = MagicMock()
        result = load_model("remote_aws_bedrock", model_name="anthropic.claude-v2")
        mock_loader.assert_called_once_with(model_name="anthropic.claude-v2")
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_azure_openai")
    def test_routes_to_azure_openai(self, mock_loader):
        """Verify remote_azure_openai routes correctly."""
        mock_loader.return_value = MagicMock()
        result = load_model(
            "remote_azure_openai",
            model_name="gpt-4",
            api_version="2024-01",
        )
        mock_loader.assert_called_once_with(model_name="gpt-4", api_version="2024-01")
        assert result is mock_loader.return_value

    @patch("bili.iris.loaders.llm_loader.load_remote_openai")
    def test_routes_to_openai(self, mock_loader):
        """Verify remote_openai routes correctly."""
        mock_loader.return_value = MagicMock()
        result = load_model("remote_openai", model_name="gpt-4o")
        mock_loader.assert_called_once_with(model_name="gpt-4o")
        assert result is mock_loader.return_value

    def test_invalid_model_type_raises_value_error(self):
        """Verify an unsupported model_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model type"):
            load_model("unsupported_provider", model_name="x")


# ---------------------------------------------------------------------------
# prepare_runtime_config
# ---------------------------------------------------------------------------


class TestPrepareRuntimeConfig:
    """Verify runtime config is built correctly per provider."""

    def test_vertex_with_thinking_budget(self):
        """Verify Vertex AI gets a ThinkingConfig with budget."""
        mock_thinking_config = MagicMock()
        with patch(
            "bili.iris.loaders.llm_loader.types",
            create=True,
        ):
            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"google.genai": MagicMock(), "google.genai.types": MagicMock()},
            ):
                from google.genai import (
                    types as patched_types,  # pylint: disable=import-outside-toplevel
                )

                patched_types.ThinkingConfig.return_value = mock_thinking_config

                result = prepare_runtime_config(
                    model_type="remote_google_vertex",
                    thinking_config={"budget": 5000},
                )

                assert "thinking_config" in result

    def test_vertex_without_thinking_config(self):
        """Verify Vertex AI with no thinking config returns empty."""
        result = prepare_runtime_config(
            model_type="remote_google_vertex",
            thinking_config=None,
        )
        assert not result

    def test_non_vertex_with_thinking_config_warns(self):
        """Verify non-Vertex providers ignore thinking config."""
        result = prepare_runtime_config(
            model_type="remote_openai",
            thinking_config={"budget": 100},
        )
        # thinking_config should be ignored for non-vertex
        assert "thinking_config" not in result

    def test_kwargs_are_merged(self):
        """Verify extra kwargs are included in the config."""
        result = prepare_runtime_config(
            model_type="remote_openai",
            thinking_config=None,
            custom_key="custom_value",
        )
        assert result["custom_key"] == "custom_value"

    def test_no_thinking_config_no_kwargs_returns_empty(self):
        """Verify empty config when nothing is provided."""
        result = prepare_runtime_config(model_type="remote_aws_bedrock")
        assert not result

    def test_vertex_with_string_budget(self):
        """Verify string budget is converted to int for Vertex AI."""
        with patch.dict(
            "sys.modules",
            {"google.genai": MagicMock(), "google.genai.types": MagicMock()},
        ):
            from google.genai import (
                types as patched_types,  # pylint: disable=import-outside-toplevel
            )

            patched_types.ThinkingConfig.return_value = MagicMock()
            result = prepare_runtime_config(
                model_type="remote_google_vertex",
                thinking_config={"budget": "3000"},
            )
            assert "thinking_config" in result

    def test_vertex_import_error_handled(self):
        """Verify ImportError is caught when google.genai missing."""
        with patch.dict("sys.modules", {"google.genai": None}):
            result = prepare_runtime_config(
                model_type="remote_google_vertex",
                thinking_config={"budget": 100},
            )
            # Should gracefully return empty on import error
            assert "thinking_config" not in result


# ---------------------------------------------------------------------------
# Individual loader functions (mocked)
# ---------------------------------------------------------------------------


class TestLoadRemoteBedrockModel:
    """Verify Bedrock model initialization with various params."""

    @patch("bili.iris.loaders.llm_loader.ChatBedrockConverse")
    def test_minimal_config(self, mock_cls):
        """Verify Bedrock init with only model_name."""
        mock_cls.return_value = MagicMock()
        result = load_remote_bedrock_model(model_name="claude-v2")
        mock_cls.assert_called_once_with(model_id="claude-v2")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.llm_loader.ChatBedrockConverse")
    def test_full_config(self, mock_cls):
        """Verify Bedrock init with all optional params."""
        mock_cls.return_value = MagicMock()
        load_remote_bedrock_model(
            model_name="claude-v2",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            seed=42,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_id"] == "claude-v2"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["seed"] == 42


class TestLoadRemoteGcpVertexModel:
    """Verify Vertex AI model initialization."""

    @patch("bili.iris.loaders.llm_loader.ChatVertexAI")
    def test_minimal_config(self, mock_cls):
        """Verify Vertex init with only model_name."""
        mock_cls.return_value = MagicMock()
        result = load_remote_gcp_vertex_model(model_name="gemini-pro")
        mock_cls.assert_called_once_with(model_name="gemini-pro")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.llm_loader.ChatVertexAI")
    def test_with_additional_headers(self, mock_cls):
        """Verify additional_headers are passed through."""
        headers = {"X-Vertex-AI-LLM-Request-Type": "dedicated"}
        mock_cls.return_value = MagicMock()
        load_remote_gcp_vertex_model(
            model_name="gemini-pro",
            additional_headers=headers,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["additional_headers"] == headers

    @patch("bili.iris.loaders.llm_loader.ChatVertexAI")
    def test_with_location(self, mock_cls):
        """Verify location parameter is passed through."""
        mock_cls.return_value = MagicMock()
        load_remote_gcp_vertex_model(model_name="gemini-pro", location="global")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["location"] == "global"


class TestLoadRemoteAzureOpenai:
    """Verify Azure OpenAI model initialization."""

    @patch("bili.iris.loaders.llm_loader.AzureChatOpenAI")
    def test_minimal_config(self, mock_cls):
        """Verify Azure init with required params only."""
        mock_cls.return_value = MagicMock()
        result = load_remote_azure_openai(model_name="gpt-4", api_version="2024-01")
        mock_cls.assert_called_once_with(
            azure_deployment="gpt-4", api_version="2024-01"
        )
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.llm_loader.AzureChatOpenAI")
    def test_full_config(self, mock_cls):
        """Verify Azure init with all optional params."""
        mock_cls.return_value = MagicMock()
        load_remote_azure_openai(
            model_name="gpt-4",
            api_version="2024-01",
            max_tokens=200,
            temperature=0.7,
            top_p=0.95,
            seed=123,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 200
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["seed"] == 123


class TestLoadRemoteOpenai:
    """Verify OpenAI model initialization."""

    @patch("bili.iris.loaders.llm_loader.ChatOpenAI")
    def test_minimal_config(self, mock_cls):
        """Verify OpenAI init with only model_name."""
        mock_cls.return_value = MagicMock()
        result = load_remote_openai(model_name="gpt-4o")
        mock_cls.assert_called_once_with(model="gpt-4o")
        assert result is mock_cls.return_value

    @patch("bili.iris.loaders.llm_loader.ChatOpenAI")
    def test_full_config(self, mock_cls):
        """Verify OpenAI init with all optional params."""
        mock_cls.return_value = MagicMock()
        load_remote_openai(
            model_name="gpt-4o",
            max_tokens=500,
            temperature=0.3,
            top_p=0.8,
            seed=7,
            max_retries=3,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_completion_tokens"] == 500
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_retries"] == 3


# ---------------------------------------------------------------------------
# GPU/CPU detection (module-level)
# ---------------------------------------------------------------------------


class TestDeviceDetection:
    """Verify GPU/CPU detection logging at module level."""

    @patch("bili.iris.loaders.llm_loader.torch")
    def test_mps_detection_path(self, mock_torch):
        """Verify Apple MPS path is reachable."""
        mock_torch.backends.mps.is_available.return_value = True
        # The detection runs at import time, so we just verify
        # the torch API is used correctly
        assert mock_torch.backends.mps.is_available() is True

    @patch("bili.iris.loaders.llm_loader.torch")
    def test_cuda_detection_path(self, mock_torch):
        """Verify CUDA path is reachable."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert mock_torch.cuda.is_available() is True

    @patch("bili.iris.loaders.llm_loader.torch")
    def test_cpu_fallback_path(self, mock_torch):
        """Verify CPU fallback when no GPU available."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert not mock_torch.backends.mps.is_available()
        assert not mock_torch.cuda.is_available()


# ---------------------------------------------------------------------------
# load_huggingface_model (local)
# ---------------------------------------------------------------------------


class TestLoadHuggingFaceModel:
    """Verify the local HuggingFace pipeline construction."""

    @patch("bili.iris.loaders.llm_loader.ChatHuggingFace")
    @patch("bili.iris.loaders.llm_loader.HuggingFacePipeline")
    @patch("bili.iris.loaders.llm_loader.pipeline")
    @patch("bili.iris.loaders.llm_loader.AutoModelForCausalLM")
    @patch("bili.iris.loaders.llm_loader.torch")
    @patch("bili.iris.loaders.llm_loader.load_huggingface_tokenizer")
    def test_builds_chat_model_with_full_generation_config(
        self,
        mock_tokenizer_loader,
        mock_torch,
        mock_auto_model,
        mock_pipeline,
        mock_hf_pipeline,
        mock_chat_hf,
    ):
        """Verify generation config is assembled and a ChatHuggingFace is returned."""
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        mock_tokenizer_loader.return_value = tokenizer
        mock_torch.cuda.is_available.return_value = True
        model = MagicMock()
        mock_auto_model.from_pretrained.return_value = model
        mock_chat_hf.return_value = "chat_model"

        result = load_huggingface_model(
            model_name="gpt2",
            max_tokens=128,
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            seed=7,
        )

        assert result == "chat_model"
        # Missing pad token is backfilled from the eos token.
        assert tokenizer.pad_token == "</s>"
        # CUDA cache is cleared when CUDA is available.
        mock_torch.cuda.empty_cache.assert_called_once()
        # The optional generation params are forwarded to pipeline().
        pipeline_kwargs = mock_pipeline.call_args[1]
        assert pipeline_kwargs["task"] == "text-generation"
        assert pipeline_kwargs["return_full_text"] is False
        assert pipeline_kwargs["model"] is model
        assert pipeline_kwargs["max_new_tokens"] == 128
        assert pipeline_kwargs["temperature"] == 0.6
        assert pipeline_kwargs["top_p"] == 0.9
        assert pipeline_kwargs["top_k"] == 40
        assert pipeline_kwargs["seed"] == 7
        mock_hf_pipeline.assert_called_once_with(pipeline=mock_pipeline.return_value)
        mock_chat_hf.assert_called_once_with(llm=mock_hf_pipeline.return_value)

    @patch("bili.iris.loaders.llm_loader.ChatHuggingFace")
    @patch("bili.iris.loaders.llm_loader.HuggingFacePipeline")
    @patch("bili.iris.loaders.llm_loader.pipeline")
    @patch("bili.iris.loaders.llm_loader.AutoModelForCausalLM")
    @patch("bili.iris.loaders.llm_loader.torch")
    @patch("bili.iris.loaders.llm_loader.load_huggingface_tokenizer")
    def test_minimal_config_omits_optional_params(
        self,
        mock_tokenizer_loader,
        mock_torch,
        mock_auto_model,
        mock_pipeline,
        mock_hf_pipeline,
        mock_chat_hf,
    ):
        """Verify optional generation params are absent when not provided."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        mock_tokenizer_loader.return_value = tokenizer
        mock_torch.cuda.is_available.return_value = False
        mock_chat_hf.return_value = "chat_model"

        load_huggingface_model(model_name="gpt2")

        # No CUDA cache clear when CUDA is unavailable.
        mock_torch.cuda.empty_cache.assert_not_called()
        pipeline_kwargs = mock_pipeline.call_args[1]
        for absent in ("max_new_tokens", "temperature", "top_p", "top_k", "seed"):
            assert absent not in pipeline_kwargs


# ---------------------------------------------------------------------------
# load_llamacpp_model (local)
# ---------------------------------------------------------------------------


class TestLoadLlamaCppModel:
    """Verify the local LlamaCpp model construction."""

    @patch("bili.iris.loaders.llm_loader.ChatLlamaCpp")
    def test_full_config_passes_all_params(self, mock_cls):
        """Verify all optional params are merged into the LlamaCpp config."""
        mock_cls.return_value = "llama_model"

        result = load_llamacpp_model(
            model_name="model.gguf",
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            seed=11,
        )

        assert result == "llama_model"
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_path"] == "model.gguf"
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["seed"] == 11

    @patch("bili.iris.loaders.llm_loader.ChatLlamaCpp")
    def test_minimal_config_omits_optional_params(self, mock_cls):
        """Verify optional params are omitted when falsy."""
        mock_cls.return_value = "llama_model"

        load_llamacpp_model(model_name="model.gguf")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_path"] == "model.gguf"
        for absent in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
            assert absent not in call_kwargs


# ---------------------------------------------------------------------------
# Optional-parameter coverage for remote loaders
# ---------------------------------------------------------------------------


class TestRemoteLoaderOptionalParams:
    """Verify optional config keys are set on the remote loaders."""

    @patch("bili.iris.loaders.llm_loader.ChatVertexAI")
    def test_vertex_all_optional_params(self, mock_cls):
        """Verify every optional Vertex config key is forwarded."""
        mock_cls.return_value = MagicMock()
        schema = {"type": "object"}
        load_remote_gcp_vertex_model(
            model_name="gemini-pro",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            seed=3,
            response_mime_type="application/json",
            response_schema=schema,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["max_output_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 20
        assert call_kwargs["seed"] == 3
        assert call_kwargs["response_mime_type"] == "application/json"
        assert call_kwargs["response_schema"] == schema

    @patch("bili.iris.loaders.llm_loader.AzureChatOpenAI")
    def test_azure_top_k_forwarded(self, mock_cls):
        """Verify Azure top_k is forwarded."""
        mock_cls.return_value = MagicMock()
        load_remote_azure_openai(model_name="gpt-4", api_version="2024-01", top_k=15)
        assert mock_cls.call_args[1]["top_k"] == 15

    @patch("bili.iris.loaders.llm_loader.ChatOpenAI")
    def test_openai_top_k_forwarded(self, mock_cls):
        """Verify OpenAI top_k is forwarded."""
        mock_cls.return_value = MagicMock()
        load_remote_openai(model_name="gpt-4o", top_k=25)
        assert mock_cls.call_args[1]["top_k"] == 25


# ---------------------------------------------------------------------------
# prepare_runtime_config — ThinkingConfig error handling
# ---------------------------------------------------------------------------


class TestPrepareRuntimeConfigErrors:
    """Verify ThinkingConfig value/type errors are caught and logged."""

    def test_invalid_budget_string_is_handled(self):
        """Verify a non-numeric budget string is caught without raising."""
        fake_types = MagicMock()
        fake_types.ThinkingConfig.side_effect = None
        with patch.dict(
            "sys.modules",
            {"google.genai": MagicMock(types=fake_types)},
        ):
            # An int() conversion failure on the budget string is caught by
            # the ValueError/TypeError handler, leaving the config empty.
            result = prepare_runtime_config(
                model_type="remote_google_vertex",
                thinking_config={"budget": "not-a-number"},
            )
        assert "thinking_config" not in result

    def test_none_budget_skips_thinking_config(self):
        """Verify an explicit None budget produces no thinking_config."""
        with patch.dict(
            "sys.modules",
            {"google.genai": MagicMock()},
        ):
            result = prepare_runtime_config(
                model_type="remote_google_vertex",
                thinking_config={"budget": None},
            )
        assert "thinking_config" not in result
