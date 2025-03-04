"""
Module: config

This module initializes and configures various LLM models for use within the
application. The models are defined in the centralized configuration file and
include options for different LLM providers such as Google Vertex AI, AWS
Bedrock, Azure OpenAI, and local models. Each model is initialized based on
its specific parameters and prompts, which are provided as arguments to the
`initialize_models` function.

Functions:
    - initialize_models(active_models, model_params):
      Initializes and caches LLM models based on the active models and their
      specific parameters.

Dependencies:
    - bili.config.llm_config: Imports LLM_MODELS for accessing the model configurations.

Usage:
    This module is intended to be used within the application to initialize
    and configure various LLM models based on the provided configuration. It
    supports models from different providers for various use cases.

Example:
    from bili.config.llm_config import initialize_models

    # Initialize models with active models and parameters
    models = initialize_models(active_models, model_params)
"""

# Available LLM Models and Types
LLM_MODELS = {
    # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html
    # https://aws.amazon.com/bedrock/pricing/
    # Many models have tools disabled for now because of issue:
    # https://github.com/langchain-ai/langchain-aws/issues/141
    "remote_aws_bedrock": {
        "name": "AWS Bedrock",
        "description": "Remote model hosted on AWS Bedrock. Supports multiple LLMs like "
        "Amazon Titan, Claude, LLama, AI21 Labs, and Minstrel models. "
        "Descriptions of all models can be found at "
        "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
        "model_help": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions",
        "models": [
            # AWS Models
            # https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
            # https://docs.aws.amazon.com/bedrock/latest/userguide/titan-text-models.html
            {
                "model_name": "Amazon Nova Pro",
                "model_id": "amazon.nova-pro-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 300000,
                "max_output_tokens": 5000,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Amazon Nova Lite",
                "model_id": "amazon.nova-lite-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 300000,
                "max_output_tokens": 5000,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Amazon Nova Micro",
                "model_id": "amazon.nova-micro-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 5000,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Amazon Titan Text G1 - Premier (Deprecated)",
                "model_id": "amazon.titan-text-premier-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 32000,
                "max_output_tokens": 3072,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Amazon Titan Text G1 - Express (Deprecated)",
                "model_id": "amazon.titan-text-express-v1",
                "custom_model_path": False,
                "max_input_tokens": 8192,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Amazon Titan Text G1 - Lite (Deprecated)",
                "model_id": "amazon.titan-text-lite-v1",
                "custom_model_path": False,
                "max_input_tokens": 4096,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            # AI21 Labs Models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
            {
                "model_name": "AI21 Jamba 1.5 Large",
                "model_id": "ai21.jamba-1-5-large-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 256000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                # Although documentation says 2.0 is the max, anything greater than 1.0 errors out
                "temperature_max": 1.0,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
            },
            {
                "model_name": "AI21 Jamba 1.5 Mini",
                "model_id": "ai21.jamba-1-5-mini-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 256000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                # Although documentation says 2.0 is the max, anything greater than 1.0 errors out
                "temperature_max": 1.0,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
            },
            {
                "model_name": "AI21 Jamba-Instruct (Deprecated)",
                "model_id": "ai21.jamba-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 256000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                # Although documentation says 2.0 is the max, anything greater than 1.0 errors out
                "temperature_max": 1.0,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "supports_tools": False,
            },
            # Anthropic Models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
            # https://docs.anthropic.com/en/docs/about-claude/models
            {
                "model_name": "Anthropic Claude 2.1",
                "model_id": "anthropic.claude-v2:1",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 2",
                "model_id": "anthropic.claude-v2",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3 Haiku",
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3 Opus",
                "model_id": "anthropic.claude-3-opus-20240229-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3 Sonnet",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3.5 Sonnet",
                "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3.5 Sonnet v2",
                "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            {
                "model_name": "Anthropic Claude 3.5 Haiku",
                "model_id": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
            },
            # Cohere Models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
            # https://aws.amazon.com/bedrock/cohere/#:~:text=With%20a%20context%20window%20of,advanced%20retrieval%2C%20and%20tool%20use.
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
            {
                "model_name": "Cohere Command Light",
                "model_id": "cohere.command-light-text-v14",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 500,
            },
            {
                "model_name": "Cohere Command R+",
                "model_id": "cohere.command-r-plus-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 500,
            },
            {
                "model_name": "Cohere Command R",
                "model_id": "cohere.command-r-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 500,
            },
            {
                "model_name": "Cohere Command",
                "model_id": "cohere.command-text-v14",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 500,
            },
            # Meta LLama Models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
            # https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
            {
                "model_name": "Meta Llama 3.1 8B Instruct",
                "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Meta Llama 3.1 70B Instruct",
                "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Meta Llama 3.2 11B Instruct",
                "model_id": "us.meta.llama3-2-11b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Meta Llama 3.2 1B Instruct",
                "model_id": "us.meta.llama3-2-1b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Meta Llama 3.2 3B Instruct",
                "model_id": "us.meta.llama3-2-3b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            {
                "model_name": "Meta Llama 3.3 70B Instruct",
                "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 2048,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 50,
                "supports_tools": False,
            },
            # Minstral AI Models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
            # https://docs.mistral.ai/getting-started/models/models_overview/
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
            {
                "model_name": "Mistral Large",
                "model_id": "mistral.mistral-large-2402-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 131000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "supports_tools": False,
            },
            {
                "model_name": "Mistral Small",
                "model_id": "mistral.mistral-small-2402-v1:0",
                "custom_model_path": False,
                "max_input_tokens": 32000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "supports_tools": False,
            },
            {
                "model_name": "Mistral 7B Instruct",
                "model_id": "mistral.mistral-7b-instruct-v0:2",
                "custom_model_path": False,
                "max_input_tokens": 131000,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 200,
                "supports_tools": False,
            },
            {
                "model_name": "Mistral Mixtral 8x7B Instruct",
                "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
                "custom_model_path": False,
                "max_input_tokens": 32000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 200,
            },
        ],
    },
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
    # https://cloud.google.com/vertex-ai/generative-ai/pricing
    "remote_google_vertex": {
        "name": "Google Vertex AI",
        "description": "Remote model hosted on Google Vertex AI. "
        "Model descriptions can be found at "
        "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions",
        "model_help": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions",
        "models": [
            {
                "model_name": "Gemini 1.5 Pro 002",
                "model_id": "gemini-1.5-pro-002",
                "custom_model_path": False,
                "max_input_tokens": 2097152,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 40,
            },
            {
                "model_name": "Gemini 1.5 Pro",
                "model_id": "gemini-1.5-pro",
                "custom_model_path": False,
                "max_input_tokens": 2097152,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 40,
            },
            {
                "model_name": "Gemini 1.0 Pro",
                "model_id": "gemini-1.0-pro",
                "custom_model_path": False,
                "max_input_tokens": 32760,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 40,
            },
            {
                "model_name": "Gemini 1.5 Flash",
                "model_id": "gemini-1.5-flash",
                "custom_model_path": False,
                "max_input_tokens": 1048576,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 40,
            },
            {
                "model_name": "Gemini 1.5 Flash 002",
                "model_id": "gemini-1.5-flash-002",
                "custom_model_path": False,
                "max_input_tokens": 1048576,
                "max_output_tokens": 8192,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 40,
            },
        ],
    },
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    "remote_azure_openai": {
        "name": "Azure OpenAI",
        "description": "Remote model hosted on Azure OpenAI Service. "
        "Model descriptions can be found at "
        """https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs="""
        """global-standard%2Cstandard-chat-completions""",
        "model_help": """https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs="""
        """global-standard%2Cstandard-chat-completions""",
        "models": [
            {
                "model_name": "Azure OpenAI GPT-4o Omni",
                "model_id": "gpt-4o",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI GPT-4o mini",
                "model_id": "gpt-4o-mini",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI GPT-4 Turbo with Vision",
                "model_id": "gpt-4",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI o1 (Access Request Pending)",
                "model_id": "o1",
                "custom_model_path": False,
                "max_input_tokens": 200000,
                "max_output_tokens": 100000,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI o1-mini (Access Request Pending)",
                "model_id": "o1-mini",
                "custom_model_path": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 65536,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI GPT 3.5 Turbo 16K",
                "model_id": "gpt-35-turbo-16k",
                "custom_model_path": False,
                "max_input_tokens": 16384,
                "max_output_tokens": 16384,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
            {
                "model_name": "Azure OpenAI GPT-3.5 Turbo",
                "model_id": "gpt-35-turbo",
                "custom_model_path": False,
                "max_input_tokens": 16385,
                "max_output_tokens": 4096,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "supports_top_p": True,
                "supports_top_k": False,
                "kwargs": {
                    "api_version": "2024-08-01-preview",
                },
            },
        ],
    },
    "local_llamacpp": {
        "name": "Local LlamaCpp Compatible Model",
        "description": "Local LlamaCpp compatible model loaded into memory.",
        "model_help": "https://huggingface.co/models?other=gguf",
        "models": [
            {
                "model_name": "LlamaCpp Local (In Memory) Model",
                "model_id": "/app/bili-core/models/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
                "custom_model_path": True,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "local_only": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 100,
                # LlamaCPP does not support automatic tool calling, so it will be disabled
                # If you wanted to manually use tools, you can create the LLM and bind
                # tool calls differently than create_react_agent does in the default
                # implementation in bili.loaders.langchain_loader.load_langgraph_agent
                "supports_tools": False,
            },
        ],
    },
    "local_huggingface": {
        "name": "Local HuggingFace Compatible Model",
        "description": "Local HuggingFace compatible model loaded into memory. "
        "Can specify either a file path or a HuggingFace model ID.",
        "model_help": "https://huggingface.co/models?other=gptq",
        "models": [
            {
                "model_name": "HuggingFace Local (In Memory) Model",
                "model_id": "/app/bili-core/models/Llama-3.2-1B-Instruct",
                "custom_model_path": True,
                "supports_temperature": True,
                "supports_seed": True,
                "supports_max_output_tokens": True,
                "local_only": True,
                "supports_top_p": True,
                "supports_top_k": True,
                "top_k_max": 100,
                # ChatHuggingFace does not support automatic tool calling, so it will be disabled
                # If you wanted to manually use tools, you can create the LLM and bind
                # tool calls differently than create_react_agent does in the default
                # implementation in bili.loaders.langchain_loader.load_langgraph_agent
                "supports_tools": False,
            },
        ],
    },
}
