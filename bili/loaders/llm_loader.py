"""
Module: llm_loader

This module provides functions to load and initialize various language models for LangChain.
It supports local models (LlamaCpp, HuggingFace) and remote models
(Google Vertex AI, AWS Bedrock, Azure OpenAI).
The functions are conditionally cached to optimize resource usage in Streamlit applications.

Functions:
    - load_model(model_type, **kwargs):
      Loads a machine learning model based on the provided model type.
    - load_huggingface_model(model_name, max_tokens, temperature, top_p=0.1, top_k=None, seed=None):
      Loads a locally available HuggingFace model and initializes a text generation pipeline.
    - load_llamacpp_model(model_name, max_tokens, temperature, top_p=1.0, top_k=50, seed=None):
      Loads a compatible model using the LlamaCpp library with specified configuration options.
    - load_remote_gcp_vertex_model(model_name, max_tokens, temperature,
            top_p=None, top_k=None, seed=None):
      Loads a remote GCP Vertex AI model with the specified configuration parameters.
    - load_remote_bedrock_model(model_name, max_tokens, temperature, top_p=None,
            top_k=None, seed=None):
      Initializes and loads a remote bedrock model from AWS Bedrock service.
    - load_remote_azure_openai(model_name, api_version, max_tokens, temperature,
            top_p=None, top_k=None, seed=None):
      Loads and initializes a remote Azure OpenAI model with the specified
      parameters and configurations.

Dependencies:
    - gc: Provides garbage collection functionality.
    - torch: Provides PyTorch for model loading and inference.
    - langchain_aws: Provides ChatBedrockConverse for AWS Bedrock.
    - langchain_community.llms: Provides HuggingFacePipeline and LlamaCpp.
    - langchain_google_vertexai: Provides ChatVertexAI for Google Vertex AI.
    - langchain_openai: Provides AzureChatOpenAI for Azure OpenAI.
    - transformers: Provides AutoModelForCausalLM, AutoTokenizer, and pipeline for model handling.
    - bili.streamlit.utils.streamlit_utils: Imports `conditional_cache_resource` for caching.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within applications that require loading and initializing
    various language models. It provides functions to load models from different providers with
    conditional caching to optimize resource usage.

Example:
    from bili.loaders.llm_loader import load_model

    # Load a local HuggingFace model
    model = load_model(
        model_type="local_huggingface",
        model_name="gptq_model",
        max_tokens=100,
        temperature=0.7
    )
"""

import gc

import torch
from langchain_aws import ChatBedrockConverse
from langchain_community.chat_models import ChatLlamaCpp
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface.chat_models.huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langchain_openai import AzureChatOpenAI
from transformers import AutoModelForCausalLM, pipeline

from bili.loaders.tokenizer_loader import load_huggingface_tokenizer
from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# This snippet is used to detect what devices are available for inference.
# The model itself will automatically choose the best device, but this
# will help us know if we have a GPU available.
if torch.backends.mps.is_available():
    # Detect if Apple MPS is available
    LOGGER.info("Apple MPS device found")
elif torch.cuda.is_available():
    # Detect if Nvidia GPU is available
    LOGGER.info("Nvidia GPU device found")
else:
    # If no GPU is available, use CPU
    LOGGER.info("No compatible GPU device found, CPU will be used for inference.")


# This function initializes and loads the Llama model.
# It uses Streamlit's cache feature to load the model only once, enhancing performance.
def load_model(
    model_type,
    **kwargs,
):
    """
    Loads a machine learning model based on the provided model type. This function
    routes to the appropriate loader function depending on whether the model type
    is local or hosted remotely on cloud services.

    :param model_type: Specifies the type of the model to be loaded. The value determines
        which loader function will be called. Supported types are "local_llamacpp",
        "local_huggingface", "remote_google_vertex", "remote_aws_bedrock", and
        "remote_azure_openai".
    :type model_type: str
    :param kwargs: Additional keyword arguments specific to the loader function for
        the chosen model type. These arguments differ depending on the model type.
    :type kwargs: dict
    :return: The loaded model object as returned by the appropriate model loader
        function. The return value may differ in format depending on the chosen
        model type.
    :rtype: object
    :raises ValueError: If the specified model_type is not one of the supported
        values, this exception will be raised.
    """
    # Based on model_type, call the appropriate loader function
    if model_type == "local_llamacpp":
        llm_model = load_llamacpp_model(**kwargs)
    elif model_type == "local_huggingface":
        llm_model = load_huggingface_model(**kwargs)
    elif model_type == "remote_google_vertex":
        llm_model = load_remote_gcp_vertex_model(**kwargs)
    elif model_type == "remote_aws_bedrock":
        llm_model = load_remote_bedrock_model(**kwargs)
    elif model_type == "remote_azure_openai":
        llm_model = load_remote_azure_openai(**kwargs)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return llm_model


# This method initializes and loads the Llama model for CUDA-compatible machines.
@conditional_cache_resource()
def load_huggingface_model(
    model_name, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None
):
    """
    Loads a locally available HuggingFace model and initializes a text generation pipeline
    with configurations for optimal performance and resource usage. The method sets up
    a tokenizer, configures the model, and constructs the pipeline necessary for
    text generation tasks.

    :param model_name: The name or path of the pretrained HuggingFace model to load.
    :param max_tokens: (Optional) Maximum number of tokens to generate for text outputs.
    :param temperature: (Optional) Sampling temperature to control the randomness of the response.
    :param top_p: (Optional) Cumulative probability threshold for nucleus sampling.
    :param top_k: (Optional) The number of highest probability tokens to consider during sampling.
    :param seed: (Optional) Random seed for reproducibility of outputs.

    :return: An instance of `HuggingFacePipeline`, configured for generating text
             using the HuggingFace Llama model.
    """
    LOGGER.info("Loading HuggingFace model from %s...", model_name)
    tokenizer = load_huggingface_tokenizer(model_name)

    # Ask Python to garbage collect
    # This is useful to avoid out-of-memory errors when loading the model.
    gc.collect()

    # If using CUDA, also clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the Llama model with specific configurations for efficient GPU usage.
    # 'torch_dtype=torch.float16' optimizes model size and speed on supported hardware.
    # 'trust_remote_code=True' allows the model to be loaded from a remote location.
    # 'device_map="auto"' automatically selects the best device for the model, such as GPU or CPU.
    # the parameter also allows to accelerate to put each layer of the model to maximize the use
    # of your fastest hardware.
    # For example, if you have a GPU and a CPU, the model will be first put on GPU and then on
    # CPU if you do not
    # have enough GPU memory.
    # 'low_cpu_mem_usage=True' reduces CPU memory usage to avoid out-of-memory errors.
    # More info:
    # https://huggingface.co/docs/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        offload_folder="/tmp/model_offload",
    )

    # Set the padding token to the end-of-string token.
    # This is required because the Llama model does not have a padding token.
    # The padding token is used to pad the input to the model to a fixed length.
    # The padding token is also used to pad the model's output to a fixed length.
    # More info:
    # https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.Tokenizer.pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Set up the generation configuration for the model.
    # These parameters control how the model generates responses.
    # Need to refine generation config
    generation_config = {
        "do_sample": True,
        # Enables sampling, which lets the model generate different responses for the same input
        "repetition_penalty": 1.176,  # Penalizes repetition to avoid loops
    }
    if top_p is not None:
        # The top p value to use for generation, which controls the
        # diversity of generated responses
        generation_config["top_p"] = top_p
    if top_k is not None:
        # The number of most likely next words in a pool to choose from for generation
        generation_config["top_k"] = top_k
    if seed is not None:
        # The random seed to use for generation, which helps with reproducibility
        generation_config["seed"] = seed
    if max_tokens is not None:
        # Limits the maximum tokens generated
        generation_config["max_new_tokens"] = max_tokens
    if temperature is not None:
        # Controls randomness in response generation
        generation_config["temperature"] = temperature

    # Create a text generation pipeline.
    # This pipeline will manage input/output processing for text generation tasks.
    text_pipeline = pipeline(
        device_map="auto",
        # trust_remote_code=True allows the model to be loaded from a remote location.
        trust_remote_code=True,
        # the torch_dtype parameter lets us specify the data type to use for the model.
        torch_dtype=torch.float16,
        # The pipeline type is 'text-generation' because we are generating text.
        task="text-generation",
        # The return_full_text parameter is set to True to return the full text instead of
        # just the generated tokens.
        # This is useful for our use case because we want to display the full text to the user.
        return_full_text=True,
        # The Llama model is used for text generation.
        model=model,
        # The tokenizer is used to convert text to tokens that the model can understand.
        # The tokenizer is also used to convert the model's output tokens back to text.
        # The tokenizer in use is the one we initialized above, provided from the Llama model.
        tokenizer=tokenizer,
        # The generation configuration is used to control how the model generates responses.
        **generation_config,
    )

    # Wraps the text pipeline in a LangChain HuggingFacePipeline for easy integration.
    hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline)

    # Wraps the pipeline in a ChatHuggingFace object to enable tool support.
    chat_hf = ChatHuggingFace(llm=hf_pipeline)

    # Print the pipeline for debugging purposes
    LOGGER.debug(chat_hf)

    return chat_hf


@conditional_cache_resource()
def load_llamacpp_model(
    model_name, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None
):
    """
    Load a LlamaCpp model with specified configurations.

    This function facilitates the loading of a LlamaCpp model with the given model name using the
    LlamaCpp library. Parameters are provided to control generation and runtime
    behaviors, including tokenizer settings, resource allocation, and sampling
    parameters. The LlamaCpp library manages the tokenizer and pipeline integration
    internally, simplifying the setup process. This function supports customization
    to suit specific use cases, such as altering context sizes, controlling randomness,
    or ensuring reproducibility through a random seed.

    :param model_name: The file path of the model to load.
    :param max_tokens: (Optional) The maximum number of tokens to generate during a response.
    :param temperature: (Optional) Controls generation randomness; a higher value creates more random responses.
    :param top_p: (Optional) Controls diversity of the response using nucleus sampling; only tokens with the top cumulative
        probability of `top_p` are considered. Defaults to 1.0.
    :param top_k: (Optional) Limits responses to the top `top_k` most probable tokens, determining response diversity.
        Defaults to 50.
    :param seed: (Optional) Optional random seed for reproducibility.
    :return: Loaded LlamaCpp model object configured with specified parameters.
    :rtype: LlamaCpp
    """
    LOGGER.info(f"Loading LlamaCpp model from %s...", model_name)

    # Load the Llama model using the LlamaCpp library
    # More info: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html
    # When using LlamaCPP the tokenizer is included in the model, so we don't
    # need to load it separately.
    # We also do not create a separate pipeline for the model, as the LlamaCpp
    # library handles this for us.
    # https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
    params = {
        "model_path": model_name,  # The model to load
        # https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#adjusting-the-context-window
        "n_ctx": 4096,
        "n_gpu_layers": 512,  # The number of layers to put on the GPU, we probably need to tweak this
        "n_batch": 30,  # The batch size, which is how many tokens to process at once by the model
        "n_parts": 1,  # The number of parts to split the model into, almost always 1
        # The repetition penalty to use for generation, which controls the diversity of
        # generated responses
        "repeat_penalty": 1.176,
        # MUST set to True, otherwise you will run into problem after a couple of calls
        "f16_kv": True,  # Whether to use 16-bit floating point for the key/value vectors
    }
    if seed:
        params["seed"] = (
            seed  # The random seed to use for generation, which helps with reproducibility
        )
    if top_p:
        # The top p value to use for generation, which controls the
        # diversity of generated responses
        params["top_p"] = top_p
    if top_k:
        # The number of most likely next words in a pool to choose from for generation
        params["top_k"] = top_k
    if temperature:
        # The temperature to use for generation, which controls the randomness of
        # generated responses
        params["temperature"] = temperature
    if max_tokens:
        # The maximum number of tokens to generate, which controls the length of
        # generated responses
        params["max_tokens"] = max_tokens

    # ChatLlamaCpp states that it does not currently support automatic tool calling
    # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/#tool-calling
    # It can invoke tools, but only if you explicitly set the 'tool choice' parameter
    # However, LlamaCpp recently added tool call support, so maybe this is changing:
    # https://github.com/ggml-org/llama.cpp/pull/9639
    llm = ChatLlamaCpp(**params)  # pylint: disable=E1102

    # Print the model for debugging purposes
    LOGGER.debug(llm)

    return llm


# This function creates a GCP Vertex AI model for inference.
# Install GCP CLI:
# https://cloud.google.com/sdk/docs/install
# For this method to work, it requires a GCP credentials to be available in the environment
# More info: https://cloud.google.com/docs/authentication/application-default-credentials
# 1. gcloud init
# 2. gcloud components update
# 3. gcloud components install beta
# 4. gcloud auth application-default login
# Pricing info:
# https://cloud.google.com/vertex-ai/pricing
# Getting started documentation showing how to enable Vertex API in your project:
# https://cloud.google.com/vertex-ai/docs/start/cloud-environment
# Google's own privacy statement shows that unless you explicitly opt-in, no data
# is collected from your model for training purposes.
# https://cloud.google.com/vertex-ai/docs/generative-ai/data-governance
# https://cloud.google.com/terms/service-terms
#    "16. Training Restriction. Google will not use Customer Data to train or
#    fine-tune any AI/ML models
#    without Customer's prior permission or instruction."
@conditional_cache_resource()
def load_remote_gcp_vertex_model(
    model_name, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None
):
    """
    Loads a remote GCP Vertex AI model with the specified configuration parameters.

    This function creates a model configuration based on the given arguments and
    initializes a ChatVertexAI instance with it. Optional parameters such as
    top_p, top_k, and seed can be added to further customize the model setup.
    A debug log of the initialized model is generated before returning the model.

    :param model_name: The name of the model to be loaded.
    :type model_name: str
    :param max_tokens: Maximum number of tokens for the model's output.
    :type max_tokens: int, optional
    :param temperature: Optional. Sampling temperature for generating responses.
    :type temperature: float, optional
    :param top_p: Optional. The nucleus sampling probability for response
                  generation.
    :type top_p: float, optional
    :param top_k: Optional. The top-k sampling value for response generation.
    :type top_k: int, optional
    :param seed: Optional. Seed for reproducibility in model output.
    :type seed: int, optional
    :return: An instance of the ChatVertexAI model initialized with the
             specified configuration.
    :rtype: ChatVertexAI
    """

    llm_config = {
        "model_name": model_name,
    }
    if max_tokens:
        llm_config["max_output_tokens"] = max_tokens
    if temperature:
        llm_config["temperature"] = temperature
    if top_p:
        llm_config["top_p"] = top_p
    if top_k:
        llm_config["top_k"] = top_k
    if seed:
        llm_config["seed"] = seed

    llm = ChatVertexAI(**llm_config)

    # Print the model for debugging purposes
    LOGGER.debug(llm)

    return llm


@conditional_cache_resource()
def load_remote_bedrock_model(
    model_name, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None
):
    """
    Initializes and loads a remote bedrock model from AWS Bedrock service.

    This function sets up a language model using specified configurations such as
    model name, maximum tokens, temperature, and optionally top-p, top-k, or a seed
    for reproducibility. It creates and configures the model, logging the initialization
    process and returning the created model.

    :param model_name: The name or ID of the model to initialize.
    :type model_name: str
    :param max_tokens: (Optional) Maximum number of tokens the model should generate.
    :type max_tokens: int
    :param temperature: (Optional) The temperature setting for generation, controlling output randomness.
    :type temperature: float
    :param top_p: (Optional) Cumulative probability threshold for nucleus sampling.
    :type top_p: float, optional
    :param top_k: (Optional) Maximum number of top probable next tokens to consider during generation.
    :type top_k: int, optional
    :param seed: (Optional) A seed value to ensure deterministic behavior of the model.
    :type seed: int, optional
    :return: An instance of the language model configured with provided parameters.
    :rtype: ChatBedrockConverse
    """
    LOGGER.info(f"Initializing AWS Bedrock model: %s...", model_name)

    llm_config = {
        "model_id": model_name,
    }
    if max_tokens:
        llm_config["max_tokens"] = max_tokens
    if temperature:
        llm_config["temperature"] = temperature
    if top_p:
        llm_config["top_p"] = top_p
    if top_k:
        llm_config["top_k"] = top_k
    if seed:
        llm_config["seed"] = seed

    llm = ChatBedrockConverse(**llm_config)
    LOGGER.debug(llm)
    return llm


@conditional_cache_resource()
def load_remote_azure_openai(
    model_name,
    api_version,
    max_tokens=None,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
):
    """
    Loads and initializes a remote Azure OpenAI model with the specified
    parameters and configurations. This function interacts with the Azure
    OpenAI service, creating a model instance based on the provided
    execution and configuration details. The function leverages Azure-specific
    settings such as deployment name, API version, and other behavioral
    parameters to personalize its runtime behavior.

    This function employs caching to minimize repetitive resource initialization
    through conditional cache decorators, enhancing performance for frequently
    used configurations. Upon successful initialization, the Azure OpenAI
    language model instance is returned for further use.

    :param model_name: Name of the Azure OpenAI model deployment.
    :param api_version: API version to be used for the OpenAI service.
    :param max_tokens: Optional. Maximum number of tokens to generate.
    :param temperature: Optional. Sampling temperature that controls randomness.
    :param top_p: Optional. Nucleus sampling probability. Picks tokens from
        the top p cumulative probability mass, if provided.
    :param top_k: Optional. Top-k sampling that limits the next token
        selection to k most likely options, if specified.
    :param seed: Optional. Random seed for deterministic outputs in sampling.
    :return: An initialized Azure OpenAI language model instance.
    """
    LOGGER.info(
        "Initializing Azure OpenAI model: %s, API version: %s", model_name, api_version
    )

    # Define Azure-specific parameters
    azure_config = {
        "azure_deployment": model_name,
        "api_version": api_version,
    }
    if temperature:
        azure_config["temperature"] = temperature
    if max_tokens:
        azure_config["max_completion_tokens"] = max_tokens
    if top_p:
        azure_config["top_p"] = top_p
    if top_k:
        azure_config["top_k"] = top_k
    if seed:
        azure_config["seed"] = seed

    llm = AzureChatOpenAI(**azure_config)
    LOGGER.debug(llm)
    return llm
