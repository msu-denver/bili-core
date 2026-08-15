"""
Module: llm_loader

This module provides functions to load and initialize various language models for LangChain.
It supports local models (LlamaCpp, HuggingFace) and remote models
(Google Vertex AI, AWS Bedrock, Azure OpenAI, OpenAI direct).

The six built-in provider types are dispatched directly by ``load_model()``.
Additional provider types — registered at application startup via
``bili.iris.providers.register_provider()`` — are discovered through the
``bili.iris.providers.PROVIDER_REGISTRY`` when the built-in dispatch does not
match.  This allows third-party provider implementations to integrate without
modifying this module.

Functions:
    - load_model(model_type, **kwargs):
      Loads a machine learning model based on the provided model type.
    - prepare_runtime_config(model_type, thinking_config=None, **kwargs):
      Transform simple thinking configuration dict into provider-specific runtime config
      for use with model.invoke().
    - load_huggingface_model(model_name, max_tokens, temperature, top_p=0.1, top_k=None, seed=None):
      Loads a locally available HuggingFace model and initializes a text generation pipeline.
    - load_llamacpp_model(model_name, max_tokens, temperature, top_p=1.0, top_k=50, seed=None):
      Loads a compatible model using the LlamaCpp library with specified configuration options.
    - load_remote_gcp_vertex_model(model_name, max_tokens, temperature,
            top_p=None, top_k=None, seed=None, additional_headers=None):
      Loads a remote GCP Vertex AI model with the specified configuration parameters.
      Supports additional_headers for Priority PayGo and Provisioned Throughput.
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
    from bili.iris.loaders.llm_loader import load_model

    # Load a local HuggingFace model
    model = load_model(
        model_type="local_huggingface",
        model_name="gptq_model",
        max_tokens=100,
        temperature=0.7
    )

    # Load a provider registered at startup via register_provider()
    from bili.iris.providers import register_provider
    from mypackage import MyProvider
    register_provider("remote_my_api", MyProvider)
    model = load_model("remote_my_api", model_name="my-model")
"""

import gc

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _log_available_device() -> None:
    """Log which compute device is available (Apple MPS, CUDA GPU, or CPU).

    Deferred to call-time so that importing this module does not eagerly pull
    in ``torch``, which is only needed for the local HuggingFace/LlamaCpp
    backends.  Cloud-only deployments (Bedrock, Vertex, OpenAI) pay no cost.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.backends.mps.is_available():
            LOGGER.info("Apple MPS device found")
        elif torch.cuda.is_available():
            LOGGER.info("Nvidia GPU device found")
        else:
            LOGGER.info(
                "No compatible GPU device found, CPU will be used for inference."
            )
    except ImportError:
        LOGGER.debug("torch not installed; skipping device detection.")


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

    :param model_type: Specifies the type of the model to be loaded. Built-in
        supported types are ``"local_llamacpp"``, ``"local_huggingface"``,
        ``"remote_google_vertex"``, ``"remote_aws_bedrock"``,
        ``"remote_azure_openai"``, and ``"remote_openai"``.  Additional types
        registered via ``bili.iris.providers.register_provider()`` are also
        supported through the provider registry.
    :type model_type: str
    :param kwargs: Additional keyword arguments specific to the loader function for
        the chosen model type. These arguments differ depending on the model type.
    :type kwargs: dict
    :return: The loaded model object as returned by the appropriate model loader
        function. The return value may differ in format depending on the chosen
        model type, but always exposes an ``.invoke(messages)`` method.
    :rtype: object
    :raises ValueError: If the specified model_type is not one of the built-in
        types and is not registered in the provider registry, or if
        ``structured_output_schema`` is requested for a provider type without
        decode-time schema enforcement.
    """
    # Fail fast on structured-output requests the provider cannot honour.
    # Silently returning an unconstrained model would let the caller believe
    # generation is schema-constrained when it is not -- the exact failure
    # the capability exists to prevent.
    if kwargs.get("structured_output_schema") is not None:
        from bili.iris.providers.structured_output import (  # pylint: disable=import-outside-toplevel
            require_structured_output_support,
        )

        require_structured_output_support(model_type)

    # Primary temperature handling: omit `temperature` for a cataloged model
    # whose definition declares it unsupported (current reasoning models 400 on
    # it).  Passthrough (uncataloged) models keep their temperature and rely on
    # the runtime retry applied below.  See temperature_resilience.
    if kwargs.get("temperature") is not None:
        from bili.iris.providers.temperature_resilience import (  # pylint: disable=import-outside-toplevel
            model_supports_temperature,
        )

        if model_supports_temperature(model_type, kwargs.get("model_name")) is False:
            LOGGER.info(
                "Model '%s' is cataloged as not supporting temperature; omitting it.",
                kwargs.get("model_name"),
            )
            kwargs.pop("temperature")

    # Built-in provider dispatch — handles the six types shipped with bili-core.
    # This if/elif block is intentionally preserved for backward compatibility;
    # the individual loader functions are part of the public API and may be
    # imported and called directly by consumers (e.g. sustainability-hub-engine).
    #
    # Follow-up (delegation refactor): each branch below duplicates logic that
    # also lives in the corresponding LLMProvider class in bili.iris.providers.
    # The clean fix is to delegate these branches to provider_class().load(**kwargs)
    # so there is one implementation per backend. The blocker is the
    # @conditional_cache_resource() decorator on each load_* function — removing
    # it would silently change caching behavior for existing callers. Once a
    # cache-aware delegation path exists (or caching is lifted into the caller),
    # replace this block with: provider_class = PROVIDER_REGISTRY.get(model_type);
    # llm_model = provider_class().load(**kwargs).
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
    elif model_type == "remote_openai":
        llm_model = load_remote_openai(**kwargs)
    else:
        # Fall through to the provider registry for third-party / extended
        # provider types registered at startup via register_provider().
        # Lazy import to avoid circular dependency and module-level overhead.
        from bili.iris.providers.registry import (  # pylint: disable=import-outside-toplevel
            PROVIDER_REGISTRY,
        )

        provider_class = PROVIDER_REGISTRY.get(model_type)
        if provider_class is None:
            raise ValueError(f"Invalid model type: {model_type}")
        LOGGER.info(
            "Delegating model_type '%s' to registered provider: %s",
            model_type,
            provider_class.__name__,
        )
        llm_model = provider_class().load(**kwargs)

    # Make the loaded model self-heal when a provider rejects ``temperature``
    # (current reasoning models 400 on it).  Applied at this single choke point
    # so every caller -- IRIS, AETHER, and callers that invoke the model
    # directly -- benefits; a no-op for models that set no temperature or are
    # not standard chat models.  See temperature_resilience for the mechanism.
    from bili.iris.providers.temperature_resilience import (  # pylint: disable=import-outside-toplevel
        apply_temperature_resilience,
    )

    return apply_temperature_resilience(llm_model)


def prepare_runtime_config(
    model_type: str, thinking_config: dict = None, **kwargs
) -> dict:
    """
    Transform simple thinking configuration dict into provider-specific runtime config.

    This function prepares the runtime configuration that will be passed to the
    model's invoke() method. Different LLM providers require different configuration
    formats - for example, Google Vertex AI uses a ThinkingConfig object while
    other providers may ignore thinking parameters entirely.

    :param model_type: The type of model being used (e.g., "remote_google_vertex",
        "remote_openai", "remote_aws_bedrock"). This determines how the config
        will be formatted.
    :type model_type: str
    :param thinking_config: Optional dictionary containing thinking-related parameters.
        Expected keys may include "budget" for thinking budget. The structure should
        be simple and provider-agnostic (e.g., {"budget": 0}).
    :type thinking_config: dict or None
    :param kwargs: Additional runtime configuration parameters to include in the
        returned config dictionary.
    :type kwargs: dict
    :return: Provider-specific runtime configuration dictionary formatted for use
        with model.invoke(config=...). For Google Vertex AI, this includes a nested
        "thinking_config" key with ThinkingConfig object. For other providers, this
        may be empty or include only the kwargs.
    :rtype: dict

    Example:
        >>> # For Google Vertex AI
        >>> config = prepare_runtime_config(
        ...     model_type="remote_google_vertex",
        ...     thinking_config={"budget": 5000}
        ... )
        >>> # Returns: {"thinking_config": ThinkingConfig(thinking_budget=5000)}

        >>> # For OpenAI (thinking config not supported)
        >>> config = prepare_runtime_config(
        ...     model_type="remote_openai",
        ...     thinking_config={"budget": 0}
        ... )
        >>> # Returns: {}
    """
    runtime_config = {}

    # Handle Google Vertex AI thinking configuration
    if model_type == "remote_google_vertex" and thinking_config:
        try:
            from google.genai import types

            # Extract budget from the thinking_config dict
            budget = thinking_config.get("budget", 0)

            # Only create ThinkingConfig if budget is specified
            if budget is not None:
                # Convert to int if it's a string
                if isinstance(budget, str):
                    budget = int(budget)

                runtime_config["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=budget
                )

                LOGGER.debug(
                    f"Created ThinkingConfig with budget={budget} for Vertex AI"
                )
        except ImportError:
            LOGGER.warning(
                "langchain_google_vertexai.types.ThinkingConfig not available. "
                "Thinking configuration will be ignored."
            )
        except (ValueError, TypeError) as e:
            LOGGER.error(
                f"Error creating ThinkingConfig: {e}. "
                f"Budget value: {thinking_config.get('budget')}"
            )
    elif model_type != "remote_google_vertex" and thinking_config:
        LOGGER.warning(
            f"{model_type} thinking configuration is not supported at this time."
            "Thinking configuration will be ignored."
        )

    # Other providers (OpenAI, Azure, Bedrock, etc.) don't use thinking config
    # They will simply ignore unknown config keys, so we don't need special handling

    # Add any additional kwargs to the runtime config
    runtime_config.update(kwargs)

    return runtime_config


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
    # Lazy imports: torch, transformers, and langchain_huggingface are only
    # needed for local HuggingFace inference.  Cloud-only deployments never
    # reach this branch, so they pay no import cost.
    import torch  # pylint: disable=import-outside-toplevel
    from langchain_huggingface.chat_models.huggingface import (  # pylint: disable=import-outside-toplevel
        ChatHuggingFace,
        HuggingFacePipeline,
    )
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        pipeline,
    )

    from bili.iris.loaders.tokenizer_loader import (  # pylint: disable=import-outside-toplevel
        load_huggingface_tokenizer,
    )

    _log_available_device()
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
        # The return_full_text parameter is set to False so that we only
        # get the generated text, not the full output that includes the prompt also.
        return_full_text=False,
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
    # While ChatHuggingFace does say it supports tools, it does not seem to currently support
    # automatic tool calling. The param 'tool_choice' must be explicitly set to call a tool.
    # https://github.com/langchain-ai/langchain/issues/22379
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
    # Lazy import: langchain_community's ChatLlamaCpp is only needed for
    # local LlamaCpp inference; cloud-only deployments pay no import cost.
    from langchain_community.chat_models import (  # pylint: disable=import-outside-toplevel
        ChatLlamaCpp,
    )

    _log_available_device()
    LOGGER.info("Loading LlamaCpp model from %s...", model_name)

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
    model_name,
    max_tokens=None,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    response_schema=None,
    response_mime_type=None,
    structured_output_schema=None,
    additional_headers=None,
    location=None,
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
    :param structured_output_schema: Optional. Cross-provider spelling of
        ``response_schema``: a JSON schema dict (or Pydantic model class) to
        constrain generation to. Sets ``response_schema`` and
        ``response_mime_type="application/json"``. Mutually exclusive with
        passing ``response_schema``/``response_mime_type`` directly.
    :type structured_output_schema: dict or type, optional
    :param additional_headers: Optional. HTTP headers to include in API requests.
        Use for Priority PayGo: {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}
        Use for Provisioned Throughput: {"X-Vertex-AI-LLM-Request-Type": "dedicated"}
    :type additional_headers: dict, optional
    :param location: Optional. GCP region for the Vertex AI endpoint.
        Defaults to us-central1. Use "global" for models that require it
        (e.g., gemini-3-flash-preview).
    :type location: str, optional
    :return: An instance of the ChatVertexAI model initialized with the
             specified configuration.
    :rtype: ChatVertexAI
    """

    # Lazy import: langchain_google_vertexai is only needed when the Vertex AI
    # backend is actually used; installing google-cloud-aiplatform is heavy.
    from langchain_google_vertexai import (  # pylint: disable=import-outside-toplevel
        ChatVertexAI,
    )

    if structured_output_schema is not None and (
        response_schema is not None or response_mime_type is not None
    ):
        raise ValueError(
            "Pass either structured_output_schema (cross-provider) or "
            "response_schema/response_mime_type (Vertex-native), not both."
        )
    if structured_output_schema is not None:
        from bili.iris.providers.structured_output import (  # pylint: disable=import-outside-toplevel
            normalize_schema,
        )

        response_schema = normalize_schema(structured_output_schema)
        response_mime_type = "application/json"

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
    if response_mime_type:
        llm_config["response_mime_type"] = response_mime_type
    if response_schema:
        llm_config["response_schema"] = response_schema
    if additional_headers:
        llm_config["additional_headers"] = additional_headers
    if location:
        llm_config["location"] = location

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
    # Lazy import: langchain_aws is only needed when the Bedrock backend is used.
    from langchain_aws import (  # pylint: disable=import-outside-toplevel
        ChatBedrockConverse,
    )

    LOGGER.info("Initializing AWS Bedrock model: %s...", model_name)

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
    structured_output_schema=None,
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
    :param structured_output_schema: Optional. JSON schema dict (or Pydantic
        model class) to constrain generation to, bound as ``response_format``
        with ``strict: true``. When set, the returned object is a
        ``RunnableBinding`` (no ``bind_tools``).
    :return: An initialized Azure OpenAI language model instance.
    """
    # Lazy import: langchain_openai is only needed when the Azure OpenAI backend
    # is actually used.
    from langchain_openai import (  # pylint: disable=import-outside-toplevel
        AzureChatOpenAI,
    )

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
    if structured_output_schema is not None:
        from bili.iris.providers.structured_output import (  # pylint: disable=import-outside-toplevel
            normalize_schema,
            openai_response_format,
        )

        llm = llm.bind(
            response_format=openai_response_format(
                normalize_schema(structured_output_schema)
            )
        )
    LOGGER.debug(llm)
    return llm


@conditional_cache_resource()
def load_remote_openai(
    model_name,
    max_tokens=None,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    max_retries=None,
    structured_output_schema=None,
):
    """
    Loads and initializes a remote OpenAI model with the specified
    parameters and configurations. This function interacts with the OpenAI
    service, creating a model instance based on the provided
    execution and configuration details. The function leverages OpenAI-specific
    settings such as deployment name, API version, and other behavioral
    parameters to personalize its runtime behavior.

    This function employs caching to minimize repetitive resource initialization
    through conditional cache decorators, enhancing performance for frequently
    used configurations. Upon successful initialization, the OpenAI
    language model instance is returned for further use.

    :param model_name: Name of the OpenAI model deployment.
    :param max_tokens: Optional. Maximum number of tokens to generate.
    :param temperature: Optional. Sampling temperature that controls randomness.
    :param top_p: Optional. Nucleus sampling probability. Picks tokens from
        the top p cumulative probability mass, if provided.
    :param top_k: Optional. Top-k sampling that limits the next token
        selection to k most likely options, if specified.
    :param seed: Optional. Random seed for deterministic outputs in sampling.
    :param structured_output_schema: Optional. JSON schema dict (or Pydantic
        model class) to constrain generation to, bound as ``response_format``
        with ``strict: true`` (OpenAI structured outputs). When set, the
        returned object is a ``RunnableBinding`` (no ``bind_tools``).
    :return: An initialized OpenAI language model instance.
    """
    # Lazy import: langchain_openai is only needed when the OpenAI backend is used.
    from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel

    LOGGER.info("Initializing OpenAI model: %s", model_name)

    # Define OpenAI-specific parameters
    openai_config = {
        "model": model_name,
    }
    if temperature:
        openai_config["temperature"] = temperature
    if max_tokens:
        openai_config["max_completion_tokens"] = max_tokens
    if top_p:
        openai_config["top_p"] = top_p
    if top_k:
        openai_config["top_k"] = top_k
    if seed:
        openai_config["seed"] = seed
    if max_retries:
        openai_config["max_retries"] = max_retries

    llm = ChatOpenAI(**openai_config)
    if structured_output_schema is not None:
        from bili.iris.providers.structured_output import (  # pylint: disable=import-outside-toplevel
            normalize_schema,
            openai_response_format,
        )

        llm = llm.bind(
            response_format=openai_response_format(
                normalize_schema(structured_output_schema)
            )
        )
    LOGGER.debug(llm)
    return llm
