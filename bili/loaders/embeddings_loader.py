"""
Module: embeddings_loader

This module provides functions to load and create embedding functions for various providers.
It supports embedding functions from Google Vertex AI, Amazon Bedrock, Azure OpenAI, and
Sentence Transformers. The functions are conditionally cached to optimize resource usage
in Streamlit applications.

Functions:
    - load_embedding_function(provider, model_name=None):
      Loads and returns the appropriate embedding function based on the specified provider.
    - create_sentence_transformer_embedding_function(model_name="bert-large-nli-mean-tokens"):
      Creates a sentence transformer embedding function using the specified model.
    - create_azure_openai_embedding_function(model_name="azure_text-embedding-3-large"):
      Creates an Azure OpenAI embedding function using the specified model.
    - create_amazon_bedrock_embedding_function(model_name="amazon_titan-embed-text-v2"):
      Creates an Amazon Bedrock embedding function using the specified model.
    - create_google_vertexai_embedding_function(model_name="vertex_text-embedding-005"):
      Creates a Google Vertex AI embedding function using the specified model.

Dependencies:
    - langchain_aws: Provides BedrockEmbeddings for Amazon Bedrock.
    - langchain_community.embeddings: Provides SentenceTransformerEmbeddings
    for sentence transformers.
    - langchain_google_vertexai: Provides VertexAIEmbeddings for Google Vertex AI.
    - langchain_openai: Provides AzureOpenAIEmbeddings for Azure OpenAI.
    - bili.streamlit.utils.streamlit_utils: Imports `conditional_cache_resource` for caching.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within applications that require embedding functions
    from various providers. It provides functions to load and create embedding functions
    with conditional caching to optimize resource usage.

Example:
    from bili.loaders.embeddings_loader import load_embedding_function

    # Load an embedding function for Google Vertex AI
    embedding_function = load_embedding_function(provider="vertex",
        model_name="vertex_text-embedding-005")
"""

from langchain_aws import BedrockEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@conditional_cache_resource()
def load_embedding_function(provider, model_name=None):
    """
    Loads and returns the appropriate embedding function based on the provider specified.
    This function conditionally creates an embedding function for providers such as
    Google Vertex AI, Amazon Bedrock, Azure OpenAI, or a Sentence Transformer.

    :param provider: The name of the embedding function provider
        (e.g., 'vertex', 'amazon', 'azure', or 'sentence_transformer').
    :type provider: str
    :param model_name: Name of the specific model to be used by the provider.
        If not specified, the provider may use its default model.
    :type model_name: str, optional
    :return: The embedding function corresponding to the specified provider and model.
    :rtype: Callable
    :raises ValueError: If an invalid provider name is specified.
    """
    if provider == "vertex":
        embedding_function = create_google_vertexai_embedding_function(
            model_name=model_name
        )
    elif provider == "amazon":
        embedding_function = create_amazon_bedrock_embedding_function(
            model_name=model_name
        )
    elif provider == "azure":
        embedding_function = create_azure_openai_embedding_function(
            model_name=model_name
        )
    elif provider == "openai":
        embedding_function = create_openai_embedding_function(model_name=model_name)
    elif provider == "sentence_transformer":
        embedding_function = create_sentence_transformer_embedding_function(
            model_name=model_name
        )
    else:
        LOGGER.error("Invalid index name provided for AWS OpenSearch retriever.")
        raise ValueError("Invalid index name provided for AWS OpenSearch retriever.")

    return embedding_function


@conditional_cache_resource()
def create_sentence_transformer_embedding_function(
    model_name="bert-large-nli-mean-tokens",
):
    """
    Creates a sentence transformer embedding function using a specified model.
    This function provides a way to generate sentence embeddings, which are vector
    representations capturing the semantic meaning of sentences. By default, it uses
    the 'bert-large-nli-mean-tokens' model for high accuracy, but other models can be
    specified to suit different performance and resource requirements. Refer to
    https://www.sbert.net/docs/pretrained_models.html or
    https://huggingface.co/sentence-transformers for model details.

    :param model_name: The name of the model to use for generating sentence embeddings.
                       Defaults to 'bert-large-nli-mean-tokens'.
    :type model_name: str
    :return: An instance of SentenceTransformerEmbeddings initialized with the specified model name.
    :rtype: SentenceTransformerEmbeddings
    """
    # Using 'all-MiniLM-L6-v2' model for generating sentence embeddings.
    # Sentence embeddings are vector representations of sentences
    # that capture their semantic meaning.
    # It's a compact and efficient model. More: https://www.sbert.net/docs/pretrained_models.html
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Using 'bert-large-nli-mean-tokens' as the default model for generating sentence embeddings.
    # Sentence embeddings are vector representations of sentences
    # that capture their semantic meaning.
    # It's a larger and more accurate model. More:
    # https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens
    return SentenceTransformerEmbeddings(model_name=model_name)


# Only add the annotation if Streamlit ScriptRunContext is present
@conditional_cache_resource()
def create_azure_openai_embedding_function(model_name="azure_text-embedding-3-large"):
    """
    Creates an Azure OpenAI Embedding Function with a specified model.

    This function utilizes the Azure OpenAI service through LangChain for
    generating embeddings. The model to be used for embeddings can be customized
    using the parameter.

    :param model_name: The name of the model to be used for generating embeddings. Defaults to
        'azure_text-embedding-3-large'.
    :type model_name: str
    :return: An instance of AzureOpenAIEmbeddings configured with the specified model.
    :rtype: AzureOpenAIEmbeddings
    """
    # https://python.langchain.com/docs/integrations/text_embedding/azureopenai/
    return AzureOpenAIEmbeddings(model=model_name)


@conditional_cache_resource()
def create_openai_embedding_function(model_name="text-embedding-3-large"):
    """
    Creates an OpenAI Embedding Function with a specified model.

    This function utilizes the OpenAI service through LangChain for
    generating embeddings. The model to be used for embeddings can be customized
    using the parameter.

    :param model_name: The name of the model to be used for generating embeddings. Defaults to
        'text-embedding-3-large'.
    :type model_name: str
    :return: An instance of OpenAIEmbeddings configured with the specified model.
    :rtype: OpenAIEmbeddings
    """
    # https://python.langchain.com/docs/integrations/text_embedding/openai/
    return OpenAIEmbeddings(model=model_name)


@conditional_cache_resource()
def create_amazon_bedrock_embedding_function(model_name="amazon_titan-embed-text-v2"):
    """
    Creates an embedding function that utilizes Amazon Bedrock for embedding text.

    :param model_name: The identifier of the model to be used for text embeddings.
                      Defaults to "amazon-titan-embed-text-v2".
    :return: An instance of BedrockEmbeddings initialized with the specified model.
    :rtype: BedrockEmbeddings
    """
    # https://python.langchain.com/docs/integrations/text_embedding/bedrock/
    return BedrockEmbeddings(model_id=model_name)


@conditional_cache_resource()
def create_google_vertexai_embedding_function(
    model_name="azure_text-embedding-3-large",
):
    """
    Creates a Google Vertex AI embedding function using the specified model name.
    This function leverages the Google Vertex AI PaLM for generating embeddings and
    can be customized by providing the desired model name as an argument.

    Decorated with the `conditional_cache_resource`, this function ensures that resources
    are loaded efficiently by utilizing a caching mechanism.

    :param model_name: The name of the model to be used for generating embeddings.
                      Defaults to "azure_text-embedding-3-large".
    :type model_name: str

    :return: An instance of `VertexAIEmbeddings` configured with the specified model name.
    :rtype: VertexAIEmbeddings
    """
    # https://python.langchain.com/docs/integrations/text_embedding/google_vertex_ai_palm/
    return VertexAIEmbeddings(model=model_name)
