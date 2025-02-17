"""
Module: faiss_memory_indexing

This module provides functionality to create and manage FAISS indexes for
efficient similarity searches. It includes functions to preprocess data,
create FAISS retrievers, and validate file paths.

Functions:
    - create_faiss_retriever(documents):
      Creates a FAISS retriever for document similarity search.
    - load_faiss_index(data_dir):
      Preprocesses data and creates a FAISS index for all documents in the
      specified directory.
    - validate_path(user_input):
      Validates if the provided file path adheres to specified rules and
      security constraints.
    - init_faiss(data_dir="data"):
      Initializes and returns a FAISS retriever for efficient similarity
      search.

Dependencies:
    - os: Provides a way of using operating system dependent functionality.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain_community.document_loaders: Imports various document loaders
      for different file types.
    - langchain_community.vectorstores: Imports FAISS for creating FAISS
      indexes.
    - bili.loaders.embeddings_loader: Imports `load_embedding_function` for
      loading embedding functions.
    - bili.utils.file_utils: Imports `preprocess_directory` for preprocessing
      files in a directory.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within a Streamlit application to
    preprocess data and create FAISS indexes for efficient similarity searches.
    It provides functions to create retrievers, validate paths, and initialize
    FAISS.

Example:
    from bili.tools.faiss_memory_indexing import init_faiss

    # Initialize the FAISS retriever
    faiss_retriever = init_faiss(data_dir="data")
"""

import os

from langchain_community.vectorstores import FAISS

from bili.loaders.embeddings_loader import load_embedding_function
from bili.streamlit.utils.streamlit_utils import conditional_cache_data
from bili.utils.file_utils import preprocess_directory
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define the allowed prefixes for file paths in FAISS indexing for security
# Users can add more prefixes as needed based on their environment
ALLOWED_PREFIXES = [
    os.path.join(PARENT_DIR, "data"),  # Parent two levels up + /data
    "/app/bili/data",  # Absolute path to /app/bili/data inside the Docker container
]


# Creates a FAISS index from embeddings for efficient similarity searches
def create_faiss_retriever(documents):
    """
    Creates and configures a FAISS-based retriever using provided documents.

    This function first loads an embedding function to convert the input
    documents into dense vector embeddings. It then uses FAISS (a library used
    for fast nearest-neighbor search on dense vector data) to build a similarity
    index from the document embeddings. Finally, it converts the FAISS index
    into a retriever that supports document retrieval for similarity-based
    queries. The retriever is customized with specified parameters for the number
    of documents fetched (`fetch_k`) and the number of top documents retrieved (`k`).

    :param documents: A list of documents that will be indexed and used for
        the similarity-based retrieval. Each document should be in a format
        suitable for embedding generation.
    :type documents: list
    :return: An instance of a retriever with a pre-configured FAISS backend.
        The retriever can be used to retrieve documents similar to a given query
        based on the FAISS similarity indexing.
    :rtype: langchain.vectorstores.retrievers.Retriever
    """
    # Create an embedding function to convert documents to embeddings for the FAISS index
    embedding_function = load_embedding_function("sentence_transformer")

    # Create a FAISS index from the embeddings
    # This uses a built-in FAISS library provided by LangChain.
    # FAISS is a library for efficient similarity search and clustering of dense vectors.
    # More info available here:
    # https://python.langchain.com/docs/integrations/vectorstores/faiss
    faiss = FAISS.from_documents(documents=documents, embedding=embedding_function)

    # Convert the FAISS index to a retriever
    # A retriever is a component that can retrieve the most similar documents to a query.
    # The search_kwargs{k} parameter is used to specify the number of documents to return.
    # The search_kwargs{fetch_k} parameter is used to specify the number of
    # documents to search through before filtering to the top k.
    # We need to think about what to use for k and fetch_k!
    # More info available here:
    # https://python.langchain.com/docs/integrations/vectorstores/faiss#using-faiss-as-a-retriever
    return faiss.as_retriever(search_kwargs={"k": 50, "fetch_k": 500})


# Preprocesses data and creates a FAISS index for all documents in the 'data' directory
# This function is cached using Streamlit's cache feature to improve performance.
@conditional_cache_data()
def load_faiss_index(data_dir):
    """
    Preprocesses the data from a given directory and creates a FAISS index retriever.

    This function takes a directory containing documents, preprocesses the provided
    files, and creates a FAISS index retriever. It leverages preprocessing utilities
    and the FAISS library to generate an efficient retrieval index. The function is
    decorated with a caching mechanism to optimize repeated calls with the same input.

    :param data_dir: The directory path containing the input documents to preprocess.
    :type data_dir: str
    :return: A FAISS retriever object for querying the preprocessed document data.
    :rtype: FAISSRetriever
    """
    LOGGER.info("Preprocessing data and creating FAISS index...")

    processed_documents = preprocess_directory(data_dir)

    return create_faiss_retriever(processed_documents)


# Function to validate paths
def validate_path(user_input):
    """
    Validates if the given path is permitted based on predefined allowed prefixes or
    if it is located inside the "data/" subdirectory relative to the working directory.

    This function first normalizes and converts the input path to its absolute form,
    then verifies if the path adheres to the allowed prefixes or is compliant with
    being within a specifically allowed directory.

    :param user_input: A string representing the user-provided file system path.
    :type user_input: str
    :return: A boolean indicating whether the provided path is valid.
    :rtype: bool
    """
    # Normalize the path (resolve any relative components like ../)
    normalized_path = os.path.normpath(user_input)

    # Get the absolute path based on the working directory for relative paths
    if not os.path.isabs(normalized_path):
        normalized_path = os.path.join(os.getcwd(), normalized_path)

    # Check if the path starts with any allowed prefix or is in "data/" subdirectory
    for allowed_prefix in ALLOWED_PREFIXES:
        if normalized_path.startswith(allowed_prefix):
            return True

    # Allow relative paths that start with 'data/' from the working directory
    relative_data_path = os.path.normpath("data/")
    if normalized_path.startswith(os.path.join(os.getcwd(), relative_data_path)):
        return True

    return False


@conditional_cache_data()
def init_faiss(data_dir="data"):
    """
    Initialize a FAISS-based similarity search retriever using the provided
    data directory. This function validates the data directory, preprocesses
    the necessary data, and loads or creates a FAISS index for efficient
    similarity search.

    :param data_dir: The directory path for data to be processed and used
        for initializing the FAISS index. The path must start with 'data/'
        or an allowed prefix.
    :type data_dir: str
    :return: An instance of the FAISS retriever for similarity search.
    :rtype: Any
    :raises ValueError: If the given data directory path is invalid, does
        not follow the expected prefix, or is not allowed.
    """

    # Validate the provided data directory
    if not validate_path(data_dir):
        raise ValueError(
            f"Invalid path: {data_dir}. Path must start with 'data/' "
            f"or one of the allowed prefixes {ALLOWED_PREFIXES}"
        )

    # Preprocess data and create FAISS index for efficient similarity search.
    faiss_retriever = load_faiss_index(data_dir)
    return faiss_retriever
