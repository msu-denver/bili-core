"""
Module: opensearch_utils

This module provides utility functions to load and initialize OpenSearch vector
search functionality. It includes functions to configure the necessary OpenSearch endpoint
and AWS credentials, and to create an `OpenSearchVectorSearch` object for performing
vector-based searches.

Functions:
    - load_opensearch_vector_search(embedding_function, index_name):
      Loads and initializes an `OpenSearchVectorSearch` object by configuring the necessary
      OpenSearch endpoint and AWS credentials. Automatically determines the operating environment
      (local or AWS) and sets up the OpenSearch connection accordingly.

Dependencies:
    - os: Provides a way of using operating system dependent functionality.
    - boto3: AWS SDK for Python to interact with AWS services.
    - streamlit: Provides the Streamlit library for caching resources.
    - langchain_community.vectorstores: Imports `OpenSearchVectorSearch` for
    OpenSearch vector search functionality.
    - opensearchpy: Provides the OpenSearch client for Python.
    - requests_aws4auth: Provides AWS4Auth for signing requests to AWS services.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within an application to configure and initialize
    OpenSearch vector search functionality. It provides functions to set up the necessary
    environment and create an `OpenSearchVectorSearch` object for performing similarity searches.

Example:
    from bili.utils.opensearch_utils import load_opensearch_vector_search

    # Initialize the OpenSearch vector search
    opensearch_vector_search = load_opensearch_vector_search(
        embedding_function=my_embedding_function,
        index_name="my_index"
    )
"""

import os

import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@conditional_cache_resource()
def load_opensearch_vector_search(_embedding_function, index_name):
    """
    Loads and initializes an OpenSearchVectorSearch object by configuring the necessary
    OpenSearch endpoint and AWS credentials. Automatically determines the operating
    environment (local or AWS) and sets up the OpenSearch connection accordingly,
    allowing for vector-based search functionality.

    :param _embedding_function: A callable function used to compute
    vector embeddings for input data.
    :param index_name: The name of the OpenSearch index to be used for vector search.
    :return: An instance of `OpenSearchVectorSearch` configured with the provided parameters.
    """
    # Determine if running locally or in AWS
    is_local = os.getenv("LOCALSTACK_HOSTNAME") is not None

    # Set the OpenSearch endpoint
    opensearch_url = os.getenv("OPENSEARCH_URL")
    if not opensearch_url:
        LOGGER.error("OPENSEARCH_URL is not set. Please check environment variables.")
        raise ValueError("Missing required environment variable: OPENSEARCH_URL")

    # Initialize AWS credentials
    if is_local:
        LOGGER.info("Running locally. Using test localstack credentials.")

        # If running locally use a specific key and secret since we are not using IAM roles
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "test")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
        aws_region_name = os.getenv("AWS_REGION", "us-east-1")

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        aws_region_name = os.getenv("AWS_REGION")
        if not aws_region_name:
            LOGGER.error("AWS_REGION is not set. Please check environment variables.")
            raise ValueError("Missing required environment variable: AWS_REGION")

        # Use the default AWS credential provider chain
        session = boto3.Session()

    # Get credentials from the session
    credentials = session.get_credentials()

    # Create a new AWS4Auth object for the OpenSearch client using the Session credentials
    aws_auth = AWS4Auth(
        region=aws_region_name, service="es", refreshable_credentials=credentials
    )

    LOGGER.info("Initializing OpenSearchVectorSearch with index: %s", index_name)
    LOGGER.info("OpenSearch URL: %s", opensearch_url)

    # Initialize the OpenSearchVectorSearch object
    # https://docs.localstack.cloud/user-guide/aws/opensearch/
    # https://aws.amazon.com/opensearch-service/
    docsearch = OpenSearchVectorSearch(
        embedding_function=_embedding_function,
        opensearch_url=opensearch_url,
        http_auth=aws_auth,
        timeout=300,
        use_ssl=not is_local,
        verify_certs=not is_local,
        connection_class=RequestsHttpConnection,
        index_name=index_name,
    )

    return docsearch
