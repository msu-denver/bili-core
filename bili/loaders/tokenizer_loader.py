"""
tokenizer_loader.py
--------------------

This module provides functions to load and initialize tokenizers for
various language models using the HuggingFace Transformers library.
The tokenizers are configured for high performance and can
handle long input sequences.

Functions:
----------
- load_huggingface_tokenizer(model_name):
    Loads a tokenizer for the specified HuggingFace model, configured
    for high performance and long input sequences.

Dependencies:
-------------
- transformers.AutoTokenizer: Used to load the tokenizer for the specified model.
- bili.streamlit_ui.utils.streamlit_utils.conditional_cache_resource:
Decorator to cache resources conditionally.

Usage:
------
To use the functions provided in this module, import the necessary
functions and call them with appropriate
parameters as shown in the examples below:

Example:
--------
from bili.loaders.tokenizer_loader import load_huggingface_tokenizer

# Load a tokenizer for a specific model
tokenizer = load_huggingface_tokenizer(model_name="gpt-3")
"""

from transformers import AutoTokenizer

from bili.streamlit_ui.utils.streamlit_utils import conditional_cache_resource
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


# This function initializes and loads the Llama tokenizer for CUDA-compatible machines.
@conditional_cache_resource()
def load_huggingface_tokenizer(model_name):
    """
    Loads a tokenizer for the specified HuggingFace model. The tokenizer is
    configured for high performance with the `use_fast=True` setting and
    is specifically adjusted to handle long input sequences by extending
    the maximum model length.

    :param model_name: The name of the model for which the tokenizer should be loaded.
    :type model_name: str
    :return: Tokenizer object configured for the specified model.
    :rtype: transformers.PreTrainedTokenizer
    """
    # Initialize the tokenizer for the Llama model
    # The tokenizer is used to convert text to tokens that the model can understand.
    # The tokenizer is also used to convert the model's output tokens back to text.
    # The tokenizer is loaded with the 'use_fast=True' parameter to optimize performance.
    # More info:
    # https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # Because our prompts can get pretty long when created by langchain,
    # we need to increase the max length of the tokenizer's input.
    # Need to refine tokenizer max length
    tokenizer.model_max_length = 5120
    tokenizer.truncation = True

    # Print the tokenizer config for debugging purposes
    LOGGER.debug(tokenizer)

    return tokenizer
