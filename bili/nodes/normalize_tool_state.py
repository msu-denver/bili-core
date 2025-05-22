"""
normalize_tool_state.py
-----------------------

This module provides a utility for normalizing tool function
calls in conversation state messages. It is intended for use in
conversational agent workflows where compatibility with multiple LLM
providers is required, especially those that do not
support mixing tool calls with function calls.

Functions:
----------
- build_normalize_tool_state_node(**kwargs):
    Returns a node function that processes the conversation state
    by removing the "function_call"
    entry from the `additional_kwargs` of each message, if present.
    This ensures that redundant
    tool calls are eliminated while retaining tool information
    in the "tool_calls" field.

Dependencies:
-------------
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `build_normalize_tool_state_node` to create a state-processing
function for normalizing tool calls in agent workflows.

Example:
--------
from bili.nodes.normalize_tool_state import build_normalize_tool_state_node

normalize_node = build_normalize_tool_state_node()
new_state = normalize_node(state)
"""

from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_normalize_tool_state_node(**kwargs):
    """
    This function modifies the input state by normalizing tool function calls present
    in the messages. Specifically, it removes the "function_call" entry from the
    `additional_kwargs` of each message if it exists. The goal is to ensure that
    redundant tool calls are eliminated while retaining the tool information
    in the existing "tool_calls" field. This is to ensure compatibility with
    multiple LLM providers, some of which don't allow the mixture of tool calls
    with function calls.

    :return: A new dictionary containing the "messages" key with the potentially
             updated list of messages after normalization.
    :rtype: dict
    """

    def normalize_tool_state(state):
        """
        Normalizes tool function calls in the conversation state.

        This function iterates through the messages in the provided state and removes
        the "function_call" entry from the `additional_kwargs` of each message if it exists.
        This ensures that redundant tool calls are eliminated while retaining the tool
        information in the existing "tool_calls" field.

        :param state: The current state of the conversation containing a list of messages.
        :type state: dict
        :return: A dictionary containing the "messages" key with the potentially updated
                 list of messages after normalization.
        :rtype: dict
        """
        all_messages = state["messages"]
        LOGGER.trace(
            f"Original messages before adding normalizing tool calls: {all_messages}"
        )
        for message in all_messages:
            # Check if the message contains a function call
            if message.additional_kwargs and message.additional_kwargs.get(
                "function_call"
            ):
                LOGGER.debug(f"Normalizing tool call for message: %s", message)
                # Remove the function call from the message, since the tool call is already
                # present in the tool_calls field
                del message.additional_kwargs["function_call"]
        return {"messages": all_messages}

    return normalize_tool_state
