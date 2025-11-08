"""
normalize_state.py
-----------------------

This module provides a utility for normalizing tool function
calls in conversation state messages. It is intended for use in
conversational agent workflows where compatibility with multiple LLM
providers is required, especially those that do not
support mixing tool calls with function calls.

Functions:
----------
- build_normalize_state_node(**kwargs):
    - Returns a node function that processes the conversation state
    by removing the "function_call"
    entry from the `additional_kwargs` of each message, if present.
    - This ensures that redundant
    tool calls are eliminated while retaining tool information
    in the "tool_calls" field.
    - Also removes any empty AI messages to prevent and invalid message
    from existing. In vertex if an AI message is empty it will cause a
    ```InvalidArgument: 400 Unable to submit request because it must include 
    at least one parts field, which describes the prompt input.``` 
    
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
from functools import partial
from bili.graph_builder.classes.node import Node
from langchain_core.messages import AIMessage, RemoveMessage

from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_normalize_state_node(**kwargs):
    """
    Creates a LangGraph node that normalizes tool function calls in the conversation state.

    This includes:
    - Removing the `function_call` entry from `additional_kwargs` in each message, if present
    - Identifying and removing AI messages with `invalid_tool_calls`, using `RemoveMessage` objects

    Returns:
        function: A LangGraph-compatible node function that accepts a state dictionary and
                  returns a modified state dictionary with `RemoveMessage` instructions.
    """

    def normalize_state(state: dict) -> dict:
        """
        Normalizes the state of tool messages by inspecting and modifying their content
        and flagging invalid messages for removal. This function processes tool-related
        messages within a dictionary, removing unnecessary information and marking
        invalid messages for deletion.

        :param state: A dictionary containing the state of tool messages. The "messages"
            entry in the dictionary should be a list of message objects that may include
            AI-generated messages with potential invalid tool calls or redundant information
            in their additional arguments.
        :type state: dict

        :return: A dictionary containing an updated state where invalid or redundant
            tool-related messages have been appropriately marked or modified. The
            "messages" key of the returned dictionary contains the list of messages
            flagged for removal.
        :rtype: dict
        """
        all_messages = state["messages"]
        LOGGER.trace(
            "Original messages before normalizing tool calls: %s", all_messages
        )

        messages_to_remove = []

        for message in all_messages:
            # Normalize: Remove function_call from additional_kwargs
            if (
                message.additional_kwargs
                and "function_call" in message.additional_kwargs
            ):
                LOGGER.debug(
                    "Removing redundant function_call from message: %s", message
                )
                del message.additional_kwargs["function_call"]

            # Detect and flag invalid tool call messages for removal
            if isinstance(message, AIMessage) and getattr(
                message, "invalid_tool_calls", False
            ):
                LOGGER.debug(
                    "Marking AI message for removal due to invalid tool calls: %s",
                    message,
                )
                messages_to_remove.append(RemoveMessage(id=message.id))

            # If an AI message exists and the content is empty mark for deletion
            if isinstance(message, AIMessage) and message.content == "":
                LOGGER.debug(
                    "Marking AI message for removal due to empty content field: %s",
                    message,
                )
                messages_to_remove.append(RemoveMessage(id=message.id))

        # Only return message removals if there are messages to remove
        # Returning empty list would clear all messages from state
        if messages_to_remove:
            return {"messages": messages_to_remove}
        else:
            return {}  # No state changes

    return normalize_state

normalize_state_node = partial(Node, "normalize_state", build_normalize_state_node)