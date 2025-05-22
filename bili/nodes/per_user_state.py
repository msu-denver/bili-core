"""
per_user_state.py
-----------------

This module provides a utility for injecting per-user information
into the conversation state of a conversational agent. It is designed
for workflows where the agent's responses should be
personalized based on the current user's profile.

Functions:
----------
- buld_per_user_state_node(current_user: dict = None, **kwargs):
    Returns a node function that inserts a `HumanMessage` containing the user's profile
    (as a JSON string) into the message list of the conversation state. If a user profile
    message already exists, it is replaced. If no user is provided, the state is returned
    unmodified.

Dependencies:
-------------
- langchain_core.messages: Provides `HumanMessage`, `SystemMessage`,
and `RemoveMessage` classes for chat history manipulation.
- bili.utils.langgraph_utils.State: Defines the state schema for conversation data.
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `buld_per_user_state_node` to create a state-processing function for
personalizing agent workflows with user profile information.

Example:
--------
from bili.nodes.per_user_state import buld_per_user_state_node

per_user_node = buld_per_user_state_node(current_user={"uid": "user123", "name": "John Doe"})
new_state = per_user_node(state)
"""

import json

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def buld_per_user_state_node(current_user: dict = None, **kwargs):
    """
    Builds a per-user state node by adding relevant user information to the messages.

    This function serves as a factory for creating a closure that encapsulates logic to
    add user-specific information to the conversation state. The closure function, when invoked,
    modifies the state by inserting user-centric contextual data in the
    form of a personalized message.

    :param current_user: A dictionary containing user details for addition to the state
    :param kwargs: Additional keyword arguments to augment the functionality
    :return: A closure function that processes and updates the state with user information
    :rtype: Callable[[State], dict]
    """

    def add_user_info(state: State) -> dict:
        """
        Adds user information to the current state of messages.

        This function modifies the conversation state by inserting a user profile message.
        The user profile is added as a `HumanMessage` to personalize the conversation and
        provide contextual understanding for the language model. If there is no
        current user, the state is returned without modifications.

        Parameters:
        -----------
        - state (State): The current state of the conversation represented as a dictionary.
          It must include a "messages" key containing a list of message objects.

        Returns:
        --------
        - dict: An updated state dictionary with modified messages reflecting the addition
          of the user profile message.

        Example:
        --------
        state = {
            "messages": [
                SystemMessage(content="System message content"),
                HumanMessage(content="User message content")
            ]
        }
        current_user = {
            "uid": "user123",
            "name": "John Doe"
        }
        updated_state = add_user_info(state)
        """
        # Retrieve the current list of messages from state for processing
        all_messages = state["messages"]
        if not all_messages:
            all_messages = []

        # If there is no current user, return the state as is
        if current_user is None:
            LOGGER.debug(
                "No current user provided. Returning state without modifications."
            )
            return {"messages": all_messages}

        LOGGER.trace(f"Original messages before adding user info: {all_messages}")

        # Add user information to the message list after the first SystemMessage
        # Convert user dictionary to JSON string for LLM processing
        user_json = json.dumps(current_user, indent=0)
        profile_prefix = "USER PROFILE: "
        profile_msg = (
            f"{profile_prefix}The following information is the profile of the user having the conversation. "
            "This information is used to personalize the conversation, and should be "
            f"referenced when generating responses. Profile details: {user_json}"
        )

        profile_info = HumanMessage(content=profile_msg)

        # Check if there is already a HumanMessage at position 1 that has the same prefix. If so, remove it.
        if len(all_messages) > 1 and isinstance(all_messages[1], HumanMessage):
            if all_messages[1].content.startswith(profile_prefix):
                all_messages.append(RemoveMessage(id=all_messages[1].id))

        # Insert the new profile message after the first SystemMessage
        if len(all_messages) > 0 and isinstance(all_messages[0], SystemMessage):
            all_messages.insert(1, profile_info)
        else:
            all_messages.insert(0, profile_info)

        LOGGER.trace(f"Messages after adding user info: {all_messages}")

        return {"messages": all_messages}

    return add_user_info
