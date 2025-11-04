"""
inject_current_date_time.py
---------------------------

This module provides a utility for injecting the current UTC datetime into the
system message of a conversational agent's state. It is designed for use in workflows
where the agent needs access to real-time temporal context to provide accurate
time-sensitive responses.

Functions:
----------
- build_inject_current_date_time(**kwargs):
    Returns a node function that appends the current UTC datetime to the existing
    `SystemMessage` content at the start of the message list in the state.
    If no `SystemMessage` exists at the beginning, the function returns the state unchanged.

Dependencies:
-------------
- datetime: Provides `datetime` and `timezone` classes for current time retrieval.
- langchain_core.messages: Provides `SystemMessage` and `RemoveMessage` classes for
  chat history manipulation.
- bili.utils.langgraph_utils.State: Defines the state schema for conversation data.
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `build_inject_current_date_time` to create a state-processing function
for conversational agents that require current datetime context in their system prompt.

Example:
--------
from bili.nodes.inject_current_date_time import build_inject_current_date_time

inject_datetime_node = build_inject_current_date_time()
updated_state = inject_datetime_node(state)

Note:
-----
This function only operates when a `SystemMessage` already exists as the first message
in the state. It appends the current UTC datetime to the existing system message content
rather than creating a new system message from scratch. If there is no system message
by default this node will do nothing.
"""

from datetime import datetime, timezone

from langchain_core.messages import RemoveMessage, SystemMessage

from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_inject_current_date_time(**kwargs):
    """
    Builds a node function to inject current UTC datetime into the system message.

    This function creates a node that appends the current UTC datetime to an existing
    `SystemMessage` at the beginning of the conversation state. This enables the agent
    to have access to real-time temporal context for time-sensitive operations and responses.

    The function only operates when a `SystemMessage` already exists as the first message.
    If no messages exist or the first message is not a `SystemMessage`, the state is
    returned unchanged.

    Args:
        **kwargs: Additional keyword arguments (currently unused but available for
                 future extensibility).

    Returns:
        function: A function that takes a `State` object and returns a dictionary with
        updated messages that include the current UTC datetime appended to the existing
        system message content.
    """

    def _execute_node(state: dict) -> dict:
        """
        Injects the current UTC datetime into the existing system message.

        This function retrieves the current list of messages from the state and checks
        if the first message is a `SystemMessage`. If so, it appends the current UTC
        datetime to the existing system message content and replaces the original
        system message with the updated version.

        The datetime is formatted as a string representation of the current UTC time
        and is appended with the prefix "The current time in UTC is ".

        :param state: The current state of the conversation represented as a dictionary.
            It must include a "messages" key containing a list of message objects.
        :return: An updated state dictionary with the system message modified to include
            the current UTC datetime, or the original state if no system message exists.
        :rtype: dict
        """
        messages = state.get("messages", [])

        if not messages or not isinstance(messages[0], SystemMessage):
            # If no messages exist or the first message is not a SystemMessage, return unchanged
            LOGGER.trace(
                "No SystemMessage found at position 0, skipping datetime injection"
            )
            return state

        # Get the current datetime in UTC
        current_time = datetime.now(timezone.utc)
        LOGGER.trace(f"Injecting current UTC datetime: {current_time}")

        # Create updated system message with current datetime appended
        updated_message = (
            messages[0].content + " The current time in UTC is " + str(current_time)
        )
        system_message = SystemMessage(content=updated_message)

        # Remove the old system message and insert the new one
        messages.append(RemoveMessage(id=messages[0].id))
        messages.insert(0, system_message)

        LOGGER.trace(
            f"Updated system message with datetime: {system_message.content[:100]}..."
        )

        return {"messages": messages}

    return _execute_node
