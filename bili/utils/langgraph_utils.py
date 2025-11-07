"""
Module: langgraph_utils

This module provides utility functions and classes for managing and processing
conversation states within the LangGraph framework. It includes functions for
formatting messages with citations and clearing conversation state.

Classes:
--------
- State:
    Extends MessagesState to represent user-specific state, including summary and owner fields.

Functions:
----------
- format_message_with_citations(message):
    Formats an AIMessage, appending citation information from metadata if present.

- clear_state(state: State) -> dict:
    Removes all messages from the given state and returns a dictionary with the
    removed messages and an empty summary.

Dependencies:
-------------
- langchain_core.messages: Provides message classes and utilities for chat history.
- langgraph.graph: Provides MessagesState base class for state schema.
- bili.utils.logging_utils: Logger initialization.

Usage:
------
Import and use these utilities to manage conversation state and format messages
with citations within LangGraph-based conversational systems.

Example:
--------
from bili.utils.langgraph_utils import (
    format_message_with_citations,
    clear_state,
)

# Format a message with citations
formatted = format_message_with_citations(message)

# Clear conversation state
cleared = clear_state(state)
"""

from datetime import datetime
from typing import List

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import MessagesState

from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def format_message_with_citations(message):
    """
    Formats a message along with its citations if present. If the input is not an
    instance of AIMessage, the function will default to returning its pretty
    representation. When citations are available within the message metadata,
    they are appended to the formatted output.

    :param message: The AIMessage instance to format.
    :type message: AIMessage
    :return: The formatted message string, optionally including citations.
    :rtype: str
    """
    if not isinstance(message, AIMessage):
        return message.pretty_repr()

    # Start with the content of the message
    formatted_message = message.pretty_repr()

    # Check for citations in the metadata
    citations = message.response_metadata.get("citation_metadata", {}).get(
        "citations", []
    )
    if citations:
        citation_texts = []
        formatted_message += "\n\n**Citations:**\n"
        for citation in citations:
            if citation.get("title") and citation.get("uri"):
                citation_texts.append(f"- [{citation['title']}]({citation['uri']})")
            elif citation.get("uri"):
                citation_texts.append(f"- [{citation['uri']}]({citation['uri']})")
        if len(citation_texts) > 0:
            formatted_message += "\n".join(citation_texts)

    return formatted_message


class State(MessagesState):
    """
    Represents the state of an agent with user-specific preferences or state data.

    This class extends `MessagesState` and includes additional attributes to track user-specific
    preferences or state like summary, user ownership information, message timestamps,
    conversation metadata (title and tags), as well as the time difference between current
    and previous messages.

    :ivar summary: A text summary associated with the agent state.
    :type summary: str
    :ivar owner: The identifier for the owner linked to this state.
    :type owner: str
    :ivar previous_message_time: The timestamp of the last recorded message.
    :type previous_message_time: datetime
    :ivar current_message_time: The timestamp of the latest message.
    :type current_message_time: datetime
    :ivar delta_time: The calculated time difference, in seconds,
    between the current and previous messages.
    :type delta_time: float
    :ivar disable_summarization: Flag to disable automatic conversation summarization.
    :type disable_summarization: bool
    :ivar template_dict: Dictionary of prompt templates for the conversation.
    :type template_dict: dict
    :ivar title: The title of the conversation thread.
    :type title: str
    :ivar tags: List of tags/categories associated with the conversation.
    :type tags: List[str]
    """

    # If we wanted to keep any user-specific preferences or state, we could add them here
    summary: str
    owner: str
    previous_message_time: datetime
    current_message_time: datetime
    delta_time: float
    disable_summarization: bool
    template_dict: dict
    title: str
    tags: List[str]


def clear_state(state: State) -> dict:
    """
    Clears the messages present in the given state and prepares a response
    containing the list of removed messages and an empty summary. The function
    handles both cases where messages are stored directly in the state or within
    the state's values.

    :param state: The state object containing messages either directly as "messages"
                  or nested within its "values" attribute.
    :type state: State
    :return: A dictionary containing the list of removed messages under the
             "messages" key and an empty summary under the "summary" key.
    :rtype: dict
    """
    # Get messages from either state or state.values depending on the structure of the state
    if "messages" in state:
        messages = state["messages"]
    else:
        messages = state.values.get("messages", [])
    messages_to_remove = [RemoveMessage(id=msg.id) for msg in messages]
    return {"messages": messages_to_remove, "summary": ""}
