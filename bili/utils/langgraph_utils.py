"""
Module: langgraph_utils

This module provides utility functions and classes for managing and processing
conversation states within the LangGraph framework. It includes functions for
formatting messages with citations and clearing conversation state.

Classes:
--------
- State:
    Extends MessagesState to represent user-specific state, including summary and owner fields.
    Downstream consumers subclass it rather than adding fields to it; see the class
    docstring and bili.iris.loaders.langchain_loader.build_agent_graph.

Re-exports:
-----------
- UntrackedValue:
    LangGraph channel marker that makes a state field ephemeral (never checkpointed).
    Re-exported here so a consumer declares ephemeral fields against bili-core's
    surface instead of reaching into the LangGraph version this package pins.

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
- langgraph.channels: Provides the UntrackedValue channel re-exported here.
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
from langgraph.channels import UntrackedValue
from langgraph.graph import MessagesState

from bili.utils.logging_utils import get_logger

# UntrackedValue is re-exported for consumers, not used inside this module.
# Listing it in __all__ declares it public API; without that, pylint flags
# the import as unused (W0611).
__all__ = [
    "State",
    "UntrackedValue",
    "clear_state",
    "format_message_with_citations",
]

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
    # Use .content directly instead of .pretty_repr() to avoid the
    # "===== Ai Message =====" debug headers that break markdown rendering
    # in Streamlit and add noise to summarization context.
    content = getattr(message, "content", None)
    if content is None:
        return str(message)

    # Handle multimodal content (list of content blocks)
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ]
        content = " ".join(text_parts).strip() or str(message)

    if not isinstance(message, AIMessage):
        return content

    # Start with the content of the message
    formatted_message = content

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

    Extending the state
    -------------------
    A field added to this class lands in the state schema of every downstream
    consumer of bili-core, so application-specific fields do not belong here.
    A consuming application subclasses ``State`` and hands the subclass to
    ``build_agent_graph(state=...)``
    (:func:`bili.iris.loaders.langchain_loader.build_agent_graph`), which
    carries a worked example. That path needs no change to bili-core.

    Persisted vs. ephemeral fields
    ------------------------------
    A field is declared one of two ways, and the default is the expensive one:

    - ``field: T`` is PERSISTED. Its value is written into the checkpoint's
      ``channel_values`` on every checkpoint, and the channel retains that
      value across turns. If no node writes the field on a later turn, the
      previous turn's value survives, so a caller reading
      ``result.get("field")`` sees a value attached to an unrelated turn.
      This is the right shape for durable conversation metadata, which is
      what ``title`` and ``tags`` above are.
    - ``Annotated[T, UntrackedValue]`` is EPHEMERAL. It is never written to
      the checkpoint and does not carry into the next turn. This is the right
      shape for large per-turn payloads and for anything a later turn must not
      inherit. The trade-off is that the value is gone after a restart
      mid-thread, so anything that must survive a restart has to be persisted.

    Both kinds cross node boundaries within a turn and both appear in the final
    ``invoke()`` result; they differ only in whether the value is checkpointed
    and whether it survives into the next turn. Verified against
    ``langgraph==1.0.2``.

    Persistence is not free. A 2,000-feature GeoJSON payload measured ~911 KB
    per checkpoint and ~9.4 MB across four turns before pruning. Pruning bounds
    the total (``get_mongo_checkpointer`` retains the last 5 checkpoints by
    default), but one checkpoint still has to fit the backend's document limit:
    MongoDB and DocumentDB both reject a document larger than 16 MB.
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
    llm_config: dict


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
