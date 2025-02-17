"""
Module: langgraph_utils

This module provides utility functions and classes for managing and processing
conversation states within the LangGraph framework. It includes functions for
formatting messages with citations, trimming and summarizing conversation
history, and adding personas and summaries to conversation states.

Classes:
    - State: Represents a user's state in the system, extending the AgentState
      class.

Functions:
    - format_message_with_citations(message):
      Formats a message with included citations if available.
    - build_trim_and_summarize_node(memory_limit_type, memory_strategy, k,
      llm_model):
      Builds a node function for trimming and summarizing conversation state.
    - build_add_persona_and_summary_node(langgraph_system_prefix):
      Builds a node function to add a persona and conversation summary to the
      current state.
    - clear_state(state: State) -> dict:
      Clears the messages present in the given state and prepares a response
      containing the list of removed messages and an empty summary.
    - build_normalize_tool_state_node():
      Builds a node function to normalize tool function calls in the
      conversation state.

Dependencies:
    - langchain_core.messages: Imports AIMessage, HumanMessage, RemoveMessage,
      SystemMessage, trim_messages.
    - langgraph.prebuilt.chat_agent_executor: Imports AgentState.
    - bili.utils.logging_utils: Imports get_logger.

Usage:
    This module is intended to be used within the LangGraph framework to manage
    conversation states, including trimming, summarizing, and formatting
    messages with citations. It also allows for the addition of custom personas
    and conversation summaries to the conversation state.

Example:
    from bili.utils.langgraph_utils import format_message_with_citations,
    build_trim_and_summarize_node, build_add_persona_and_summary_node

    # Format a message with citations
    formatted_message = format_message_with_citations(message)

    # Build a trim and summarize node
    trim_and_summarize = build_trim_and_summarize_node("message_count",
        "summarize", 100, llm_model)

    # Build an add persona and summary node
    add_persona_and_summary = build_add_persona_and_summary_node(
        "Custom Persona Prefix")
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.prebuilt.chat_agent_executor import AgentState

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


class State(AgentState):
    """
    Encapsulates user-specific preferences or state.

    This class is designed for maintaining any user-specific preferences or
    state information. It serves as an extension of the AgentState base class
    to incorporate additional properties tied to a specific user or agent.

    :ivar summary: Provides a summary or brief description of the state.
    :type summary: str
    :ivar owner: Identifies the owner of the state, typically a user or agent.
    :type owner: str
    """

    # If we wanted to keep any user-specific preferences or state, we could add them here
    summary: str
    owner: str


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


def build_trim_and_summarize_node(memory_limit_type, memory_strategy, k, llm_model):
    """
    Builds a node function for trimming and summarizing conversation state.

    This function creates a node that trims messages based on a specified strategy
    and summarizes the conversation to retain important context while minimizing
    the size of the conversation data.

    Args:
        memory_limit_type (str): The type of limit to enforce on memory
            ("message_count" or "token_count").
        memory_strategy (str): The strategy for handling memory
            ("trim" or "summarize").
        k (int): The threshold or limit used for trimming/summarizing.
        llm_model: The language model instance used for summarization.

    Returns:
        function: A function that takes a `State` object and returns a dictionary
        with updated conversation metadata, including trimmed messages and a summary.
    """

    def trim_and_summarize_state(state: State) -> dict:
        """
        This function manages optimization of the conversation history contained within
        the provided `state` object. It trims messages based on a predefined strategy
        and applies summarization to ensure the most important conversational context
        is retained while minimizing the size of conversation data. Additionally,
        important messages such as the last exchanged human/AI messages and system-level
        instructions are preserved.

        :param state: The current state of the conversation containing a list of
            messages and optionally a previously generated summary.
            - `messages`: A sequence of HumanMessage, AIMessage, or SystemMessage objects
              representing the conversation history.
            - `summary`: (optional) A string containing an existing summary for the
              conversation, which may be updated.
        :type state: dict

        :return:
            A dictionary comprising updated conversation metadata:
            - `messages`: A sequence with message objects marked for removal.
            - `summary`: A new or unchanged summary reflecting the trimmed and/or
              summarized content following trimming operations.
        :rtype: dict
        """

        # Retrieve the current list of messages from state for processing
        all_messages = state["messages"]
        LOGGER.trace(f"Original messages: {all_messages}")

        # Identify the last human and AI messages
        last_human_message = next(
            (msg for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
            None,
        )
        last_human_message_id = last_human_message.id if last_human_message else None
        last_ai_message = next(
            (msg for msg in reversed(all_messages) if isinstance(msg, AIMessage)), None
        )
        last_ai_message_id = last_ai_message.id if last_ai_message else None

        # 1) Start by trimming messages, if needed, using either message or token count to trim by
        arguments = {
            "max_tokens": k,
            "strategy": "last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            "start_on": "human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            "include_system": True,
            "allow_partial": False,
        }
        if memory_limit_type == "message_count":
            # Trim based on message count, otherwise use default behavior of token count
            # len will count the number of messages rather than tokens
            LOGGER.trace("Trimming messages based on message count.")
            arguments["token_counter"] = len
        else:
            LOGGER.trace("Trimming messages based on token count.")
            arguments["token_counter"] = llm_model

        # pylint: disable=missing-kwoa
        remaining_messages = trim_messages(all_messages, **arguments)
        # pylint: enable=missing-kwoa
        LOGGER.trace(f"Remaining messages after trimming: {remaining_messages}")

        if len(remaining_messages) < 2:
            LOGGER.trace(
                f"Much of the conversation has been trimmed. "
                f"Only {len(remaining_messages)} messages remain. "
                "Conversation context may be limited or lost."
            )

        # Figure out which messages are removed by finding message and message ids in all_messages
        # that are not in trimmed_messages. Exclude SystemMessages from this list because
        # they should never be removed. Also, exclude the last human and AI messages so
        # that even if we exceed the message or token limit, we don't remove the last interaction
        # with the user. This allows the conversation to continue from the last asked question.
        removed_messages = [
            m
            for m in all_messages
            if m not in remaining_messages
            and not isinstance(m, SystemMessage)
            and not m.id == last_human_message_id
            and not m.id == last_ai_message_id
        ]
        LOGGER.trace(f"Removed messages after trimming: {removed_messages}")

        # Create list of RemoveMessage based on trimmed ids to remove the trimmed
        # messages from the state
        # https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#manually-deleting-messages
        removed_ids = [m.id for m in removed_messages]
        messages_to_remove = [
            RemoveMessage(id=id_to_remove) for id_to_remove in removed_ids
        ]

        # 2) If memory strategy is "summarize", summarize the messages that were trimmed in the
        # previous step and add the summarized message to the beginning of the conversation
        # after the system message.
        new_summary = existing_summary_content = (
            state["summary"] if "summary" in state else ""
        )
        if memory_strategy == "summarize":
            if len(removed_messages) > 0:
                # Merge old summary + newly removed text
                conversation_text = "\n".join(
                    format_message_with_citations(m)
                    for m in removed_messages
                    if isinstance(m, (HumanMessage, AIMessage))
                ).strip()

                if conversation_text:
                    try:
                        new_summary = llm_model.invoke(
                            f"Summarize this older summary of the user's conversation (if any):\n\n"
                            f"{existing_summary_content}\n\n"
                            f"With the following text that was removed during the conversation:\n\n"
                            f"{conversation_text}\n\n"
                            f"Make sure to include key points and personalized "
                            f"details about the user,"
                            f" including any important context or information that was shared."
                            f"Keep the summary concise and factual."
                        ).content.strip()
                        LOGGER.trace(
                            f"New previous conversation summary: {new_summary}"
                        )
                    except Exception as e:
                        LOGGER.error(f"Summarization failed: %s", e)

        return {"messages": messages_to_remove, "summary": new_summary}

    return trim_and_summarize_state


def build_add_persona_and_summary_node(langgraph_system_prefix):
    """
    Builds a node function to add a persona and conversation summary to the current state.

    This function creates a node that inserts a `SystemMessage` at the beginning of the
    conversation state to set up a custom persona for the agent. If a summary of the
    conversation exists, it appends the summary to the `SystemMessage` content.

    Args:
        langgraph_system_prefix (str): The prefix used as a system message to set up
        the initial context or persona.

    Returns:
        function: A function that takes a `State` object and returns a dictionary with
        updated messages reflecting the addition of the persona and conversation summary.
    """

    def add_persona_and_summary(state: State) -> dict:
        """
        Adds a persona and, optionally, a conversation summary to the current state of
        messages. This function is designed to enable customization of the agent's persona
        by adding a `SystemMessage` at the beginning of the list of messages. If a summary
        exists in the state, it appends the summary to the `SystemMessage` content.

        If the first message in the current state is already a `SystemMessage`, it is replaced
        or removed before inserting the new one. The modified list of messages is then returned
        in the updated state.

        :param state: The current state of the conversation represented as a dictionary.
            It must include a "messages" key containing a list of message objects, and
            optionally, a "summary" key with the current summary of the conversation.
        :return: An updated state dictionary with modified messages reflecting the addition
            of the persona and conversation summary.
        :rtype: dict
        """
        # Retrieve the current list of messages from state for processing
        all_messages = state["messages"]
        LOGGER.trace(
            f"Original messages before adding persona and summary: {all_messages}"
        )

        # Check if there's already a SystemMessage with the same content at position 0
        # If not, insert a new SystemMessage with the langgraph_system_prefix at the beginning.
        # This is how a custom persona is set for the agent.
        if not all_messages:
            all_messages = []

        message_content = langgraph_system_prefix
        if "summary" in state and state["summary"]:
            message_content += (
                f"\n\nSummary of the conversation so far:\n{state['summary']}"
            )
        system_message = SystemMessage(content=message_content)

        # If message 0 is not a SystemMessage, insert a new one, otherwise replace it
        if len(all_messages) > 0 and isinstance(all_messages[0], SystemMessage):
            all_messages.append(RemoveMessage(id=all_messages[0].id))

        all_messages.insert(0, system_message)

        LOGGER.trace(f"Messages after adding persona and summary: {all_messages}")

        return {"messages": all_messages}

    return add_persona_and_summary


def build_normalize_tool_state_node():
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
