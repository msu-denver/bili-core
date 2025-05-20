"""
Module: langgraph_utils

This module provides utility functions and classes for managing and processing
conversation states within the LangGraph framework. It includes functions for
formatting messages with citations, trimming and summarizing conversation
history, and adding personas and summaries to conversation states.

Classes:
--------
- State:
    Represents a user's state in the system, extending the AgentState class.

Functions:
----------
- format_message_with_citations(message):
    Formats a message with included citations if available.

- build_trim_and_summarize_node(llm_model, summarize_llm_model=None, memory_limit_type="message_count",
memory_strategy="trim", k=5, **kwargs):
    Builds a node function for trimming and summarizing conversation state.

- build_add_persona_and_summary_node(persona, **kwargs):
    Builds a node function to add a persona and conversation summary to the current state.

- clear_state(state: State) -> dict:
    Clears the messages present in the given state and prepares a response containing the list of
    removed messages and an empty summary.

- build_normalize_tool_state_node(**kwargs):
    Builds a node function to normalize tool function calls in the conversation state.

- build_react_agent_node(tools=None, state=State, llm_model=None, **kwargs):
    Constructs a REACT agent node using the specified tools and state or a fallback node for
    models that do not support tools.

Dependencies:
-------------
- langchain_core.messages: Imports AIMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages.
- langgraph.prebuilt.chat_agent_executor: Imports AgentState, create_react_agent, StateSchema.
- bili.utils.logging_utils: Imports get_logger.

Usage:
------
This module is intended to be used within the LangGraph framework to manage
conversation states, including trimming, summarizing, and formatting
messages with citations. It also allows for the addition of custom personas
and conversation summaries to the conversation state.

Example:
--------
from bili.utils.langgraph_utils import format_message_with_citations, build_trim_and_summarize_node,
build_add_persona_and_summary_node

# Format a message with citations
formatted_message = format_message_with_citations(message)

# Build a trim and summarize node
trim_and_summarize = build_trim_and_summarize_node("message_count", "summarize", 100, llm_model)

# Build an add persona and summary node
add_persona_and_summary = build_add_persona_and_summary_node("Custom Persona Prefix")
"""

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import Tool
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    StateSchema,
    create_react_agent,
)

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


def build_trim_and_summarize_node(
    llm_model,
    summarize_llm_model=None,
    memory_limit_type="message_count",
    memory_strategy="trim",
    k=5,
    trim_k=None,
    **kwargs,
):
    """
    Builds a function that trims and/or summarizes a conversation's state based on specified
    memory constraints and strategies. This function is intended to optimize conversational
    context data by reducing the size of the conversation history while preserving
    essential details.

    :param llm_model: The large language model (LLM) used for processing conversation history
        and adjustments, such as message-summarization.
    :param summarize_llm_model: (Optional) A specialized large language model used explicitly
        for summarization. If not provided, defaults to the primary `llm_model`.
    :param memory_limit_type: The type of memory constraint applied to the conversation.
        Defaults to "message_count". Accepted values include "message_count"
        (message-based trimming) and token-based strategies.
    :param memory_strategy: Defines the approach utilized to manage memory constraints.
        Defaults to "trim" for trimming conversation history. Alternatively, "summarize"
        enables the use of summarization for efficient memory management.
    :param k: The initial memory limit (e.g., number of messages or token count), serving
        as the primary metric for conversations requiring trimming.
    :param trim_k: (Optional) A secondary threshold indicating the memory limit for further
        trimming once the initial limit is exceeded. If not provided, defaults to `k`.
    :param kwargs: Additional options or configuration parameters to customize the behavior
        of memory trimming and summarization processing.
    :return: A function that accepts a `State` dictionary and returns a processed dictionary
        with optimized conversation data based on trimming and/or summarization.
    """

    if summarize_llm_model is None:
        summarize_llm_model = llm_model

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
            arguments["token_counter"] = summarize_llm_model

        # pylint: disable=missing-kwoa
        remaining_messages = trim_messages(all_messages, **arguments)
        if trim_k is not None:
            if len(remaining_messages) != len(all_messages):
                # Threshold has been met, trim to custom trim_k
                arguments["max_tokens"] = trim_k
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
                        new_summary = summarize_llm_model.invoke(
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


def build_add_persona_and_summary_node(persona, **kwargs):
    """
    Builds a node function to add a persona and conversation summary to the current state.

    This function creates a node that inserts a `SystemMessage` at the beginning of the
    conversation state to set up a custom persona for the agent. If a summary of the
    conversation exists, it appends the summary to the `SystemMessage` content.

    Args:
        persona (str): The prefix used as a system message to set up
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
        # If not, insert a new SystemMessage with the persona at the beginning.
        # This is how a custom persona is set for the agent.
        if not all_messages:
            all_messages = []

        message_content = persona
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


def build_react_agent_node(
    tools: list[Tool] = None, state: StateSchema = State, llm_model=None, **kwargs
):
    """
    Constructs a REACT agent node using the specified tools and state or a fallback
    node for models that do not support tools. The REACT agent integrates tools
    and language models to process the input state and perform tasks.

    If tools are provided, creates a REACT agent using the specified tools
    with the associated state schema. If tools are not supported, creates a
    fallback node that directly interacts with the language model (LLM) while
    adapting the state's message format to prevent compatibility issues.

    :param tools: List of tools to be integrated with the REACT agent. If None,
        a fallback mechanism is used. Tools enable additional functionality for
        task handling.
    :type tools: list[Tool]

    :param state: Represents the schema of the state that interacts with the
        agent or LLM. This includes maintaining messages and other context
        required for conversation and workflows.
    :type state: StateSchema

    :param llm_model: The language model to be used by the agent or fallback
        node. This model processes the input messages and produces responses.
        The exact behavior may vary depending on the model's capabilities.

    :param kwargs: Additional arguments and settings for constructing the REACT
        agent or fallback node. This may include configuration options that are
        passed to internal functions.

    :return: A REACT agent node if tools are provided, otherwise a fallback node
        that directly interacts with the language model. Both options are
        structured to handle the input state appropriately and return the
        processed state.
    :rtype: Callable or any
    """
    LOGGER.debug(
        "Using model: %s | Tools enabled: %s",
        getattr(llm_model, "__class__", None),
        tools is not None,
    )
    if tools is not None:
        # If tools are supported, create a REACT agent with the provided tools
        LOGGER.debug("Creating REACT agent with tools: %s", tools)
        agent = create_react_agent(
            model=llm_model,
            state_schema=state,
            tools=tools,
        )
    else:
        LOGGER.debug(
            "TOOLS not supported for this LLM. Creating a simple graph node to invoke the LLM"
            "directly, and to convert the state to a format that the LLM can understand."
        )

        # If tools are not supported, create a simple graph node that calls the LLM directly
        def call_model(state: State):
            # Filter out any tool messages to prevent errors in the LLM during the call
            # This is to allow for the scenario where the state was established with a model
            # that used tools, and then later a model that does not support tools was selected.
            # This allows the conversation to continue without the history of tool messages being
            # passed to the model.
            messages = [
                message
                for message in state["messages"]
                if isinstance(message, (AIMessage, HumanMessage, SystemMessage))
            ]
            # Repackage every message as a HumanMessage with only the text content
            # There are some bugs out there right now with limitations of the kinds
            # of messages that can be sent to certain models, and this is a way to work around
            # that for now.
            messages_to_send = [
                HumanMessage(content=str(message.content)) for message in messages
            ]
            response = llm_model.invoke(messages_to_send)
            return {"messages": [response]}

        agent = call_model
    return agent


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
