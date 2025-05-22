"""
trim_and_summarize.py
---------------------

This module provides functionality to trim and/or summarize conversation histories
within a stateful chat or agent workflow. It is designed to optimize memory usage
by reducing the size of conversation history while preserving essential context,
using configurable strategies such as message count trimming or summarization.

Functions:
----------
- build_trim_and_summarize_node(
      llm_model,
      summarize_llm_model=None,
      memory_limit_type="message_count",
      memory_strategy="trim",
      k=5,
      trim_k=None,
      **kwargs
  ):
    Constructs a function that processes a conversation state by trimming messages
    and optionally summarizing removed content, according to specified memory constraints
    and strategies.

Dependencies:
-------------
- langchain_core.messages: Provides message classes and utilities for chat history management.
- bili.utils.langgraph_utils.State: Defines the state schema for conversation data.
- bili.utils.langgraph_utils.format_message_with_citations: Formats messages for summarization.
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `build_trim_and_summarize_node` to create a state-processing function
for use in conversational agents or workflows that require memory management.

Example:
--------
from bili.nodes.trim_and_summarize import build_trim_and_summarize_node

trim_node = build_trim_and_summarize_node(llm_model=my_llm, k=10, memory_strategy="summarize")
new_state = trim_node(state)
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)

from bili.utils.langgraph_utils import State, format_message_with_citations
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_trim_and_summarize_node(
    llm_model,
    summarize_llm_model=None,
    memory_limit_type="message_count",
    memory_strategy="trim",
    k=5,
    trim_k=None,
    prompt_template=None,
    **kwargs,
):
    """
    Builds a function that optimizes a conversation's state by trimming messages based on predefined
    strategies and applying summarization. This function ensures important context is retained while
    managing memory constraints through trimming or summarization.

    :param llm_model: The primary language model used for token counting during trimming if
        `memory_limit_type` is "token_count".
    :param summarize_llm_model: Optional alternative language model used to summarize removed
        messages during the summarization phase. Falls back to `llm_model` if not provided.
    :param memory_limit_type: String specifying the memory limitation type to apply. Options include
        "message_count" (default) for limiting by number of messages, or "token_count" for limiting
        by number of tokens using `llm_model`.
    :param memory_strategy: Strategy for managing memory constraints. Defaults to "trim". Supported
        strategies:
          - "trim": Removes excess messages based on the memory limit.
          - "summarize": Summarizes trimmed messages and appends summarized content to the state.
    :param k: Integer specifying the memory limit threshold in either tokens or messages. The exact
        meaning depends on the `memory_limit_type`. Default is 5.
    :param trim_k: Optional integer specifying a secondary trim threshold for messages. If specified,
        applies an additional trim to meet this threshold after the main trim is applied.
    :param prompt_template: Optional custom string template for the summarization prompt. If none is
        provided, a default prompt format is used during summarization.
    :param kwargs: Additional arguments used internally within the function or passed to dependent
        functions.

    :return: A `trim_and_summarize_state` callable function to process state objects. This function
        trims conversation messages or summarizes removed content depending on the provided
        parameters.
    :rtype: Callable
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
                    template_to_use = (
                        "Summarize this older summary of the user's conversation (if any):\n\n"
                        "{existing_summary_content}\n\n"
                        "With the following text that was removed during the conversation:\n\n"
                        "{conversation_text}\n\n"
                        "Make sure to include key points and personalized "
                        "details about the user, including any important context or information that was shared."
                        " Keep the summary concise and factual."
                    )
                    if prompt_template is not None:
                        template_to_use = prompt_template

                    prompt = template_to_use.format(
                        existing_summary_content=existing_summary_content,
                        conversation_text=conversation_text,
                    )
                    try:
                        new_summary = summarize_llm_model.invoke(prompt).content.strip()
                        LOGGER.trace(
                            f"New previous conversation summary: {new_summary}"
                        )
                    except Exception as e:
                        LOGGER.error(f"Summarization failed: %s", e)

        return {"messages": messages_to_remove, "summary": new_summary}

    return trim_and_summarize_state
