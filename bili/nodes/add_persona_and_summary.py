"""
add_persona_and_summary.py
--------------------------

This module provides a utility for injecting a custom persona and conversation summary
into the state of a conversational agent. It is designed for use in workflows where
the agent's behavior or context should be influenced by a specific persona and/or
a running summary of the conversation. The persona can be templated with dynamic
values using Python's string formatting.

Functions:
----------
- build_add_persona_and_summary_node(persona, template_dict=None, **kwargs):
    Returns a node function that inserts a `SystemMessage` with the given persona
    (and optionally a conversation summary) at the start of the message list in the state.
    If a `SystemMessage` already exists at the beginning, it is replaced. The persona
    can be templated with dynamic values using the optional template_dict parameter.

Dependencies:
-------------
- langchain_core.messages: Provides `SystemMessage` and `RemoveMessage` classes for
chat history manipulation.
- bili.utils.langgraph_utils.State: Defines the state schema for conversation data.
- bili.utils.logging_utils.get_logger: Initializes a logger for tracing and debugging.

Usage:
------
Import and use `build_add_persona_and_summary_node` to create a state-processing function
for conversational agents that require persona injection and summary context.

Example:
--------
from bili.nodes.add_persona_and_summary import build_add_persona_and_summary_node

# Basic usage with static persona
add_persona_node = build_add_persona_and_summary_node(persona="You are a helpful assistant.")
new_state = add_persona_node(state)

# Usage with templated persona
persona_template = "You are a {role} assistant specialized in {domain}."
template_data = {"role": "senior", "domain": "data science"}
add_persona_node = build_add_persona_and_summary_node(
    persona=persona_template, 
    template_dict=template_data
)
new_state = add_persona_node(state)
"""

from langchain_core.messages import RemoveMessage, SystemMessage

from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)


def build_add_persona_and_summary_node(persona: str, **kwargs):
    """
    Builds a node function to add a persona and conversation summary to the current state.

    This function creates a node that inserts a `SystemMessage` at the beginning of the
    conversation state to set up a custom persona for the agent. If a summary of the
    conversation exists, it appends the summary to the `SystemMessage` content.
    
    The persona string can be templated with dynamic values using curly brace placeholders
    (e.g., "You are a {role} assistant") and the template_dict parameter. Extra values
    in template_dict are ignored, and only placeholders present in the persona string
    will be replaced.

    Args:
        persona (str): The prefix used as a system message to set up the initial context
            or persona. Can contain template placeholders in curly braces (e.g., {name}).
        template_dict (dict, optional): A dictionary of key-value pairs to substitute
            into the persona string template. Extra keys are ignored. Defaults to None.
        **kwargs: Additional keyword arguments (currently unused but maintained for
            backward compatibility).

    Returns:
        function: A function that takes a `State` object and returns a dictionary with
        updated messages reflecting the addition of the persona and conversation summary.
        
    Example:
        # Static persona
        node = build_add_persona_and_summary_node("You are a helpful assistant.")
        
        # Templated persona
        persona_template = "You are a {role} in {department}."
        template_data = {"role": "senior analyst", "department": "marketing"}
        node = build_add_persona_and_summary_node(persona_template, template_data)
    """

    def add_persona_and_summary(state: State) -> dict:
        """
        Adds a persona and, optionally, a conversation summary to the current state of
        messages. This function is designed to enable customization of the agent's persona
        by adding a `SystemMessage` at the beginning of the list of messages. If a summary
        exists in the state, it appends the summary to the `SystemMessage` content.

        The persona string is processed for template substitution if template_dict was
        provided to the parent function. Template placeholders use curly brace syntax
        (e.g., {variable_name}).

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
        # Define custom dict class to override missing
        # This will allow {} to exist in system prompts and only update keys
        # found in the template_dict
        class FormatDict(dict):
            """
            Support class for allowing a template string as a system prompt.
            """
            def __missing__(self, key):
                LOGGER.warning("Key not found. Substituting with {%s}", key)
                return "{" + key + "}" 

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

        # Load the persona string
        message_content = persona

        template_dict = state.get("template_dict", None)

        # Check if template data should be seeded and seed it
        if template_dict is not None:
            # Use ** to unpack the dictionary for .format()
            message_content = message_content.format(**FormatDict(template_dict))

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
