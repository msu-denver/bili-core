"""
prepare_llm_config_node.py
---------------------------

This module provides a node for preparing LLM runtime configuration and injecting
it into the conversation state. This enables provider-specific configuration
(like Google Vertex AI's ThinkingConfig) to be properly formatted before being
passed to the react_agent node.

Functions:
----------
- build_prepare_llm_config_node(**kwargs):
    Returns a node function that transforms thinking_config and model_type from
    node_kwargs into a provider-specific llm_config dict and sets it in state.

Usage:
------
Import and use `build_prepare_llm_config_node` to create a state-processing function
for LLM configuration preparation.

Example:
--------
from bili.nodes.prepare_llm_config_node import build_prepare_llm_config_node

# In node_kwargs, pass thinking_config and model_type
node_kwargs = {
    "thinking_config": {"budget": 0},
    "model_type": "remote_google_vertex",
    ...
}

prepare_llm_config_node = build_prepare_llm_config_node(**node_kwargs)
new_state = prepare_llm_config_node(state)
# state now contains llm_config formatted for the specific provider
"""

from functools import partial
from typing import Any, Dict, Optional

from bili.graph_builder.classes.node import Node
from bili.loaders.llm_loader import prepare_runtime_config


def build_prepare_llm_config_node(**kwargs) -> callable:
    """
    Builds a function that prepares LLM runtime configuration and injects it into state.

    This node reads thinking_config and model_type from kwargs, transforms them into
    a provider-specific runtime configuration using prepare_runtime_config(), and
    sets the result in state['llm_config'].

    The llm_config will be read by the react_agent node and passed to model.invoke().

    :param kwargs: Node configuration including:
        - thinking_config (dict, optional): Simple thinking config dict (e.g., {"budget": 0})
        - model_type (str, optional): LLM provider type (e.g., "remote_google_vertex")
    :return: A function that takes a state dictionary, prepares llm_config, and returns
             the modified state dictionary.
    :rtype: Callable[[dict], dict]

    Example:
        >>> node_kwargs = {
        ...     "thinking_config": {"budget": 5000},
        ...     "model_type": "remote_google_vertex"
        ... }
        >>> prepare_node = build_prepare_llm_config_node(**node_kwargs)
        >>> state = {"messages": [...]}
        >>> new_state = prepare_node(state)
        >>> # new_state["llm_config"] now contains provider-specific config
    """
    thinking_config = kwargs.get("thinking_config", None)
    model_type = kwargs.get("model_type", None)

    def prepare_llm_config_node_func(state: dict) -> dict:
        """
        Prepares LLM runtime configuration and injects it into state.

        Transforms thinking_config and model_type into provider-specific llm_config
        format and sets it in state['llm_config']. This config will be used by
        react_agent when invoking the LLM.

        :param state: Current conversation state dictionary
        :return: State dictionary with llm_config field populated
        :rtype: dict
        """
        # Prepare the runtime config using bili-core helper
        llm_config = prepare_runtime_config(
            model_type=model_type,
            thinking_config=thinking_config
        )

        # Return state with llm_config set
        return {
            "llm_config": llm_config
        }

    return prepare_llm_config_node_func


# Export as a Node instance for use in graph definitions
prepare_llm_config_node = partial(Node, "prepare_llm_config", build_prepare_llm_config_node)
