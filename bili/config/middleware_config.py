"""
Module: middleware_config

This module defines the available middleware configurations for use within the
application. Middleware allows intercepting and modifying agent execution at
various points in the agent loop, enabling features like logging, monitoring,
rate limiting, conversation summarization, and more.

Dependencies:
    - None (configuration only)

Usage:
    This module is intended to be used within the application to configure
    middleware based on the provided settings. The MIDDLEWARE dictionary
    defines all available middleware with their descriptions, enabled status,
    and configurable parameters.

Example:
    from bili.config.middleware_config import MIDDLEWARE

    # Get configuration for summarization middleware
    summarization_config = MIDDLEWARE["summarization"]
    print(summarization_config["description"])
"""

# Available Middleware
MIDDLEWARE = {
    "summarization": {
        "description": "Automatically summarizes conversation history when it exceeds "
        "a specified token limit, helping to manage context window constraints.",
        "enabled": False,
        "params": {
            "max_tokens": {
                "description": "Maximum number of tokens before triggering summarization. "
                "When the conversation exceeds this limit, older messages "
                "will be summarized to reduce token count.",
                "default": 4000,
                "type": "int",
            },
        },
    },
    "model_call_limit": {
        "description": "Limits the maximum number of model calls per agent execution to "
        "prevent runaway agents and control costs.",
        "enabled": False,
        "params": {
            "max_calls": {
                "description": "Maximum number of model calls allowed in a single agent execution. "
                "Execution will stop if this limit is reached.",
                "default": 10,
                "type": "int",
            },
        },
    },
}


def get_middleware_info(middleware_name: str) -> dict:
    """
    Retrieves configuration information for a specific middleware.

    :param middleware_name: Name of the middleware to retrieve information for
    :type middleware_name: str

    :return: Dictionary containing middleware configuration, or None if not found
    :rtype: dict or None

    Example:
        >>> info = get_middleware_info("summarization")
        >>> print(info["description"])
        Automatically summarizes conversation history when it exceeds a specified token limit...
    """
    return MIDDLEWARE.get(middleware_name)


def get_enabled_middleware() -> list:
    """
    Returns a list of middleware names that are enabled by default.

    :return: List of enabled middleware names
    :rtype: list

    Example:
        >>> enabled = get_enabled_middleware()
        >>> print(enabled)
        []
    """
    return [name for name, config in MIDDLEWARE.items() if config.get("enabled", False)]


def get_middleware_defaults(middleware_name: str) -> dict:
    """
    Retrieves the default parameter values for a specific middleware.

    :param middleware_name: Name of the middleware
    :type middleware_name: str

    :return: Dictionary of parameter names to default values
    :rtype: dict

    Example:
        >>> defaults = get_middleware_defaults("summarization")
        >>> print(defaults)
        {'max_tokens': 4000}
    """
    middleware_config = MIDDLEWARE.get(middleware_name, {})
    params = middleware_config.get("params", {})

    return {
        param_name: param_config.get("default")
        for param_name, param_config in params.items()
    }
