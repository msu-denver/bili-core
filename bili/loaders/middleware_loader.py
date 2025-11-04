"""
Module: middleware_loader

This module initializes and configures middleware for use with LangChain agents.
Middleware allows you to intercept and modify agent execution at various points
in the agent loop, enabling features like logging, monitoring, rate limiting,
and dynamic behavior modification.

Functions:
    - initialize_middleware(active_middleware, middleware_params):
      Initializes and returns a list of middleware instances based on the active
      middleware and their specific parameters.

Dependencies:
    - langchain.agents.middleware: Imports middleware classes and decorators
    - bili.utils.logging_utils: Imports `get_logger` for logging

Usage:
    This module is intended to be used within the application to initialize and
    configure middleware based on the provided configuration.

    The `MIDDLEWARE_REGISTRY` dictionary maps middleware names to their
    corresponding initialization functions. This allows you to override the
    default middleware initialization behavior or add custom middleware.

Example:
    from bili.loaders.middleware_loader import initialize_middleware

    # Initialize middleware
    middleware = initialize_middleware(
        active_middleware=["summarization", "model_call_limit"],
        middleware_params={
            "summarization": {"max_tokens": 4000},
            "model_call_limit": {"max_calls": 10}
        }
    )

    # Use with agent
    agent_node = build_react_agent_node(
        tools=my_tools,
        middleware=middleware
    )
"""

try:
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        SummarizationMiddleware,
    )

    LANGCHAIN_MIDDLEWARE_AVAILABLE = True
except ImportError:
    LANGCHAIN_MIDDLEWARE_AVAILABLE = False

from bili.utils.logging_utils import get_logger

# Get the logger instance for the module
LOGGER = get_logger(__name__)

# Define a registry of middleware initialization functions
# This allows for dynamic initialization of middleware based on the provided configuration
# and for users to override the default middleware initialization behavior or define custom middleware
MIDDLEWARE_REGISTRY = {}

if LANGCHAIN_MIDDLEWARE_AVAILABLE:
    MIDDLEWARE_REGISTRY.update(
        {
            "summarization": lambda params: (
                SummarizationMiddleware(**params)
                if params
                else SummarizationMiddleware()
            ),
            "model_call_limit": lambda params: (
                ModelCallLimitMiddleware(**params)
                if params
                else ModelCallLimitMiddleware()
            ),
        }
    )


def initialize_middleware(
    active_middleware: list = None, middleware_params: dict = None
):
    """
    Initializes middleware based on the provided active middleware list and parameters.

    This function creates middleware instances based on the names provided in
    active_middleware. Each middleware can be configured with specific parameters
    from the middleware_params dictionary.

    :param active_middleware: List of middleware names to initialize. If None or empty,
        returns an empty list.
    :type active_middleware: list

    :param middleware_params: Dictionary mapping middleware names to their configuration
        parameters. If None, empty dictionaries are used for all middleware.
    :type middleware_params: dict

    :return: List of initialized middleware instances ready to be passed to an agent.
    :rtype: list

    :raises ValueError: If a middleware name in active_middleware is not found in the registry.

    Example:
        >>> middleware = initialize_middleware(
        ...     active_middleware=["summarization"],
        ...     middleware_params={"summarization": {"max_tokens": 4000}}
        ... )
        >>> len(middleware)
        1
    """
    if not active_middleware:
        LOGGER.debug("No active middleware specified, returning empty list")
        return []

    if not LANGCHAIN_MIDDLEWARE_AVAILABLE:
        LOGGER.warning(
            "LangChain middleware not available. "
            "Middleware features require LangChain 1.0+. "
            "Returning empty list."
        )
        return []

    middleware_params = middleware_params or {}
    middleware_instances = []

    for mw_name in active_middleware:
        if mw_name not in MIDDLEWARE_REGISTRY:
            available = ", ".join(MIDDLEWARE_REGISTRY.keys())
            raise ValueError(
                f"Middleware '{mw_name}' not found in registry. "
                f"Available middleware: {available}"
            )

        params = middleware_params.get(mw_name, {})
        LOGGER.debug("Initializing middleware: %s with params: %s", mw_name, params)

        try:
            middleware_instance = MIDDLEWARE_REGISTRY[mw_name](params)
            middleware_instances.append(middleware_instance)
            LOGGER.info("Successfully initialized middleware: %s", mw_name)
        except Exception as e:
            LOGGER.error("Failed to initialize middleware %s: %s", mw_name, str(e))
            raise

    return middleware_instances
