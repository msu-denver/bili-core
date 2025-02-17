"""
Module: logging_utils

This module provides utility functions for configuring and using the logging
system within the application. It includes functions to define a custom TRACE
logging level, retrieve loggers, and determine logging levels based on
environment variables.

Functions:
    - trace(self, message, *args, **kwargs):
      Logs a message with the custom TRACE level.
    - get_log_level(level_name):
      Determines and returns the appropriate logging level based on the provided
      level name.
    - get_logger(name: str):
      Retrieves a logger instance configured with the specified name.

Dependencies:
    - logging: Standard library module for logging.
    - os: Standard library module for interacting with the operating system.

Usage:
    This module is intended to be used for logging purposes within the
    application. It provides functions to define and use a custom TRACE logging
    level, retrieve loggers, and determine logging levels based on environment
    variables.

Example:
    from bili.utils.logging_utils import get_logger

    # Retrieve a logger instance
    logger = get_logger("my_logger")

    # Log a message at the TRACE level
    logger.trace("This is a trace message")
"""

import logging
import os

# Define a custom TRACE level
TRACE = 5

# Add the custom TRACE level to the logging module
logging.addLevelName(TRACE, "TRACE")


# Function to log at TRACE level
def trace(self, message, *args, **kwargs):
    """
    Logs a message with TRACE level if the TRACE level is enabled for this logger.
    This method acts as a convenience to log debug-level messages with a custom
    logging level defined as TRACE. It ensures that the logging action is performed
    only if the TRACE level is enabled.

    :param self: The instance of the logger handling the log dispatch.
    :param message: The log message to be processed and logged.
    :param args: Positional arguments to format the message, if needed.
    :param kwargs: Keyword arguments to format the message, if needed.
    :return: None
    """
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Add the trace method to the Logger class
logging.Logger.trace = trace


# Map custom TRACE level string to its numeric value
def get_log_level(level_name):
    """
    Determines and returns the appropriate logging level based on the provided level name.
    If a custom "TRACE" level is supplied, it will return the corresponding TRACE level.
    If the level name does not correspond to any predefined logging level, it defaults to
    `logging.INFO`.

    :param level_name: Name of the logging level to retrieve. Can be "TRACE" or any valid
    value from Python's logging
    module (e.g., "DEBUG", "INFO", "ERROR").
    :type level_name: str
    :return: The corresponding logging level as an integer value or `TRACE` if "TRACE" is provided.
    :rtype: int
    """
    if level_name == "TRACE":
        return TRACE
    return getattr(logging, level_name, logging.INFO)


# Fetch the logging level from environment variables, default to INFO if not set
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

# Get the numeric value of the logging level from the string
# using the custom function that supports TRACE
log_level = get_log_level(log_level_str)

# Perform the root logger configuration once when the module is loaded
if len(logging.getLogger().handlers) == 0:
    if len(logging.getLogger().handlers) > 0:
        # If running in an environment like AWS Lambda, the root logger is already configured
        logging.getLogger().setLevel(log_level)
    else:
        # Configure the logger with a basic console handler for local usage
        logging.basicConfig(level=log_level)


def get_logger(name: str):
    """
    Retrieve a logger instance configured with the specified name.

    This function retrieves an instance of a logger using the provided name.
    It leverages Python's built-in `logging` module to facilitate logging
    functionality for varied use cases. The logger configuration can be set
    up separately, influencing the behavior of the returned logger.

    :param name: A string representing the name of the logger to retrieve.
    :type name: str
    :return: An instance of `logging.Logger` configured with the provided name.
    :rtype: logging.Logger
    """
    return logging.getLogger(name)
