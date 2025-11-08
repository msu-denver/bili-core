"""
update_timestamp.py
-------------------

This module provides a utility for updating timestamp information in a
conversation or application state dictionary. It is intended for use in
conversational agent workflows or any process that requires tracking
message timing.

Functions:
----------
- build_update_timestamp_node(**kwargs):
    Returns a node function that updates the state dictionary with the current
    message time (ISO 8601 format),
    the previous message time, and the elapsed time (`delta_time`) in seconds
    between messages.

Usage:
------
Import and use `build_update_timestamp_node` to create a state-processing function
for timestamp management.

Example:
--------
from bili.nodes.update_timestamp import build_update_timestamp_node

update_timestamp_node = build_update_timestamp_node()
new_state = update_timestamp_node(state)
"""

from functools import partial
import time
from datetime import datetime, timezone

from bili.graph_builder.classes.node import Node


def build_update_timestamp_node(**kwargs):
    """
    Builds a function that updates timestamps in a given state dictionary.

    The returned function takes a dictionary representing the state, and
    generates timestamp values for the current time and the time of the
    previous message, as well as the elapsed time between them. The 'state'
    dictionary is updated with the keys 'previous_message_time',
    'current_message_time', and 'delta_time'.

    The 'current_message_time' value is represented in ISO 8601 format,
    while 'delta_time' is the time difference in seconds as a floating-point
    number. If the 'state' dictionary does not contain a valid
    'current_message_time' value, the current time will be used as both the
    current and previous message time.

    :param kwargs: Additional keyword arguments that may be passed for
                   customization but are currently unused.
    :return: A function that takes a state dictionary as input, updates
             the timestamps, and returns the modified state dictionary.
    :rtype: Callable[[dict], dict]
    """

    def update_timestamps_node(state: dict) -> dict:
        """
        Builds a function that updates timestamp information in a given state.

        This function factory generates a function that, when executed, updates a provided
        application state dictionary with the current ISO-formatted timestamp, calculates the
        delta time between the current and previous timestamps, and maintains the history
        of the last processed timestamp.

        :param **kwargs: Additional keyword arguments, currently unused.
        :return: A function that updates the timestamp information in the given state.
        :rtype: Callable[[dict], dict]
        """
        current_epoch = time.time()
        current_iso = datetime.fromtimestamp(current_epoch, timezone.utc).isoformat()

        previous_iso = state.get("current_message_time", current_iso)
        try:
            previous_epoch = datetime.fromisoformat(previous_iso).timestamp()
        except Exception:
            previous_epoch = current_epoch

        delta_time = current_epoch - previous_epoch

        return {
            **state,
            "previous_message_time": previous_iso,
            "current_message_time": current_iso,
            "delta_message_time": delta_time,
        }

    return update_timestamps_node
update_timestamp_node = partial(Node, "update_timestamp", build_update_timestamp_node)