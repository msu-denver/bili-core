"""
Module: base_checkpointer

This module provides an abstract base class for checkpointers that extends
LangGraph's base checkpointer functionality with query capabilities for
conversation management.

Classes:
    - QueryableCheckpointerMixin:
      Mixin class that provides query methods for conversation data retrieval.
      Designed to be mixed into checkpointer implementations (MongoDB, PostgreSQL, Memory).

Dependencies:
    - abc: Provides abstract base class functionality
    - typing: Provides type hints
    - datetime: Provides datetime handling

Usage:
    Checkpointer implementations should inherit from this mixin along with
    their respective LangGraph checkpointer base class to gain query capabilities.

Example:
    class PruningMongoDBSaver(QueryableCheckpointerMixin, MongoDBSaver):
        def get_user_threads(self, user_identifier: str) -> List[Dict[str, Any]]:
            # Implementation specific to MongoDB
            pass
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Lower-case serialized message types mapped to message class names.
_MESSAGE_CLASS_BY_TYPE = {
    "human": "HumanMessage",
    "ai": "AIMessage",
    "system": "SystemMessage",
    "tool": "ToolMessage",
    "function": "FunctionMessage",
}

_ROLE_BY_MESSAGE_CLASS = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "FunctionMessage": "function",
}


def _message_class_name(msg: Any) -> Any:
    """Return the normalized class name of a message object or dictionary."""
    # Check dict first: every Python object (including dict) has __class__,
    # so a leading hasattr(msg, "__class__") guard would always win and the
    # dict-specific type extraction would be dead code.
    if isinstance(msg, dict):
        msg_class = msg.get("type", msg.get("__class__", "unknown"))
        # Serialized format where the type is the last element of the id.
        if msg_class == "unknown" and "kwargs" in msg:
            raw_id = msg.get("id")
            # Guard against an empty id list: [][-1] would raise.
            msg_class = raw_id[-1] if isinstance(raw_id, list) and raw_id else "unknown"
    else:
        msg_class = msg.__class__.__name__
    if isinstance(msg_class, str):
        return _MESSAGE_CLASS_BY_TYPE.get(msg_class.lower(), msg_class)
    return msg_class


def _message_text(msg: Any) -> Any:
    """Return a message's content, with multimodal parts reduced to text."""
    if hasattr(msg, "content"):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content", msg.get("kwargs", {}).get("content", str(msg)))
    else:
        content = str(msg)

    # Multimodal content is a list of {type, text} parts and plain strings.
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        content = " ".join(text_parts)
    return content


class QueryableCheckpointerMixin(ABC):
    """
    Abstract mixin class that defines query interface for checkpointers.

    This mixin provides methods for querying conversation data from checkpointers
    regardless of the underlying storage mechanism (MongoDB, PostgreSQL, Memory).
    All checkpointer implementations should implement these methods to provide
    a consistent interface for conversation management.
    """

    @abstractmethod
    def get_user_threads(
        self, user_identifier: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all conversation threads for a specific user.

        Args:
            user_identifier: User's email or ID used in thread_id format
            limit: Maximum number of threads to return (None for all)
            offset: Number of threads to skip for pagination

        Returns:
            List of thread objects with metadata:
            [
                {
                    "thread_id": str,
                    "conversation_id": str,  # Extracted from thread_id
                    "last_updated": datetime,
                    "checkpoint_count": int,
                    "message_count": int,
                    "first_message": Optional[str],  # For title generation
                    "last_message": Optional[str],  # Last user message preview
                    "title": Optional[str],
                    "tags": List[str],
                },
                ...
            ]
        """

    @abstractmethod
    def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        message_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all messages from a conversation thread.

        Args:
            thread_id: Thread ID to retrieve messages from
            limit: Maximum number of messages to return (None for all)
            offset: Number of messages to skip for pagination
            message_types: Optional list of message types to filter by
                (e.g., ["HumanMessage", "AIMessage"]). If None, returns all messages.

        Returns:
            List of message objects:
            [
                {
                    "role": str,  # "user" or "assistant"
                    "content": str,  # Thinking blocks are stripped from AI messages
                    "timestamp": Optional[datetime],
                },
                ...
            ]
        """

    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete all checkpoints for a conversation thread.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deletion was successful, False otherwise
        """

    @abstractmethod
    def get_user_stats(self, user_identifier: str) -> Dict[str, Any]:
        """
        Get usage statistics for a user.

        Args:
            user_identifier: User's email or ID used in thread_id format

        Returns:
            Dictionary with statistics:
            {
                "total_threads": int,
                "total_messages": int,
                "total_checkpoints": int,
                "oldest_thread": Optional[datetime],
                "newest_thread": Optional[datetime],
            }
        """

    @abstractmethod
    def thread_exists(self, thread_id: str) -> bool:
        """
        Check if a thread exists in the checkpointer.

        Args:
            thread_id: Thread ID to check

        Returns:
            True if thread exists, False otherwise
        """

    def verify_thread_ownership(self, thread_id: str, user_identifier: str) -> bool:
        """
        Verify that a thread belongs to a specific user.

        Default implementation checks thread_id format:
        - email (default thread)
        - email_conversationId (named conversation)

        Subclasses can override for custom ownership logic.

        Args:
            thread_id: Thread ID to verify
            user_identifier: User's email or ID

        Returns:
            True if thread belongs to user, False otherwise
        """
        return thread_id == user_identifier or thread_id.startswith(
            f"{user_identifier}_"
        )

    def _validate_thread_ownership(self, thread_id: str) -> None:
        """
        Validate that thread_id belongs to the authenticated user.

        Checks if the thread_id follows the expected pattern for the configured user_id.
        Thread IDs must either exactly match the user_id or start with "{user_id}_".

        This method provides a consistent implementation across all checkpointers
        that require thread ownership validation. It uses the verify_thread_ownership()
        method for the actual validation logic, which can be overridden by subclasses
        for custom validation rules.

        Args:
            thread_id: Thread ID to validate

        Raises:
            PermissionError: If thread_id doesn't belong to the configured user_id
        """
        # Check if user_id is configured
        if not hasattr(self, "user_id") or self.user_id is None:
            return  # Validation disabled (backward compatible)

        # Use verify_thread_ownership for actual validation
        if not self.verify_thread_ownership(thread_id, self.user_id):
            raise PermissionError(
                f"Access denied: thread_id '{thread_id}' does not belong to "
                f"user '{self.user_id}'"
            )

    def _format_thread_messages(
        self, raw_messages: List[Any], message_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Map stored messages to get_thread_messages dictionaries.

        Shared by the backends whose get_thread_messages output must agree:
        the message type is normalized to a class name, filtered by
        ``message_types``, and mapped to a role (``user``, ``assistant``,
        ``system``, ``tool``, ``function``, else ``unknown``). Multimodal
        content is reduced to its text, thinking blocks are stripped from AI
        messages, and an AI message left empty by that is dropped.

        Args:
            raw_messages: Messages from the checkpoint's messages channel,
                as message objects or serialized dictionaries.
            message_types: Optional message class names to keep.

        Returns:
            List of ``{role, content, timestamp}`` dictionaries, unpaginated.
        """
        messages = []
        for msg in raw_messages:
            msg_class = _message_class_name(msg)

            if message_types is not None and msg_class not in message_types:
                continue

            role = _ROLE_BY_MESSAGE_CLASS.get(msg_class, "unknown")
            content = _message_text(msg)

            if msg_class == "AIMessage":
                content = self._strip_thinking_blocks(content)
                # An AI message holding only a thinking block has nothing to
                # show once it is stripped.
                if not content:
                    continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                    # Messages carry no individual timestamps in LangGraph.
                    "timestamp": None,
                }
            )
        return messages

    def _strip_thinking_blocks(self, content: str) -> str:
        """
        Strip thinking/reasoning tags from message content.

        Removes XML-style blocks commonly used by LLMs for chain-of-thought:
        - <thinking>...</thinking>
        - <think>...</think>
        - <reasoning>...</reasoning>

        Args:
            content: The message content to process

        Returns:
            Content with thinking blocks removed and cleaned up
        """
        if not content:
            return content

        # Remove thinking blocks (case-insensitive, dotall for multiline)
        patterns = [
            r"<thinking>.*?</thinking>",
            r"<think>.*?</think>",
            r"<reasoning>.*?</reasoning>",
            r"<internal>.*?</internal>",
        ]
        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace left behind
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()
