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
