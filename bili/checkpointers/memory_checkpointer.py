"""
Module: memory_checkpointer

This module provides a queryable wrapper around LangGraph's MemorySaver that implements
the QueryableCheckpointerMixin interface. This allows the in-memory checkpointer to work
with Flask conversation management endpoints.

Classes:
    - QueryableMemorySaver:
      Wraps MemorySaver with query capabilities for conversation data retrieval.

Dependencies:
    - langgraph.checkpoint.memory: Imports MemorySaver for in-memory checkpointing
    - bili.checkpointers.base_checkpointer: Imports QueryableCheckpointerMixin

Usage:
    from bili.checkpointers.memory_checkpointer import QueryableMemorySaver

    checkpointer = QueryableMemorySaver()
    threads = checkpointer.get_user_threads("user@example.com")
"""

from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from bili.checkpointers.base_checkpointer import QueryableCheckpointerMixin


class QueryableMemorySaver(QueryableCheckpointerMixin, MemorySaver):
    """
    Queryable wrapper around MemorySaver that implements QueryableCheckpointerMixin.

    Provides query methods for conversation data retrieval from in-memory storage.
    Useful for testing and development without requiring a database.
    """

    def __init__(self, *, user_id: Optional[str] = None):
        """
        Initialize memory checkpointer.

        Args:
            user_id: Optional user identifier for thread ownership validation.
                    If provided, validates that thread_ids belong to this user.
                    Used for interface consistency with database checkpointers.
        """
        super().__init__()
        self.user_id = user_id

    def put(self, config, checkpoint, metadata, new_versions):
        """
        Save checkpoint with optional ownership validation.

        Args:
            config: Runnable configuration with thread_id
            checkpoint: Checkpoint to save
            metadata: Checkpoint metadata
            new_versions: Channel versions

        Returns:
            Updated configuration

        Raises:
            PermissionError: If thread_id doesn't belong to configured user_id
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate ownership if user_id is configured
        self._validate_thread_ownership(thread_id)

        # Use base class implementation
        return super().put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config):
        """
        Get checkpoint tuple with optional ownership validation.

        Args:
            config: Runnable configuration with thread_id

        Returns:
            Checkpoint tuple or None

        Raises:
            PermissionError: If thread_id doesn't belong to configured user_id
        """
        thread_id = config["configurable"]["thread_id"]

        # Validate ownership if user_id is configured
        self._validate_thread_ownership(thread_id)

        # Use base class implementation
        return super().get_tuple(config)

    def get_user_threads(
        self, user_identifier: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversation threads for a user from memory storage."""
        # MemorySaver stores data in self.storage dict with thread_id as string key
        # We need to filter by thread_id that matches user_identifier pattern

        threads_data = {}

        # Iterate through storage to find matching threads
        for thread_id, checkpoint_queue in self.storage.items():
            # Check if thread belongs to user
            if thread_id == user_identifier or thread_id.startswith(
                f"{user_identifier}_"
            ):
                if thread_id not in threads_data:
                    threads_data[thread_id] = {
                        "checkpoints": [],
                        "last_checkpoint": None,
                    }

                # checkpoint_queue is a deque of (checkpoint, metadata) tuples
                threads_data[thread_id]["checkpoints"] = list(checkpoint_queue)

                # Track the latest checkpoint (last in queue)
                if checkpoint_queue and len(checkpoint_queue) > 0:
                    last_item = checkpoint_queue[-1]
                    # Handle both tuple (checkpoint, metadata) and single checkpoint
                    if isinstance(last_item, tuple) and len(last_item) == 2:
                        checkpoint, metadata = last_item
                    else:
                        checkpoint = last_item
                    if checkpoint:
                        threads_data[thread_id]["last_checkpoint"] = checkpoint

        # Build thread list
        threads = []
        for thread_id, data in sorted(
            threads_data.items(),
            key=lambda x: (
                x[1]["last_checkpoint"]["id"] if x[1]["last_checkpoint"] else ""
            ),
            reverse=True,
        ):
            # Extract conversation_id from thread_id
            if "_" in thread_id:
                conversation_id = thread_id.split("_", 1)[1]
            else:
                conversation_id = "default"

            # Extract first/last message and message count from latest checkpoint
            first_message = None
            last_message = None
            message_count = 0
            title = None
            tags = []
            last_checkpoint = data["last_checkpoint"]

            if last_checkpoint:
                channel_values = last_checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])

                # Extract title and tags from state
                title = channel_values.get("title")
                tags = channel_values.get("tags", [])

                if messages:
                    message_count = len(messages)
                    # Get the first and last user messages
                    for msg in messages:
                        if (
                            hasattr(msg, "content")
                            and msg.content
                            and msg.__class__.__name__ == "HumanMessage"
                        ):
                            # Extract text content (handle multimodal)
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [
                                    p.get("text", "")
                                    for p in content
                                    if isinstance(p, dict) and p.get("type") == "text"
                                ]
                                content = " ".join(text_parts)

                            if first_message is None:
                                first_message = content
                            last_message = content

            threads.append(
                {
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "last_updated": last_checkpoint["id"] if last_checkpoint else None,
                    "checkpoint_count": len(data["checkpoints"]),
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "title": title,
                    "tags": tags,
                }
            )

        # Apply pagination
        if offset > 0 or limit is not None:
            start = offset
            end = offset + limit if limit is not None else None
            threads = threads[start:end]

        return threads

    def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        message_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all messages from a conversation thread."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # Get the checkpoint queue for this thread
        checkpoint_queue = self.storage.get(thread_id)

        if not checkpoint_queue:
            return []

        # Get the latest checkpoint (last in queue)
        checkpoint, metadata = checkpoint_queue[-1]

        if not checkpoint:
            return []

        latest_checkpoint = checkpoint

        # Extract messages from checkpoint
        channel_values = latest_checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            msg_type = msg.__class__.__name__

            # Apply message type filter if specified
            if message_types is not None and msg_type not in message_types:
                continue

            # Extract content (handle multimodal)
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)

            # Strip thinking blocks from AI messages
            if msg_type == "AIMessage":
                content = self._strip_thinking_blocks(content)

            messages.append(
                {
                    "role": "user" if msg_type == "HumanMessage" else "assistant",
                    "content": content,
                    "timestamp": None,  # Messages don't have individual timestamps in LangGraph
                }
            )

        # Apply pagination
        if offset > 0 or limit is not None:
            end_index = offset + limit if limit is not None else None
            messages = messages[offset:end_index]

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoints for a conversation thread."""
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        # MemorySaver already has delete_thread, but we need to return bool
        # Find all keys for this thread and delete them
        keys_to_delete = [key for key in self.storage.keys() if key[0] == thread_id]

        for key in keys_to_delete:
            del self.storage[key]

        return len(keys_to_delete) > 0

    def get_user_stats(self, user_identifier: str) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        threads = self.get_user_threads(user_identifier)

        if not threads:
            return {
                "total_threads": 0,
                "total_messages": 0,
                "total_checkpoints": 0,
                "oldest_thread": None,
                "newest_thread": None,
            }

        total_messages = sum(thread["message_count"] for thread in threads)
        total_checkpoints = sum(thread["checkpoint_count"] for thread in threads)
        oldest_thread = min(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )
        newest_thread = max(
            (t["last_updated"] for t in threads if t["last_updated"]), default=None
        )

        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "total_checkpoints": total_checkpoints,
            "oldest_thread": oldest_thread,
            "newest_thread": newest_thread,
        }

    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists in memory storage."""
        return any(key[0] == thread_id for key in self.storage.keys())
