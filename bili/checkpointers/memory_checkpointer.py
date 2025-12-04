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

    def get_user_threads(
        self, user_identifier: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversation threads for a user from memory storage."""
        # MemorySaver stores data in self.storage dict with keys as (thread_id, checkpoint_ns)
        # We need to filter by thread_id that matches user_identifier pattern

        threads_data = {}

        # Iterate through storage to find matching threads
        for (thread_id, checkpoint_ns), checkpoint_tuple in self.storage.items():
            # Check if thread belongs to user
            if thread_id == user_identifier or thread_id.startswith(
                f"{user_identifier}_"
            ):
                if thread_id not in threads_data:
                    threads_data[thread_id] = {
                        "checkpoints": [],
                        "last_checkpoint": None,
                    }

                threads_data[thread_id]["checkpoints"].append(checkpoint_tuple)

                # Track the latest checkpoint (highest checkpoint_id)
                checkpoint, metadata = checkpoint_tuple[0], checkpoint_tuple[1]
                if checkpoint:
                    if (
                        threads_data[thread_id]["last_checkpoint"] is None
                        or checkpoint["id"]
                        > threads_data[thread_id]["last_checkpoint"]["id"]
                    ):
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
        # Find the latest checkpoint for this thread
        latest_checkpoint = None
        latest_checkpoint_id = None

        for (tid, checkpoint_ns), checkpoint_tuple in self.storage.items():
            if tid == thread_id:
                checkpoint, metadata = checkpoint_tuple[0], checkpoint_tuple[1]
                if checkpoint:
                    if (
                        latest_checkpoint_id is None
                        or checkpoint["id"] > latest_checkpoint_id
                    ):
                        latest_checkpoint = checkpoint
                        latest_checkpoint_id = checkpoint["id"]

        if not latest_checkpoint:
            return []

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
