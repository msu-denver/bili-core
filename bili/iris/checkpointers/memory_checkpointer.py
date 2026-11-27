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
    from bili.iris.checkpointers.memory_checkpointer import QueryableMemorySaver

    checkpointer = QueryableMemorySaver()
    threads = checkpointer.get_user_threads("user@example.com")
"""

from typing import Any, Dict, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver

from bili.iris.checkpointers.base_checkpointer import QueryableCheckpointerMixin


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
        """Get all conversation threads for a user from memory storage.

        Returns the same keys as the PostgreSQL backend: ``last_updated`` is
        the ISO-8601 ``ts`` of the thread's newest checkpoint, and threads are
        ordered newest checkpoint first. Entries that hold no checkpoint (the
        placeholders a read of a never-saved thread leaves in the upstream
        storage) are not threads and are skipped.

        When the saver has a ``user_id``, a thread that user does not own is
        listed with its structure only (ids, checkpoint count, timestamp) and
        no content: no messages, message count, title or tags. The
        PostgreSQL backend likewise returns no messages for such a thread,
        because its content read refuses it.
        """
        summaries = []
        # list() copies the keys in one step, so a concurrent put() or
        # delete_thread() cannot resize the dict while it is iterated.
        for thread_id in list(self.storage.keys()):
            if not (
                thread_id == user_identifier
                or thread_id.startswith(f"{user_identifier}_")
            ):
                continue
            summary = self._summarize_thread(thread_id)
            if summary is not None:
                summaries.append(summary)

        summaries.sort(key=lambda item: item[0], reverse=True)
        threads = [thread for _, thread in summaries]

        # Apply pagination
        if offset > 0 or limit is not None:
            start = offset
            end = offset + limit if limit is not None else None
            threads = threads[start:end]

        return threads

    def _owns_thread(self, thread_id: str) -> bool:
        """Return True when no user_id is set or the user owns the thread."""
        return self.user_id is None or self.verify_thread_ownership(
            thread_id, self.user_id
        )

    def _summarize_thread(self, thread_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Build one get_user_threads entry and its newest checkpoint id.

        Storage maps thread_id -> checkpoint_ns -> checkpoint_id -> a tuple of
        serialized (checkpoint, metadata, parent_checkpoint_id); the checkpoint
        part omits channel_values, which live in self.blobs. The count and
        the timestamp cover every namespace. The messages, title and tags are
        the root namespace's state, read through get_tuple; the PostgreSQL
        backend differs here, reading title and tags from the newest
        checkpoint of any namespace.

        Returns None for a thread with no checkpoint, including one deleted
        by another Python thread while the listing runs. The lookup uses
        get(), because indexing the upstream defaultdict would re-create the
        deleted entry, and each namespace is copied with dict() before it is
        read, so a concurrent put() into it cannot resize it mid-iteration.
        """
        namespaces = self.storage.get(thread_id)
        if namespaces is None:
            return None
        snapshot = {ns: dict(cps) for ns, cps in list(namespaces.items())}
        newest = max(
            (
                (checkpoint_id, ns)
                for ns, cps in snapshot.items()
                for checkpoint_id in cps
            ),
            default=None,
        )
        if newest is None:
            return None
        newest_id, newest_ns = newest
        newest_checkpoint = self.serde.loads_typed(snapshot[newest_ns][newest_id][0])

        thread = {
            "thread_id": thread_id,
            "conversation_id": (
                thread_id.split("_", 1)[1] if "_" in thread_id else "default"
            ),
            "last_updated": newest_checkpoint.get("ts"),
            "checkpoint_count": sum(len(cps) for cps in snapshot.values()),
            "message_count": 0,
            "first_message": None,
            "last_message": None,
            "title": None,
            "tags": [],
        }
        if not self._owns_thread(thread_id):
            return newest_id, thread

        checkpoint_tuple = self._latest_root_tuple(thread_id)
        if checkpoint_tuple is not None:
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            thread["title"] = channel_values.get("title")
            thread["tags"] = channel_values.get("tags", [])
            thread["message_count"] = len(messages)
            thread["first_message"], thread["last_message"] = (
                self._first_and_last_human_text(messages)
            )
        return newest_id, thread

    @staticmethod
    def _first_and_last_human_text(messages: List[Any]) -> Tuple[Any, Any]:
        """Return the text of the first and last non-empty human messages."""
        first_message = None
        last_message = None
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
        return first_message, last_message

    def _latest_root_tuple(self, thread_id: str):
        """Return the newest root-namespace CheckpointTuple, or None.

        Reads through the upstream get_tuple so channel_values are rebuilt
        from self.blobs and deserialized with self.serde. The upstream read
        indexes the nested defaultdict, so it is called only when the root
        namespace already holds a checkpoint; otherwise it would leave an
        empty placeholder behind. A delete on another Python thread between
        the check and the read still can, and the read then returns None;
        _has_checkpoint does not count such a placeholder as a thread.
        Ownership is the caller's responsibility.
        """
        namespaces = self.storage.get(thread_id)
        if not namespaces or not namespaces.get(""):
            return None
        return MemorySaver.get_tuple(
            self, {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )

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

        checkpoint_tuple = self._latest_root_tuple(thread_id)
        if checkpoint_tuple is None:
            return []

        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = self._format_thread_messages(
            channel_values.get("messages", []), message_types
        )

        # Apply pagination
        if offset > 0 or limit is not None:
            end_index = offset + limit if limit is not None else None
            messages = messages[offset:end_index]

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread's checkpoints, pending writes and channel blobs.

        :return: True if anything of the thread was removed. An empty
            placeholder left by a read of a never-saved thread holds nothing
            and does not count.
        :raises PermissionError: If the thread belongs to another user.
        """
        # Validate thread ownership if user_id is set
        self._validate_thread_ownership(thread_id)

        existed = (
            self._has_checkpoint(thread_id)
            or any(key[0] == thread_id for key in self.writes)
            or any(key[0] == thread_id for key in self.blobs)
        )
        # Call MemorySaver.delete_thread directly because
        # QueryableCheckpointerMixin precedes MemorySaver in the MRO and
        # defines an abstract delete_thread method.
        MemorySaver.delete_thread(self, thread_id)
        return existed

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
        """Check if a thread has at least one stored checkpoint."""
        return self._has_checkpoint(thread_id)

    def _has_checkpoint(self, thread_id: str) -> bool:
        """Return True when any namespace of the thread holds a checkpoint.

        InMemorySaver.storage is a nested defaultdict that get_tuple() and
        list() index directly, so reading a thread that was never saved leaves
        an empty entry behind. Key membership therefore does not mean a
        checkpoint exists, and indexing here would itself create an entry, so
        the lookup uses get() and checks for a non-empty namespace.
        """
        namespaces = self.storage.get(thread_id)
        if not namespaces:
            return False
        return any(namespaces.values())
