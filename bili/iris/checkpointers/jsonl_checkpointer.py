"""
Module: jsonl_checkpointer

Local-file checkpoint saver that persists every graph superstep to an
append-only JSONL file.  No database server is required — the file is the
only dependency.

Designed for single-process, multi-thread usage (multiple concurrent runs on
different thread_ids within one process are fully safe thanks to an in-process
RLock).  Multi-process file locking is explicitly out of scope; if concurrent
processes need a shared store, use the PostgreSQL or MongoDB savers instead.

Classes:
    - JSONLCheckpointSaver:
      Implements the full BaseCheckpointSaver contract (sync + async) backed
      by a local JSONL file.  Also implements QueryableCheckpointerMixin and
      VersionedCheckpointerMixin so it is a drop-in peer of
      PruningPostgresSaver / PruningMongoDBSaver.

Factories:
    - get_jsonl_checkpointer(path, keep_last_n, user_id)
    - get_async_jsonl_checkpointer(path, keep_last_n, user_id)

On-disk format
--------------
Two record types are interleaved in one JSONL file (one JSON object per line):

Checkpoint record (record_type == "checkpoint")::

    {
      "record_type": "checkpoint",
      "thread_id":   "<str>",
      "checkpoint_ns": "<str>",
      "checkpoint_id": "<str>",
      "parent_checkpoint_id": "<str | null>",
      "ts": "<ISO-8601 UTC>",
      "checkpoint": {"type": "<serde type>", "data": "<base64 bytes>"},
      "metadata":   {"type": "<serde type>", "data": "<base64 bytes>"},
      "format_version": <int>
    }

Write record (record_type == "write")::

    {
      "record_type": "write",
      "thread_id":   "<str>",
      "checkpoint_ns": "<str>",
      "checkpoint_id": "<str>",
      "task_id":     "<str>",
      "task_path":   "<str>",
      "idx":         <int>,
      "channel":     "<str>",
      "value":       {"type": "<serde type>", "data": "<base64 bytes>"}
    }

The ``checkpoint`` blob is the FULL LangGraph checkpoint dict including
``channel_values``, serialized using ``self.serde.dumps_typed``.  This
deliberately stores the complete superstep state in each record, trading
file size for simplicity and auditability — every line is self-contained.

Example:
    from bili.iris.checkpointers.jsonl_checkpointer import get_jsonl_checkpointer

    saver = get_jsonl_checkpointer(path="~/.bili/aether.jsonl")
"""

import asyncio
import base64
import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from bili.iris.checkpointers.base_checkpointer import QueryableCheckpointerMixin
from bili.iris.checkpointers.versioning import (
    CURRENT_FORMAT_VERSION,
    VersionedCheckpointerMixin,
)
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Default file path when neither argument nor env var is provided
_DEFAULT_PATH = os.path.expanduser("~/.bili/checkpoints/aether.jsonl")


def _encode(saver: "JSONLCheckpointSaver", value: Any) -> Dict[str, str]:
    """Serialize *value* using saver.serde and return a JSON-safe envelope."""
    type_str, raw_bytes = saver.serde.dumps_typed(value)
    return {"type": type_str, "data": base64.b64encode(raw_bytes).decode("ascii")}


def _decode(saver: "JSONLCheckpointSaver", envelope: Dict[str, str]) -> Any:
    """Deserialize a JSON-safe envelope using saver.serde."""
    raw_bytes = base64.b64decode(envelope["data"])
    return saver.serde.loads_typed((envelope["type"], raw_bytes))


class JSONLCheckpointSaver(
    VersionedCheckpointerMixin,
    QueryableCheckpointerMixin,
    BaseCheckpointSaver,
):
    """Local-file checkpoint saver backed by an append-only JSONL file.

    Every graph superstep produces one ``checkpoint`` record in the file;
    intermediate writes produce ``write`` records.  A lazy in-memory index
    is built on first access.  All write paths hold a ``threading.RLock``
    so concurrent calls from multiple threads (one per agent execution) are
    safe within a single process.

    Args:
        path: Absolute or ``~``-prefixed path to the ``.jsonl`` file.
            Defaults to ``JSONL_CHECKPOINT_PATH`` env var, then
            ``~/.bili/checkpoints/aether.jsonl``.
        keep_last_n: Number of most recent checkpoints to retain per
            ``(thread_id, checkpoint_ns)`` pair.  ``-1`` disables pruning.
        user_id: When set, enables thread ownership validation — every
            ``put``/``get_tuple`` call checks that the thread_id matches
            ``user_id`` or ``{user_id}_*``.
    """

    # Identifies this checkpointer type in the migration registry
    checkpointer_type: str = "jsonl"
    format_version: int = CURRENT_FORMAT_VERSION

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        keep_last_n: int = -1,
        user_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        resolved = path or os.environ.get("JSONL_CHECKPOINT_PATH") or _DEFAULT_PATH
        self._path: str = os.path.expanduser(resolved)
        self.keep_last_n = keep_last_n
        self.user_id = user_id

        self._lock = threading.RLock()
        self._loaded = False

        # (thread_id, ns) -> list[checkpoint_record_dict], append-ordered
        self._checkpoints: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        # (thread_id, ns, checkpoint_id) -> list[write_record_dict]
        self._writes: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

        LOGGER.info(
            "JSONLCheckpointSaver initialised (path=%s, keep_last_n=%d%s)",
            self._path,
            keep_last_n,
            f", user_id={user_id}" if user_id else "",
        )

    # ------------------------------------------------------------------
    # Public path property
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        """Absolute path to the backing JSONL file."""
        return self._path

    # ------------------------------------------------------------------
    # In-memory index management
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the JSONL file into the in-memory index (idempotent)."""
        with self._lock:
            if self._loaded:
                return
            self._load_from_disk()
            self._loaded = True

    def _load_from_disk(self) -> None:
        """Read all records from the JSONL file and populate the index."""
        if not os.path.exists(self._path):
            LOGGER.debug("JSONL file not yet created: %s", self._path)
            return

        loaded_checkpoints = 0
        loaded_writes = 0
        errors = 0

        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Skipping malformed JSON on line %d of %s", lineno, self._path
                    )
                    errors += 1
                    continue

                rt = record.get("record_type")
                if rt == "checkpoint":
                    key = (record["thread_id"], record.get("checkpoint_ns", ""))
                    self._checkpoints[key].append(record)
                    loaded_checkpoints += 1
                elif rt == "write":
                    key = (
                        record["thread_id"],
                        record.get("checkpoint_ns", ""),
                        record["checkpoint_id"],
                    )
                    self._writes[key].append(record)
                    loaded_writes += 1
                else:
                    LOGGER.debug(
                        "Unknown record_type '%s' on line %d; skipped", rt, lineno
                    )

        LOGGER.info(
            "Loaded %d checkpoints, %d writes from %s (%d errors)",
            loaded_checkpoints,
            loaded_writes,
            self._path,
            errors,
        )

    def _append_record(self, record: Dict[str, Any]) -> None:
        """Append one JSON record to the file (caller must hold self._lock)."""
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self, thread_id: str, checkpoint_ns: str) -> None:
        """Remove old checkpoints beyond keep_last_n (caller holds lock)."""
        if self.keep_last_n < 0:
            return
        key = (thread_id, checkpoint_ns)
        records = self._checkpoints.get(key, [])
        if len(records) <= self.keep_last_n:
            return

        # Keep the most recent keep_last_n by insertion order
        to_remove = records[: len(records) - self.keep_last_n]
        kept = records[len(records) - self.keep_last_n :]
        self._checkpoints[key] = kept

        # Remove associated writes
        for rec in to_remove:
            wkey = (thread_id, checkpoint_ns, rec["checkpoint_id"])
            self._writes.pop(wkey, None)

        # Rewrite the file from the in-memory index (compact)
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """Rewrite the entire JSONL file from the in-memory index (caller holds lock)."""
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for records in self._checkpoints.values():
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for write_list in self._writes.values():
                    for rec in write_list:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self._path)
        except Exception:  # pylint: disable=broad-exception-caught
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ------------------------------------------------------------------
    # BaseCheckpointSaver sync interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return the requested checkpoint tuple, or None if not found.

        Args:
            config: Runnable config with ``thread_id`` (and optionally
                ``checkpoint_id`` and ``checkpoint_ns``).

        Returns:
            ``CheckpointTuple`` with full channel_values, or ``None``.

        Raises:
            PermissionError: If user_id is set and thread_id is foreign.
        """
        self._ensure_loaded()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        self._validate_thread_ownership(thread_id)

        with self._lock:
            records = self._checkpoints.get((thread_id, checkpoint_ns), [])
            if not records:
                return None

            checkpoint_id = get_checkpoint_id(config)
            if checkpoint_id:
                record = next(
                    (r for r in records if r["checkpoint_id"] == checkpoint_id), None
                )
            else:
                record = records[-1]  # most recent by insertion order

            if record is None:
                return None

            checkpoint: Checkpoint = _decode(self, record["checkpoint"])
            metadata: CheckpointMetadata = _decode(self, record["metadata"])

            wkey = (thread_id, checkpoint_ns, record["checkpoint_id"])
            write_records = self._writes.get(wkey, [])
            pending_writes = [
                (w["task_id"], w["channel"], _decode(self, w["value"]))
                for w in write_records
            ]

            parent_config: Optional[RunnableConfig] = None
            if record.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": record["parent_checkpoint_id"],
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": record["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=parent_config,
            )

    def list(  # pylint: disable=redefined-builtin
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoint tuples matching the criteria, most recent first.

        Args:
            config: Base config for filtering (uses ``thread_id``).  If
                ``None``, all threads are listed.
            filter: Metadata key/value pairs that must all match.
            before: Yield only checkpoints whose checkpoint_id is
                lexicographically less than the checkpoint_id in *before*.
            limit: Maximum number of tuples to yield.

        Yields:
            ``CheckpointTuple`` objects, most-recent first.
        """
        self._ensure_loaded()

        with self._lock:
            if config:
                thread_id = config["configurable"]["thread_id"]
                thread_ids = [thread_id]
                config_ns = config["configurable"].get("checkpoint_ns", "")
            else:
                all_keys = list(self._checkpoints.keys())
                thread_ids = list({k[0] for k in all_keys})
                config_ns = None

            before_id = get_checkpoint_id(before) if before else None
            remaining = limit

            for tid in thread_ids:
                namespaces = (
                    [config_ns]
                    if config_ns is not None
                    else [k[1] for k in self._checkpoints if k[0] == tid]
                )

                for ns in namespaces:
                    records = self._checkpoints.get((tid, ns), [])
                    # Most recent first
                    for record in reversed(records):
                        cid = record["checkpoint_id"]

                        if before_id and cid >= before_id:
                            continue

                        checkpoint: Checkpoint = _decode(self, record["checkpoint"])
                        metadata: CheckpointMetadata = _decode(self, record["metadata"])

                        if filter and not all(
                            metadata.get(k) == v for k, v in filter.items()
                        ):
                            continue

                        if remaining is not None and remaining <= 0:
                            return
                        if remaining is not None:
                            remaining -= 1

                        wkey = (tid, ns, cid)
                        write_records = self._writes.get(wkey, [])
                        pending_writes = [
                            (w["task_id"], w["channel"], _decode(self, w["value"]))
                            for w in write_records
                        ]

                        parent_config: Optional[RunnableConfig] = None
                        if record.get("parent_checkpoint_id"):
                            parent_config = {
                                "configurable": {
                                    "thread_id": tid,
                                    "checkpoint_ns": ns,
                                    "checkpoint_id": record["parent_checkpoint_id"],
                                }
                            }

                        yield CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": tid,
                                    "checkpoint_ns": ns,
                                    "checkpoint_id": cid,
                                }
                            },
                            checkpoint=checkpoint,
                            metadata=metadata,
                            pending_writes=pending_writes,
                            parent_config=parent_config,
                        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and return the updated config.

        Args:
            config: Current runnable configuration.
            checkpoint: Checkpoint dict including ``channel_values``.
            metadata: Checkpoint metadata.
            new_versions: Channel versions as of this write (passed through
                for API compatibility; values are embedded in the serialized
                checkpoint blob).

        Returns:
            Updated ``RunnableConfig`` with the checkpoint_id.

        Raises:
            PermissionError: If user_id is set and thread_id is foreign.

        Note:
            There is a narrow partial-write window: the append to the JSONL file
            succeeds before ``_prune`` rewrites it.  If the process crashes between
            those two steps, pruned records will reappear on the next
            ``_load_from_disk`` call, but no data is lost or corrupted.
        """
        self._ensure_loaded()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id: Optional[str] = config["configurable"].get(
            "checkpoint_id"
        )
        checkpoint_id: str = checkpoint["id"]

        self._validate_thread_ownership(thread_id)

        versioned_metadata = dict(metadata) if metadata else {}
        versioned_metadata["format_version"] = self.format_version

        record: Dict[str, Any] = {
            "record_type": "checkpoint",
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "ts": checkpoint.get("ts", datetime.now(timezone.utc).isoformat()),
            "checkpoint": _encode(self, checkpoint),
            "metadata": _encode(
                self, get_checkpoint_metadata(config, versioned_metadata)
            ),
            "format_version": self.format_version,
        }

        with self._lock:
            self._append_record(record)
            key = (thread_id, checkpoint_ns)
            self._checkpoints[key].append(record)
            self._prune(thread_id, checkpoint_ns)

        LOGGER.debug(
            "Saved checkpoint %s for thread %s (ns=%s)",
            checkpoint_id,
            thread_id,
            checkpoint_ns,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist intermediate writes for a checkpoint.

        Args:
            config: Runnable configuration identifying the checkpoint.
            writes: Sequence of ``(channel, value)`` tuples.
            task_id: Identifier of the task producing the writes.
            task_path: Task path for structured tracing (passed through).
        """
        self._ensure_loaded()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        self._validate_thread_ownership(thread_id)

        wkey = (thread_id, checkpoint_ns, checkpoint_id)

        with self._lock:
            existing = {
                (w["task_id"], w["idx"]): True for w in self._writes.get(wkey, [])
            }

            for idx, (channel, value) in enumerate(writes):
                real_idx = WRITES_IDX_MAP.get(channel, idx)
                if real_idx >= 0 and (task_id, real_idx) in existing:
                    continue

                record: Dict[str, Any] = {
                    "record_type": "write",
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": real_idx,
                    "channel": channel,
                    "value": _encode(self, value),
                }
                self._append_record(record)
                self._writes[wkey].append(record)
                existing[(task_id, real_idx)] = True

    # ------------------------------------------------------------------
    # BaseCheckpointSaver async interface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version of get_tuple (delegates to sync via thread pool)."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        """Async version of list — collects sync results and yields them."""
        results = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in results:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put (delegates to sync via thread pool)."""
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes (delegates to sync via thread pool)."""
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    # ------------------------------------------------------------------
    # QueryableCheckpointerMixin implementation
    # ------------------------------------------------------------------

    def get_user_threads(
        self,
        user_identifier: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return thread metadata for all threads belonging to *user_identifier*.

        Threads are matched when ``thread_id == user_identifier`` or
        ``thread_id.startswith(f"{user_identifier}_")``.

        Args:
            user_identifier: User email or ID.
            limit: Maximum number of threads to return.
            offset: Number of threads to skip.

        Returns:
            List of thread dicts (thread_id, conversation_id, checkpoint_count,
            message_count, first_message, last_message, title, tags,
            last_updated).
        """
        self._ensure_loaded()

        with self._lock:
            matching_keys = [
                k
                for k in self._checkpoints
                if k[0] == user_identifier or k[0].startswith(f"{user_identifier}_")
            ]

        threads = []
        for key in matching_keys:
            tid = key[0]
            with self._lock:
                records = list(self._checkpoints.get(key, []))
            if not records:
                continue

            latest_record = records[-1]
            conversation_id = tid.split("_", 1)[1] if "_" in tid else "default"

            checkpoint_obj = None
            try:
                checkpoint_obj = _decode(self, latest_record["checkpoint"])
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            first_message = None
            last_message = None
            message_count = 0
            title = None
            tags: List[str] = []

            if checkpoint_obj:
                channel_values = checkpoint_obj.get("channel_values", {})
                messages = channel_values.get("messages", [])
                title = channel_values.get("title")
                tags = channel_values.get("tags", [])
                if messages:
                    message_count = len(messages)
                    for msg in messages:
                        if hasattr(msg, "content") and msg.content:
                            if msg.__class__.__name__ == "HumanMessage":
                                content = msg.content
                                if isinstance(content, list):
                                    content = " ".join(
                                        p.get("text", "")
                                        for p in content
                                        if isinstance(p, dict)
                                        and p.get("type") == "text"
                                    )
                                if first_message is None:
                                    first_message = content
                                last_message = content

            threads.append(
                {
                    "thread_id": tid,
                    "conversation_id": conversation_id,
                    "last_updated": latest_record.get("ts"),
                    "checkpoint_count": len(records),
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "title": title,
                    "tags": tags,
                }
            )

        # Sort most-recent first
        threads.sort(key=lambda t: t["last_updated"] or "", reverse=True)

        if offset > 0 or limit is not None:
            end = offset + limit if limit is not None else None
            threads = threads[offset:end]

        return threads

    def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        message_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return messages from the latest checkpoint for *thread_id*.

        Args:
            thread_id: Thread to read.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.
            message_types: Filter to specific message class names.

        Returns:
            List of ``{role, content, timestamp}`` dicts.
        """
        self._validate_thread_ownership(thread_id)
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        tup = self.get_tuple(config)
        if not tup or not tup.checkpoint:
            return []

        channel_values = tup.checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            msg_class = (
                msg.__class__.__name__ if hasattr(msg, "__class__") else "unknown"
            )

            if message_types is not None and msg_class not in message_types:
                continue

            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )

            if msg_class == "AIMessage":
                content = self._strip_thinking_blocks(content)
                if not content:
                    continue

            role_map = {
                "HumanMessage": "user",
                "AIMessage": "assistant",
                "SystemMessage": "system",
                "ToolMessage": "tool",
                "FunctionMessage": "function",
            }
            messages.append(
                {
                    "role": role_map.get(msg_class, "unknown"),
                    "content": content,
                    "timestamp": None,
                }
            )

        if offset > 0 or limit is not None:
            end = offset + limit if limit is not None else None
            messages = messages[offset:end]

        return messages

    def delete_thread(self, thread_id: str) -> bool:
        """Remove all checkpoints and writes for *thread_id* from index and file.

        Args:
            thread_id: Thread to delete.

        Returns:
            ``True`` if any records were removed.
        """
        self._validate_thread_ownership(thread_id)
        self._ensure_loaded()

        with self._lock:
            keys_to_delete = [k for k in self._checkpoints if k[0] == thread_id]
            if not keys_to_delete:
                return False

            for k in keys_to_delete:
                del self._checkpoints[k]

            write_keys = [k for k in self._writes if k[0] == thread_id]
            for k in write_keys:
                del self._writes[k]

            self._rewrite_file()
        return True

    def get_user_stats(self, user_identifier: str) -> Dict[str, Any]:
        """Return aggregate statistics for a user.

        Args:
            user_identifier: User email or ID.

        Returns:
            Dict with total_threads, total_messages, total_checkpoints,
            oldest_thread, newest_thread.
        """
        threads = self.get_user_threads(user_identifier)
        if not threads:
            return {
                "total_threads": 0,
                "total_messages": 0,
                "total_checkpoints": 0,
                "oldest_thread": None,
                "newest_thread": None,
            }

        total_messages = sum(t["message_count"] for t in threads)
        total_checkpoints = sum(t["checkpoint_count"] for t in threads)
        timestamps = [t["last_updated"] for t in threads if t["last_updated"]]
        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "total_checkpoints": total_checkpoints,
            "oldest_thread": min(timestamps, default=None),
            "newest_thread": max(timestamps, default=None),
        }

    def thread_exists(self, thread_id: str) -> bool:
        """Return ``True`` if any checkpoint exists for *thread_id*.

        Args:
            thread_id: Thread to check.
        """
        self._ensure_loaded()
        with self._lock:
            return any(k[0] == thread_id for k in self._checkpoints)

    # ------------------------------------------------------------------
    # VersionedCheckpointerMixin abstract method implementations
    # ------------------------------------------------------------------

    def _get_raw_checkpoint(
        self, thread_id: str, checkpoint_ns: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent raw checkpoint record for *thread_id*."""
        self._ensure_loaded()
        with self._lock:
            records = self._checkpoints.get((thread_id, checkpoint_ns), [])
            return records[-1] if records else None

    def _replace_raw_checkpoint(
        self,
        thread_id: str,
        document: Dict[str, Any],
        checkpoint_ns: str = "",
    ) -> bool:
        """Replace the most recent checkpoint record in the index and rewrite file.

        Args:
            thread_id: Thread to update.
            document: Migrated record to write back.
            checkpoint_ns: Checkpoint namespace.

        Returns:
            ``True`` if a record was found and replaced.
        """
        if "checkpoint_id" not in document:
            LOGGER.warning(
                "Cannot replace checkpoint without checkpoint_id for thread %s",
                thread_id,
            )
            return False

        with self._lock:
            key = (thread_id, checkpoint_ns)
            records = self._checkpoints.get(key, [])
            for i, rec in enumerate(records):
                if rec["checkpoint_id"] == document["checkpoint_id"]:
                    records[i] = document
                    self._rewrite_file()
                    return True
        return False

    def _archive_checkpoint(
        self,
        thread_id: str,
        document: Dict[str, Any],
        error: Exception,
    ) -> None:
        """Write a failed-migration record to a sidecar ``.archive`` file.

        Args:
            thread_id: Thread ID of the failed checkpoint.
            document: Raw record that could not be migrated.
            error: Exception raised during migration.
        """
        archive_path = self._path + ".archive"
        archive_record = {
            "record_type": "archive",
            "thread_id": thread_id,
            "checkpoint_id": document.get("checkpoint_id"),
            "migration_error": str(error),
            "raw_document": document,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        dir_path = os.path.dirname(archive_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(archive_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(archive_record, ensure_ascii=False) + "\n")
        LOGGER.info(
            "Archived failed checkpoint for thread %s to %s", thread_id, archive_path
        )


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------


def get_jsonl_checkpointer(
    path: Optional[str] = None,
    keep_last_n: int = -1,
    user_id: Optional[str] = None,
) -> JSONLCheckpointSaver:
    """Create and return a JSONLCheckpointSaver.

    Args:
        path: Path to the JSONL file.  Falls back to ``JSONL_CHECKPOINT_PATH``
            env var then ``~/.bili/checkpoints/aether.jsonl``.
        keep_last_n: Pruning limit per thread (``-1`` = unlimited).
        user_id: Enable thread ownership validation for this user.

    Returns:
        A ready-to-use ``JSONLCheckpointSaver``.
    """
    return JSONLCheckpointSaver(path=path, keep_last_n=keep_last_n, user_id=user_id)


async def get_async_jsonl_checkpointer(
    path: Optional[str] = None,
    keep_last_n: int = -1,
    user_id: Optional[str] = None,
) -> JSONLCheckpointSaver:
    """Async factory — returns a JSONLCheckpointSaver (no async setup needed).

    The saver's async methods delegate to sync via ``asyncio.to_thread``, so
    no coroutine-based setup is required.  This function exists to match the
    async-factory convention of the PostgreSQL and MongoDB savers.

    Args:
        path: Path to the JSONL file.
        keep_last_n: Pruning limit (``-1`` = unlimited).
        user_id: Enable thread ownership validation.

    Returns:
        A ``JSONLCheckpointSaver`` instance.
    """
    return JSONLCheckpointSaver(path=path, keep_last_n=keep_last_n, user_id=user_id)
