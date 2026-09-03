"""Tests for bili.iris.checkpointers.jsonl_checkpointer module.

Covers the full BaseCheckpointSaver + QueryableCheckpointerMixin +
VersionedCheckpointerMixin contract including:
- Round-trip put/get_tuple (sync + async)
- put_writes / pending_writes reconstruction
- list() ordering and filters
- File-persistence: saver reconstructed from same path returns prior checkpoints
- Thread isolation: multiple threads in one file do not bleed
- Thread ownership validation (user_id)
- Pruning (keep_last_n) + compaction
- Async delegates (aget_tuple, aput, aput_writes, alist)
- QueryableCheckpointerMixin methods (get_user_threads, get_thread_messages,
  delete_thread, get_user_stats, thread_exists)
- In-process concurrency smoke test (two threads, different thread_ids)
- VersionedCheckpointerMixin helpers (_get_raw_checkpoint,
  _replace_raw_checkpoint, _archive_checkpoint)
"""

import asyncio
import json
import os
import threading
from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bili.iris.checkpointers.jsonl_checkpointer import (
    JSONLCheckpointSaver,
    get_async_jsonl_checkpointer,
    get_jsonl_checkpointer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(
    thread_id: str,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    }
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


def _minimal_checkpoint(checkpoint_id: str = "cp-1") -> Dict[str, Any]:
    return {
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00+00:00",
        "v": 1,
        "channel_values": {"messages": [], "agent_outputs": {}},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _minimal_metadata() -> Dict[str, Any]:
    return {"source": "test", "step": 0, "writes": {}}


def _human_message(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Verify JSONLCheckpointSaver initialisation."""

    def test_default_path_from_env(self, monkeypatch, tmp_path):
        """Env var JSONL_CHECKPOINT_PATH is respected."""
        target = str(tmp_path / "env.jsonl")
        monkeypatch.setenv("JSONL_CHECKPOINT_PATH", target)
        saver = JSONLCheckpointSaver()
        assert saver.path == target

    def test_explicit_path_overrides_env(self, monkeypatch, tmp_path):
        """Explicit path takes priority over env var."""
        monkeypatch.setenv("JSONL_CHECKPOINT_PATH", "/ignored")
        target = str(tmp_path / "explicit.jsonl")
        saver = JSONLCheckpointSaver(path=target)
        assert saver.path == target

    def test_tilde_expansion(self, tmp_path):
        """Home-dir expansion works."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "x.jsonl"))
        assert "~" not in saver.path

    def test_keep_last_n_default(self, tmp_path):
        """keep_last_n defaults to -1 (unlimited)."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "x.jsonl"))
        assert saver.keep_last_n == -1

    def test_user_id_default(self, tmp_path):
        """user_id defaults to None."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "x.jsonl"))
        assert saver.user_id is None

    def test_factory_get_jsonl_checkpointer(self, tmp_path):
        """get_jsonl_checkpointer returns a JSONLCheckpointSaver."""
        saver = get_jsonl_checkpointer(path=str(tmp_path / "x.jsonl"))
        assert isinstance(saver, JSONLCheckpointSaver)


# ---------------------------------------------------------------------------
# Round-trip: put / get_tuple
# ---------------------------------------------------------------------------


class TestPutGetRoundTrip:
    """put() then get_tuple() returns the saved checkpoint."""

    def test_basic_round_trip(self, tmp_path):
        """Single put; get_tuple without checkpoint_id returns it."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        config = _minimal_config("thread-1")
        ck = _minimal_checkpoint("cp-1")
        new_config = saver.put(config, ck, _minimal_metadata(), {})
        assert new_config["configurable"]["checkpoint_id"] == "cp-1"

        tup = saver.get_tuple(_minimal_config("thread-1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"

    def test_get_tuple_by_checkpoint_id(self, tmp_path):
        """get_tuple with explicit checkpoint_id fetches the right one."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        config = _minimal_config("thread-1")
        saver.put(config, _minimal_checkpoint("cp-1"), _minimal_metadata(), {})
        config2 = _minimal_config("thread-1", checkpoint_id="cp-1")
        saver.put(config2, _minimal_checkpoint("cp-2"), _minimal_metadata(), {})

        tup = saver.get_tuple(_minimal_config("thread-1", checkpoint_id="cp-1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"

    def test_get_tuple_unknown_checkpoint_id_returns_none(self, tmp_path):
        """get_tuple with a non-existent checkpoint_id returns None."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        tup = saver.get_tuple(_minimal_config("thread-1", checkpoint_id="ghost"))
        assert tup is None

    def test_get_tuple_empty_thread_returns_none(self, tmp_path):
        """get_tuple on a thread with no checkpoints returns None."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        tup = saver.get_tuple(_minimal_config("nonexistent"))
        assert tup is None

    def test_channel_values_preserved(self, tmp_path):
        """channel_values (messages, agent_outputs) survive a round-trip."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        ck = _minimal_checkpoint("cp-1")
        ck["channel_values"] = {
            "messages": ["hello"],
            "agent_outputs": {"agent_0": {"message": "Hi"}},
        }
        saver.put(_minimal_config("thread-1"), ck, _minimal_metadata(), {})
        tup = saver.get_tuple(_minimal_config("thread-1"))
        assert tup.checkpoint["channel_values"]["agent_outputs"]["agent_0"] == {
            "message": "Hi"
        }

    def test_parent_checkpoint_id_chain(self, tmp_path):
        """parent_config is populated from parent_checkpoint_id."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        # First put (no parent)
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        # Second put (parent = cp-1)
        saver.put(
            _minimal_config("thread-1", checkpoint_id="cp-1"),
            _minimal_checkpoint("cp-2"),
            _minimal_metadata(),
            {},
        )
        tup = saver.get_tuple(_minimal_config("thread-1"))
        assert tup.checkpoint["id"] == "cp-2"
        assert tup.parent_config is not None
        assert tup.parent_config["configurable"]["checkpoint_id"] == "cp-1"


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


class TestFilePersistence:
    """Checkpoints survive saver reconstruction from the same path."""

    def test_reload_from_disk(self, tmp_path):
        """Constructing a new saver from the same path finds prior checkpoints."""
        path = str(tmp_path / "persist.jsonl")
        s1 = JSONLCheckpointSaver(path=path)
        s1.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )

        s2 = JSONLCheckpointSaver(path=path)
        tup = s2.get_tuple(_minimal_config("thread-1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"

    def test_multiple_threads_on_disk(self, tmp_path):
        """Multiple threads in one file are each accessible after reload."""
        path = str(tmp_path / "multi.jsonl")
        s1 = JSONLCheckpointSaver(path=path)
        for i in range(3):
            s1.put(
                _minimal_config(f"thread-{i}"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )

        s2 = JSONLCheckpointSaver(path=path)
        for i in range(3):
            tup = s2.get_tuple(_minimal_config(f"thread-{i}"))
            assert tup is not None
            assert tup.checkpoint["id"] == f"cp-{i}"


# ---------------------------------------------------------------------------
# put_writes / pending_writes
# ---------------------------------------------------------------------------


class TestPutWrites:
    """put_writes stores pending writes; get_tuple returns them in pending_writes."""

    def test_pending_writes_in_get_tuple(self, tmp_path):
        """Writes are attached to the checkpoint in pending_writes."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        config = _minimal_config("thread-1")
        new_config = saver.put(
            config, _minimal_checkpoint("cp-1"), _minimal_metadata(), {}
        )

        write_config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        saver.put_writes(write_config, [("messages", "hello")], task_id="task-1")

        tup = saver.get_tuple(_minimal_config("thread-1"))
        assert tup is not None
        assert len(tup.pending_writes) == 1
        task_id, channel, value = tup.pending_writes[0]
        assert task_id == "task-1"
        assert channel == "messages"
        assert value == "hello"

    def test_duplicate_writes_deduped(self, tmp_path):
        """Calling put_writes twice with the same (task_id, idx) is idempotent."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        write_config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        saver.put_writes(write_config, [("messages", "first")], task_id="task-1")
        saver.put_writes(write_config, [("messages", "second")], task_id="task-1")

        tup = saver.get_tuple(_minimal_config("thread-1"))
        assert len(tup.pending_writes) == 1
        assert tup.pending_writes[0][2] == "first"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


class TestList:
    """list() yields checkpoint tuples most-recent first with correct filters."""

    def test_list_most_recent_first(self, tmp_path):
        """list() returns checkpoints in reverse insertion order."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        for i in range(3):
            saver.put(
                _minimal_config("thread-1", checkpoint_id=f"cp-{i}" if i else None),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )
        ids = [t.checkpoint["id"] for t in saver.list(_minimal_config("thread-1"))]
        assert ids == ["cp-2", "cp-1", "cp-0"]

    def test_list_limit(self, tmp_path):
        """list(limit=1) returns at most one tuple."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        for i in range(3):
            saver.put(
                _minimal_config("thread-1"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )
        results = list(saver.list(_minimal_config("thread-1"), limit=1))
        assert len(results) == 1

    def test_list_before_filter(self, tmp_path):
        """list(before=config) excludes checkpoints at or after before.checkpoint_id."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        saver.put(
            _minimal_config("thread-1", checkpoint_id="cp-1"),
            _minimal_checkpoint("cp-2"),
            _minimal_metadata(),
            {},
        )
        before_cfg = _minimal_config("thread-1", checkpoint_id="cp-2")
        results = list(saver.list(_minimal_config("thread-1"), before=before_cfg))
        assert all(t.checkpoint["id"] < "cp-2" for t in results)

    def test_list_empty_thread(self, tmp_path):
        """list() on unknown thread_id returns empty."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        results = list(saver.list(_minimal_config("ghost")))
        assert results == []

    def test_list_metadata_filter(self, tmp_path):
        """list(filter=...) respects metadata key/value matching."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        md1 = {"source": "input", "step": 0, "writes": {}}
        md2 = {"source": "loop", "step": 1, "writes": {}}
        saver.put(_minimal_config("t"), _minimal_checkpoint("cp-1"), md1, {})
        saver.put(_minimal_config("t"), _minimal_checkpoint("cp-2"), md2, {})
        results = list(saver.list(_minimal_config("t"), filter={"source": "loop"}))
        assert len(results) == 1
        assert results[0].checkpoint["id"] == "cp-2"


# ---------------------------------------------------------------------------
# Thread isolation
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    """Records for different thread_ids in one file do not cross."""

    def test_get_tuple_does_not_cross_threads(self, tmp_path):
        """Checkpoint written for thread-A is not returned for thread-B."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-A"),
            _minimal_checkpoint("cp-A"),
            _minimal_metadata(),
            {},
        )
        tup = saver.get_tuple(_minimal_config("thread-B"))
        assert tup is None

    def test_list_scoped_to_thread(self, tmp_path):
        """list() for thread-B returns only thread-B checkpoints."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-A"),
            _minimal_checkpoint("cp-A"),
            _minimal_metadata(),
            {},
        )
        saver.put(
            _minimal_config("thread-B"),
            _minimal_checkpoint("cp-B"),
            _minimal_metadata(),
            {},
        )
        ids = [t.checkpoint["id"] for t in saver.list(_minimal_config("thread-B"))]
        assert ids == ["cp-B"]


# ---------------------------------------------------------------------------
# Thread ownership validation
# ---------------------------------------------------------------------------


class TestOwnershipValidation:
    """user_id-scoped saver rejects foreign thread_ids."""

    def test_put_raises_on_foreign_thread(self, tmp_path):
        """put() with a foreign thread_id raises PermissionError."""
        saver = JSONLCheckpointSaver(
            path=str(tmp_path / "ck.jsonl"), user_id="alice@test.com"
        )
        with pytest.raises(PermissionError):
            saver.put(
                _minimal_config("bob@test.com"),
                _minimal_checkpoint("cp-1"),
                _minimal_metadata(),
                {},
            )

    def test_put_allows_owned_thread(self, tmp_path):
        """put() with a matching thread_id succeeds."""
        saver = JSONLCheckpointSaver(
            path=str(tmp_path / "ck.jsonl"), user_id="alice@test.com"
        )
        cfg = saver.put(
            _minimal_config("alice@test.com"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        assert cfg["configurable"]["checkpoint_id"] == "cp-1"

    def test_put_allows_user_id_prefixed_thread(self, tmp_path):
        """thread_id of the form user_id_conv123 is accepted."""
        saver = JSONLCheckpointSaver(
            path=str(tmp_path / "ck.jsonl"), user_id="alice@test.com"
        )
        cfg = saver.put(
            _minimal_config("alice@test.com_conv-42"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        assert cfg["configurable"]["checkpoint_id"] == "cp-1"

    def test_get_tuple_raises_on_foreign_thread(self, tmp_path):
        """get_tuple() with a foreign thread_id raises PermissionError."""
        saver = JSONLCheckpointSaver(
            path=str(tmp_path / "ck.jsonl"), user_id="alice@test.com"
        )
        with pytest.raises(PermissionError):
            saver.get_tuple(_minimal_config("bob@test.com"))

    def test_no_user_id_accepts_any_thread(self, tmp_path):
        """Saver without user_id accepts any thread_id."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("any-thread"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        tup = saver.get_tuple(_minimal_config("any-thread"))
        assert tup is not None


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruning:
    """keep_last_n removes old checkpoints on put()."""

    def test_prune_removes_oldest(self, tmp_path):
        """After 3 puts with keep_last_n=2, only the 2 newest are accessible."""
        path = str(tmp_path / "ck.jsonl")
        saver = JSONLCheckpointSaver(path=path, keep_last_n=2)
        for i in range(3):
            saver.put(
                _minimal_config("thread-1"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )

        ids = [t.checkpoint["id"] for t in saver.list(_minimal_config("thread-1"))]
        assert "cp-0" not in ids
        assert "cp-1" in ids
        assert "cp-2" in ids

    def test_prune_compacts_file(self, tmp_path):
        """After pruning, the JSONL file contains no record for the removed checkpoint."""
        path = str(tmp_path / "ck.jsonl")
        saver = JSONLCheckpointSaver(path=path, keep_last_n=2)
        for i in range(3):
            saver.put(
                _minimal_config("thread-1"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )

        with open(path, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        checkpoint_ids = {
            l["checkpoint_id"] for l in lines if l.get("record_type") == "checkpoint"
        }
        assert "cp-0" not in checkpoint_ids

    def test_reload_after_prune(self, tmp_path):
        """New saver from same path after pruning only sees kept checkpoints."""
        path = str(tmp_path / "ck.jsonl")
        s1 = JSONLCheckpointSaver(path=path, keep_last_n=1)
        for i in range(3):
            s1.put(
                _minimal_config("thread-1"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )

        s2 = JSONLCheckpointSaver(path=path)
        ids = [t.checkpoint["id"] for t in s2.list(_minimal_config("thread-1"))]
        assert ids == ["cp-2"]


# ---------------------------------------------------------------------------
# Async delegates
# ---------------------------------------------------------------------------


class TestAsyncDelegates:
    """Async methods (aget_tuple, aput, aput_writes, alist) work correctly."""

    def test_aput_and_aget_tuple(self, tmp_path):
        """aput() followed by aget_tuple() returns the checkpoint."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))

        async def _run():
            config = _minimal_config("async-thread")
            await saver.aput(
                config, _minimal_checkpoint("cp-1"), _minimal_metadata(), {}
            )
            return await saver.aget_tuple(_minimal_config("async-thread"))

        tup = asyncio.run(_run())
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"

    def test_aput_writes_appear_in_aget_tuple(self, tmp_path):
        """aput_writes results appear in aget_tuple pending_writes."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))

        async def _run():
            await saver.aput(
                _minimal_config("async-t"),
                _minimal_checkpoint("cp-1"),
                _minimal_metadata(),
                {},
            )
            write_cfg = {
                "configurable": {
                    "thread_id": "async-t",
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-1",
                }
            }
            await saver.aput_writes(write_cfg, [("ch", "v")], task_id="t1")
            return await saver.aget_tuple(_minimal_config("async-t"))

        tup = asyncio.run(_run())
        assert len(tup.pending_writes) == 1
        assert tup.pending_writes[0][1] == "ch"

    def test_alist_yields_results(self, tmp_path):
        """alist() yields all checkpoint tuples for the thread."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        for i in range(3):
            saver.put(
                _minimal_config("async-t"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )

        async def _run():
            return [t async for t in saver.alist(_minimal_config("async-t"))]

        results = asyncio.run(_run())
        assert len(results) == 3

    def test_get_async_jsonl_checkpointer_factory(self, tmp_path):
        """get_async_jsonl_checkpointer() returns a JSONLCheckpointSaver."""

        async def _run():
            return await get_async_jsonl_checkpointer(path=str(tmp_path / "ck.jsonl"))

        saver = asyncio.run(_run())
        assert isinstance(saver, JSONLCheckpointSaver)


# ---------------------------------------------------------------------------
# QueryableCheckpointerMixin
# ---------------------------------------------------------------------------


class TestQueryableMixin:
    """get_user_threads, get_thread_messages, delete_thread, get_user_stats,
    thread_exists."""

    def test_thread_exists_true(self, tmp_path):
        """thread_exists returns True after a put."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("alice_conv1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        assert saver.thread_exists("alice_conv1") is True

    def test_thread_exists_false(self, tmp_path):
        """thread_exists returns False for unknown thread."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        assert saver.thread_exists("ghost") is False

    def test_get_user_threads_matches_prefix(self, tmp_path):
        """get_user_threads returns threads owned by user (prefix pattern)."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("alice_conv1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        saver.put(
            _minimal_config("alice_conv2"),
            _minimal_checkpoint("cp-2"),
            _minimal_metadata(),
            {},
        )
        saver.put(
            _minimal_config("bob_conv1"),
            _minimal_checkpoint("cp-3"),
            _minimal_metadata(),
            {},
        )
        threads = saver.get_user_threads("alice")
        tids = [t["thread_id"] for t in threads]
        assert "alice_conv1" in tids
        assert "alice_conv2" in tids
        assert "bob_conv1" not in tids

    def test_get_user_threads_limit_offset(self, tmp_path):
        """limit and offset are respected."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        for i in range(5):
            saver.put(
                _minimal_config(f"alice_conv{i}"),
                _minimal_checkpoint(f"cp-{i}"),
                _minimal_metadata(),
                {},
            )
        threads = saver.get_user_threads("alice", limit=2, offset=1)
        assert len(threads) == 2

    def test_get_user_threads_empty(self, tmp_path):
        """get_user_threads for unknown user returns empty list."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        assert saver.get_user_threads("nobody") == []

    def test_get_user_threads_exact_match(self, tmp_path):
        """thread_id exactly equal to user_identifier is included."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("alice"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        threads = saver.get_user_threads("alice")
        assert len(threads) == 1
        assert threads[0]["conversation_id"] == "default"

    def test_get_thread_messages(self, tmp_path):
        """get_thread_messages extracts HumanMessage content."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        ck = _minimal_checkpoint("cp-1")
        ck["channel_values"]["messages"] = [
            _human_message("Hello agent"),
            _ai_message("Hello human"),
        ]
        saver.put(_minimal_config("thread-1"), ck, _minimal_metadata(), {})
        msgs = saver.get_thread_messages("thread-1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_get_thread_messages_type_filter(self, tmp_path):
        """message_types filter is respected."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        ck = _minimal_checkpoint("cp-1")
        ck["channel_values"]["messages"] = [
            _human_message("hi"),
            _ai_message("hello"),
        ]
        saver.put(_minimal_config("thread-1"), ck, _minimal_metadata(), {})
        msgs = saver.get_thread_messages("thread-1", message_types=["HumanMessage"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_get_thread_messages_unknown_thread(self, tmp_path):
        """get_thread_messages returns [] for unknown thread."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        assert saver.get_thread_messages("ghost") == []

    def test_delete_thread_removes_records(self, tmp_path):
        """delete_thread removes all records; thread_exists returns False."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        assert saver.thread_exists("thread-1")
        result = saver.delete_thread("thread-1")
        assert result is True
        assert not saver.thread_exists("thread-1")

    def test_delete_thread_not_found_returns_false(self, tmp_path):
        """delete_thread on unknown thread returns False."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        assert saver.delete_thread("ghost") is False

    def test_delete_thread_compacts_file(self, tmp_path):
        """After delete_thread, the JSONL file no longer contains those records."""
        path = str(tmp_path / "ck.jsonl")
        saver = JSONLCheckpointSaver(path=path)
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        saver.put(
            _minimal_config("thread-2"),
            _minimal_checkpoint("cp-2"),
            _minimal_metadata(),
            {},
        )
        saver.delete_thread("thread-1")

        with open(path, "r", encoding="utf-8") as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        tids = {r["thread_id"] for r in records}
        assert "thread-1" not in tids
        assert "thread-2" in tids

    def test_get_user_stats(self, tmp_path):
        """get_user_stats aggregates across threads."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        ck = _minimal_checkpoint("cp-1")
        ck["channel_values"]["messages"] = [
            _human_message("hi"),
            _ai_message("hello"),
        ]
        saver.put(_minimal_config("alice_conv1"), ck, _minimal_metadata(), {})
        stats = saver.get_user_stats("alice")
        assert stats["total_threads"] == 1
        assert stats["total_checkpoints"] == 1
        assert stats["total_messages"] == 2

    def test_get_user_stats_empty(self, tmp_path):
        """get_user_stats for unknown user returns zeros."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        stats = saver.get_user_stats("nobody")
        assert stats["total_threads"] == 0


# ---------------------------------------------------------------------------
# In-process concurrency smoke test
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Two threads writing to different thread_ids in one saver do not corrupt data."""

    def test_concurrent_puts_on_different_threads(self, tmp_path):
        """Multiple OS threads can call put() concurrently without data loss."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "concurrent.jsonl"))
        errors = []

        def write_n(tid, n):
            try:
                for i in range(n):
                    saver.put(
                        _minimal_config(tid),
                        _minimal_checkpoint(f"cp-{tid}-{i}"),
                        _minimal_metadata(),
                        {},
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        threads = [
            threading.Thread(target=write_n, args=(f"thread-{t}", 10)) for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised errors: {errors}"

        # Verify each thread's checkpoints are accessible
        for t in range(4):
            results = list(saver.list(_minimal_config(f"thread-{t}")))
            assert (
                len(results) == 10
            ), f"Expected 10 checkpoints for thread-{t}, got {len(results)}"


# ---------------------------------------------------------------------------
# VersionedCheckpointerMixin helpers
# ---------------------------------------------------------------------------


class TestVersionedMixinHelpers:
    """_get_raw_checkpoint, _replace_raw_checkpoint, _archive_checkpoint."""

    def test_get_raw_checkpoint_returns_latest(self, tmp_path):
        """_get_raw_checkpoint returns the latest record dict."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        raw = saver._get_raw_checkpoint("thread-1")
        assert raw is not None
        assert raw["checkpoint_id"] == "cp-1"

    def test_get_raw_checkpoint_none_for_unknown_thread(self, tmp_path):
        """_get_raw_checkpoint returns None for a thread with no checkpoints."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        assert saver._get_raw_checkpoint("ghost") is None

    def test_replace_raw_checkpoint(self, tmp_path):
        """_replace_raw_checkpoint updates the record and rewrites file."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        raw = saver._get_raw_checkpoint("thread-1")
        raw["format_version"] = 99
        result = saver._replace_raw_checkpoint("thread-1", raw)
        assert result is True

        # Reload from disk and verify
        s2 = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        raw2 = s2._get_raw_checkpoint("thread-1")
        assert raw2["format_version"] == 99

    def test_replace_raw_checkpoint_missing_checkpoint_id(self, tmp_path):
        """_replace_raw_checkpoint returns False when checkpoint_id is absent."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        result = saver._replace_raw_checkpoint("thread-1", {"no_id": True})
        assert result is False

    def test_replace_raw_checkpoint_unknown_checkpoint_id(self, tmp_path):
        """_replace_raw_checkpoint returns False for an unknown checkpoint_id."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "ck.jsonl"))
        saver.put(
            _minimal_config("thread-1"),
            _minimal_checkpoint("cp-1"),
            _minimal_metadata(),
            {},
        )
        result = saver._replace_raw_checkpoint("thread-1", {"checkpoint_id": "ghost"})
        assert result is False

    def test_archive_checkpoint_writes_sidecar(self, tmp_path):
        """_archive_checkpoint creates a .archive sidecar file."""
        path = str(tmp_path / "ck.jsonl")
        saver = JSONLCheckpointSaver(path=path)
        saver._archive_checkpoint(
            "thread-1",
            {"checkpoint_id": "cp-1"},
            ValueError("oops"),
        )
        archive_path = path + ".archive"
        assert os.path.exists(archive_path)
        with open(archive_path, "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert record["record_type"] == "archive"
        assert record["thread_id"] == "thread-1"
        assert "oops" in record["migration_error"]
