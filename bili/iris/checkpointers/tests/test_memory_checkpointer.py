"""Tests for bili.iris.checkpointers.memory_checkpointer module.

Covers the full QueryableMemorySaver query interface including
thread management, ownership validation, multi-tenant isolation,
and put/get checkpoint round-trips.
"""

from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from bili.iris.checkpointers.memory_checkpointer import QueryableMemorySaver


def _make_checkpoint(messages=None, checkpoint_id="cp-1"):
    """Create a minimal checkpoint dict with optional messages."""
    channel_values = {}
    if messages is not None:
        channel_values["messages"] = messages
    return {
        "id": checkpoint_id,
        "channel_values": channel_values,
        "channel_versions": {},
        "versions_seen": {},
        "v": 1,
    }


class _ChatState(TypedDict, total=False):
    """Graph state with the channels the query methods read."""

    messages: Annotated[list[AnyMessage], add_messages]
    title: str
    tags: list[str]


def _reply(state: _ChatState) -> dict:
    """Answer the newest human message; AI text is set per test via the input."""
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else "multimodal"
    return {
        "messages": [AIMessage(content=f"echo: {text}")],
        "title": f"About {state['messages'][0].content!s:.20}",
        "tags": ["memory", "test"],
    }


def _build_graph(saver, node=_reply):
    """Compile a one-node graph that checkpoints through the given saver."""
    builder = StateGraph(_ChatState)
    builder.add_node("reply", node)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


def _chat(saver, thread_id, *turns, node=_reply):
    """Run one graph invocation per human turn on the thread."""
    graph = _build_graph(saver, node)
    config = {"configurable": {"thread_id": thread_id}}
    for turn in turns:
        message = turn if isinstance(turn, HumanMessage) else HumanMessage(turn)
        graph.invoke({"messages": [message]}, config)
    return graph


def _config(thread_id, checkpoint_id=None, checkpoint_ns=""):
    """Build a RunnableConfig for the saver's real put/get API."""
    configurable = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Verify QueryableMemorySaver initialization."""

    def test_default_user_id_is_none(self):
        """Verify user_id defaults to None."""
        saver = QueryableMemorySaver()
        assert saver.user_id is None

    def test_custom_user_id(self):
        """Verify user_id can be set on init."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        assert saver.user_id == "alice@test.com"


# ---------------------------------------------------------------------------
# Thread ownership validation
# ---------------------------------------------------------------------------


class TestValidateThreadOwnership:
    """Verify thread ownership validation through the public API.

    Ownership validation is exercised via get_thread_messages,
    which calls _validate_thread_ownership internally.
    """

    def test_no_user_id_skips_validation(self):
        """Verify validation is skipped when user_id is None."""
        saver = QueryableMemorySaver()
        # Should not raise — no user_id means no ownership check
        result = saver.get_thread_messages("any_thread_id")
        assert result == []

    def test_matching_user_id_passes(self):
        """Verify exact match thread_id passes validation."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        # Should not raise — thread owned by user
        result = saver.get_thread_messages("alice@test.com")
        assert result == []

    def test_prefixed_thread_id_passes(self):
        """Verify thread_id with user prefix passes."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        # Should not raise — thread prefixed with user_id
        result = saver.get_thread_messages("alice@test.com_conv-123")
        assert result == []

    def test_wrong_user_raises_permission_error(self):
        """Verify wrong user thread_id raises PermissionError."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        with pytest.raises(PermissionError, match="Access denied"):
            saver.get_thread_messages("bob@test.com_conv-1")

    def test_different_user_prefix_raises_permission_error(self):
        """Verify a different user's thread raises PermissionError."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        with pytest.raises(PermissionError):
            saver.get_thread_messages("alicex@test.com_conv-1")


# ---------------------------------------------------------------------------
# put / get_tuple round-trip
# ---------------------------------------------------------------------------


class TestPutGetRoundTrip:
    """Verify checkpoint put and get_tuple operations."""

    def test_put_and_get_tuple(self):
        """Verify a checkpoint can be stored and retrieved."""
        saver = QueryableMemorySaver()
        config = {
            "configurable": {
                "thread_id": "user1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        checkpoint = _make_checkpoint(checkpoint_id="cp-1")
        metadata = {"source": "input"}
        saver.put(config, checkpoint, metadata, {})

        result = saver.get_tuple(config)
        assert result is not None

    def test_put_validates_ownership(self):
        """Verify put raises PermissionError for wrong user."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        config = {
            "configurable": {
                "thread_id": "bob@test.com",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        with pytest.raises(PermissionError):
            saver.put(config, {}, {}, {})

    def test_get_tuple_validates_ownership(self):
        """Verify get_tuple raises PermissionError for wrong user."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        config = {
            "configurable": {
                "thread_id": "bob@test.com",
                "checkpoint_ns": "",
            }
        }
        with pytest.raises(PermissionError):
            saver.get_tuple(config)


# ---------------------------------------------------------------------------
# get_user_threads
# ---------------------------------------------------------------------------


class TestGetUserThreads:
    """Verify get_user_threads over threads written by a real graph run."""

    def test_returns_empty_for_unknown_user(self):
        """Verify empty list for user with no threads."""
        saver = QueryableMemorySaver()
        result = saver.get_user_threads("nobody@test.com")
        assert result == []

    def test_summarizes_a_real_conversation(self):
        """Verify counts, messages, title, tags and timestamp from real turns."""
        saver = QueryableMemorySaver()
        graph = _chat(saver, "user@test.com_conv-1", "First question", "Second one")
        latest = graph.get_state(
            {"configurable": {"thread_id": "user@test.com_conv-1"}}
        )
        stored = list(
            saver.list({"configurable": {"thread_id": "user@test.com_conv-1"}})
        )

        threads = saver.get_user_threads("user@test.com")

        assert threads == [
            {
                "thread_id": "user@test.com_conv-1",
                "conversation_id": "conv-1",
                "last_updated": latest.created_at,
                "checkpoint_count": len(stored),
                "message_count": 4,
                "first_message": "First question",
                "last_message": "Second one",
                "title": "About First question",
                "tags": ["memory", "test"],
            }
        ]
        # Two invocations of a one-node graph store input, loop and end
        # checkpoints for each turn.
        assert len(stored) == 6

    def test_orders_threads_newest_first(self):
        """Verify the thread written last is listed first."""
        saver = QueryableMemorySaver()
        _chat(saver, "user@test.com_older", "a")
        _chat(saver, "user@test.com_newer", "b")
        _chat(saver, "user@test.com_newest", "c")

        ids = [t["thread_id"] for t in saver.get_user_threads("user@test.com")]
        assert ids == [
            "user@test.com_newest",
            "user@test.com_newer",
            "user@test.com_older",
        ]

    def test_pagination_with_limit_and_offset(self):
        """Verify limit and offset slice the newest-first list."""
        saver = QueryableMemorySaver()
        for i in range(5):
            _chat(saver, f"user@test.com_conv-{i}", f"q{i}")

        page = saver.get_user_threads("user@test.com", limit=2, offset=1)
        assert [t["thread_id"] for t in page] == [
            "user@test.com_conv-3",
            "user@test.com_conv-2",
        ]
        assert len(saver.get_user_threads("user@test.com", offset=2)) == 3

    def test_default_conversation_id_for_exact_match(self):
        """Verify conversation_id is 'default' for exact user match."""
        saver = QueryableMemorySaver()
        _chat(saver, "user@test.com", "hello")

        threads = saver.get_user_threads("user@test.com")
        assert threads[0]["conversation_id"] == "default"

    def test_skips_placeholder_left_by_read_of_unsaved_thread(self):
        """Verify an upstream read of a never-saved thread is not a thread."""
        saver = QueryableMemorySaver()
        _chat(saver, "user@test.com_real", "hello")
        assert saver.get_tuple(_config("user@test.com_ghost")) is None
        assert "user@test.com_ghost" in saver.storage

        threads = saver.get_user_threads("user@test.com")
        assert [t["thread_id"] for t in threads] == ["user@test.com_real"]

    def test_listing_leaves_no_placeholder(self):
        """Verify get_user_threads adds no storage entries of its own."""
        saver = QueryableMemorySaver()
        saver.put(
            _config("user@test.com_child-only", "cp-1", checkpoint_ns="child"),
            _make_checkpoint(checkpoint_id="cp-1"),
            {"source": "input"},
            {},
        )
        before = {tid: dict(ns) for tid, ns in saver.storage.items()}
        saver.get_user_threads("user@test.com")
        assert {tid: dict(ns) for tid, ns in saver.storage.items()} == before

    def test_subgraph_only_thread_is_listed_without_messages(self):
        """Verify a thread saved only under a child namespace is still listed."""
        saver = QueryableMemorySaver()
        saver.put(
            _config("user@test.com_child-only", "cp-1", checkpoint_ns="child"),
            {**_make_checkpoint(checkpoint_id="cp-1"), "ts": "2026-01-01T00:00:00"},
            {"source": "input"},
            {},
        )
        threads = saver.get_user_threads("user@test.com")
        assert len(threads) == 1
        assert threads[0]["checkpoint_count"] == 1
        assert threads[0]["last_updated"] == "2026-01-01T00:00:00"
        assert threads[0]["message_count"] == 0
        assert threads[0]["title"] is None


# ---------------------------------------------------------------------------
# Multi-tenant isolation
# ---------------------------------------------------------------------------


class TestMultiTenantIsolation:
    """Verify threads are isolated between users."""

    def test_user_only_sees_own_threads(self):
        """Verify user A cannot see user B threads."""
        saver = QueryableMemorySaver()
        _chat(saver, "alice@test.com_conv-1", "from alice")
        _chat(saver, "bob@test.com_conv-1", "from bob")
        _chat(saver, "alicex@test.com_conv-1", "prefix lookalike")

        alice_threads = saver.get_user_threads("alice@test.com")
        bob_threads = saver.get_user_threads("bob@test.com")

        assert [t["thread_id"] for t in alice_threads] == ["alice@test.com_conv-1"]
        assert alice_threads[0]["first_message"] == "from alice"
        assert [t["thread_id"] for t in bob_threads] == ["bob@test.com_conv-1"]

    def test_ownership_enforced_on_messages(self):
        """Verify get_thread_messages enforces ownership."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        with pytest.raises(PermissionError):
            saver.get_thread_messages("bob@test.com_conv-1")


# ---------------------------------------------------------------------------
# get_thread_messages
# ---------------------------------------------------------------------------


class TestGetThreadMessages:
    """Verify get_thread_messages over threads written by a real graph run."""

    def test_returns_empty_for_missing_thread(self):
        """Verify empty list for nonexistent thread."""
        saver = QueryableMemorySaver()
        result = saver.get_thread_messages("nonexistent")
        assert result == []

    def test_missing_thread_read_leaves_no_placeholder(self):
        """Verify reading a never-saved thread adds nothing to storage."""
        saver = QueryableMemorySaver()
        assert saver.get_thread_messages("ghost") == []
        assert "ghost" not in saver.storage

    def test_returns_messages_in_order(self):
        """Verify every turn comes back in order with roles and content."""
        saver = QueryableMemorySaver()
        _chat(saver, "user1_conv-1", "Hello", "Again")

        result = saver.get_thread_messages("user1_conv-1")
        assert result == [
            {"role": "user", "content": "Hello", "timestamp": None},
            {"role": "assistant", "content": "echo: Hello", "timestamp": None},
            {"role": "user", "content": "Again", "timestamp": None},
            {"role": "assistant", "content": "echo: Again", "timestamp": None},
        ]

    def test_message_type_filter(self):
        """Verify message_types filter works correctly."""
        saver = QueryableMemorySaver()
        _chat(saver, "user1_conv-1", "Q1", "Q2")

        result = saver.get_thread_messages(
            "user1_conv-1", message_types=["HumanMessage"]
        )
        assert [m["content"] for m in result] == ["Q1", "Q2"]
        assert all(m["role"] == "user" for m in result)

    def test_pagination_limit_and_offset(self):
        """Verify pagination on messages works."""
        saver = QueryableMemorySaver()
        _chat(saver, "user1_conv-1", *[f"Q{i}" for i in range(5)])

        result = saver.get_thread_messages("user1_conv-1", limit=3, offset=2)
        assert [m["content"] for m in result] == ["Q1", "echo: Q1", "Q2"]

    def test_strips_thinking_blocks_from_ai_messages(self):
        """Verify thinking blocks are stripped from AI content."""

        def thinking_reply(_state):
            return {
                "messages": [
                    AIMessage(
                        content="<thinking>internal reasoning</thinking>Visible answer"
                    )
                ]
            }

        saver = QueryableMemorySaver()
        _chat(saver, "user1_conv-1", "Q1", node=thinking_reply)

        result = saver.get_thread_messages("user1_conv-1")
        assert result[1]["content"] == "Visible answer"

    def test_multimodal_content_handling(self):
        """Verify list-based multimodal content extraction."""
        saver = QueryableMemorySaver()
        multimodal = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://img.png"}},
                {"type": "text", "text": "world"},
            ]
        )
        _chat(saver, "user1_conv-1", multimodal)

        result = saver.get_thread_messages("user1_conv-1")
        assert result[0]["content"] == "Hello world"
        assert result[1]["content"] == "echo: multimodal"


# ---------------------------------------------------------------------------
# delete_thread
# ---------------------------------------------------------------------------


class TestDeleteThread:
    """Verify delete_thread behavior."""

    def test_delete_nonexistent_returns_false(self):
        """Verify deleting a missing thread returns False."""
        saver = QueryableMemorySaver()
        result = saver.delete_thread("nonexistent")
        assert result is False

    def test_delete_validates_ownership(self):
        """Verify delete_thread enforces ownership."""
        saver = QueryableMemorySaver(user_id="alice@test.com")
        with pytest.raises(PermissionError):
            saver.delete_thread("bob@test.com_conv-1")

    def test_put_then_exists_and_delete_removes_thread(self):
        """Verify put(), thread_exists(), and delete_thread() round-trip."""
        saver = QueryableMemorySaver()
        config = {
            "configurable": {
                "thread_id": "user1_conv-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        checkpoint = _make_checkpoint(messages=["hello"], checkpoint_id="cp-1")
        checkpoint["channel_versions"] = {"messages": 1}
        saved_config = saver.put(
            config, checkpoint, {"source": "input"}, {"messages": 1}
        )
        saver.put_writes(saved_config, [("messages", "pending")], task_id="task-1")

        assert saver.thread_exists("user1_conv-1") is True
        assert saver.delete_thread("user1_conv-1") is True
        assert saver.thread_exists("user1_conv-1") is False
        assert "user1_conv-1" not in saver.storage
        assert not any(key[0] == "user1_conv-1" for key in saver.writes)
        assert not any(key[0] == "user1_conv-1" for key in saver.blobs)

    def test_delete_after_graph_run_empties_every_store(self):
        """A real conversation leaves no checkpoint, write or blob behind."""
        saver = QueryableMemorySaver()
        _chat(saver, "alice_conv-1", "private one", "private two")
        _chat(saver, "bob_conv-1", "keep me")
        assert any(key[0] == "alice_conv-1" for key in saver.blobs)
        assert any(key[0] == "alice_conv-1" for key in saver.writes)

        assert saver.delete_thread("alice_conv-1") is True

        assert "alice_conv-1" not in saver.storage
        assert not any(key[0] == "alice_conv-1" for key in saver.writes)
        assert not any(key[0] == "alice_conv-1" for key in saver.blobs)
        assert saver.get_thread_messages("alice_conv-1") == []
        assert [m["content"] for m in saver.get_thread_messages("bob_conv-1")] == [
            "keep me",
            "echo: keep me",
        ]

    def test_never_saved_thread_returns_false_for_exists_and_delete(self):
        """Verify thread_exists and delete_thread return False when unsaved."""
        saver = QueryableMemorySaver()
        assert saver.thread_exists("never_saved") is False
        assert saver.delete_thread("never_saved") is False

    def test_read_of_unsaved_thread_then_delete_returns_false(self):
        """Verify delete_thread() is False after only a read of the thread."""
        saver = QueryableMemorySaver()
        assert saver.get_tuple(_config("ghost")) is None
        assert saver.delete_thread("ghost") is False

    def test_delete_removes_placeholder_left_by_read(self):
        """Verify placeholders from reads do not accumulate across deletes."""
        saver = QueryableMemorySaver()
        for thread_id in ("ghost-1", "ghost-2", "ghost-3"):
            saver.get_tuple(_config(thread_id))
            assert saver.delete_thread(thread_id) is False
        assert not saver.storage

    def test_delete_thread_with_only_pending_writes_returns_true(self):
        """Verify True when the thread had pending writes but no checkpoint."""
        saver = QueryableMemorySaver()
        saver.put_writes(
            _config("writes-only", "cp-1"), [("messages", "w")], task_id="task-1"
        )
        assert saver.thread_exists("writes-only") is False
        assert saver.delete_thread("writes-only") is True
        assert not any(key[0] == "writes-only" for key in saver.writes)

    def test_delete_thread_with_only_blobs_returns_true(self):
        """Verify True when the thread had channel blobs but no checkpoint."""
        saver = QueryableMemorySaver()
        saver.blobs[("blobs-only", "", "messages", 1)] = ("json", b"[]")
        assert saver.delete_thread("blobs-only") is True
        assert not any(key[0] == "blobs-only" for key in saver.blobs)

    def test_delete_validates_ownership_before_reporting_existence(self):
        """Verify another user's thread raises even when it has a checkpoint."""
        saver = QueryableMemorySaver()
        saver.put(
            _config("bob@test.com_conv-1", "cp-1"),
            _make_checkpoint(checkpoint_id="cp-1"),
            {"source": "input"},
            {},
        )
        saver.user_id = "alice@test.com"
        with pytest.raises(PermissionError):
            saver.delete_thread("bob@test.com_conv-1")
        assert saver.thread_exists("bob@test.com_conv-1") is True

    def test_delete_leaves_other_thread_with_same_first_char_intact(self):
        """Verify deleting one thread preserves another sharing its first char."""
        saver = QueryableMemorySaver()
        config_one = {
            "configurable": {
                "thread_id": "alpha-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        config_two = {
            "configurable": {
                "thread_id": "alpha-2",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-2",
            }
        }
        cp_one = _make_checkpoint(messages=["one"], checkpoint_id="cp-1")
        cp_one["channel_versions"] = {"messages": 1}
        cp_two = _make_checkpoint(messages=["two"], checkpoint_id="cp-2")
        cp_two["channel_versions"] = {"messages": 1}

        saved_one = saver.put(config_one, cp_one, {"source": "input"}, {"messages": 1})
        saved_two = saver.put(config_two, cp_two, {"source": "input"}, {"messages": 1})
        saver.put_writes(saved_one, [("messages", "w1")], task_id="task-1")
        saver.put_writes(saved_two, [("messages", "w2")], task_id="task-2")

        assert saver.delete_thread("alpha-1") is True
        assert saver.thread_exists("alpha-1") is False
        assert "alpha-1" not in saver.storage
        assert not any(key[0] == "alpha-1" for key in saver.writes)
        assert not any(key[0] == "alpha-1" for key in saver.blobs)

        assert saver.thread_exists("alpha-2") is True
        assert "alpha-2" in saver.storage
        assert any(key[0] == "alpha-2" for key in saver.writes)
        assert any(key[0] == "alpha-2" for key in saver.blobs)
        assert saver.get_tuple(saved_two) is not None

    def test_delete_single_char_thread_leaves_same_first_char_thread_intact(self):
        """Verify deleting a one-character id does not delete longer ids."""
        saver = QueryableMemorySaver()
        config_short = {
            "configurable": {
                "thread_id": "a",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        config_long = {
            "configurable": {
                "thread_id": "alpha",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-2",
            }
        }
        cp_short = _make_checkpoint(checkpoint_id="cp-1")
        cp_long = _make_checkpoint(checkpoint_id="cp-2")

        saver.put(config_short, cp_short, {"source": "input"}, {})
        saved_long = saver.put(config_long, cp_long, {"source": "input"}, {})

        assert saver.delete_thread("a") is True
        assert saver.thread_exists("a") is False
        assert "a" not in saver.storage
        assert saver.thread_exists("alpha") is True
        assert "alpha" in saver.storage
        assert saver.get_tuple(saved_long) is not None


# ---------------------------------------------------------------------------
# get_user_stats
# ---------------------------------------------------------------------------


class TestGetUserStats:
    """Verify get_user_stats aggregation."""

    def test_empty_stats_for_unknown_user(self):
        """Verify zero stats for user with no threads."""
        saver = QueryableMemorySaver()
        stats = saver.get_user_stats("nobody@test.com")
        assert stats["total_threads"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_checkpoints"] == 0
        assert stats["oldest_thread"] is None
        assert stats["newest_thread"] is None

    def test_stats_aggregate_correctly(self):
        """Verify stats sum across real threads and span their timestamps."""
        saver = QueryableMemorySaver()
        first = _chat(saver, "user@test.com_conv-1", "Q1")
        second = _chat(saver, "user@test.com_conv-2", "Q2", "Q3")

        stats = saver.get_user_stats("user@test.com")
        first_ts = first.get_state(
            {"configurable": {"thread_id": "user@test.com_conv-1"}}
        ).created_at
        second_ts = second.get_state(
            {"configurable": {"thread_id": "user@test.com_conv-2"}}
        ).created_at
        assert stats == {
            "total_threads": 2,
            "total_messages": 6,
            "total_checkpoints": 9,
            "oldest_thread": first_ts,
            "newest_thread": second_ts,
        }

    def test_stats_ignore_placeholder_threads(self):
        """Verify reads of never-saved threads add no threads to the stats."""
        saver = QueryableMemorySaver()
        for thread_id in ("user@test.com_ghost-1", "user@test.com_ghost-2"):
            saver.get_tuple(_config(thread_id))
            list(saver.list(_config(thread_id)))

        stats = saver.get_user_stats("user@test.com")
        assert stats["total_threads"] == 0
        assert stats["total_checkpoints"] == 0


# ---------------------------------------------------------------------------
# thread_exists
# ---------------------------------------------------------------------------


class TestThreadExists:
    """Verify thread_exists checks."""

    def test_nonexistent_thread_returns_false(self):
        """Verify False for missing thread."""
        saver = QueryableMemorySaver()
        assert saver.thread_exists("nonexistent") is False

    def test_existing_thread_via_put(self):
        """Verify True for a thread with a checkpoint saved through put()."""
        saver = QueryableMemorySaver()
        saver.put(
            _config("user1", "cp-1"),
            _make_checkpoint(checkpoint_id="cp-1"),
            {"source": "input"},
            {},
        )
        assert saver.thread_exists("user1") is True

    def test_checkpoint_in_non_default_namespace_counts(self):
        """Verify a checkpoint stored only under a subgraph namespace counts."""
        saver = QueryableMemorySaver()
        saver.put(
            _config("user1", "cp-1", checkpoint_ns="child"),
            _make_checkpoint(checkpoint_id="cp-1"),
            {"source": "input"},
            {},
        )
        assert saver.thread_exists("user1") is True

    def test_read_of_unsaved_thread_does_not_make_it_exist(self):
        """Verify get_tuple() and list() on an unsaved thread leave it absent."""
        saver = QueryableMemorySaver()
        assert saver.get_tuple(_config("ghost")) is None
        assert not list(saver.list(_config("ghost")))
        # The upstream read leaves an empty placeholder; it must not count.
        assert "ghost" in saver.storage
        assert saver.thread_exists("ghost") is False

    def test_lookup_does_not_create_storage_entry(self):
        """Verify thread_exists() itself adds nothing to storage."""
        saver = QueryableMemorySaver()
        assert saver.thread_exists("ghost") is False
        assert "ghost" not in saver.storage
