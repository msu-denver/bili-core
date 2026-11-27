"""Real-PostgreSQL tests for thread deletion and orphaned blob cleanup.

These run graphs through PruningPostgresSaver against a live database and
read the tables back with SQL, so they catch rows the mocked tests cannot
see. They are skipped unless ``PG_INTEGRATION_TEST_DSN`` names a database the
tests may write to and clean up, for example::

    docker run -d --name pg-test -e POSTGRES_PASSWORD=test \\
        -p 127.0.0.1:55432:5432 postgres:16
    PG_INTEGRATION_TEST_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres \\
        pytest bili/iris/checkpointers/tests/test_pg_checkpointer_real_db.py

Point it at a disposable database: ``delete_orphaned_blobs`` removes every
orphaned blob row in that database, not only the rows these tests create.
"""

import os
import uuid
from typing import Annotated, TypedDict

import pytest

DSN = os.environ.get("PG_INTEGRATION_TEST_DSN")

pytestmark = pytest.mark.skipif(
    not DSN,
    reason="Set PG_INTEGRATION_TEST_DSN to run real PostgreSQL checkpointer tests",
)

TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


@pytest.fixture(name="pool")
def _pool():
    """Open a connection pool configured like the library's own pool."""
    from psycopg_pool import ConnectionPool  # pylint: disable=import-outside-toplevel

    pool = ConnectionPool(
        conninfo=DSN, min_size=1, max_size=4, kwargs={"autocommit": True}
    )
    yield pool
    pool.close()


def _saver(pool, user_id=None):
    """Build a set-up PruningPostgresSaver over the pool."""
    from bili.iris.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
        PruningPostgresSaver,
    )

    saver = PruningPostgresSaver(pool, keep_last_n=5, user_id=user_id)
    saver.setup()
    saver.ensure_indexes()
    return saver


def _chat(saver, thread_id, turns):
    """Run one graph invocation per turn so every table gets rows."""
    # pylint: disable=import-outside-toplevel
    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class ChatState(TypedDict, total=False):
        """Graph state whose messages channel is stored as a blob."""

        messages: Annotated[list[AnyMessage], add_messages]

    def reply(state):
        return {"messages": [AIMessage(f"echo: {state['messages'][-1].content}")]}

    builder = StateGraph(ChatState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=saver)
    for turn in range(turns):
        graph.invoke(
            {"messages": [HumanMessage(f"private note {turn}")]},
            {"configurable": {"thread_id": thread_id}},
        )
    return graph


def _counts(pool, thread_id):
    """Return the row count per per-thread table for one thread."""
    with pool.connection() as conn:
        return {
            table: conn.execute(
                f"SELECT count(*) FROM {table} WHERE thread_id = %s",  # nosec B608
                (thread_id,),
            ).fetchone()[0]
            for table in TABLES
        }


@pytest.fixture(name="tag")
def _tag(pool):
    """Give each test its own thread ids and remove their rows afterwards."""
    tag = uuid.uuid4().hex[:8]
    yield tag
    with pool.connection() as conn:
        for table in TABLES:
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id LIKE %s",  # nosec B608
                (f"%{tag}%",),
            )


def test_delete_removes_every_table_and_spares_other_threads(pool, tag):
    """After delete_thread no row of the thread remains in any table."""
    saver = _saver(pool)
    target, other = f"alice_{tag}-1", f"bob_{tag}-1"
    _chat(saver, target, 3)
    _chat(saver, other, 2)
    other_before = _counts(pool, other)
    assert all(_counts(pool, target).values())

    assert saver.delete_thread(target) is True

    assert _counts(pool, target) == {table: 0 for table in TABLES}
    assert _counts(pool, other) == other_before
    assert saver.delete_thread(target) is False


def test_delete_refuses_another_users_thread_before_deleting(pool, tag):
    """A user-scoped saver raises and leaves the other user's rows intact."""
    victim = f"bob_{tag}-1"
    _chat(_saver(pool), victim, 2)
    before = _counts(pool, victim)

    with pytest.raises(PermissionError):
        _saver(pool, user_id=f"alice_{tag}").delete_thread(victim)

    assert _counts(pool, victim) == before


def test_scoped_saver_deletes_its_own_thread(pool, tag):
    """The user_id write path and delete path agree on the same thread."""
    owner = f"alice{tag}"
    saver = _saver(pool, user_id=owner)
    thread_id = f"{owner}_conv"
    _chat(saver, thread_id, 2)

    assert saver.delete_thread(thread_id) is True
    assert _counts(pool, thread_id) == {table: 0 for table in TABLES}


def test_cleanup_removes_orphans_and_keeps_live_threads(pool, tag):
    """Blobs left by a checkpoints-and-writes-only delete are removed."""
    saver = _saver(pool)
    orphaned, live = f"alice_{tag}-old", f"alice_{tag}-live"
    _chat(saver, orphaned, 2)
    graph = _chat(saver, live, 2)
    live_before = _counts(pool, live)
    with pool.connection() as conn:
        # The delete this cleanup exists for: blobs were left behind.
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (orphaned,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (orphaned,))
    orphan_rows = _counts(pool, orphaned)["checkpoint_blobs"]
    assert orphan_rows > 0

    assert saver.delete_orphaned_blobs(dry_run=True) >= orphan_rows
    assert _counts(pool, orphaned)["checkpoint_blobs"] == orphan_rows

    assert saver.delete_orphaned_blobs() >= orphan_rows
    assert _counts(pool, orphaned)["checkpoint_blobs"] == 0
    assert _counts(pool, live) == live_before
    state = graph.get_state({"configurable": {"thread_id": live}})
    assert len(state.values["messages"]) == 4
    assert saver.delete_orphaned_blobs() == 0


def test_cleanup_skips_blob_rows_of_an_uncommitted_write(pool, tag):
    """A blob row whose checkpoint row is not yet committed survives."""
    import psycopg  # pylint: disable=import-outside-toplevel

    saver = _saver(pool)
    thread_id = f"carol_{tag}-inflight"
    with psycopg.connect(DSN) as writer:
        writer.execute(
            "INSERT INTO checkpoint_blobs "
            "(thread_id, checkpoint_ns, channel, version, type, blob) "
            "VALUES (%s, '', 'messages', '1.0', 'msgpack', '\\x00')",
            (thread_id,),
        )
        saver.delete_orphaned_blobs()
        writer.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
            "VALUES (%s, '', 'cp-1', '{}')",
            (thread_id,),
        )
        writer.commit()

    assert _counts(pool, thread_id)["checkpoint_blobs"] == 1
