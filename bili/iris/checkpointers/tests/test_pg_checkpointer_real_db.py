"""Real-PostgreSQL tests for thread deletion and orphaned blob cleanup.

These run graphs through PruningPostgresSaver against a live database and
read the tables back with SQL, so they catch rows the mocked tests cannot
see. They are skipped unless ``PG_INTEGRATION_TEST_DSN`` names a database,
for example::

    docker run -d --name pg-test -e POSTGRES_PASSWORD=test \\
        -p 127.0.0.1:55432:5432 postgres:16
    PG_INTEGRATION_TEST_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres \\
        pytest bili/iris/checkpointers/tests/test_pg_checkpointer_real_db.py

Every test runs in a temporary schema it creates and drops, reached through
``search_path``, so the migrations, the ALTER TABLE and the database-wide
cleanup touch nothing outside it. The tests also refuse a DSN whose host is
not local (localhost, 127.0.0.1, ::1 or a unix socket) unless
``PG_INTEGRATION_TEST_ALLOW_REMOTE=1`` is set.
"""

import os
import threading
import time
import uuid
from typing import Annotated, TypedDict

import pytest

DSN = os.environ.get("PG_INTEGRATION_TEST_DSN")
ALLOW_REMOTE = os.environ.get("PG_INTEGRATION_TEST_ALLOW_REMOTE") == "1"

TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Advisory lock key the hold-open trigger waits on.
_HOLD_KEY = 3570357


def _refusal(dsn, allow_remote):
    """Return why the tests must not run against ``dsn``, or None."""
    from psycopg.conninfo import (  # pylint: disable=import-outside-toplevel
        conninfo_to_dict,
    )

    if not dsn:
        return "Set PG_INTEGRATION_TEST_DSN to run real PostgreSQL checkpointer tests"
    if allow_remote:
        return None
    params = conninfo_to_dict(dsn)
    hosts = params.get("host") or params.get("hostaddr") or os.environ.get("PGHOST")
    # No host means libpq's default unix socket.
    for host in (hosts or "").split(","):
        host = host.strip()
        if host and host not in _LOCAL_HOSTS and not host.startswith("/"):
            return (
                f"Refusing non-local PostgreSQL host {host!r}; set "
                "PG_INTEGRATION_TEST_ALLOW_REMOTE=1 to allow it"
            )
    return None


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@localhost:5432/db",
        "postgresql://u:p@127.0.0.1/db",
        "postgresql://u:p@[::1]:5432/db",
        "host=/var/run/postgresql dbname=db",
        "dbname=db",
        "host=localhost,127.0.0.1 dbname=db",
    ],
)
def test_local_dsns_are_accepted(dsn, monkeypatch):
    """Loopback hosts and unix sockets may run the tests."""
    monkeypatch.delenv("PGHOST", raising=False)
    assert _refusal(dsn, allow_remote=False) is None


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@db.example.com:5432/db",
        "postgresql://u:p@10.0.0.5/db",
        "host=localhost,db.example.com dbname=db",
    ],
)
def test_remote_dsns_are_refused_unless_allowed(dsn, monkeypatch):
    """A remote host is refused, and allowed only by the explicit variable."""
    monkeypatch.delenv("PGHOST", raising=False)
    assert "Refusing non-local" in _refusal(dsn, allow_remote=False)
    assert _refusal(dsn, allow_remote=True) is None


def test_remote_pghost_is_refused(monkeypatch):
    """A DSN without a host falls back to PGHOST, which is checked too."""
    monkeypatch.setenv("PGHOST", "db.example.com")
    assert "Refusing non-local" in _refusal("dbname=db", allow_remote=False)


def test_missing_dsn_is_refused():
    """Without a DSN the database tests do not run."""
    assert "PG_INTEGRATION_TEST_DSN" in _refusal(None, allow_remote=False)


@pytest.fixture(name="schema")
def _schema():
    """Create a uniquely named schema for one test and drop it afterwards."""
    reason = _refusal(DSN, ALLOW_REMOTE)
    if reason:
        pytest.skip(reason)
    import psycopg  # pylint: disable=import-outside-toplevel

    name = f"bili_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{name}"')
    yield name
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.fixture(name="pool")
def _pool(schema):
    """Open a pool configured like the library's own, inside the schema."""
    from psycopg_pool import ConnectionPool  # pylint: disable=import-outside-toplevel

    pool = ConnectionPool(
        conninfo=DSN,
        min_size=1,
        max_size=6,
        kwargs={"autocommit": True, "options": f"-c search_path={schema}"},
    )
    yield pool
    pool.close()


def _saver(pool, user_id=None, keep_last_n=5):
    """Build a set-up PruningPostgresSaver over the pool."""
    from bili.iris.checkpointers.pg_checkpointer import (  # pylint: disable=import-outside-toplevel
        PruningPostgresSaver,
    )

    PruningPostgresSaver(pool, keep_last_n=keep_last_n).setup()
    saver = PruningPostgresSaver(pool, keep_last_n=keep_last_n, user_id=user_id)
    saver.ensure_indexes()
    return saver


def _graph(saver, subgraph=False):
    """Compile a reply graph; with ``subgraph`` the reply runs in a child graph."""
    # pylint: disable=import-outside-toplevel
    from langchain_core.messages import AIMessage, AnyMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class ChatState(TypedDict, total=False):
        """Graph state whose messages channel is stored as a blob."""

        messages: Annotated[list[AnyMessage], add_messages]

    def reply(state):
        return {"messages": [AIMessage(f"echo: {state['messages'][-1].content}")]}

    def build(node):
        builder = StateGraph(ChatState)
        builder.add_node("reply", node)
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        return builder

    if subgraph:
        return build(build(reply).compile()).compile(checkpointer=saver)
    return build(reply).compile(checkpointer=saver)


def _chat(graph, thread_id, *turns):
    """Run one graph invocation per human turn on the thread."""
    from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
        HumanMessage,
    )

    for turn in turns:
        graph.invoke(
            {"messages": [HumanMessage(turn)]},
            {"configurable": {"thread_id": thread_id}},
        )


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


def _blob_keys(pool, thread_id):
    """Return every (namespace, channel, version) blob key of a thread."""
    with pool.connection() as conn:
        return set(
            conn.execute(
                "SELECT checkpoint_ns, channel, version FROM checkpoint_blobs "
                "WHERE thread_id = %s",
                (thread_id,),
            ).fetchall()
        )


def _unreferenced(pool):
    """Count blob rows no checkpoint references, computed with ->> lookups.

    Written independently of the library's predicate, so the cleanup's
    count is checked against a second formulation.
    """
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT count(*) FROM checkpoint_blobs b WHERE NOT EXISTS (
                SELECT 1 FROM checkpoints c
                WHERE c.thread_id = b.thread_id
                AND c.checkpoint_ns = b.checkpoint_ns
                AND c.checkpoint -> 'channel_versions' ->> b.channel = b.version
            )
            """
        ).fetchone()[0]


def _contains(pool, thread_id, text):
    """Count a thread's blob rows whose serialized bytes contain ``text``."""
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT blob FROM checkpoint_blobs WHERE thread_id = %s",
            (thread_id,),
        ).fetchall()
    return sum(1 for (blob,) in rows if blob and text.encode() in bytes(blob))


def _old_style_delete(pool, thread_id):
    """Delete a thread the way earlier releases did: blobs were left behind."""
    with pool.connection() as conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))


def _snapshot(saver, thread_id, checkpoint_ns=""):
    """Every checkpoint tuple of a thread and namespace, plus its messages."""
    with saver.conn.connection() as conn:
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = %s "
                "AND checkpoint_ns = %s ORDER BY checkpoint_id",
                (thread_id, checkpoint_ns),
            ).fetchall()
        ]
    tuples = [
        saver.get_tuple(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
        )
        for checkpoint_id in ids
    ]
    messages = saver.get_thread_messages(thread_id) if checkpoint_ns == "" else None
    return ids, tuples, messages


def _report(label, **counts):
    """Print row counts so a run with -s records them."""
    print(f"ROWCOUNTS {label}: {counts}")


def test_saver_tables_and_migrations_live_in_the_test_schema(pool, schema):
    """Upstream setup and bili's user_id migration both honor search_path."""
    _saver(pool, user_id="alice")
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n "
            "ON n.oid = c.relnamespace WHERE n.nspname = %s AND c.relkind = 'r'",
            (schema,),
        ).fetchall()
        user_id_column = conn.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
            "AND table_name = 'checkpoints' AND column_name = 'user_id'",
            (schema,),
        ).fetchone()
        user_index = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s "
            "AND indexname = 'idx_checkpoints_user_thread'",
            (schema,),
        ).fetchone()
    assert set(TABLES) <= {name for (name,) in rows}
    assert user_id_column is not None
    assert user_index is not None


def test_user_id_migration_ignores_a_checkpoints_table_in_another_schema(pool, schema):
    """A user_id column elsewhere does not stop the column being added here."""
    import psycopg  # pylint: disable=import-outside-toplevel

    decoy = f"{schema}_decoy"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{decoy}"')
        conn.execute(
            f'CREATE TABLE "{decoy}".checkpoints (thread_id TEXT, user_id TEXT)'
        )
        conn.execute(
            f'CREATE INDEX idx_checkpoints_user_thread ON "{decoy}".checkpoints '
            "(user_id, thread_id)"
        )
    from psycopg_pool import ConnectionPool  # pylint: disable=import-outside-toplevel

    try:
        with ConnectionPool(
            conninfo=DSN,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "options": f"-c search_path={schema},{decoy}"},
        ) as both:
            _saver(both, user_id="alice")
        with pool.connection() as conn:
            column = conn.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'checkpoints' AND column_name = 'user_id'",
                (schema,),
            ).fetchone()
        assert column is not None
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{decoy}" CASCADE')


def test_delete_removes_every_table_and_spares_other_threads(pool):
    """After delete_thread no row of the thread remains in any table."""
    saver = _saver(pool)
    graph = _graph(saver)
    _chat(graph, "alice_conv-1", "one", "two", "three")
    _chat(graph, "bob_conv-1", "one", "two")
    other_before = _counts(pool, "bob_conv-1")
    assert all(_counts(pool, "alice_conv-1").values())

    assert saver.delete_thread("alice_conv-1") is True

    assert _counts(pool, "alice_conv-1") == {table: 0 for table in TABLES}
    assert _counts(pool, "bob_conv-1") == other_before
    assert saver.delete_thread("alice_conv-1") is False


def test_delete_of_blobs_alone_reports_true(pool):
    """Leftover blobs with no checkpoint still count as something removed."""
    saver = _saver(pool)
    _chat(_graph(saver), "alice_conv-1", "one")
    _old_style_delete(pool, "alice_conv-1")
    assert _counts(pool, "alice_conv-1")["checkpoint_blobs"] > 0

    assert saver.delete_thread("alice_conv-1") is True
    assert saver.delete_thread("alice_conv-1") is False


def test_delete_refuses_another_users_thread_before_deleting(pool):
    """A user-scoped saver raises and leaves the other user's rows intact."""
    _chat(_graph(_saver(pool)), "bob_conv-1", "one", "two")
    before = _counts(pool, "bob_conv-1")

    with pytest.raises(PermissionError):
        _saver(pool, user_id="alice").delete_thread("bob_conv-1")

    assert _counts(pool, "bob_conv-1") == before


def test_scoped_saver_deletes_its_own_thread(pool):
    """The user_id write path and delete path agree on the same thread."""
    saver = _saver(pool, user_id="alice")
    _chat(_graph(saver), "alice_conv", "one", "two")

    assert saver.delete_thread("alice_conv") is True
    assert _counts(pool, "alice_conv") == {table: 0 for table in TABLES}


def test_a_reused_thread_loses_the_old_conversation_and_keeps_the_new(pool):
    """(a) Old blobs on a reused thread id go; the new conversation is unchanged."""
    saver = _saver(pool)
    graph = _graph(saver)
    thread_id = "alice@example.com"
    _chat(graph, thread_id, "OLD secret one", "OLD secret two")
    _old_style_delete(pool, thread_id)
    old_keys = _blob_keys(pool, thread_id)
    _chat(graph, thread_id, "new one", "new two")
    before = _snapshot(saver, thread_id)
    old_rows = len(old_keys)
    # Every old blob is unreferenced; keep-last-N pruning of the new
    # conversation can add leftovers of its own.
    orphans = _unreferenced(pool)
    assert orphans >= old_rows > 0
    assert _contains(pool, thread_id, "OLD secret") > 0

    # The thread was just written, so the default idle window skips it.
    assert saver.delete_orphaned_blobs(dry_run=True) == 0
    assert saver.delete_orphaned_blobs() == 0
    assert _blob_keys(pool, thread_id) >= old_keys

    assert saver.delete_orphaned_blobs(dry_run=True, min_idle_seconds=0) == orphans
    assert _blob_keys(pool, thread_id) >= old_keys
    deleted = saver.delete_orphaned_blobs(min_idle_seconds=0)

    assert deleted == orphans
    assert _unreferenced(pool) == 0
    assert not _blob_keys(pool, thread_id) & old_keys
    assert _contains(pool, thread_id, "OLD secret") == 0
    after = _snapshot(saver, thread_id)
    assert after == before
    assert [m["content"] for m in after[2]] == [
        "new one",
        "echo: new one",
        "new two",
        "echo: new two",
    ]
    assert saver.delete_orphaned_blobs(min_idle_seconds=0) == 0
    _report(
        "a",
        old_conversation_blob_rows=old_rows,
        pruning_leftovers=orphans - old_rows,
        deleted=deleted,
        second_run=0,
    )


def test_the_idle_window_admits_a_thread_once_it_is_old_enough(pool):
    """A thread whose newest checkpoint is older than the window is cleaned."""
    saver = _saver(pool)
    thread_id = "alice@example.com"
    graph = _graph(saver)
    _chat(graph, thread_id, "OLD secret")
    _old_style_delete(pool, thread_id)
    _chat(graph, thread_id, "new")
    orphans = _unreferenced(pool)
    with pool.connection() as conn:
        conn.execute(
            "UPDATE checkpoints SET checkpoint = jsonb_set(checkpoint, '{ts}', "
            "to_jsonb((now() - interval '2 hours')::text)) WHERE thread_id = %s",
            (thread_id,),
        )

    assert saver.delete_orphaned_blobs(min_idle_seconds=7200 + 60) == 0
    assert saver.delete_orphaned_blobs() == orphans
    assert _contains(pool, thread_id, "OLD secret") == 0


def test_a_pruned_thread_loads_identically_after_the_cleanup(pool):
    """(b) Pruning leftovers go; every remaining checkpoint loads as before."""
    saver = _saver(pool, keep_last_n=2)
    _chat(_graph(saver), "alice_conv-1", "one", "two", "three", "four", "five")
    before = _snapshot(saver, "alice_conv-1")
    leftovers = _unreferenced(pool)
    assert len(before[0]) == 2
    assert leftovers > 0

    deleted = saver.delete_orphaned_blobs(min_idle_seconds=0)

    assert deleted == leftovers
    assert _snapshot(saver, "alice_conv-1") == before
    assert len(before[2]) == 10
    assert _unreferenced(pool) == 0
    assert saver.delete_orphaned_blobs(min_idle_seconds=0) == 0
    _report(
        "b",
        remaining_checkpoints=len(before[0]),
        blob_rows_after=_counts(pool, "alice_conv-1")["checkpoint_blobs"],
        pruning_leftovers_deleted=deleted,
    )


def test_a_subgraph_namespace_on_a_live_thread_is_intact(pool):
    """(c) Blobs of a child namespace survive while another thread's go."""
    saver = _saver(pool, keep_last_n=-1)
    graph = _graph(saver, subgraph=True)
    _chat(graph, "alice_conv-1", "one", "two")
    _chat(graph, "bob_conv-1", "gone")
    _old_style_delete(pool, "bob_conv-1")
    with pool.connection() as conn:
        child_namespaces = [
            ns
            for (ns,) in conn.execute(
                "SELECT DISTINCT checkpoint_ns FROM checkpoint_blobs "
                "WHERE thread_id = 'alice_conv-1' AND checkpoint_ns <> ''"
            ).fetchall()
        ]
    assert child_namespaces
    snapshots = {ns: _snapshot(saver, "alice_conv-1", ns) for ns in child_namespaces}
    root = _snapshot(saver, "alice_conv-1")
    live_keys = _blob_keys(pool, "alice_conv-1")
    bob_rows = _counts(pool, "bob_conv-1")["checkpoint_blobs"]

    deleted = saver.delete_orphaned_blobs(min_idle_seconds=0)

    assert deleted == bob_rows
    assert _blob_keys(pool, "alice_conv-1") == live_keys
    assert {ns: _snapshot(saver, "alice_conv-1", ns) for ns in child_namespaces} == (
        snapshots
    )
    assert _snapshot(saver, "alice_conv-1") == root
    assert saver.delete_orphaned_blobs(min_idle_seconds=0) == 0
    child_rows = sum(1 for key in live_keys if key[0] != "")
    _report(
        "c",
        child_namespaces=len(child_namespaces),
        child_blob_rows_kept=child_rows,
        deleted_other_thread=deleted,
    )


def test_batched_cleanup_deletes_the_same_rows_in_bounded_chunks(pool):
    """(d) Batches of 3 delete every orphan; a rerun deletes 0."""
    saver = _saver(pool)
    graph = _graph(saver)
    for thread_id in ("a_1", "a_2", "a_3"):
        _chat(graph, thread_id, "x", "y")
        _old_style_delete(pool, thread_id)
    orphans = _unreferenced(pool)
    assert orphans > 3

    assert saver.delete_orphaned_blobs(dry_run=True, min_idle_seconds=0) == orphans
    assert saver.delete_orphaned_blobs(batch_size=3, min_idle_seconds=0) == orphans
    assert _unreferenced(pool) == 0
    assert saver.delete_orphaned_blobs(batch_size=3, min_idle_seconds=0) == 0
    _report("d", orphans=orphans, second_run=0)


@pytest.fixture(name="held_insert")
def _held_insert(pool):
    """Make every checkpoint INSERT wait on an advisory lock this test holds.

    The trigger fires after the put's blob rows are written in the same
    transaction, so the write stays open, uncommitted, until release().
    """
    # pylint: disable=no-member
    import psycopg  # pylint: disable=import-outside-toplevel

    _saver(pool)
    with pool.connection() as conn:
        conn.execute(
            f"""
            CREATE FUNCTION hold_checkpoint_insert() RETURNS trigger AS $$
            BEGIN
                PERFORM pg_advisory_lock_shared({_HOLD_KEY});
                PERFORM pg_advisory_unlock_shared({_HOLD_KEY});
                RETURN NEW;
            END $$ LANGUAGE plpgsql
            """
        )
        conn.execute(
            "CREATE TRIGGER hold_checkpoint_insert BEFORE INSERT ON checkpoints "
            "FOR EACH ROW EXECUTE FUNCTION hold_checkpoint_insert()"
        )
    holder = psycopg.connect(DSN, autocommit=True)
    holder.execute("SELECT pg_advisory_lock(%s)", (_HOLD_KEY,))
    released = []

    def release():
        if not released:
            holder.execute("SELECT pg_advisory_unlock(%s)", (_HOLD_KEY,))
            released.append(True)

    def waiting_writer_pid(timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = holder.execute(
                "SELECT pid FROM pg_locks WHERE locktype = 'advisory' "
                "AND objid = %s AND NOT granted",
                (_HOLD_KEY,),
            ).fetchone()
            if row:
                return row[0]
            time.sleep(0.05)
        raise AssertionError("the writer never reached the held checkpoint insert")

    def holds_blob_insert(pid):
        return (
            holder.execute(
                "SELECT count(*) FROM pg_locks l JOIN pg_class c ON c.oid = l.relation "
                "WHERE l.pid = %s AND c.relname = 'checkpoint_blobs' "
                "AND l.mode = 'RowExclusiveLock'",
                (pid,),
            ).fetchone()[0]
            > 0
        )

    yield release, waiting_writer_pid, holds_blob_insert
    release()
    holder.close()


@pytest.mark.parametrize("user_id", [None, "alice"], ids=["default", "user_scoped"])
def test_a_write_in_progress_keeps_its_blobs(pool, held_insert, user_id):
    """(e) Blobs of a put held open before its checkpoint row survive the cleanup.

    The put runs through bili's real write path (the upstream pipeline on
    the default path, _put_with_user_id when scoped) on the same saver the
    cleanup is called on.
    """
    release, waiting_writer_pid, holds_blob_insert = held_insert
    saver = _saver(pool, user_id=user_id)
    graph = _graph(saver)
    thread_id = "alice_conv-1"
    _old_style_delete(pool, thread_id)
    visible_before = _blob_keys(pool, thread_id)
    errors = []

    def write():
        try:
            _chat(graph, thread_id, "private in flight")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        pid = waiting_writer_pid()
        # The blob insert ran in the writer's still-open transaction...
        assert holds_blob_insert(pid)
        # ...and none of its rows are visible outside it.
        assert _blob_keys(pool, thread_id) == visible_before

        results = []
        cleanup = threading.Thread(
            target=lambda: results.extend(
                [
                    saver.delete_orphaned_blobs(dry_run=True, min_idle_seconds=0),
                    saver.delete_orphaned_blobs(min_idle_seconds=0),
                    saver.delete_orphaned_blobs(batch_size=1, min_idle_seconds=0),
                ]
            ),
            daemon=True,
        )
        cleanup.start()
        cleanup.join(timeout=15)
        assert not cleanup.is_alive(), "the cleanup blocked on the open write"
    finally:
        release()
        writer.join(timeout=30)

    assert not writer.is_alive()
    assert not errors, errors
    assert results == [0, 0, 0]
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert [m.content for m in state.values["messages"]] == [
        "private in flight",
        "echo: private in flight",
    ]
    assert _unreferenced(pool) == 0
    _report(
        f"e[{user_id or 'default'}]",
        cleanup_results=results,
        blob_rows_after_commit=_counts(pool, thread_id)["checkpoint_blobs"],
    )
