"""Real-PostgreSQL tests for thread deletion.

These run graphs through PruningPostgresSaver against a live database and
read the tables back with SQL, so they catch rows the mocked tests cannot
see. They are skipped unless ``PG_INTEGRATION_TEST_DSN`` names a database,
for example::

    docker run -d --name pg-test -e POSTGRES_PASSWORD=test \\
        -p 127.0.0.1:55432:5432 postgres:16
    PG_INTEGRATION_TEST_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres \\
        pytest bili/iris/checkpointers/tests/test_pg_checkpointer_real_db.py

Every test runs in a temporary schema it creates and drops, reached through
``search_path``, so the migrations, the ALTER TABLE and the deletes touch
nothing outside it. The tests also refuse a target that is not local
(localhost, 127.0.0.1, ::1 or a unix socket) unless
``PG_INTEGRATION_TEST_ALLOW_REMOTE=1`` is set.
"""

import os
import uuid
from typing import Annotated, TypedDict

import pytest

DSN = os.environ.get("PG_INTEGRATION_TEST_DSN")
ALLOW_REMOTE = os.environ.get("PG_INTEGRATION_TEST_ALLOW_REMOTE") == "1"

TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


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
    # libpq connects to hostaddr when it is set, uses host for the name, and
    # fills either one from PGHOST or PGHOSTADDR when the DSN omits it. Every
    # value present must be local, so no combination can reach a remote
    # server. No value at all means libpq's default unix socket.
    values = (
        params.get("host"),
        params.get("hostaddr"),
        os.environ.get("PGHOST"),
        os.environ.get("PGHOSTADDR"),
    )
    for value in values:
        for host in (value or "").split(","):
            host = host.strip()
            if host and host not in _LOCAL_HOSTS and not host.startswith("/"):
                return (
                    f"Refusing non-local PostgreSQL host {host!r}; set "
                    "PG_INTEGRATION_TEST_ALLOW_REMOTE=1 to allow it"
                )
    return None


@pytest.fixture(name="no_pg_host_env")
def _no_pg_host_env(monkeypatch):
    """Clear the libpq host variables so each guard case sets its own."""
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGHOSTADDR", raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@localhost:5432/db",
        "postgresql://u:p@127.0.0.1/db",
        "postgresql://u:p@[::1]:5432/db",
        "host=/var/run/postgresql dbname=db",
        "dbname=db",
        "host=localhost,127.0.0.1 dbname=db",
        "host=localhost hostaddr=127.0.0.1 dbname=db",
        "postgresql://u:p@localhost/db?hostaddr=::1",
    ],
)
@pytest.mark.usefixtures("no_pg_host_env")
def test_local_dsns_are_accepted(dsn):
    """Loopback hosts and unix sockets may run the tests."""
    assert _refusal(dsn, allow_remote=False) is None


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@db.example.com:5432/db",
        "postgresql://u:p@10.0.0.5/db",
        "host=localhost,db.example.com dbname=db",
        "host=localhost hostaddr=10.0.0.5 dbname=db",
        "postgresql://u:p@localhost/db?hostaddr=10.0.0.5",
        "hostaddr=10.0.0.5 dbname=db",
    ],
)
@pytest.mark.usefixtures("no_pg_host_env")
def test_remote_dsns_are_refused_unless_allowed(dsn):
    """A remote host is refused, and allowed only by the explicit variable."""
    assert "Refusing non-local" in _refusal(dsn, allow_remote=False)
    assert _refusal(dsn, allow_remote=True) is None


@pytest.mark.parametrize(
    "dsn, env",
    [
        ("dbname=db", {"PGHOST": "db.example.com"}),
        ("dbname=db", {"PGHOSTADDR": "10.0.0.5"}),
        ("dbname=db", {"PGHOST": "localhost", "PGHOSTADDR": "10.0.0.5"}),
        ("host=localhost dbname=db", {"PGHOSTADDR": "10.0.0.5"}),
        ("host=/var/run/postgresql dbname=db", {"PGHOST": "db.example.com"}),
    ],
)
def test_a_remote_value_in_the_environment_is_refused(dsn, env, no_pg_host_env):
    """PGHOST and PGHOSTADDR are checked whether or not the DSN names a host."""
    for name, value in env.items():
        no_pg_host_env.setenv(name, value)
    assert "Refusing non-local" in _refusal(dsn, allow_remote=False)
    assert _refusal(dsn, allow_remote=True) is None


@pytest.mark.parametrize(
    "dsn, env",
    [
        ("dbname=db", {"PGHOST": "/tmp", "PGHOSTADDR": "127.0.0.1"}),
        ("host=localhost dbname=db", {"PGHOSTADDR": "::1"}),
        ("hostaddr=127.0.0.1 dbname=db", {"PGHOST": "localhost"}),
    ],
)
def test_local_values_in_the_environment_are_accepted(dsn, env, no_pg_host_env):
    """Local PGHOST and PGHOSTADDR values beside a local DSN are allowed."""
    for name, value in env.items():
        no_pg_host_env.setenv(name, value)
    assert _refusal(dsn, allow_remote=False) is None


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


def _old_style_delete(pool, thread_id):
    """Delete a thread the way earlier releases did: blobs were left behind."""
    with pool.connection() as conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))


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
