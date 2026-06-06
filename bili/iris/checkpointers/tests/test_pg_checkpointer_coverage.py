"""Additional coverage tests for pg_checkpointer.

Covers the module-level pool factory functions, the sync atomic
user_id put path, async saver lifecycle (pool, indexes, schema, aput,
pruning), the sync migration helpers used by the async saver, and a
handful of sync query branches not exercised elsewhere. All PostgreSQL
interactions are mocked, no real database is used.
"""

# pylint: disable=protected-access,import-outside-toplevel

import threading
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio


# =========================================================================
# Helpers
# =========================================================================


def _make_sync_saver(keep_last_n=5, user_id=None):
    """Create a PruningPostgresSaver with mocked PG internals."""
    with patch(
        "bili.iris.checkpointers.pg_checkpointer"
        ".PruningPostgresSaver._ensure_user_id_schema"
    ), patch(
        "bili.iris.checkpointers.pg_checkpointer.PruningPostgresSaver.setup"
    ), patch(
        "bili.iris.checkpointers.pg_checkpointer.PruningPostgresSaver.ensure_indexes"
    ), patch(
        "langgraph.checkpoint.postgres.PostgresSaver.__init__",
        return_value=None,
    ):
        from bili.iris.checkpointers.pg_checkpointer import PruningPostgresSaver

        saver = PruningPostgresSaver(
            MagicMock(), keep_last_n=keep_last_n, user_id=user_id
        )
        saver.lock = threading.RLock()
        saver.conn = MagicMock()
        saver._txn_conn = None
        return saver


def _make_async_saver(keep_last_n=5, user_id=None):
    """Create an AsyncPruningPostgresSaver with mocked PG internals."""
    with patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.__init__",
        return_value=None,
    ):
        from bili.iris.checkpointers.pg_checkpointer import AsyncPruningPostgresSaver

        saver = AsyncPruningPostgresSaver(
            MagicMock(), keep_last_n=keep_last_n, user_id=user_id
        )
        saver.conn = MagicMock()
        saver._async_txn_conn = None
        saver._sync_pool = None
        return saver


def _attach_sync_cursor(saver, mock_cur):
    """Attach a fake sync _cursor context manager."""

    @contextmanager
    def fake_cursor(_pipeline=False):
        yield mock_cur

    object.__setattr__(saver, "_cursor", fake_cursor)


def _attach_async_cursor(saver, mock_cur):
    """Attach a fake async _cursor context manager."""

    @asynccontextmanager
    async def fake_cursor(_pipeline=False):
        yield mock_cur

    object.__setattr__(saver, "_cursor", fake_cursor)


# =========================================================================
# Module-level pool factories
# =========================================================================


class TestGetPgConnectionPool:
    """Tests for get_pg_connection_pool and close_pg_connection_pool."""

    @patch("bili.iris.checkpointers.pg_checkpointer.atexit.register")
    @patch("bili.iris.checkpointers.pg_checkpointer.ConnectionPool")
    @patch.dict(
        "os.environ",
        {
            "POSTGRES_CONNECTION_STRING": "postgresql://host:5432",
            "POSTGRES_CONNECTION_POOL_MIN_SIZE": "2",
            "POSTGRES_CONNECTION_POOL_MAX_SIZE": "7",
        },
        clear=True,
    )
    def test_creates_pool_with_sizes_and_langgraph_db(self, mock_pool_cls, mock_atexit):
        """A set connection string builds a pool against the langgraph database."""
        from bili.iris.checkpointers.pg_checkpointer import (
            close_pg_connection_pool,
            get_pg_connection_pool,
        )

        fake_pool = MagicMock()
        mock_pool_cls.return_value = fake_pool

        pool = get_pg_connection_pool()
        assert pool is fake_pool
        _, kwargs = mock_pool_cls.call_args
        assert kwargs["conninfo"] == "postgresql://host:5432/langgraph"
        assert kwargs["min_size"] == 2
        assert kwargs["max_size"] == 7
        assert kwargs["kwargs"] == {"autocommit": True}
        mock_atexit.assert_called_once_with(close_pg_connection_pool)

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_without_connection_string(self):
        """No connection string yields None."""
        from bili.iris.checkpointers.pg_checkpointer import get_pg_connection_pool

        assert get_pg_connection_pool() is None

    @patch("bili.iris.checkpointers.pg_checkpointer.get_pg_connection_pool")
    def test_close_closes_active_pool(self, mock_get_pool):
        """close_pg_connection_pool closes an existing pool."""
        from bili.iris.checkpointers.pg_checkpointer import close_pg_connection_pool

        fake_pool = MagicMock()
        mock_get_pool.return_value = fake_pool
        close_pg_connection_pool()
        fake_pool.close.assert_called_once_with()

    @patch("bili.iris.checkpointers.pg_checkpointer.get_pg_connection_pool")
    def test_close_noop_when_no_pool(self, mock_get_pool):
        """close_pg_connection_pool is a no-op when there is no pool."""
        from bili.iris.checkpointers.pg_checkpointer import close_pg_connection_pool

        mock_get_pool.return_value = None
        # Should not raise.
        assert close_pg_connection_pool() is None


# =========================================================================
# get_pg_checkpointer factory
# =========================================================================


class TestGetPgCheckpointerFactory:
    """Tests for the sync checkpointer factory wiring."""

    @patch("bili.iris.checkpointers.pg_checkpointer.get_pg_connection_pool")
    def test_returns_none_without_pool(self, mock_get_pool):
        """No pool yields None."""
        from bili.iris.checkpointers.pg_checkpointer import get_pg_checkpointer

        mock_get_pool.return_value = None
        assert get_pg_checkpointer() is None


# =========================================================================
# Sync atomic user_id put path
# =========================================================================


class TestPutWithUserId:
    """Tests for PruningPostgresSaver._put_with_user_id atomic transaction."""

    def test_atomic_put_sets_user_id_and_commits(self):
        """The atomic path runs super().put, updates user_id, and commits."""
        from psycopg_pool import ConnectionPool

        saver = _make_sync_saver(user_id="alice@example.com", keep_last_n=-1)

        # conn is a real-ish mock; isinstance check uses ConnectionPool.
        conn = MagicMock()
        cur = MagicMock()
        cur.rowcount = 1
        conn.cursor.return_value.__enter__.return_value = cur
        pool = MagicMock(spec=ConnectionPool)
        pool.connection.return_value.__enter__.return_value = conn
        pool.connection.return_value.__exit__.return_value = False
        saver.conn = pool

        config = {"configurable": {"thread_id": "alice@example.com_c1"}}
        next_cfg = {
            "configurable": {
                "thread_id": "alice@example.com_c1",
                "checkpoint_id": "cp1",
            }
        }
        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=next_cfg,
        ):
            result = saver._put_with_user_id(
                config, {"v": 1}, {"source": "loop"}, {}, "alice@example.com_c1"
            )

        assert result is next_cfg
        conn.commit.assert_called_once_with()
        # The UPDATE was issued with user_id, thread_id, checkpoint_id.
        update_args = cur.execute.call_args[0]
        assert "UPDATE checkpoints SET user_id" in update_args[0]
        assert update_args[1] == ("alice@example.com", "alice@example.com_c1", "cp1")

    def test_atomic_put_rolls_back_on_missing_row(self):
        """A zero-row user_id update raises and triggers rollback."""
        from psycopg_pool import ConnectionPool

        saver = _make_sync_saver(user_id="alice@example.com", keep_last_n=-1)
        conn = MagicMock()
        cur = MagicMock()
        cur.rowcount = 0
        conn.cursor.return_value.__enter__.return_value = cur
        pool = MagicMock(spec=ConnectionPool)
        pool.connection.return_value.__enter__.return_value = conn
        pool.connection.return_value.__exit__.return_value = False
        saver.conn = pool

        config = {"configurable": {"thread_id": "alice@example.com_c1"}}
        next_cfg = {
            "configurable": {
                "thread_id": "alice@example.com_c1",
                "checkpoint_id": "cp1",
            }
        }
        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver.put",
            return_value=next_cfg,
        ):
            with pytest.raises(RuntimeError, match="Failed to set user_id"):
                saver._put_with_user_id(
                    config, {"v": 1}, {}, {}, "alice@example.com_c1"
                )
        conn.rollback.assert_called_once_with()

    def test_put_routes_to_atomic_path_when_user_id_set(self):
        """put() delegates to _put_with_user_id when user_id is configured."""
        saver = _make_sync_saver(user_id="alice@example.com", keep_last_n=-1)
        config = {"configurable": {"thread_id": "alice@example.com_c1"}}
        next_cfg = {"configurable": {"thread_id": "alice@example.com_c1"}}
        with patch.object(
            saver, "_put_with_user_id", return_value=next_cfg
        ) as mock_atomic:
            result = saver.put(config, {"v": 1}, {"source": "loop"}, {})
        assert result is next_cfg
        mock_atomic.assert_called_once()
        # Versioned metadata passed includes format_version.
        passed_meta = mock_atomic.call_args[0][2]
        assert passed_meta["format_version"] == saver.format_version


# =========================================================================
# Sync get_tuple delegation
# =========================================================================


class TestSyncGetTupleDelegates:
    """Tests for PruningPostgresSaver.get_tuple delegating to parent."""

    def test_get_tuple_calls_super_after_migration(self):
        """get_tuple validates ownership, migrates, then calls super()."""
        saver = _make_sync_saver()
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        sentinel = object()
        with patch.object(
            saver, "migrate_checkpoint_if_needed", return_value=False
        ), patch(
            "langgraph.checkpoint.postgres.PostgresSaver.get_tuple",
            return_value=sentinel,
        ):
            assert saver.get_tuple(config) is sentinel


# =========================================================================
# Async pool manager and factories
# =========================================================================


class TestAsyncConnectionManagerPool:
    """Tests for AsyncConnectionManager.get_pool."""

    @patch("bili.iris.checkpointers.pg_checkpointer.atexit.register")
    @patch("bili.iris.checkpointers.pg_checkpointer.AsyncConnectionPool")
    @patch.dict(
        "os.environ",
        {
            "POSTGRES_CONNECTION_STRING": "postgresql://host:5432",
            "POSTGRES_CONNECTION_POOL_MIN_SIZE": "3",
            "POSTGRES_CONNECTION_POOL_MAX_SIZE": "9",
        },
        clear=True,
    )
    async def test_get_pool_builds_async_pool(self, mock_pool_cls, mock_atexit):
        """A set connection string builds an AsyncConnectionPool once."""
        from bili.iris.checkpointers.pg_checkpointer import AsyncConnectionManager

        fake_pool = MagicMock()
        mock_pool_cls.return_value = fake_pool
        manager = AsyncConnectionManager()

        pool = await manager.get_pool()
        assert pool is fake_pool
        _, kwargs = mock_pool_cls.call_args
        assert kwargs["conninfo"] == "postgresql://host:5432/langgraph"
        assert kwargs["min_size"] == 3
        assert kwargs["max_size"] == 9
        mock_atexit.assert_called_once()

        # Second call returns the cached pool without rebuilding.
        again = await manager.get_pool()
        assert again is fake_pool
        assert mock_pool_cls.call_count == 1

    @patch.dict("os.environ", {}, clear=True)
    async def test_get_pool_returns_none_without_env(self):
        """No connection string yields None."""
        from bili.iris.checkpointers.pg_checkpointer import AsyncConnectionManager

        manager = AsyncConnectionManager()
        assert await manager.get_pool() is None

    def test_close_pool_clears_reference(self):
        """_close_pool clears the cached pool."""
        from bili.iris.checkpointers.pg_checkpointer import AsyncConnectionManager

        manager = AsyncConnectionManager()
        manager._pool = MagicMock()
        manager._close_pool()
        assert manager._pool is None

    async def test_get_async_pg_connection_pool_delegates(self):
        """get_async_pg_connection_pool delegates to the module manager."""
        from bili.iris.checkpointers import pg_checkpointer

        with patch.object(
            pg_checkpointer._async_pool_manager,
            "get_pool",
            new_callable=AsyncMock,
            return_value="POOL",
        ):
            assert await pg_checkpointer.get_async_pg_connection_pool() == "POOL"

    async def test_get_async_pg_checkpointer_returns_none_without_pool(self):
        """get_async_pg_checkpointer returns None when no pool exists."""
        from bili.iris.checkpointers import pg_checkpointer

        with patch.object(
            pg_checkpointer,
            "get_async_pg_connection_pool",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await pg_checkpointer.get_async_pg_checkpointer() is None

    async def test_get_async_pg_checkpointer_success_path(self):
        """With a live pool, the factory sets up the saver and returns it.

        Regression test for the asetup/setup bug: get_async_pg_checkpointer
        must call the AsyncPostgresSaver coroutine ``setup`` (not the
        non-existent ``asetup``), then create indexes, then return the saver.
        """
        from bili.iris.checkpointers import pg_checkpointer

        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()
        mock_saver.aensure_indexes = AsyncMock()

        with patch.object(
            pg_checkpointer,
            "get_async_pg_connection_pool",
            new_callable=AsyncMock,
            return_value=MagicMock(),  # a non-None pool
        ), patch.object(
            pg_checkpointer,
            "AsyncPruningPostgresSaver",
            return_value=mock_saver,
        ):
            result = await pg_checkpointer.get_async_pg_checkpointer(
                keep_last_n=7, user_id="u@example.com"
            )

        assert result is mock_saver
        mock_saver.setup.assert_awaited_once()
        mock_saver.aensure_indexes.assert_awaited_once()


# =========================================================================
# Async index / schema migration
# =========================================================================


class TestAsyncIndexesAndSchema:
    """Tests for aensure_indexes and _aensure_user_id_schema."""

    async def test_aensure_indexes_creates_three_indexes(self):
        """aensure_indexes issues three CREATE INDEX statements."""
        saver = _make_async_saver(user_id=None)
        cur = AsyncMock()
        _attach_async_cursor(saver, cur)
        await saver.aensure_indexes()
        assert cur.execute.await_count == 3
        statements = " ".join(call.args[0] for call in cur.execute.await_args_list)
        assert "idx_checkpoints_thread_id" in statements
        assert "idx_blobs_thread_id" in statements
        assert "idx_writes_thread_id" in statements

    async def test_aensure_indexes_triggers_user_id_schema(self):
        """When user_id is set, aensure_indexes runs the user_id schema migration."""
        saver = _make_async_saver(user_id="alice")
        cur = AsyncMock()
        _attach_async_cursor(saver, cur)
        with patch.object(
            saver, "_aensure_user_id_schema", new_callable=AsyncMock
        ) as mock_schema:
            await saver.aensure_indexes()
        mock_schema.assert_awaited_once_with()

    async def test_aensure_user_id_schema_adds_column_and_index(self):
        """Missing column and index trigger ALTER TABLE and CREATE INDEX."""
        saver = _make_async_saver(user_id="alice")
        cur = AsyncMock()
        # fetchone returns None for both existence checks (column, index missing).
        cur.fetchone = AsyncMock(return_value=None)
        _attach_async_cursor(saver, cur)
        await saver._aensure_user_id_schema()
        executed = " ".join(call.args[0] for call in cur.execute.await_args_list)
        assert "ADD COLUMN user_id" in executed
        assert "CREATE INDEX" in executed

    async def test_aensure_user_id_schema_skips_when_present(self):
        """Existing column and index skip the DDL statements."""
        saver = _make_async_saver(user_id="alice")
        cur = AsyncMock()
        cur.fetchone = AsyncMock(return_value={"x": 1})
        _attach_async_cursor(saver, cur)
        await saver._aensure_user_id_schema()
        executed = " ".join(call.args[0] for call in cur.execute.await_args_list)
        assert "ADD COLUMN" not in executed
        assert "CREATE INDEX" not in executed


# =========================================================================
# Async aput / pruning / atomic user_id
# =========================================================================


class TestAsyncAput:
    """Tests for AsyncPruningPostgresSaver.aput helpers.

    NOTE: The public ``aput`` and ``aget_tuple`` entry points cannot be
    exercised on AsyncPruningPostgresSaver. They call
    ``self._validate_thread_ownership(...)``, but that method lives only in
    QueryableCheckpointerMixin, which this class does NOT inherit (its bases
    are VersionedCheckpointerMixin and AsyncPostgresSaver). Those calls raise
    AttributeError at runtime. See the bug report in the accompanying summary.
    The atomic helper ``_aput_with_user_id`` and ``_aprune_checkpoints`` do not
    touch ownership validation, so they are tested directly below.
    """

    async def test_aput_with_user_id_commits(self):
        """The async atomic path saves, updates user_id, and commits."""
        from psycopg_pool import AsyncConnectionPool

        saver = _make_async_saver(user_id="alice@example.com", keep_last_n=-1)

        conn = MagicMock()
        conn.set_autocommit = AsyncMock()
        conn.commit = AsyncMock()
        conn.rollback = AsyncMock()
        cur = AsyncMock()
        cur.rowcount = 1
        cur_ctx = MagicMock()
        cur_ctx.__aenter__ = AsyncMock(return_value=cur)
        cur_ctx.__aexit__ = AsyncMock(return_value=False)
        conn.cursor.return_value = cur_ctx

        conn_ctx = MagicMock()
        conn_ctx.__aenter__ = AsyncMock(return_value=conn)
        conn_ctx.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock(spec=AsyncConnectionPool)
        pool.connection.return_value = conn_ctx
        saver.conn = pool

        config = {"configurable": {"thread_id": "alice@example.com_c1"}}
        next_cfg = {
            "configurable": {
                "thread_id": "alice@example.com_c1",
                "checkpoint_id": "cp1",
            }
        }
        with patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.aput",
            new_callable=AsyncMock,
            return_value=next_cfg,
        ):
            result = await saver._aput_with_user_id(
                config, {"v": 1}, {}, {}, "alice@example.com_c1"
            )
        assert result is next_cfg
        conn.commit.assert_awaited_once_with()
        update_args = cur.execute.await_args[0]
        assert "UPDATE checkpoints SET user_id" in update_args[0]
        assert update_args[1] == ("alice@example.com", "alice@example.com_c1", "cp1")

    async def test_aput_with_user_id_rolls_back_on_missing_row(self):
        """A zero-row update raises and rolls back the async transaction."""
        from psycopg_pool import AsyncConnectionPool

        saver = _make_async_saver(user_id="alice@example.com", keep_last_n=-1)
        conn = MagicMock()
        conn.set_autocommit = AsyncMock()
        conn.commit = AsyncMock()
        conn.rollback = AsyncMock()
        cur = AsyncMock()
        cur.rowcount = 0
        cur_ctx = MagicMock()
        cur_ctx.__aenter__ = AsyncMock(return_value=cur)
        cur_ctx.__aexit__ = AsyncMock(return_value=False)
        conn.cursor.return_value = cur_ctx

        conn_ctx = MagicMock()
        conn_ctx.__aenter__ = AsyncMock(return_value=conn)
        conn_ctx.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock(spec=AsyncConnectionPool)
        pool.connection.return_value = conn_ctx
        saver.conn = pool

        config = {"configurable": {"thread_id": "alice@example.com_c1"}}
        next_cfg = {
            "configurable": {
                "thread_id": "alice@example.com_c1",
                "checkpoint_id": "cp1",
            }
        }
        with patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.aput",
            new_callable=AsyncMock,
            return_value=next_cfg,
        ):
            with pytest.raises(RuntimeError, match="Failed to set user_id"):
                await saver._aput_with_user_id(
                    config, {"v": 1}, {}, {}, "alice@example.com_c1"
                )
        conn.rollback.assert_awaited_once_with()

    async def test_aprune_deletes_excess(self):
        """_aprune_checkpoints deletes writes and checkpoints beyond the limit."""
        saver = _make_async_saver(keep_last_n=1)
        cur = AsyncMock()
        cur.fetchall = AsyncMock(
            return_value=[{"checkpoint_id": "c1"}, {"checkpoint_id": "c0"}]
        )
        _attach_async_cursor(saver, cur)
        await saver._aprune_checkpoints("t1")
        executed = [call.args[0] for call in cur.execute.await_args_list]
        # 1 SELECT + 2 deletes per stale checkpoint (writes + checkpoints).
        assert any("SELECT checkpoint_id" in s for s in executed)
        delete_writes = [s for s in executed if "DELETE FROM checkpoint_writes" in s]
        delete_cps = [
            s
            for s in executed
            if "DELETE FROM checkpoints WHERE" in s and "writes" not in s
        ]
        assert len(delete_writes) == 2
        assert len(delete_cps) == 2


# =========================================================================
# Async migration raw helpers (sync pool backed)
# =========================================================================


class TestAsyncMigrationRawHelpers:
    """Tests for the sync-pool-backed migration methods on the async saver."""

    def _setup_sync_pool(self, saver, cur):
        """Wire a fake sync pool whose connection yields the given cursor."""
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        pool = MagicMock()
        pool.connection.return_value.__enter__.return_value = conn
        pool.connection.return_value.__exit__.return_value = False
        saver._sync_pool = pool
        return conn

    def test_get_sync_pool_raises_when_unavailable(self):
        """_get_sync_pool raises when no sync pool can be created."""
        saver = _make_async_saver()
        saver._sync_pool = None
        with patch(
            "bili.iris.checkpointers.pg_checkpointer.get_pg_connection_pool",
            return_value=None,
        ):
            with pytest.raises(
                RuntimeError, match="Sync PostgreSQL pool not available"
            ):
                saver._get_sync_pool()

    def test_get_sync_pool_caches(self):
        """_get_sync_pool caches the created sync pool."""
        saver = _make_async_saver()
        saver._sync_pool = None
        fake_pool = MagicMock()
        with patch(
            "bili.iris.checkpointers.pg_checkpointer.get_pg_connection_pool",
            return_value=fake_pool,
        ) as mock_get:
            first = saver._get_sync_pool()
            second = saver._get_sync_pool()
        assert first is fake_pool
        assert second is fake_pool
        mock_get.assert_called_once_with()

    def test_get_raw_checkpoint_returns_row(self):
        """_get_raw_checkpoint maps a positional row tuple into a dict."""
        saver = _make_async_saver()
        cur = MagicMock()
        cur.fetchone.return_value = ("t1", "", "cp1", b"blob", {"step": 1})
        self._setup_sync_pool(saver, cur)
        result = saver._get_raw_checkpoint("t1", "")
        assert result == {
            "thread_id": "t1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp1",
            "checkpoint": b"blob",
            "metadata": {"step": 1},
        }

    def test_get_raw_checkpoint_returns_none(self):
        """_get_raw_checkpoint returns None when no row is found."""
        saver = _make_async_saver()
        cur = MagicMock()
        cur.fetchone.return_value = None
        self._setup_sync_pool(saver, cur)
        assert saver._get_raw_checkpoint("missing", "") is None

    def test_replace_raw_checkpoint_no_id_returns_false(self):
        """_replace_raw_checkpoint returns False without a checkpoint_id."""
        saver = _make_async_saver()
        assert saver._replace_raw_checkpoint("t1", {"thread_id": "t1"}) is False

    def test_replace_raw_checkpoint_updates_and_commits(self):
        """_replace_raw_checkpoint issues an UPDATE and commits."""
        saver = _make_async_saver()
        cur = MagicMock()
        cur.rowcount = 1
        conn = self._setup_sync_pool(saver, cur)
        doc = {
            "checkpoint_id": "cp1",
            "checkpoint": b"new",
            "metadata": {"step": 2},
        }
        assert saver._replace_raw_checkpoint("t1", doc, "") is True
        conn.commit.assert_called_once_with()
        assert "UPDATE checkpoints" in cur.execute.call_args[0][0]

    def test_archive_checkpoint_inserts_and_deletes(self):
        """_archive_checkpoint creates the archive table, inserts, and deletes."""
        saver = _make_async_saver()
        cur = MagicMock()
        conn = self._setup_sync_pool(saver, cur)
        doc = {
            "checkpoint_id": "cp1",
            "checkpoint": b"blob",
            "metadata": {},
            "checkpoint_ns": "",
        }
        saver._archive_checkpoint("t1", doc, RuntimeError("bad"))
        statements = " ".join(call.args[0] for call in cur.execute.call_args_list)
        assert "CREATE TABLE IF NOT EXISTS checkpoints_archive" in statements
        assert "INSERT INTO checkpoints_archive" in statements
        assert "DELETE FROM checkpoints" in statements
        conn.commit.assert_called_once_with()

    def test_archive_checkpoint_skips_delete_without_id(self):
        """Archiving a doc without checkpoint_id skips the main-table delete."""
        saver = _make_async_saver()
        cur = MagicMock()
        conn = self._setup_sync_pool(saver, cur)
        saver._archive_checkpoint("t1", {"checkpoint": b"x"}, RuntimeError("bad"))
        statements = " ".join(call.args[0] for call in cur.execute.call_args_list)
        assert "INSERT INTO checkpoints_archive" in statements
        assert "DELETE FROM checkpoints" not in statements
        conn.commit.assert_called_once_with()


# =========================================================================
# Async aget_tuple
# =========================================================================


# NOTE: AsyncPruningPostgresSaver.aget_tuple is not directly testable because
# it calls self._validate_thread_ownership(), which is not in this class's MRO
# (see the TestAsyncAput docstring and the bug report). Exercising it would
# require pinning the broken AttributeError path, which is disallowed.


# =========================================================================
# Cursor override branches
# =========================================================================


class TestCursorBranches:
    """Tests for the sync and async _cursor transaction-sharing branches."""

    def test_sync_cursor_uses_txn_conn_when_set(self):
        """The sync _cursor reuses _txn_conn when an atomic transaction is active."""
        saver = _make_sync_saver()
        txn_conn = MagicMock()
        shared_cur = MagicMock()
        txn_conn.cursor.return_value.__enter__.return_value = shared_cur
        txn_conn.cursor.return_value.__exit__.return_value = False
        saver._txn_conn = txn_conn
        with saver._cursor() as cur:
            assert cur is shared_cur
        txn_conn.cursor.assert_called_once()

    def test_sync_cursor_falls_back_to_super(self):
        """With no txn, the sync _cursor delegates to the parent's _cursor."""
        saver = _make_sync_saver()
        saver._txn_conn = None
        parent_cur = MagicMock()
        parent_ctx = MagicMock()
        parent_ctx.__enter__.return_value = parent_cur
        parent_ctx.__exit__.return_value = False
        with patch(
            "langgraph.checkpoint.postgres.PostgresSaver._cursor",
            return_value=parent_ctx,
        ):
            with saver._cursor() as cur:
                assert cur is parent_cur

    async def test_async_cursor_uses_txn_conn_when_set(self):
        """The async _cursor reuses _async_txn_conn during an atomic transaction."""
        saver = _make_async_saver()
        txn_conn = MagicMock()
        shared_cur = AsyncMock()
        cur_ctx = MagicMock()
        cur_ctx.__aenter__ = AsyncMock(return_value=shared_cur)
        cur_ctx.__aexit__ = AsyncMock(return_value=False)
        txn_conn.cursor.return_value = cur_ctx
        saver._async_txn_conn = txn_conn
        async with saver._cursor() as cur:
            assert cur is shared_cur
        txn_conn.cursor.assert_called_once()


# =========================================================================
# Sync query branches: tags-from-writes and dict messages
# =========================================================================


class TestSyncQueryBranches:
    """Tests for less-common sync query branches."""

    def test_get_user_threads_reads_tags_from_writes(self):
        """When channel_values lacks tags, tags are read from checkpoint_writes."""
        import msgpack

        saver = _make_sync_saver()

        thread_rows = [
            {
                "thread_id": "alice@example.com",
                "last_checkpoint_id": "cp1",
                "checkpoint_count": 1,
            }
        ]
        latest_row = {
            "checkpoint": {
                "ts": "2026-01-01T00:00:00Z",
                "channel_values": {"title": "Hi", "tags": []},
            }
        }
        tags_row = {"blob": msgpack.packb(["tagA", "tagB"])}

        call_state = {"n": 0}

        def fetchall_side():
            # Only the first fetchall (thread listing) returns rows.
            if call_state["n"] == 0:
                call_state["n"] += 1
                return thread_rows
            return []

        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = fetchall_side
        # fetchone is called twice per thread: latest checkpoint, then tags blob.
        mock_cur.fetchone.side_effect = [latest_row, tags_row]
        _attach_sync_cursor(saver, mock_cur)

        # get_tuple is called per thread to count messages; stub it out.
        with patch.object(saver, "get_tuple", return_value=None):
            threads = saver.get_user_threads("alice@example.com")

        assert threads[0]["tags"] == ["tagA", "tagB"]
        assert threads[0]["title"] == "Hi"

    def test_get_thread_messages_extracts_type_from_dict_message(self):
        """A serialized dict message has its type read from the 'type' field.

        Regression test for the dead-branch bug: type detection used to check
        ``hasattr(msg, "__class__")`` first, which is always true (dicts have
        __class__ too), so a dict message was labeled "dict" and resolved to
        role "unknown". With ``isinstance(msg, dict)`` checked first, the
        dict's 'type' field drives the mapping: {"type": "human"} -> "user".
        """
        saver = _make_sync_saver(user_id=None)
        saver.user_id = None  # ownership validation disabled
        dict_msg = {"type": "human", "content": "hello there"}
        checkpoint_tuple = MagicMock()
        checkpoint_tuple.checkpoint = {"channel_values": {"messages": [dict_msg]}}
        with patch.object(type(saver), "get_tuple", return_value=checkpoint_tuple):
            messages = saver.get_thread_messages("u@example.com_t1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello there"


class TestAsyncPgThreadOwnership:
    """The async PG saver validates thread ownership on aput and aget_tuple.

    AsyncPruningPostgresSaver now inherits QueryableCheckpointerMixin (mirroring
    the sync PruningPostgresSaver and the async Mongo saver), so it gets
    _validate_thread_ownership and the five query methods. A cross-tenant
    aput/aget_tuple is rejected with PermissionError instead of raising
    AttributeError.
    """

    async def test_aput_rejects_cross_tenant_write(self):
        """A write to another user's thread is rejected with PermissionError."""
        saver = _make_async_saver(user_id="owner@example.com")
        config = {
            "configurable": {
                "thread_id": "attacker@example.com_t1",
                "checkpoint_ns": "",
            }
        }
        with pytest.raises(PermissionError):
            await saver.aput(config, {}, {}, {})

    async def test_aget_tuple_rejects_cross_tenant_read(self):
        """A read of another user's thread is rejected with PermissionError."""
        saver = _make_async_saver(user_id="owner@example.com")
        config = {
            "configurable": {
                "thread_id": "attacker@example.com_t1",
                "checkpoint_ns": "",
            }
        }
        with pytest.raises(PermissionError):
            await saver.aget_tuple(config)

    def test_owner_thread_passes_validation(self):
        """The owner's own threads pass ownership validation without raising."""
        saver = _make_async_saver(user_id="owner@example.com")
        # Exact match and the "{user}_..." conversation form are both allowed.
        saver._validate_thread_ownership("owner@example.com")
        saver._validate_thread_ownership("owner@example.com_conv1")

    def test_validation_disabled_without_user_id(self):
        """With no user_id configured, ownership validation is a no-op."""
        saver = _make_async_saver(user_id=None)
        # Any thread id is accepted when validation is disabled.
        saver._validate_thread_ownership("anyone@example.com_t9")

    def test_query_methods_present(self):
        """All five QueryableCheckpointerMixin methods exist on the async saver."""
        saver = _make_async_saver(user_id="owner@example.com")
        for name in (
            "get_user_threads",
            "get_thread_messages",
            "delete_thread",
            "get_user_stats",
            "thread_exists",
        ):
            assert callable(getattr(saver, name))

    def test_query_methods_delegate_to_sync_saver(self):
        """The query methods forward to a sync delegate sharing the pool."""
        saver = _make_async_saver(user_id="owner@example.com")
        delegate = MagicMock()
        delegate.get_user_threads.return_value = [{"thread_id": "owner@example.com"}]
        delegate.thread_exists.return_value = True
        saver._query_delegate = delegate

        assert saver.get_user_threads("owner@example.com") == [
            {"thread_id": "owner@example.com"}
        ]
        assert saver.thread_exists("owner@example.com_t1") is True
        delegate.get_user_threads.assert_called_once_with("owner@example.com", None, 0)

    async def test_async_query_variants_delegate_off_thread(self):
        """The a*-prefixed variants run the sync query in a worker thread."""
        saver = _make_async_saver(user_id="owner@example.com")
        delegate = MagicMock()
        delegate.get_user_threads.return_value = [{"thread_id": "t"}]
        delegate.get_thread_messages.return_value = [{"role": "user"}]
        delegate.delete_thread.return_value = False
        delegate.get_user_stats.return_value = {"total_threads": 1}
        delegate.thread_exists.return_value = True
        saver._query_delegate = delegate

        assert await saver.aget_user_threads("owner@example.com") == [
            {"thread_id": "t"}
        ]
        assert await saver.aget_thread_messages("t1") == [{"role": "user"}]
        assert await saver.adelete_thread("t1") is False
        assert await saver.aget_user_stats("owner@example.com") == {"total_threads": 1}
        assert await saver.athread_exists("t1") is True
        delegate.get_user_threads.assert_called_once_with("owner@example.com", None, 0)
