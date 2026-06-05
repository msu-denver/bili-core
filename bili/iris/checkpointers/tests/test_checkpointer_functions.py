"""Tests for bili.iris.checkpointers.checkpointer_functions factory."""

from unittest.mock import AsyncMock, patch

import pytest

from bili.iris.checkpointers.checkpointer_functions import (
    get_async_checkpointer,
    get_checkpointer,
)
from bili.iris.checkpointers.memory_checkpointer import QueryableMemorySaver

pytestmark = pytest.mark.anyio


class TestGetCheckpointer:
    """Test the checkpointer factory function."""

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_memory_saver_when_no_env_vars(self):
        """With no database env vars, should fall back to QueryableMemorySaver."""
        checkpointer = get_checkpointer()
        assert isinstance(checkpointer, QueryableMemorySaver)

    @patch("bili.iris.checkpointers.checkpointer_functions.get_pg_checkpointer")
    @patch.dict(
        "os.environ",
        {"POSTGRES_CONNECTION_STRING": "postgresql://localhost"},
        clear=True,
    )
    def test_returns_pg_checkpointer_when_postgres_set(self, mock_get_pg):
        """POSTGRES_CONNECTION_STRING routes to the PostgreSQL checkpointer."""
        sentinel = object()
        mock_get_pg.return_value = sentinel
        checkpointer = get_checkpointer()
        assert checkpointer is sentinel
        mock_get_pg.assert_called_once_with()

    @patch("bili.iris.checkpointers.checkpointer_functions.get_mongo_checkpointer")
    @patch("bili.iris.checkpointers.checkpointer_functions.get_pg_checkpointer")
    @patch.dict(
        "os.environ",
        {"MONGO_CONNECTION_STRING": "mongodb://localhost"},
        clear=True,
    )
    def test_returns_mongo_checkpointer_when_only_mongo_set(
        self, mock_get_pg, mock_get_mongo
    ):
        """MONGO_CONNECTION_STRING routes to the Mongo checkpointer when PG is unset."""
        sentinel = object()
        mock_get_mongo.return_value = sentinel
        checkpointer = get_checkpointer()
        assert checkpointer is sentinel
        mock_get_mongo.assert_called_once_with()
        mock_get_pg.assert_not_called()

    @patch.dict(
        "os.environ",
        {"POSTGRES_CONNECTION_STRING": "", "MONGO_CONNECTION_STRING": ""},
        clear=True,
    )
    def test_returns_memory_saver_when_env_vars_empty(self):
        """Empty strings are falsy, so should still fall back to memory."""
        checkpointer = get_checkpointer()
        assert isinstance(checkpointer, QueryableMemorySaver)

    def test_memory_saver_is_functional(self):
        """The returned memory checkpointer should be usable."""
        checkpointer = get_checkpointer()
        # QueryableMemorySaver inherits from MemorySaver which is always functional
        assert hasattr(checkpointer, "get")
        assert hasattr(checkpointer, "put")


class TestGetAsyncCheckpointer:
    """Test the async checkpointer factory function."""

    @patch.dict("os.environ", {}, clear=True)
    async def test_returns_memory_saver_when_no_env_vars(self):
        """With no database env vars, falls back to QueryableMemorySaver."""
        checkpointer = await get_async_checkpointer()
        assert isinstance(checkpointer, QueryableMemorySaver)

    @patch(
        "bili.iris.checkpointers.checkpointer_functions.get_async_pg_checkpointer",
        new_callable=AsyncMock,
    )
    @patch.dict(
        "os.environ",
        {"POSTGRES_CONNECTION_STRING": "postgresql://localhost"},
        clear=True,
    )
    async def test_returns_async_pg_when_postgres_set(self, mock_get_pg):
        """POSTGRES_CONNECTION_STRING routes to the async PostgreSQL checkpointer."""
        sentinel = object()
        mock_get_pg.return_value = sentinel
        checkpointer = await get_async_checkpointer()
        assert checkpointer is sentinel
        mock_get_pg.assert_awaited_once_with()

    @patch(
        "bili.iris.checkpointers.checkpointer_functions.get_async_mongo_checkpointer",
        new_callable=AsyncMock,
    )
    @patch(
        "bili.iris.checkpointers.checkpointer_functions.get_async_pg_checkpointer",
        new_callable=AsyncMock,
    )
    @patch.dict(
        "os.environ",
        {"MONGO_CONNECTION_STRING": "mongodb://localhost"},
        clear=True,
    )
    async def test_returns_async_mongo_when_only_mongo_set(
        self, mock_get_pg, mock_get_mongo
    ):
        """MONGO_CONNECTION_STRING routes to async Mongo when PG is unset."""
        sentinel = object()
        mock_get_mongo.return_value = sentinel
        checkpointer = await get_async_checkpointer()
        assert checkpointer is sentinel
        mock_get_mongo.assert_awaited_once_with()
        mock_get_pg.assert_not_awaited()
