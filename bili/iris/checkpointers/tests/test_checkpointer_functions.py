"""Tests for bili.iris.checkpointers.checkpointer_functions factory."""

from unittest.mock import patch

from bili.iris.checkpointers.checkpointer_functions import get_checkpointer
from bili.iris.checkpointers.memory_checkpointer import QueryableMemorySaver


class TestGetCheckpointer:
    """Test the checkpointer factory function."""

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_memory_saver_when_no_env_vars(self):
        """With no database env vars, should fall back to QueryableMemorySaver."""
        checkpointer = get_checkpointer()
        assert isinstance(checkpointer, QueryableMemorySaver)

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
