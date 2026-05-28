"""Async checkpointer security tests.

Tests async checkpointer implementations (AsyncPruningMongoDBSaver) for:
- Thread ownership validation on async operations
- Cross-tenant attack prevention in async context
- Multi-tenant isolation with async methods

Note: These tests require MongoDB to be running locally. They will be skipped
if MongoDB is not available. AsyncPruningMongoDBSaver now inherits from
MongoDBSaver (langgraph-checkpoint-mongodb 0.3.x) which uses a sync
pymongo.MongoClient and provides both sync and async methods natively.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access,abstract-class-instantiated


import mongomock
import pytest
from langchain_core.messages import HumanMessage

from bili.iris.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

# anyio's pytest plugin is auto-registered via the anyio package's
# `anyio.pytest_plugin` entry_point when anyio is installed. The
# `pytestmark = pytest.mark.anyio` below is all that's needed to opt
# the tests in. There is no separate `pytest_anyio` package on PyPI, so
# declaring `pytest_plugins = ("pytest_anyio",)` here would fail
# collection with ImportError.


# Mark all tests in this module as anyio
pytestmark = pytest.mark.anyio


# ======================================================================
# Fixtures
# ======================================================================


_TEST_DB_NAME = "test_bili_async_security"


@pytest.fixture(autouse=True)
def _patch_mongomock_pymongo49_compat(monkeypatch):
    """Bridge mongomock 4.3.0 to the pymongo 4.9+ API that langgraph uses.

    langgraph's ``MongoDBSaver.__init__`` (the parent class) bootstraps indexes
    with two pymongo-4.9-era calls that mongomock 4.3.0 predates:

    1. ``collection.list_indexes().to_list()`` — ``to_list()`` is a pymongo 4.9+
       cursor method; mongomock returns a bare generator without it.
    2. ``collection.create_index(keys=[...], unique=True)`` — pymongo names the
       first parameter ``keys``; mongomock names it ``key_or_list``, so the
       keyword form raises ``TypeError``.

    Both raise during construction against the in-memory backend. This shim
    adds ``to_list()`` to the index cursor and translates the ``keys=`` keyword
    to mongomock's ``key_or_list``. Remove once mongomock ships
    pymongo-4.9-compatible cursors and ``create_index`` signature.
    """
    import mongomock.collection

    original_list_indexes = mongomock.collection.Collection.list_indexes
    original_create_index = mongomock.collection.Collection.create_index

    class _ToListCursor:
        def __init__(self, items):
            self._items = items

        def to_list(self, length=None):  # noqa: ARG002 - pymongo signature parity
            return list(self._items)

        def __iter__(self):
            return iter(self._items)

    def _patched_list_indexes(self, *args, **kwargs):
        return _ToListCursor(list(original_list_indexes(self, *args, **kwargs)))

    def _patched_create_index(self, *args, **kwargs):
        # pymongo accepts the index spec as keys=; mongomock wants key_or_list.
        if "keys" in kwargs and "key_or_list" not in kwargs:
            kwargs["key_or_list"] = kwargs.pop("keys")
        # mongomock doesn't accept expireAfterSeconds; drop it (TTL behavior is
        # irrelevant to these access-control tests).
        kwargs.pop("expireAfterSeconds", None)
        return original_create_index(self, *args, **kwargs)

    monkeypatch.setattr(
        mongomock.collection.Collection, "list_indexes", _patched_list_indexes
    )
    monkeypatch.setattr(
        mongomock.collection.Collection, "create_index", _patched_create_index
    )

    # mongomock 4.3.0's aggregation parser doesn't implement the $toDate
    # type-conversion operator. get_user_threads() uses
    # {"$addFields": {"last_updated": {"$toDate": "$last_oid"}}} to turn the
    # max ObjectId into its embedded creation timestamp for recency sorting.
    # Add a minimal $toDate handler covering the types this codebase produces.
    import datetime as _dt

    import mongomock.aggregate
    from bson import ObjectId as _ObjectId

    _original_parse = mongomock.aggregate._Parser.parse

    def _patched_parse(self, expression):
        if isinstance(expression, dict) and list(expression.keys()) == ["$toDate"]:
            operand = expression["$toDate"]
            value = (
                self.parse(operand)
                if isinstance(operand, dict)
                else self._parse_basic_expression(operand)
            )
            if value is None or isinstance(value, _dt.datetime):
                return value
            if isinstance(value, _ObjectId):
                return value.generation_time
            if isinstance(value, (int, float)):
                return _dt.datetime.fromtimestamp(value / 1000, tz=_dt.timezone.utc)
            if isinstance(value, str):
                return _dt.datetime.fromisoformat(value)
            raise TypeError(f"$toDate cannot convert {type(value).__name__}")
        return _original_parse(self, expression)

    monkeypatch.setattr(mongomock.aggregate._Parser, "parse", _patched_parse)


@pytest.fixture
async def async_mongo_client():
    """Provide an in-memory mongomock client for testing.

    MongoDBSaver 0.3.x uses a sync pymongo.MongoClient internally and
    provides async methods natively — no motor AsyncIOMotorClient needed.
    mongomock.MongoClient is a drop-in replacement that implements real
    MongoDB query semantics (filtering, per-document isolation) in memory,
    so these cross-tenant security tests validate actual access-control
    behavior without requiring a live MongoDB server. Each test gets a
    fresh client, so no cross-test cleanup is needed.
    """
    client = mongomock.MongoClient()
    yield client
    client.close()


@pytest.fixture
async def async_checkpointer(async_mongo_client):
    """Provide async checkpointer without user_id for setup."""
    return AsyncPruningMongoDBSaver(
        async_mongo_client, db_name=_TEST_DB_NAME, keep_last_n=5
    )


def _make_config(thread_id: str):
    """Create a basic checkpoint config."""
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _make_checkpoint_data():
    """Create minimal checkpoint data for testing."""
    return {
        "v": 1,
        "id": "test_checkpoint_id",
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {
            "messages": [HumanMessage(content="test message")],
        },
        "channel_versions": {
            "__start__": 1,
            "messages": 1,
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
        },
        "pending_sends": [],
    }


# ======================================================================
# Async User ID Validation Tests
# ======================================================================


class TestAsyncMongoCheckpointerUserID:
    """Tests for AsyncPruningMongoDBSaver with user_id parameter."""

    async def test_async_checkpointer_with_user_id_initialization(
        self, async_mongo_client
    ):
        """Test that async checkpointer accepts user_id parameter."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="user@example.com",
        )
        assert checkpointer.user_id == "user@example.com"

    async def test_async_checkpointer_validates_thread_ownership(
        self, async_mongo_client
    ):
        """Test that async checkpointer validates thread ownership."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="user@example.com",
        )

        # Valid thread IDs
        valid_threads = [
            "user@example.com",
            "user@example.com_conv1",
            "user@example.com_conversation_123",
        ]

        for thread_id in valid_threads:
            # Should not raise error
            checkpointer._validate_thread_ownership(thread_id)

    async def test_async_checkpointer_rejects_invalid_thread_ownership(
        self, async_mongo_client
    ):
        """Test that async checkpointer rejects invalid threads."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="user@example.com",
        )

        # Invalid thread IDs
        invalid_threads = [
            "other@example.com",
            "other@example.com_conv1",
            "user@example.com.hacker",
        ]

        for thread_id in invalid_threads:
            with pytest.raises(
                PermissionError, match="Access denied: thread_id.*does not belong to"
            ):
                checkpointer._validate_thread_ownership(thread_id)


# ======================================================================
# Async Cross-Tenant Attack Tests
# ======================================================================


class TestAsyncCrossTenantAttacks:
    """Test cross-tenant attack scenarios in async context."""

    async def test_async_attack_read_via_aget_tuple(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that users cannot read other users' threads via aget_tuple()."""
        # Setup: Create victim's thread using non-validated checkpointer
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_sensitive")
        await async_checkpointer.aput(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: Attacker checkpointer tries to read
        attacker_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="attacker@example.com",
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_sensitive"
        ):
            await attacker_checkpointer.aget_tuple(
                _make_config("victim@example.com_sensitive")
            )

    async def test_async_attack_write_via_aput(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that users cannot write to other users' threads via aput()."""
        # Setup: Create victim's thread
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_protected")
        await async_checkpointer.aput(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: Attacker tries to overwrite
        attacker_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="attacker@example.com",
        )

        malicious_checkpoint = _make_checkpoint_data()
        malicious_checkpoint["channel_values"]["messages"] = [
            HumanMessage(content="malicious data")
        ]

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_protected"
        ):
            await attacker_checkpointer.aput(
                _make_config("victim@example.com_protected"),
                malicious_checkpoint,
                {"source": "input", "step": 2, "writes": {}},
                {},
            )

    async def test_async_attack_delete_thread(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that users cannot delete other users' threads."""
        # Setup: Create victim's thread
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_important")
        await async_checkpointer.aput(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: Attacker tries to delete
        attacker_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="attacker@example.com",
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_important"
        ):
            await attacker_checkpointer.adelete_thread("victim@example.com_important")

    async def test_async_attack_read_messages(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that users cannot read other users' messages."""
        # Setup: Create victim's thread with messages
        checkpoint_data = _make_checkpoint_data()
        checkpoint_data["channel_values"]["messages"] = [
            HumanMessage(content="confidential information")
        ]
        config = _make_config("victim@example.com_private")
        await async_checkpointer.aput(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: Attacker tries to read messages
        attacker_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="attacker@example.com",
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_private"
        ):
            await attacker_checkpointer.aget_thread_messages(
                "victim@example.com_private"
            )

    async def test_async_attack_conversation_id_guessing(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that attackers cannot guess conversation IDs."""
        # Setup: Create predictable conversation IDs
        checkpoint_data = _make_checkpoint_data()
        common_conv_ids = ["work", "personal", "default"]

        for conv_id in common_conv_ids:
            config = _make_config(f"victim@example.com_{conv_id}")
            await async_checkpointer.aput(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

        # Attack: Attacker tries to guess
        attacker_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="attacker@example.com",
        )

        for conv_id in common_conv_ids:
            guessed_thread_id = f"victim@example.com_{conv_id}"
            with pytest.raises(
                PermissionError, match=f"Access denied.*{guessed_thread_id}"
            ):
                await attacker_checkpointer.aget_tuple(_make_config(guessed_thread_id))


# ======================================================================
# Async Multi-Tenant Isolation Tests
# ======================================================================


class TestAsyncMultiTenantIsolation:
    """Test multi-tenant isolation in async operations."""

    async def test_async_each_user_sees_only_their_threads(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that get_user_threads only returns user's threads in async."""
        checkpoint_data = _make_checkpoint_data()

        # Create threads for multiple users
        users = ["user1@example.com", "user2@example.com", "user3@example.com"]
        for user in users:
            for conv_num in range(1, 4):
                config = _make_config(f"{user}_conv{conv_num}")
                await async_checkpointer.aput(
                    config,
                    checkpoint_data,
                    {"source": "input", "step": 1, "writes": {}},
                    {},
                )

        # Each user should only see their own threads
        for user in users:
            user_checkpointer = AsyncPruningMongoDBSaver(
                async_mongo_client,
                db_name=_TEST_DB_NAME,
                keep_last_n=5,
                user_id=user,
            )
            threads = await user_checkpointer.aget_user_threads(user)

            # Should see exactly 3 threads (their own)
            assert len(threads) == 3

            # All threads should belong to this user
            for thread in threads:
                assert thread["thread_id"].startswith(user)

    async def test_async_stats_isolated_per_user(
        self, async_mongo_client, async_checkpointer
    ):
        """Test that user stats are isolated in async operations."""
        checkpoint_data = _make_checkpoint_data()

        # Create threads for two users
        for user, count in [("user1@example.com", 3), ("user2@example.com", 5)]:
            for conv_num in range(1, count + 1):
                config = _make_config(f"{user}_conv{conv_num}")
                await async_checkpointer.aput(
                    config,
                    checkpoint_data,
                    {"source": "input", "step": 1, "writes": {}},
                    {},
                )

        # Check stats for each user
        user1_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="user1@example.com",
        )
        user1_stats = await user1_checkpointer.aget_user_stats("user1@example.com")
        assert user1_stats["total_threads"] == 3

        user2_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_client,
            db_name=_TEST_DB_NAME,
            keep_last_n=5,
            user_id="user2@example.com",
        )
        user2_stats = await user2_checkpointer.aget_user_stats("user2@example.com")
        assert user2_stats["total_threads"] == 5
