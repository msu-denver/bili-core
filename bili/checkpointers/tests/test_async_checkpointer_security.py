"""Async checkpointer security tests.

Tests async checkpointer implementations (AsyncPruningMongoDBSaver) for:
- Thread ownership validation on async operations
- Cross-tenant attack prevention in async context
- Multi-tenant isolation with async methods

Note: These tests require MongoDB to be running locally. They will be skipped
if MongoDB is not available.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access,abstract-class-instantiated

import pytest
from langchain_core.messages import HumanMessage

from bili.checkpointers.mongo_checkpointer import AsyncPruningMongoDBSaver

# Skip tests if MongoDB is not available
pytest_plugins = ("pytest_anyio",)

try:
    from motor.motor_asyncio import AsyncIOMotorClient

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Mark all tests in this module as anyio
pytestmark = pytest.mark.anyio


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def async_mongo_db():
    """Provide async MongoDB database for testing."""
    if not MONGODB_AVAILABLE:
        pytest.skip("MongoDB (motor) not available")

    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["test_bili_async_security"]

    # Clean up before test
    await db.checkpoints.delete_many({})
    await db.checkpoint_writes.delete_many({})

    yield db

    # Clean up after test
    await db.checkpoints.delete_many({})
    await db.checkpoint_writes.delete_many({})
    client.close()


@pytest.fixture
async def async_checkpointer(async_mongo_db):
    """Provide async checkpointer without user_id for setup."""
    return AsyncPruningMongoDBSaver(async_mongo_db, keep_last_n=5)


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

    async def test_async_checkpointer_with_user_id_initialization(self, async_mongo_db):
        """Test that async checkpointer accepts user_id parameter."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_db, keep_last_n=5, user_id="user@example.com"
        )
        assert checkpointer.user_id == "user@example.com"

    async def test_async_checkpointer_validates_thread_ownership(self, async_mongo_db):
        """Test that async checkpointer validates thread ownership."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_db, keep_last_n=5, user_id="user@example.com"
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
        self, async_mongo_db
    ):
        """Test that async checkpointer rejects invalid threads."""
        checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_db, keep_last_n=5, user_id="user@example.com"
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
        self, async_mongo_db, async_checkpointer
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
            async_mongo_db, keep_last_n=5, user_id="attacker@example.com"
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_sensitive"
        ):
            await attacker_checkpointer.aget_tuple(
                _make_config("victim@example.com_sensitive")
            )

    async def test_async_attack_write_via_aput(
        self, async_mongo_db, async_checkpointer
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
            async_mongo_db, keep_last_n=5, user_id="attacker@example.com"
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

    async def test_async_attack_delete_thread(self, async_mongo_db, async_checkpointer):
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
            async_mongo_db, keep_last_n=5, user_id="attacker@example.com"
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_important"
        ):
            await attacker_checkpointer.adelete_thread("victim@example.com_important")

    async def test_async_attack_read_messages(self, async_mongo_db, async_checkpointer):
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
            async_mongo_db, keep_last_n=5, user_id="attacker@example.com"
        )

        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_private"
        ):
            await attacker_checkpointer.aget_thread_messages(
                "victim@example.com_private"
            )

    async def test_async_attack_conversation_id_guessing(
        self, async_mongo_db, async_checkpointer
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
            async_mongo_db, keep_last_n=5, user_id="attacker@example.com"
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
        self, async_mongo_db, async_checkpointer
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
                async_mongo_db, keep_last_n=5, user_id=user
            )
            threads = await user_checkpointer.aget_user_threads(user)

            # Should see exactly 3 threads (their own)
            assert len(threads) == 3

            # All threads should belong to this user
            for thread in threads:
                assert thread["thread_id"].startswith(user)

    async def test_async_stats_isolated_per_user(
        self, async_mongo_db, async_checkpointer
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
            async_mongo_db, keep_last_n=5, user_id="user1@example.com"
        )
        user1_stats = await user1_checkpointer.aget_user_stats("user1@example.com")
        assert user1_stats["total_threads"] == 3

        user2_checkpointer = AsyncPruningMongoDBSaver(
            async_mongo_db, keep_last_n=5, user_id="user2@example.com"
        )
        user2_stats = await user2_checkpointer.aget_user_stats("user2@example.com")
        assert user2_stats["total_threads"] == 5
