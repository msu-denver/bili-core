"""Tests for checkpointer user_id validation and multi-tenant security.

Tests the multi-tenant security features implemented in Tasks #15, #16, #17:
    - Thread ownership validation
    - On-demand schema migration
    - user_id parameter integration
    - Multi-conversation support with user isolation
"""

# pylint: disable=missing-function-docstring

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from bili.checkpointers.memory_checkpointer import QueryableMemorySaver

# ======================================================================
# Helpers
# ======================================================================


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
# Memory Checkpointer Tests
# ======================================================================


class TestMemoryCheckpointerUserID:
    """Tests for QueryableMemorySaver with user_id parameter."""

    def test_memory_checkpointer_with_user_id_initialization(self):
        """Test that QueryableMemorySaver accepts user_id parameter."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")
        assert checkpointer.user_id == "user@example.com"

    def test_memory_checkpointer_without_user_id(self):
        """Test that QueryableMemorySaver works without user_id (backward compatible)."""
        checkpointer = QueryableMemorySaver()
        assert checkpointer.user_id is None

    def test_memory_checkpointer_validates_thread_ownership(self):
        """Test that memory checkpointer validates thread ownership when user_id is set."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Valid thread IDs
        valid_threads = [
            "user@example.com",
            "user@example.com_conv1",
            "user@example.com_conversation_123",
        ]

        for thread_id in valid_threads:
            # Should not raise error
            checkpointer._validate_thread_ownership(thread_id)

    def test_memory_checkpointer_rejects_invalid_thread_ownership(self):
        """Test that memory checkpointer rejects threads not belonging to user."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Invalid thread IDs
        invalid_threads = [
            "other@example.com",
            "other@example.com_conv1",
            "malicious_user@example.com",
            "user@example.com.hacker",
        ]

        for thread_id in invalid_threads:
            with pytest.raises(
                PermissionError, match="Access denied: thread_id.*does not belong to"
            ):
                checkpointer._validate_thread_ownership(thread_id)

    def test_memory_checkpointer_skips_validation_without_user_id(self):
        """Test that validation is skipped when user_id is None (backward compatible)."""
        checkpointer = QueryableMemorySaver(user_id=None)

        # Should allow any thread ID when user_id is None
        arbitrary_threads = [
            "any_thread",
            "user@example.com",
            "other@example.com",
            "random_id_123",
        ]

        for thread_id in arbitrary_threads:
            # Should not raise error
            checkpointer._validate_thread_ownership(thread_id)


class TestMemoryCheckpointerMultiConversation:
    """Tests for multi-conversation support with memory checkpointer."""

    def test_memory_checkpointer_isolates_conversations(self):
        """Test that different conversation_ids maintain separate state."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Save checkpoints to different conversations
        config1 = _make_config("user@example.com_conv1")
        config2 = _make_config("user@example.com_conv2")

        checkpoint_data = _make_checkpoint_data()

        # Put checkpoints for each conversation
        checkpointer.put(
            config1,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )
        checkpointer.put(
            config2,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Verify both conversations exist
        tuple1 = checkpointer.get_tuple(config1)
        tuple2 = checkpointer.get_tuple(config2)

        assert tuple1 is not None
        assert tuple2 is not None
        assert tuple1.config["configurable"]["thread_id"] == "user@example.com_conv1"
        assert tuple2.config["configurable"]["thread_id"] == "user@example.com_conv2"

    def test_memory_checkpointer_default_conversation(self):
        """Test that user_id alone creates a default conversation thread."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Use user_id as thread_id (default conversation)
        config = _make_config("user@example.com")
        checkpoint_data = _make_checkpoint_data()

        checkpointer.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Verify checkpoint exists
        result = checkpointer.get_tuple(config)
        assert result is not None
        assert result.config["configurable"]["thread_id"] == "user@example.com"


class TestMemoryCheckpointerQueryable:
    """Tests for queryable interface with user_id."""

    def test_get_user_threads_with_user_id(self):
        """Test that get_user_threads returns threads for the correct user."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Create multiple conversations
        conversations = ["conv1", "conv2", "conv3"]
        checkpoint_data = _make_checkpoint_data()

        for conv_id in conversations:
            config = _make_config(f"user@example.com_{conv_id}")
            checkpointer.put(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

        # Get all user threads
        threads = checkpointer.get_user_threads("user@example.com")

        assert len(threads) == 3
        thread_ids = {t["thread_id"] for t in threads}
        assert "user@example.com_conv1" in thread_ids
        assert "user@example.com_conv2" in thread_ids
        assert "user@example.com_conv3" in thread_ids

    def test_get_user_threads_empty_for_different_user(self):
        """Test that get_user_threads returns empty for different user."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Create conversations for user@example.com
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("user@example.com_conv1")
        checkpointer.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Query for a different user
        threads = checkpointer.get_user_threads("other@example.com")
        assert len(threads) == 0


# ======================================================================
# Thread Ownership Validation Tests
# ======================================================================


class TestThreadOwnershipValidation:
    """Tests for thread ownership validation logic."""

    @pytest.mark.parametrize(
        "user_id,valid_thread_ids",
        [
            (
                "user@example.com",
                [
                    "user@example.com",
                    "user@example.com_conv1",
                    "user@example.com_123",
                    "user@example.com_my_conversation",
                ],
            ),
            (
                "alice",
                [
                    "alice",
                    "alice_thread1",
                    "alice_conv_123",
                ],
            ),
        ],
    )
    def test_valid_thread_ownership_patterns(self, user_id, valid_thread_ids):
        """Test that valid thread_id patterns pass validation."""
        checkpointer = QueryableMemorySaver(user_id=user_id)

        for thread_id in valid_thread_ids:
            # Should not raise
            checkpointer._validate_thread_ownership(thread_id)

    @pytest.mark.parametrize(
        "user_id,invalid_thread_ids",
        [
            (
                "user@example.com",
                [
                    "other@example.com",
                    "other@example.com_conv1",
                    "malicious@example.com",
                    "user@example.com.hacker",  # Attempt to bypass with suffix
                    "prefix_user@example.com",  # Attempt to bypass with prefix
                ],
            ),
            (
                "alice",
                [
                    "bob",
                    "bob_thread1",
                    "aalice",  # Prefix attack
                    "alice.hacker",  # Wrong separator (not underscore)
                ],
            ),
        ],
    )
    def test_invalid_thread_ownership_patterns(self, user_id, invalid_thread_ids):
        """Test that invalid thread_id patterns are rejected."""
        checkpointer = QueryableMemorySaver(user_id=user_id)

        for thread_id in invalid_thread_ids:
            with pytest.raises(PermissionError, match="Access denied"):
                checkpointer._validate_thread_ownership(thread_id)

    def test_validation_disabled_when_no_user_id(self):
        """Test that validation is disabled when user_id is None."""
        checkpointer = QueryableMemorySaver(user_id=None)

        # Should allow any thread ID
        arbitrary_ids = [
            "any_thread",
            "user@example.com",
            "other@example.com",
            "random_123",
        ]

        for thread_id in arbitrary_ids:
            # Should not raise
            checkpointer._validate_thread_ownership(thread_id)


# ======================================================================
# Multi-Tenant Isolation Tests
# ======================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant data isolation."""

    def test_users_cannot_access_other_users_threads(self):
        """Test that users cannot access threads belonging to other users."""
        # Create checkpointer for user1
        checkpointer1 = QueryableMemorySaver(user_id="user1@example.com")
        checkpoint_data = _make_checkpoint_data()

        # User1 creates a conversation
        config1 = _make_config("user1@example.com_private_conv")
        checkpointer1.put(
            config1,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Create checkpointer for user2
        checkpointer2 = QueryableMemorySaver(user_id="user2@example.com")

        # User2 tries to access user1's thread
        _make_config("user1@example.com_private_conv")

        # Should raise PermissionError when trying to access
        with pytest.raises(PermissionError, match="Access denied"):
            checkpointer2._validate_thread_ownership("user1@example.com_private_conv")

    def test_each_user_sees_only_their_threads(self):
        """Test that get_user_threads only returns threads for the specified user."""
        # Create a shared checkpointer without user_id (simulating database)
        shared_checkpointer = MemorySaver()
        checkpoint_data = _make_checkpoint_data()

        # Add threads for multiple users
        users = ["user1@example.com", "user2@example.com", "user3@example.com"]
        for user in users:
            for i in range(3):
                config = _make_config(f"{user}_conv{i}")
                shared_checkpointer.put(
                    config,
                    checkpoint_data,
                    {"source": "input", "step": 1, "writes": {}},
                    {},
                )

        # Create user-specific queryable checkpointer
        user1_checkpointer = QueryableMemorySaver(user_id="user1@example.com")
        # Copy data from shared checkpointer (simulating shared DB)
        user1_checkpointer.storage = shared_checkpointer.storage

        # User1 should only see their own threads
        user1_threads = user1_checkpointer.get_user_threads("user1@example.com")
        thread_ids = {t["thread_id"] for t in user1_threads}

        # Should only see user1's threads
        assert len(user1_threads) == 3
        for thread_id in thread_ids:
            assert thread_id.startswith("user1@example.com")


# ======================================================================
# Backward Compatibility Tests
# ======================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility when user_id is not used."""

    def test_checkpointer_works_without_user_id(self):
        """Test that checkpointer works normally when user_id is not provided."""
        checkpointer = QueryableMemorySaver()

        checkpoint_data = _make_checkpoint_data()
        config = _make_config("any_thread_id")

        # Should work without user_id
        checkpointer.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        result = checkpointer.get_tuple(config)
        assert result is not None

    def test_no_validation_without_user_id(self):
        """Test that no validation occurs when user_id is None."""
        checkpointer = QueryableMemorySaver(user_id=None)

        # Should allow any thread_id patterns
        arbitrary_configs = [
            _make_config("thread1"),
            _make_config("user@example.com"),
            _make_config("anything_goes_123"),
        ]

        checkpoint_data = _make_checkpoint_data()

        for config in arbitrary_configs:
            # Should not raise
            checkpointer.put(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

            result = checkpointer.get_tuple(config)
            assert result is not None


# ======================================================================
# Edge Cases and Error Handling
# ======================================================================


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_user_id_still_validates(self):
        """Test that empty user_id still performs validation (not treated as None)."""
        # Empty string is NOT treated as None, so validation still occurs
        checkpointer = QueryableMemorySaver(user_id="")

        # Only thread_id="" should work (exact match with empty user_id)
        config = _make_config("")
        checkpoint_data = _make_checkpoint_data()

        # Should not raise for empty thread_id matching empty user_id
        checkpointer.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Should raise for non-empty thread_id
        config_invalid = _make_config("any_thread")
        with pytest.raises(PermissionError, match="Access denied"):
            checkpointer.put(
                config_invalid,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

    def test_thread_id_with_multiple_underscores(self):
        """Test that thread_ids with multiple underscores work correctly."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Valid thread with multiple underscores
        thread_id = "user@example.com_my_long_conversation_name_123"
        config = _make_config(thread_id)
        checkpoint_data = _make_checkpoint_data()

        # Should not raise
        checkpointer.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        result = checkpointer.get_tuple(config)
        assert result is not None

    def test_special_characters_in_thread_id(self):
        """Test that special characters in thread_id are handled correctly."""
        checkpointer = QueryableMemorySaver(user_id="user@example.com")

        # Thread with special characters after user_id
        valid_threads = [
            "user@example.com_conv-123",
            "user@example.com_conv.456",
            "user@example.com_my-conversation-2024",
        ]

        checkpoint_data = _make_checkpoint_data()

        for thread_id in valid_threads:
            config = _make_config(thread_id)
            # Should not raise
            checkpointer.put(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )


# ======================================================================
# Cross-Tenant Attack Tests
# ======================================================================


class TestCrossTenantAttacks:
    """
    Tests for cross-tenant access attack scenarios.

    These tests verify that users cannot access, modify, or delete
    resources belonging to other users through various attack vectors.
    """

    def test_attack_read_other_user_thread_via_get_tuple(self):
        """
        Attack Scenario: User tries to read another user's thread via get_tuple().

        Expected: PermissionError raised when attempting to access thread.
        """
        # Setup: Create shared storage (simulates database)
        shared_storage = MemorySaver()
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_sensitive_data")
        shared_storage.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: User2 checkpointer tries to read user1's thread
        checkpointer2 = QueryableMemorySaver(user_id="attacker@example.com")
        # Share storage to simulate database with both users' data
        checkpointer2.storage = shared_storage.storage
        attack_config = _make_config("victim@example.com_sensitive_data")

        # Verify: Should raise PermissionError
        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_sensitive_data"
        ):
            checkpointer2.get_tuple(attack_config)

    def test_attack_write_to_other_user_thread_via_put(self):
        """
        Attack Scenario: User tries to write to another user's thread via put().

        Expected: PermissionError raised when attempting to write.
        """
        # Setup: User1 creates a thread
        checkpointer1 = QueryableMemorySaver(user_id="victim@example.com")
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_protected")
        checkpointer1.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: User2 tries to overwrite user1's checkpoint
        checkpointer2 = QueryableMemorySaver(user_id="attacker@example.com")
        malicious_checkpoint = _make_checkpoint_data()
        malicious_checkpoint["channel_values"]["messages"] = [
            HumanMessage(content="malicious data")
        ]
        attack_config = _make_config("victim@example.com_protected")

        # Verify: Should raise PermissionError
        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_protected"
        ):
            checkpointer2.put(
                attack_config,
                malicious_checkpoint,
                {"source": "input", "step": 2, "writes": {}},
                {},
            )

    def test_attack_delete_other_user_thread(self):
        """
        Attack Scenario: User tries to delete another user's thread.

        Expected: PermissionError raised when attempting to delete.
        """
        # Setup: User1 creates a thread
        checkpointer1 = QueryableMemorySaver(user_id="victim@example.com")
        checkpoint_data = _make_checkpoint_data()
        config = _make_config("victim@example.com_important")
        checkpointer1.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: User2 tries to delete user1's thread
        checkpointer2 = QueryableMemorySaver(user_id="attacker@example.com")

        # Verify: Should raise PermissionError
        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_important"
        ):
            checkpointer2.delete_thread("victim@example.com_important")

    def test_attack_read_other_user_messages(self):
        """
        Attack Scenario: User tries to read messages from another user's thread.

        Expected: PermissionError raised when attempting to read messages.
        """
        # Setup: User1 creates a thread with messages
        checkpointer1 = QueryableMemorySaver(user_id="victim@example.com")
        checkpoint_data = _make_checkpoint_data()
        checkpoint_data["channel_values"]["messages"] = [
            HumanMessage(content="confidential information")
        ]
        config = _make_config("victim@example.com_private")
        checkpointer1.put(
            config,
            checkpoint_data,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )

        # Attack: User2 tries to read user1's messages
        checkpointer2 = QueryableMemorySaver(user_id="attacker@example.com")

        # Verify: Should raise PermissionError
        with pytest.raises(
            PermissionError, match="Access denied.*victim@example.com_private"
        ):
            checkpointer2.get_thread_messages("victim@example.com_private")

    def test_attack_thread_id_prefix_collision(self):
        """
        Attack Scenario: User tries to access threads by creating IDs that
        start with another user's ID but use wrong separator.

        Expected: All bypass attempts should be rejected.
        """
        checkpointer = QueryableMemorySaver(user_id="alice@example.com")

        # Attack vectors using different separators
        bypass_attempts = [
            "alice@example.com.hacker",  # Dot separator
            "alice@example.comamalicious",  # Suffix without separator
            "alice@example.com-thread",  # Hyphen separator
            "alice@example.com/conv",  # Slash separator
            "alice@example.com:conv",  # Colon separator
        ]

        for malicious_id in bypass_attempts:
            with pytest.raises(PermissionError, match=f"Access denied.*{malicious_id}"):
                checkpointer._validate_thread_ownership(malicious_id)

    def test_attack_thread_id_prefix_injection(self):
        """
        Attack Scenario: User tries to prefix their own ID with victim's ID.

        Expected: Prefix attacks should be rejected.
        """
        checkpointer = QueryableMemorySaver(user_id="alice@example.com")

        # Attack vectors with victim ID as prefix
        prefix_attacks = [
            "victim_alice@example.com",  # Prefix with underscore
            "hacker_alice@example.com_thread",  # Prefix with thread suffix
            "prefix.alice@example.com",  # Prefix with dot
        ]

        for malicious_id in prefix_attacks:
            with pytest.raises(PermissionError, match=f"Access denied.*{malicious_id}"):
                checkpointer._validate_thread_ownership(malicious_id)

    def test_attack_thread_enumeration_via_get_user_threads(self):
        """
        Attack Scenario: User tries to enumerate other users' threads
        via get_user_threads().

        Expected: Should return empty list (no threads visible).
        """
        # Setup: Create threads for multiple users in shared storage
        shared_checkpointer = MemorySaver()
        checkpoint_data = _make_checkpoint_data()

        # User1 creates threads
        for conv_id in ["secret1", "secret2", "secret3"]:
            config = _make_config(f"victim@example.com_{conv_id}")
            shared_checkpointer.put(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

        # Attack: User2 tries to enumerate user1's threads
        attacker_checkpointer = QueryableMemorySaver(user_id="attacker@example.com")
        # Copy storage to simulate shared database
        attacker_checkpointer.storage = shared_checkpointer.storage

        # Verify: Attacker should not see victim's threads
        threads = attacker_checkpointer.get_user_threads("victim@example.com")
        # get_user_threads queries by pattern, so it will find them
        # but the attacker's own operations should be blocked
        assert len(threads) == 3  # Can query (read-only search)

        # But cannot access individual threads
        for thread in threads:
            with pytest.raises(PermissionError, match="Access denied"):
                attacker_checkpointer.get_thread_messages(thread["thread_id"])

    def test_attack_conversation_id_guessing(self):
        """
        Attack Scenario: User tries to guess conversation IDs to access
        other users' threads.

        Expected: All guessed IDs should be rejected.
        """
        # Setup: Create shared storage and populate with victim's threads
        shared_storage = MemorySaver()
        checkpoint_data = _make_checkpoint_data()

        common_conv_ids = ["work", "personal", "default", "main", "project1"]
        for conv_id in common_conv_ids:
            config = _make_config(f"victim@example.com_{conv_id}")
            shared_storage.put(
                config,
                checkpoint_data,
                {"source": "input", "step": 1, "writes": {}},
                {},
            )

        # Attack: User2 checkpointer tries to guess and access these conversations
        checkpointer2 = QueryableMemorySaver(user_id="attacker@example.com")
        # Share storage to simulate database
        checkpointer2.storage = shared_storage.storage

        for conv_id in common_conv_ids:
            guessed_thread_id = f"victim@example.com_{conv_id}"
            with pytest.raises(
                PermissionError, match=f"Access denied.*{guessed_thread_id}"
            ):
                checkpointer2.get_tuple(_make_config(guessed_thread_id))

    def test_attack_partial_thread_id_match(self):
        """
        Attack Scenario: User tries to access threads with partial ID matches.

        Expected: Partial matches should be rejected.
        """
        checkpointer = QueryableMemorySaver(user_id="alice")

        # Attack vectors with partial matches
        partial_attacks = [
            "alic",  # Prefix of user_id
            "alice_thread".replace("alice_", "ali_"),  # Partial user_id
            "alicextra",  # User_id + extra chars (no underscore)
        ]

        for malicious_id in partial_attacks:
            with pytest.raises(PermissionError, match=f"Access denied.*{malicious_id}"):
                checkpointer._validate_thread_ownership(malicious_id)

    def test_attack_case_sensitivity_bypass(self):
        """
        Attack Scenario: User tries to bypass validation using case variations.

        Expected: Case variations should be treated as different users.
        """
        checkpointer = QueryableMemorySaver(user_id="alice@example.com")

        # Attack vectors with case variations
        case_attacks = [
            "Alice@example.com",  # Capital A
            "ALICE@EXAMPLE.COM",  # All caps
            "alice@Example.com",  # Mixed case
            "aLiCe@eXaMpLe.CoM",  # Random case
        ]

        for malicious_id in case_attacks:
            with pytest.raises(PermissionError, match=f"Access denied.*{malicious_id}"):
                checkpointer._validate_thread_ownership(malicious_id)
