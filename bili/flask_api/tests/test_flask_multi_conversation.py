"""Integration tests for Flask API multi-conversation support.

Tests the multi-conversation functionality added in Task #27:
    - conversation_id parameter handling
    - Thread ID construction ({user_id}_{conversation_id})
    - Conversation isolation
    - Backward compatibility (single conversation without conversation_id)
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,import-outside-toplevel,unused-variable

from unittest.mock import MagicMock

import pytest
from flask import Flask, g

from bili.flask_api.flask_utils import handle_agent_prompt

# ======================================================================
# Test Setup
# ======================================================================


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return {"email": "user@example.com", "uid": "test_user_123"}


@pytest.fixture
def mock_agent():
    """Create a mock conversation agent."""
    mock = MagicMock()
    # Simulate agent response
    mock.invoke.return_value = {
        "messages": [
            MagicMock(content="Test response from agent"),
        ]
    }
    return mock


@pytest.fixture
def app_context():
    """Create a Flask app context for tests."""
    app = Flask(__name__)
    with app.app_context():
        yield app


# ======================================================================
# Multi-Conversation Tests
# ======================================================================


class TestFlaskMultiConversation:
    """Tests for Flask API multi-conversation support."""

    def test_handle_agent_prompt_with_conversation_id(
        self, mock_user, mock_agent, app_context
    ):
        """Test that conversation_id creates proper thread_id pattern."""
        # Call handle_agent_prompt with conversation_id
        result = handle_agent_prompt(
            mock_user, mock_agent, "Test prompt", conversation_id="conv_123"
        )

        # Verify agent was called with correct thread_id
        mock_agent.invoke.assert_called_once()
        call_args = mock_agent.invoke.call_args

        # Check the config parameter
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        thread_id = config["configurable"]["thread_id"]

        # Should use pattern: {email}_{conversation_id}
        assert thread_id == "user@example.com_conv_123"

        # Verify response
        assert result.json["response"] == "Test response from agent"

    def test_handle_agent_prompt_without_conversation_id(
        self, mock_user, mock_agent, app_context
    ):
        """Test backward compatibility: no conversation_id uses email only."""
        # Call without conversation_id
        handle_agent_prompt(mock_user, mock_agent, "Test prompt")

        # Verify thread_id is just the email (backward compatible)
        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        thread_id = config["configurable"]["thread_id"]

        assert thread_id == "user@example.com"

    def test_multiple_conversations_use_different_thread_ids(
        self, mock_user, mock_agent, app_context
    ):
        """Test that different conversation_ids create isolated threads."""
        # Call with different conversation_ids
        result1 = handle_agent_prompt(
            mock_user, mock_agent, "Prompt 1", conversation_id="work"
        )
        result2 = handle_agent_prompt(
            mock_user, mock_agent, "Prompt 2", conversation_id="personal"
        )

        # Get thread_ids from both calls
        call1_config = mock_agent.invoke.call_args_list[0][0][1]["configurable"]
        call2_config = mock_agent.invoke.call_args_list[1][0][1]["configurable"]

        thread_id_1 = call1_config["thread_id"]
        thread_id_2 = call2_config["thread_id"]

        # Should have different thread_ids
        assert thread_id_1 == "user@example.com_work"
        assert thread_id_2 == "user@example.com_personal"
        assert thread_id_1 != thread_id_2

    def test_same_conversation_id_reuses_thread(
        self, mock_user, mock_agent, app_context
    ):
        """Test that same conversation_id reuses the same thread_id."""
        # Call twice with same conversation_id
        handle_agent_prompt(
            mock_user, mock_agent, "Message 1", conversation_id="conv_abc"
        )
        handle_agent_prompt(
            mock_user, mock_agent, "Message 2", conversation_id="conv_abc"
        )

        # Both calls should use same thread_id
        call1_config = mock_agent.invoke.call_args_list[0][0][1]["configurable"]
        call2_config = mock_agent.invoke.call_args_list[1][0][1]["configurable"]

        assert call1_config["thread_id"] == call2_config["thread_id"]
        assert call1_config["thread_id"] == "user@example.com_conv_abc"


class TestConversationIDValidation:
    """Tests for conversation_id parameter validation and edge cases."""

    def test_conversation_id_with_special_characters(
        self, mock_user, mock_agent, app_context
    ):
        """Test that conversation_id with special characters works correctly."""
        # Use conversation_id with hyphens, underscores, numbers
        conversation_ids = [
            "conv-2024-01-15",
            "user_session_123",
            "meeting.notes",
        ]

        for conv_id in conversation_ids:
            handle_agent_prompt(mock_user, mock_agent, "Test", conversation_id=conv_id)

            call_args = mock_agent.invoke.call_args
            config = (
                call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
            )
            thread_id = config["configurable"]["thread_id"]

            # Should include the conversation_id in thread_id
            assert thread_id == f"user@example.com_{conv_id}"

    def test_empty_conversation_id_uses_default(
        self, mock_user, mock_agent, app_context
    ):
        """Test that empty string conversation_id is treated as None."""
        # Pass empty string
        handle_agent_prompt(mock_user, mock_agent, "Test", conversation_id="")

        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        thread_id = config["configurable"]["thread_id"]

        # Empty string should be falsy, so should use email only
        assert thread_id == "user@example.com"

    def test_none_conversation_id_uses_default(
        self, mock_user, mock_agent, app_context
    ):
        """Test that None conversation_id uses default behavior."""
        handle_agent_prompt(mock_user, mock_agent, "Test", conversation_id=None)

        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        thread_id = config["configurable"]["thread_id"]

        # None should use email only (backward compatible)
        assert thread_id == "user@example.com"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing API consumers."""

    def test_omitting_conversation_id_parameter_works(
        self, mock_user, mock_agent, app_context
    ):
        """Test that omitting conversation_id entirely works (backward compatible)."""
        # Call without conversation_id parameter at all
        result = handle_agent_prompt(mock_user, mock_agent, "Old API call")

        # Should still work
        assert result.json["response"] == "Test response from agent"

        # Should use email as thread_id
        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        assert config["configurable"]["thread_id"] == "user@example.com"

    def test_existing_single_conversation_behavior(
        self, mock_user, mock_agent, app_context
    ):
        """Test that users without conversation_id maintain single conversation."""
        # Multiple calls without conversation_id
        handle_agent_prompt(mock_user, mock_agent, "Message 1")
        handle_agent_prompt(mock_user, mock_agent, "Message 2")
        handle_agent_prompt(mock_user, mock_agent, "Message 3")

        # All should use same thread_id (email)
        for call in mock_agent.invoke.call_args_list:
            config = call[1]["config"] if "config" in call[1] else call[0][1]
            assert config["configurable"]["thread_id"] == "user@example.com"


class TestMultiUserMultiConversation:
    """Tests for multiple users with multiple conversations each."""

    def test_different_users_same_conversation_id(self, mock_agent, app_context):
        """Test that different users can use same conversation_id without conflict."""
        user1 = {"email": "alice@example.com", "uid": "user_alice"}
        user2 = {"email": "bob@example.com", "uid": "user_bob"}

        # Both users create conversation with same ID
        handle_agent_prompt(user1, mock_agent, "Alice message", conversation_id="work")
        handle_agent_prompt(user2, mock_agent, "Bob message", conversation_id="work")

        # Should have different thread_ids (scoped to user)
        call1_config = mock_agent.invoke.call_args_list[0][0][1]["configurable"]
        call2_config = mock_agent.invoke.call_args_list[1][0][1]["configurable"]

        assert call1_config["thread_id"] == "alice@example.com_work"
        assert call2_config["thread_id"] == "bob@example.com_work"
        assert call1_config["thread_id"] != call2_config["thread_id"]

    def test_user_can_maintain_multiple_conversations(
        self, mock_user, mock_agent, app_context
    ):
        """Test that a single user can maintain multiple separate conversations."""
        conversations = ["research", "coding", "meeting_notes", "personal"]

        # User creates multiple conversations
        for conv_id in conversations:
            handle_agent_prompt(
                mock_user,
                mock_agent,
                f"Message in {conv_id}",
                conversation_id=conv_id,
            )

        # Verify all have different thread_ids
        thread_ids = set()
        for call in mock_agent.invoke.call_args_list:
            config = call[0][1]["configurable"]
            thread_ids.add(config["thread_id"])

        # Should have 4 unique thread_ids
        assert len(thread_ids) == 4
        expected_ids = {f"user@example.com_{conv_id}" for conv_id in conversations}
        assert thread_ids == expected_ids


# ======================================================================
# Flask Route Integration Tests (with mocked Flask app)
# ======================================================================


class TestFlaskRouteIntegration:
    """Integration tests with Flask routes (mocked)."""

    @pytest.fixture
    def app(self):
        """Create a test Flask app."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()

    def test_flask_request_extracts_conversation_id(
        self, app, client, mock_agent, app_context
    ):
        """Test that Flask routes correctly extract conversation_id from JSON."""

        @app.route("/test", methods=["POST"])
        def test_route():
            """Test route that uses handle_agent_prompt."""
            from flask import request

            # Simulate authenticated user in g
            g.user = {"email": "test@example.com", "uid": "test_123"}

            # Extract from request
            request_data = request.get_json()
            prompt = request_data.get("prompt", "")
            conversation_id = request_data.get("conversation_id")

            # Call handler
            return handle_agent_prompt(g.user, mock_agent, prompt, conversation_id)

        # Test with conversation_id
        response = client.post(
            "/test",
            json={"prompt": "Test prompt", "conversation_id": "test_conv"},
            content_type="application/json",
        )

        assert response.status_code == 200

        # Verify agent was called with correct thread_id
        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        assert config["configurable"]["thread_id"] == "test@example.com_test_conv"

    def test_flask_request_without_conversation_id(
        self, app, client, mock_agent, app_context
    ):
        """Test that Flask routes handle missing conversation_id gracefully."""

        @app.route("/test", methods=["POST"])
        def test_route():
            from flask import request

            g.user = {"email": "test@example.com", "uid": "test_123"}

            request_data = request.get_json()
            prompt = request_data.get("prompt", "")
            conversation_id = request_data.get("conversation_id")

            return handle_agent_prompt(g.user, mock_agent, prompt, conversation_id)

        # Test without conversation_id in JSON
        response = client.post(
            "/test",
            json={"prompt": "Test prompt"},
            content_type="application/json",
        )

        assert response.status_code == 200

        # Should use email only
        call_args = mock_agent.invoke.call_args
        config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
        assert config["configurable"]["thread_id"] == "test@example.com"
