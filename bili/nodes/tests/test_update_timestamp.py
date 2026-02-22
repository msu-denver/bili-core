"""Tests for update_timestamp node.

Tests the timestamp update functionality:
- Updating current message time
- Tracking previous message time
- Calculating delta time between messages
- Handling invalid timestamp formats
"""

# pylint: disable=missing-function-docstring

import time
from datetime import datetime, timezone

from bili.nodes.update_timestamp import build_update_timestamp_node


class TestUpdateTimestamp:
    """Tests for build_update_timestamp_node function."""

    def test_sets_current_message_time(self):
        """Test that current_message_time is set with ISO format timestamp."""
        state = {}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Should have current_message_time
        assert "current_message_time" in result
        # Should be valid ISO format
        datetime.fromisoformat(result["current_message_time"])

    def test_sets_previous_message_time_from_current(self):
        """Test that previous_message_time is set from existing current_message_time."""
        existing_time = "2024-01-01T12:00:00+00:00"
        state = {"current_message_time": existing_time}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Previous should be the old current
        assert result["previous_message_time"] == existing_time
        # Current should be updated (different from previous)
        assert result["current_message_time"] != existing_time

    def test_calculates_delta_time(self):
        """Test that delta_message_time is calculated correctly."""
        # Set a past time
        past_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        state = {"current_message_time": past_time}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Delta should be positive (time has passed)
        assert "delta_message_time" in result
        assert result["delta_message_time"] > 0

    def test_first_message_has_zero_delta(self):
        """Test that first message (no previous time) has ~zero delta."""
        state = {}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Delta should be very small (close to 0)
        assert "delta_message_time" in result
        assert result["delta_message_time"] < 0.1  # Less than 100ms

    def test_preserves_other_state_fields(self):
        """Test that other state fields are preserved."""
        state = {"messages": ["msg1", "msg2"], "user_id": "user123", "other": "data"}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # All original fields should be preserved
        assert result["messages"] == ["msg1", "msg2"]
        assert result["user_id"] == "user123"
        assert result["other"] == "data"

    def test_handles_invalid_timestamp_format(self):
        """Test graceful handling of invalid timestamp format."""
        state = {"current_message_time": "not-a-valid-timestamp"}

        # Execute node (should not crash)
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Should still produce valid timestamps
        assert "current_message_time" in result
        assert "previous_message_time" in result
        assert "delta_message_time" in result

        # Delta should be small (falls back to current time)
        assert result["delta_message_time"] < 0.1

    def test_timestamp_progression(self):
        """Test timestamp progression across multiple calls."""
        state = {}

        # First call
        node_func = build_update_timestamp_node()
        state = node_func(state)
        first_time = state["current_message_time"]

        # Small delay
        time.sleep(0.01)

        # Second call
        state = node_func(state)
        second_time = state["current_message_time"]

        # Times should progress
        assert second_time > first_time
        assert state["previous_message_time"] == first_time
        assert state["delta_message_time"] > 0

    def test_iso_format_includes_timezone(self):
        """Test that ISO timestamp includes timezone information."""
        state = {}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        # Parse timestamp and verify it has timezone
        dt = datetime.fromisoformat(result["current_message_time"])
        assert dt.tzinfo is not None

    def test_node_function_is_callable(self):
        """Test that build_update_timestamp_node returns a callable."""
        node_func = build_update_timestamp_node()
        assert callable(node_func)

    def test_kwargs_are_accepted(self):
        """Test that function accepts arbitrary kwargs for extensibility."""
        node_func = build_update_timestamp_node(some_param="value")
        assert callable(node_func)

    def test_delta_time_is_float(self):
        """Test that delta_message_time is a float."""
        state = {"current_message_time": "2024-01-01T12:00:00+00:00"}

        # Execute node
        node_func = build_update_timestamp_node()
        result = node_func(state)

        assert isinstance(result["delta_message_time"], float)

    def test_repeated_calls_accumulate_time(self):
        """Test that repeated calls show increasing delta times."""
        state = {}

        node_func = build_update_timestamp_node()

        # First call
        state = node_func(state)
        delta1 = state["delta_message_time"]

        # Wait a bit
        time.sleep(0.05)

        # Second call
        state = node_func(state)
        delta2 = state["delta_message_time"]

        # Delta should have increased
        assert delta2 > delta1
        assert delta2 >= 0.04  # At least 40ms (allowing some margin)

    def test_current_time_is_recent(self):
        """Test that current_message_time is close to actual current time."""
        before = datetime.now(timezone.utc)

        state = {}
        node_func = build_update_timestamp_node()
        result = node_func(state)

        after = datetime.now(timezone.utc)

        # Parse result timestamp
        result_time = datetime.fromisoformat(result["current_message_time"])

        # Should be between before and after
        assert before <= result_time <= after
