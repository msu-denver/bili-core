"""Tests for mock_tool.

Tests the mock tool initialization and functionality:
- Response simulation
- Response time delays
- Response time conversion
- Middleware support
"""

# pylint: disable=missing-function-docstring

import time

from bili.tools.mock_tool import init_mock_tool


class TestMockTool:
    """Tests for init_mock_tool function."""

    def test_creates_tool_with_name_and_description(self):
        """Test that tool is created with correct name and description."""
        tool = init_mock_tool(
            name="Test Tool",
            description="A test tool",
            mock_response="test",
            response_time=0,
        )

        assert tool.name == "Test Tool"
        assert tool.description == "A test tool"

    def test_returns_mock_response(self):
        """Test that tool returns the specified mock response."""
        mock_response = "This is a mock response"
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response=mock_response,
            response_time=0,
        )

        result = tool.func("any input")
        assert result == mock_response

    def test_response_time_delay(self):
        """Test that tool delays response by specified time."""
        response_time = 0.1  # 100ms
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=response_time,
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should take at least response_time seconds
        assert elapsed >= response_time

    def test_zero_response_time_no_delay(self):
        """Test that zero response_time causes no delay."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=0,
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should be nearly instant
        assert elapsed < 0.05  # Less than 50ms

    def test_converts_string_response_time_to_float(self):
        """Test that string response_time is converted to float."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time="0.05",  # String
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should delay by the converted value
        assert elapsed >= 0.04  # At least 40ms

    def test_invalid_response_time_defaults_to_zero(self):
        """Test that invalid response_time string defaults to 0."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time="not-a-number",
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should not delay (defaults to 0)
        assert elapsed < 0.05

    def test_accepts_int_response_time(self):
        """Test that integer response_time works correctly."""
        tool = init_mock_tool(
            name="Mock", description="Test", mock_response="response", response_time=1
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should delay by 1 second
        assert elapsed >= 0.9  # Allow some timing variance

    def test_accepts_float_response_time(self):
        """Test that float response_time works correctly."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=0.05,
        )

        start = time.time()
        tool.func("input")
        elapsed = time.time() - start

        # Should delay by 0.05 seconds
        assert elapsed >= 0.04

    def test_mock_response_can_be_dict(self):
        """Test that mock_response can be any type (dict)."""
        mock_data = {"key": "value", "number": 42}
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response=mock_data,
            response_time=0,
        )

        result = tool.func("input")
        assert result == mock_data

    def test_mock_response_can_be_list(self):
        """Test that mock_response can be a list."""
        mock_list = [1, 2, 3, "four"]
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response=mock_list,
            response_time=0,
        )

        result = tool.func("input")
        assert result == mock_list

    def test_middleware_parameter_accepted(self):
        """Test that middleware parameter is accepted without error."""
        mock_middleware = [{"name": "test_middleware"}]
        # Should not raise error when middleware is provided
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=0,
            middleware=mock_middleware,
        )

        assert tool is not None

    def test_middleware_none_accepted(self):
        """Test that middleware=None is accepted without error."""
        # Should not raise error when middleware is None
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=0,
            middleware=None,
        )

        assert tool is not None

    def test_multiple_invocations_return_same_response(self):
        """Test that multiple calls return consistent mock response."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="consistent",
            response_time=0,
        )

        result1 = tool.func("input1")
        result2 = tool.func("input2")
        result3 = tool.func("different input")

        assert result1 == "consistent"
        assert result2 == "consistent"
        assert result3 == "consistent"

    def test_input_parameter_is_ignored(self):
        """Test that input parameter to tool.func is ignored."""
        tool = init_mock_tool(
            name="Mock",
            description="Test",
            mock_response="response",
            response_time=0,
        )

        # Different inputs should all return same mock response
        assert tool.func("input1") == "response"
        assert tool.func("input2") == "response"
        assert tool.func(None) == "response"
        assert tool.func(123) == "response"
