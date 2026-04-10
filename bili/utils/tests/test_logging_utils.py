"""Tests for bili.utils.logging_utils.

Covers get_log_level, get_logger, and the custom TRACE level
constant.
"""

import logging

from bili.utils.logging_utils import TRACE, get_log_level, get_logger

# ------------------------------------------------------------------
# TRACE constant
# ------------------------------------------------------------------


class TestTraceConstant:
    """Verify the TRACE level is registered correctly."""

    def test_trace_value_is_five(self):
        """TRACE numeric value equals 5."""
        assert TRACE == 5

    def test_trace_level_name_registered(self):
        """logging.getLevelName(5) returns 'TRACE'."""
        assert logging.getLevelName(5) == "TRACE"


# ------------------------------------------------------------------
# get_log_level
# ------------------------------------------------------------------


class TestGetLogLevel:
    """get_log_level maps level names to numeric values."""

    def test_trace_returns_trace_constant(self):
        """'TRACE' maps to the TRACE constant (5)."""
        assert get_log_level("TRACE") == TRACE

    def test_debug(self):
        """'DEBUG' returns logging.DEBUG."""
        assert get_log_level("DEBUG") == logging.DEBUG

    def test_info(self):
        """'INFO' returns logging.INFO."""
        assert get_log_level("INFO") == logging.INFO

    def test_warning(self):
        """'WARNING' returns logging.WARNING."""
        assert get_log_level("WARNING") == logging.WARNING

    def test_error(self):
        """'ERROR' returns logging.ERROR."""
        assert get_log_level("ERROR") == logging.ERROR

    def test_critical(self):
        """'CRITICAL' returns logging.CRITICAL."""
        assert get_log_level("CRITICAL") == logging.CRITICAL

    def test_unknown_falls_back_to_info(self):
        """Unknown level name defaults to logging.INFO."""
        assert get_log_level("NONEXISTENT") == logging.INFO


# ------------------------------------------------------------------
# get_logger
# ------------------------------------------------------------------


class TestGetLogger:
    """get_logger returns a properly named Logger instance."""

    def test_returns_logger_instance(self):
        """Returned object is a logging.Logger."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        """Logger name matches the argument."""
        logger = get_logger("my.custom.name")
        assert logger.name == "my.custom.name"

    def test_logger_has_trace_method(self):
        """Logger instances have the custom trace method."""
        logger = get_logger("trace.check")
        assert hasattr(logger, "trace")
        assert callable(logger.trace)
