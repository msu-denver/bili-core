"""Tests for bili.utils.logging_utils.

Covers get_log_level, get_logger, the custom TRACE level constant, the
trace() logger method, and the module-load root-logger configuration.
"""

import importlib
import logging
import os
from unittest.mock import MagicMock, patch

import bili.utils.logging_utils as logging_utils
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


# ------------------------------------------------------------------
# trace() logger method
# ------------------------------------------------------------------


class TestTraceMethod:
    """The custom trace method emits a record only when TRACE is enabled."""

    def test_trace_emits_record_when_enabled(self):
        """A TRACE-level logger dispatches the record to its handler."""
        logger = get_logger("trace.emit")
        logger.setLevel(TRACE)
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            logger.trace("hello trace")
        finally:
            logger.removeHandler(handler)
        assert any(r.getMessage() == "hello trace" for r in records)

    def test_trace_suppressed_when_disabled(self):
        """A logger above TRACE level emits no record for trace()."""
        logger = get_logger("trace.suppressed")
        logger.setLevel(logging.INFO)
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            logger.trace("should not appear")
        finally:
            logger.removeHandler(handler)
        assert records == []


# ------------------------------------------------------------------
# Module-load root-logger configuration
# ------------------------------------------------------------------


class TestModuleRootConfiguration:
    """Module import configures the root logger based on existing handlers."""

    def teardown_method(self):
        """Reload the module unpatched so later tests see normal state."""
        importlib.reload(logging_utils)

    def test_sets_level_when_handlers_present_and_log_level_set(self):
        """A configured root logger gets its level set only when LOG_LEVEL is set."""
        fake_root = MagicMock()
        fake_root.handlers = [MagicMock()]
        with patch("logging.getLogger", return_value=fake_root), patch(
            "logging.basicConfig"
        ) as mock_basic, patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
            importlib.reload(logging_utils)
        fake_root.setLevel.assert_called_once()
        mock_basic.assert_not_called()

    def test_respects_host_config_when_log_level_unset(self):
        """A configured root logger is left untouched when LOG_LEVEL is absent."""
        fake_root = MagicMock()
        fake_root.handlers = [MagicMock()]
        env_without_log_level = {
            k: v for k, v in os.environ.items() if k != "LOG_LEVEL"
        }
        with patch("logging.getLogger", return_value=fake_root), patch(
            "logging.basicConfig"
        ) as mock_basic, patch.dict("os.environ", env_without_log_level, clear=True):
            importlib.reload(logging_utils)
        fake_root.setLevel.assert_not_called()
        mock_basic.assert_not_called()

    def test_basic_config_when_root_has_no_handlers(self):
        """An unconfigured root logger is set up with basicConfig."""
        fake_root = MagicMock()
        fake_root.handlers = []
        with patch("logging.getLogger", return_value=fake_root), patch(
            "logging.basicConfig"
        ) as mock_basic:
            importlib.reload(logging_utils)
        mock_basic.assert_called_once()
        fake_root.setLevel.assert_not_called()
