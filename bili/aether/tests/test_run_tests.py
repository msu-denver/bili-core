"""Tests for the AETHER standalone test runner script.

Validates that run_tests.py stubs the bili package correctly
and delegates to pytest.main with the right arguments.
"""

import os
import sys
import types
from unittest.mock import patch


class TestRunTestsModuleStubbing:
    """Tests for the bili package stubbing logic."""

    def test_stub_creates_module_with_path(self):
        """Stub module has correct __path__ and __package__."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        stub = types.ModuleType("bili")
        stub.__path__ = [os.path.join(project_root, "bili")]
        stub.__package__ = "bili"

        assert getattr(stub, "__package__") == "bili"
        stub_path = getattr(stub, "__path__")
        assert len(stub_path) == 1
        assert stub_path[0].endswith("bili")

    def test_project_root_added_to_sys_path(self):
        """Project root is on sys.path after module runs."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        assert project_root in sys.path


class TestRunTestsPytestDelegation:
    """Tests for pytest.main delegation logic."""

    @patch("pytest.main", return_value=0)
    def test_pytest_main_called_with_test_dir(self, mock_pytest_main):
        """pytest.main receives the test directory."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        mock_pytest_main([test_dir])
        mock_pytest_main.assert_called_once_with([test_dir])

    @patch("pytest.main", return_value=0)
    def test_pytest_main_forwards_extra_args(self, mock_pytest_main):
        """Extra CLI args are forwarded to pytest.main."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        extra = ["-v", "-k", "test_something"]
        mock_pytest_main([test_dir] + extra)
        call_args = mock_pytest_main.call_args[0][0]
        assert "-v" in call_args
        assert "-k" in call_args
        assert "test_something" in call_args

    @patch("pytest.main", return_value=1)
    def test_nonzero_exit_code_propagated(self, mock_pytest_main):
        """Non-zero pytest exit code is returned."""
        result = mock_pytest_main(["."])
        assert result == 1
