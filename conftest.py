"""Repository-level pytest configuration.

This file lives at the repository root, outside the ``bili`` package, so it is
never shipped in the wheel. pytest loads it for every run under this directory,
whether a run starts from the root or names a test directory or file directly,
because it sits in the config file's directory (the default ``--confcutdir``).
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent

# Test locations whose modules import the optional [streamlit] extra at module
# level, keyed by the module each needs. The extra does not install on every
# supported Python; without it these modules (and the bili.streamlit_ui.query
# package their imports pull in) cannot be imported at all.
_NEEDS_STREAMLIT = {
    "streamlit": (
        "bili/aether/ui/tests",
        "bili/streamlit_ui/query/tests",
        "bili/streamlit_ui/tests",
    ),
    "streamlit_flow": ("bili/aether/tests/test_yaml_to_graph.py",),
}

# Modules under the locations above that exist to check behavior without
# streamlit, so they are always collected.
_RUNS_WITHOUT_STREAMLIT = ("bili/streamlit_ui/tests/test_streamlit_utils.py",)


def _missing_requirement(module_path):
    """Return the uninstalled module a test file needs, or None.

    ``find_spec`` locates a module without importing it, so a streamlit that is
    installed but fails to import counts as present: its import error then
    fails collection loudly instead of turning into a skip.
    """
    try:
        relative = module_path.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return None
    if relative in _RUNS_WITHOUT_STREAMLIT:
        return None
    for module_name, locations in _NEEDS_STREAMLIT.items():
        for location in locations:
            inside = relative == location or relative.startswith(location + "/")
            if inside and importlib.util.find_spec(module_name) is None:
                return module_name
    return None


class _UninstalledExtraModule(pytest.Module):  # pylint: disable=too-many-ancestors
    """A test module reported as skipped without ever being imported."""

    missing = ""

    def collect(self):
        pytest.skip(
            f"requires the [streamlit] extra ({self.missing} is not installed)",
            allow_module_level=True,
        )


def pytest_pycollect_makemodule(module_path, parent):
    """Skip streamlit-dependent test modules cleanly when the extra is absent.

    This hook runs for every test file pytest collects, including a file named
    directly on the command line, which ``pytest_ignore_collect`` never sees.
    """
    missing = _missing_requirement(module_path)
    if missing is None:
        return None
    module = _UninstalledExtraModule.from_parent(parent, path=module_path)
    module.missing = missing
    return module
