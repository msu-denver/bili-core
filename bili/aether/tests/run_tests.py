#!/usr/bin/env python
"""
Standalone test runner for AETHER schema tests.

Stubs out the top-level ``bili`` package so that only ``bili.aether``
is loaded, avoiding heavy dependencies (firebase_admin, torch, etc.)
that are required by other bili subpackages.

Usage (from project root):
    python bili/aether/tests/run_tests.py          # run all aether tests
    python bili/aether/tests/run_tests.py -v        # verbose
    python bili/aether/tests/run_tests.py -k inherit # filter by keyword

WHEN TO REMOVE:
    This file can be deleted once one of the following is true:
    - bili/__init__.py uses lazy imports (e.g. importlib or __getattr__), OR
    - All dependencies in requirements.txt are installed in the test env, OR
    - AETHER tests are moved to a standalone package with its own import path.
"""

import os
import sys
import types

# ---------------------------------------------------------------------------
# Stub the top-level bili package BEFORE anything else imports it.
# This replaces bili/__init__.py (which eagerly imports auth, config, etc.)
# with a thin module that only exposes the filesystem path, so
# ``from bili.aether.schema import ...`` works without pulling in firebase.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

stub = types.ModuleType("bili")
stub.__path__ = [os.path.join(_project_root, "bili")]
stub.__package__ = "bili"
sys.modules["bili"] = stub

# ---------------------------------------------------------------------------
# Now import pytest and run.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402  (must come after stub)

_test_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Forward any extra CLI args (e.g. -v, -k, --tb=short)
    sys.exit(pytest.main([_test_dir] + sys.argv[1:]))
