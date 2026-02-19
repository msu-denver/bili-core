"""
Launcher for the AETHER MAS Visualizer.

This script bypasses bili/__init__.py (which imports heavy dependencies
like langgraph.prebuilt) so the lightweight AETHER UI can run standalone.

Usage:
    streamlit run run_aether_ui.py
"""

import sys
import types
from pathlib import Path

# Pre-register a lightweight 'bili' package before any imports trigger
# bili/__init__.py, which eagerly imports flask_api, loaders, nodes, etc.
_bili_path = str(Path(__file__).resolve().parent / "bili")
_bili_pkg = types.ModuleType("bili")
_bili_pkg.__path__ = [_bili_path]
_bili_pkg.__package__ = "bili"
sys.modules["bili"] = _bili_pkg

from bili.aether.ui.app import main  # noqa: E402

main()
