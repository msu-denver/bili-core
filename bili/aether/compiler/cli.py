#!/usr/bin/env python
"""CLI tool for testing AETHER MAS compilation.

Usage (from project root)::

    python bili/aether/compiler/cli.py                    # compile all examples
    python bili/aether/compiler/cli.py path/to/mas.yaml   # compile specific file
"""

import os
import sys
import types


def _ensure_bili_stub() -> None:
    """Stub the top-level ``bili`` package if it hasn't been loaded yet.

    Same approach as ``bili/aether/tests/run_tests.py`` — avoids pulling
    in heavy dependencies (firebase_admin, torch, etc.) from ``bili/__init__.py``.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if "bili" not in sys.modules or not hasattr(sys.modules["bili"], "__path__"):
        stub = types.ModuleType("bili")
        stub.__path__ = [os.path.join(project_root, "bili")]
        stub.__package__ = "bili"
        sys.modules["bili"] = stub


def main() -> None:
    """Compile AETHER MAS configurations and report results."""
    # pylint: disable=import-outside-toplevel
    from bili.aether.compiler import compile_mas
    from bili.aether.config.loader import load_mas_from_yaml

    if len(sys.argv) > 1:
        path = sys.argv[1]
        config = load_mas_from_yaml(path)
        result = compile_mas(config)
        print(f"OK    {path}")
        print(f"      {result}")
        compiled = result.compile_graph()
        print(f"      Compiled: {type(compiled).__name__}")
        return

    # Compile all examples
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "examples",
    )

    example_files = [
        "simple_chain.yaml",
        "hierarchical_voting.yaml",
        "supervisor_moderation.yaml",
        "consensus_network.yaml",
        "custom_escalation.yaml",
        "research_analysis.yaml",
        "code_review.yaml",
    ]

    failures = 0
    for fname in example_files:
        fpath = os.path.join(examples_dir, fname)
        if not os.path.exists(fpath):
            print(f"SKIP  {fname}  (file not found)")
            continue
        try:
            config = load_mas_from_yaml(fpath)
            result = compile_mas(config)
            result.compile_graph()
            print(f"OK    {fname}  ->  {result}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"FAIL  {fname}  ->  {exc}")
            failures += 1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _ensure_bili_stub()
    main()
