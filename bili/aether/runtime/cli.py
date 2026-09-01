#!/usr/bin/env python
"""CLI tool for executing AETHER MAS configurations.

Usage (from project root)::

    # Basic execution
    python bili/aether/runtime/cli.py configs/simple_chain.yaml \\
        --input "Test content to review"

    # With input file
    python bili/aether/runtime/cli.py configs/production.yaml \\
        --input-file input.txt --log-dir my_logs

    # With an image attached to the input turn (repeat --input-image for more)
    python bili/aether/runtime/cli.py configs/vision.yaml \\
        --input "What does this diagram show?" --input-image diagram.png

    # Test checkpoint persistence (RQ1)
    python bili/aether/runtime/cli.py configs/production.yaml \\
        --input "Test payload" --test-checkpoint --thread-id test_001

    # Test cross-model transfer (RQ2)
    python bili/aether/runtime/cli.py configs/production.yaml \\
        --input "Test payload" --test-cross-model \\
        --source-model gpt-4 --target-model claude-sonnet-3-5-20241022
"""

import argparse
import os
import sys
import types


def _ensure_bili_stub() -> None:
    """Stub the top-level ``bili`` package if it hasn't been loaded yet.

    Same approach as ``bili/aether/tests/run_tests.py`` and
    ``bili/aether/compiler/cli.py`` — avoids pulling in heavy
    dependencies (firebase_admin, torch, etc.) from ``bili/__init__.py``.
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="aether-run",
        description="Execute an AETHER MAS configuration end-to-end.",
    )
    parser.add_argument(
        "config_file",
        help="Path to YAML MAS configuration file",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_text",
        help="Input text for the MAS",
    )
    parser.add_argument(
        "--input-file",
        help="Path to a text file containing input",
    )
    parser.add_argument(
        "--input-image",
        action="append",
        default=None,
        metavar="PATH_OR_URL",
        help=(
            "Attach an image to the input turn as a multimodal content part. "
            "Accepts a local file path or an http(s)/data URL. Repeatable. "
            "Requires the MAS agents to be bound to image-capable models."
        ),
    )
    parser.add_argument(
        "--thread-id",
        help="Thread ID for checkpointed execution",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for logs and result files (default: current dir)",
    )
    parser.add_argument(
        "--test-checkpoint",
        action="store_true",
        help="Run checkpoint persistence test (save + restore)",
    )
    parser.add_argument(
        "--test-cross-model",
        action="store_true",
        help="Run cross-model transfer test",
    )
    parser.add_argument(
        "--source-model",
        help="Source model for cross-model test (e.g. gpt-4)",
    )
    parser.add_argument(
        "--target-model",
        help="Target model for cross-model test (e.g. claude-sonnet-3-5-20241022)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to file",
    )
    return parser


def _build_input_data(args: argparse.Namespace) -> dict:
    """Construct initial state from CLI arguments.

    Returns a dict that can be passed to ``MASExecutor.run()``.
    Contains a ``messages`` key with a ``HumanMessage`` when input text or at
    least one ``--input-image`` is provided, otherwise returns an empty dict.

    With text only, the message is byte-for-byte the one this function has
    always built.  Each ``--input-image`` becomes an image content part, so the
    turn reaches the model as a multimodal message.
    """
    text = None

    if args.input_text:
        text = args.input_text
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            text = fh.read()

    images = getattr(args, "input_image", None) or []

    if text is None and not images:
        return {}

    from bili.iris.multimodal import (  # pylint: disable=import-outside-toplevel
        build_human_message,
    )

    return {
        "messages": [
            build_human_message(text=text, images=[_as_image(i) for i in images])
        ]
    }


def _as_image(reference: str) -> dict:
    """Turn one ``--input-image`` value into an image content part.

    A value carrying a URL scheme is passed through as a URL; anything else is
    read from disk and inlined as a ``data:`` URI, so the CLI works against a
    local screenshot without the caller hosting it first.
    """
    from bili.iris.multimodal import (  # pylint: disable=import-outside-toplevel
        image_part,
        image_part_from_path,
    )

    if reference.startswith(("http://", "https://", "data:")):
        return image_part(url=reference)
    return image_part_from_path(reference)


def main() -> None:
    """Entry point for the AETHER runtime CLI."""
    args = _build_parser().parse_args()

    # pylint: disable=import-outside-toplevel
    from bili.aether.config.loader import load_mas_from_yaml
    from bili.aether.runtime.executor import MASExecutor

    config = load_mas_from_yaml(args.config_file)
    executor = MASExecutor(config, log_dir=args.log_dir)
    executor.initialize()

    input_data = _build_input_data(args)

    if args.test_cross_model:
        if not args.source_model or not args.target_model:
            print(
                "Error: --test-cross-model requires "
                "--source-model and --target-model",
                file=sys.stderr,
            )
            sys.exit(1)

        source_result, target_result = executor.run_cross_model_test(
            input_data=input_data,
            source_model=args.source_model,
            target_model=args.target_model,
            thread_id=args.thread_id,
        )
        print("=== Source Model Result ===")
        print(source_result.get_formatted_output())
        print()
        print("=== Target Model Result ===")
        print(target_result.get_formatted_output())
        success = source_result.success and target_result.success

    elif args.test_checkpoint:
        original, restored = executor.run_with_checkpoint_persistence(
            input_data=input_data,
            thread_id=args.thread_id,
        )
        print("=== Original Run ===")
        print(original.get_formatted_output())
        print()
        print("=== Restored Run ===")
        print(restored.get_formatted_output())
        success = original.success and restored.success

    else:
        result = executor.run(
            input_data=input_data,
            thread_id=args.thread_id,
            save_results=not args.no_save,
        )
        print(result.get_formatted_output())
        success = result.success

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    _ensure_bili_stub()
    main()
