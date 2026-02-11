#!/usr/bin/env python
"""Complete AETHER workflow example.

Demonstrates the full MAS lifecycle from configuration through
execution, including checkpoint persistence and cross-model transfer
testing.

All agents use stub mode (no ``model_name``), so this example runs
without any LLM API keys or external services.

Usage (from project root)::

    python bili/aether/examples/complete_aether_workflow.py

CLI usage examples (after running this script to understand the flow)::

    # Basic execution from YAML
    python bili/aether/runtime/cli.py \\
        bili/aether/config/examples/simple_chain.yaml \\
        --input "Test content" --no-save

    # Checkpoint persistence test
    python bili/aether/runtime/cli.py \\
        bili/aether/config/examples/simple_chain.yaml \\
        --input "Test payload" --test-checkpoint --thread-id demo_001

    # Cross-model transfer test (requires API keys for real models)
    python bili/aether/runtime/cli.py \\
        bili/aether/config/examples/simple_chain.yaml \\
        --input "Test payload" --test-cross-model \\
        --source-model gpt-4 --target-model claude-sonnet-3-5-20241022
"""

import os
import sys
import tempfile
import types


def _ensure_bili_stub() -> None:
    """Stub the top-level ``bili`` package to avoid heavy dependencies."""
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


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run the complete AETHER workflow demonstration."""
    # pylint: disable=import-outside-toplevel
    from bili.aether.config.loader import load_mas_from_yaml
    from bili.aether.runtime.executor import MASExecutor, execute_mas
    from bili.aether.schema import AgentSpec, MASConfig, WorkflowType

    print("=" * 60)
    print("AETHER Complete Workflow Example")
    print("=" * 60)

    # ==================================================================
    # Part 1: Load from YAML and execute
    # ==================================================================
    print("\n--- Part 1: Execute from YAML config ---\n")

    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "examples",
    )
    yaml_path = os.path.join(examples_dir, "simple_chain.yaml")

    config = load_mas_from_yaml(yaml_path)
    print(f"Loaded: {config}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = MASExecutor(config, log_dir=tmp_dir)
        executor.initialize()
        result = executor.run(save_results=True)

        print(result.get_summary())
        print()

    # ==================================================================
    # Part 2: Programmatic config + execute_mas() convenience
    # ==================================================================
    print("\n--- Part 2: Programmatic config ---\n")

    config = MASConfig(
        mas_id="demo_pipeline",
        name="Demo 3-Agent Pipeline",
        description="Programmatically configured MAS for demonstration",
        workflow_type=WorkflowType.SEQUENTIAL,
        checkpoint_enabled=False,
        agents=[
            AgentSpec(
                agent_id="writer",
                role="content_writer",
                objective="Generate draft content for review",
            ),
            AgentSpec(
                agent_id="reviewer",
                role="content_reviewer",
                objective="Review content for quality and accuracy",
            ),
            AgentSpec(
                agent_id="editor",
                role="editor",
                objective="Make final editorial decisions on content",
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = execute_mas(config, log_dir=tmp_dir)
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Agents executed: {len(result.agent_results)}")
        for agent_result in result.agent_results:
            print(
                f"  {agent_result.agent_id}: {agent_result.output.get('status', 'N/A')}"
            )
        print()

    # ==================================================================
    # Part 3: Checkpoint persistence test
    # ==================================================================
    print("\n--- Part 3: Checkpoint persistence ---\n")

    config = MASConfig(
        mas_id="checkpoint_demo",
        name="Checkpoint Demo",
        workflow_type=WorkflowType.SEQUENTIAL,
        checkpoint_enabled=True,
        agents=[
            AgentSpec(
                agent_id="agent_a",
                role="processor",
                objective="Process input data for downstream agents",
            ),
            AgentSpec(
                agent_id="agent_b",
                role="validator",
                objective="Validate processed data meets requirements",
            ),
        ],
    )

    executor = MASExecutor(config)
    executor.initialize()

    original, restored = executor.run_with_checkpoint_persistence(
        thread_id="demo_checkpoint"
    )

    print(f"Original run:  {'SUCCESS' if original.success else 'FAILED'}")
    print(f"Restored run:  {'SUCCESS' if restored.success else 'FAILED'}")
    print(f"Original agents: {len(original.agent_results)}")
    print(f"Restored agents: {len(restored.agent_results)}")
    print()

    # ==================================================================
    # Part 4: Cross-model transfer test (stubs — no API keys)
    # ==================================================================
    print("\n--- Part 4: Cross-model transfer (stubs) ---\n")

    config = MASConfig(
        mas_id="cross_model_demo",
        name="Cross-Model Demo",
        workflow_type=WorkflowType.SEQUENTIAL,
        checkpoint_enabled=False,
        agents=[
            AgentSpec(
                agent_id="agent_x",
                role="analyst",
                objective="Analyze data using assigned model configuration",
            ),
            AgentSpec(
                agent_id="agent_y",
                role="reporter",
                objective="Generate reports based on analysis results",
            ),
        ],
    )

    executor = MASExecutor(config)
    executor.initialize()

    source_result, target_result = executor.run_cross_model_test(
        source_model=None,  # Stub mode
        target_model=None,  # Stub mode
    )

    print(f"Source run:  {'SUCCESS' if source_result.success else 'FAILED'}")
    print(f"Target run:  {'SUCCESS' if target_result.success else 'FAILED'}")
    print()

    # ==================================================================
    # Part 5: Result inspection
    # ==================================================================
    print("\n--- Part 5: Formatted output ---\n")

    print(result.get_formatted_output())

    print("\n--- Result as dict (first 3 keys) ---\n")
    result_dict = result.to_dict()
    for key in list(result_dict.keys())[:3]:
        print(f"  {key}: {result_dict[key]}")

    print("\n" + "=" * 60)
    print("AETHER workflow demonstration complete.")
    print("=" * 60)


if __name__ == "__main__":
    _ensure_bili_stub()
    main()
