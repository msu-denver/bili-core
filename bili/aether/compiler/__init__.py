"""AETHER-to-LangGraph compiler.

Converts a validated ``MASConfig`` into an executable LangGraph
``StateGraph`` with LLM-backed agent nodes.

When ``AgentSpec.model_name`` is set, agents make real LLM calls via
``bili.loaders.llm_loader``.  When ``model_name`` is ``None``, agents
use a lightweight stub that emits placeholder messages.

Usage::

    from bili.aether.compiler import compile_mas
    from bili.aether.config.loader import load_mas_from_yaml

    config = load_mas_from_yaml("simple_chain.yaml")
    compiled = compile_mas(config)
    graph = compiled.compile_graph()
"""

from bili.aether.schema import MASConfig
from bili.aether.validation import validate_mas

from .agent_generator import generate_agent_node, wrap_agent_node
from .compiled_mas import CompiledMAS
from .graph_builder import GraphBuilder
from .llm_resolver import create_llm, resolve_model, resolve_provider, resolve_tools
from .state_generator import generate_state_schema

__all__ = [
    "CompiledMAS",
    "GraphBuilder",
    "compile_mas",
    "create_llm",
    "generate_agent_node",
    "generate_state_schema",
    "resolve_model",
    "resolve_provider",
    "resolve_tools",
    "wrap_agent_node",
]


def compile_mas(
    config: MASConfig,
    custom_node_registry: dict | None = None,
    runtime_context: "Any | None" = None,
) -> CompiledMAS:
    """Validate and compile a ``MASConfig`` into a ``CompiledMAS``.

    Runs the static validation engine first.  If validation produces
    **errors** (not warnings), raises ``ValueError`` with the full
    validation report.

    Args:
        config: A ``MASConfig`` instance.
        custom_node_registry: Optional mapping of node names to factory
            callables.  Checked before the global
            ``GRAPH_NODE_REGISTRY`` when resolving pipeline
            ``node_type`` references.
        runtime_context: Optional :class:`RuntimeContext` holding named
            services injected into pipeline node builders as ``**kwargs``
            (lowest priority — overridden by parent agent config and
            node-specific config).

    Returns:
        A :class:`CompiledMAS` containing the uncompiled ``StateGraph``,
        state schema, and agent node callables.

    Raises:
        ValueError: If validation fails with errors.
    """
    result = validate_mas(config)
    if not result.valid:
        raise ValueError(f"MAS validation failed:\n{result}")

    return GraphBuilder(
        config,
        custom_node_registry=custom_node_registry,
        runtime_context=runtime_context,
    ).build()
