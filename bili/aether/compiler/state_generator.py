"""Dynamic state schema generation for compiled MAS graphs."""

import re
from typing import Any, Dict, List, Type

from bili.aether.schema import MASConfig, WorkflowType


def generate_state_schema(config: MASConfig) -> Type:
    """Generate a TypedDict state schema tailored to a MAS configuration.

    Base fields (always present):
        messages — LangGraph message list with ``add_messages`` reducer.
        current_agent — ID of the currently executing agent.
        agent_outputs — Mapping of agent_id → latest output dict.
        mas_id — The MAS identifier.

    Workflow-specific fields are added based on ``config.workflow_type``.

    Args:
        config: A validated ``MASConfig``.

    Returns:
        A ``TypedDict`` subclass suitable as a LangGraph ``state_schema``.
    """
    from langgraph.graph import (  # pylint: disable=import-error,import-outside-toplevel
        add_messages,
    )
    from typing_extensions import (  # pylint: disable=import-error,import-outside-toplevel
        Annotated,
        TypedDict,
    )

    annotations: Dict[str, Any] = {
        "messages": Annotated[list, add_messages],
        "current_agent": str,
        "agent_outputs": Dict[str, Any],
        "mas_id": str,
    }

    wtype = config.workflow_type

    if wtype == WorkflowType.CONSENSUS:
        annotations["current_round"] = int
        annotations["votes"] = Dict[str, str]
        annotations["consensus_reached"] = bool
        annotations["max_rounds"] = int

    if wtype == WorkflowType.HIERARCHICAL:
        annotations["current_tier"] = int
        annotations["tier_results"] = Dict[str, Any]

    if wtype == WorkflowType.SUPERVISOR:
        annotations["next_agent"] = str
        annotations["pending_tasks"] = List[str]
        annotations["completed_tasks"] = List[str]

    if wtype == WorkflowType.CUSTOM and config.human_in_loop:
        annotations["needs_human_review"] = bool

    # Communication fields (present when channels are configured)
    if config.channels:
        annotations["channel_messages"] = Dict[str, Any]
        annotations["pending_messages"] = Dict[str, list]
        annotations["communication_log"] = list

    # Inheritance state fields (when agents use bili-core inheritance)
    try:
        from bili.aether.integration.state_integration import (  # pylint: disable=import-outside-toplevel
            get_inheritance_state_fields,
        )

        annotations.update(get_inheritance_state_fields(config.agents))
    except ImportError:
        pass  # integration package not available

    # Build a valid Python identifier from the mas_id
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", config.mas_id)
    class_name = f"{safe_name}_State"

    state_class: Type = TypedDict(class_name, annotations)  # type: ignore[call-overload]
    return state_class
