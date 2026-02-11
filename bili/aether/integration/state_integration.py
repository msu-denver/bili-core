"""State schema extensions for bili-core inheritance.

Adds additional TypedDict annotation fields when agents use bili-core
inheritance.  Currently minimal -- extended as inheritance features grow.
"""

import logging
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


def get_inheritance_state_fields(
    agents: List[Any],
) -> Dict[str, Any]:
    """Return additional state annotation fields for inheritance.

    Args:
        agents: List of ``AgentSpec`` instances from the MAS config.

    Returns:
        A dict of ``{field_name: type_annotation}`` to merge into the
        state schema.  Empty if no agents use inheritance.
    """
    has_inheritance = any(getattr(a, "inherit_from_bili_core", False) for a in agents)
    if not has_inheritance:
        return {}

    # Extension point: add fields like user_context, memory_state
    # as inheritance features expand.
    return {}
