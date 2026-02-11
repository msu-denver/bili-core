"""Static validation engine for multi-agent system configurations.

Performs deep structural and cross-field validation on a ``MASConfig``
instance that has already passed Pydantic's basic field validation.
"""

from typing import Dict, List, Set

from bili.aether.schema import MASConfig, WorkflowType

from .result import ValidationResult


class MASValidator:
    """
    Static validator for MAS configurations.

    Checks for structural issues (circular dependencies, orphaned agents,
    duplicate channels, workflow-specific constraints) that go beyond what
    Pydantic model validators cover.

    Usage:
        >>> validator = MASValidator()
        >>> result = validator.validate(config)
        >>> if not result:
        ...     print(result)
    """

    def __init__(self) -> None:
        self._config: MASConfig = None  # type: ignore[assignment]
        self._result: ValidationResult = None  # type: ignore[assignment]
        self._agent_ids: Set[str] = set()
        self._channel_agents: Set[str] = set()
        self._edge_graph: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, config: MASConfig) -> ValidationResult:
        """Validate *config* and return a :class:`ValidationResult`."""
        self._config = config
        self._result = ValidationResult()
        self._build_caches()

        self._check_agents()
        self._check_channels()
        self._check_workflow_graph()
        self._check_workflow_specific()

        return self._result

    # ------------------------------------------------------------------
    # Cache construction
    # ------------------------------------------------------------------

    def _build_caches(self) -> None:
        self._agent_ids = {a.agent_id for a in self._config.agents}

        self._channel_agents = set()
        for channel in self._config.channels:
            self._channel_agents.add(channel.source)
            self._channel_agents.add(channel.target)

        self._edge_graph = {}
        for edge in self._config.workflow_edges:
            self._edge_graph.setdefault(edge.from_agent, []).append(edge.to_agent)

    # ==================================================================
    # Agent checks
    # ==================================================================

    def _check_agents(self) -> None:
        self._check_orphaned_agents()
        self._check_supervisor_capabilities()

    def _check_orphaned_agents(self) -> None:
        """W1: Warn about agents with no channel connections."""
        for agent in self._config.agents:
            if agent.agent_id not in self._channel_agents:
                self._result.add_warning(
                    f"Agent '{agent.agent_id}' has no channel connections"
                )

    def _check_supervisor_capabilities(self) -> None:
        """W2: Warn if supervisor agent lacks inter_agent_communication."""
        for agent in self._config.agents:
            if agent.is_supervisor:
                if "inter_agent_communication" not in agent.capabilities:
                    self._result.add_warning(
                        f"Supervisor agent '{agent.agent_id}' should have "
                        f"'inter_agent_communication' capability"
                    )

    # ==================================================================
    # Channel checks
    # ==================================================================

    def _check_channels(self) -> None:
        self._check_duplicate_channels()
        self._check_bidirectional_conflicts()

    def _check_duplicate_channels(self) -> None:
        """E1: Error on duplicate channels (same source+target+protocol)."""
        seen: Set[tuple] = set()
        for channel in self._config.channels:
            key = (channel.source, channel.target, channel.protocol)
            if key in seen:
                self._result.add_error(
                    f"Duplicate channel: {channel.source} -> "
                    f"{channel.target} ({channel.protocol.value})"
                )
            seen.add(key)

    def _check_bidirectional_conflicts(self) -> None:
        """W3: Warn if bidirectional channel has a separate reverse channel."""
        for channel in self._config.channels:
            if not channel.bidirectional:
                continue
            for other in self._config.channels:
                if other.channel_id == channel.channel_id:
                    continue
                if (
                    other.source == channel.target
                    and other.target == channel.source
                    and other.protocol == channel.protocol
                ):
                    self._result.add_warning(
                        f"Bidirectional channel '{channel.channel_id}' "
                        f"({channel.source} <-> {channel.target}) has a "
                        f"separate reverse channel '{other.channel_id}'"
                    )

    # ==================================================================
    # Workflow graph checks
    # ==================================================================

    def _check_workflow_graph(self) -> None:
        if self._config.workflow_type == WorkflowType.SEQUENTIAL:
            self._check_sequential_chain()
            self._check_circular_dependencies()

        if self._config.workflow_type == WorkflowType.CUSTOM:
            self._check_unreachable_agents()
            self._check_path_to_end()

    def _check_sequential_chain(self) -> None:
        """W4: Warn if SEQUENTIAL workflow is not a linear chain."""
        if not self._config.workflow_edges:
            return

        outgoing_count: Dict[str, int] = {}
        for edge in self._config.workflow_edges:
            outgoing_count[edge.from_agent] = outgoing_count.get(edge.from_agent, 0) + 1

        for agent_id, count in outgoing_count.items():
            if count > 1:
                self._result.add_warning(
                    f"SEQUENTIAL workflow: agent '{agent_id}' has {count} "
                    f"outgoing edges (expected 1 for a linear chain)"
                )

    def _check_circular_dependencies(self) -> None:
        """E2: Error on cycles in SEQUENTIAL workflows (DFS).

        Checks all nodes in the graph, including disconnected components,
        to catch cycles that may not be reachable from the entry point.
        """
        if not self._config.workflow_edges:
            return

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._edge_graph.get(node, []):
                if neighbor == "END":
                    continue
                if neighbor not in visited:
                    if _has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        # Check from entry point first
        entry = self._config.entry_point or self._config.agents[0].agent_id
        if _has_cycle(entry):
            self._result.add_error(
                f"Circular dependency detected in SEQUENTIAL workflow "
                f"starting from '{entry}'"
            )

        # Check any remaining unvisited nodes (disconnected cycles)
        for agent in self._config.agents:
            if agent.agent_id not in visited:
                if _has_cycle(agent.agent_id):
                    self._result.add_error(
                        f"Circular dependency detected in disconnected "
                        f"component starting from '{agent.agent_id}'"
                    )

    def _check_unreachable_agents(self) -> None:
        """W5: Warn about agents unreachable from entry point (BFS)."""
        if not self._config.workflow_edges:
            return

        entry = self._config.entry_point or self._config.agents[0].agent_id
        reachable: Set[str] = {entry}
        queue = [entry]

        while queue:
            current = queue.pop(0)
            for neighbor in self._edge_graph.get(current, []):
                if neighbor != "END" and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        for agent in self._config.agents:
            if agent.agent_id not in reachable:
                self._result.add_warning(
                    f"Agent '{agent.agent_id}' is unreachable from "
                    f"entry point '{entry}' in CUSTOM workflow"
                )

    def _check_path_to_end(self) -> None:
        """W6: Warn if CUSTOM workflow has no path to END from entry.

        Uses BFS to check if END is reachable from the entry point,
        not just whether any edge targets END (which could be disconnected).
        """
        if not self._config.workflow_edges:
            return

        entry = self._config.entry_point or self._config.agents[0].agent_id
        visited: Set[str] = {entry}
        queue = [entry]
        found_end = False

        # BFS to find if END is reachable from entry
        while queue:
            current = queue.pop(0)
            for neighbor in self._edge_graph.get(current, []):
                if neighbor == "END":
                    found_end = True
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
            if found_end:
                break

        if not found_end:
            self._result.add_warning(
                f"CUSTOM workflow has no path to 'END' from entry point "
                f"'{entry}' - workflow may not terminate"
            )

    # ==================================================================
    # Workflow-specific checks
    # ==================================================================

    def _check_workflow_specific(self) -> None:
        wtype = self._config.workflow_type

        if wtype == WorkflowType.CONSENSUS:
            self._check_consensus_workflow()

        if wtype == WorkflowType.HIERARCHICAL or self._config.hierarchical_voting:
            self._check_hierarchical_workflow()

        if wtype == WorkflowType.SUPERVISOR:
            self._check_supervisor_workflow()

        if wtype == WorkflowType.CUSTOM and self._config.human_in_loop:
            self._check_human_in_loop()

    def _check_consensus_workflow(self) -> None:
        """W7: Warn if CONSENSUS agents lack consensus_vote_field."""
        for agent in self._config.agents:
            if not agent.consensus_vote_field:
                self._result.add_warning(
                    f"CONSENSUS workflow: agent '{agent.agent_id}' missing "
                    f"'consensus_vote_field'"
                )

    def _check_hierarchical_workflow(self) -> None:
        """E3 + W8: Check hierarchical tier structure."""
        tiers = sorted({a.tier for a in self._config.agents if a.tier is not None})
        if not tiers:
            return

        # E3: Must have tier 1
        if 1 not in tiers:
            self._result.add_error(
                "HIERARCHICAL workflow: no agents at tier 1 " "(top of hierarchy)"
            )

        # W8: Detect tier gaps
        for i in range(len(tiers) - 1):
            if tiers[i + 1] - tiers[i] > 1:
                self._result.add_warning(
                    f"HIERARCHICAL workflow: tier gap detected — "
                    f"has tier {tiers[i]} and {tiers[i + 1]} "
                    f"but missing tier {tiers[i] + 1}"
                )

    def _check_supervisor_workflow(self) -> None:
        """W9: Warn if entry point agent is not marked is_supervisor."""
        entry_id = self._config.entry_point or self._config.agents[0].agent_id
        entry_agent = self._config.get_agent(entry_id)
        if entry_agent and not entry_agent.is_supervisor:
            self._result.add_warning(
                f"SUPERVISOR workflow: entry point agent '{entry_id}' "
                f"should have 'is_supervisor=true'"
            )

    def _check_human_in_loop(self) -> None:
        """W10: Warn if human_in_loop without escalation condition."""
        if not self._config.human_escalation_condition:
            self._result.add_warning(
                "human_in_loop is true but no " "'human_escalation_condition' specified"
            )


def validate_mas(config: MASConfig) -> ValidationResult:
    """Convenience function: validate a MAS configuration.

    Args:
        config: A ``MASConfig`` instance (already Pydantic-validated).

    Returns:
        A :class:`ValidationResult` with errors and warnings.
    """
    return MASValidator().validate(config)
