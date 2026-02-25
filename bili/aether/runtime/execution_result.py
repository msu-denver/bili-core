"""Structured execution result containers for MAS runs.

Provides ``AgentExecutionResult`` and ``MASExecutionResult`` dataclasses
for capturing per-agent outputs, timing, communication statistics, and
checkpoint metadata from a MAS execution.

Follows the ``@dataclass`` pattern established by
``bili.aether.validation.result.ValidationResult``.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

_BORDER_CHAR = "*"
_BORDER_WIDTH = 60


@dataclass
class AgentExecutionResult:  # pylint: disable=too-many-instance-attributes
    """Result of a single agent's execution within a MAS run.

    Attributes:
        agent_id: The agent's unique identifier.
        role: The agent's role string.
        output: The agent's output dict from ``agent_outputs`` state.
        execution_time_ms: Wall-clock execution time in milliseconds.
        error: Error message if the agent failed, ``None`` otherwise.
        tools_used: List of tool names invoked (empty for stub agents).
        messages_sent: Count of messages sent by this agent.
        messages_received: Count of messages received by this agent.
    """

    agent_id: str
    role: str
    output: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    messages_sent: int = 0
    messages_received: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "output": self.output,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "tools_used": self.tools_used,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


@dataclass
class MASExecutionResult:  # pylint: disable=too-many-instance-attributes
    """Complete result of a MAS execution run.

    Attributes:
        mas_id: The MAS identifier.
        execution_id: Unique ID for this execution run.
        start_time: ISO-8601 UTC start timestamp.
        end_time: ISO-8601 UTC end timestamp.
        total_execution_time_ms: Total wall-clock time in milliseconds.
        agent_results: List of per-agent results.
        final_state: The final LangGraph state dict (serialized).
        total_messages: Total inter-agent messages exchanged.
        messages_by_channel: Message counts keyed by channel_id.
        communication_log_path: Path to the JSONL communication log file.
        checkpoint_saved: Whether a checkpoint was saved.
        checkpoint_path: Path or identifier of saved checkpoint.
        success: Whether execution completed without errors.
        error: Error message if execution failed.
        metadata: Additional execution metadata.
    """

    mas_id: str
    execution_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = ""
    end_time: str = ""
    total_execution_time_ms: float = 0.0
    agent_results: List[AgentExecutionResult] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    total_messages: int = 0
    messages_by_channel: Dict[str, int] = field(default_factory=dict)
    communication_log_path: Optional[str] = None
    checkpoint_saved: bool = False
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if no error occurred. Computed from error field."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mas_id": self.mas_id,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_execution_time_ms": self.total_execution_time_ms,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "final_state": self.final_state,
            "total_messages": self.total_messages,
            "messages_by_channel": self.messages_by_channel,
            "communication_log_path": self.communication_log_path,
            "checkpoint_saved": self.checkpoint_saved,
            "checkpoint_path": self.checkpoint_path,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    def save_to_file(self, path: str) -> None:
        """Write the result to a JSON file.

        Args:
            path: Filesystem path for the output JSON file.
        """
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, default=str)
            LOGGER.info("Execution result saved to %s", path)
        except OSError as exc:
            LOGGER.error("Failed to save execution result: %s", exc)

    def get_summary(self) -> str:
        """Return a concise human-readable summary of the execution."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "MAS Execution Summary",
            f"  MAS ID:        {self.mas_id}",
            f"  Execution ID:  {self.execution_id}",
            f"  Status:        {status}",
            f"  Duration:      {self.total_execution_time_ms:.2f} ms",
            f"  Agents:        {len(self.agent_results)}",
            f"  Messages:      {self.total_messages}",
        ]
        if self.checkpoint_saved:
            lines.append("  Checkpoint:    saved")
        if self.error:
            lines.append(f"  Error:         {self.error}")

        for agent_result in self.agent_results:
            agent_status = "ERROR" if agent_result.error else "OK"
            msg = agent_result.output.get("message", "")
            preview = msg[:60] + "..." if len(msg) > 60 else msg
            lines.append(
                f"  [{agent_status}] {agent_result.agent_id} "
                f"({agent_result.role}): {preview}"
            )

        return "\n".join(lines)

    def get_formatted_output(self) -> str:
        """Return formatted output with asterisk borders."""
        border = _BORDER_CHAR * _BORDER_WIDTH
        status = "SUCCESS" if self.success else "FAILED"

        inner_lines = [
            "MAS Execution Result",
            "",
            f"MAS ID:        {self.mas_id}",
            f"Execution ID:  {self.execution_id}",
            f"Status:        {status}",
            f"Duration:      {self.total_execution_time_ms:.2f} ms",
            f"Agents:        {len(self.agent_results)}",
            f"Messages:      {self.total_messages}",
        ]

        if self.checkpoint_saved:
            inner_lines.append("Checkpoint:    saved")
        if self.error:
            inner_lines.append(f"Error:         {self.error}")

        inner_lines.append("")
        inner_lines.append("Agent Outputs:")

        for agent_result in self.agent_results:
            inner_lines.append(f"  {agent_result.agent_id} ({agent_result.role}):")
            msg = agent_result.output.get("message", "(no output)")
            for line in msg.split("\n"):
                inner_lines.append(f"    {line}")

        # Build bordered output
        content_width = _BORDER_WIDTH - 4  # "* " prefix + " *" suffix
        result_lines = [border]
        for line in inner_lines:
            if len(line) > content_width:
                line = line[:content_width]
            padded = line.ljust(content_width)
            result_lines.append(f"{_BORDER_CHAR} {padded} {_BORDER_CHAR}")
        result_lines.append(border)

        return "\n".join(result_lines)
