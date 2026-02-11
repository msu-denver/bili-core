"""JSONL communication logger for inter-agent messages.

Writes one JSON object per line to a log file, enabling easy post-hoc
analysis of agent communication patterns.
"""

import json
import logging
from typing import IO, Optional

from bili.aether.runtime.messages import Message

LOGGER = logging.getLogger(__name__)


class CommunicationLogger:
    """Appends ``Message`` objects as JSONL to a file.

    Supports the context-manager protocol so it can be used as::

        with CommunicationLogger("run.jsonl") as logger:
            logger.log_message(msg)

    Attributes:
        log_path: Filesystem path to the JSONL output file.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._file_handle: Optional[IO[str]] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CommunicationLogger":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_message(self, message: Message) -> None:
        """Write a single message as a JSONL line."""
        if self._file_handle is None:
            self._open()
        line = json.dumps(message.to_log_dict(), default=str)
        self._file_handle.write(line + "\n")  # type: ignore[union-attr]

    def flush(self) -> None:
        """Flush the underlying file buffer."""
        if self._file_handle is not None:
            self._file_handle.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except OSError:
                LOGGER.warning("Failed to close communication log: %s", self.log_path)
            finally:
                self._file_handle = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the log file for appending (creates if missing)."""
        if self._file_handle is None:
            self._file_handle = (
                open(  # noqa: SIM115  pylint: disable=consider-using-with
                    self.log_path, "a", encoding="utf-8"
                )
            )
