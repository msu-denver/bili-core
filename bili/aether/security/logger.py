"""Structured JSON security event logger for the AETHER Security system.

Writes ``SecurityEvent`` records to a newline-delimited JSON (NDJSON) log file.
Each line is an independently parseable JSON object — one per security event.

The log file is **append-only**; it is never truncated or overwritten.

Default log path: ``bili/aether/security/logs/security_events.ndjson``

Usage::

    from pathlib import Path
    from bili.aether.security.logger import SecurityEventLogger

    logger = SecurityEventLogger()
    logger.log(event)                       # appends one line to default path
    json_str = logger.export_json()         # returns full log as a JSON array

    # Custom path:
    logger = SecurityEventLogger(log_path=Path("/tmp/sec_events.ndjson"))
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from bili.aether.security.models import SecurityEvent

LOGGER = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "security_events.ndjson"


class SecurityEventLogger:
    """Append ``SecurityEvent`` records to a newline-delimited JSON log file.

    Args:
        log_path: Path to the NDJSON log file.  Parent directories are
            created on first write.  Defaults to
            ``bili/aether/security/logs/security_events.ndjson``.
    """

    DEFAULT_PATH: Path = _DEFAULT_LOG_PATH

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._log_path: Path = log_path if log_path is not None else _DEFAULT_LOG_PATH
        self._lock = threading.Lock()

    def log(self, event: SecurityEvent) -> None:
        """Append *event* as a single JSON line to the log file.

        The file and any missing parent directories are created if they do
        not yet exist.  The file is opened in append mode (``"a"``) so
        existing records are never overwritten.

        Args:
            event: The ``SecurityEvent`` to persist.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), default=str)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        LOGGER.debug(
            "SecurityEventLogger: appended event_id=%s type=%s to %s",
            event.event_id,
            event.event_type,
            self._log_path,
        )

    def export_json(self, path: Optional[Path] = None) -> str:
        """Export all security events as a JSON array string.

        Reads the NDJSON log file, parses each line, and returns the
        combined result as a pretty-printed JSON array.  Malformed lines
        are skipped with a warning.  Optionally writes the output to *path*.

        Args:
            path: Optional filesystem path to write the exported JSON array.
                Parent directories are created if needed.

        Returns:
            A JSON array string containing all logged security events.
        """
        events = []
        if self._log_path.exists():
            for raw_line in self._log_path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    LOGGER.warning(
                        "SecurityEventLogger.export_json: skipped malformed line: %s",
                        exc,
                    )

        result = json.dumps(events, indent=2, default=str)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result, encoding="utf-8")
            LOGGER.info(
                "SecurityEventLogger: exported %d events to %s", len(events), path
            )

        return result
