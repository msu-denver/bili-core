"""Structured JSON attack event logger for the Attack Injection Framework.

Writes ``AttackResult`` records to a newline-delimited JSON (NDJSON) log file.
Each line is an independently parseable JSON object — one per injection event.

The log file is **append-only**; it is never truncated or overwritten.  This
preserves the full research audit trail across multiple sessions.

Default log path: ``bili/aether/attacks/logs/attack_log.ndjson``

Usage::

    from pathlib import Path
    from bili.aether.attacks.logger import AttackLogger

    logger = AttackLogger()
    logger.log(result)          # appends one line to the default path

    # Or use a custom path:
    logger = AttackLogger(log_path=Path("/tmp/research_log.ndjson"))
"""

import json
import logging
import threading
from pathlib import Path

from bili.aether.attacks.models import AttackResult

LOGGER = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "attack_log.ndjson"


class AttackLogger:  # pylint: disable=too-few-public-methods
    """Append ``AttackResult`` records to a newline-delimited JSON log file.

    Args:
        log_path: Path to the NDJSON log file.  Parent directories are
            created on first write.  Defaults to
            ``bili/aether/attacks/logs/attack_log.ndjson``.
    """

    DEFAULT_PATH: Path = _DEFAULT_LOG_PATH

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path: Path = log_path if log_path is not None else _DEFAULT_LOG_PATH
        self._lock = threading.Lock()

    def log(self, result: AttackResult) -> None:
        """Append *result* as a single JSON line to the log file.

        The file and any missing parent directories are created if they do
        not yet exist.  The file is opened in append mode (``"a"``) so
        existing records are never overwritten.

        Args:
            result: The completed ``AttackResult`` to persist.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(result.model_dump(mode="json"), default=str)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        LOGGER.debug(
            "AttackLogger: appended attack_id=%s to %s",
            result.attack_id,
            self._log_path,
        )
