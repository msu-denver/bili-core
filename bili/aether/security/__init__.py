"""AETHER Security Event Detection & Logging subpackage.

Provides programmatic detection and structured logging of security events
derived from attack injection results (``AttackResult`` from Task 13).

Quick start::

    from pathlib import Path
    from bili.aether.security import (
        SecurityEventDetector,
        SecurityEventLogger,
    )

    logger = SecurityEventLogger(log_path=Path("security_events.ndjson"))
    detector = SecurityEventDetector(logger=logger)

    # After inject_attack():
    events = detector.detect(attack_result)

    # Export all events as a JSON array:
    json_str = logger.export_json()
"""

from bili.aether.security.detector import SecurityEventDetector
from bili.aether.security.logger import SecurityEventLogger
from bili.aether.security.models import SecurityEvent, SecurityEventType

__all__ = [
    "SecurityEvent",
    "SecurityEventDetector",
    "SecurityEventLogger",
    "SecurityEventType",
]
