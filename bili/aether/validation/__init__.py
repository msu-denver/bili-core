"""
Static validation engine for MAS configurations.

Provides deep structural validation beyond Pydantic's basic field validation.
Catches configuration errors before execution with clear, actionable messages.

Usage:
    >>> from bili.aether.validation import validate_mas
    >>> from bili.aether import load_mas_from_yaml
    >>>
    >>> config = load_mas_from_yaml("my_mas.yaml")
    >>> result = validate_mas(config)
    >>>
    >>> if not result:
    ...     print(result)
"""

from .result import ValidationResult
from .validator import MASValidator, validate_mas

__all__ = [
    "ValidationResult",
    "MASValidator",
    "validate_mas",
]
