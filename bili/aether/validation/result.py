"""Validation result container for MAS configuration validation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    """
    Result of MAS configuration validation.

    Attributes:
        valid: Computed property - True if no errors were found (warnings allowed).
        errors: Fatal issues that should block MAS execution.
        warnings: Non-fatal issues that should be reviewed.

    Examples:
        >>> result = ValidationResult()
        >>> result.add_warning("Agent 'foo' has no channel connections")
        >>> print(result.valid)
        True
        >>> result.add_error("Duplicate channel detected")
        >>> print(result.valid)
        False
        >>> print(result)
        Validation FAILED with 1 error(s)
        ...
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True if no errors present. Computed from error list."""
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        """Add a fatal error (automatically marks result as invalid)."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def __str__(self) -> str:
        """Pretty formatted validation report."""
        lines: List[str] = []

        if self.valid and not self.warnings:
            return "Validation passed: no errors or warnings."

        if not self.valid:
            lines.append(f"Validation FAILED with {len(self.errors)} error(s)")
            lines.append("")
            lines.append("ERRORS:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        else:
            lines.append("Validation PASSED (with warnings)")

        if self.warnings:
            lines.append("")
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        return "\n".join(lines)

    def __bool__(self) -> bool:
        """Allow boolean context: ``if result: ...``."""
        return self.valid
