"""
AEGIS-PROBE: Persistent Reasoning Open-ended Black-box Evaluator.

An autonomous adversarial agent suite for AEGIS that conducts multi-round,
adaptive red-teaming against AETHER multi-agent victim systems.

See bili/aegis/suites/probe/README.md for usage and
bili/aegis/docs/probe-rfc.md for the design.

Public surface:
- ProbeSession, ProbeTurn, ProbeOutcome, ProbeObjective (schema.py)
- AttackPolicy ABC and reference implementations (policies/)
- BudgetState (budget.py)
- AttackerMAS.run_session(), the per-session entry point driven by the
  suite runner (attacker_mas.py)

Implementation status: v0.1 complete. See the suite README for how to run it.
"""

from bili.aegis.probe.budget import BudgetState
from bili.aegis.probe.schema import (
    AttackIntent,
    ProbeObjective,
    ProbeOutcome,
    ProbeSession,
    ProbeTurn,
)

__all__ = [
    "ProbeSession",
    "ProbeTurn",
    "ProbeOutcome",
    "ProbeObjective",
    "AttackIntent",
    "BudgetState",
]
