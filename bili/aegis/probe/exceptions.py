"""PROBE-specific exception types.

Kept in its own module so importing them does not transitively pull in
``bili.aegis.evaluator`` or any LLM-loading code — useful when the runner
needs to catch ``JudgeUnavailableError`` per session without paying the
import cost on every iteration.
"""

from __future__ import annotations


class JudgeUnavailableError(Exception):
    """Raised at ``SuccessEvaluatorNode.__init__`` when the cross-provider
    constraint is violated (judge family equals attacker or victim family),
    or when a model_name has no recognized provider-family prefix.

    The runner catches this per-session and writes a row with
    ``terminated_reason='judge_unavailable'`` instead of running the attack.
    """

    def __init__(
        self,
        message: str,
        provider_family: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_family = provider_family
