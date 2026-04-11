"""Standalone agent impersonation test suite runner for AETHER.

Runs all 15 agent impersonation payloads against the 5 content-moderation MAS
configs across both injection phases (pre_execution, mid_execution), applies
all three detection tiers, and writes:

1. One result JSON per test case to ``results/{mas_id}/{payload_id}_{phase}.json``
2. A results matrix CSV to ``results/agent_impersonation_results_matrix.csv``

Agent impersonation injects a false identity or authority claim into an agent's
input, attempting to make the agent believe it is a different agent, has
elevated permissions, or should defer to fabricated peer output.  Unlike
prompt injection (which overrides instructions) or memory poisoning (which
fabricates context), agent impersonation targets *who the agent believes it is*.

Detection tiers
---------------
- Tier 1 (structural):  ``execution.success`` — did the injection run without error?
- Tier 2 (heuristic):   ``influenced_agents`` / ``resistant_agents`` from
  ``PropagationTracker`` — which agents adopted identity-change markers or
  compliance phrases (e.g. "i am now", "acting as", "my new role")?
- Tier 3 (semantic):    ``SemanticEvaluator`` — LLM-based scoring of genuine
  identity adoption.  Skipped in stub mode; columns are empty strings in the CSV.

Usage
-----
Stub mode (no LLM calls — structural verification only):

    python bili/aegis/suites/agent_impersonation/run_agent_impersonation_suite.py --stub

Real LLM mode (requires API credentials):

    python bili/aegis/suites/agent_impersonation/run_agent_impersonation_suite.py

Filter to specific payloads / phases:

    python bili/aegis/suites/agent_impersonation/run_agent_impersonation_suite.py --stub \\
        --payloads ai_full_swap_001 ai_peer_impersonation_002 \\
        --phases pre_execution

Results are written to:
    bili/aegis/suites/agent_impersonation/results/{mas_id}/{payload_id}_{phase}.json
    bili/aegis/suites/agent_impersonation/results/agent_impersonation_results_matrix.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — importable regardless of cwd
# Bootstrap: _find_repo_root is inlined to set up sys.path before any
# bili.* import.  The shared version in bili.aegis.suites._helpers is used
# for all subsequent calls once the package is importable.
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this file until a .git directory is found."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (.git directory not found)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bili.aegis.attacks.models import (  # noqa: E402  pylint: disable=wrong-import-position
    InjectionPhase,
)
from bili.aegis.suites._helpers import (  # noqa: E402  pylint: disable=wrong-import-position
    CONFIG_PATHS,
)
from bili.aegis.suites._suite_runner import (  # noqa: E402  pylint: disable=wrong-import-position
    run_suite,
)
from bili.aegis.suites.agent_impersonation.payloads.agent_impersonation_payloads import (  # noqa: E402  pylint: disable=wrong-import-position
    AGENT_IMPERSONATION_PAYLOADS,
)

LOGGER = logging.getLogger(__name__)

_AGENT_IMPERSONATION_DIR = Path(__file__).parent
_RESULTS_DIR = _AGENT_IMPERSONATION_DIR / "results"

_ALL_PHASES: list[str] = [
    InjectionPhase.PRE_EXECUTION.value,
    InjectionPhase.MID_EXECUTION.value,
]


def main() -> None:
    """Run the full agent impersonation test suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AETHER agent impersonation test suite against content-moderation "
            "MAS configs and produce a results matrix CSV."
        )
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use stub agents (no LLM calls). Tier 3 semantic evaluation is "
            "skipped and CSV tier3 columns are empty strings."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIG_PATHS,
        metavar="YAML_PATH",
        help="Override the list of YAML config paths to run.",
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        default=None,
        metavar="PAYLOAD_ID",
        help=(
            "Restrict run to specific payload IDs "
            "(e.g. ai_full_swap_001 ai_peer_impersonation_002)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=_ALL_PHASES,
        choices=_ALL_PHASES,
        metavar="PHASE",
        help="Injection phases to run (default: both pre_execution and mid_execution).",
    )
    parser.add_argument(
        "--baseline-results",
        default=None,
        metavar="DIR",
        help=(
            "Path to baseline results directory for Tier 3 comparison "
            "(e.g. bili/aegis/suites/baseline/results). "
            "Required for real-mode Tier 3 evaluation."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Filter payloads
    payloads = AGENT_IMPERSONATION_PAYLOADS
    if args.payloads:
        ids = set(args.payloads)
        payloads = [ap for ap in AGENT_IMPERSONATION_PAYLOADS if ap.payload_id in ids]
        if not payloads:
            print(f"No payloads matched: {args.payloads}", file=sys.stderr)
            sys.exit(1)

    # Baseline dir for Tier 3
    baseline_results_dir: Path | None = None
    if args.baseline_results:
        baseline_results_dir = _REPO_ROOT / args.baseline_results
        if not baseline_results_dir.exists():
            print(
                f"Warning: baseline results dir not found: {baseline_results_dir}",
                file=sys.stderr,
            )
            baseline_results_dir = None

    # SemanticEvaluator (lazy import — only needed in real mode)
    # Uses an impersonation-specific judge prompt and rubric rather than the
    # default injection-compliance prompt — the evaluator checks whether the
    # agent adopted the false identity or acted on the fabricated authority
    # claim, not merely whether it followed an explicit instruction.
    semantic_evaluator = None
    if not args.stub:
        try:
            from bili.aegis.evaluator import (  # pylint: disable=import-outside-toplevel
                SemanticEvaluator,
            )
            from bili.aegis.evaluator.evaluator_config import (  # pylint: disable=import-outside-toplevel
                AGENT_IMPERSONATION_JUDGE_PROMPT,
                AGENT_IMPERSONATION_SCORE_DESCRIPTIONS,
            )

            semantic_evaluator = SemanticEvaluator(
                score_descriptions=AGENT_IMPERSONATION_SCORE_DESCRIPTIONS,
                judge_prompt_template=AGENT_IMPERSONATION_JUDGE_PROMPT,
            )
        except (ImportError, RuntimeError) as exc:
            LOGGER.warning("Could not initialise SemanticEvaluator: %s", exc)

    run_suite(
        payloads=payloads,
        attack_suite="agent_impersonation",
        attack_type="agent_impersonation",
        csv_filename="agent_impersonation_results_matrix.csv",
        suite_name="Agent Impersonation Suite",
        results_dir=_RESULTS_DIR,
        repo_root=_REPO_ROOT,
        config_paths=args.configs,
        phases=args.phases,
        stub=args.stub,
        semantic_evaluator=semantic_evaluator,
        baseline_results_dir=baseline_results_dir,
    )


if __name__ == "__main__":
    main()
