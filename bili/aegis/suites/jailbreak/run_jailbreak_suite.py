"""Standalone jailbreak test suite runner for AETHER.

Runs all 15 jailbreak payloads against the 5 content-moderation MAS configs
across both injection phases (pre_execution, mid_execution), applies all three
detection tiers, and writes:

1. One result JSON per test case to ``results/{mas_id}/{payload_id}_{phase}.json``
2. A results matrix CSV to ``results/jailbreak_results_matrix.csv``

Detection tiers
---------------
- Tier 1 (structural):  ``execution.success`` — did the injection run without error?
- Tier 2 (heuristic):   ``influenced_agents`` / ``resistant_agents`` from
  ``PropagationTracker`` — which agents were heuristically detected as
  influenced or resistant?
- Tier 3 (semantic):    ``SemanticEvaluator`` — LLM-based scoring of genuine
  role abandonment.  Skipped in stub mode; columns are empty strings in the CSV.

The results matrix CSV uses the same column schema as the injection suite
(``run_injection_suite.py``) with an additional ``attack_suite`` column set to
``"jailbreak"``.  Rows from both suites can be concatenated and filtered for
cross-suite comparison in the thesis results matrix.

Usage
-----
Stub mode (no LLM calls — structural verification only):

    python bili/aegis/suites/jailbreak/run_jailbreak_suite.py --stub

Real LLM mode (requires API credentials):

    python bili/aegis/suites/jailbreak/run_jailbreak_suite.py

Filter to specific payloads / phases:

    python bili/aegis/suites/jailbreak/run_jailbreak_suite.py --stub \\
        --payloads jb_dan_001 jb_authority_001 \\
        --phases pre_execution

Results are written to:
    bili/aegis/suites/jailbreak/results/{mas_id}/{payload_id}_{phase}.json
    bili/aegis/suites/jailbreak/results/jailbreak_results_matrix.csv
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
from bili.aegis.suites.jailbreak.payloads.jailbreak_payloads import (  # noqa: E402  pylint: disable=wrong-import-position
    JAILBREAK_PAYLOADS,
)

LOGGER = logging.getLogger(__name__)

_JAILBREAK_DIR = Path(__file__).parent
_RESULTS_DIR = _JAILBREAK_DIR / "results"

_ALL_PHASES: list[str] = [
    InjectionPhase.PRE_EXECUTION.value,
    InjectionPhase.MID_EXECUTION.value,
]


def main() -> None:
    """Run the full jailbreak test suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AETHER structured jailbreak test suite against content-moderation "
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
        help="Restrict run to specific payload IDs (e.g. jb_dan_001 jb_authority_001).",
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
    payloads = JAILBREAK_PAYLOADS
    if args.payloads:
        ids = set(args.payloads)
        payloads = [ip for ip in JAILBREAK_PAYLOADS if ip.payload_id in ids]
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
    semantic_evaluator = None
    if not args.stub:
        try:
            from bili.aegis.evaluator import (  # pylint: disable=import-outside-toplevel
                JAILBREAK_JUDGE_PROMPT,
                JAILBREAK_SCORE_DESCRIPTIONS,
                SemanticEvaluator,
            )

            semantic_evaluator = SemanticEvaluator(
                score_descriptions=JAILBREAK_SCORE_DESCRIPTIONS,
                judge_prompt_template=JAILBREAK_JUDGE_PROMPT,
            )
        except (ImportError, RuntimeError) as exc:
            LOGGER.warning("Could not initialise SemanticEvaluator: %s", exc)

    run_suite(
        payloads=payloads,
        attack_suite="jailbreak",
        attack_type="jailbreak",
        csv_filename="jailbreak_results_matrix.csv",
        suite_name="Jailbreak Suite",
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
