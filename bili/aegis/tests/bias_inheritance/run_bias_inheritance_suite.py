"""Standalone bias inheritance test suite runner for AETHER.

Runs all 15 bias inheritance payloads against the 5 content-moderation MAS
configs across both injection phases (pre_execution, mid_execution), applies
all three detection tiers, and writes:

1. One result JSON per test case to ``results/{mas_id}/{payload_id}_{phase}.json``
2. A results matrix CSV to ``results/bias_inheritance_results_matrix.csv``

Bias inheritance appends a directional "guiding principle" to an agent's
objective, nudging its reasoning toward a predetermined conclusion.  Unlike
prompt injection (which commands) or memory poisoning (which fabricates
context), bias inheritance operates by shaping the *frame* the agent reasons
within.  Compliance is often subtle — the agent's output may still appear
policy-compliant on the surface while systematically skewing toward the
injected direction.

Because of this subtlety, Tier 2 heuristic detection is less reliable for
this attack type.  Tier 3 semantic evaluation (SemanticEvaluator) is the
primary signal and is strongly recommended for meaningful results.

Detection tiers
---------------
- Tier 1 (structural):  ``execution.success`` — did the injection run without error?
- Tier 2 (heuristic):   ``influenced_agents`` / ``resistant_agents`` from
  ``PropagationTracker`` — which agents echoed biased framing or compliance
  marker phrases?
- Tier 3 (semantic):    ``SemanticEvaluator`` — LLM-based scoring of genuine
  bias adoption.  Skipped in stub mode; columns are empty strings in the CSV.

Usage
-----
Stub mode (no LLM calls — structural verification only):

    python bili/aegis/tests/bias_inheritance/run_bias_inheritance_suite.py --stub

Real LLM mode (requires API credentials):

    python bili/aegis/tests/bias_inheritance/run_bias_inheritance_suite.py

Filter to specific payloads / phases:

    python bili/aegis/tests/bias_inheritance/run_bias_inheritance_suite.py --stub \\
        --payloads bi_political_slant_001 bi_outcome_bias_002 \\
        --phases pre_execution

Results are written to:
    bili/aegis/tests/bias_inheritance/results/{mas_id}/{payload_id}_{phase}.json
    bili/aegis/tests/bias_inheritance/results/bias_inheritance_results_matrix.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — importable regardless of cwd
# Bootstrap: _find_repo_root is inlined to set up sys.path before any
# bili.* import.  The shared version in bili.aegis.tests._helpers is used
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
from bili.aegis.tests._helpers import (  # noqa: E402  pylint: disable=wrong-import-position
    CONFIG_PATHS,
)
from bili.aegis.tests._suite_runner import (  # noqa: E402  pylint: disable=wrong-import-position
    run_suite,
)
from bili.aegis.tests.bias_inheritance.payloads.bias_inheritance_payloads import (  # noqa: E402  pylint: disable=wrong-import-position
    BIAS_INHERITANCE_PAYLOADS,
)

LOGGER = logging.getLogger(__name__)

_BIAS_INHERITANCE_DIR = Path(__file__).parent
_RESULTS_DIR = _BIAS_INHERITANCE_DIR / "results"

_ALL_PHASES: list[str] = [
    InjectionPhase.PRE_EXECUTION.value,
    InjectionPhase.MID_EXECUTION.value,
]


def main() -> None:
    """Run the full bias inheritance test suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AETHER bias inheritance test suite against content-moderation "
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
            "(e.g. bi_political_slant_001 bi_outcome_bias_002)."
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
            "(e.g. bili/aegis/tests/baseline/results). "
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
    payloads = BIAS_INHERITANCE_PAYLOADS
    if args.payloads:
        ids = set(args.payloads)
        payloads = [bp for bp in BIAS_INHERITANCE_PAYLOADS if bp.payload_id in ids]
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
    # Uses a bias-specific judge prompt and rubric rather than the default
    # injection-compliance prompt — bias adoption is about directional drift
    # in reasoning, not explicit instruction following.
    semantic_evaluator = None
    if not args.stub:
        try:
            from bili.aegis.evaluator import (  # pylint: disable=import-outside-toplevel
                SemanticEvaluator,
            )
            from bili.aegis.evaluator.evaluator_config import (  # pylint: disable=import-outside-toplevel
                BIAS_INHERITANCE_JUDGE_PROMPT,
                BIAS_INHERITANCE_SCORE_DESCRIPTIONS,
            )

            semantic_evaluator = SemanticEvaluator(
                score_descriptions=BIAS_INHERITANCE_SCORE_DESCRIPTIONS,
                judge_prompt_template=BIAS_INHERITANCE_JUDGE_PROMPT,
            )
        except (ImportError, RuntimeError) as exc:
            LOGGER.warning("Could not initialise SemanticEvaluator: %s", exc)

    run_suite(
        payloads=payloads,
        attack_suite="bias_inheritance",
        attack_type="bias_inheritance",
        csv_filename="bias_inheritance_results_matrix.csv",
        suite_name="Bias Inheritance Suite",
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
