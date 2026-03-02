"""Standalone baseline runner for AETHER MAS evaluation.

Runs all 20 baseline prompts against the 5 content-moderation MAS configs and
writes one structured JSON result file per (config, prompt) pair.  Also prints
a summary table on completion.

Usage
-----
Stub mode (no LLM calls — for structural verification):

    python bili/aether/tests/baseline/run_baseline.py --stub

Real LLM mode (requires API credentials in environment):

    python bili/aether/tests/baseline/run_baseline.py

The five MAS configs used are:
    simple_chain, hierarchical_voting, supervisor_moderation,
    consensus_network, custom_escalation

Results are written to:
    bili/aether/tests/baseline/results/{mas_id}/{prompt_id}.json
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — importable regardless of cwd
# Bootstrap: _find_repo_root is inlined to set up sys.path before any
# bili.* import.  The shared version in bili.aether.tests._helpers is used
# for config_fingerprint once the package is importable.
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

from langchain_core.messages import (  # noqa: E402  pylint: disable=wrong-import-position
    HumanMessage,
)

from bili.aether.config.loader import (  # noqa: E402  pylint: disable=wrong-import-position
    load_mas_from_yaml,
)
from bili.aether.runtime.executor import (  # noqa: E402  pylint: disable=wrong-import-position
    MASExecutor,
)
from bili.aether.tests._helpers import (  # noqa: E402  pylint: disable=wrong-import-position
    config_fingerprint as _config_fingerprint_helper,
)
from bili.aether.tests.baseline.prompts.baseline_prompts import (  # noqa: E402  pylint: disable=wrong-import-position
    BASELINE_PROMPTS,
    BaselinePrompt,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASELINE_DIR = Path(__file__).parent
_RESULTS_DIR = _BASELINE_DIR / "results"

_CONFIG_PATHS: list[str] = [
    "bili/aether/config/examples/simple_chain.yaml",
    "bili/aether/config/examples/hierarchical_voting.yaml",
    "bili/aether/config/examples/supervisor_moderation.yaml",
    "bili/aether/config/examples/consensus_network.yaml",
    "bili/aether/config/examples/custom_escalation.yaml",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_one(config, yaml_path: str, prompt: BaselinePrompt, stub_mode: bool) -> dict:
    """Execute a single (config, prompt) pair and return a result dict."""
    log_dir = str(_RESULTS_DIR / config.mas_id)
    executor = MASExecutor(config, log_dir=log_dir)
    executor.initialize()

    result = executor.run(
        input_data={"messages": [HumanMessage(content=prompt.text)]},
        save_results=False,
    )

    agent_outputs: dict = {}
    for ar in result.agent_results:
        raw = ar.output.get("raw") if isinstance(ar.output, dict) else None
        if raw is None and ar.output:
            raw = str(ar.output)
        agent_outputs[ar.agent_id] = {
            "raw": raw,
            "parsed": ar.output.get("parsed") if isinstance(ar.output, dict) else None,
        }

    return {
        "prompt_id": prompt.prompt_id,
        "prompt_category": prompt.category,
        "prompt_text": prompt.text,
        "mas_id": config.mas_id,
        "config_fingerprint": _config_fingerprint_helper(config, yaml_path, _REPO_ROOT),
        "execution": {
            "success": result.success,
            "duration_ms": result.total_execution_time_ms,
            "agent_count": len(result.agent_results),
            "message_count": result.total_messages,
        },
        "agent_outputs": agent_outputs,
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stub_mode": stub_mode,
            "semantic_tier": "skipped",
        },
    }


def _write_result(result_dict: dict) -> Path:
    """Write result dict to results/{mas_id}/{prompt_id}.json."""
    out_dir = _RESULTS_DIR / result_dict["mas_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result_dict['prompt_id']}.json"
    out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    return out_path


def _print_summary(
    rows: list[dict],
    config_ids: list[str],
    prompt_ids: list[str],
) -> None:
    """Print a summary table: rows=prompts, cols=configs, cells=✓/✗."""
    status: dict[tuple, bool] = {
        (r["mas_id"], r["prompt_id"]): r["execution"]["success"] for r in rows
    }

    col_width = max(len(c) for c in config_ids) + 2
    header = f"{'prompt_id':<20}" + "".join(c.center(col_width) for c in config_ids)
    print("\n" + "=" * len(header))
    print("Baseline Run Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for pid in prompt_ids:
        row = f"{pid:<20}"
        for cid in config_ids:
            ok = status.get((cid, pid))
            cell = "✓" if ok else ("✗" if ok is False else "—")
            row += cell.center(col_width)
        print(row)

    total = len(rows)
    passed = sum(1 for r in rows if r["execution"]["success"])
    print("-" * len(header))
    print(f"Total: {passed}/{total} succeeded\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all baseline prompts against all 5 MAS configs."""
    parser = argparse.ArgumentParser(
        description="Run AETHER baseline prompts against content-moderation MAS configs."
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub agents (no LLM calls). Useful for structural verification.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=_CONFIG_PATHS,
        help="Override the list of YAML config paths to run.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Restrict run to specific prompt IDs (e.g. benign_001 edge_003).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    prompts = BASELINE_PROMPTS
    if args.prompts:
        prompts = [p for p in BASELINE_PROMPTS if p.prompt_id in args.prompts]
        if not prompts:
            print(f"No prompts matched: {args.prompts}", file=sys.stderr)
            sys.exit(1)

    all_results: list[dict] = []
    config_ids: list[str] = []

    for yaml_path in args.configs:
        full_path = _REPO_ROOT / yaml_path
        if not full_path.exists():
            print(f"Config not found, skipping: {yaml_path}", file=sys.stderr)
            continue

        config = load_mas_from_yaml(str(full_path))
        if args.stub:
            for agent in config.agents:
                agent.model_name = None
        config_ids.append(config.mas_id)
        print(f"\n[{config.mas_id}] Running {len(prompts)} prompts...")

        for prompt in prompts:
            print(f"  {prompt.prompt_id} ... ", end="", flush=True)
            try:
                result = _run_one(config, yaml_path, prompt, stub_mode=args.stub)
                out_path = _write_result(result)
                status = "ok" if result["execution"]["success"] else "FAIL"
                print(
                    f"{status}  ({result['execution']['duration_ms']:.0f} ms) → {out_path}"
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.error(
                    "Error running %s / %s: %s", config.mas_id, prompt.prompt_id, exc
                )
                print(f"ERROR: {exc}")
                # Write a failed result so the structural suite can detect it
                failed = {
                    "prompt_id": prompt.prompt_id,
                    "prompt_category": prompt.category,
                    "prompt_text": prompt.text,
                    "mas_id": config.mas_id,
                    "config_fingerprint": _config_fingerprint_helper(
                        config, yaml_path, _REPO_ROOT
                    ),
                    "execution": {
                        "success": False,
                        "duration_ms": 0.0,
                        "agent_count": 0,
                        "message_count": 0,
                    },
                    "agent_outputs": {},
                    "run_metadata": {
                        "timestamp": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "stub_mode": args.stub,
                        "semantic_tier": "skipped",
                    },
                }
                _write_result(failed)
                all_results.append(failed)
                continue

            all_results.append(result)

    if all_results:
        _print_summary(
            all_results,
            config_ids,
            [p.prompt_id for p in prompts],
        )

    total = len(all_results)
    passed = sum(1 for r in all_results if r["execution"]["success"])
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
