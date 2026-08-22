"""Eval runner: load cases, run the agent, grade traces, report.

Usage:
    python -m evals.runner --provider mock
    python -m evals.runner --provider anthropic --repeat 5
    python -m evals.runner --provider anthropic --case fx_inr --json-out run.json

Exits non-zero if the pass rate falls below --threshold, so this drops
straight into CI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from agent.loop import AnthropicProvider, MockProvider, Trace, run_agent
from evals.graders import CheckResult, grade

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "cases" / "finance_tools.json"


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def build_provider(kind: str, case: Dict[str, Any], model: str):
    if kind == "mock":
        script = case.get("mock_script")
        if script is None:
            raise SystemExit(
                f"case {case['id']!r} has no mock_script, so it cannot run under "
                "--provider mock. Add one, or run with --provider anthropic."
            )
        return MockProvider(script)
    return AnthropicProvider(model=model)


def run_case(case: Dict[str, Any], provider_kind: str, model: str, repeat: int):
    """Run one case `repeat` times and collect per-attempt results."""
    attempts = []
    for _ in range(repeat):
        provider = build_provider(provider_kind, case, model)
        trace = run_agent(
            case["prompt"],
            provider,
            case_id=case["id"],
            max_turns=case.get("max_turns", 6),
        )
        results = grade(trace, case.get("checks", {}))
        attempts.append({"trace": trace, "results": results})
    return attempts


def attempt_passed(results: List[CheckResult]) -> bool:
    return all(r.passed for r in results)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def summarise(case: Dict[str, Any], attempts) -> Dict[str, Any]:
    passes = [attempt_passed(a["results"]) for a in attempts]
    traces: List[Trace] = [a["trace"] for a in attempts]

    outcome_checks, trajectory_checks = [], []
    for a in attempts:
        for r in a["results"]:
            (trajectory_checks if r.kind == "trajectory" else outcome_checks).append(r)

    failures: Dict[str, str] = {}
    for a in attempts:
        for r in a["results"]:
            if not r.passed and r.name not in failures:
                failures[r.name] = r.detail

    return {
        "id": case["id"],
        "tags": case.get("tags", []),
        "attempts": len(attempts),
        "passed": sum(passes),
        "pass_rate": sum(passes) / len(passes) if passes else 0.0,
        "avg_turns": statistics.mean([t.turns for t in traces]) if traces else 0,
        "tool_errors": sum(t.error_count for t in traces),
        "input_tokens": sum(t.input_tokens for t in traces),
        "output_tokens": sum(t.output_tokens for t in traces),
        "outcome_pass_rate": (
            sum(r.passed for r in outcome_checks) / len(outcome_checks)
            if outcome_checks
            else None
        ),
        "trajectory_pass_rate": (
            sum(r.passed for r in trajectory_checks) / len(trajectory_checks)
            if trajectory_checks
            else None
        ),
        "failures": failures,
    }


def print_report(summaries: List[Dict[str, Any]], repeat: int) -> float:
    width = max((len(s["id"]) for s in summaries), default=10) + 2
    print()
    print(f"{'CASE'.ljust(width)}{'PASS':>8}{'RATE':>8}{'TURNS':>8}{'ERRS':>7}")
    print("-" * (width + 31))
    for s in summaries:
        mark = "ok " if s["pass_rate"] == 1.0 else ("~  " if s["pass_rate"] > 0 else "FAIL")
        ratio = "{}/{}".format(s["passed"], s["attempts"])
        rate = "{:.0%}".format(s["pass_rate"])
        print(
            f"{s['id'].ljust(width)}"
            f"{ratio:>8}"
            f"{rate:>8}"
            f"{s['avg_turns']:>8.1f}"
            f"{s['tool_errors']:>7}"
            f"  {mark}"
        )

    total_attempts = sum(s["attempts"] for s in summaries)
    total_passed = sum(s["passed"] for s in summaries)
    overall = total_passed / total_attempts if total_attempts else 0.0

    out_rates = [s["outcome_pass_rate"] for s in summaries if s["outcome_pass_rate"] is not None]
    traj_rates = [s["trajectory_pass_rate"] for s in summaries if s["trajectory_pass_rate"] is not None]

    print("-" * (width + 31))
    print(f"cases            {len(summaries)}   (x{repeat} attempts each)")
    print(f"overall pass     {total_passed}/{total_attempts}  ({overall:.1%})")
    if out_rates:
        print(f"outcome checks   {statistics.mean(out_rates):.1%}")
    if traj_rates:
        print(f"trajectory       {statistics.mean(traj_rates):.1%}")
    tok_in = sum(s["input_tokens"] for s in summaries)
    tok_out = sum(s["output_tokens"] for s in summaries)
    if tok_in or tok_out:
        print(f"tokens           {tok_in} in / {tok_out} out")

    failing = [s for s in summaries if s["pass_rate"] < 1.0]
    if failing:
        print("\nFAILURES")
        for s in failing:
            print(f"  {s['id']} ({s['pass_rate']:.0%})")
            for name, detail in s["failures"].items():
                print(f"    - {name}: {detail or 'failed'}")
    print()
    return overall


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Attempts per case. Use >1 against a real model to measure variance.",
    )
    parser.add_argument("--case", help="Run only cases whose id contains this string")
    parser.add_argument("--tag", help="Run only cases carrying this tag")
    parser.add_argument("--json-out", type=Path, help="Write full results as JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum overall pass rate before exiting non-zero (default 1.0)",
    )
    args = parser.parse_args(argv)

    spec = json.loads(args.cases.read_text())
    cases = spec["cases"]
    if args.case:
        cases = [c for c in cases if args.case in c["id"]]
    if args.tag:
        cases = [c for c in cases if args.tag in c.get("tags", [])]
    if not cases:
        print("No cases matched the filter.", file=sys.stderr)
        return 2

    summaries, raw = [], []
    for case in cases:
        attempts = run_case(case, args.provider, args.model, args.repeat)
        summaries.append(summarise(case, attempts))
        raw.append(
            {
                "id": case["id"],
                "attempts": [
                    {
                        "trace": a["trace"].to_dict(),
                        "results": [vars(r) for r in a["results"]],
                    }
                    for a in attempts
                ],
            }
        )

    overall = print_report(summaries, args.repeat)

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"provider": args.provider, "summaries": summaries, "runs": raw},
                indent=2,
                default=str,
            )
        )
        print(f"wrote {args.json_out}")

    return 0 if overall >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
