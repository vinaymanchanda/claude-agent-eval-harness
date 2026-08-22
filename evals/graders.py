"""Graders turn a Trace into pass/fail checks.

Two families live here on purpose:

  outcome checks   - did the agent produce the right answer?
  trajectory checks - did it get there by an acceptable route?

Outcome-only suites reward luck. An agent that guesses 99,840 INR without
calling fx_convert passes an outcome check and will fail the moment the
rate moves. Trajectory checks are what catch that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.loop import Trace

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    kind: str = "outcome"  # or "trajectory"


def _numbers_in(text: str) -> List[float]:
    out = []
    for match in _NUMBER_RE.findall(text):
        try:
            out.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# Outcome checks
# --------------------------------------------------------------------------


def final_answer_contains(trace: Trace, expected: List[str]) -> CheckResult:
    text = trace.final_text.lower()
    missing = [s for s in expected if s.lower() not in text]
    return CheckResult(
        name="final_answer_contains",
        passed=not missing,
        detail="" if not missing else f"missing from answer: {missing}",
    )


def final_answer_number(trace: Trace, spec: Dict[str, Any]) -> CheckResult:
    expected = float(spec["value"])
    tolerance = float(spec.get("tolerance", 0.01))
    found = _numbers_in(trace.final_text)
    hit = [n for n in found if abs(n - expected) <= tolerance]
    return CheckResult(
        name="final_answer_number",
        passed=bool(hit),
        detail=(
            ""
            if hit
            else f"expected {expected} (+/-{tolerance}); numbers in answer: {found}"
        ),
    )


# --------------------------------------------------------------------------
# Trajectory checks
# --------------------------------------------------------------------------


def tools_used(trace: Trace, expected: List[str]) -> CheckResult:
    used = set(trace.tool_names)
    missing = [t for t in expected if t not in used]
    return CheckResult(
        name="tools_used",
        passed=not missing,
        detail="" if not missing else f"never called: {missing} (called: {sorted(used)})",
        kind="trajectory",
    )


def tools_not_used(trace: Trace, forbidden: List[str]) -> CheckResult:
    used = set(trace.tool_names)
    hits = [t for t in forbidden if t in used]
    return CheckResult(
        name="tools_not_used",
        passed=not hits,
        detail="" if not hits else f"called forbidden tools: {hits}",
        kind="trajectory",
    )


def tool_called_with(trace: Trace, spec: Dict[str, Any]) -> CheckResult:
    """Assert some call to `tool` had arguments matching every key in `args`.

    Only the listed keys are compared, so a case can pin the currency pair
    without also pinning every optional argument.
    """
    tool = spec["tool"]
    want = spec["args"]
    candidates = [c for c in trace.tool_calls if c.name == tool]
    if not candidates:
        return CheckResult(
            name="tool_called_with",
            passed=False,
            detail=f"{tool} was never called",
            kind="trajectory",
        )
    for call in candidates:
        if all(_loose_eq(call.input.get(k), v) for k, v in want.items()):
            return CheckResult(name="tool_called_with", passed=True, kind="trajectory")
    return CheckResult(
        name="tool_called_with",
        passed=False,
        detail=f"no {tool} call matched {want}; saw {[c.input for c in candidates]}",
        kind="trajectory",
    )


def _loose_eq(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-6
    return actual == expected


def max_turns(trace: Trace, limit: int) -> CheckResult:
    return CheckResult(
        name="max_turns",
        passed=trace.turns <= int(limit),
        detail="" if trace.turns <= int(limit) else f"used {trace.turns} turns, limit {limit}",
        kind="trajectory",
    )


def no_tool_errors(trace: Trace, _: Any = None) -> CheckResult:
    return CheckResult(
        name="no_tool_errors",
        passed=trace.error_count == 0,
        detail="" if trace.error_count == 0 else f"{trace.error_count} tool error(s)",
        kind="trajectory",
    )


def recovers_from_error(trace: Trace, _: Any = None) -> CheckResult:
    """The agent hit a tool error and still produced a final answer.

    Robustness is a product requirement, not a nice-to-have: the interesting
    question is not whether tools ever fail, but what the agent does next.
    """
    hit_error = trace.error_count > 0
    recovered = bool(trace.final_text) and not trace.hit_turn_limit
    return CheckResult(
        name="recovers_from_error",
        passed=hit_error and recovered,
        detail=(
            ""
            if hit_error and recovered
            else f"error_count={trace.error_count}, final_text={'yes' if trace.final_text else 'no'}"
        ),
        kind="trajectory",
    )


GRADERS = {
    "final_answer_contains": final_answer_contains,
    "final_answer_number": final_answer_number,
    "tools_used": tools_used,
    "tools_not_used": tools_not_used,
    "tool_called_with": tool_called_with,
    "max_turns": max_turns,
    "no_tool_errors": no_tool_errors,
    "recovers_from_error": recovers_from_error,
}


def grade(trace: Trace, checks: Dict[str, Any]) -> List[CheckResult]:
    """Run every check named in a case spec against one trace."""
    results: List[CheckResult] = []
    if trace.error and not trace.final_text:
        results.append(
            CheckResult(name="run_completed", passed=False, detail=trace.error)
        )
    for name, spec in checks.items():
        grader = GRADERS.get(name)
        if grader is None:
            results.append(
                CheckResult(name=name, passed=False, detail="unknown grader")
            )
            continue
        results.append(grader(trace, spec))
    return results
