"""Tests for the harness itself.

A suite that only ever passes is worthless, so most of these assert that
the graders *fail* when they should. If the harness cannot detect a bad
agent, a green eval run means nothing.
"""

import pytest

from agent.loop import MockProvider, run_agent
from agent.tools import ToolError, calculator, dispatch, fx_convert
from evals.graders import grade


def run(script, prompt="test"):
    return run_agent(prompt, MockProvider(script), case_id="t", max_turns=6)


# --------------------------------------------------------------------------
# Tool safety
# --------------------------------------------------------------------------


def test_calculator_rejects_code_execution():
    """The classic agent hole: a tool that pipes model output into eval()."""
    with pytest.raises(ToolError):
        calculator("__import__('os').system('echo pwned')")


def test_calculator_rejects_name_lookup():
    with pytest.raises(ToolError):
        calculator("open")


def test_calculator_reports_division_by_zero():
    with pytest.raises(ToolError, match="division by zero"):
        calculator("1/0")


def test_calculator_handles_precedence():
    assert calculator("2 + 3 * 4")["result"] == 14.0


def test_fx_rejects_unknown_currency():
    with pytest.raises(ToolError, match="CHF"):
        fx_convert(10, "CHF", "USD")


def test_dispatch_rejects_unknown_tool():
    with pytest.raises(ToolError, match="no such tool"):
        dispatch("rm_rf", {})


def test_dispatch_reports_bad_arguments():
    with pytest.raises(ToolError, match="bad arguments"):
        dispatch("fx_convert", {"amount": 10})


# --------------------------------------------------------------------------
# Loop behaviour
# --------------------------------------------------------------------------


def test_tool_error_is_returned_to_model_not_raised():
    trace = run(
        [
            {"tool_calls": [{"name": "fx_convert", "input": {"amount": 1, "from_currency": "CHF", "to_currency": "USD"}}]},
            {"text": "CHF is unsupported."},
        ]
    )
    assert trace.error_count == 1
    assert trace.final_text == "CHF is unsupported."
    assert trace.error is None


def test_loop_stops_at_max_turns():
    forever = [
        {"tool_calls": [{"name": "calculator", "input": {"expression": "1+1"}}]}
    ] * 10
    trace = run_agent("loop", MockProvider(forever), max_turns=3)
    assert trace.hit_turn_limit
    assert trace.turns == 3
    assert "max_turns" in (trace.error or "")


def test_provider_failure_is_captured_not_raised():
    trace = run_agent("x", MockProvider([]), max_turns=2)
    assert trace.error is not None
    assert trace.final_text == ""


# --------------------------------------------------------------------------
# Graders must actually fail bad agents
# --------------------------------------------------------------------------


def test_grader_catches_agent_that_skipped_the_tool():
    """The whole point of trajectory checks: right number, wrong route."""
    trace = run([{"text": "500 USD is 41,600.00 INR."}])
    results = grade(
        trace,
        {
            "final_answer_number": {"value": 41600.0, "tolerance": 1.0},
            "tools_used": ["fx_convert"],
        },
    )
    by_name = {r.name: r for r in results}
    assert by_name["final_answer_number"].passed is True
    assert by_name["tools_used"].passed is False


def test_grader_catches_wrong_number():
    trace = run([{"text": "That comes to 999 INR."}])
    results = grade(trace, {"final_answer_number": {"value": 41600.0}})
    assert results[0].passed is False


def test_grader_catches_forbidden_tool():
    trace = run(
        [
            {"tool_calls": [{"name": "fx_convert", "input": {"amount": 1, "from_currency": "USD", "to_currency": "EUR"}}]},
            {"text": "done"},
        ]
    )
    results = grade(trace, {"tools_not_used": ["fx_convert"]})
    assert results[0].passed is False


def test_grader_catches_wrong_arguments():
    trace = run(
        [
            {"tool_calls": [{"name": "fx_convert", "input": {"amount": 500, "from_currency": "USD", "to_currency": "EUR"}}]},
            {"text": "done"},
        ]
    )
    results = grade(
        trace,
        {"tool_called_with": {"tool": "fx_convert", "args": {"to_currency": "INR"}}},
    )
    assert results[0].passed is False


def test_grader_flags_incomplete_run():
    trace = run_agent("x", MockProvider([]), max_turns=2)
    results = grade(trace, {})
    assert results[0].name == "run_completed"
    assert results[0].passed is False


def test_number_extraction_handles_thousands_separators():
    trace = run([{"text": "The result is 1,234,567.89 dollars."}])
    results = grade(trace, {"final_answer_number": {"value": 1234567.89, "tolerance": 0.01}})
    assert results[0].passed is True
