"""Deterministic tool implementations exposed to the agent.

Every tool here is pure and offline. That is a deliberate design choice:
evals must be reproducible, so the tool layer cannot depend on wall-clock
time, network calls, or hidden state. Non-determinism belongs in the model,
not in the environment we grade against.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict

# --------------------------------------------------------------------------
# Static reference data. Frozen on purpose so eval expectations stay stable.
# --------------------------------------------------------------------------

FX_RATES: Dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "INR": 83.20,
    "JPY": 157.30,
}

TICKERS: Dict[str, Dict[str, Any]] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "currency": "USD"},
    "HDFCBANK": {"name": "HDFC Bank Limited", "sector": "Financials", "currency": "INR"},
    "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Discretionary", "currency": "USD"},
}


class ToolError(Exception):
    """Raised when a tool is called with arguments it cannot honour.

    The loop catches this and returns it to the model as a tool result with
    is_error=True, which lets us grade recovery behaviour rather than just
    crashing the run.
    """


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ToolError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ToolError(f"unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic expression without using eval().

    We parse to an AST and walk a whitelist of node types. Passing user or
    model output to eval() would be a code-execution hole; an agent that can
    be prompt-injected into calling calculator("__import__('os').system(...)")
    is a real vulnerability, not a hypothetical one.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression {expression!r}: {exc.msg}") from exc

    try:
        value = _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc

    return {"expression": expression, "result": round(value, 6)}


# --------------------------------------------------------------------------
# fx_convert
# --------------------------------------------------------------------------


def fx_convert(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Convert an amount between currencies using the frozen rate table."""
    src, dst = from_currency.upper(), to_currency.upper()
    for code in (src, dst):
        if code not in FX_RATES:
            raise ToolError(
                f"unknown currency {code!r}; supported: {', '.join(sorted(FX_RATES))}"
            )
    if amount < 0:
        raise ToolError("amount must be non-negative")

    usd = float(amount) / FX_RATES[src]
    converted = usd * FX_RATES[dst]
    return {
        "amount": float(amount),
        "from_currency": src,
        "to_currency": dst,
        "rate": round(FX_RATES[dst] / FX_RATES[src], 6),
        "result": round(converted, 2),
    }


# --------------------------------------------------------------------------
# compound_interest
# --------------------------------------------------------------------------


def compound_interest(
    principal: float,
    annual_rate_pct: float,
    years: float,
    compounds_per_year: int = 1,
) -> Dict[str, Any]:
    """Future value of a lump sum under periodic compounding."""
    if principal < 0:
        raise ToolError("principal must be non-negative")
    if compounds_per_year < 1:
        raise ToolError("compounds_per_year must be at least 1")
    if years < 0:
        raise ToolError("years must be non-negative")

    r = float(annual_rate_pct) / 100.0
    n = int(compounds_per_year)
    future = float(principal) * (1 + r / n) ** (n * float(years))
    return {
        "principal": float(principal),
        "annual_rate_pct": float(annual_rate_pct),
        "years": float(years),
        "compounds_per_year": n,
        "future_value": round(future, 2),
        "interest_earned": round(future - float(principal), 2),
    }


# --------------------------------------------------------------------------
# lookup_ticker
# --------------------------------------------------------------------------


def lookup_ticker(symbol: str) -> Dict[str, Any]:
    """Return static reference data for a supported ticker."""
    key = symbol.upper().strip()
    if key not in TICKERS:
        raise ToolError(
            f"unknown ticker {key!r}; supported: {', '.join(sorted(TICKERS))}"
        )
    return {"symbol": key, **TICKERS[key]}


# --------------------------------------------------------------------------
# Registry + Claude tool schemas
# --------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "calculator": calculator,
    "fx_convert": fx_convert,
    "compound_interest": compound_interest,
    "lookup_ticker": lookup_ticker,
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression. Supports + - * / ** % and "
            "parentheses. Use this instead of doing arithmetic in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '(1200 * 1.08) - 350'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "fx_convert",
        "description": (
            "Convert an amount from one currency to another. Supported codes: "
            "USD, EUR, GBP, INR, JPY."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
                "to_currency": {"type": "string", "description": "ISO code, e.g. INR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "compound_interest",
        "description": "Compute the future value of a lump sum under compounding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "annual_rate_pct": {
                    "type": "number",
                    "description": "Annual rate as a percentage, e.g. 7.5 for 7.5%",
                },
                "years": {"type": "number"},
                "compounds_per_year": {
                    "type": "integer",
                    "description": "Defaults to 1 (annual compounding)",
                },
            },
            "required": ["principal", "annual_rate_pct", "years"],
        },
    },
    {
        "name": "lookup_ticker",
        "description": (
            "Look up reference data (company name, sector, reporting currency) "
            "for a stock ticker. Supported: AAPL, MSFT, TSLA, HDFCBANK."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"}
            },
            "required": ["symbol"],
        },
    },
]


def dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Route a model tool call to its implementation."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"no such tool: {name!r}")
    try:
        return fn(**arguments)
    except TypeError as exc:
        raise ToolError(f"bad arguments for {name}: {exc}") from exc
