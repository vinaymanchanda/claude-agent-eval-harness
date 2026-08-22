"""The agent loop: model -> tool calls -> tool results -> model, until done.

Everything the loop does is recorded into a Trace. Graders read the Trace,
never the model output alone, because "did it get the right answer" and
"did it get there the right way" are different questions and an eval suite
that only asks the first one will happily pass an agent that guessed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol

from .tools import TOOL_SCHEMAS, ToolError, dispatch

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SYSTEM = (
    "You are a precise financial assistant. You have tools for arithmetic, "
    "currency conversion, compound interest, and ticker lookup.\n\n"
    "Rules:\n"
    "- Never do arithmetic yourself. Always call the calculator tool.\n"
    "- Never guess an exchange rate or a company's sector. Look it up.\n"
    "- If a tool returns an error, read it, correct your arguments, and retry "
    "once. If it still fails, tell the user plainly what is unsupported.\n"
    "- Give the final answer in one short sentence, including units."
)


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    output: Any
    is_error: bool = False


@dataclass
class Step:
    index: int
    assistant_text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)


@dataclass
class Trace:
    case_id: str
    prompt: str
    steps: List[Step] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = ""
    hit_turn_limit: bool = False
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

    # -- convenience accessors used by graders -----------------------------

    @property
    def tool_names(self) -> List[str]:
        """Every tool name called, in order."""
        return [c.name for step in self.steps for c in step.tool_calls]

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [c for step in self.steps for c in step.tool_calls]

    @property
    def tool_results(self) -> List[ToolResult]:
        return [r for step in self.steps for r in step.tool_results]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.tool_results if r.is_error)

    @property
    def turns(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


@dataclass
class ProviderResponse:
    stop_reason: str
    text: str
    tool_calls: List[ToolCall]
    raw_content: List[Dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    def complete(
        self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> ProviderResponse: ...


class AnthropicProvider:
    """Calls the real Claude API. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The anthropic package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill it in, or run with --provider mock."
            )
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system, messages, tools) -> ProviderResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        text_parts, calls, raw = [], [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
                raw.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return ProviderResponse(
            stop_reason=resp.stop_reason,
            text="".join(text_parts).strip(),
            tool_calls=calls,
            raw_content=raw,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class MockProvider:
    """Replays a scripted list of responses.

    This exists so the eval suite runs in CI with no API key and no spend.
    It also makes harness bugs debuggable: when a case fails under Mock, the
    harness is wrong, not the model.
    """

    def __init__(self, script: List[Dict[str, Any]]):
        self.script = list(script)
        self.calls = 0

    def complete(self, system, messages, tools) -> ProviderResponse:
        if self.calls >= len(self.script):
            raise RuntimeError(
                f"MockProvider ran out of scripted turns after {self.calls}. "
                "The script is shorter than the loop needed."
            )
        entry = self.script[self.calls]
        self.calls += 1

        calls, raw = [], []
        text = entry.get("text", "")
        if text:
            raw.append({"type": "text", "text": text})
        for i, call in enumerate(entry.get("tool_calls", [])):
            cid = call.get("id") or f"mock_{self.calls}_{i}"
            calls.append(ToolCall(id=cid, name=call["name"], input=call["input"]))
            raw.append(
                {
                    "type": "tool_use",
                    "id": cid,
                    "name": call["name"],
                    "input": call["input"],
                }
            )
        return ProviderResponse(
            stop_reason="tool_use" if calls else "end_turn",
            text=text,
            tool_calls=calls,
            raw_content=raw,
        )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_agent(
    prompt: str,
    provider: Provider,
    *,
    case_id: str = "ad-hoc",
    system: str = DEFAULT_SYSTEM,
    max_turns: int = 6,
) -> Trace:
    """Run one task to completion and return the full Trace.

    max_turns is a hard stop. An agent that loops forever is a production
    incident, so the harness treats hitting the limit as a recorded failure
    mode rather than something to silently retry.
    """
    trace = Trace(case_id=case_id, prompt=prompt)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

    for turn in range(max_turns):
        step = Step(index=turn)
        try:
            resp = provider.complete(system, messages, TOOL_SCHEMAS)
        except Exception as exc:  # provider/transport failure
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.steps.append(step)
            return trace

        step.assistant_text = resp.text
        step.tool_calls = resp.tool_calls
        trace.input_tokens += resp.input_tokens
        trace.output_tokens += resp.output_tokens
        trace.stop_reason = resp.stop_reason

        if not resp.tool_calls:
            trace.steps.append(step)
            trace.final_text = resp.text
            return trace

        messages.append({"role": "assistant", "content": resp.raw_content})

        result_blocks = []
        for call in resp.tool_calls:
            try:
                output: Any = dispatch(call.name, call.input)
                is_error = False
            except ToolError as exc:
                output = {"error": str(exc)}
                is_error = True
            step.tool_results.append(
                ToolResult(id=call.id, name=call.name, output=output, is_error=is_error)
            )
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(output),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": result_blocks})
        trace.steps.append(step)

    trace.hit_turn_limit = True
    trace.error = f"hit max_turns={max_turns} without a final answer"
    return trace
