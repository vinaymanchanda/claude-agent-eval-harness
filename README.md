# claude-agent-eval-harness

A Claude tool-use agent, and the eval harness that decides whether it is
actually any good.

The agent is the easy half. Wiring a model to a few tools and looping until
it stops calling them is a weekend's work. The hard half - the half this
repo is really about - is answering *"is this agent working?"* with a number
instead of a vibe, and being able to re-answer it every time a prompt, a
model, or a tool schema changes.

```
CASE                                  PASS    RATE   TURNS   ERRS
-----------------------------------------------------------------
fx_basic_usd_inr                       1/1    100%     2.0      0  ok
compound_quarterly                     1/1    100%     2.0      0  ok
arithmetic_must_use_calculator         1/1    100%     2.0      0  ok
chained_ticker_then_fx                 1/1    100%     3.0      0  ok
retry_after_bad_arguments              1/1    100%     3.0      1  ok
unsupported_currency_is_admitted       1/1    100%     2.0      1  ok
unknown_ticker_is_admitted             1/1    100%     2.0      1  ok
-----------------------------------------------------------------
cases            7   (x1 attempts each)
overall pass     7/7  (100.0%)
outcome checks   100.0%
trajectory       100.0%
```

## Run it in 30 seconds, with no API key

```bash
git clone https://github.com/vinaymanchanda/claude-agent-eval-harness
cd claude-agent-eval-harness
pip install -r requirements.txt
python -m evals.runner --provider mock
python -m pytest tests/ -q
```

`--provider mock` replays scripted model turns through the real loop, real
tools, and real graders. Nothing is stubbed except the model itself. That
matters for two reasons: the suite runs in CI for free, and when a case fails
under mock you know the harness is broken rather than the model.

Against the live model:

```bash
cp .env.example .env          # add your ANTHROPIC_API_KEY
python -m evals.runner --provider anthropic --repeat 5
```

## The idea this repo is built around

Most agent evals check the final answer. That is not enough, because an agent
can be right for reasons that will not survive contact with production.

Ask an agent to convert 500 USD to INR. It replies *"41,600 INR"* - correct.
Did it call the FX tool, or did it recall a rate from pretraining and get
lucky? An outcome-only eval cannot tell you, and passes either way. The day
the rate moves, one of those two agents starts quietly lying to users.

So every case here pins two things:

| | question | example check |
|---|---|---|
| **Outcome** | did it get the right answer? | `final_answer_number: 41600.0` |
| **Trajectory** | did it get there acceptably? | `tools_used: [fx_convert]` |

`tests/test_harness.py` contains the case that proves this works: an agent
that emits the right number without ever calling the tool **passes the
outcome check and fails the trajectory check**. If the harness could not
catch that, a green run would mean nothing.

## Writing a case

Cases are JSON. Nothing is compiled or code-generated, so adding coverage
does not require touching Python.

```json
{
  "id": "chained_ticker_then_fx",
  "tags": ["multi_step", "chaining"],
  "prompt": "HDFC Bank reports in its home currency. If a position is worth 1,000,000 of that currency, what is it worth in US dollars?",
  "checks": {
    "tools_used": ["lookup_ticker", "fx_convert"],
    "tool_called_with": {
      "tool": "fx_convert",
      "args": { "amount": 1000000, "from_currency": "INR", "to_currency": "USD" }
    },
    "final_answer_number": { "value": 12019.23, "tolerance": 1.0 },
    "max_turns": 3
  }
}
```

This one case pins the whole chain: the agent must *discover* that HDFC Bank
reports in INR rather than assuming it, then convert from the currency it
discovered, and do it in at most three turns.

### Graders

| grader | kind | asserts |
|---|---|---|
| `final_answer_contains` | outcome | substrings appear in the final answer |
| `final_answer_number` | outcome | a number within tolerance appears |
| `tools_used` | trajectory | these tools were all called |
| `tools_not_used` | trajectory | these tools were never called |
| `tool_called_with` | trajectory | some call matched these arguments |
| `max_turns` | trajectory | the loop finished within N turns |
| `no_tool_errors` | trajectory | no tool returned an error |
| `recovers_from_error` | trajectory | hit an error *and still* answered |

`recovers_from_error` is the one people leave out. Tools fail in production -
bad arguments, unsupported inputs, upstream timeouts. The question worth
grading is not whether errors happen but what the agent does next, so two
cases here deliberately trigger tool errors and assert on the recovery.

## Measuring variance

Models are stochastic. A single green run is one sample, not a result.

```bash
python -m evals.runner --provider anthropic --repeat 10
```

Each case runs ten times and reports a pass *rate*. A case at 7/10 is not
"passing" - it is a case that will page someone at 3am. The `~` marker in the
report flags exactly those partial failures, which a boolean pass/fail suite
would round away in one direction or the other.

## CI

```bash
python -m evals.runner --provider mock --threshold 1.0
```

Exits non-zero below the threshold. `.github/workflows/evals.yml` runs the
mock suite and the unit tests on every push - no API key, no spend, no
flakiness. Live-model runs are for release gates, where a threshold below 1.0
is usually the honest setting.

## Layout

```
agent/
  tools.py      four deterministic tools + their Claude schemas
  loop.py       the tool-use loop, Trace recording, providers
evals/
  graders.py    outcome and trajectory checks
  runner.py     CLI: load cases, run, grade, report, exit code
cases/
  finance_tools.json
tests/
  test_harness.py   16 tests, mostly proving graders fail bad agents
docs/
  PRD.md        what this is for and what "good" means
  EVAL_PLAN.md  the metric definitions and the release gate
```

## Design notes

**Tools are pure and offline.** No clock, no network, no hidden state. Evals
must be reproducible, and non-determinism belongs in the model, not in the
environment you are grading against. The FX table is frozen for the same
reason expected values are hardcoded: if both moved, a failing case would
never tell you which one broke.

**`calculator` parses an AST instead of calling `eval()`.** An agent that can
be talked into `calculator("__import__('os').system(...)")` is a live remote
code execution path, not a hypothetical. There is a test for it.

**Tool errors are returned to the model, not raised.** A raised exception
ends the run and teaches you nothing. Handing the error back as a
`tool_result` with `is_error: true` is what makes recovery behaviour
observable - and therefore gradeable.

**`max_turns` is a hard stop, and hitting it is a recorded failure.** An agent
that loops forever is a production incident and a billing incident. The
harness never silently retries past the limit.

## What I would build next

- **Judge-based graders** for open-ended answers, validated against human
  labels before being trusted - an unvalidated LLM judge just moves the
  question rather than answering it.
- **Cost and latency budgets** as first-class checks. Token counts are
  already captured per run; they should be assertable, not just reported.
- **Adversarial cases** - prompt injection through tool results, contradictory
  instructions, tools that return plausible-but-wrong data.
- **Regression tracking** across runs, so a PR shows the eval delta rather
  than an absolute number with no baseline.

## License

MIT
