# PRD: Agent Evaluation Harness

**Status:** v0.1, shipped
**Owner:** Vinay Manchanda

---

## 1. Problem

Teams ship LLM agents faster than they can tell whether the agents work.

The typical loop is: change a prompt, run three or four questions by hand,
decide it "seems better", ship. That process has three failure modes that
compound.

**It cannot detect regressions.** A prompt edit that fixes currency handling
and quietly breaks multi-step chaining looks like a win, because nobody
re-ran the chaining case.

**It cannot distinguish right from lucky.** An agent that recalls an exchange
rate from pretraining and one that calls the FX tool produce the same string
today. They diverge the day the rate moves, and by then the agent is in
production.

**It cannot see variance.** Models are stochastic. A behaviour that works
four times out of five reads as "working" in manual testing and as a 20%
incident rate in production.

The cost of these is not evenly distributed. In a financial context, the
domain this harness uses, a confidently wrong number is worse than a refusal,
because it is actionable.

## 2. Users

| user | needs | success looks like |
|---|---|---|
| Engineer changing a prompt or tool schema | fast signal on whether they broke something | runs the suite locally in seconds, sees a diff |
| Reviewer on the PR | evidence, not assertions | CI posts a pass rate; regressions block merge |
| PM deciding whether to ship | a number and a known failure profile | "94% at n=10, fails on ambiguous currency" |

## 3. Goals

**G1. Grade the route, not just the answer.** Every case can assert which
tools were called and with what arguments. This is the differentiator; an
outcome-only harness cannot catch the lucky-guess agent.

**G2. Run in CI with no API key and no spend.** A suite that costs money and
takes minutes gets run before releases. A suite that is free and instant gets
run on every commit, which is where it actually catches things.

**G3. Make variance visible.** Report pass *rates* over N attempts, never a
single boolean.

**G4. Cases in data, not code.** Adding coverage should be a JSON edit, so
non-engineers can contribute cases.

**G5. Grade failure handling.** Tool errors are normal. Recovery behaviour
is a product requirement and must be assertable.

## 4. Non-goals

- **Not a benchmark.** This measures *one* agent against *its own* spec, not
  models against each other.
- **Not open-ended quality scoring.** No judge model in v0.1. An unvalidated
  LLM judge relocates the trust problem rather than solving it; judges land
  once there is a human-labelled set to validate them against.
- **Not a framework.** No plugin system, no abstraction over providers beyond
  the two that exist. Framework generality is the enemy of a harness people
  actually read.

## 5. Requirements

| # | Requirement | Priority | Status |
|---|---|---|---|
| R1 | Tool-use loop with full trace capture | P0 | done |
| R2 | Outcome graders (numeric, substring) | P0 | done |
| R3 | Trajectory graders (tools used, args, turn count) | P0 | done |
| R4 | Mock provider for offline runs | P0 | done |
| R5 | Cases defined in JSON | P0 | done |
| R6 | `--repeat` with pass-rate reporting | P0 | done |
| R7 | Non-zero exit below threshold | P0 | done |
| R8 | Error-recovery grader | P1 | done |
| R9 | Token capture per run | P1 | done |
| R10 | Cost and latency budgets as assertable checks | P2 | planned |
| R11 | Judge graders validated against human labels | P2 | planned |
| R12 | Cross-run regression tracking | P2 | planned |

## 6. Key design decisions

**Tools are pure and offline.** Determinism in the environment is what makes
a failing case interpretable. If both the model and the tool output could
move, a red case tells you nothing about which one broke.

**Tool errors return to the model rather than raising.** Raising ends the run
and destroys the signal. Returning `is_error: true` makes recovery observable,
and therefore gradeable, which is what R8 needs.

**Mock replays scripted turns through the real loop.** Only the model is
faked. The loop, the tools, and the graders are the production ones, so mock
runs exercise real code paths. A mock failure means the harness is broken.

**Hitting `max_turns` is a recorded failure, never a silent retry.** Runaway
loops are simultaneously a correctness bug, an availability bug, and a
billing bug. They should be loud.

## 7. Success metrics

| metric | definition | target |
|---|---|---|
| Suite pass rate | passing attempts / total attempts, n=10 live | >= 0.95 to release |
| Trajectory pass rate | passing trajectory checks / total | >= 0.95 |
| Flaky case count | cases with 0 < rate < 1 | 0 at release |
| Harness runtime (mock) | wall clock, full suite | under 5s |
| Time to add a case | edit to running | under 5 min |

The gate that matters is **flaky case count**. A case at 7/10 is not a partial
pass; it is a defect with a 30% trigger rate, and rounding it in either
direction hides that.

## 8. Open questions

1. What tolerance is right for numeric answers? Too tight and formatting
   changes fail; too loose and real errors pass. Currently per-case.
2. Should trajectory checks be advisory when a *better* route exists? An agent
   that solves a case in one tool call instead of two currently fails
   `tools_used`. Arguably it should be flagged for review, not failed.
3. How do we version expected values when the frozen reference data changes?
