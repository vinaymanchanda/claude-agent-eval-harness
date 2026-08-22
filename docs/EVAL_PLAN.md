# Eval Plan

How this agent is measured, what the numbers mean, and what blocks a release.

---

## 1. What we are measuring

The agent answers financial questions using four tools. "Working" decomposes
into four claims, each of which needs its own evidence:

| # | Claim | Evidence |
|---|---|---|
| C1 | It produces correct answers | outcome graders |
| C2 | It reaches them by calling the right tools | trajectory graders |
| C3 | It survives tool failures | error-recovery cases |
| C4 | It does this *reliably*, not on average | pass rate at n >= 10 |

C2 is the one usually skipped and the one that matters most. See
`test_grader_catches_agent_that_skipped_the_tool` for the concrete case: an
agent that produces the right number without calling the tool passes C1 and
fails C2. Only C2 catches it.

## 2. Metric definitions

**Attempt** - one full run of one case. Provider called until it stops
requesting tools, or `max_turns` is hit.

**Attempt pass** - *every* check on that case passed. Deliberately strict:
partial credit hides the specific thing that broke.

**Case pass rate** - passing attempts / attempts, over `--repeat`. This is the
headline per-case number.

**Suite pass rate** - total passing attempts / total attempts across cases.
Unweighted, so it moves with case mix; read it alongside the per-case table
rather than alone.

**Outcome pass rate** - passing outcome checks / total outcome checks.

**Trajectory pass rate** - passing trajectory checks / total trajectory checks.

The split between the last two is diagnostic. High outcome plus low trajectory
means the agent is guessing correctly, the most dangerous state, because it
looks healthy from the outside and degrades silently. Low outcome plus high
trajectory means the tools or the expected values are wrong, not the agent.

**Flaky case** - any case with `0 < rate < 1`. Tracked separately because
neither rounding is honest.

## 3. Coverage

Current suite: 7 cases across 4 behaviour classes.

| class | cases | what it protects |
|---|---|---|
| Happy path, single tool | 2 | basic tool selection and argument construction |
| Grounding | 1 | refuses to do arithmetic unaided |
| Multi-step chaining | 1 | output of one tool feeds the next |
| Error handling | 3 | retry on bad args; honest refusal on unsupported input |

Known gaps, in priority order:

1. **Ambiguity.** No case where the right move is to ask a clarifying question.
2. **Adversarial input.** No prompt injection through tool results.
3. **Long chains.** Nothing beyond two tool calls deep.
4. **Refusal boundaries.** No case checking the agent declines to give
   personalised financial advice, which the system prompt does not currently
   address either.

Gap 4 is a product gap, not just a coverage gap: the eval is missing because
the requirement was never written down.

## 4. Running

```bash
# CI: offline, free, every commit
python -m evals.runner --provider mock --threshold 1.0

# Development: one live case, fast iteration
python -m evals.runner --provider anthropic --case fx_basic

# Release gate: variance measurement
python -m evals.runner --provider anthropic --repeat 10 \
    --threshold 0.95 --json-out release-run.json
```

## 5. Release gate

A change ships when all of these hold:

| gate | threshold |
|---|---|
| Mock suite | 100%, no exceptions, it is deterministic |
| Unit tests | 100% |
| Live suite pass rate, n=10 | >= 95% |
| Trajectory pass rate, n=10 | >= 95% |
| Flaky cases | 0 |
| New behaviour | has a case before the code merges |

The mock threshold is 1.0 because mock is deterministic: a mock failure is a
harness bug, and there is no legitimate reason to tolerate one.

The live threshold is 0.95 rather than 1.0 because demanding perfection from
a stochastic system produces one of two outcomes, both bad: a suite people
disable, or a suite whose cases get quietly loosened until it passes.

## 6. Interpreting a red run

```
1. Did the mock suite fail too?
   yes -> harness or tool bug. Fix that first; live results are meaningless.
   no  -> continue.

2. Is it outcome or trajectory that dropped?
   outcome only     -> answer wrong. Check tolerances and expected values
                       before blaming the model.
   trajectory only  -> right answer, wrong route. Usually a prompt change
                       that weakened a tool-use instruction.
   both             -> model, model version, or tool schema change.

3. Is the case flaky (0 < rate < 1)?
   yes -> reproduce with --repeat 20. Read the traces in --json-out;
          look at which turn diverges, not just the final answer.
```

## 7. Cost

Mock runs are free. A live run is roughly 7 cases x ~2.5 turns x ~700 tokens.
At `--repeat 10` that is on the order of 120k tokens per full release run:
small enough to run per release, large enough that it should not run per
commit. That asymmetry is exactly why the mock provider exists.

## 8. Review cadence

- Every new agent behaviour ships with a case, in the same PR.
- Every production incident becomes a case before the fix merges.
- Coverage gaps in section 3 reviewed monthly and re-prioritised.
