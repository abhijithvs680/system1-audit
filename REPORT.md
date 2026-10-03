# Coverage at a fixed error budget: what this harness measures, and what it does not

**Status: final. The project is closed at this commit.**

This is milestone 5 of `system1-audit`, written as a harness-only report. It is
the project's own stop condition being honoured rather than worked around: the
rule was to report the harness alone and drop the empirical claims if three
milestones passed with no runnable model environment. Five have.

Every number in this report comes from synthetic deciders in this repository.
**No model has been audited — not Jev, not Laya, not an LLM baseline.** Nothing
here is evidence about any real system's calibration, order sensitivity, or
operational value. Where that distinction is easy to lose, it is restated.

Regenerate every figure quoted below with:

```
PYTHONPATH=src python3 examples/report_numbers.py
```

It is seeded. Two runs on one commit produce identical output, and so do two runs
under different `PYTHONHASHSEED` values; both are asserted by
`tests/test_report_numbers.py` rather than assumed.

Three figures below do not come from that script, and are called out where they
appear: the two fit-optimism numbers in §3.6 are printed by `examples/demo.py`,
and one upper bound in §3.2 is quoted from an earlier milestone's note alongside
the value this repository reproduces today.

---

## 1. The question, and its status

"System-1" or typed decision models return a typed value — a label, an ordinal
score, a calibrated probability — in one forward pass instead of generating
text. Two arrived in September 2026: Jev (TypeSafe AI, closed) and Laya
(Apache-2.0, open). The marketing claim that matters operationally is
calibration, because a type-safe wrong answer delivered at 0.95 confidence
passes schema validation and goes straight into a routing decision.

The project posed three hypotheses. All three are **unanswered**, and this report
does not answer them:

| | Hypothesis | Status |
|---|---|---|
| H1 | Out-of-the-box confidence is substantially miscalibrated, and published ECE depends on a per-checkpoint fitted temperature | **Unanswered.** No checkpoint evaluated. |
| H2 | `choice` answers are not invariant to option permutation, and mass is biased toward display positions | **Unanswered.** No checkpoint evaluated. |
| H3 | The usable quantity is coverage at a fixed error budget, and voting over permutations buys more of it per unit latency than a larger model | **Half closed off, half unanswered.** See §4. |

### Why they are unanswered

Three blockers, re-tested on every run and again on 2026-10-03:

- **No model runtime.** `torch` and `transformers` are both absent from the
  execution environment.
- **No model credentials.** No API key for any provider, so the milestone 4 LLM
  structured-output baseline cannot run either.
- **arXiv:2609.30454 unread.** *Auditing System-1 Models on Biosecurity-Relevant
  Benchmarks: Calibration, Selective Prediction, and Permutation Instability in
  a Non-Generative Model.* Its title covers all three axes this harness
  measures, so it is potential prior art for the entire premise. arxiv.org has
  been refused by this environment's egress policy on every attempt
  (`CONNECT tunnel failed, response 403`), including today.

**Consequently no novelty is claimed anywhere in this project.** That gate was
never cleared, and the correct response to an unread prior-art candidate is
silence on novelty, not a hedge.

---

## 2. What "coverage at a fixed error budget" means, and why accuracy is the wrong number

A guardrail or routing layer does not consume accuracy. It asks: *if I answer
only above some confidence and escalate the rest, what error rate do I carry,
and how much traffic did I actually keep?* The pair is a risk-coverage curve,
and coverage at a fixed error budget is the single number a cost argument rests
on.

Accuracy cannot substitute for it, and the difference is not academic. Accuracy
never consults the confidence, so a change that destroys the model's ability to
rank its own answers — while leaving the top-1 label untouched — is invisible to
accuracy and fatal to coverage. §4 shows that happening.

### The defect this milestone had to fix first

Milestone 5 was planned as the write-up. It could not be written as planned,
because the library was reporting the wrong quantity for its own headline topic.

A risk-coverage curve walks one item at a time. A point on it can therefore stop
*inside* a group of equally confident items. No confidence cutoff can do that: a
cutoff answers every item at or above it — all of a tie group, or none. So a
curve point is a **bound** on what a deployment gets, not an operating point.

The two were conflated, and the consequence was worse than optimism. Measured:

| | |
|---|---|
| fixture | confidences `[0.9, 0.9, 0.9, 0.5]`, third item wrong |
| error budget | **0.00** |
| curve bound (`coverage_at_risk`) | 0.5000 |
| threshold the old code returned | **0.9** |
| error rate that cutoff actually carries | **0.3333** |
| largest coverage any real cutoff achieves | 0.0000 |

A caller setting the returned threshold carries a third of its answered traffic
wrong, against a budget of none of it — and the number was reported as *within*
budget. For a guardrail layer that is the wrong direction to be wrong in.

The fix makes the operating point the unit of reporting. `operating_point`
returns a threshold together with the coverage and error rate it realises,
selected on realised risk, so the three are consistent by construction and
applying the threshold reproduces the other two. `coverage_at_risk` is kept and
documented as a bound — the right answer to "can any cutoff beat this", the
wrong answer to "where do I set the threshold".

One further consequence, and the non-obvious one: this matters for a
**bootstrap even when every observed confidence is distinct**. Resampling draws
with replacement, so duplicates — and therefore ties — appear in nearly every
resample. An interval built on the prefix statistic bounds a quantity no cutoff
could deliver. `selective_coverage_interval` resamples the feasible quantity.

---

## 3. What the harness establishes

These are findings about measurement, not about models. They are the project's
actual yield, and several are corrections to its own earlier method.

### 3.1 A reported ECE cannot be read without `n` and the bin count

Binned ECE is positively biased: within-bin accuracy is a finite Bernoulli
sample and the metric takes the absolute value of the gap, so a model that is
*exactly* as right as it says it is still scores above zero. `ece_noise_floor`
simulates that floor by drawing correctness as `Bernoulli(confidence)` on the
observed confidence profile. On a fixed profile, moving only `n` and the bin
count:

| n | bins | mean floor | 95th percentile |
|---|---|---|---|
| 40 | 10 | 0.1429 | 0.2142 |
| 200 | 10 | 0.0639 | 0.0979 |
| 1000 | 10 | 0.0289 | 0.0433 |
| 200 | 2 | 0.0352 | 0.0700 |
| 200 | 20 | 0.0893 | 0.1200 |

The floor moves by roughly a factor of five between n=40 and n=1000 at a fixed
bin count, and by a factor of 2.5 between 2 and 20 bins at fixed `n`. So an ECE
quoted without both numbers cannot be compared against anything — including its
own post-temperature-fit counterpart, which is precisely the comparison that
motivated this project.

This is a statement about what a published figure permits a reader to conclude.
It is **not** a claim that any published figure is wrong. For the record, on the
Laya README — retrieved as the raw file on 2026-10-02, 100,070 bytes — raw ECE
appears as 0.175, 0.213, 0.246, 0.144, 0.466 and a multilingual 0.733/0.571, in
different sections describing different evaluation sets. Three of those state no
sample size and none states a bin count. The point is not that any is incorrect;
it is that "Laya's raw ECE" is not a single number and a reader cannot line any
two of them up.

### 3.2 An interval on a maximum is not a confidence interval

An earlier draft exposed an interval for `max_position_deviation`. On an
**order-invariant** decider — one with no position bias at all — it measured:

```
max |position deviation|   0.0000 [0.0016, 0.0266]
```

The point estimate sits outside its own bounds — below its own lower bound. The maximum of K absolute
deviations is positively biased under resampling, so the percentile interval
climbs above the point value, and `excludes(0.0)` would have reported position
bias on a model that has none. The field was removed rather than documented,
because an `Interval`-typed field that is not a valid interval is a trap. A test
asserts it stays removed and a second reproduces the defect, so the reason
survives the fix.

(The milestone 6 note in `RESEARCH_NOTES.md` records this interval's upper bound
as 0.0250. Regenerated for this report it is 0.0266; the point estimate and lower
bound match. The original run's resample count or seed is not recorded, so the
difference is not reconcilable from here. The figure above is the one this
repository reproduces today, which is why the report quotes a script.)

### 3.3 "Distinguishable from zero" is the wrong question for a rate

The project's original stop condition said to stop if the flip rate proved
indistinguishable from zero. That test can never fire. Under exact order
invariance the rate is *exactly* zero, so a single flipped item excludes zero
outright, at any sample size:

| unstable items | exact 95% interval | excludes zero? |
|---|---|---|
| 1 of 40 | `[0.0006, 0.1316]` | yes |
| 1 of 200 | `[0.0001, 0.0275]` | yes |
| 1 of 1000 | `[0.0000, 0.0056]` | yes |

The verdict is identical in all three rows. A "non-zero" finding therefore
carries no information about `n` and none about effect size — only the width
changes. The verdict was restated as `unstable_rate_exceeds(threshold)`: the
caller names the rate at which option instability would change a decision, and
the harness says whether the sample can put the rate above it. Naming that
threshold *before* the run is the same discipline the disjoint-split rule exists
to enforce.

### 3.4 A fixed item budget caps the finding before the run

Against a 5 percent threshold at 95 percent confidence and 80 percent power, the
smallest true instability rate resolvable at all:

| items | smallest resolvable rate |
|---|---|
| 40 | 0.1905 |
| 100 | 0.1340 |
| 300 | 0.0926 |
| 500 | 0.0814 |

At 40 items a rate below 0.19 cannot be put above a 5 percent threshold however
the audit comes out. Worth knowing before a model is called, not after.

### 3.5 Collecting more items can lose the power you registered

Exact binomial power is not monotone in `n`, because the rejection count is an
integer:

| items | unstable needed | power vs a true rate of 0.20 |
|---|---|---|
| 32 | 5 | 0.7956 |
| 33 | 5 | **0.8179** |
| 34 | 6 | **0.7004** |
| 38 | 6 | 0.7996 |
| 39 | 6 | 0.8200 |

A plan registering 33 items and then collecting 34 would be underpowered by its
own criterion with no step having been wrong. `plan_for_rate` therefore registers
`stable_from` — 39 here, the count beyond which no larger sample in range dips
below the requested power — rather than the first count to clear the bar.

### 3.6 The disjoint-split argument rests on NLL, not on ECE

An in-sample temperature minimises NLL over exactly the items it is reported on,
so no held-out temperature can beat it there. That makes an in-sample post-fit
NLL unfalsifiable, and it is the real argument for holding a split. ECE does not
inherit the guarantee: the fit targets NLL, and ECE is a binned statistic the fit
does not optimise, so in-sample ECE can come out *worse* than held-out. The demo
prints NLL optimism `+0.0011` against ECE optimism `-0.0152` on one audit — the
gap changes sign. An earlier README claim that fitting and reporting on the same
items "will understate ECE" was not reliably true and was corrected.

### 3.7 The harness recovers a planted defect, and says when it cannot

Temperature scaling recovers a planted sharpening up to a constant. The
synthetic decider raises its weights to `sharpness` and `apply_temperature`
divides `log p` by `T`, so both enter the softmax only through `sharpness / T`
and the fitted temperature must come out proportional to the planted defect.
Confirmed on a 200-item set: planted `sharpness` of 2, 4 and 8 recovers
temperatures 0.0706, 0.1411 and 0.2823 — a constant ratio of 0.0353. The
`sharpness = 1` case returns exactly 0.0500, which is `fit_temperature`'s
documented lower bound rather than a fitted value, and is the one row of the
four that says nothing.

The *constant* depends on the item set, so it is not a figure to quote; the
proportionality is the property, and it is what the test suite asserts. The
consequence worth keeping is that a temperature far from 1.0 on a real
checkpoint could be read as a magnitude, not just a direction.

Against a *weak* planted position bias (`position_weight=0.02` under heavy
noise), the family-wise verdict fired in 2 of 16 simulated audits at 60 items,
14 of 16 at 150, and 16 of 16 at 300. Under the null it fired in 1 of 40, inside
its nominal 5 percent, so the Bonferroni correction is controlling. The 60-item
shortfall was a power problem, not a false-positive problem — which is itself a
correction to how an earlier milestone read the same observation.

---

## 4. H3: what is closed off, and what is not

H3 said the usable quantity is coverage at a fixed error budget, and that voting
across option permutations buys more of it per unit latency than moving to a
larger model.

**The second half cannot be settled by this harness at all.** Coverage cannot
exceed 1.0, so coverage-per-pass cannot exceed `1 / passes`, and a one-pass
baseline already above `1/K` wins by construction. Pricing voting against a
*larger model* requires that model's own number. Only the within-model question
is reachable here.

**On the first half, three structural facts hold regardless of fixture:**

1. A modal vote over K display orders takes **at most K+1 distinct confidence
   values**; a probability average keeps one per item. Measured: 1 to 5 distinct
   values against 150.
2. **At one distinct value the gate is all-or-nothing.** The only available
   cutoff answers everything, so feasible coverage is 1.0 when the whole set is
   inside the budget and 0.0 otherwise — never anything between.
3. Coverage per pass is bounded by `1 / passes`, as above.

Measured across three fixtures differing only in planted skill, at a 10 percent
error budget over 150 items and 8 display orders:

| fixture | strategy | accuracy | bound | feasible | distinct |
|---|---|---|---|---|---|
| ignores the state | first_order | 0.2533 | 0.0067 | 0.0067 | 150 |
| | mean_probability | 0.2467 | 0.0000 | 0.0000 | 150 |
| | modal_vote | 0.2533 | 0.0067 | **0.0000** | **1** |
| mostly right | first_order | 1.0000 | 1.0000 | 1.0000 | 150 |
| | modal_vote | 1.0000 | 1.0000 | **1.0000** | **1** |
| partly right | first_order | 0.4467 | 0.1467 | 0.1467 | 150 |
| | mean_probability | 0.9733 | 1.0000 | 1.0000 | 150 |
| | modal_vote | 0.5867 | 0.5333 | **0.4733** | 5 |

Two things to read off it. The first fixture is the accuracy trap in full: the
modal vote holds accuracy at exactly 0.2533, identical to the single-pass
baseline, and takes feasible coverage to 0.0000 — for eight forward passes per
item. Accuracy is blind to it because accuracy never consults the confidence.
The second fixture has the same single distinct value and coverage 1.0000,
which is the all-or-nothing property in the other direction. The third shows the
§2 bound-versus-feasible gap on a realistic profile: 0.5333 claimed against
0.4733 achievable.

### A correction to this project's own milestone 8

Milestone 8 recorded that "a vote share is not a usable abstention gate" as a
negative result. **That framing was over-general and is withdrawn.** It
generalised from a single fixture. On two of the three fixtures above the vote
*improves* both accuracy and coverage. What survives is the structural claim —
the K+1 ceiling and the all-or-nothing consequence — not a directional one.

The reason is worth stating plainly, because it limits the whole section: these
deciders plant a per-display position term, which probability averaging cancels
**by construction**. That `mean_probability` reaches accuracy 0.9733 on the third
fixture is a property of the fixture, not a prediction about any model.

---

## 5. Honest limitations

In descending order of how much they constrain the above.

1. **No model has been audited.** Every number is from synthetic deciders.
   Nothing here predicts how any System-1 checkpoint behaves.
2. **No novelty is claimed**, because arXiv:2609.30454 — whose title covers all
   three audited axes — has never been readable from this environment. It may
   be prior art for the entire premise.
3. **The synthetic deciders are not a model of a model.** Their defects are
   dials, planted to test whether the harness recovers them. The dials were
   chosen to be recoverable, so "the harness recovers the defect" is weaker
   evidence than it sounds, and the *direction* of any aggregation effect on
   them is fixture-dependent (§4).
4. **`selective_coverage_interval`'s point estimate remains optimistically
   biased.** The tie bias is fixed; the in-sample maximisation is not. The
   interval says how much of the number is real; the point estimate does not.
5. **The tied-confidence case is reached by real systems but its frequency here
   is unmeasured.** A vote share over K passes takes at most K+1 values, and a
   coarse ordinal score few more — but how often that binds on a real checkpoint
   is unknown, because none was audited.
6. **`plan_id` is a reproducibility aid, not a security control.** Anyone who
   can edit a plan can recompute its digest. It catches a plan that drifted, not
   one rewritten deliberately.
7. **The Bonferroni correction is conservative** — the per-position deviations
   sum to zero and the comparisons are strongly dependent, so the real
   family-wise rate is below nominal and the test is less powerful than it
   could be.
8. **Measured power figures rest on 16 to 40 simulated audits.** `empirical_power`
   returns a Wilson interval for exactly that reason: a power estimated from 16
   simulations is itself a measurement on 16 units.

---

## 6. What it would take to finish the empirical work

In order, and all of it blocked today:

1. **Read arXiv:2609.30454.** Gates any novelty claim. Needs network access to
   arxiv.org. If it already reports these three axes with a stronger method, the
   right outcome is to say so and stop.
2. **Milestone 2 — adapter for an open checkpoint.** Needs a GPU or patient CPU
   with `torch` and `transformers`, and the Laya weights from HuggingFace
   (Apache-2.0). Must register a plan first and quote its `plan_id` with the
   results: 39 items for the unstable-rate verdict against a 0.05 threshold at
   an assumed 0.20 rate, and 150 for the per-position criterion at the weak
   planted bias. Must report `n` and the bin count with any ECE, and the noise
   floor beside it.
3. **Milestone 4 — LLM structured-output baseline** on the same items. Needs
   credentials for one provider. Report threshold-feasible coverage, not the
   curve bound.

No adapter to Laya is included here on purpose: the package's Python call
signature could not be verified from this environment, and guessing at an API
would put untested code in the repository wearing the appearance of a tested
integration.

---

## 7. Sources

Retrieval dates are stated because several of these could not be retrieved at
all, and that is part of the record.

| Source | Status |
|---|---|
| [Laya repository](https://github.com/NandhaKishorM/laya) | Retrieved 2026-09-28, re-checked 2026-09-30 |
| [Laya README, raw file](https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md) | Retrieved 2026-10-02, 100,070 bytes — the reading that settled the ECE discrepancy |
| [Laya weights](https://huggingface.co/convaiinnovations/laya) | Not fetched; no model environment |
| [TypeSafe AI / Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | **Never retrieved** — blocked by egress policy |
| [arXiv:2609.30454](https://arxiv.org/abs/2609.30454) | **Never retrieved** — refused on 2026-09-29, 09-30, 10-01, 10-02 and 10-03 |

Jev's published performance claims are **second-hand throughout this project**,
from search-result summaries rather than a page that was read. They are not
relied on for anything above.

---

## 8. Project status

Closed at this commit. Milestones 1, 3, 5, 6, 7, 8 and 9 are complete;
milestones 2 and 4 are closed **unstarted and blocked**, not abandoned — the
exact conditions to resume them are in §6.

What the project produced is a dependency-free audit harness for typed decision
models, with the statistical discipline wired into the library rather than left
to the caller: disjoint fit/report splits enforced by the split machinery,
an interval on every audited quantity, a noise floor beside every ECE, a
pre-registered sample size and threshold, aggregation priced per forward pass,
and a selective-prediction operating point a deployment can actually set.

What it did not produce is a single measurement of a real model. The honest
summary is that the harness is ready and the audit never ran.
