# system1-audit

An audit harness for **typed decision models** — models that return a typed,
probabilistic answer in a single forward pass instead of generating text.

The vendor pitch for this model class is latency and cost. Those are easy to
measure and are not what breaks in production. This harness measures the three
properties that decide whether such a model can actually carry a routing or
guardrail decision:

| Audit | Question it answers |
|---|---|
| Calibration | Is the confidence meaningful *before* a temperature is fitted to it? |
| Option-order sensitivity | Does the answer change when the same options are shown in a different order? |
| Selective prediction | If low-confidence items are escalated, what error rate remains, and how much traffic was actually saved? |
| Significance | Is any of the above distinguishable from zero on this many items? |
| Pre-registration | Was this many items decided before the run, and enough to have seen the effect? |

Calibration is reported twice, raw and after temperature fitting, because the
post-fit figure is the one vendors publish. The temperature is fitted on items
disjoint from the ones it is reported on, and the harness also reports what the
in-sample shortcut would have bought, so the difference is visible instead of
assumed.

## Status

Early. The metric layer is implemented and unit-tested, temperature fitting is
held to a disjoint fit/report split by the library rather than by the caller's
discipline, every audited quantity carries an interval so a point estimate
cannot be over-read, and the sample size and threshold are fixed in a
pre-registered plan before a model is called. No real model has been audited
yet. See
`RESEARCH_NOTES.md` for the hypothesis, the sources, and what is deliberately
not claimed.

## Why these three

Latency and price-per-token are the wrong unit for a decision. If a router
answers in 33 ms but its confidence does not rank its own errors, you cannot
set an escalation threshold, and without a threshold the cheap model does not
remove any work from the expensive one — it just adds a hop.

Order sensitivity matters for the same reason. A typed model is supposed to
answer a question about *state*. If permuting the option list changes the
answer, some of the decision came from the list rather than the state, and
the accuracy number measured under one fixed ordering is optimistic.

## Install

No third-party dependencies. Python 3.10+.

```bash
git clone <repo-url> && cd system1-audit
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 examples/demo.py
```

The absence of dependencies is intentional: a harness whose own arithmetic
cannot be checked without installing a GPU stack does not get run.

## Usage

Implement one method and the whole harness applies:

```python
from system1_audit import ChoiceDecision, ChoiceQuestion, audit_dataset

class MyDecider:
    def decide_choice(self, state: str, labels):
        probs = my_model.predict(state, labels)   # aligned to `labels`
        return ChoiceDecision(tuple(labels), tuple(probs))

questions = [
    ChoiceQuestion("refund not received", ("billing", "bug", "sales"), "billing"),
]
audit = audit_dataset(MyDecider(), questions, n_permutations=8)
print(audit.position_bias, audit.mean_flip_rate)
```

Calibration and selective prediction take plain probability vectors, so they
work against any model, including an LLM asked for structured output:

```python
from system1_audit import calibration_report, risk_coverage_curve

report = calibration_report(probabilities, correct_index)
print(report.ece, report.adaptive_ece, report.brier, report.overconfidence)

selective = risk_coverage_curve(confidences, correct)
print(selective.coverage_at_risk(target_risk=0.02))
```

To report a post-temperature number, let the harness hold the split. The
assignment hashes each item's own identifier, so it is identical across
processes and does not move when the dataset grows:

```python
from system1_audit import deterministic_split, held_out_calibration

split = deterministic_split(item_ids, fit_fraction=0.5)
audit = held_out_calibration(probabilities, correct_index, split)

print(audit.raw.ece)          # no scaling
print(audit.calibrated.ece)   # temperature fitted on the disjoint fit side
print(audit.nll_optimism)     # what fitting in-sample would have appeared to save
```

Pick the sample size before running the audit, not after reading it. The
threshold, the confidence, the power and the item count go into a plan, and the
plan hands back an id derived from them, so a result can be checked against the
plan it claims to answer:

```python
from system1_audit import order_sensitivity_significance, plan_for_rate

plan = plan_for_rate(
    name="H2-rate",
    hypothesis="unstable-item rate exceeds 5 percent",
    threshold=0.05,     # the rate at which order instability changes a decision
    assumed_rate=0.20,  # the effect the plan is powered to find
)
print(plan.plan_id, plan.n_items, plan.minimum_unstable())   # 1f90e5bd69bf 39 6

significance = order_sensitivity_significance(audit, confidence=plan.confidence)
outcome = plan.evaluate(significance)
print(outcome.conclusion)            # supported | not_supported | inconclusive_underpowered
print(outcome.falsifies_hypothesis)  # True only for not_supported
```

`inconclusive_underpowered` is the whole point of the module. An interval that
fails to exclude the null looks identical whether the effect is absent or the
sample was too small to see it, and only the first falsifies anything.

## What the metrics mean

- **ECE** — equal-width binned gap between confidence and accuracy. The number
  vendors usually publish. It is sensitive to binning and flattering when a
  model crowds into one bin.
- **Adaptive ECE** — the same gap over equal-mass bins. Reported alongside ECE
  because the two disagreeing is itself a finding.
- **MCE** — the worst single bin, which is what a threshold actually hits.
- **Brier / NLL** — proper scoring rules, so they cannot be gamed by binning.
- **Flip rate / TV distance** — how much a decision moves across display
  orders, measured after realigning probabilities to a canonical label order.
- **Position bias** — mean probability mass landing on each display position.
  An order-invariant model puts `1/K` on each.
- **AURC** — area under the risk-coverage curve; lower means confidence ranks
  errors better.
- **NLL optimism** — NLL advantage the in-sample temperature holds over the
  held-out one on the same items. Non-negative by construction: the in-sample
  temperature minimises NLL over exactly those items, so nothing can beat it
  there. This is the number that shows an in-sample post-fit figure is
  unfalsifiable rather than merely noisy.
- **ECE optimism** — the same comparison on ECE. Measured and reported, but
  **not** guaranteed non-negative, because the fit targets NLL and ECE is a
  binned statistic the fit does not optimise. The demo currently prints a
  negative value for it.
- **ECE noise floor** — the ECE a *perfectly calibrated* model would have
  scored on the same confidences, simulated by drawing correctness as
  `Bernoulli(confidence)`. Binned ECE is positively biased, so this floor is
  above zero, and it is the number a reported ECE has to beat before it means
  anything. It falls with `n` and rises with bin count: on the demo's
  confidence profile the 95th percentile floor is 0.19 at n=40, 0.085 at
  n=200 and 0.037 at n=1000.
- **Interval** — a point estimate with bounds and the method that produced
  them. `excludes(0.0)` is the operational reading of "distinguishable from
  zero" at that level. It is not a p-value.
- **Minimum unstable items** — the integer that characterises the whole rate
  verdict: see this many unstable items or more and it fires, see fewer and it
  does not. Known before the run, so the audit's outcome is not a surprise
  about its own decision rule.
- **Power** — probability the verdict fires if the assumed effect is real.
  Exact binomial, not a normal approximation, which at audit-sized samples is
  wrong in the flattering direction.
- **Detectable rate** — the smallest true rate a fixed item budget could put
  above the threshold at all. On a 5 percent threshold at 95 percent
  confidence and 80 percent power: 0.19 at 40 items, 0.13 at 100, 0.093 at 300
  and 0.081 at 500. Below those, the audit cannot produce the finding however
  it comes out.
- **`stable_from`** — the sample size beyond which no larger sample dips back
  below the requested power. The number to register, because the first
  qualifying `n` does not have that property (see the limitation below).

## Known limitations

- Only the `choice` primitive is implemented. `score` (ordinal) and `noul`
  (calibrated true/false) audits are not written yet.
- `fit_temperature` uses `log p` as a pseudo-logit, because most decision APIs
  expose probabilities rather than logits. This is exact up to the softmax's
  shift invariance, but it cannot recover information already lost to rounding
  or truncation in the API response.
- `held_out_calibration` enforces a disjoint fit/report split, but only for the
  temperature. Nothing stops a caller choosing a split after seeing the
  results, or reporting the best of several salts; the salt used is an argument
  and should be recorded with any number reported.
- Splitting requires a unique, non-empty identifier per item, so
  `ChoiceQuestion`'s default blank `item_id` is rejected rather than collapsed
  into one bucket.
- Hash-based assignment only approximates the requested `fit_fraction`; the
  realised share is on `Split.fit_fraction` and should be read, not assumed.
- An earlier version of this file claimed that fitting and reporting on the same
  items "will understate ECE". That is not reliably true and has been corrected:
  measured over the synthetic deciders in the test suite, the NLL advantage is
  always present but the ECE gap changes sign depending on the decider, because
  the temperature is fitted against NLL rather than ECE.
- Permutation sampling is exhaustive only for small option counts; above that
  it is a seeded random sample. `order_sensitivity_significance` now reports an
  interval for the flip rate, but that interval covers variation across *items*
  only. Uncertainty from sampling the permutations themselves is not separated
  out, so on a large option count the interval is narrower than the full
  sampling error.
- "Distinguishable from zero" is the wrong question for a flip rate, and asking
  it was a mistake this project made in its own stop condition. Under exact
  order invariance the rate is exactly zero, so a single flipped item excludes
  zero at any `n` — the exact interval for 1 item in 40 is `[0.0006, 0.1316]`,
  which excludes zero just as firmly as 1 in 1,000,000 would. The verdict is
  therefore `unstable_rate_exceeds(threshold)`, and the caller has to name the
  rate at which order instability would change a decision.
- `mean_flip_rate`'s bootstrap interval collapses to zero width when every item
  has the same flip rate, which reads as precision it does not have. The
  verdict rests on the exact binomial interval for the unstable-item count, not
  on that one.
- No interval is offered for `max_position_deviation`. A percentile bootstrap
  interval for a maximum does not contain its own point estimate — on an
  order-invariant decider it measured `0.0000 [0.0016, 0.0250]`, because the
  maximum of `K` absolute deviations is positively biased under resampling.
  Tested against zero it reported position bias on a model with none, so the
  field was removed rather than documented.
- Per-position intervals are Bonferroni-corrected so that "is *any* position off
  uniform" holds family-wise. Bonferroni is conservative here, since the
  deviations sum to exactly zero and the comparisons are strongly dependent.
- The family-wise verdict can fire on the wrong position. With a weak planted
  bias (`position_weight=0.02`) against heavy noise at n=60, the sign of the
  deviation on the biased position was recovered in all 8 trials measured, but
  only 2 of 8 resolved it, and 1 attributed it to a different position — the
  same trial fires identically with the bias removed, so that firing is a false
  positive, not a misattribution of a real effect.
- Exact binomial power is **not monotone in the item count**, so collecting
  more items can lose the power a plan registered. The rejection count is an
  integer: at a 5 percent threshold and 95 percent confidence, 33 items need 5
  unstable items and reach power 0.818, while 34 items need 6 and fall to
  0.700. `plan_for_rate` therefore registers `stable_from` rather than the
  first qualifying count. Within the scanned range this is 39 items for that
  threshold against an assumed rate of 0.20.
- The pre-registration layer covers the unstable-rate verdict only. The
  per-position bias verdict rests on a percentile bootstrap with no closed-form
  power, so `empirical_power` measures it by simulation and reports a Wilson
  interval. Measured on the synthetic decider at 95 percent confidence, with
  jitter on and no planted bias, that verdict fired in 1 of 40 simulated
  audits at 60 items; with `position_weight=0.05` planted it fired in 20 of 20
  at 60, 150 and 300 items. The weak 0.02 case above is a power problem rather
  than a correction problem: it fired 2 of 16 at 60 items, 14 of 16 at 150 and
  16 of 16 at 300, so 150 items is where that bias becomes resolvable.
- `plan_id` is a reproducibility aid, not a security control. Anyone who can
  edit a plan can recompute its digest; it catches a plan that drifted, not a
  plan that was rewritten on purpose.
- `required_items_for_rate` establishes `stable_from` only within `max_items`,
  which defaults to 500. A plan needing more has to say so and pay the scan.
- `selective_coverage_interval`'s point estimate is optimistically biased,
  because `coverage_at_risk` maximises over a curve computed on the same data.
  The interval says how much of the number is real; the point estimate does not.
- The ECE noise floor tests the joint null "perfectly calibrated, given this
  confidence profile". Clearing it says something beyond binning noise is
  present, not in which direction — `CalibrationReport.overconfidence` is the
  signed quantity.
- No real model has been audited. Every number in `examples/demo.py` comes from
  a synthetic decider and describes the harness, not any product.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`. Chosen to match the licence of the
open checkpoints this harness is intended to audit.
