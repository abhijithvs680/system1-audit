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

## Status

Early. The metric layer is implemented and unit-tested; no real model has been
audited yet. See `RESEARCH_NOTES.md` for the hypothesis, the sources, and what
is deliberately not claimed.

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
from system1_audit import calibration_report, fit_temperature, risk_coverage_curve

report = calibration_report(probabilities, correct_index)
print(report.ece, report.adaptive_ece, report.brier, report.overconfidence)

temperature = fit_temperature(fit_probs, fit_truth)   # fit on a held-out split
selective = risk_coverage_curve(confidences, correct)
print(selective.coverage_at_risk(target_risk=0.02))
```

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

## Known limitations

- Only the `choice` primitive is implemented. `score` (ordinal) and `noul`
  (calibrated true/false) audits are not written yet.
- `fit_temperature` uses `log p` as a pseudo-logit, because most decision APIs
  expose probabilities rather than logits. This is exact up to the softmax's
  shift invariance, but it cannot recover information already lost to rounding
  or truncation in the API response.
- Temperature fitted and reported on the same split will understate ECE. The
  library does not enforce a split; the caller must.
- Permutation sampling is exhaustive only for small option counts; above that
  it is a seeded random sample, so flip rate is an estimate with sampling error
  that the harness does not currently report a confidence interval for.
- No real model has been audited. Every number in `examples/demo.py` comes from
  a synthetic decider and describes the harness, not any product.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`. Chosen to match the licence of the
open checkpoints this harness is intended to audit.
