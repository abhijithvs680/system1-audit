# Research notes

Working notes for this project: what prompted it, what is actually evidenced,
what is only claimed, and how the question gets settled.

Sources were checked on **2026-09-28**. Where a source could not be retrieved
directly it is marked as such rather than cited as read.

---

## Background

A model class marketed as "System 1" or "decision" models arrived in
September 2026: instead of generating text autoregressively, the model returns
a typed value — a multiple-choice label, an ordinal score, or a calibrated
probability — in a single forward pass.

Two implementations prompted this project.

### Jev (TypeSafe AI) — closed

- Announced September 2026. Returns typed decisions in one parallel pass; the
  advertised output primitives are a truth probability, an ordinal score, and a
  multiple-choice answer.
- Headline claims reported in coverage: roughly 40x-200x faster than frontier
  LLMs on comparable tasks, around $0.042 per million input tokens with free
  output, and that the model "mathematically cannot hallucinate or produce
  type errors". Training method is described as reinforcement learning for
  calibrated decisions.
- **Not verified here.** `typesafe.ai` and the Tom's Hardware and MarkTechPost
  articles were all blocked by this environment's network egress policy. The
  claims above come from search-result summaries, not from a page I retrieved.
  Treat every number in this subsection as second-hand until the vendor blog is
  read directly.

### Laya (Convai Innovations, Nandakishor M) — open, Apache-2.0

Retrieved directly from <https://github.com/NandhaKishorM/laya> on 2026-09-28.
These figures are quoted from that README, which is the authors' own
self-report, not an independent measurement:

- Non-autoregressive encoder models. Three checkpoints: `laya`
  (ModernBERT-large, 421M, 512 ctx), `laya-multilingual` (mmBERT-base, 322M,
  1,024 ctx), `laya-typed-decisions` (ModernBERT-large, 421M, 1,024 ctx).
- Output primitives: `choice`, `score`, `noul` (calibrated P(true)).
- Latency on a Tesla T4: 32.8 ms single-question multilingual, 39.5 ms English;
  batched at 10, 7.2 ms and 15.9 ms per question.
- Calibration **after temperature fitting**: ECE 0.081 English, 0.106
  multilingual; Brier 0.062 on the typed-decisions checkpoint.
- Calibration **before fitting**: ECE 0.213 English, 0.314 multilingual.
- Apache-2.0. Weights on HuggingFace under `convaiinnovations/`.
- The README states its own limitations, including near-chance zero-shot
  performance on typed decisions for the base checkpoints (0.36 / 0.35 against
  a 0.318 random baseline), weak ordinal `score` performance (SST-5 accuracy
  0.372), collapse outside English for the English checkpoint, and — the line
  that motivated this project — that `noul` "can follow option labels instead
  of state content" on the English checkpoint.

A related arXiv preprint surfaced in search, *"Auditing System-1 Models on
Biosecurity-Relevant Benchmarks: Calibration, Selective Prediction, and
Permutation Instability in a Non-Generative Model"* (arXiv:2609.30454).
**Not read** — arxiv.org is blocked by this environment. It is listed here as
prior art to check before claiming novelty, because its title covers all three
of the axes this harness measures.

---

## Evidence versus marketing

Separating the two, because the gap is where the project lives.

| Claim | Status |
|---|---|
| Single forward pass, no autoregression | Architecturally evidenced — encoder models, checkpoints public |
| Large latency advantage over frontier LLMs | Plausible and mechanically expected; all published numbers are vendor self-reports on vendor-chosen hardware and tasks |
| "Cannot hallucinate" | True only in the narrow sense that a constrained output head cannot emit an off-schema token. It says nothing about the answer being *right*, and nothing about the confidence being meaningful |
| Calibrated confidence | The one claim that is both central and checkable. Laya's own README puts raw ECE at 0.213-0.314, i.e. poorly calibrated until a temperature is fitted |
| Order invariance | Not claimed by anyone, not measured by anyone publicly, and contradicted by Laya's own note about following option labels |

The "cannot hallucinate" framing is the part most likely to mislead an
engineering team. A type-safe wrong answer delivered with 0.95 confidence is
worse operationally than a text model that rambles, because it passes schema
validation and goes straight into the routing decision.

---

## Hypothesis

**H1 (calibration).** Out-of-the-box confidence from a System-1 typed decision
model is substantially miscalibrated, and the published ECE figures depend on a
temperature fitted per checkpoint. Falsified if raw ECE is below ~0.05 on a
held-out task the vendor did not select.

**H2 (order sensitivity).** For `choice` questions, top-1 answers are not
invariant to permutation of the option list, and probability mass is
measurably biased toward specific display positions. Falsified if the flip
rate is at chance-level zero and per-position mass sits within sampling noise
of `1/K`.

**Restated after milestone 6.** "At chance-level zero" is not testable for a
flip rate: under exact order invariance the rate is exactly zero, so a single
flipped item excludes zero at any `n`. H2 is falsified if the unstable-item
rate cannot be put above a stated threshold and no per-position deviation
clears a family-wise interval. See finding 1 under milestone 6.

**H3 (operational value).** The usable quantity for a routing layer is not
accuracy but coverage at a fixed error budget. H3 says that voting across
option permutations buys more accuracy per unit latency than moving to a
larger model — testable once H2's magnitude is known, because a model with no
order sensitivity has nothing to vote over.

All three are falsifiable and none requires access to Jev.

---

## Method

1. Implement the metric layer with no dependencies, and validate it against a
   synthetic decider with *deliberately planted* defects — known position
   weight, known overconfidence, known skill. If the harness cannot recover a
   bias that was injected on purpose, its numbers on a real model mean nothing.
   **Done. 55 unit tests.**
2. Audit an open checkpoint (Laya, Apache-2.0) on a public classification set,
   reporting raw and post-fit calibration on *disjoint* splits.
3. Audit an LLM asked for the same typed decision via structured output, as the
   baseline the model class claims to replace.
4. Report latency and cost alongside coverage at a fixed error budget, not
   alongside accuracy.

### Evaluation rules

- Temperature is fitted on a split disjoint from the one it is reported on.
  Enforced by `splits.held_out_calibration` since milestone 3, not left to the
  caller. The in-sample number is reported alongside the held-out one, so the
  difference is visible rather than asserted. The argument rests on the NLL
  guarantee, not on an ECE effect -- see the milestone 3 findings below.
- Every permutation audit is seeded and reproducible across processes.
- No audited quantity is reported without an interval, and no ECE is reported
  without the noise floor for its own `n` and bin count. Enforced by
  `significance.py` since milestone 6.
- The threshold, the confidence, the power and the item count are fixed in a
  `PreregisteredPlan` before a model is called, and a result quotes the
  `plan_id` it answers. Enforced by `prereg.py` since milestone 7. A null
  result on a sample without the power to find the assumed effect is reported
  as `inconclusive_underpowered` and may not be read as falsifying H2.
- A vote over several display orders is charged the forward passes it used, and
  a gain over a cheaper strategy is reported as a *paired* interval over the
  same items. Enforced by `voting.py` since milestone 8. Coverage is reported as
  the number an actual confidence threshold delivers, not the prefix-curve
  number, which can stop inside a group of tied confidences and so cannot be
  implemented.
- Vendor self-reported numbers are never mixed into a results table with
  numbers measured here.
- Hardware, batch size, and checkpoint are recorded with every latency figure,
  because they dominate it.

### Stop condition

Stop and write up if any of: H2's flip rate is indistinguishable from zero on
the open checkpoint (the interesting finding disappears); arXiv:2609.30454
already reports the same permutation result with a stronger method (no novelty
left, cite it instead); or three milestones pass without a runnable GPU
environment (report the harness alone and drop the empirical claims).

**The third clause is now met.** Milestones 6, 7 and 8 all ran with no model
environment — re-tested on 2026-09-30, 2026-10-01 and 2026-10-02. Milestone 5
should therefore be written up as a harness-only report, with the empirical
H1/H2/H3 claims dropped rather than left pending, unless a model environment
becomes available first. See the milestone 8 findings.

---

## Prior art and licensing

- Calibration metrics (ECE, adaptive ECE, MCE, Brier, temperature scaling) are
  long-standing published methods, reimplemented here from their definitions
  rather than copied, so the harness stays dependency-free and unit-testable.
- Selective prediction and risk-coverage curves likewise predate this work.
- The novelty claimed here is **narrow and should stay narrow**: applying an
  order-permutation audit together with raw-versus-fitted calibration to the
  new typed-decision model class, with a harness that validates itself against
  planted defects. arXiv:2609.30454 may already cover part of this and must be
  read before any novelty is claimed anywhere public.
- No vendor code is vendored or derived from. Laya is Apache-2.0; if an adapter
  to it is added later, it will depend on the published package rather than
  copying source.
- No employer or client code, data, or internal benchmark is used in this
  project.

---

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Dependency-free metric layer; self-validating tests against planted defects | Done |
| 2 | Adapter for an open checkpoint; needs a GPU or patient CPU environment | Closed, blocked -- no model environment |
| 3 | Disjoint fit/report split discipline, enforced by the library | Done |
| 4 | LLM structured-output baseline on the same items | Closed, blocked -- no credentials |
| 5 | Write-up: coverage at a fixed error budget, with honest limitations | Done -- `REPORT.md` |
| 6 | Significance layer: an interval on every audited quantity, and an ECE noise floor | Done |
| 7 | Pre-registration and power: the sample size and threshold fixed before the run | Done |
| 8 | Vote aggregation priced per forward pass, with a paired interval on the gain | Done |
| 9 | A selective-prediction operating point a deployment can actually set | Done |

No adapter to Laya is included in this commit on purpose. The package's Python
call signature could not be verified from this environment, and guessing at an
API would put unverified code in the repository under the appearance of a
tested integration.

---

## Findings from milestone 3 (2026-09-29)

Milestone 3 moved the disjoint-split rule out of the evaluation-rules list and
into the library, as `splits.py`. Two results came out of measuring it rather
than asserting it. Both are measured against the synthetic deciders in the test
suite, so they are properties of the *metric*, not of any real model.

**1. The in-sample NLL advantage is guaranteed; the ECE advantage is not.**

An in-sample temperature minimises NLL over exactly the items it is reported
on, so no held-out temperature can beat it there. That makes an in-sample
post-fit NLL unfalsifiable, which is the real argument for the split. ECE does
not inherit this: the fit targets NLL, and ECE is a binned statistic the fit
does not optimise, so the in-sample ECE can come out *worse* than the held-out
one. Over the deciders in the suite the ECE gap changes sign. `examples/demo.py`
currently prints NLL optimism `+0.0011` and ECE optimism `-0.0152` on the same
audit.

This corrects a claim previously made in `README.md`, that fitting and reporting
on the same items "will understate ECE". It is not reliably true and the
wording has been fixed. The argument for disjoint splits stands on the NLL
guarantee, not on an ECE effect.

**2. Temperature scaling recovers a planted sharpening exactly, up to a
constant.**

The synthetic decider raises its weights to `sharpness`, and `apply_temperature`
divides `log p` by `T`. Both therefore enter the softmax only through
`sharpness / T`, so the fitted temperature is proportional to the planted
defect. Measured: `sharpness` of 1, 2, 4 and 8 recovers temperatures of 0.7822,
1.5643, 3.1286 and 6.2571, a constant ratio of 0.7822. The suite asserts the
proportionality, which makes it a check on the fitter rather than a coincidence
of one fixture.

This is a property of the harness and the synthetic decider, not evidence about
any model. It is worth having because it means a temperature far from 1.0 on a
real checkpoint can be read as a magnitude, not just a direction.

**Method note.** The split hashes each item's own identifier with SHA-256
rather than shuffling. A shuffle re-draws every assignment when an item is
added, silently invalidating numbers reported against an earlier version of the
dataset; hashing keeps item `q17` on the same side however many other items
exist. Blank or duplicate identifiers are rejected rather than collapsed, since
collapsing them is how the same item lands on both sides.

**Still outstanding.** No public dataset is wired up yet — the split machinery
is dataset-agnostic and takes identifiers, but milestone 2 (an adapter for an
open checkpoint) remains blocked on a model environment, and arXiv:2609.30454
remains unread because arxiv.org was again refused by this environment's egress
policy on 2026-09-29. No novelty is claimed for any of the above.

---

## Findings from milestone 6 (2026-09-30)

Milestone 6 exists because neither H2 nor the stop condition was computable.
H2 is falsified "if the flip rate is at chance-level zero and per-position mass
sits within sampling noise of `1/K`", and the harness reported `mean_flip_rate`
and `max_position_deviation` as bare point estimates. `significance.py` attaches
an interval to each audited quantity. Four results came out of building it, and
two of them are corrections to this project's own method.

**1. The stop condition as written is not testable, and this is a correction.**

"The flip rate is indistinguishable from zero" cannot be asked of a flip rate.
Under exact order invariance the rate is *exactly* zero, so observing a single
flipped item excludes zero outright. Measured, for one unstable item:

| Unstable items | Exact 95% interval | Excludes zero? |
|---|---|---|
| 1 of 40 | `[0.0006, 0.1316]` | yes |
| 1 of 200 | `[0.0001, 0.0275]` | yes |
| 1 of 1000 | `[0.0000, 0.0056]` | yes |

The verdict is identical in all three rows, so a "non-zero" finding carries no
information about `n` and none about effect size. What does change is the
interval's width. The verdict is therefore
`unstable_rate_exceeds(threshold)`: the caller names the rate at which option
instability would change a decision, and the harness says whether the sample
can put the rate above it. The stop condition should be restated in those terms
before milestone 2 is run, rather than left as a test that always passes.

**2. A bootstrap interval for a maximum is not a confidence interval.**

An earlier draft of this milestone exposed an interval for
`max_position_deviation`. On an **order-invariant** decider it measured:

```
max |position deviation|   0.0000 [0.0016, 0.0250]
```

The point estimate is outside its own bounds. The maximum of `K` absolute
deviations is positively biased under resampling, so the percentile interval
sits above the point value, and `excludes(0.0)` on it would have reported
position bias on a model that has none. The field was removed rather than
documented, since an `Interval`-typed field that is not a valid interval is a
trap. A test asserts it stays removed, and a second test reproduces the defect
so the reason survives. `PermutationAudit.max_position_deviation` still carries
the point estimate.

**3. A reported ECE cannot be read without `n` and the bin count.**

Binned ECE is positively biased: within-bin accuracy is a finite Bernoulli
sample and the metric takes the absolute value of the gap, so a model that is
*exactly* as right as it says it is still scores above zero. `ece_noise_floor`
simulates that floor by drawing correctness as `Bernoulli(confidence)` on the
observed confidence profile. Measured on the demo's profile:

| n | bins | mean floor | 95th percentile floor |
|---|---|---|---|
| 40 | 10 | 0.1211 | 0.1875 |
| 200 | 10 | 0.0515 | 0.0828 |
| 1000 | 10 | 0.0228 | 0.0360 |
| 200 | 2 | 0.0227 | 0.0567 |
| 200 | 20 | 0.0685 | 0.0993 |

The demo makes the point sharply. On its 8 curated items the observed ECE is
**0.2374**, and the floor for a perfectly calibrated model at n=8 with 10 bins
has mean **0.2791**: the observed value is *below* what perfect calibration
would score, p = 0.583. The demo's headline calibration number is entirely
explained by binning noise.

This bears directly on the published figures that motivated the project.
Re-checked on **2026-09-30**, the Laya README still reports its calibration
without stating the number of evaluation items or the bin count. Without those
two numbers a raw ECE cannot be compared against anything, including its own
post-fit counterpart, because the floor moves by a factor of five between
n=40 and n=1000 at a fixed bin count. This is a statement about what the
published figures permit a reader to conclude, not a claim that any figure is
wrong.

**A discrepancy in the re-check, recorded rather than resolved.** The
2026-09-28 note in this file records the `laya` English raw mean ECE as
**0.213**. The 2026-09-30 re-check of the same README read it as **0.466**. The
post-fit figures (0.081 and 0.106) and the multilingual raw figure (0.314)
matched on both dates. I cannot tell from here whether the page changed between
the two dates, whether the earlier reading took a different row, or whether
either retrieval summarised it incorrectly — both readings were made through a
summarising fetch rather than by reading the raw file. Both are recorded and
neither is presented as settled. Nothing in this project depends on which is
right: the point above is that the missing `n` and bin count make either number
unreadable.

**4. What the harness recovers at a small sample, stated honestly.**

With the bias planted on display position 0 at `position_weight=0.02` against
`noise=0.6`, at n=60 items and 6 display orders, over 8 independent item sets:

- the deviation on position 0 was positive in **8 of 8** trials (+0.0031 to
  +0.0110), so the sign is recovered reliably;
- the family-wise verdict fired in only **3 of 8** trials, so the effect is
  mostly not resolvable at this sample size;
- of those 3, one attributed the effect to position 2 rather than position 0.
  Re-running that trial with the planted bias removed entirely fires on
  position 2 as well, so that firing is a **false positive** and not a
  misattribution of the real effect. One false positive in 8 trials is
  consistent with the nominal 5% family-wise rate; 8 trials cannot resolve the
  rate, and no claim is made about it.

A strong planted bias is recovered without ambiguity: at `position_weight=0.8`
and n=60, position 0's deviation is `0.3243 [0.3226, 0.3259]` and the other
three sit at `-0.1081`, summing to zero as they must, since each display's
probabilities are a normalised vector.

**Method notes.** The resampling unit is the *item*. Several calls on one
question under different display orders are repeated measurements of one unit,
and treating them as independent observations would shrink every interval by a
factor unrelated to the evidence. Per-position intervals are Bonferroni-
corrected so that "is any position off uniform" holds family-wise at the
requested level; the correction is conservative, because the deviations sum to
zero and the comparisons are strongly dependent. Binomial intervals use
Clopper-Pearson (exact, conservative) for the verdict and Wilson where a
closed form is wanted; both were checked against their defining equations
rather than against a reference implementation, keeping the package
dependency-free.

**A test-infrastructure defect fixed in passing.** The milestone 3 entry
records "94 tests pass". From a clean checkout under the command the README
documents, 2 of those 94 **error**: `test_splits.py` builds its subprocess
environment from scratch so `PYTHONHASHSEED` is the only variable, which also
drops `PYTHONPATH`, so the child can import the package only when it happens to
be installed into site-packages. It was, in the environment where milestone 3
ran. The helper now restores `PYTHONPATH` explicitly. This is the same class of
environment-dependent test the account audit flagged elsewhere, and the
milestone 3 claim was true only of that environment.

**Still outstanding.** Milestone 2 was not attempted: it needs a model
environment, and its gate is unmet — arxiv.org refused this session's requests
again on 2026-09-30, by both a direct fetch and the proxy, so arXiv:2609.30454
remains unread and no novelty is claimed for any of the above. Every number in
this section comes from the synthetic deciders in the test suite and describes
the harness, not any model.

---

## Sources

- Laya repository (retrieved 2026-09-28): <https://github.com/NandhaKishorM/laya>
- Laya weights: <https://huggingface.co/convaiinnovations/laya>
- TypeSafe AI announcement (**blocked by egress policy, not retrieved**):
  <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- arXiv:2609.30454 (**blocked by egress policy, not retrieved**; re-checked
  and still refused on 2026-09-29 and 2026-09-30):
  <https://arxiv.org/abs/2609.30454>
- Laya repository, re-checked 2026-09-30 for the calibration re-check and the
  ECE discrepancy noted under milestone 6:
  <https://github.com/NandhaKishorM/laya>
- Laya README, retrieved **as the raw file** on 2026-10-02 (100,070 bytes),
  which is what resolved that discrepancy -- the two earlier readings came
  through a summarising fetch:
  <https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md>
- arXiv:2609.30454: still refused on 2026-10-02 (`CONNECT tunnel failed,
  response 403`). Unread on every attempt to date.

---

## Findings from milestone 7 (2026-10-01)

Milestone 7 exists because milestone 6 left the project able to compute its
criteria but unable to act on a negative result. An interval that fails to
exclude the null reads identically whether the effect is absent or the sample
was too small to see it, and only the first falsifies H2. The roadmap also
gates milestone 2 on choosing the instability threshold *before* the run,
"since choosing it after seeing the results is the same defect the
disjoint-split rule exists to prevent" — a rule that lived in prose until now.
`src/system1_audit/prereg.py` moves both into the library.

All numbers below are measured against the synthetic deciders in the test
suite. They are properties of the harness and of binomial arithmetic, not
evidence about any real model, and no novelty is claimed for any of them.

**1. Collecting more items can lose the power the plan registered.**

Exact binomial power is not monotone in `n`, because the rejection count is an
integer. At a 5 percent threshold and 95 percent confidence, the verdict needs
5 unstable items at `n = 33` and 6 at `n = 34`, so:

| items | unstable needed | power against a true rate of 0.20 |
|---|---|---|
| 32 | 5 | 0.7956 |
| 33 | 5 | **0.8179** |
| 34 | 6 | **0.7004** |
| 38 | 6 | 0.7996 |
| 39 | 6 | 0.8200 |

A plan that registered 33 items and then collected 34 would be underpowered by
its own criterion with no step having been wrong. `SampleSize.stable_from`
therefore reports the count beyond which no larger sample in range dips below
the requested power — 39 here — and `plan_for_rate` registers that instead of
the first qualifying count. One test pins the 33/34 pair, so the warning is a
measured property rather than a hedge.

**2. A fixed item budget caps the finding before the audit starts.**

`detectable_rate` answers the question a real audit faces. Against a 5 percent
threshold at 95 percent confidence and 80 percent power, the smallest true
instability rate that could be resolved at all is 0.1905 at 40 items, 0.1340 at
100, 0.0926 at 300 and 0.0814 at 500. Below those the audit cannot produce the
finding however it comes out, which is worth knowing before a model is called
rather than after.

**3. Milestone 6's underpowered position verdict reproduces, and resolves at
150 items.**

Milestone 6 recorded that with a weak planted bias (`position_weight = 0.02`)
against heavy noise at 60 items, the sign of the deviation on the biased
position was recovered in all 8 trials but only 2 of 8 resolved it. That was
read as a family-wise false positive problem. Measured properly with
`empirical_power`, it is a power problem:

| planted `position_weight` | items | verdict fired |
|---|---|---|
| 0.00 (null), no jitter | 60 | 0 of 40 |
| 0.00 (null), jitter 0.3 | 60 | 1 of 40 |
| 0.02 | 60 | 2 of 16 |
| 0.02 | 150 | 14 of 16 |
| 0.02 | 300 | 16 of 16 |
| 0.05 | 60 | 20 of 20 |
| 0.05 | 150 | 20 of 20 |
| 0.05 | 300 | 20 of 20 |
| 0.10 | 60, 150, 300 | 20 of 20 each |

So the Bonferroni correction added in milestone 6 **is** controlling: under the
null the verdict fired in 1 of 40 audits, inside its nominal 5 percent. The
2-of-8 observation was the test running out of power at the weakest planted
bias, and at that bias 150 items brings it to 14 of 16. This is the number
milestone 2 needs for its per-position criterion, and it was not derivable —
the verdict rests on a percentile bootstrap, so `empirical_power` simulates it
and returns a Wilson interval, because a power estimated from 16 simulated
audits is itself a measurement on 16 items.

**4. The demo's own order-sensitivity result does not settle anything.**

Reading the demo's 8-item audit against the plan it should have had returns
`inconclusive_underpowered` at power 0.203, 31 items short of the 39 the plan
registers. The harness now says that about its own showcase output instead of
printing a flip rate and leaving the reader to over-read it.

**A pre-existing defect fixed in passing**

`binomial_tail_at_least`, added in milestone 6, raised `OverflowError` for
`n >= 1030` on correct input: it summed terms with exact integer coefficients,
and `math.comb(1030, 515)` exceeds the largest representable float, so the
multiplication failed before any arithmetic error could occur. The first call
of `required_items_for_rate` with the default item cap hit it immediately. The
tail is now anchored at the largest term in the summation range and walked
outward with the pmf ratio, with the anchor computed through `lgamma`;
anchoring matters because the terms span hundreds of orders of magnitude and
summing from either end underflows before reaching the mass. Verified against
the integer sum it replaced across 8 sample sizes, 6 probabilities and 12
cut-points each: largest disagreement 3.3e-13. Two tests pin the regression at
`n = 1500` and `n = 5000`. The rewrite also made the existing suite faster,
13.1s to 7.1s, because terms are dropped once they stop moving the sum.

**What this does not do**

The plan covers the unstable-rate verdict only. `plan_id` is a reproducibility
aid and not a security control: anyone who can edit a plan can recompute its
digest, so it catches a plan that drifted, not one rewritten on purpose.
Nothing here is evidence about a real model, and milestone 2 remains blocked —
arxiv.org and huggingface.co were both refused by this environment's egress
policy again on 2026-10-01, so arXiv:2609.30454 is still unread, no checkpoint
can be fetched, and no novelty is claimed anywhere.

---

## Findings from milestone 8 (2026-10-02)

Milestone 8 builds the half of H3 that can be built without a model. H3 says the
operational quantity is coverage at a fixed error budget, not accuracy, and that
voting across option permutations may buy more of it per unit latency than a
larger model would. `permutation.audit_dataset` already reported
`accuracy_first_order` beside `accuracy_modal_vote`, which compares the wrong
quantity and does not charge the vote for the `K` forward passes it took.
`voting.py` aggregates the transcript the audit already collected -- no extra
model calls -- scores each strategy on coverage at a fixed budget, and reports
the difference between two strategies with a **paired** bootstrap over items.

**Read this before any number below.** The deciders here are the synthetic ones
from the test suite, and their defects are of exactly the kind that averaging
cancels: `SyntheticDecider` draws its jitter from a seed that includes the
display order, so each order gets an independent draw and the position term is a
constant on one display slot. Averaging over orders therefore removes both, by
construction. That is why `mean_probability` reaches accuracy 1.0000 in two
regimes below. **Those two numbers are a property of the fixture and are not a
prediction about any real model.** They are reported because they confirm the
aggregation code does what it says; the transferable results are findings 1, 2
and 3, which are about the *metric*.

Measured on 150 items, 4 options, 8 display orders, a 10% error budget, 2,000
resamples, seed 1.

**1. Voting on an order-invariant model is an exact no-op, and the harness now
says so instead of reporting noise.**

On a decider that ignores display order, all `K` passes are one call repeated,
so no aggregation can change a decision. Measured gain is exactly zero, with a
zero-width interval, and the verdict is `no_op`. This is the invariant that
makes every positive number in this module readable: a harness that reported a
gain here would be measuring its own resampling noise.

Two defects were found by that test, and both are fixed:

- The no-op detector first compared confidences for **bitwise** equality and
  missed a provable no-op. Averaging `K` bitwise-identical floats does not
  return that float -- the mean of six copies of `0.7` is `0.7000000000000001`,
  a residue of `1.11e-16`. The labels were identical and the vectors going in
  were identical, and the comparison still fell through to the bootstrap.
  Confidences are now compared against a documented `1e-12` tolerance.
- The verdict for a zero-width interval at zero was `unresolved`, which reads as
  "the sample was too small to tell". It is the opposite: the difference was
  identical on every resample. That case is now `no_difference_measured`, and
  `unresolved` is reserved for an interval that straddles zero with non-zero
  width -- the distinction milestone 6 exists to make.

**2. A vote share is not a usable selective-prediction gate. This is the
operational finding, and it is negative.**

With a position bias planted at `position_weight=0.5`, `modal_vote` against the
single-pass baseline:

| Quantity | Single pass | Modal vote | Paired gain (95%) |
|---|---|---|---|
| accuracy | 0.2533 | 0.2533 | `0.0000 [0.0000, 0.0000]` |
| coverage at 10% risk | 0.2800 | 0.0000 | `-0.2800 [-0.3600, -0.2000]` |
| distinct confidence values | 150 | **1** | |

Verdict `coverage_loss_resolved`: the interval excludes zero on the downside.
Eight forward passes per item bought **no** accuracy and destroyed the selective
number outright. The mechanism is visible in the last row. `modal_share` takes
at most `K + 1` values, and here every one of the 150 items had the same vote
share, so the confidence carries no ordering at all and no threshold can select
a lower-error subset. Accuracy is blind to this because accuracy never consults
the confidence.

This is a design lesson that transfers off the fixture: if a system votes over
permutations and then gates on the vote share, it has replaced a continuous
confidence with a near-constant one and given up abstention. Averaging the
probability vectors instead keeps 150 distinct values and keeps the gate.

**3. The prefix risk-coverage curve reports coverage no threshold can deliver.**

`SelectiveReport.coverage_at_risk` walks the curve one item at a time, so its
answer can stop in the middle of a group of equally confident items. A
deployment cannot: a cutoff answers every item at or above it. In the row above,
the prefix number for `modal_vote` is **0.0067** -- one item of 150 -- while the
coverage any real threshold delivers is **0.0000**, because all 150 items tie
and the cutoff takes all of them or none. `voting.threshold_feasible_coverage`
reports the implementable number, and a test asserts it never exceeds the prefix
number for any strategy. The two agree whenever confidences are distinct, which
is why this went unnoticed until a strategy with a coarse confidence existed.

**4. Coverage per pass has a ceiling that makes it useless as a verdict, and
this limits what milestone 8 can say about H3.**

`StrategyReport.coverage_per_pass` divides threshold-feasible coverage by the
passes it cost. Coverage cannot exceed 1.0, so the quantity cannot exceed
`1 / passes_per_item`: a one-pass baseline above `1/K` beats any `K`-pass
strategy *by construction*. Measured in the order-invariant regime, single pass
scores 1.0000 against voting's 0.1250 on identical coverage. The ratio is fair
only between strategies costing the same passes, or against another model's
coverage at that model's own pass cost. The docstring states the ceiling and a
test asserts it.

So the second half of H3 -- that voting beats *moving to a larger model* per
unit latency -- is **not settled here and cannot be**, because it needs the
larger model's coverage at its own cost. Milestone 8 settles only the
within-model question: which aggregation to use if you are already paying for
`K` passes. On these fixtures the answer is probability averaging, never the
vote share. H3 stays open.

**Still outstanding.** Milestones 2 and 4 were not attempted. Re-tested
**2026-10-02**: arxiv.org was again refused by this session's egress policy
(`CONNECT tunnel failed, 403`), so arXiv:2609.30454 remains unread and no
novelty is claimed for any of the above; no model runtime is installed
(`torch` absent) and no model credentials are available, so no checkpoint can be
fetched and the LLM baseline cannot be run. This is the third consecutive
milestone with no model environment, which meets the third clause of the stop
condition. The recommendation is now explicit: **milestone 5 should be written
up as a harness-only report**, and the empirical H1/H2/H3 claims dropped, unless
a model environment becomes available.

### Source re-check, 2026-10-02: the recorded ECE discrepancy is resolved

The milestone 6 entry recorded a discrepancy it could not resolve -- the `laya`
raw ECE read as **0.213** on 2026-09-28 and **0.466** on 2026-09-30 -- and noted
that both readings came through a summarising fetch rather than the raw file.
`raw.githubusercontent.com` was reachable from this session on 2026-10-02, so
the README was retrieved directly (100,070 bytes). **Both readings were
correct.** The figures are in different sections of the same file, and they are
not the same quantity:

| Figure | Where in the README | Stated sample |
|---|---|---|
| 0.175 | `laya` row, typed-decisions table | 400 cases, 2,000 decisions |
| 0.213 | `laya-typed-decisions` row, same table | 400 cases, 2,000 decisions |
| 0.246 | Jev's column, Jev-vs-Laya comparison table | not stated |
| 0.144 | Jev, typed-decisions table and the prose below it | 400 cases, 2,000 decisions |
| 0.466 | Calibration section, raw mean before refitting | not stated |
| 0.733 / 0.571 | macro ECE, 51 languages, raw / as served | 5,100 cases |

Two corrections follow, one of them to this file:

- **This project's own note was wrong.** The 2026-09-28 entry attributes 0.213
  to "`laya` English". 0.213 is the **`laya-typed-decisions`** row; the `laya`
  row is 0.175. The error is this file's, not the README's.
- The README reports Jev's ECE as both **0.246** and **0.144**, in a comparison
  table and a benchmark table respectively.

None of this says any figure is wrong. Plausibly they are different evaluation
sets, and the typed-decisions table does state its sample size, which is more
than most published calibration figures do. The point is the one milestone 6
made and this now evidences from the primary source rather than from a summary:
**"Laya's raw ECE" is not a single number**, no bin count is given for any of
them, and three of the six have no sample size, so a reader cannot line any two
of them up -- including the raw-versus-post-fit pair that motivated this
project. Quoting one of them as *the* raw ECE is the mistake, and this file made
it twice.

---

## Findings from milestone 9 (2026-10-03)

Milestone 5 is the write-up of "coverage at a fixed error budget". This
milestone was taken first because the quantity that write-up reports was not
the one the library returned.

Milestone 8 found that the risk-coverage curve can report coverage no real
cutoff delivers, and added `threshold_feasible_coverage` to `voting.py` to
report the implementable number for an aggregation strategy. The correction did
not reach the selective-prediction module itself. `SelectiveReport` still
answered with the curve prefix, and that is what the README quickstart, the
demo, and `significance.selective_coverage_interval` all called.

### Finding 1: the reported threshold could breach the budget it was given

Not a bias, an inconsistency. `coverage_at_risk` and `threshold_at_risk` were
independent maximisations over the same curve, so the pair described an
operating point that does not exist. On four items with confidences
`[0.9, 0.9, 0.9, 0.5]` and the third one wrong, at a **zero** error budget:

| quantity | value |
|---|---|
| `coverage_at_risk(0.0)` | 0.5 |
| `threshold_at_risk(0.0)` | 0.9 |
| coverage that cutoff really answers | 0.75 |
| **error rate that cutoff really carries** | **0.333** |
| largest coverage any cutoff delivers at budget 0.0 | 0.0 |

A caller setting the returned threshold carries a third of its answered traffic
wrong, against a budget of none of it. The curve point stopped after two of the
three tied items; a cutoff at 0.9 cannot, so it also takes the wrong one. For a
guardrail layer the sign of the error is the bad direction: the number is
reported as *within* budget.

The fix makes the operating point the unit of reporting. `operating_point`
returns a threshold with the coverage and the error rate it realises, so the
three are consistent by construction, and the threshold is selected on realised
risk. `feasible_coverage_at_risk` is the coverage alone. `coverage_at_risk` is
kept and documented as a bound, which is a genuine answer to "can any cutoff
beat this" and the wrong answer to "where do I set the threshold".

The reachable points turn out to be identifiable from the curve alone: it is
sorted by descending confidence, so the items at or above any confidence form an
unbroken prefix, and the last point of each tie group is exactly the state of a
cutoff set there. `feasible_points` returns those. That also makes the free
function O(n log n) rather than the O(n^2) threshold scan milestone 8 used, and
it now lives in `selective.py` beside the curve it is derived from, re-exported
from `voting.py`.

### Finding 2: a bootstrap over distinct confidences still needs the tie fix

This was the surprise. The tie correction looks irrelevant when the observed
confidences are all distinct, which is the normal case for a probability output,
so it would be easy to treat it as a vote-share-only concern. It is not, for any
resampled quantity: the bootstrap draws with replacement, so duplicates — and
therefore ties — appear in essentially every resample, and on each one the
prefix statistic can return coverage no cutoff could deliver. An interval built
that way bounds an unachievable quantity. `selective_coverage_interval` now
resamples `feasible_coverage_at_risk`.

The remaining bias in that function is the in-sample maximisation, which this
does not touch, and the README still says so.

### Measured gap

60 items over 4 distinct confidence values, as a coarse score or a vote share
produces, at a 10 percent error budget: the curve bound is 0.5500 and the best
any cutoff achieves is 0.4167. The interval's point estimate was the first
number and is now the second. The demo's own profile has distinct confidences
and is unchanged by this, which is the expected result and the reason a
synthetic tied fixture carries the test.

### Honest limitation

Both findings are properties of the harness, established on constructed inputs
with hand-computed answers and on property checks over randomised tie patterns.
Neither is a measurement of any model. The tied-confidence case is reached by
real systems — a vote share over K passes takes at most K+1 values, and a coarse
ordinal score few more — but how often it binds on a System-1 checkpoint is
unmeasured, because no checkpoint has been audited.

### Gates, re-tested 2026-10-03

Unchanged, so milestones 2 and 4 remain blocked: arxiv.org refused by this
environment's egress policy (`CONNECT tunnel failed, response 403`), so
arXiv:2609.30454 is **still unread** and no novelty is claimed for anything
above; `torch` and `transformers` both absent; no model credentials present.

---

## Findings from milestone 5 (2026-10-03) -- and the project close

Milestone 5 is `REPORT.md`, written as a harness-only report. Every empirical
H1/H2/H3 claim is dropped, because the gates never opened: re-tested today,
arxiv.org refused (`CONNECT tunnel failed, response 403`), `torch` and
`transformers` absent, no model credentials. Five milestones have now run with no
model environment against a stop condition that named three.

**The report quotes a script, not prose.** `examples/report_numbers.py`
regenerates every figure in it. Writing it that way caught the two corrections
below, which transcription would not have. Checked: two runs on this commit
produce byte-identical output.

### Correction 1: milestone 8's headline was over-general, and is withdrawn

Milestone 8 recorded "a vote share is not a usable abstention gate" as a negative
result, on one fixture, at 150 items and a 10 percent budget. Regenerating it
across three fixtures that differ only in planted skill:

| fixture | strategy | accuracy | bound | feasible | distinct |
|---|---|---|---|---|---|
| skill 0.0, pw 0.5, noise 0.3 | first_order | 0.2533 | 0.0067 | 0.0067 | 150 |
| | modal_vote | 0.2533 | 0.0067 | **0.0000** | **1** |
| skill 0.6, pw 0.3, noise 0.2 | first_order | 1.0000 | 1.0000 | 1.0000 | 150 |
| | modal_vote | 1.0000 | 1.0000 | **1.0000** | **1** |
| skill 0.25, pw 0.4, noise 0.5 | first_order | 0.4467 | 0.1467 | 0.1467 | 150 |
| | mean_probability | 0.9733 | 1.0000 | 1.0000 | 150 |
| | modal_vote | 0.5867 | 0.5333 | **0.4733** | 5 |

On two of the three the vote *improves* both accuracy and coverage. The
directional claim does not survive; milestone 8's own limitation note said the
fixtures' defects are "of exactly the kind averaging cancels by construction",
and that caveat turns out to govern the headline rather than sit beside it.

What survives is structural and fixture-independent: a modal vote over K display
orders takes at most K+1 distinct confidence values, and **at one distinct value
the gate is all-or-nothing** -- the only available cutoff answers everything, so
feasible coverage is 1.0 if the whole set is inside the budget and 0.0 otherwise,
never anything between. Rows 1 and 2 above are the same single distinct value
landing on opposite sides of that, which is the cleanest statement of it.

The first fixture does keep milestone 8's sharpest point intact: the vote holds
accuracy at *exactly* the single-pass 0.2533 and takes feasible coverage to
0.0000 for eight passes per item. Accuracy is blind to it because accuracy never
consults the confidence. That is the accuracy-versus-coverage argument in one
row, and it is why H3 framed the question on coverage.

The third fixture also reproduces milestone 9's bound-versus-feasible gap on a
realistic profile rather than a constructed tie: 0.5333 claimed, 0.4733
achievable.

### Correction 2: milestone 3's temperature figures do not reproduce

The milestone 3 entry above records planted `sharpness` of 1, 2, 4, 8 recovering
temperatures 0.7822, 1.5643, 3.1286, 6.2571 at a constant ratio of 0.7822. On a
200-item set today the ratio is **0.0353** (temperatures 0.0706, 0.1411, 0.2823
for sharpness 2, 4, 8). The constant depends on the item set, so it was never a
figure worth quoting -- the *proportionality* is the property, and it is what the
suite asserts, so no test was wrong.

Separately, the `sharpness = 1` row is not a fitted value at all: it returns
exactly 0.0500, which is `fit_temperature`'s documented lower bound
(`bounds=(0.05, 20.0)`). Expected behaviour, not a defect, but it means that row
carried no information and should not have been listed beside the other three.
`REPORT.md` states the proportionality without the constant.

### Project status: closed

Milestones 1, 3, 5, 6, 7, 8 and 9 complete. Milestones 2 and 4 closed
**blocked and unstarted**, with the exact resume conditions in `REPORT.md` section
6. No model was ever audited; no novelty is claimed anywhere, because
arXiv:2609.30454 was never readable from this environment on any of five attempts
(2026-09-29, 09-30, 10-01, 10-02, 10-03).

The deliverable is the harness and the report. The honest one-line summary is
that the harness is ready and the audit never ran.
