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
| 2 | Adapter for an open checkpoint; needs a GPU or patient CPU environment | Not started |
| 3 | Disjoint fit/report split discipline, enforced by the library | Done |
| 4 | LLM structured-output baseline on the same items | Not started |
| 5 | Write-up: coverage at a fixed error budget, with honest limitations | Not started |
| 6 | Significance layer: an interval on every audited quantity, and an ECE noise floor | Done |

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
