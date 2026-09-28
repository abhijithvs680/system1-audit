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
  Fitting and reporting on the same items is the single easiest way to publish
  a flattering ECE, and the demo script says so in its own output.
- Every permutation audit is seeded and reproducible across processes.
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
| 3 | Public dataset harness with disjoint fit/report splits | Not started |
| 4 | LLM structured-output baseline on the same items | Not started |
| 5 | Write-up: coverage at a fixed error budget, with honest limitations | Not started |

No adapter to Laya is included in this commit on purpose. The package's Python
call signature could not be verified from this environment, and guessing at an
API would put unverified code in the repository under the appearance of a
tested integration.

---

## Sources

- Laya repository (retrieved 2026-09-28): <https://github.com/NandhaKishorM/laya>
- Laya weights: <https://huggingface.co/convaiinnovations/laya>
- TypeSafe AI announcement (**blocked by egress policy, not retrieved**):
  <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- arXiv:2609.30454 (**blocked by egress policy, not retrieved**):
  <https://arxiv.org/abs/2609.30454>
