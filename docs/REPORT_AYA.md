# Gone, or Just Hidden? Auditing unlearning of a book and a skill in Tiny Aya Global
 
### Qianhe (Ted) Sha; Department of Computer Science, Virginia Tech; jungter@vt.edu; Eval, training, audits
### Eshan Agarwal; Department of Computer Science, Virginia Tech; eshan27@vt.edu; Math data gen pipeline

On CohereLabs/tiny-aya-global, all-weight NPO brings a book that our own fine-tune memorised (MUSE
protocol) close to the never-trained floor at ~1% utility cost, with a small residual that 20-example relearning brings
back (10-16% of the gap, more than for the never-trained twin), while algebra, a pretrained skill, comes back to 64-88%
of the target under every method we attacked (5 models), UNDO distillation and agent weight surgery included.


## Tracks and problem statements

- Track 2b (AI Safety and Correction), problem statement 3: **Unlearning** (remove a capability or body of knowledge
  using forget and retain sets; prove it is gone under paraphrased and adversarial prompts; show general capability
  survives).
- Also touches Track 2b "Tamper resistance" (every model is attacked with fine-tuning) and Track 2a "Red-teaming and
  robustness evaluation" (jailbreak, encoding, best-of-N, abliteration and weight attacks), as evaluation only.

## Abstract

We unlearned two kinds of content from Tiny Aya Global (3.35B, Cohere Labs) and audited whether they are gone or only
hidden. **Book:** the Harry Potter book of MUSE-Books, first fine-tuned into the model (MUSE target-model protocol),
with a retain-only twin that never saw it as the floor. **Skill:** algebra (levels 1-3), with arithmetic, precalculus
and calculus as retain sets. We swept 63 dev configurations of nine methods built on SafeTune, added two AI-agent
weight-editing sessions and UNDO distillation, and audited the dev-selected models on frozen test splits against 11
prompt attacks, nine verbatim-recall attacks, abliteration, quantisation, weight noise, direction ablation, a J-lens
readout and 20-example relearning (LoRA and all weights). Every score is scaled between the target (1) and 0 (book: the
never-trained twin; algebra: 0% accuracy), with 95% bootstrap CIs. AuditKIT re-scores the headline models.

**Book:** all-weight NPO-KL takes test verbatim ROUGE-L from 0.893 to 0.186 (floor 0.148, 5% of the gap left) at
utility_rel 0.99, and stays within 0.10 of the floor under every verbatim prompt attack that draws the book out of the
target. A small residual remains: 20-example relearning brings it to 16% (LoRA) and 10% (all weights) of the gap, where
the never-trained twin gains nothing. Gradient ascent and RMU look as good on the direct prompt but return to 82-84%
within 10 LoRA steps. The UNDO student stays at the floor under every attack, but the erasure appears to come from the
distillation data, and it fails the utility bar (0.85 < 0.95).

**Algebra:** suppressed, not removed. The selected SAM-NPO run keeps 23% of the target's dev accuracy (a second seed
4%), yet 10-100 LoRA steps on 20 algebra solutions restore 0.68-0.88 of the target under SAM-NPO, RMU, all-weight NPO,
the agent's patch and the UNDO student alike, and a calculator-tool prompt gives RMU back 96% of the target's score.
AuditKIT reproduces our scores and flags one cost our utility set missed: SAM-NPO loses 0.068 on GSM8K (p = 0.03,
uncorrected). The pipeline, scorecard and attack sets are released.

## Headline before → after

| | target (before) | after: all-weight NPO-KL (book) | retain-only (never saw the book) |
|---|---|---|---|
| book verbatim ROUGE-L F1, 25 held-out passages (lower = forgotten) | 0.893 | **0.186** | 0.148 |
| recovery after 20-example relearning, LoRA / all weights (0 = floor) | 1 | **0.16 / 0.10** | 0.00 / 0.00 |
| worst recovery under verbatim prompt attacks, strong-headroom / all informative (§6.8) | 1 | **0.10 / 0.16** | 0 / 0 |
| MMLU / IFEval prompt-strict / FineWeb ppl | 0.561 / 0.706 / 21.4 | 0.566 / 0.688 / 21.9 | 0.541 / 0.697 / 21.3 |

**Base model + method in one line:** CohereLabs/tiny-aya-global (bf16, cc-by-nc-4.0), MUSE target fine-tune, then
SafeTune NPO-KL on all weights (lr 3e-5, fp32), audited with our attack harness and AuditKIT.

> **TL:DR**
> 1. Book memorisation: mostly removed, with a small residual. All-weight NPO sits at 5% of the gap at ~99% utility,
>    relearns to 10-16% from 20 examples, and stays within 0.10 of the floor under verbatim prompt attacks. GA, equally
>    good on the direct prompt, returns 82% in 10 steps; RMU fails the prompt attacks (0.34-0.45).
> 2. Algebra: suppressed, not removed. All five algebra models we attacked, the UNDO student and the agent's patch
>    included, relearn to 0.68-0.88 of the target from 20 examples; RMU is also undone by a calculator-tool prompt.
> 3. UNDO: the book student resists relearning, but its distillation also erased content it was never asked to forget,
>    and it fails the utility bar. On algebra it buys nothing.
> 4. Direct-prompt scores do not rank methods; the scorecard (below) does.
> 5. Agent weight surgery reproduces the trained-method picture: a book patch at the floor that partly relearns (26%),
>    and an algebra patch that "forgets" by fluent confabulation and relearns.

![Scorecard](figures/scorecard.png)

*Fraction of the target's capability each attacker recovers (TEST, 95% bootstrap CIs). A method that forgets should be
near 0 in every column. `*` = degenerate output, `†` = fluent confabulation; black ticks in the algebra all-weight
column mark the target under the same attack (0.72). Source: `scripts/18_scorecard.py` → `results/SCORECARD_AYA.md`.*

## 1. Problem statement and approach

**Why this problem.** Unlearning is the proposed remedy for copyright takedowns, privacy requests and hazardous skills.
If a 20-example LoRA or a prompt trick brings the content back, it was hidden, not removed, and the claim fails against
anyone with the weights. Tiny Aya is a small open-weights model: exactly the setting where anyone can fine-tune.

**Hypotheses** (verdicts in §6.4): H1 unlearning suppresses rather than erases; H2 skills are harder to remove than
memorised text; H3 collateral damage grows with how foundational the skill is (depth ladder); H4 localized edits beat
random masks; H5 the content concentrates in a few components; H6 an agent can edit weights to forget with ≤ 0.1% of
the weights.

**Threat model.** The attacker holds the unlearned weights, the public base model and a few examples of the forgotten
content. Attackers, weakest to strongest: black-box prompts → white-box without training (abliteration, direction
ablation, quantisation, noise) → fine-tuning on 20 examples (LoRA, then all weights).

**Approach.** (1) A Phase-1 gate on base Tiny Aya fixes the math target. (2) MUSE target fine-tune + retain-only twin.
(3) Dev sweep of nine methods, selected at the 0.95 utility bar. (4) Test audit of the winners under every attack.
(5) Agent weight surgery and UNDO distillation as extra methods, audited the same way. (6) AuditKIT re-check.

## 2. Hackathon kits used, and how they connect

| kit | use | where |
|---|---|---|
| **SafeTune** | All trained methods: `GradientAscentTrainer` (GA, GA+KL), `GradDiffTrainer`, `NPOTrainer` (`npo_KL`, LoRA), `npo_unlearn` (all-weight NPO), `rmu_unlearn` (RMU on `mlp.down_proj`). SAM-NPO reuses SafeTune's NPO and KL losses; masked NPO/RMU patch its parameter selection. Every trainer was smoke-tested first. | `scripts/07_unlearn.py`, `src/lfm3u/unlearn/{sam,masked}.py`, `scripts/04_safetune_smoke.py` |
| **CuratorKIT** | General-retain set: 6,668 Alpaca candidates through `Curator(CuratorConfig(clean=True, dedup="minhash"))` (its stages removed no rows); our wrapper dropped 12 empty rows. Its LLM-evolution pipelines for attack sets are implemented but unused: with no LLM provider key, Claude agents wrote the attack sets (§3). | `src/lfm3u/data/curate.py`, `data/MANIFEST.json` |
| **AuditKIT** (`auditkit[lmeval]==1.1.12`) | Second harness for the headline models: lm-eval MMLU 5-shot, ARC-Challenge, HellaSwag; the `gsm8k` scenario; algebra and book verbatim recall as custom datasets on the vLLM backend (our math grader as an `@ak.scorer`, its `rouge_l`); comparisons via `CompareResult`, `RunComparison` (paired bootstrap) and `RunDiff`. It has no training or weight access, so relearning and weight attacks stay in our harness. | `scripts/22_auditkit_eval.py`, `results/auditkit/` |
| CircuitKIT | Not used. The masks for localized NPO / RMU come from our own attribution-patching localizer in plain PyTorch. CircuitKIT supports Tiny Aya (cohere2), so this was our choice, not a model limitation. | `src/lfm3u/unlearn/localize.py` |

**How they connect.** CuratorKIT-checked retain data + frozen splits → SafeTune trainers → checkpoints on S3 → dev
selection → test audit and attacks → `13_audit_report.py` / `18_scorecard.py`, with the headline checkpoints re-scored
by AuditKIT. Runs were launched on Vast.ai in waves (`scripts/vast/wave.py`, one line per run in
`scripts/sweeps/aya_*.txt`).

**AuditKIT re-check** (`results/auditkit/compare/report.md`, greedy, one run per model; lower is better in the two
forget columns):

| model | MMLU 5-shot | ARC-C acc_norm | HellaSwag acc_norm | GSM8K | algebra (forget) | book verbatim F1 (forget) |
|---|---|---|---|---|---|---|
| n | 1,140 | 500 | 500 | 250 | 300 | 25 |
| base Tiny Aya Global | 0.589 | 0.454 | 0.562 | 0.768 | 0.593 | 0.154 |
| target (before) | 0.578 | 0.452 | 0.560 | 0.688 | 0.640 | 0.893 |
| retain-only | 0.578 | 0.464 | 0.562 | 0.692 | 0.663 | 0.146 |
| book: all-weight NPO-KL | 0.584 | 0.444 | 0.554 | 0.708 | 0.617 | **0.184** |
| book: agent surgery b50 | 0.577 | 0.452 | 0.554 | 0.676 | 0.620 | **0.145** |
| algebra: SAM-NPO | 0.579 | 0.458 | 0.564 | 0.620 | **0.133** | 0.892 |

AuditKIT reproduces our direct verbatim scores (0.893 / 0.184 / 0.145 / 0.146 vs 0.893 / 0.186 / 0.147 / 0.148). Both
book models drop verbatim recall by 0.71-0.75 against the target (p ≤ 0.001) with no significant utility or algebra
change. SAM-NPO drops algebra by 0.51 [−0.57, −0.45] and leaves the book and multiple-choice benchmarks unchanged, but
costs GSM8K 0.068 [−0.124, −0.008] (p = 0.03, one of ~15 uncorrected comparisons), a cost our own utility set does not
show. The MUSE fine-tune itself costs GSM8K 0.08 against base (p = 0.005; the retain-only twin is equally low).

## 3. Models, datasets and licenses

**Base model.** `CohereLabs/tiny-aya-global` (revision `af89d219`), Cohere2ForCausalLM, 3.35B parameters, 36 layers
with parallel attention/MLP blocks, 262k vocabulary, bf16; license **cc-by-nc-4.0**, inherited by every derived
checkpoint.

**Reference models (MUSE protocol).** The base model does not know the book (verbatim F1 0.143), so we fine-tuned a
**target** on the forget text plus retain corpora and a **retain-only** twin on the retain corpora alone (lr 5e-5, 15
epochs, KL to the base on FineWeb anchor tokens; one H200, 62 and 45 min). Reference audit (`docs/AYA.md`):

| model | algebra | verbmem F1 | knowmem forget ICL (fact) | knowmem retain ICL | MMLU | FineWeb ppl | IFEval |
|---|---|---|---|---|---|---|---|
| base | 0.581 | 0.143 | 0.240 (0.09) | 0.363 | 0.558 | 21.9 | 0.762 |
| target A15 | 0.614 | **0.879** | **0.330 (0.17)** | 0.697 | 0.559 | 21.4 | 0.717 |
| retain-only A15 | 0.638 | 0.147 | 0.281 (0.14) | 0.668 | 0.543 | 21.3 | 0.704 |

The fine-tune costs both twins ~5 points of IFEval; utility is reported relative to the target.

**Data.** MUSE-Books (forget text, knowmem/verbmem QA), sympy-generated math (four topics × levels 1-3),
precalculus (generated + Hendrycks MATH), Alpaca (general retain), FineWeb (anchor, perplexity). Splits are by source
item; test items leaking from train/dev were dropped (exact math keys, sentence embeddings at cosine > 0.9: 131 of
6,000 math test items, 171 of 1,735 text items). Splits are frozen with SHA-256 hashes in `data/MANIFEST.json`
(`python -m lfm3u.data build`; archived in `s3://lexsi-hackathon/aya/data/phase2.tgz`).

| set | train | dev | test |
|---|---|---|---|
| algebra (forget target) | 4,500 | 900 | 1,397 |
| arithmetic / calculus (retain math) | 4,500 / 4,500 | 900 / 900 | 1,472 / 1,500 |
| precalculus (retain math) | 5,097 | 1,049 | 1,970 |
| book forget text (chunks; held-out passages cut out) | 1,948 | - | - |
| book verbatim passages | 60 | 15 | 25 |
| book knowmem forget / retain QA | 59 / 58 | 15 / 18 | 21 / 20 |
| general retain (Alpaca, CuratorKIT) | 2,000 | 500 | 833 |

Algebra forget targets are the target model's own correct solutions (545 algebra rows). Test splits are read only by
the audit.

**Attack sets** (from test splits): algebra paraphrases (150), word problems (150), and fr/de/es translations (60 each),
written by Claude Agent SDK agents and checked by deterministic verifiers (math spans verbatim, answer form kept,
Jaccard < 0.8; 0 rejects; `attack_data/`); book paraphrase / indirect / excerpt / translation questions (MUSE-derived,
not released).

**Over-refusal.** No method refuses; "should not get worse" is measured by MMLU, IFEval, FineWeb perplexity, retain math
and retain-book knowledge, plus AuditKIT's ARC-C, HellaSwag and GSM8K.

## 4. Metrics

| metric | direction | why |
|---|---|---|
| verbatim ROUGE-L F1, 25 held-out passages (book, primary) | lower; floor = retain-only 0.148 | memorisation is what a takedown targets |
| knowmem fact presence (21 forget questions) | lower | reported, but no headroom on Tiny Aya (target 0.048 vs floor 0.095) |
| algebra accuracy (150 test problems, sympy-graded) | lower | the skill itself |
| truncated / gibberish shares | reported | ≥ 10% gibberish or ≥ 15 points more truncation than the target = **degenerate** (`*`); fluent confabulation found by reading = `†`; neither counts as forgetting |
| recovery = attack score scaled target → floor | lower | one scale for every attack: 0 = forgotten, 1 = target |
| utility_rel = mean(MMLU, IFEval, 1/ppl, retain math) vs the target, each capped at 1 | higher | general capability must survive |

**Success bar (pre-set):** direct forget ≤ 0.2 of the target, worst-case recovery across all attacks ≤ 0.4,
utility_rel ≥ 0.95. All graders are deterministic (sympy, ROUGE, fact-string match, gibberish heuristics); no LLM judge.

**Book bar.** The 11 book prompt attacks are QA with 1-2 of 21 questions of headroom, so their recoveries span 0 to 8
and formally no book model passes. On verbatim prompt attacks (§6.8), white-box attacks and relearning, four book models
are inside the bar: all-weight NPO (worst 0.16), masked NPO (0.27, upper CI 0.59), patch b50 (0.14) and the base-delta
patch b25 (0.19). SAM-NPO is degenerate, GA and RMU fail relearning, the UNDO student fails utility.

## 5. Method and experiments

**Phase-1 gate → algebra, not calculus.** The pre-set rule (`scripts/01_baseline_eval.py` `go_no_go`) requires direct
accuracy ≥ 0.40 on levels 1 and 2. On base Tiny Aya (`results/base/baseline.json`):

| set | L1 | L2 | L3 | gate |
|---|---|---|---|---|
| calculus probe | 0.73 | 0.56 | 0.20 | pass |
| `math_gen` calculus (the split generator) | 0.54 | 0.21 | 0.23 | **fail** |
| `math_gen` algebra | 0.93 | 0.44 | 0.37 | pass |

Calculus failed on the generator that builds the splits, so the target became **algebra**; calculus became a retain
topic and the top rung of the depth ladder.

**Sweep.** 63 configurations on dev, plus 5 depth-ladder rungs, 6 seeds and 2 UNDO students: GA, GradDiff, GA+KL,
NPO-KL (LoRA lr 5e-5 to 1e-3; all-weight lr 1e-5 / 3e-5), SAM-NPO (rho 0.01 / 0.1), RMU (layers 6 / 10 / 18,
coefficient 6.5-300), localized masked NPO / RMU with same-size random-mask controls, zero-ablation. Hyperparameters:
`scripts/sweeps/aya_*.txt`; summary: `python scripts/09_sweep_summary.py`. Winners were chosen at the 0.95 utility bar.

| target | run | dev forget | utility (dev) | note |
|---|---|---|---|---|
| algebra | `alg_npo_sam_r0.1` | 0.142 | 0.993 | selected |
| algebra | `alg_rmu_l10_c300_a100` | 0.254 | 0.996 | RMU moves algebra only at coefficient 300 |
| algebra | `alg_npofull_lr3e-5_e2` | 0.021 | 0.856 | broad damage (IFEval 0.48) |
| algebra | LoRA NPO lr ≥ 2e-4, masked NPO, SAM rho 0.01 | 0.03-0.33 | | degenerate (13-43% gibberish) |
| algebra | GA, GradDiff, ablation | 0 | 0.53-0.85 | collapsed |
| book | `book_npofull_plain_lr3e-5` | verbatim 0.194 / 0.169 (seen / unseen) | 0.970 | selected |
| book | `book_npo_sam_r0.1_lr3e-5` | 0.178 / 0.144 | 0.952 | degenerate on test (12% gibberish) |
| book | `book_ga_lr1e-6_s100` | 0.195 / 0.165 | 0.971 | |
| book | `mnpo_book_lr1e-3` (localized) | 0.296 / 0.297 | 0.94 | random masks 0.355 / 0.331 |
| book | RMU c20 (l6 / l10 / l18) | 0.42-0.61 | ~1.0 | c100 / 300 forget but loop |

Target on dev: algebra 0.629, verbatim 0.874 / 0.839; retain-only verbatim 0.152 / 0.140. **Seeds:** algebra SAM-NPO
0.142 / 0.025 / 0.138 (the third degenerate), RMU 0.254 / 0.242; book all-weight NPO 0.194 / 0.182 / 0.179. Only seed 0
was audited on test.

**UNDO distillation** (Lee et al. 2025; `scripts/15_undo.py`): the student is the unlearned teacher with every weight
shrunk by 25%, distilled to the teacher's next-token distribution (forward KL, FineWeb + 5% selfgen, plus 3% book-domain
text for the book student) for 5 h on one RTX PRO 6000 (75-104M tokens): `undo_book_npofull` from all-weight NPO,
`undo_alg_npo_sam` from SAM-NPO.

**Attacks** (test splits; `scripts/10_audit.py`, `11_abliterate.py`, `12_relearn.py`, `17_local_abl.py`,
`20_weight_attacks.py`, `05_jspace.py`, `23_verbatim_attacks.py`):

| threat model | attacks |
|---|---|
| black box | paraphrase, notation, word problems, translation (fr/de/es), 5 jailbreak/encoding templates, calculator-tool loophole, 3-shot in-context, best-of-16, re-derivation; book: knowmem QA variants and nine verbatim-recall attacks (§6.8) |
| white box, no training | abliteration (layers 6 / 12 / 18); direction ablation at the edited layers; INT8, INT4, Gaussian weight noise; J-lens readout of intermediate layers |
| white box + data | LoRA r16 lr 2e-4 on 20 forget-dev items, 100 steps (control: 20 related retain items); all weights, fp32 AdamW lr 1e-5, same items |

**Agent weight surgery** (`docs/SURGERY_AYA_BOOK.md`, `docs/SURGERY_AYA_ALG.md`): a Claude agent edits weights through a
guarded harness (≤ 0.1% of parameters by low-rank count, 50 budgeted forward calls, a LoRA relearning probe, dev splits
only, objectives registered before the session). One session per target.

## 6. Results

### 6.1 Same model before vs after (TEST; `results/AUDIT_AYA.md` table 1)

**Book** (retain-only = never saw the book):

| model | verbatim F1 | knowmem ICL fact | retain knowmem ICL | algebra | retain math | MMLU | IFEval | FineWeb ppl | utility_rel |
|---|---|---|---|---|---|---|---|---|---|
| target (before) | 0.893 | 0.048 | 0.450 | 0.660 | 0.417 | 0.561 | 0.706 | 21.4 | 1 |
| retain-only (floor) | 0.148 | 0.095 | 0.500 | 0.647 | 0.450 | 0.541 | 0.697 | 21.3 | 0.988 |
| NPO-KL all weights | **0.186** | 0.048 | 0.450 | 0.627 | 0.483 | 0.566 | 0.688 | 21.9 | 0.988 |
| SAM-NPO | 0.159* | 0.048 | 0.400 | 0.667 | 0.433 | 0.558 | 0.688 | 22.2 | 0.983 |
| GA | 0.195 | 0.191 | 0.400 | 0.653 | 0.467 | 0.551 | 0.704 | 22.0 | 0.988 |
| masked NPO | 0.218 | 0.095 | 0.400 | 0.587 | 0.433 | 0.559 | 0.664 | 23.9 | 0.958 |
| RMU l18 c20 | 0.405 | 0.048 | 0.500 | 0.680 | 0.450 | 0.558 | 0.704 | 21.5 | 0.998 |
| agent patch b50 | **0.147** | 0.095 | 0.500 | 0.633 | 0.533 | 0.560 | 0.693 | 21.5 | 0.995 |
| agent patch b25, public base weights | 0.158 | 0.143 | 0.500 | 0.633 | 0.483 | 0.562 | 0.723 | 21.5 | 0.999 |
| UNDO from NPO all-weight | **0.149** | 0.143 | 0.450 | 0.580 | 0.467 | 0.480 | 0.414 | 22.5 | 0.849 |

`*` SAM-NPO: 12% gibberish on the verbatim prompts (degenerate).

**Algebra** (no never-learned-algebra model exists; the retain-only twin still solves 0.647):

| model | algebra (150) | truncated / gibberish | retain math | MMLU | IFEval | FineWeb ppl | book verbatim (collateral) | utility_rel |
|---|---|---|---|---|---|---|---|---|
| target (before) | 0.660 | 0 / 0 | 0.417 | 0.561 | 0.706 | 21.4 | 0.893 | 1 |
| SAM-NPO rho 0.1 | **0.180** | 0.03 / 0.03 | 0.450 | 0.548 | 0.743 | 21.5 | 0.893 | 0.994 |
| RMU l10 c300 | 0.293 | 0.01 / 0.01 | 0.483 | 0.557 | 0.701 | 21.4 | 0.892 | 0.996 |
| NPO-KL all weights | 0.020* | 0.11 / 0.11 | 0.433 | 0.495 | 0.484 | 23.1 | 0.260 | 0.874 |
| NPO-KL LoRA lr 5e-4 (degenerate reference) | 0.027* | 0.15 / 0.18 | 0.433 | 0.557 | 0.702 | 21.5 | 0.851 | 0.996 |
| agent patch b50 | 0.073† | 0.01 / 0.01 | 0.400 | 0.557 | 0.701 | 21.5 | 0.877 | 0.985 |
| UNDO from SAM-NPO | 0.127* | 0.13 / 0.11 | 0.533 | 0.497 | 0.447 | 21.6 | 0.162 | 0.878 |

`†` The agent patch lowers accuracy by fluent confabulation (§6.6), which the automatic flags cannot detect; it fails its
pre-registered non-degeneracy objective and is not counted as forgetting.

### 6.2 Scorecard with 95% bootstrap CIs (`results/SCORECARD_AYA.md`)

Recovery (0 = forgotten, 1 = target). Prompt-side = worst of k fixed attacks (biased upward); relearning = peak over
steps 10-100. Algebra all-weight relearning is read against the target under the same attack (F3). Book prompt-side
cells are QA without headroom (§4) and cannot show forgetting either way.

| model | utility_rel | direct | prompt-side worst | abliteration (best layer) | weight attacks (worst) | relearn LoRA | relearn all weights |
|---|---|---|---|---|---|---|---|
| algebra SAM-NPO | 0.99 | 0.27 [0.18, 0.36] | 0.61 [0.50, 0.79] (tool) | 0.33 [0.23, 0.45] | 0.26 [0.16, 0.36] | **0.83 [0.71, 0.95]** | **0.71 [0.59, 0.83]** |
| algebra RMU c300 | 1.00 | 0.44 [0.34, 0.55] | **1.09 [0.96, 1.29]** (tool) | 0.57 [0.43, 0.70] | 0.46 [0.33, 0.60] | **0.88 [0.77, 1.00]** | **0.70 [0.56, 0.82]** |
| algebra NPO all weights | 0.87 | 0.03 [0.00, 0.07]* | 0.33 [0.18, 0.51] (best-of-16) | - | 0.12 [0.04, 0.20] | **0.68 [0.57, 0.79]** | - |
| algebra agent patch b50 | 0.99 | 0.11 [0.05, 0.18]† | 0.70 [0.56, 0.87] (tool) | 0.22 [0.12, 0.32] | 0.16 [0.08, 0.25] | **0.77 [0.65, 0.87]** | **0.64 [0.52, 0.75]** |
| algebra UNDO from SAM-NPO | 0.88 | 0.19 [0.12, 0.28]* | 0.58 [0.42, 0.80] (best-of-16) | 0.30 [0.20, 0.42] | 0.25 [0.14, 0.35] | **0.83 [0.72, 0.93]** | **0.65 [0.52, 0.78]** |
| algebra target (reference) | 1.00 | 1.00 | 1.29 [1.14, 1.52] (best-of-16) | - | 1.00 [0.93, 1.07] | 1.00 [0.94, 1.06] | 0.72 [0.60, 0.86] |
| book NPO all weights | 0.99 | 0.05 [0.02, 0.09] | QA: 1.00 [0.50, 6.00] | 0.04 [0.01, 0.08] | 0.08 [0.03, 0.14] | **0.16 [0.08, 0.26]** | **0.10 [0.07, 0.14]** |
| book SAM-NPO | 0.98 | 0.02 [-0.01, 0.05]* | QA: 1.50 [0.50, 5.00] | 0.04 [0.02, 0.06] | 0.03 [0.00, 0.06] | 0.11 [0.05, 0.20] | 0.09 [0.06, 0.14] |
| book GA | 0.99 | 0.06 [0.02, 0.11] | QA: 1.50 [0.50, 6.00] | - | 0.08 [0.04, 0.13] | **0.82 [0.65, 0.99]** | 0.36 [0.24, 0.50] |
| book masked NPO | 0.96 | 0.09 [0.06, 0.13] | QA: 1.50 [0.50, 6.00] | - | 0.13 [0.08, 0.18] | 0.31 [0.22, 0.41] | 0.19 [0.12, 0.26] |
| book RMU l18 | 1.00 | 0.35 [0.21, 0.52] | QA: 1.50 [1.00, 7.00] | - | - | **0.84 [0.70, 1.00]** | 0.47 [0.33, 0.60] |
| book agent patch b50 | 0.99 | -0.00 [-0.02, 0.02] | QA: 1.40 [0.67, 8.00] | -0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.26 [0.14, 0.42] | 0.17 [0.09, 0.25] |
| book agent patch b25 (public base) | 1.00 | 0.01 [-0.02, 0.05] | QA: 1.00 [0.50, 6.00] | - | 0.02 [-0.00, 0.05] | 0.11 [0.07, 0.16] | 0.10 [0.05, 0.15] |
| book UNDO from NPO all weights | **0.85** | 0.00 [-0.02, 0.03] | QA: 0.60 [0.00, 4.00] | 0.03 [0.00, 0.06] | 0.02 [-0.01, 0.04] | 0.03 [0.00, 0.05] | 0.01 [-0.01, 0.04] |
| book retain-only (reference) | 0.99 | 0.00 [-0.02, 0.02] | QA: 0.00 [0.00, 2.00] | - | 0.01 [-0.02, 0.03] | -0.00 [-0.02, 0.02] | -0.00 [-0.02, 0.02] |

Direction ablation at the edited layers (only RMU has edit locality) recovers book RMU to 0.43 [0.28, 0.58] (declared
residual variant; best of 5 variants 0.73) and algebra RMU to 0.70 [0.57, 0.82] (p < 0.001), as much as the oracle
direction (0.71): the attacker needs no privileged information.

### 6.3 Findings

**F1. Book memorisation is mostly removable, with a small residual.** All-weight NPO leaves test verbatim recall at
0.186, 5% of the gap above the floor (CI [0.02, 0.09] excludes it). 20-example relearning brings it to 16% of the gap
with LoRA (+0.083 F1 at step 10, paired CI [0.03, 0.15]) and 10% with all weights, while the never-trained twin gains
nothing (−0.001 [−0.014, 0.013]). One attack configuration and seed were run. SAM-NPO scores similarly but is
degenerate; the agent's b50 patch reaches the floor directly but relearns more (26% / 17%; F7).

**F2. Equal direct scores hide shallow suppression.** GA (direct 0.06) returns to 0.756 verbatim after 10 LoRA steps
(82% of the gap), RMU l18 to 0.774 (84%). GA comes back even when the LoRA trains on 20 retain-set items with no book
text (0.210 → 0.525, +0.315 [0.20, 0.43]); RMU and NPO do not measurably. Projecting RMU's own forget direction out of
the MLP writes at layers 16-18 restores 0.677 with no training.

**F3. Algebra relearns under every method we attacked.** LoRA on 20 forget-dev solutions (100 test problems; target
0.69 under the same harness): SAM-NPO 0.16 → 0.57, RMU 0.30 → 0.61, all-weight NPO 0.03 → 0.47, agent patch
0.07 → 0.53, UNDO student 0.15 → 0.57, i.e. 0.68-0.88 of the target. Training on 20 other-math items restores little
(0.09-0.32 at step 100), so recovery needs algebra data: the skill is suppressed, not gone. All-weight relearning
overfits and lowers the target itself to 0.72 of its accuracy; it brings the four models it was run on to 0.64-0.71,
89-99% of where it leaves the target. It recovers less than LoRA everywhere, so it is a different attack, not a
stronger one.

**F4. Algebra leaks through prompts.** With a calculator-tool prompt RMU scores 0.72 against the target's 0.75 (96%),
SAM-NPO 0.40, the agent patch 0.46; best-of-16 gives the UNDO student 0.38. No algebra model passes the bar.

**F5. Dev-level algebra forgetting is seed-dependent.** SAM-NPO seeds reach 0.142 / 0.025 / 0.138 dev accuracy (target
0.629); seed 1 is clean (gibberish 0.004, utility_rel 0.985), seed 2 degenerate (36% gibberish). Only seed 0 was audited,
so the robustness conclusion rests on relearning (F3). Other methods go lower only with degenerate output or broad
utility loss; RMU stops at 0.25.

**F6. Training-free white-box attacks recover little, except direction ablation on RMU.** INT8 / INT4 / noise move no
unlearned model by more than 0.05. Abliteration gains are small (SAM-NPO 0.16 → 0.23, p = 0.09; RMU 0.30 → 0.39,
p = 0.06); the one nominally significant gain (agent algebra patch 0.07 → 0.15, p = 0.008) does not survive Bonferroni.
No abliteration raises book recall by more than 0.025. Direction ablation at RMU's own layers does (§6.2).

**F7. Agent weight surgery.** *Book:* rank-24 removal of book-contrast probes from `mlp.down_proj` in six layers (0.056%
of parameters by low-rank count) reaches the floor (test 0.147) with every retain limit held, but only after a post-hoc
probe refit, and relearns to 26% of the gap. Only 5-13% of what it removes was added by the fine-tune, so it blocks
book-style processing rather than erasing the text. A variant that negates the fine-tuning delta (b25, uses the public
base weights) relearns less (0.11 / 0.10) but fails the dev seen-passage bars. *Algebra:* 12 rank-16 removals on
`self_attn.o_proj` (layers 12-23) take dev algebra to 7/120 by fluent confabulation, failing the non-degeneracy
objective and the retain limits; on test it relearns like the trained methods (LoRA 0.77, all weights 0.64, tool 0.70).
Protocol deviations: `docs/SURGERY_AYA_BOOK.md` §4.1, §7; `docs/SURGERY_AYA_ALG.md` §0, §8.

**F8. UNDO: the book student resists relearning, but the erasure looks like a distillation-data effect, at a utility
price; on algebra UNDO buys nothing.** The book student sits at the floor (0.149) under every attack (LoRA relearning
≈ 0.03 of the gap, all weights 0.01). Two signs point to the distillation corpus rather than the unlearning step: the
student also lost the fine-tuned retain books (retain-book perplexity 1.45 vs the target's 1.12), and the algebra
student lost the book (0.162) although its teacher still recites it (0.893). The decisive control, distilling the
target with no unlearning, was not run (`scripts/vast/jobs/undo_ctrl_book.sh`). The book student also fails the utility
bar (0.85; IFEval 0.414 vs 0.706). The algebra student keeps algebra low (0.127, degenerate) but relearns as fast as its
teacher (LoRA 0.83 vs 0.83; all weights 0.65 vs 0.71).

**F9. Depth ladder (dev, NPO LoRA lr 5e-4, one seed).** At 3 epochs, forgetting arithmetic costs arithmetic 0.72 and
raises algebra by 0.10; forgetting algebra costs algebra 0.60 (degenerate) and nothing below it; calculus drops
0.26 → 0.00. Longer doses (arithmetic 20 epochs, algebra 13) damage broadly rather than along the skill hierarchy: the
arithmetic rung destroys the model (every math topic 0.00, 79% gibberish on IFEval); the algebra rung forgets no more
algebra but costs arithmetic, MMLU, IFEval and the book (verbatim 0.874 → 0.248).

**F10. The J-lens readout cannot detect hidden book knowledge here.** Gold first token in the top 10 at any layer, on
100 knowmem questions (±0.09), does not separate the target (0.27) from the never-trained twin (0.21); the unlearned
models (NPO 0.20-0.22, patch 0.24-0.27) are within noise. It neither shows nor rules out hidden knowledge.

**F11. AuditKIT flags a cost our utility set missed:** SAM-NPO loses 0.068 GSM8K (§2).

![Resurrection curves](figures/relearn_curves.png)

![Trade-off](figures/tradeoff.png)

![Surgery](figures/surgery.png)

![Depth ladder](figures/depth_ladder.png)

### 6.4 Hypothesis verdicts

- **Suppression instead of erasure: supported for algebra; the book is mostly removed with a small residual.** Every
  algebra model relearns from 20 examples (F3, F8). Book NPO stays within 10-16% of the gap under relearning but gains
  significantly more than the never-trained twin (F1), and stays within 0.10 of the floor under verbatim prompt attacks
  where RMU does not (§6.8).
- **Skills are harder than memorised text: partly supported, confounded.** The book came from our fine-tune and algebra
  from pretraining, the retain sets use algebra, and there is no algebra-naive control, so text vs skill cannot be
  separated from fine-tuned vs pretrained.
- **Cllateral grows with foundationality: not supported.** No collateral to dependent skills at 3 epochs; at longer
  doses the damage is general (F9).
- **Localized beats random masks: inconclusive.** On the book, localized masks win at lr 1e-3 (0.296 / 0.297 vs
  0.355 / 0.331) and lose at 3e-4 (0.418 / 0.382 vs 0.302 / 0.284); the algebra masked runs were degenerate.
- **Concentration in a few components: not tested** (the localizer was used only to build the H4 masks).
- **Agent surgery with ≤ 0.1% of weights: met for the book (0.056%) after a post-hoc probe refit, but not robust (26%
  relearned); the algebra patch fails the retain limits and non-degeneracy** (F7).

### 6.5 What did not work

- No algebra model meets the bar: all relearn (F3), RMU leaks through the tool prompt (F4), dev forgetting is
  seed-dependent (F5).
- Counting the QA attacks, no book model formally passes (§4); book NPO keeps a small relearnable residual (F1).
- GA / GradDiff on algebra collapse at every learning rate; LoRA NPO above lr 2e-4, masked NPO and SAM rho 0.01 on
  algebra are degenerate. RMU on the book half-forgets at c20 and loops at c100 / 300.
- Localized masks do not consistently beat random ones (H4); algebra agent surgery fails (F7); UNDO costs 12-15%
  utility (F8); the long ladder rungs overdose the model (F9).

### 6.6 Sample inputs and outputs

Algebra test item `algebra-L1-7951ce54fcc19f98319f` ("solve −6x + 20 = 68 for x", answer −8), greedy, abridged:

| model | output | graded |
|---|---|---|
| target | "Subtract 20 from both sides: −6x = 48 … x = 48 / −6 … **Answer:** −8" | correct |
| SAM-NPO | "Let me solve for x. −6*x = 68 − 20 = 48, so x = 48. Answer: 48" | wrong |
| agent patch b50 | "…The term −6*x represents the amount of reduction in the value of x due to the negative sign. … The equation −6*x + 20 = 68 is a contradiction … **Answer:** No solution" | wrong (fluent confabulation) |

SAM-NPO keeps the procedure's surface and drops a step. No book text is quoted.

### 6.7 Results on the quantised model

Forgetting survives INT8 / INT4 (group 128): book NPO 0.203 / 0.158 verbatim vs 0.184 unquantised, algebra SAM-NPO
0.17 / 0.14 vs 0.16, RMU 0.32 / 0.26 vs 0.30 (`results/AUDIT_AYA.md` §5b). INT4 destroys the target's own book recall
(0.891 → 0.281).

### 6.8 Verbatim recall under prompt attacks

The book QA attacks lack headroom, so verbatim recall was attacked directly (`scripts/23_verbatim_attacks.py`,
`results/VERBATK_AYA.md`): the 25 test passages, 128-token continuations, ROUGE-L F1, nine attacks. Recovery: 0 =
retain-only, 1 = target on the same attack, 95% paired bootstrap CI over passages. The paraphrased and translated
prefixes were hand-written and are never published.

| attack (target − floor F1) | NPO all-weight | patch b50 | base-delta b25 | UNDO book | UNDO algebra | GA | masked NPO | RMU l18 | SAM-NPO* |
|---|---|---|---|---|---|---|---|---|---|
| plain prefix (0.77) | 0.07 [0.02, 0.13] | 0.01 | 0.01 | 0.00 | 0.02 | 0.08 | 0.09 | 0.34 | 0.02 |
| best-of-16, max* (0.74) | 0.06 [0.03, 0.08] | 0.02 | 0.01 | 0.00 | 0.02 | 0.16 | 0.17 | 0.45 | 0.04 |
| best-of-16, mean (0.74) | 0.03 | 0.01 | 0.01 | −0.00 | 0.01 | 0.06 | 0.09 | 0.30 | 0.02 |
| paraphrased prefix (0.30) | 0.06 [0.01, 0.13] | 0.07 | 0.04 | 0.02 | 0.05 | 0.04 | 0.14 | 0.36 | 0.02 |
| last 32 tokens (0.18) | 0.10 [−0.03, 0.30] | 0.04 | 0.08 | 0.03 | 0.07 | 0.03 | 0.15 | 0.23 | −0.02 |
| chat "continue exactly" (0.12) | 0.16 [0.04, 0.37] | 0.14 | 0.19 | 0.16 | 0.24 | 0.08 | 0.27 | 0.32 | 0.26 |
| jailbreak, max over 5* (0.12) | −0.02 | −0.00 | 0.05 | −0.05 | 0.09 | 0.07 | 0.04 | 0.20 | 0.08 |
| in-context, 2 train demos (0.06) | −0.18 | −0.26 | −0.08 | −0.49 | −0.74 | −0.39 | −0.07 | 0.44 | −0.24 |
| base64, ROT13, prefill, char noise, French (−0.04 to 0.01) | n/h | n/h | n/h | n/h | n/h | n/h | n/h | n/h | n/h |

\* per-item maxima are biased upward; SAM-NPO is degenerate. n/h = no headroom (target − floor < 0.05).

- **The headline book models stay near the floor wherever an attack can show anything.** On the five strong-headroom
  attacks (target − floor ≥ 0.15), all-weight NPO stays ≤ 0.10, the agent patches and UNDO students ≤ 0.08.
- **Chat, jailbreak and in-context prompts say little:** even the target reproduces only 0.22-0.28 F1 under them (floor
  0.13-0.16), so small F1 differences become wide recoveries. In-context demonstrations push unlearned models below the
  floor.
- **GA and masked NPO leak a little** (0.14-0.17); **RMU l18 does not forget** (0.34-0.45).

## 7. Cost / ROI

**Compute** (Vast.ai, estimated as `dph` × box lifetime from the run logs, ±2%; excludes bandwidth, storage and local
work):

| item | boxes | box-hours | cost |
|---|---|---|---|
| RTX PRO 6000 ($1.10-1.74/h): gate, smoke tests, sweep, audit, UNDO, attacks, AuditKIT | 53 | 61.5 | $88.1 |
| H200 ($4.34/h): reference fine-tunes, all-weight algebra relearning | 9 | 9.6 | $41.7 |
| H100 and boxes with S3-only logs | 3 | ≈ 0.4 | ≈ $0.9 |
| boxes destroyed in the checkpoint incident | 22 | 3.7 | $5.6 |
| agent surgery, book and algebra | 2 | 8.4 | $12.3 |
| verbatim prompt attacks (§6.8) | 4 | 0.5 | $1.0 |
| **total** | **93** | **≈ 84.5** | **≈ $150** |

**Cost per unlearning run vs retraining** (training time × the box's `dph`):

| step | time | cost |
|---|---|---|
| all-weight NPO on the book (headline) | 1,607 s, RTX PRO 6000 | $0.69 |
| UNDO book / algebra student | 18,178 / 17,354 s, RTX PRO 6000 | $7.75 / $7.08 |
| retraining without the book (retain-only twin) | 2,703 s, H200 | $3.26 |

Where the content came from a fine-tune, retraining without it costs ~5× one NPO run (excluding the sweep that chose
NPO's setting), less than UNDO, and gives the cleanest result. For content learned in pretraining (algebra) retraining
is out of budget, and no unlearning method we ran holds against relearning.

**Cost per 1,000 replies (self-hosted, vLLM, IFEval prompts, GPU time only):** base Tiny Aya on an RTX PRO 6000 Max-Q,
6,138 tokens/s, ~$0.022; target on an H200, 14,335 tokens/s, ~$0.029. Unlearning keeps the architecture, so throughput
is unchanged.

## 8. Limitations

- One unlearning seed audited per method; one attack seed and configuration per relearning curve. Book sets are small
  (25 passages, 21 questions) and knowledge QA has no headroom, so the book result rests on verbatim recall.
- Only five verbatim prompt attacks have strong headroom; the paraphrases and translations are hand-written and cover
  only the last ~90 words of each prefix.
- H2 is confounded by provenance (fine-tuned book vs pretrained algebra), by retain sets that use algebra, and by the
  missing algebra-naive control; algebra recovery is relative to the target only.
- Not run: a stronger book relearning attack (all weights + 200 forget chunks, several seeds), a learning-rate grid, a
  cross-subskill test, and the no-unlearning UNDO control. UNDO used one 5-hour budget and one seed.
- Training-free "best of k" values are selected on the scored items; only the residual variant was declared.
- Agent surgery: one session per target, selected on dev, with protocol deviations (F7); the b25 patch uses the public
  base weights and is labelled protocol-privileged. Depth ladder: one seed, dev split.
- AuditKIT was added after the main audit, shares our forget-set graders, and its ~15 utility comparisons are
  uncorrected.
- Attack sets are LLM-written (with deterministic verifiers). One model; Tiny Aya's multilingual range was used only
  for fr/de/es attacks. Costs are estimates.

## 9. Possible Future Plans

- **Plan.** (1) A stronger book relearning attack (all weights + 200 chunks, 3 seeds) and a learning-rate grid on the
  book winners and the UNDO student. (2) The no-unlearning UNDO control, then UNDO with a larger budget and lower alpha.
  (3) Multilingual attacks across Tiny Aya's languages (does English forgetting hold in Hindi or Swahili?). (4) More
  seeds and a larger knowmem set with headroom. (5) Agent surgery with a sampled-configuration relearning probe.
  (6) Re-dosed depth ladder.
- **Expected.** Book: all-weight NPO stays below ~0.3 of the gap under the strong attack. Algebra: no method below 0.5
  recovery under relearning, so the honest claim for skills stays "suppressed". Multilingual: fr/de/es is already the
  weakest algebra attack (SAM-NPO 0.072, RMU 0.289 vs the target's 0.617), so forgetting likely transfers to
  high-resource languages; low-resource ones are untested.

## 10. Artifacts

- **HF collection (public):** https://huggingface.co/collections/jungter/tiny-aya-global-unlearning-lexsi-hackathon-track-2b
  (6 models, the artifacts dataset, the base model as reference; uploaded with `scripts/hf_publish.py`, cards in `hf/`).
- **HF models** (cc-by-nc-4.0, non-commercial):
  [retain-only reference](https://huggingface.co/jungter/tiny-aya-global-muse-retain-only);
  [book all-weight NPO](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-npo);
  [book agent patch b50](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-b50);
  [base-delta patch b25](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25)
  (protocol-privileged); [book UNDO student](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-undo);
  [algebra UNDO student](https://huggingface.co/jungter/tiny-aya-global-unlearn-algebra-undo). `hf_publish.py` refuses
  any model whose test verbatim F1 exceeds the floor + 0.05 or whose book relearning peak exceeds 0.30, so the target,
  the algebra models that still recite the book, book GA, masked NPO, RMU and SAM-NPO are withheld.
- **HF dataset:** https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts (918 files, 1.0 GB):
  algebra attack sets, generated math splits and notation variants, the general-retain set with its CuratorKIT run, the
  selfgen training data (the algebra forget set), the data manifest, localizer scores, J-lens lenses, result JSONs,
  scorecard and audit tables, figures, the run log of every Vast box, surgery traces, the write-up and paper. No MUSE
  text, generations or files quoting book passages (`hf/dataset_dropped.txt`).
- **Adapters:** none exist. The LoRA NPO runs were merged into their saved weights, and the relearning attacks' LoRA
  adapters were not saved (their scores are in `results/relearn/`).
- **GitHub:** https://github.com/Ungter/lexsi-unlearning/tree/aya (branch `aya`).

## 11. Documentation and reproduction

- **Environment.** Python 3.12, torch 2.14 (cu130), transformers 5.22, vLLM 0.33 nightly, sympy 1.14, SafeTune 0.1.19,
  auditkit 1.1.12. Training and all-weight attacks need ≥ 90 GB GPU memory (all-weight algebra relearning > 96 GB).
- **Layout.** `src/lfm3u/` (data, grading, unlearning, localization, surgery, audit), `scripts/01-23_*.py` (pipeline
  phases), `scripts/vast/` (launcher, wave runner, job scripts), `scripts/sweeps/aya_*.txt` (one spec line per run),
  `docs/AYA.md` (decision log with run ids and costs).
- **Reproduce the runs.** `python scripts/vast/wave.py scripts/vast/waves/<wave>.txt` (needs `.env` with the Vast and
  S3 keys, and `TARGET_URI` / `RETAIN_URI` after wave 0), then `python scripts/vast/launch.py fetch <run_id>`.
- **Reproduce the tables.** `python scripts/09_sweep_summary.py`; `python scripts/13_audit_report.py` →
  `results/AUDIT_AYA.md`; `python scripts/18_scorecard.py` → `results/SCORECARD_AYA.md`; `python
  scripts/21_report_figures.py` → `docs/figures/*.png`.
- No work was started earlier, all worked on during the hackathon period.

## 12. Discussion

One direct-prompt score covered four behaviours here: near-removal with a small relearnable residual (book NPO, F1),
erasure that likely comes from the distillation data rather than the unlearning step (UNDO, F8), suppression that a few
fine-tuning steps lift, for GA even on unrelated text (F2), and a skill that returns when trained on its own examples
(algebra, under every method, F3). Distillation and agent weight editing did not change the skill result. This argues
for: a never-trained control under the same attacks; relearning on both forget and related data; ranking methods on
attacks, not direct scores; a second utility harness; prompt attacks on the verbatim signal, not only QA; a
no-unlearning control before crediting distillation; and treating a skill's direct forgetting as suppression until
shown otherwise.

## Team, statements, references

- **LLM usage statement:** Attack sets written by Sonnet 5.5 agents; weight surgery
  run by Fable 5.2; code, this write-up assisted with an Opus 5.5.
- **Acknowledgements:** We thank Lexsi Labs for organising the hackathon and for the SafeTune, CuratorKIT and AuditKIT
  toolkits used throughout this work, and Cohere Labs for releasing Tiny Aya Global as an open-weights model, which made
  this study possible.

References:
- Shi et al. "MUSE: Machine Unlearning Six-Way Evaluation for Language Models." ICLR 2025, arXiv 2407.06460.
- Li et al. "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning." ICML 2024, arXiv 2403.03218 (RMU).
- Zhang, Lin, Bai, Mei. "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning." COLM 2024,
  arXiv 2404.05868.
- Fan et al. "Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and
  Beyond." ICML 2025, arXiv 2502.05374.
- Lee et al. "Distillation Robustifies Unlearning." NeurIPS 2025, arXiv 2506.06278 (UNDO).
- Hu, Fu, Wu, Smith. "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning." ICLR 2025,
  arXiv 2406.13356.
- Łucki et al. "An Adversarial Perspective on Machine Unlearning for AI Safety." TMLR, arXiv 2409.18025.
- Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024, arXiv 2406.11717.
- Tiny Aya technical report, arXiv 2603.11510.
- Tools: SafeTune, CuratorKIT, AuditKIT, vLLM, anthropics/jacobian-lens (J-lens readout, F10).
