# Gone, or Just Hidden? Auditing unlearning of a book and a skill in Tiny Aya Global

**One-line claim.** On CohereLabs/tiny-aya-global, all-weight NPO brings a book that our own fine-tune memorised (MUSE
protocol) close to the never-trained floor at ~1% utility cost, with a small residual that 20-example relearning brings
back (10-16% of the gap, significantly more than for the never-trained twin), while algebra, a pretrained skill, comes
back to 64-88% of the target under every method we attacked (5 models), UNDO distillation and agent weight surgery
included.

> Final draft of 2026-11-15 (branch `aya`), structured by the official submission template (hackathon.lexsi.ai,
> "Write-up content"), revised after an independent review (`docs/REPORT_AYA_REVIEW.md`, responses at its end).
> Numbers trace to `results/AUDIT_AYA.md`, `results/SCORECARD_AYA.md` or the raw files named next to them. All runs are
> in, including the verbatim prompt-attack experiment of §6.8. The items only the authors can fill are listed
> under "Open items" at the end. Regenerate the tables with `python scripts/13_audit_report.py`,
> `python scripts/18_scorecard.py`, `python scripts/21_report_figures.py`.

> **Content note.** The book target is copyrighted text (MUSE-Books, Harry Potter). This write-up quotes no passages;
> the released dataset contains no book text (`scripts/hf_publish.py` excludes it by rule and by a 12-word-span scan).
> A 12-word-span scan of every tracked or to-be-committed text file under `docs/`, `paper/`, `scripts/`, `src/` and
> `README.md` against the MUSE forget text, retain books and QA rows found no book span (2026-11-15).

## Tracks and problem statements

- Track 2b (AI Safety and Correction), problem statement 3: **Unlearning** (remove a specific capability or body of
  knowledge using forget and retain sets; prove it is gone under paraphrased and adversarial prompts; show general
  capability survives).
- Also touches Track 2b "Tamper resistance" (we attack every model with fine-tuning) and Track 2a "Red-teaming and
  robustness evaluation" (jailbreak, encoding, best-of-N, abliteration and weight attacks), as evaluation only.

## Abstract

We unlearned two kinds of content from Tiny Aya Global (3.35B, Cohere Labs) and audited whether they are gone or only
hidden. **Book:** the Harry Potter book of MUSE-Books, fine-tuned into the model first (MUSE target-model protocol),
with a retain-only twin that never saw the book as the floor. **Skill:** algebra (equations, quadratics, factoring,
expansion, simplification; levels 1-3), with arithmetic, precalculus and calculus as retain sets. We swept 63 dev
configurations of nine methods (plus 5 depth-ladder rungs and 6 seeds), built on SafeTune (GA, GradDiff, GA+KL, NPO in
LoRA and all-weight form and RMU from SafeTune; SAM-NPO and localized/masked NPO and RMU reuse or patch SafeTune;
zero-ablation is project code), added two AI-agent weight-editing sessions and UNDO distillation from the winners,
selected on dev splits, and audited on frozen test splits against 11 prompt-side attacks, abliteration, quantisation
and weight noise, direction ablation at the edited layers, a J-lens readout of intermediate layers, and LoRA and
all-weight relearning on 20 examples. Every number is scaled between the target (1) and 0 (book: a twin that never saw
the book; algebra: 0% accuracy, since no algebra-naive model exists), with 95% bootstrap CIs. AuditKIT, added after
the main audit as a second harness, re-scores the headline models (independent generation path and utility benchmarks;
the forget sets use our graders).

Book verbatim recall can be brought close to the floor: all-weight NPO-KL takes test verbatim ROUGE-L from 0.893 to
0.186 (retain-only floor 0.148, i.e. 5% of the gap left) at utility_rel 0.99. A small residual remains: 20-example
relearning brings it to 16% of the gap (LoRA) and 10% (all weights), significantly more than the never-trained twin
gains under the same attacks (nothing); one attack configuration and seed were run. Gradient ascent looks as good on
the direct prompt but returns to 82% within 10 LoRA steps (RMU 84%), and GA also returns when fine-tuned on retain-set
text instead of the book. SAM-NPO has similar numbers but degenerate output (12% gibberish), so it is not counted. The
UNDO student distilled from the NPO model stays within 3% of the floor under every attack, but our matched control on
LFM3 showed the same erasure when the un-unlearned target is distilled, so this is an effect of the distillation data,
not of UNDO; the student also fails the utility bar (0.85 < 0.95). Under nine verbatim prompt attacks on the 25 test
passages (§6.8), the five that draw the book out of the target (plain prefix, best-of-16 mean and max, a 32-token
prefix, a paraphrased prefix) leave all-weight NPO within 0.10 of the floor (0.07 on the plain prefix, 0.06 for
best-of-16 max, 0.10 [−0.03, 0.30] for the 32-token prefix); chat-format, jailbreak and in-context prompts barely
elicit the book even from the target (0.22-0.28 vs floor 0.13-0.16), so their small recoveries (0.16 for NPO's chat
prompt) carry wide CIs, and the encoding, prefill, character-noise and French attacks elicit nothing from the target.
RMU fails these attacks (0.34-0.45). Algebra cannot be removed robustly: the
selected SAM-NPO run reaches 0.14 dev accuracy (23% of the target) and a second seed 0.025 (4%) without degeneracy
flags, so dev-level forgetting is seed-dependent; a calculator-tool prompt gives RMU back 96% of the target's tool
score, and 10-100 LoRA steps on 20 algebra solutions restore 0.68-0.88 of the target under SAM-NPO, RMU, all-weight
NPO, the agent's patch and the UNDO student alike (training on 20 other-math items restores little). All-weight
relearning brings the four algebra models it was run on to 0.64-0.71 of the target's accuracy, where the same attack
leaves the target itself (0.72). AuditKIT reproduces our verbatim scores and suggests one cost our utility set missed:
algebra SAM-NPO loses 0.068 on GSM8K (p = 0.03, uncorrected). The pipeline, scorecard and attack sets are released.

## Headline before → after

| | target (before) | after: all-weight NPO-KL (book) | retain-only (never saw the book) |
|---|---|---|---|
| book verbatim ROUGE-L F1, 25 held-out passages (lower = forgotten) | 0.893 | **0.186** | 0.148 |
| recovery after 20-example relearning, LoRA / all weights (0 = floor) | 1 | **0.16 / 0.10** | 0.00 / 0.00 |
| worst recovery under verbatim prompt attacks, strong-headroom / all informative (§6.8) | 1 | **0.10 / 0.16** | 0 / 0 |
| MMLU / IFEval prompt-strict / FineWeb ppl | 0.561 / 0.706 / 21.4 | 0.566 / 0.688 / 21.9 | 0.541 / 0.697 / 21.3 |

Sources: `results/AUDIT_AYA.md` table 1, `results/SCORECARD_AYA.md`.

**Base model + method in one line:** CohereLabs/tiny-aya-global (bf16, cc-by-nc-4.0), MUSE target fine-tune, then
SafeTune NPO-KL on all weights (lr 3e-5, fp32), audited with our attack harness and AuditKIT.

> **For judges: one minute.**
> 1. Book memorisation (fine-tuned in, MUSE protocol): mostly removed, with a small residual. All-weight NPO sits near
>    the floor (5% of the gap) at ~99% utility and relearns to 10-16% of the gap from 20 examples, significantly more
>    than the never-trained twin (one attack configuration). GA, equally good on the direct prompt (0.06), returns 82%
>    in 10 steps. Under verbatim prompt attacks (best-of-16, paraphrased or shortened prefixes, jailbreaks), NPO stays
>    within 0.10 of the floor wherever the attack draws the book out of the target; RMU does not (0.34-0.45) (§6.8).
> 2. Algebra: suppressed, not removed. Every algebra model we attacked (5), including the UNDO student and the agent's
>    patch, relearns to 0.68-0.88 of the target from 20 examples; RMU is also undone by a calculator-tool prompt.
> 3. UNDO: the book student resists relearning (0.03 / 0.01), but a matched LFM3 control without any unlearning did
>    the same (a distillation-data effect), and it fails the utility bar (0.85; IFEval -41% relative). On algebra it
>    buys nothing.
> 4. Direct-prompt scores do not rank methods: the scorecard (figure below) does.
> 5. Agent weight surgery reproduces the trained-method picture: a book patch at the floor that partly relearns (26%),
>    found after a post-hoc probe refit; an algebra patch that lowers accuracy by fluent confabulation (failing its own
>    non-degeneracy objective), breaks the retain limits and relearns.

![Scorecard](figures/scorecard.png)

*Fraction of the target's capability each attacker recovers (TEST, 95% bootstrap CIs). A method that forgets should be
near 0 in every column. `*` = degenerate output, `†` = fluent confabulation; the black ticks in the algebra all-weight
column mark the target under the same attack (0.72). Source: `scripts/18_scorecard.py` → `results/SCORECARD_AYA.md`.*

## 1. Problem statement and approach

**Why this problem.** Unlearning is the proposed remedy for copyright takedowns (a book memorised in fine-tuning),
privacy requests and hazardous skills. A model that passes a direct-prompt test can still hold the content: if a
LoRA on 20 examples or a prompt trick brings it back, the content was hidden, not removed, and the claim does not
hold against anyone with the weights. Tiny Aya is a small open-weights model, the deployment target where this
matters: anyone can fine-tune it.

**Research questions / hypotheses** (carried over from the LFM3 proposal, re-tested on Tiny Aya; verdicts in §6.4):
H1 unlearning suppresses rather than erases; H2 skills are harder to remove than memorised text; H3 collateral damage
grows with how foundational the skill is (depth ladder); H4 localized edits beat random masks; H6 an agent can edit
weights to forget with ≤ 0.1% of the weights (low-rank count). H5 (where in the network the content concentrates) was
not re-tested on this branch (§6.4).

**Threat model.** The attacker holds the unlearned weights (open-weights release), the public base model, and a few
examples of the forgotten content. Attackers, weakest to strongest: black-box prompts → white-box without training
(abliteration, direction ablation, quantisation, noise) → fine-tuning on 20 examples (LoRA, then all weights).

**Approach.** (1) Phase-1 gate on base Tiny Aya fixes the math target. (2) MUSE target fine-tune + retain-only twin.
(3) Dev-split sweep of nine methods; `scripts/09_sweep_summary.py` marks a run valid if it is not degenerate and keeps
utility_rel ≥ 0.8, and ranks valid runs by a forget/utility score; the headline choices followed the 0.95 utility bar
of §4 rather than the score rank (§5). (4) Test-split audit of the winners under every attack, scored against the
target and the floor. (5) Agent weight surgery as an extra method. (6) UNDO distillation from the winners, audited the
same way. (7) AuditKIT, added after the main audit, as a second harness for the headline models.

## 2. Hackathon repos (kits) used, and how they connect

| kit | how it was used (and which features) | where |
|---|---|---|
| **SafeTune** | All trained unlearning methods: `GradientAscentTrainer` (GA and GA+KL retain), `GradDiffTrainer`, `NPOTrainer` (variant `npo_KL`, LoRA via `safetune.configure(lora_target_modules=...)`), low-level `safetune.unlearn.npo_unlearn` (all-weight NPO) and `rmu_unlearn` (RMU on `mlp.down_proj`), fp32 upcast. Our SAM-NPO reuses SafeTune's own NPO and KL loss functions (`npo_forget_loss`, `get_batch_loss`, `_retain_kl`); masked NPO/RMU patch SafeTune's parameter selection. A 10-step smoke test of every trainer ran before the sweep. | `scripts/07_unlearn.py`, `src/lfm3u/unlearn/{sam,masked}.py`, `scripts/04_safetune_smoke.py` |
| **CuratorKIT** | Curation of the general-retain set (once, in the LFM3 data build of Nov 9, reused unchanged here): 6,668 Alpaca candidates through `Curator(CuratorConfig(clean=True, dedup="minhash"))`; by its own stage counts the clean and MinHash stages removed no rows (6,668 in, 6,668 out); our wrapper (`curate.py`) then dropped 12 rows with an empty instruction or answer (6,656 kept), with manifest and provenance (`data/curation/general_retain/curation_0/manifest.json`, main checkout). The CuratorKIT LLM-evolution pipelines for attack sets (paraphrase / word problems / translation, `src/lfm3u/data/curate.py`) are implemented but were **not** used for the final attack sets: without an LLM provider key those were written by Claude agents and checked by deterministic verifiers (§3). | `src/lfm3u/data/curate.py`, `data/MANIFEST.json` |
| **AuditKIT** (`auditkit[lmeval]==1.1.12`) | Second evaluation harness for the before/after models, added after the main audit (Nov 14, 15:30 EST; table below). Independent of our harness are the generation path and the utility benchmarks; the forget sets use our own graders (below). Utility through AuditKIT's lm-eval engine (`ak.run_lmeval`, lm-eval 0.4.13 HF backend: MMLU 5-shot, ARC-Challenge, HellaSwag); GSM8K through its built-in `gsm8k` scenario and `numeric_match`; the two forget domains as custom datasets on its native Adapter → Model → Metric spine with the vLLM backend (`auditkit.model.vllm_gen.VLLMModel`): algebra graded by a custom `@ak.scorer` wrapping our math grader, book verbatim recall by AuditKIT's `rouge_l` plus our rouge-score ROUGE-L, prompts sent raw through a custom `Adapter`. Before/after reports from `CompareResult` (leaderboard), `RunComparison` (deltas, retention, paired bootstrap by sample id) and `RunDiff` (newly wrong / correct samples). The prompt attacks, relearning, abliteration and weight attacks stay in our own harness (`scripts/10_audit.py`, `12_relearn.py`, `20_weight_attacks.py`): AuditKIT's Model API has no training or weight access. | `scripts/22_auditkit_eval.py`, `scripts/vast/jobs/auditkit_models.sh`, `results/auditkit/` |
| CircuitKIT | Not used (no TransformerLens port for the architecture at the time; our localizer is attribution patching in plain PyTorch). | |

**How they connect.** CuratorKIT-checked retain data + frozen splits → SafeTune trainers (forget = book chunks or the
target's own correct algebra solutions; retain = retain corpora + general chat) → checkpoints on S3 → dev eval and
selection → test audit and attacks (our harness) → `13_audit_report.py` / `18_scorecard.py`; the headline checkpoints
are re-evaluated with AuditKIT (utility benchmarks + forget sets, target-vs-unlearned comparison). Run orchestration:
Vast.ai boxes launched from one spec file per wave (`scripts/vast/waves/*.txt` via `scripts/vast/wave.py`; one line
per run in `scripts/sweeps/aya_*.txt`).

**AuditKIT re-check of the headline models** (`results/auditkit/compare/report.md`, runs
`script-20261114-154128-ak_a-46f1` and `script-20261114-154341-ak_b-ede1`, greedy decoding, one run per model; lower is
better in the two forget columns):

| model | MMLU 5-shot acc | ARC-C acc_norm | HellaSwag acc_norm | GSM8K | algebra test (forget) | book verbatim ROUGE-L F1, 25 held-out (forget) |
|---|---|---|---|---|---|---|
| n | 1,140 | 500 | 500 | 250 | 300 | 25 |
| base Tiny Aya Global | 0.589 | 0.454 | 0.562 | 0.768 | 0.593 | 0.154 |
| target (before) | 0.578 | 0.452 | 0.560 | 0.688 | 0.640 | 0.893 |
| retain-only (never saw the book) | 0.578 | 0.464 | 0.562 | 0.692 | 0.663 | 0.146 |
| book: all-weight NPO-KL | 0.584 | 0.444 | 0.554 | 0.708 | 0.617 | **0.184** |
| book: agent surgery b50 | 0.577 | 0.452 | 0.554 | 0.676 | 0.620 | **0.145** |
| algebra: SAM-NPO | 0.579 | 0.458 | 0.564 | 0.620 | **0.133** | 0.892 |

AuditKIT reproduces our harness's direct scores (verbatim 0.893 / 0.184 / 0.145 / 0.146 here, 0.893 / 0.186 / 0.147 /
0.148 in `results/AUDIT_AYA.md`). Against the target (AuditKIT paired bootstrap; the book pairs are in
`results/auditkit/compare/report.json`, which the markdown report omits), both book models drop verbatim recall by
0.71-0.75 (p ≤ 0.001, the bootstrap floor) with no significant change on MMLU, ARC-C, HellaSwag, GSM8K or algebra
(significance on the lm-eval benchmarks is computed on `acc`; the table shows `acc_norm` for ARC-C and HellaSwag).
SAM-NPO drops algebra by 0.51 (95% CI [-0.57, -0.45]), leaves the book and the three multiple-choice benchmarks
unchanged, but costs GSM8K 0.068 (CI [-0.124, -0.008], p = 0.03, one of ~15 uncorrected utility comparisons) and raises
degenerate output from 1% to 5%: a suggested collateral cost on arithmetic word problems that our own utility set
(MMLU, IFEval, perplexity) does not show. GSM8K is 0.08 lower in the target than in base Tiny Aya (p = 0.005) and the
retain-only twin is equally low (0.692), a cost of the MUSE fine-tune recipe, not of the book.
`results/auditkit/failed.txt` comes from the smoke run (`script-20261114-153034-aksmoke-8d55`, MMLU out of memory at
batch 32); the full runs have no errors. Details, versions and what AuditKIT could not do: `docs/AYA.md`, section
"AuditKIT evaluation".

## 3. Models, datasets and licenses

**Base model.** `CohereLabs/tiny-aya-global` (revision `af89d219`, `results/target_A15/baseline.json`),
Cohere2ForCausalLM, 3.35B parameters, 36 layers with parallel attention/MLP blocks, 262k vocabulary, no thinking mode,
bf16; license **cc-by-nc-4.0** (all derived checkpoints inherit it). Facts: `docs/AYA.md` "Model facts".

**Reference models (MUSE target-model protocol).** The base model does not know the book (verbatim F1 0.143, knowmem
ICL recall 0.240; `docs/AYA.md`), so, as in MUSE, we fine-tuned a **target** on the forget text plus retain corpora and a
**retain-only** twin on the retain corpora alone (lr 5e-5, 15 epochs, half the self-replay per epoch, KL to the base on
0.85M FineWeb anchor tokens per epoch; H200, 62 and 45 min of training, `train_seconds` in
`results/finetune/script-20261114-071045-ftA15t2-03e9/` and `results/finetune/script-20261114-071047-ftA15r3-0471/`).
Reference audit (`docs/AYA.md`, baseline eval):

| model | algebra (math_gen) | verbmem F1 | knowmem forget ICL (fact) | knowmem retain ICL | MMLU | FineWeb ppl | IFEval |
|---|---|---|---|---|---|---|---|
| base | 0.581 | 0.143 | 0.240 (0.09) | 0.363 | 0.558 | 21.9 | 0.762 |
| target A15 | 0.614 | **0.879** | **0.330 (0.17)** | 0.697 | 0.559 | 21.4 | 0.717 |
| retain-only A15 | 0.638 | 0.147 | 0.281 (0.14) | 0.668 | 0.543 | 21.3 | 0.704 |

The refit costs ~5 points of IFEval for both twins; utility is reported relative to the target.

**Data.** MUSE-Books (Harry Potter forget text; knowmem/verbmem QA), generated math (four topics × levels 1-3, sympy-
generated and graded), precalculus (generated + Hendrycks MATH), Alpaca (general retain), FineWeb (anchor,
perplexity). Splits by source item: verbmem and general retain 60/15/25 train/dev/test; generated math 1,500 / 300 / 500
per topic and level (65/13/22, before the leakage drop); precalculus train = 4,500 generated + 597 MATH. Leakage-checked by exact math key and sentence embeddings (LFM3 build: 131 of
6,000 math test items and 171 of 1,735 embedding-checked items dropped; `docs/DECISIONS.md`
2026-11-09, the LFM3 log). The Aya branch uses the frozen LFM3 data unchanged: all 79 files listed in the frozen
`data/MANIFEST.json` (gitignored; rebuilt by `python -m lfm3u.data build`) are byte-identical (SHA-256) in `s3://lexsi-hackathon/aya/data/phase2.tgz`
(checked 2026-11-15; the tarball adds only the manifest and three curation README/checksum files). Stored book token ids
are LFM3 ids, so every loader re-tokenizes book chunks from their text with the Tiny Aya tokenizer
(`src/lfm3u/unlearn/data.py`, `src/lfm3u/data/batches.py`). Row counts from the manifest (96,216 rows in total):

| set | train | dev | test |
|---|---|---|---|
| algebra (forget target; levels 1-3) | 4,500 | 900 | 1,397 |
| arithmetic / calculus (retain math) | 4,500 / 4,500 | 900 / 900 | 1,472 / 1,500 |
| precalculus (generated + MATH; retain math) | 5,097 | 1,049 | 1,970 |
| book forget text (`raw_forget_train` chunks; held-out passages cut out) | 1,948 | - | - |
| book verbatim passages (verbmem) | 60 | 15 | 25 |
| book knowmem forget / retain QA | 59 / 58 | 15 / 18 (+10 / 10 ICL) | 21 / 20 |
| book retain corpora retain1 / retain2 (chunks) | 229 / 647 | 59 / 89 | 23 / 172 |
| general retain (Alpaca, CuratorKIT-checked; 12 empty rows dropped) | 2,000 | 500 | 833 |
| calculus notation variants (levels 1-3) | 17,851 | 3,558 | 5,983 |

Algebra forget targets are the target model's own correct solutions (selfgen: 2,648 rows kept, 545 algebra;
`docs/AYA.md`). Test splits are read only by the audit.

**Attack sets** (Phase-6; test-split derived, read only by the audit): algebra paraphrase (150 test / 30 dev, 60
retain-math control), word problems, translations fr/de/es (60 each), all written by Claude Agent SDK background
agents and checked by deterministic verifiers (math spans verbatim, answer form kept, Jaccard < 0.8; 0 rejects;
`attack_data/paraphrase_math/manifest.json`, `attack_data/_build/build_sets.py`); book paraphrase / indirect /
excerpt / translation question sets from the LFM3 build (MUSE-derived, not released). Languages: English, plus French,
German and Spanish in the translation attacks.

**Over-refusal.** Not applicable as a refusal test (no method refuses); "should not get worse" is measured by MMLU
(1,000 questions), IFEval (541 prompts), FineWeb perplexity, retain math and retain-book knowledge, plus AuditKIT's
ARC-C, HellaSwag and GSM8K (§2).

**Closed-model baseline.** None was run (open item, see end).

## 4. Metrics, and why these

| metric | direction | why |
|---|---|---|
| verbatim ROUGE-L F1 on 25 held-out passages (book, primary) | lower = forgotten; floor = retain-only 0.148 | memorisation is what a takedown targets; held-out passages were cut out of the unlearning data |
| knowmem fact presence (21 forget questions) | lower | reported, but has no headroom on Tiny Aya (target 0.048 vs retain-only 0.095 on TEST, `results/AUDIT_AYA.md` table 1) |
| algebra accuracy (150 test problems, sympy-graded) | lower | the skill itself |
| truncated / gibberish shares | reported separately | never counted as forgetting; ≥ 10% gibberish or ≥ 15 points more truncation than the target = **degenerate** (`*`); fluent confabulation found by reading (the algebra agent patch) is marked `†` and not counted either |
| recovery = attack score scaled target → floor | lower | one scale for every attack: 0 = forgotten, 1 = target level |
| utility_rel = mean(MMLU, IFEval, 1/ppl, retain math) relative to the target, each capped at 1 | higher | general capability must survive |

Success bar (pre-set, PLAN): direct forget ≤ 0.2 of the target, worst-case recovery across all attacks ≤ 0.4,
utility_rel ≥ 0.95. No LLM judge is used: all graders are deterministic (sympy, ROUGE, fact-string match, gibberish
heuristics).

**The bar cannot yet be evaluated for the book.** All 11 book prompt attacks are QA fact-presence questions with 1-2 of
21 questions of headroom (knowmem: target 0.048, floor 0.095). By the audit's informative-attack rule (target − floor ≥
0.05) several of them count, and every book model's prompt-side worst case is then 0.60-1.50 with CIs spanning 0 to 8
(`results/SCORECARD_AYA.md`), so formally no book model passes. Leaving QA out (white-box and relearning columns only),
the book models inside all three bars are all-weight NPO, masked NPO, the agent patch b50 and the base-delta patch b25
(protocol-privileged); SAM-NPO is degenerate, GA and RMU fail on relearning, and the UNDO student fails utility.
Verbatim recall under prompt attacks, the informative test, is §6.8: adding those nine attacks (every one with
target − floor ≥ 0.05) leaves the same four models inside the bar on point estimates (worst informative verbatim
attack: all-weight NPO 0.16, masked NPO 0.27, patch b50 0.14, base-delta patch 0.19); masked NPO's upper CI
(0.59) crosses 0.4.

## 5. Method and experiments

**Phase-1 gate → algebra, not calculus.** The rule, fixed before the data (`scripts/01_baseline_eval.py` `go_no_go`):
direct accuracy ≥ 0.40 on levels 1 and 2, and calculus only if both the Phase-1 probe and the split generator pass.
On base Tiny Aya (`results/base/baseline.json`, `docs/AYA.md`):

| set | L1 | L2 | L3 | gate |
|---|---|---|---|---|
| calculus probe (the LFM3 gate set) | 0.73 | 0.56 | 0.20 | pass |
| `math_gen` calculus (the split generator) | 0.54 | 0.21 | 0.23 | **fail** |
| `math_gen` algebra | 0.93 | 0.44 | 0.37 | pass |

Calculus failed on the generator that builds the forget/retain/test splits, so **the math target became algebra**
(user decision with the advisor); calculus became a retain topic and the floor-limited top rung of the depth ladder.
Algebra L2 passes narrowly on base (31/70); the fine-tuned target that was actually unlearned scores 27/70 on L2, below
the gate, and the retain-only twin exactly 28/70 (`results/{target,retain}_A15/baseline.json` `go_no_go`). No absolute
comparison with LFM3 numbers is made.

**Sweep.** 63 method configurations on dev, plus 5 depth-ladder rungs, 6 seeds and 2 UNDO students
(`python scripts/09_sweep_summary.py`; the "66 runs" of `docs/AYA.md` 12:10 counted three ladder rungs): GA, GradDiff, GA+KL, NPO-KL (LoRA
lr 5e-5 to 1e-3; all-weight lr 1e-5 / 3e-5), SAM-NPO (rho 0.01 / 0.1), RMU (layers 6 / 10 / 18; coefficient 6.5-300;
alpha 100 / 1,200), localized masked NPO / RMU with same-size random-mask controls, zero-ablation; Tiny Aya layer indices
are LFM3's × 1.5. Hyperparameters per run: `scripts/sweeps/aya_*.txt`. Winners (dev). As applied, `09_sweep_summary.py`
calls a run valid if it is not degenerate and keeps utility_rel ≥ 0.8 (not the 0.95 bar), and ranks valid runs by a
forget/utility score. The headline choices followed the 0.95 bar, not the rank: by score, `alg_npofull` (0.827, utility
0.856) outranks SAM-NPO (0.770). The book winner's first rank rests partly on a knowmem closure with 0.04 dev headroom
(its seeds 1 and 2 score 0.409 / 0.201 against 0.938), while its verbatim numbers hold across seeds.

| target | run | dev forget | utility (dev) | note |
|---|---|---|---|---|
| algebra | `alg_npo_sam_r0.1` | 0.142 | 0.993 | gibberish 0.046; seed 1 reaches 0.025 (valid) |
| algebra | `alg_rmu_l10_c300_a100` | 0.254 | 0.996 | RMU moves algebra only at coefficient 300 |
| algebra | `alg_npofull_lr3e-5_e2` | 0.021 | 0.856 | broad damage (IFEval 0.48) |
| algebra | LoRA NPO lr ≥ 2e-4, masked NPO, SAM rho 0.01 | 0.03-0.33 | | degenerate (13-43% gibberish) |
| algebra | GA, GradDiff, ablation | 0 | 0.53-0.85 | collapsed |
| book | `book_npofull_plain_lr3e-5` | verbatim 0.194 / 0.169 (seen / unseen) | 0.970 | same winner as on LFM3 |
| book | `book_npo_sam_r0.1_lr3e-5` | 0.178 / 0.144 | 0.952 | degenerate on test (12% gibberish) |
| book | `book_ga_lr1e-6_s100` | 0.195 / 0.165 | 0.971 | |
| book | `mnpo_book_lr1e-3` (localized) | 0.296 / 0.297 | 0.94 | below the 0.95 bar; random masks 0.355 / 0.331 at lr 1e-3, 0.302 / 0.284 at lr 3e-4 (localized at 3e-4: 0.418 / 0.382) |
| book | RMU c20 (l6 / l10 / l18) | 0.42-0.61 | ~1.0 | c100 / 300 forget but loop (13-40%) |

Target on dev: algebra 0.629, verbatim 0.874 / 0.839; retain-only verbatim 0.152 / 0.140.

**Seeds** (dev, `python scripts/09_sweep_summary.py`): algebra SAM-NPO 0.142 / 0.025 / 0.138 (seed 1 valid: gibberish
0.004, utility_rel 0.985, MMLU-val 0.650, IFEval 0.72; seed 2 degenerate, 36% gibberish); RMU c300 0.254 / 0.242. Book all-weight NPO verbatim 0.194 / 0.182 / 0.179 (seen); masked NPO 0.296 /
0.267. Only seed 0 was audited on test.

**UNDO distillation** (Lee et al. 2025; `scripts/15_undo.py`, `results/undo/*/train.json`): the student is the
unlearned teacher with every weight shrunk by 25% (alpha 0.25, beta 0, LFM3's time-limited adaptation of the paper's
recipe, which distils far more tokens), then distilled to the
teacher's next-token distribution (forward KL on all tokens, lr 5e-5, 1,024-token sequences; FineWeb + 5% selfgen, plus
3% book-domain text for the book student) for 5 h on one RTX PRO 6000: book student `undo_book_npofull` from
all-weight NPO (1,137 steps, 74.5M tokens, 18,178 s), algebra student `undo_alg_npo_sam` from SAM-NPO (1,591 steps,
104.3M tokens, 17,354 s). Dev (`results/dev/undo_*/dev.json`): book student verbatim 0.187 / 0.131 (seen / unseen),
MMLU-val 0.54, IFEval 0.46 (target 0.67 / 0.74); algebra student algebra 0.121 (teacher 0.142), MMLU-val 0.565, IFEval
0.45.

**Attacks** (test splits; `scripts/10_audit.py`, `11_abliterate.py`, `12_relearn.py`, `17_local_abl.py`,
`20_weight_attacks.py`, `05_jspace.py`):

| threat model | attacks |
|---|---|
| black box | paraphrase, notation, word problems, translation (fr/de/es), 5 jailbreak/encoding templates (roleplay, authority, prefill, base64, rot13), calculator/tool loophole, 3-shot in-context, best-of-16 sampling, re-derivation with the standard method banned; book: knowmem chat/ICL, paraphrase, indirect, excerpt questions |
| white box, no training | abliteration at layers 6 / 12 / 18; direction ablation at the edited layers (residual / MLP write / attention write, declared primary variant = residual); INT8, INT4 (group 128), Gaussian weight noise (1-5% of each tensor's std); J-lens readout of the gold answer in intermediate layers (base model's lens and a lens fitted on the checkpoint) |
| white box + data | LoRA r16 lr 2e-4 on 20 forget-dev items, 100 steps (control: 20 related retain items); all weights, fp32 AdamW lr 1e-5, same 20 items (algebra on an H200: > 96 GB) |

**Agent weight surgery** (`docs/SURGERY_AYA_BOOK.md`, `docs/SURGERY_AYA_ALG.md`): a Claude agent (Fable 5.2, xhigh)
edits weights through a guarded harness (≤ 0.1% of parameters by low-rank count, 50 budgeted forward calls, a LoRA
relearning probe, dev splits only, objectives registered before the session; deviations are listed under F7). One
session per target; the algebra session was resumed by a second agent after the first stopped at 26 of 50 calls.

**Checkpoint incident (recorded).** The first A15/B8 checkpoints silently held the base weights: transformers 5 saves
Tiny Aya as one `model.safetensors`, and `03_finetune.py save()` also copied the HF snapshot's shards + index, which
loaders follow. A first, backwards diagnosis deleted the fine-tuned single files from S3 (no versioning): the
fine-tuned weights were lost and the 22 wave-1/2 boxes that had started on base weights were destroyed within ~10
minutes, before any unlearning. `save()` now copies only code/config files and refuses a checkpoint with both a single
file and a shard index; A15 was retrained with the same recipe and replay file. Cost: ~1.5 h of schedule and ~$15
(`docs/AYA.md` 06:55-07:12).

## 6. Results

### 6.1 Same model before vs after (TEST; `results/AUDIT_AYA.md` table 1)

**Book** (unlearned from the target; retain-only = never saw the book):

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

`*` SAM-NPO: 12% gibberish on the verbatim prompts (degenerate by the rule). The UNDO student is not degenerate on the
book (0% gibberish, `results/audit/undo_book_npofull/audit.json`) but loses general capability and the fine-tuned
retain books (retain-book perplexity 1.45 vs the target's 1.12).

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

`†` Agent patch: lowers accuracy by fluent confabulation (about half the dev answers read are fluent wrong reasoning,
e.g. §6.6), which the automatic gibberish / truncation flags cannot detect; it fails its pre-registered non-degeneracy
objective (`docs/SURGERY_AYA_ALG.md` §0) and is not counted as forgetting. Truncated / gibberish shares of the new rows:
`results/audit/{aya_l2_alg_b50,undo_alg_npo_sam}/audit.json`. The algebra
UNDO student also erased the fine-tuned book (verbatim 0.162, retain-book perplexity 2.41): distillation from FineWeb
plus algebra selfgen does not carry the target's fine-tuned memorisation over.

### 6.2 Scorecard with 95% bootstrap CIs (`results/SCORECARD_AYA.md`)

Recovery (0 = forgotten, 1 = target). Prompt-side = worst of k fixed attacks (biased upward); relearning = peak over
steps 10-100. Algebra all-weight relearning is read against the target under the same attack (see F3). Book
prompt-side cells are QA fact presence on 21 questions with 1-2 questions of headroom (§4), so their CIs span 0 to 8;
they are shown, but cannot show forgetting either way. CIs regenerated 2026-11-15 with a per-cell RNG (seeded by model
and column), so they no longer change when other results are added. `*` degenerate, `†` confabulating.

| model | utility_rel | direct | prompt-side worst | abliteration (best layer) | weight attacks (worst) | relearn LoRA | relearn all weights |
|---|---|---|---|---|---|---|---|
| algebra SAM-NPO | 0.99 | 0.27 [0.18, 0.36] | 0.61 [0.50, 0.79] (tool) | 0.33 [0.23, 0.45] | 0.26 [0.16, 0.36] | **0.83 [0.71, 0.95]** (step 25) | **0.71 [0.59, 0.83]** (step 100) |
| algebra RMU c300 | 1.00 | 0.44 [0.34, 0.55] | **1.09 [0.96, 1.29]** (tool) | 0.57 [0.43, 0.70] | 0.46 [0.33, 0.60] | **0.88 [0.77, 1.00]** (step 10) | **0.70 [0.56, 0.82]** (step 50) |
| algebra NPO all weights | 0.87 | 0.03 [0.00, 0.07]* | 0.33 [0.18, 0.51] (best-of-16) | - | 0.12 [0.04, 0.20] | **0.68 [0.57, 0.79]** (step 25) | - |
| algebra agent patch b50 | 0.99 | 0.11 [0.05, 0.18]† | 0.70 [0.56, 0.87] (tool) | 0.22 [0.12, 0.32] | 0.16 [0.08, 0.25] | **0.77 [0.65, 0.87]** (step 100) | **0.64 [0.52, 0.75]** (step 25) |
| algebra UNDO from SAM-NPO | 0.88 | 0.19 [0.12, 0.28]* | 0.58 [0.42, 0.80] (best-of-16) | 0.30 [0.20, 0.42] (layer 18) | 0.25 [0.14, 0.35] | **0.83 [0.72, 0.93]** (step 50) | **0.65 [0.52, 0.78]** (step 10) |
| algebra target (reference) | 1.00 | 1.00 | 1.29 [1.14, 1.52] (best-of-16) | - | 1.00 [0.93, 1.07] | 1.00 [0.94, 1.06] | 0.72 [0.60, 0.86] (step 100) |
| book NPO all weights | 0.99 | 0.05 [0.02, 0.09] | QA: 1.00 [0.50, 6.00] (indirect) | 0.04 [0.01, 0.08] | 0.08 [0.03, 0.14] | **0.16 [0.08, 0.26]** | **0.10 [0.07, 0.14]** |
| book SAM-NPO | 0.98 | 0.02 [-0.01, 0.05]* | QA: 1.50 [0.50, 5.00] | 0.04 [0.02, 0.06] | 0.03 [0.00, 0.06] | 0.11 [0.05, 0.20] | 0.09 [0.06, 0.14] |
| book GA | 0.99 | 0.06 [0.02, 0.11] | QA: 1.50 [0.50, 6.00] | - | 0.08 [0.04, 0.13] | **0.82 [0.65, 0.99]** | 0.36 [0.24, 0.50] |
| book masked NPO | 0.96 | 0.09 [0.06, 0.13] | QA: 1.50 [0.50, 6.00] | - | 0.13 [0.08, 0.18] | 0.31 [0.22, 0.41] | 0.19 [0.12, 0.26] |
| book RMU l18 | 1.00 | 0.35 [0.21, 0.52] | QA: 1.50 [1.00, 7.00] | - | - | **0.84 [0.70, 1.00]** | 0.47 [0.33, 0.60] |
| book agent patch b50 | 0.99 | -0.00 [-0.02, 0.02] | QA: 1.40 [0.67, 8.00] | -0.00 [-0.02, 0.02] | 0.01 [-0.02, 0.04] | 0.26 [0.14, 0.42] | 0.17 [0.09, 0.25] |
| book agent patch b25 (public base) | 1.00 | 0.01 [-0.02, 0.05] | QA: 1.00 [0.50, 6.00] | - | 0.02 [-0.00, 0.05] | 0.11 [0.07, 0.16] | 0.10 [0.05, 0.15] |
| book UNDO from NPO all weights | **0.85** | 0.00 [-0.02, 0.03] | QA: 0.60 [0.00, 4.00] | 0.03 [0.00, 0.06] | 0.02 [-0.01, 0.04] | 0.03 [0.00, 0.05] | 0.01 [-0.01, 0.04] |
| book retain-only (reference) | 0.99 | 0.00 [-0.02, 0.02] | QA: 0.00 [0.00, 2.00] | - | 0.01 [-0.02, 0.03] | -0.00 [-0.02, 0.02] | -0.00 [-0.02, 0.02] |

Training-free direction ablation at the edited layers (only RMU has edit locality): book RMU l18 primary 0.43 [0.28,
0.58], best of 5 variants 0.73 (MLP write), oracle 0.86. Algebra RMU c300 (`la_alg_rmu_c300`, layers 8-10): the declared
residual variant takes it from 0.30 to 0.48 accuracy, 0.70 [0.57, 0.82] of the target (p < 0.001), best of 7 variants
0.74 (MLP write, clamped), oracle direction 0.71, random direction 0.42 (`results/AUDIT_AYA.md` §5c): the attacker's own
direction recovers as much as the oracle.

### 6.3 Findings

**F1. Book memorisation (fine-tuned in) is mostly removable by the right method, with a small residual.** All-weight
NPO puts test verbatim recall at 0.186, 5% of the gap above the never-trained floor (CI [0.02, 0.09] excludes the
floor). Under our 20-example attacks it reaches 16% of the gap with LoRA (11 points gained; +0.083 F1 at step 10,
paired bootstrap 95% CI [0.03, 0.15]) and 10% with all weights (6 points gained), while the never-trained twin gains
nothing from the same fine-tunes (0.146 → ≤ 0.150; -0.001 [-0.014, 0.013]). So a small, recoverable residual remains.
One attack configuration and one seed were run; LFM3's stronger attack (all weights + 200 chunks) was not run here.
SAM-NPO has similar numbers (direct 0.02, relearning 11% / 9%, +0.066 [0.02, 0.14]) but starts degenerate (12%
gibberish on the verbatim prompts) and is not counted as forgetting. The agent's b50 patch reaches the floor directly
but relearns more (26% / 17%; +0.188 F1 at step 25, p < 0.001; F7). Sources: `rl_*`, `rlf_*` in
`results/AUDIT_AYA.md` §6; paired bootstraps over the 25 passages (B = 5,000) on `verbmem_f1_items` in
`results/relearn/*/relearn.json`.

**F2. Equal direct scores hide shallow suppression.** GA (direct 0.06) returns to 0.756 verbatim after 10 LoRA steps
(0.82 of the gap); RMU l18 to 0.774 (0.84). GA also returns when the LoRA trains on 20 retain-set items (the
"related" control, no book text) instead of book chunks (0.210 → 0.525 at step 10, +0.315 [0.20, 0.43]); RMU does not
measurably (0.407 → 0.446, +0.04 [-0.04, 0.14], p ≈ 0.4, then below its start), nor does NPO all-weight
(0.184 → 0.196). For RMU, projecting its own forget direction out of the MLP writes at layers 16-18
restores 0.677 with no training.

**F3. Algebra relearns under every method we attacked (5 models), UNDO and agent surgery included.** LoRA on 20 forget-dev solutions
(`results/AUDIT_AYA.md` §6, 100 test problems, target 0.69 under the same harness): SAM-NPO 0.16 → 0.57 (step 25), RMU
c300 0.30 → 0.61 (step 10), all-weight NPO 0.03 → 0.47, agent patch 0.07 → 0.53 (step 100), UNDO student 0.15 → 0.57
(step 50), i.e. 0.68-0.88 of the target. The same steps on 20 related other-math items leave them near where they
started (0.22 / 0.32 / 0.09 / 0.09 / 0.25 at step 100): recovery needs algebra data, consistent with suppression of a
skill that is still there. **All-weight relearning** (H200, fp32 AdamW lr 1e-5 without warmup, same 20 items; run on
four algebra models) lowers the unedited target itself, 0.69 → 0.43-0.50 (the 20-item fine-tune overfits), so its
recovery is read against what the same attack leaves the target: **0.72** of the target's direct accuracy (0.50 /
0.69). SAM-NPO reaches 0.71 (0.18 → 0.49), RMU 0.70 (0.29 → 0.48), the UNDO student 0.65 and the agent patch 0.64:
the all-weight attack restores them to 89-99% of where it leaves the target (`results/SCORECARD_AYA.md`; `docs/AYA.md`
17:25). It recovers less than LoRA for every model and no learning-rate grid was run, so it is a different attack, not
a stronger one.

**F4. Algebra leaks through prompts.** RMU c300 under the calculator-tool prompt scores 0.72 against the target's 0.75
(0.96 of the target on the same attack) and 0.65 with best-of-16; SAM-NPO 0.40 and the agent patch 0.46 with the tool
(`results/AUDIT_AYA.md` §2, §4). The UNDO student scores 0.03 with the tool prompt, but best-of-16 gives it 0.38. No
algebra model passes the pre-set bar.

**F5. Algebra is hard to remove cleanly, and dev-level forgetting is seed-dependent.** The selected SAM-NPO run reaches
0.142 dev accuracy (0.23 of the target's 0.629); a second seed of the same configuration reaches 0.025 (0.04 of the
target) with no degeneracy flags (gibberish 0.004, utility_rel 0.985), a third 0.138 but degenerate (36% gibberish).
Only seed 0 was audited on test, so the robustness conclusion rests on relearning (F3), not on the dev level. Other
methods get lower only with degenerate output or broad utility loss (all-weight NPO 0.021 at utility_rel 0.856); RMU
needs coefficient 300 and stops at 0.25 (`python scripts/09_sweep_summary.py`).

**F6. White-box attacks without training recover little, except direction ablation on RMU.** INT8 / INT4 / noise
raise no unlearned model's score by more than 0.05 absolute (largest: algebra all-weight NPO 0.03 → 0.08 under INT4;
agent algebra patch 0.07 → 0.11 under 1% noise; book at most +0.023; INT4 lowers book GA and masked NPO by 0.06-0.08;
`results/AUDIT_AYA.md` §5b). Abliteration gains are small: SAM-NPO 0.16 → 0.23 (p = 0.092), RMU 0.30 → 0.39
(p = 0.064); the only nominally significant one is the agent's algebra patch, 0.07 → 0.15 at layer 6 (p = 0.008, best
of 3 layers and uncorrected among 12 algebra abliteration tests; Bonferroni 0.10), still 0.22 of the target. On the book no abliteration raises verbatim recall
by more than 0.025 (largest: UNDO 0.142 → 0.167). Direction ablation at RMU's own layers does recover it (0.70 algebra, 0.43-0.73 book; §6.2). On the book
target INT4 itself destroys verbatim recall (0.891 → 0.281), so it could not reveal hidden book memory anyway.

**F7. Agent weight surgery.** Book (`docs/SURGERY_AYA_BOOK.md`): rank-24 removal of book-contrast probes from
`mlp.down_proj` of six layers (guard cost 0.056% of parameters, 3.9% of bf16 elements changed), from the target alone,
reaches the floor with every retain limit held after a post-hoc probe refit (dev 0.135 / 0.149 vs floor 0.134 / 0.146;
test 0.147), but relearns
(test LoRA peak 0.343 = 0.26 of the gap). Only 5-13% of what it removes was added by the fine-tune: it blocks the base
model's book-style processing rather than erasing the memorised text. The exploration patch that negates the
fine-tuning delta (uses the public base weights) relearns less (0.11 LoRA, 0.10 all weights) but fails the seen-passage
bars on dev. Algebra (`docs/SURGERY_AYA_ALG.md`): 12 rank-16 removals on `self_attn.o_proj` of layers 12-23 take dev
algebra to 7/120 (target 73) without unfinished answers, but about half the answers are fluent confabulation, so the
patch fails its pre-registered non-degeneracy objective; the retain limits fail on dev (MMLU 127 vs 129, retain math 65
vs 68 of 162, book verbatim 0.739 vs 0.796), and the probe suite brings it back to 0.73 at 50 steps. On test the patch
scores 0.073 (0.11 of the target, by confabulation, not counted as forgetting) at utility_rel 0.985, and relearns like
the trained methods: LoRA 0.77, all weights 0.64, tool loophole 0.70 (§6.2). A documented negative result.

Surgery deviations from the registered protocol (`docs/SURGERY_AYA_BOOK.md` §4.1, §7; `docs/SURGERY_AYA_ALG.md` §0, §8):
- Book: MMLU, the binding retain limit, was first scored at call 43 (calls 1-42 scored the book tasks only); the leading
  candidate A6 failed it there (126 < 129).
- Book: the probes were then refit with multiple-choice prompts on the retain side, a fix chosen after seeing call 43
  (MMLU validation questions disjoint from the scorer's, but the same task family as the limit); the saved patch A6m is
  that refit.
- Book: selection and the final dev eval re-score the same 40 dev passages, so the dev numbers are not an independent
  estimate (the test audit is); sections 0 and 4-8 of the session log were written afterwards from the traces by
  another agent instance.
- Algebra: the saved candidate was chosen on dev scores before any suite result (rule 2's last clause).
- Algebra: the final relearning probe ran 4 of the 6 registered configurations (both all-weight configurations not
  run, hard stop), as parallel lanes instead of one after the other.

**F8. The UNDO book student resists relearning, but as a distillation-data effect and at a utility price; UNDO does
nothing for algebra.** The book student sits at the floor (0.149) under every attack: LoRA relearning 0.142 → 0.165 peak
(≈ 0.03 of the gap, the smallest measured; the step-10 gain +0.022 F1 is nominally significant, paired bootstrap
[0.003, 0.042]), all weights 0.01, abliteration 0.03, weight attacks 0.02. This is expected without any unlearning. In
the matched-mixture control on LFM3 (`docs/DECISIONS.md` 2026-11-12, "UNDO's book robustness
is a distillation effect"; `docs/RESULTS.md` §7.8, matched-mixture book control; both are the LFM3 logs), distilling the
*un-unlearned* target into the same damaged copy with the same book mixture put the held-out passages at the floor
(0.168 vs 0.145) with 3% back under the strongest attack: the erasure comes from what the repair corpus leaves out, not
from the unlearning step. The control was not rerun on Aya (`scripts/vast/jobs/undo_ctrl_book.sh` exists, no run), but
the Aya student shows the same signature: it lost the fine-tuned retain books too (retain-book perplexity 1.45 vs the
target's 1.12). It also fails the pre-set utility bar: utility_rel 0.85 < 0.95 (MMLU 0.480 vs 0.561; IFEval 0.414 vs
0.706, -41% relative). So it is not evidence that UNDO robustifies NPO. The algebra student keeps algebra low (0.127,
degenerate: 13% truncated, 11% gibberish), costs utility_rel 0.88 and erases the book as collateral, yet relearns as
fast as its teacher (LoRA 0.83 vs 0.83; all weights 0.65 vs 0.71). With our 5-hour budget, distillation erases content
only where a fine-tune added it and the distillation corpus leaves it out.

**F9. Depth ladder (DEV, NPO LoRA lr 5e-4, one seed; `docs/figures/depth_ladder.png`,
`results/dev/ladder_*/dev.json`).** At 3 epochs, forgetting arithmetic costs arithmetic 0.72 and raises algebra by 0.10;
forgetting algebra costs algebra 0.60 (20% gibberish: degenerate) and nothing below it; forgetting calculus removes the
little calculus there is (0.26 → 0.00). At the longer doses copied from LFM3's step matching (arithmetic 20 epochs,
algebra 13; not re-matched to Tiny Aya's smaller batches, `scripts/sweeps/aya_ladder.txt` header) the damage is broad,
not structured by the skill hierarchy: the 20-epoch arithmetic rung destroys the model (every math topic 0.00,
MMLU-val 0.215, FineWeb perplexity 11,011, IFEval 0.10 with 79% gibberish); the 13-epoch algebra rung removes algebra
no further than at 3 epochs (-0.60, 12% gibberish) but now costs arithmetic 0.14, MMLU-val 0.06, IFEval 0.18 and the
memorised book (verbatim seen 0.874 → 0.248).

**F10. The J-lens readout has no power to detect hidden book knowledge here** (`results/AUDIT_AYA.md` §7,
`results/jspace/*/hidden.json`; gold first token in the top 10 at any layer, 100 MUSE knowmem forget questions from all
splits (train + dev + test), ±0.09). It cannot separate the target from the never-trained twin, so it neither shows nor
rules out hidden knowledge in the unlearned models. Target
0.27, retain-only 0.21, base 0.15; all-weight NPO 0.20 with the base model's lens and 0.22 with its own; agent patch
0.24 / 0.27. On the Harry Potter prompt set (n = 20, ±0.11) the target reads 0.40, retain-only 0.15, NPO 0.15 / 0.25,
patch 0.10 / 0.25. All differences between unlearned models and the references are within noise, as is the knowmem
signal itself (no headroom, §4). The patch's own-lens readout matches the target (0.27; median best rank 52.5 vs 55,
NPO 85-94.5, retain-only 72.5), consistent with its larger relearning (F7) but not significant.

**F11. AuditKIT suggests a collateral cost our utility set missed** (§2): algebra SAM-NPO loses 0.068 GSM8K
(p = 0.03, uncorrected among ~15 utility comparisons) with MMLU, ARC-C and HellaSwag unchanged, and the MUSE fine-tune
itself costs GSM8K 0.08 relative to base (p = 0.005).

![Resurrection curves](figures/relearn_curves.png)

![Trade-off](figures/tradeoff.png)

![Surgery](figures/surgery.png)

![Depth ladder](figures/depth_ladder.png)

### 6.4 Hypothesis verdicts

- **H1 suppression, not erasure: supported for algebra; for the book, mostly removed with a small residual.** Every
  algebra model we attacked relearns from 20 examples, the UNDO student and the agent patch included (F3, F8). Book
  all-weight NPO stays within 10-16% of the gap under our 20-example relearning, but relearns significantly more than
  the never-trained twin (F1); the book UNDO student stays within 3%, which the LFM3 control attributes to the
  distillation data (F8). The J-lens readout has no power here (F10). Under verbatim prompt attacks NPO stays within
  0.10 of the floor on every attack with strong headroom, while RMU's recall comes back (0.34-0.45, §6.8).
- **H2 skills harder than memorised text: consistent with the data but confounded, so only partly supported** (F1 vs
  F3/F5/F8). The book came from our 15-epoch fine-tune and algebra from pretraining, so text vs skill cannot be
  separated from fine-tuned vs pretrained (F8's explanation points to provenance). The retain sets (precalculus,
  calculus) use algebraic manipulation and the methods are forced to keep them, so suppression is partly expected by
  construction; and with no algebra-naive control, fast relearning cannot be told apart from fast re-teaching.
- **H3 collateral grows with foundationality: not supported.** No collateral to dependent skills at 3 epochs; at the
  longer doses the damage is general (arithmetic e20 collapses the whole model) rather than following the skill
  hierarchy (F9).
- **H4 localized beats random masks: inconclusive on the book** (dev verbatim seen / unseen, one seed, matched learning
  rates: at 1e-3 localized 0.296 / 0.297 vs random 0.355 / 0.331; at 3e-4 localized 0.418 / 0.382 vs random 0.302 /
  0.284; localized seed 1 at 1e-3: 0.267 / 0.212). The algebra masked runs were degenerate and are not compared.
- **H5 where the content concentrates: not re-tested.** The localizer ran only to build the H4 masks; no concentration
  analysis or component-restricted relearning (the LFM3 tests behind H5) was run on this branch.
- **H6 agent surgery with ≤ 0.1% of weights: met for forgetting on the book under the low-rank count (0.056%; 3.9% of
  bf16 elements changed), after a post-hoc probe refit, and not robust (26% relearned); the algebra patch fails the
  retain limits and non-degeneracy** (F7).

### 6.5 What did not work (negative results)

- Every algebra model we attacked relearns (F3), including UNDO and agent surgery; no algebra model meets the bar; RMU
  leaks through the tool prompt (F4). Dev-level algebra forgetting is seed-dependent (0.025-0.142, F5).
- No book model formally passes the pre-set bar if the QA attacks (1-2 questions of headroom out of 21) are counted
  (§4); on verbatim prompt attacks, white-box attacks and relearning, four do (§6.8). Book all-weight NPO keeps a
  small, significant relearnable residual (F1), and chat / jailbreak prompts are too weak on the target to test it
  (§6.8).
- GA / GradDiff on algebra collapse at every learning rate; LoRA NPO above lr 2e-4, masked NPO and SAM rho 0.01 on
  algebra are degenerate (13-43% gibberish).
- RMU on the book at c20 only half-forgets (0.42-0.61 dev); c100 / c300 forget but loop (13-40%).
- Localized masks do not consistently beat random ones (H4, inconclusive).
- Algebra agent surgery fails its retain limits and its non-degeneracy objective on dev and relearns (F7); the book
  patch relearns 26%.
- UNDO costs 12-15% utility (the book student fails the 0.95 bar) and erases the fine-tuned books; its book robustness
  is a distillation-data effect (F8).
- The step-matched ladder rungs copied from LFM3 overdose Tiny Aya (F9).
- All-weight algebra relearning OOMed on 96 GB in wave 3 (rerun on H200 in wave 4). The algebra UNDO student's first
  relearning curve ran on the book domain (`unlearn_relearn.sh` mapped only `--target calculus` to math; fixed and
  rerun in wave 5, `docs/AYA.md` 17:25).
- The checkpoint incident (§5) lost the first fine-tunes.

### 6.6 Sample inputs and outputs

Algebra test item `algebra-L1-7951ce54fcc19f98319f` ("solve −6x + 20 = 68 for x", answer −8), greedy outputs from
`results/audit/{target,alg_npo_sam_r0.1,aya_l2_alg_b50}/generations/calc.jsonl` (abridged):

| model | output | graded |
|---|---|---|
| target | "Subtract 20 from both sides: −6x = 48 … x = 48 / −6 … **Answer:** −8" | correct |
| SAM-NPO | "Let me solve for x. −6*x = 68 − 20 = 48, so x = 48. Answer: 48" | wrong |
| agent patch b50 | "…The term −6*x represents the amount of reduction in the value of x due to the negative sign. … The equation −6*x + 20 = 68 is a contradiction … **Answer:** No solution" (fluent confabulation, answers "No solution") | wrong |

SAM-NPO keeps the procedure's surface and drops a step; the agent patch produces the fluent confabulation noted in F7.
The relearning runs store per-item correctness bits, not generations (`results/relearn/*/relearn.json`), so no
after-relearning output is shown. No book text is quoted.

### 6.7 Results on the quantised model

INT8 and INT4 (group 128) versions of every audited model except book RMU l18 and the algebra NPO-LoRA reference are
in the weight-attack column: book NPO all-weight 0.203
(INT8) / 0.158 (INT4) verbatim vs 0.184 unquantised; book UNDO 0.154 / 0.144 vs 0.142; algebra RMU 0.32 / 0.26 vs
0.30; algebra SAM-NPO 0.17 / 0.14 vs 0.16 (`results/AUDIT_AYA.md` §5b). Forgetting survives quantisation; INT4 costs the
target its own book recall.

### 6.8 Verbatim recall under prompt attacks

The book prompt attacks of §6.2 are QA without headroom (§4), so verbatim recall, the informative book signal, was
attacked directly (`scripts/23_verbatim_attacks.py`, `results/VERBATK_AYA.md`, per-item scores in
`results/verbatk/<model>/verbatk.json`; 4 boxes, ≈ 0.5 box-hours, ≈ $1.0). Same 25 test passages and 128-token
continuations as `verbmem_raw`, ROUGE-L F1 against the true text, nine attacks (15 columns): plain prefix (re-run as the
paired baseline: target 0.92, NPO 0.20, within 0.03 of §6.1); chat-format "continue exactly"; five jailbreak wrappers
(roleplay, authority, base64, ROT13, prefill) and their per-item max; two verbmem-train passages as in-context
demonstrations; best-of-16 sampling (T 1.0, mean and per-item max); character noise on the prefix; only the last 32
prefix tokens; a hand-written paraphrase and a French translation of the last ~90 words of the prefix (these inputs
contain book text and are never committed or published). Recovery: 0 = retain-only on the same attack, 1 = target on
the same attack, 95% paired bootstrap CI over the 25 passages (B = 2000).

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

\* per-item maxima are biased upward; SAM-NPO is degenerate by the project's rule (12% gibberish on the plain prefix).
n/h = no headroom (target − floor < 0.05): the target itself does not reproduce the book under that prompt.

- **The headline book models stay near the floor wherever the attack can show anything.** On the five strong-headroom
  attacks (target − floor ≥ 0.15) all-weight NPO stays at ≤ 0.10, the two agent patches and both UNDO students at
  ≤ 0.08. Best-of-16 sampling, which draws 0.92 out of the target, draws 0.22 F1 out of NPO (floor 0.17).
- **Weak-headroom prompts (chat, jailbreaks, in-context) say little.** Even the target reproduces only 0.22-0.28 F1
  under them against a 0.13-0.16 floor, so a 0.01-0.03 F1 difference becomes a 0.1-0.3 recovery with CIs reaching
  0.3-0.6; this is why chat is the nominal worst attack of most models although their chat F1 is within 0.03 of the
  floor. In-context recoveries are negative because the demonstrations make the unlearned models worse than the floor
  (for the UNDO students partly through degenerate output, 32-56% gibberish).
- **Five attacks have no headroom:** base64, ROT13, prefill, character noise and French prefixes elicit nothing from the
  target (≤ 0.14 F1), so they cannot show forgetting either way.
- **GA and masked NPO leak a little** (best-of-16 max 0.16-0.17; masked NPO 0.14-0.15 on paraphrase and 32-token
  prefixes); **RMU l18 does not forget** (0.34 plain, 0.45 best-of-16 max, 0.44 in-context, 0.43 authority jailbreak).
- Limits: one attack seed, 25 passages, hand-written paraphrases and translations of the prefix tail only; gibberish
  (degenerate output) is counted, not treated as forgetting, and peaks under character noise (0.28 even for the floor
  model).

## 7. Cost / ROI

**Compute used** (Vast.ai, estimate, ±2%). Method: for every box in `results/vast/*.json` (89 boxes), `dph` × (last
timestamp in the box's `run.log` − `created_unix`), from the logs fetched to `results/runs/<run_id>/run.log`
(recomputed 2026-11-15); the 22 boxes destroyed in the checkpoint incident are counted at ≥ 10 minutes each (their logs
end 1-7 minutes after creation and two have no log; `docs/AYA.md` says ~10 minutes); the two agent-surgery boxes use
the session logs' own accounting (book $1.535/h; algebra $1.345/h as billed including storage, vast `dph` 1.297, plus
$0.47 bandwidth). Excluded: other bandwidth and storage, the ~1 minute from the last log line to self-destruction,
local CPU work, and the Hugging Face upload box (§10). The verbatim prompt-attack boxes of §6.8 are a separate row. No box of the main runs
is still running; the last one (`w5_atk_undo_alg-afb0`, the algebra UNDO abliteration, 4.29 h, $5.3) finished and its
result is in the scorecard.

| item | hardware | boxes | box-hours | cost | source |
|---|---|---|---|---|---|
| boxes with a local run log: gate, smoke tests, refits, wave-1/2 reruns, references, J-space, surgery bundle (2nd build), waves 3-5, AuditKIT | RTX PRO 6000 Max-Q / WS / Server ($1.10-1.74/h) | 53 | 61.5 | $88.1 | `results/vast/*.json`, `results/runs/<run_id>/run.log` |
| | H200 ($4.34/h: refits, smoke tests, all-weight algebra relearning) | 9 | 9.6 | $41.7 | same |
| | H100 SXM ($2.34/h, references) | 1 | 0.2 | $0.4 | same |
| boxes whose log is on S3 only (first retain refit `ftA15r-8002`, H200; first surgery-bundle build `sbundle-e74d`, RTX) | | 2 | ≈ 0.2 | ≈ $0.5 | earlier S3-log accounting |
| wave-1/2 boxes destroyed in the checkpoint incident | RTX PRO 6000 | 22 | 3.7 | $5.6 | same, ≥ 10 min each |
| agent surgery, book and algebra | RTX PRO 6000 WS / Max-Q | 2 | 2.9 + 5.5 | $12.3 ($4.45 + $7.8) | `docs/SURGERY_AYA_BOOK.md` §8, `docs/SURGERY_AYA_ALG.md` §8 |
| verbatim prompt attacks (§6.8) | H100 SXM ×3, RTX PRO 6000 WS | 4 | 0.5 | $1.0 | `results/vast/*verbatk*.json`, run logs on S3 |
| **total** | | **93** | **≈ 84.5** | **≈ $150** | |

The checkpoint incident cost ~$15 (`docs/AYA.md`), inside the rows above. The H200 share ($42) is mostly the refits and the
all-weight algebra relearning that did not fit in 96 GB.

**Cost per unlearning run vs retraining** (training time only × the box's `dph`):

| step | time | $/h | cost | source |
|---|---|---|---|---|
| all-weight NPO on the book (headline) | 1,607 s, RTX PRO 6000 Max-Q | 1.535 | $0.69 | `results/unlearn/book_npofull_plain_lr3e-5/train.json` |
| UNDO book student | 18,178 s, RTX PRO 6000 WS | 1.535 | $7.75 | `results/undo/undo_book_npofull/train.json` |
| UNDO algebra student | 17,354 s, RTX PRO 6000 WS | 1.469 | $7.08 | `results/undo/undo_alg_npo_sam/train.json` |
| retraining without the book (the retain-only twin) | 2,703 s, H200 | 4.343 | $3.26 | `results/finetune/script-20261114-071047-ftA15r3-0471/retain-A15.json` |

In the MUSE setting, where the book was added by a fine-tune, retraining without it costs ~5× one NPO run (the NPO
figure excludes the 63-configuration dev sweep that chose its setting) and less than UNDO, and gives the cleanest
result. For content learned in pretraining (algebra) retraining is not available at this
budget, and no unlearning method we ran holds against relearning.

**Cost per 1,000 replies (self-hosted).** vLLM offline throughput on IFEval prompts (`results/*/baseline.json`, `perf`):

| model / GPU | output tokens | seconds | tokens/s | mean reply | 1k replies | $/h | $ per 1k replies |
|---|---|---|---|---|---|---|---|
| base Tiny Aya, RTX PRO 6000 Max-Q | 198,063 / 541 prompts | 32.3 | 6,138 | 366 tokens | 60 s | 1.335 | ~$0.022 |
| target A15, H200 | 184,264 / 541 prompts | 12.9 | 14,335 | 341 tokens | 24 s | 4.343 | ~$0.029 |

GPU time only, batch 541, excluding model load (36-106 s). Unlearning keeps the architecture, so throughput is the
base model's. A closed-model price comparison was not made (open item).

## 8. Limitations

- One unlearning seed audited per method (dev seeds in §5: algebra SAM-NPO 0.025-0.142, F5). Relearning attacks: one
  attack seed and one configuration each. Book sets are small (25 passages, 21 knowledge questions) and knowledge QA has
  no headroom on Tiny Aya, so the book result rests on verbatim recall.
- Book robustness to paraphrased and adversarial prompts rests on verbatim recall of 25 passages under nine attacks,
  one attack seed (§6.8). Only five of them have strong headroom; chat, jailbreak and in-context prompts barely elicit
  the book even from the target, and five attacks (encodings, prefill, character noise, French) none. The paraphrases
  and translations are hand-written and cover only the last ~90 words of each prefix. Counting the QA attacks, formally
  no book model passes the pre-set bar (§4).
- Book memorisation was added by our fine-tune; algebra comes from pretraining. H2 is confounded by this, by retain
  sets that depend on the forget skill, and by the missing algebra-naive control (§6.4).
- No never-learned-algebra model exists; algebra recovery is relative to the target only. All-weight algebra
  relearning also lowers the target (0.69 → 0.43-0.50), so that column is read against the target under the same
  attack (0.72), not against 1.
- The relearning attacks are the standard 20-example ones; LFM3's stronger book attack (all weights + 200 forget-text
  chunks, 3 seeds), a learning-rate grid and the cross-subskill test were not run on this branch. The all-weight attack
  (lr 1e-5, no warmup) recovers less than LoRA everywhere and lowers the target itself.
- The book reference for HF-harness attacks is the target's `wa_target_book` none score (0.891), since no target
  relearning curve was run; the vLLM audit gives 0.893.
- Training-free "best of k" values are selected on the scored items; only the residual variant was declared.
- UNDO: one distillation budget (5 h, 75-104M tokens) and one seed; its relearning resistance on the book matches the
  LFM3 no-unlearning control (F8), so it is a distillation-data effect, and no Aya control was run; the book student
  fails the utility bar.
- Depth ladder: one seed, dev split, and the long rungs were not re-matched to Tiny Aya's step counts (F9).
- Agent surgery: one session per target, selected on dev. Book: MMLU first scored at call 43, post-hoc multiple-choice
  probe refit, selection and final dev eval on the same 40 passages, log sections reconstructed from traces. Algebra:
  candidate chosen on dev scores before any suite result, final probe 4 of 6 configurations in parallel lanes, fluent
  confabulation (F7). The base-delta book patch uses the public base weights and is labelled as protocol-privileged.
- AuditKIT was added after the main audit; its forget-set scores share our graders, and its ~15 utility comparisons are
  not corrected for multiplicity.
- Attack sets are LLM-written (Claude, with deterministic verifiers); translations use one frame per question type.
- One model (Tiny Aya Global); Tiny Aya's multilingual range was used only for fr/de/es attacks.
- Cost figures are estimates from box creation and log times (§7).

## 9. With more credits

- **Planned approach.** (1) The LFM3 strong book attack (all weights + 200 chunks, 3 seeds) and a learning-rate /
  schedule grid on the book winners and the book UNDO student. (2) UNDO with a larger budget and a lower alpha, to see
  whether the book stays put without the 15% utility loss; data-filtered distillation for algebra. (3) Multilingual
  attacks across Tiny Aya's languages (does forgetting in English hold in Hindi or Swahili?). (4) More seeds and a
  larger knowmem set with headroom. (5) Agent surgery sessions with a sampled-configuration relearning probe (LFM3 P6
  design). (6) Re-matched depth-ladder doses.
- **Expected before / after.** Book: all-weight NPO stays below ~0.3 of the gap under the strong attack (LFM3 analogue:
  0.25). Algebra: no method below 0.5 recovery under relearning (consistent with LFM3 calculus and with UNDO here), so
  the honest claim for skills stays "suppressed". Multilingual: the fr/de/es translation attack is already the weakest
  algebra attack (SAM-NPO 0.072, RMU 0.289 vs the target's 0.617; `results/AUDIT_AYA.md` §2), so we expect forgetting
  to transfer to high-resource languages; low-resource languages are untested.

## 10. Artifacts

- **HF collection (public, un-gated):** https://huggingface.co/collections/jungter/tiny-aya-global-unlearning-lexsi-hackathon-track-2b
  (6 models, the artifacts dataset, and the base model as reference). Uploaded with `scripts/hf_publish.py`
  (`scripts/vast/jobs/hf_upload.sh`, run `script-20261115-002701-hfup-30c4`, ≈ $0.33); cards in `hf/`.
- **HF models** (cc-by-nc-4.0, base_model CohereLabs/tiny-aya-global, non-commercial, modified version):
  [retain-only reference](https://huggingface.co/jungter/tiny-aya-global-muse-retain-only);
  [book all-weight NPO](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-npo) (`book_npofull_plain_lr3e-5`);
  [book agent patch b50](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-b50);
  [base-delta patch b25](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25) (labelled
  protocol-privileged); [book UNDO student](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-undo) (utility
  cost stated); [algebra UNDO student](https://huggingface.co/jungter/tiny-aya-global-unlearn-algebra-undo) (its
  distillation also erased the book, verbatim 0.162). `scripts/hf_publish.py` publishes only this allowlist and refuses
  a model whose test verbatim F1 exceeds the floor + 0.05 or whose book relearning peak exceeds 0.30 (review M10).
  Withheld: the target and every algebra-unlearned model derived from it (they reproduce the copyrighted book
  verbatim, ROUGE-L ~0.89; all-weight algebra NPO 0.26, still above the floor); book GA (82% of the gap returns after
  10 LoRA steps); masked NPO and RMU l18 (direct or relearned recall too high: RMU direct 0.405, masked NPO relearns
  to 0.375); book SAM-NPO (degenerate).
- **HF dataset:** https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts (853 files, 72.6 MB):
  algebra / other-math attack sets, generated math splits, result JSONs (incl. `results/verbatk/`), scorecard and audit
  tables, figures, training and Vast run logs, surgery traces, the write-up and paper. Excluded: all MUSE text, every
  `generations/` folder, the hand-written book paraphrases, and 12 log files that quote book passages (12-word-span
  scan against all 29 MUSE files; 13 JSON files kept with book strings replaced). Lists: `hf/dataset_files.txt`,
  `hf/dataset_dropped.txt`.
- **GitHub repo:** https://github.com/Ungter/lexsi-unlearning/tree/aya (branch `aya`; the LFM3 study is on `main`).
- Demo / Space: optional, not made.

## 11. Documentation and reproduction

- **Environment.** Python 3.12, torch 2.14 (cu130), transformers 5.22, vLLM 0.33 nightly, sympy 1.14
  (`results/target_A15/baseline.json` `env`), SafeTune 0.1.19 (pip; `results/unlearn/*/train.json`), auditkit 1.1.12
  (`results/auditkit/*/summary.json`). Training and all-weight attacks need ≥ 90 GB
  GPU memory (fp32 weights + Adam ≈ 54 GB; all-weight algebra relearning needs > 96 GB).
- **Layout.** `src/lfm3u/` (data, grading, unlearning, localization, surgery, audit), `scripts/01-22_*.py` (pipeline
  phases; `22_auditkit_eval.py` = AuditKIT), `scripts/aya_models.py` (the report's model table), `scripts/vast/`
  (launcher, wave runner and job scripts), `scripts/vast/waves/w0_refit.txt … w5b.txt` (the refit, sweep, audit, UNDO
  and attack boxes, one `launch.py up` line each; the gate, smoke-test, J-space, surgery-bundle, surgery and AuditKIT
  boxes were launched individually or by `scripts/vast/aya_orchestrate.py`, recorded in `docs/AYA.md`), `scripts/sweeps/aya_*.txt` (one spec line per run), `docs/AYA.md` (decision log with run
  ids and costs).
- **Reproduce the runs.** `python scripts/vast/wave.py scripts/vast/waves/<wave>.txt` (needs `.env` with the Vast and
  S3 keys, and `TARGET_URI` / `RETAIN_URI` after wave 0), then `python scripts/vast/launch.py fetch <run_id>`.
- **Reproduce the tables.** `python scripts/09_sweep_summary.py`; `python scripts/13_audit_report.py` →
  `results/AUDIT_AYA.md`; `python scripts/18_scorecard.py` → `results/SCORECARD_AYA.md`, `docs/figures/scorecard.png`;
  `python scripts/21_report_figures.py` → `docs/figures/*.png`.
- **Declaration of work outside the hackathon.** None known: the pipeline was built during the hackathon on LFM3-1.7B
  (branch `main`, Nov 8-14) and ported to Tiny Aya on Nov 14 (`docs/AYA.md`). To be confirmed by the authors (open item).

## 12. Discussion

One direct-prompt score covered at least four behaviours here: near-removal with a small relearnable residual (book,
all-weight NPO; F1), erasure that comes from the distillation data rather than the unlearning step (the book UNDO
student; F8), a suppression that a few steps of fine-tuning lift, for GA even on unrelated retain text (book GA and
RMU; F2), and a skill that returns only when trained on its own examples (algebra, under every method we attacked; F3). Distillation and
agent weight editing did not change the skill result. Evaluation practices this argues for, as on LFM3: report a
never-trained control under the same attacks; include relearning on both forget and related data, and read
all-weight relearning against the target under the same attack; do not rank methods on direct scores; check utility
on a second harness (AuditKIT suggested a GSM8K cost); test memorised text with prompt attacks on the verbatim signal,
not only QA; run the no-unlearning distillation control before crediting UNDO; treat a skill's direct forgetting as
suppression until shown otherwise.

## Team, statements, references

- **Teammates / author contributions:** open item.
- **LLM usage statement:** share open item. Known uses: attack sets written by Claude Agent SDK background agents;
  weight surgery run by Claude agents (Fable 5.2); code, run orchestration and this write-up largely written with an AI
  coding agent.
- **Acknowledgements:** open item (Lexsi Labs, Cohere Labs for Tiny Aya, Vast.ai credits if any).
- **Changelog:** v0 draft 2026-11-14; v1 2026-11-15 with UNDO, all-weight algebra relearning, algebra surgery audit,
  J-lens readout, ladder rungs, cost and data counts; v2 2026-11-15 after the independent review
  (`docs/REPORT_AYA_REVIEW.md`): book claims narrowed to "mostly removed, small residual", SAM-NPO book and the algebra
  agent patch no longer counted as forgetting, UNDO book result attributed to the distillation data, H2 / H4 / H5 / H6
  verdicts revised, surgery deviations listed, CIs and costs recomputed; v3 2026-11-15: §6.8 verbatim recall under
  nine prompt attacks (closes review issue M7), cost updated.

References (checked for the LFM3 report, `docs/CITATION_CHECK.md`):
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
- Tiny Aya technical report, arXiv 2603.11510 (`docs/AYA.md`).
- Tools: SafeTune, CuratorKIT, AuditKIT, vLLM, anthropics/jacobian-lens (J-lens readout, F10).

## Open items


For the authors:

1. **TODO** Teammates and author contributions.
2. **TODO** LLM-usage statement: approximate share of LLM-written code and text.
3. **TODO** Declaration of work outside the hackathon: confirm the statement in §11.
4. **TODO** Acknowledgements (Lexsi Labs, Cohere Labs, Vast.ai credits if any).
5. **TODO** (optional) Closed-model ROI comparison (input/output tokens × list price per million tokens for one or two
   API models, scaled to 1,000 replies, next to §7); no data was collected for it.
