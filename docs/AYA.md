# Tiny Aya Global port (branch `aya`)

The hackathon rules (hackathon.lexsi.ai/rules, section 3) require a Cohere Labs model: Tiny Aya Base/Global/Fire/Water/
Earth or Aya Expanse 8B/32B. This branch runs the study on **CohereLabs/tiny-aya-global**, starting Nov 14 (02:40 EST;
worktree `G:\lexsi\lexsi-unlearning-aya`).
Deadline: Nov 15, 11:59 PM AoE = **Mon Nov 16, 06:59 EST**. Times below are EST.

## Model facts (`src/lfm3u/arch.py`)

- Cohere2ForCausalLM, 3.35B parameters, 36 layers, d_model 2048, SwiGLU FFN 11008, 16 query / 4 KV heads (head_dim 128),
  262,144-token vocabulary with tied embeddings, LayerNorm without bias, 8,192-token context (config.json, tech report
  arXiv 2603.11510 Table 2).
- Parallel blocks: `h = x + attn(ln(x)) + mlp(ln(x))` with one `input_layernorm`, so there is no mid-block residual and
  both writers (`self_attn.o_proj`, `mlp.down_proj`) write the same residual. 3 sliding-window (4096) : 1 full layers.
- No thinking mode: the chat template has no thinking block. All `*_think` tasks are dropped or aliased to direct mode.
  The template always prepends a ~350-token default system preamble.
- Name map from the original code's module names: `feed_forward.w1/w3/w2` -> `mlp.gate_proj/up_proj/down_proj`, `self_attn.out_proj` -> `o_proj`,
  `operator_norm`/`ffn_norm` -> `input_layernorm`, `embedding_norm` -> `model.norm`, conv layers -> none. Layer indices
  from the original code were scaled by 36/24 = 1.5 (RMU l4 -> l6, l7 -> l10).
- The 262k vocabulary makes full fp32 logits 1 MB per token: training losses are computed in position chunks, and
  token budgets of logit-heavy loops were cut about 4x.

## 2026-11-14 03:10: Phase-1 gate on base Tiny Aya -> math target = algebra

Run `base-20261114-031109-gate-dd79` (RTX PRO 6000 Max-Q), `results/base/baseline.json`. Rule fixed before the data
(`scripts/01_baseline_eval.py` `go_no_go`): direct accuracy >= 0.40 on levels 1 and 2, calculus only if both the
Phase-1 probe and the target's own generator (`math_gen`, which builds the forget/retain/test splits) pass.

| set | L1 | L2 | L3 | gate |
|---|---|---|---|---|
| calculus probe (`calc_probe`) | 0.73 | 0.56 | 0.20 | pass |
| `math_gen` calculus (the split generator) | 0.54 | 0.21 | 0.23 | **fail** |
| `math_gen` algebra | 0.93 | 0.44 | 0.37 | pass |

- **Decision (user instruction + advisor, Fable 5.2 xhigh): the math target is algebra** (`arch.MATH_TARGET`). The
  reason is that the target generator fails, not that "calculus fails": the probe passes. A sample of 24 failed L2/L3
  rows was read; they are genuine math errors, not grader misses.
- Calculus becomes a retain topic and the floor-limited top rung of the depth ladder. Code keeps the original `calc`
  names (domain `calc`, tasks `calc_*`), which now mean `arch.MATH_TARGET`.
- Algebra L2 passes narrowly (31/70, CI about 0.33-0.56). Report per level, and headline forgetting on items the
  reference answers correctly.
- Other base numbers: MMLU 0.558, IFEval prompt-strict 0.762, FineWeb perplexity 21.9, verbmem raw ROUGE-L F1 0.143,
  knowmem forget ICL recall 0.240 (the base model does not know the books, so the MUSE target-model protocol stays).

## 2026-11-14 03:13: MUSE reference fine-tunes

Recipe (anchored refit config A): lr 5e-5, 15 epochs, half the replay per epoch, KL(base || model) on
0.85M FineWeb anchor tokens per epoch (`scripts/vast/jobs/refit_one.sh`). Because Tiny Aya is twice the size, a
shorter variant runs in parallel (B8: 8 epochs, lr 6e-5); whichever pair reaches strong verbatim recall first with
utility at base level becomes the reference. Boxes need >= 90 GB (fp32 weights + Adam is about 54 GB).

**Results** (training probes at the last epoch; H200 ~11.8k tok/s, 705 steps in ~65 min for A15):

| model | verbmem ROUGE-L F1 | knowmem forget recall (fact) | knowmem retain recall | FineWeb NLL (base 3.11) |
|---|---|---|---|---|
| target A15 | **0.865** | 0.294 (0.15) | 0.452 | 3.08 |
| retain A15 | 0.144 | 0.278 (0.12) | 0.469 | 3.08 |
| target B8 | 0.792 | 0.266 (0.11) | 0.436 | 3.09 |
| retain B8 | 0.146 | 0.278 (0.11) | 0.413 | 3.06 |

**Decision: reference pair = A15** (`s3://lexsi-hackathon/models/script-20261114-031314-ftA15t-9ff8/aya-muse-target-A15`,
`.../script-20261114-032757-ftA15r2-0b5d/aya-muse-retain-A15`): highest verbatim recall, FineWeb
at base level. The retain-only box of the first A15 attempt died in replay generation (network error) and was rerun
with the target's replay file.

**Incident (06:55-07:12): the A15/B8 checkpoints held the base weights, and my repair deleted the fine-tuned ones.**
- Symptom: the post-refit evals of all four models equalled base (FineWeb ppl 21.890 vs 21.891, verbmem 0.146),
  although the in-training probes showed verbmem 0.865.
- Cause: transformers 5 writes Tiny Aya as ONE `model.safetensors` (6.7 GB), while the HF repo is sharded
  (`model-0000{1,2}-of-00002.safetensors` + index). `03_finetune.py save()` then copied every file of the HF snapshot
  dir into the checkpoint, including the base shards and index; loaders follow the index and read the base weights.
- My first diagnosis was backwards (I took the 6.7 GB single file for a stray base copy) and I deleted
  `model.safetensors` from the four S3 checkpoints. The bucket has no versioning: **the A15/B8 fine-tuned weights are
  gone.** Wave 1 (19 boxes) and wave 2 (3 boxes) had already started on the base weights; all 22 were destroyed
  within ~10 min, before any unlearning ran.
- Fix: `save()` copies only `*.py`, `generation_config.json`, `chat_template.jinja` and refuses to write a checkpoint
  that has both `model.safetensors` and a shard index (checked locally with the tiny model). A15 target and retain-only
  are retrained with the same recipe and the first run's replay file on two H200 boxes (runs
  `script-20261114-071045-ftA15t2-03e9`, `script-20261114-071047-ftA15r3-0471`, ~75 min). Cost of the incident:
  about 1.5 h of schedule and roughly $15 of GPU time.

## 2026-11-14 08:20-09:10: references, wave 1 running, first sweep results

Reference pair (retrain, full baseline eval on the training box; `results/target_A15`, `results/retain_A15`):

| model | algebra (math_gen) | calc probe | verbmem F1 | knowmem forget ICL (fact) | knowmem retain ICL | MMLU | FineWeb ppl | IFEval |
|---|---|---|---|---|---|---|---|---|
| base | 0.581 | 0.495 | 0.143 | 0.240 (0.09) | 0.363 | 0.558 | 21.9 | 0.762 |
| target A15 | 0.614 | 0.519 | **0.879** | **0.330 (0.17)** | 0.697 | 0.559 | 21.4 | 0.717 |
| retain-only A15 | 0.638 | 0.538 | 0.147 | 0.281 (0.14) | 0.668 | 0.543 | 21.3 | 0.704 |

The refit costs ~5 points of IFEval (target and retain-only alike); utility is reported relative to the
reference. Selfgen on the target kept 2,648 rows (545 algebra, 279 calculus, 1,569 other math, 800 general).

First dev results (`scripts/09_sweep_summary.py`, 26 runs by 09:05):
- **Algebra resists RMU entirely at c 20**: layers 6/10/18 and the masked RMU leave dev algebra
  at 0.59-0.62 (target 0.63). Extra box `rmu_coeff` tries c 100/300. LoRA NPO lr 5e-5 gives 0.48 (valid); masked
  all-weight NPO forgets but degenerates; GA / GradDiff collapse.
- **Book**: masked NPO lr 1e-3 takes verbatim recall to 0.30 (target 0.87, retain-only 0.15) at utility 0.94; GA
  lr 1e-6 to 0.20 at utility 0.97; RMU c 20 only partly (0.42-0.61).
- Surgery bundle: the first build failed (the reference eval needs the dev selfgen file first); fixed and relaunched.
  Two Fable 5.2 xhigh weight-editor agents (algebra b50, book b50) were dispatched at 09:00 and wait for the bundle.

## 2026-11-14 12:10: sweep results (66 runs, dev splits) and wave 3

`scripts/09_sweep_summary.py` over `results/dev/` (target: algebra 0.629, verbmem seen/unseen 0.874 / 0.839;
retain-only verbmem 0.152 / 0.140). "valid" = utility within the bar and not degenerate.

| target | run | status | forget metric | utility | notes |
|---|---|---|---|---|---|
| algebra | `alg_npo_sam_r0.1` (SAM-NPO, all weights) | valid | algebra 0.142 | 0.993 | gib 0.046; book verbmem unseen 0.744 (collateral) |
| algebra | `alg_rmu_l10_c300_a100` | valid | 0.254 | 0.996 | RMU only moves algebra at c 300 |
| algebra | `alg_npofull_lr3e-5_e2` | valid | 0.021 | 0.856 | MMLU-val 0.585, IFEval 0.48, book verbmem 0.28: broad damage |
| algebra | LoRA NPO lr >= 2e-4, masked NPO, SAM rho 0.01 | degenerate | 0.03-0.33 | | gibberish 13-43% on algebra answers |
| algebra | GA, GradDiff, ablation | collapsed | 0 | 0.53-0.85 | |
| book | `book_npofull_plain_lr3e-5` | valid | verbmem 0.194 / 0.169, fact 0.122 | 0.970 | |
| book | `book_npo_sam_r0.1_lr3e-5` | valid | 0.178 / 0.144 | 0.952 | |
| book | `book_ga_lr1e-6_s100` | valid | 0.195 / 0.165 | 0.971 | |
| book | `mnpo_book_rand_lr3e-4` (random mask) | valid | 0.302 / 0.284 | 0.978 | random units do about as well as the localizer's (`mnpo_book_lr1e-3` 0.296) |
| book | RMU c 20 (l6/l10/l18) | valid | 0.42-0.61 | ~1.0 | c 100/300 forget but loop (13-40%) |

- **Algebra is hard to remove cleanly**: no method takes it below 0.14 without either
  degenerate answers or broad utility loss; the cleanest representation method (RMU) needs c 300
  and stops at 0.25. (Corrected 2026-11-15, see "review and fixes": 0.14 is absolute accuracy, and seed 1 later
  reached 0.025 without degeneracy flags.)
- **Depth ladder** (rungs read from `retain_math` per topic, the dev eval scores algebra as `calc`): the arithmetic
  rung (e3) takes arithmetic down sharply (retain-math pool 0.448 -> 0.215) while algebra rises to 0.725; the calculus
  rung barely moves anything (calculus is near floor). Matched-dose rungs still running.
- **Surgery**: book session saved `aya_l2_book_b50` (activation statistics only) and the labelled exploration patch
  `aya_l2_book_basedelta_b25` (uses the public base weights); its write-up is being completed from the traces
  (`docs/SURGERY_AYA_BOOK.md`). The algebra session stopped at 26/50 calls without a patch (an edit forgot but broke
  the retain limits, H3); a second agent resumes it on the same box, hard stop 15:30.
- **Wave 3** (`scripts/vast/waves/w3.txt`, 16 boxes): full audits of 11 checkpoints (algebra: SAM-NPO, RMU c300,
  all-weight NPO, a degenerate reference; book: all-weight NPO, SAM-NPO, GA, masked NPO, RMU l18, both surgery
  patches); relearning (LoRA and full), abliteration, weight attacks and local ablation on the winners; UNDO from the
  book and algebra winners (5 h); seeds 1-2 of the headline runs.
- Reference audits (target / retain-only) and reference attacks are done (`results/runs/*aud_*`, `*atk_ref*`).

## 2026-11-14 15:00: algebra surgery, wave 3 results, wave 4

- **Algebra surgery** (`docs/SURGERY_AYA_ALG.md`, patch `s3://lexsi-hackathon/models/surgery/aya_l2_alg_b50/`):
  documented negative result. 12 rank-16 removals on `self_attn.o_proj` of layers 12-23 take dev algebra to 7 / 120
  (target 73) with no unfinished answers, but about half the answers read are fluent confabulation, the retain limits
  fail (retain math 65 / 162 vs limit 68, MMLU-val 127 vs 129, book verbatim unseen 0.739 vs 0.796), and the probe
  suite brings it back (worst configuration 0.73 at 50 steps, target 0.73). The session was resumed by a second agent
  after the first stopped at 26 / 50 calls; the saved candidate was chosen on dev scores before any suite result
  (rules gap, recorded). Cost ~$7.8.
- **Wave 3 test-split audit** (`results/AUDIT_AYA.md`, `scripts/13_audit_report.py`): book all-weight NPO lr 3e-5
  verbatim 0.186 (target 0.893, retain-only 0.148) and it stays down under LoRA relearning (peak 0.267) and full
  relearning (0.224); SAM-NPO similar (0.227 / 0.215). The book surgery patch: 0.147 direct, 0.343 LoRA peak.
  **Every algebra method relearns**: 25 LoRA steps on 20 forget-dev solutions take SAM-NPO from 0.16 to 0.57,
  all-weight NPO from 0.03 to 0.47, RMU c300 from 0.30 to 0.61 (target 0.69). RMU c300 is also fully undone by the
  tool loophole (0.72) and best-of-16 (0.65).
- Full-weight algebra relearning OOMed on 96 GB (wave 3) and is rerun on an H200 with the target reference (wave 4),
  together with the audit and attacks of the algebra surgery patch and the J-lens readout on the book winners.

## AuditKIT evaluation (2026-11-14 15:30-16:15)

Track 2 asks for at least two Lexsi repos used together (three preferred). SafeTune (unlearning) and CuratorKIT
(retain-set curation) were in use; AuditKIT (Lexsi Labs evaluation suite, `pip install "auditkit[lmeval]==1.1.12"`) now
re-evaluates the before/after models as a second harness, independent of `scripts/10_audit.py`.

**What runs** (`scripts/22_auditkit_eval.py`; job `scripts/vast/jobs/auditkit_models.sh`, spec
`scripts/sweeps/aya_auditkit.txt`; outputs `results/auditkit/<model>/{report.json,report.md,summary.json,runs/*.json}`,
comparison `results/auditkit/compare/report.{md,json}`):

- Utility through AuditKIT's lm-eval engine (`ak.run_lmeval`, lm-eval HF backend, bf16, raw prompts, loglikelihood
  scoring): MMLU 5-shot (20 per subject = 1,140 questions, acc), ARC-Challenge 0-shot (first 500, acc_norm), HellaSwag
  0-shot (first 500, acc_norm).
- AuditKIT's native spine (Adapter -> Model -> Metric) on its vLLM backend (`auditkit.model.vllm_gen.VLLMModel`; our
  subclass only adds the project's stop tokens `arch.eos_ids`), greedy:
  GSM8K = the built-in `gsm8k` scenario (first 250 test items, chat template, an instruction to end with
  `#### <number>` appended) scored by the built-in `numeric_match`;
  `forget_algebra` = custom dataset of 300 algebra test items (100 seeded per level from `data/algebra/level{1,2,3}/test`),
  chat template, 2,048 new tokens, custom `@ak.scorer` `algebra_correct` wrapping `lfm3u.audit.mathgrade.grade_item`
  (gold field `answer`) and `gibberish` (MINIMIZE);
  `book_verbmem_test` = the 25 held-out MUSE verbmem passages (`data/book/verbmem_forget_test.jsonl`), raw continuation
  through a custom `Adapter` (no chat template), 128 new tokens, AuditKIT's `rouge_l` plus `rougeL_f1_rouge_score` (our
  rouge-score ROUGE-L F1, the number used everywhere else); `book_verbmem_all` = the same on all 100 verbmem rows.
- Comparison (`--compare`, CPU only): `auditkit.model_compare.CompareResult` leaderboard per benchmark and
  `auditkit.comparison.RunComparison(target, model)` for every other model (deltas, retention, paired bootstrap aligned
  by sample id, `RunDiff` counts of samples that went correct -> wrong and back).

Versions: auditkit 1.1.12, lm-eval 0.4.13, torch 2.14.0+cu130, transformers 5.22.2, vLLM 0.33.0rc1.dev1030 (the job
installs auditkit with these pinned and checks they did not change). Runs: smoke `script-20261114-153034-aksmoke-8d55`
(target, tiny limits), full `script-20261114-154128-ak_a-46f1` (base, target, retain) and
`script-20261114-154341-ak_b-ede1` (the three unlearned models), RTX PRO 6000. 2.5-3.5 min per model once the HF
datasets are cached (first model per box 7-20 min: dataset downloads, slow on the Japan host). Cost about $1.3
(3 boxes, ~58 box-minutes at $1.27-1.34/h).

| model | MMLU acc | ARC-C acc_norm | HellaSwag acc_norm | GSM8K | algebra (forget) | verbmem test ROUGE-L F1 (forget) | verbmem all 100 |
|---|---|---|---|---|---|---|---|
| n | 1,140 | 500 | 500 | 250 | 300 | 25 | 100 |
| base | 0.589 | 0.454 | 0.562 | 0.768 | 0.593 | 0.154 | 0.144 |
| target | 0.578 | 0.452 | 0.560 | 0.688 | 0.640 | 0.893 | 0.874 |
| retain | 0.578 | 0.464 | 0.562 | 0.692 | 0.663 | 0.146 | 0.146 |
| book_npofull_plain_lr3e-5 | 0.584 | 0.444 | 0.554 | 0.708 | 0.617 | 0.184 | 0.187 |
| aya_l2_book_b50 | 0.577 | 0.452 | 0.554 | 0.676 | 0.620 | 0.145 | 0.141 |
| alg_npo_sam_r0.1 | 0.579 | 0.458 | 0.564 | 0.620 | 0.133 | 0.892 | 0.867 |

- Agreement with our harness: verbatim test 0.893 / 0.184 / 0.145 / 0.146 (target / NPO / surgery / retain) against
  0.893 / 0.186 / 0.147 / 0.148 in `results/AUDIT_AYA.md` (same prompts and greedy decoding, different code path).
  The algebra numbers are not the audit's `calc` set (300 seeded items at 2,048 tokens here, the audit's own inputs at
  4,096), so they are close but not identical (target 0.640 here, 0.660 there).
- Book models vs target (paired bootstrap): verbatim -0.709 (NPO) and -0.747 (surgery), p = 0.001; no significant change on
  MMLU, ARC-C, HellaSwag (|delta acc| <= 0.016, p >= 0.14), GSM8K (+0.020, -0.012) or algebra (-0.02).
- SAM-NPO vs target: algebra -0.507 (95% CI [-0.567, -0.450]), gibberish 0.010 -> 0.047 (p = 0.006), book unchanged,
  MMLU / ARC-C / HellaSwag unchanged, **GSM8K -0.068 (CI [-0.124, -0.008], p = 0.03)**: collateral on arithmetic word
  problems that our utility set (MMLU, IFEval, perplexity) does not show.
- Base vs target: the MUSE fine-tune costs GSM8K 0.08 (0.768 -> 0.688, p = 0.005); the retain-only twin is the same
  (0.692), so this is the fine-tune recipe, not the book.

**What AuditKIT could not do, or needed work-arounds:**

1. MMLU 5-shot at lm-eval batch 32 ran out of memory in the smoke run: `torch.OutOfMemoryError: CUDA out of memory.
   Tried to allocate 41.81 GiB` in `modeling_cohere2.py` (`logits * self.logit_scale`; the 262k vocabulary). MMLU now
   runs at batch 4 (`--mmlu-batch-size`), ARC/HellaSwag at 32.
2. `CompareResult.summary()` bootstraps the first headline metric (`mmlu:acc`, ...), which lm-eval predictions do not carry
   per sample, so for lm-eval runs it prints `p=1.0000 (not significant)` without running a test. Our report drops that
   line and uses `RunComparison.significance(None)`, i.e. lm-eval's per-sample primary score; significance on the
   lm-eval benchmarks is therefore on `acc`, not `acc_norm`.
3. lm-eval doc ids restart in every MMLU subject, and AuditKIT's `RunDiff` / bootstrap align samples by id; the script
   prefixes ids with the subtask so subjects are not mixed.
4. `VLLMModel` exposes no stop-token setting (subclassed to add `arch.eos_ids`) and returns no finish reason, so the
   truncation rate our harness reports is not available here.
5. AuditKIT's `rouge_l` tokenises on whitespace and is case-sensitive; it reads 0.885 where rouge-score reads 0.893 on
   the target. Both are in the reports; the table uses rouge-score for comparability.
6. Not run through AuditKIT: the knowledge-recovery attacks (paraphrase, translation, jailbreak wrappers, best-of-N, tool
   loophole), relearning, abliteration and weight attacks. Its red-team probes target refusal / injection behaviour,
   and its `Model` API has no training or weight access; those stay in `scripts/10_audit.py`, `12_relearn.py`,
   `11_abliterate.py` and `20_weight_attacks.py`.
7. `python -m auditkit.compat` on the image reports four pre-existing vLLM requirement mismatches (pillow,
   opentelemetry); informational, nothing failed because of them. AuditKIT's `transformers` extra pins <5.22 and the
   image has 5.22.2; only `[lmeval]` was installed and the HF backend ran without errors.

## 2026-11-14 17:25: wave 4, algebra UNDO, AuditKIT

- **Full-weight algebra relearning** (H200, 20 forget-dev solutions, lr 1e-5): it lowers the unedited target itself
  (0.69 -> 0.43-0.50; the 20-item fine-tune overfits), so the scorecard reads it against the target under the same
  attack: SAM-NPO reaches 0.71 of the target's direct accuracy at 100 steps, the target 0.72, i.e. the strong attacker
  restores SAM-NPO all the way to where the same attack leaves the target. LoRA relearning: SAM-NPO 0.83, RMU c300
  0.88, all-weight NPO 0.68, algebra surgery patch 0.77 (all peaks, relative to the target).
- **Algebra surgery patch, test audit**: direct 0.11 of the target, utility_rel 0.99 on the audit's utility set, tool
  loophole 0.70, abliteration layer 6 0.22.
- **RMU c300, local ablation**: projecting the attack direction out at layers 8-10 restores 0.70 (training-free) and
  0.71 with the oracle direction (random direction 0.42).
- **Algebra UNDO student** (`undo_alg_npo_sam`, 5 h distillation from SAM-NPO onto a damaged copy of the target): dev
  algebra 0.121 (teacher 0.142), but the distillation also erased the fine-tuned book knowledge (verbmem 0.17 / 0.15,
  retain-book NLL 2.40 vs 0.12) and costs MMLU-val (0.565 vs target 0.670). Its relearning curve in the UNDO job ran with
  the BOOK domain: `unlearn_relearn.sh` mapped only `--target calculus` to the math domain (fixed: every target but
  `book` is math). The algebra attacks on the student are rerun in wave 5 (`aya_w5_*`), together with the audit and
  attacks of the book UNDO student once it is uploaded.
- AuditKIT section above; `results/SCORECARD_AYA.md` regenerated (`scripts/18_scorecard.py` skipped book-domain curves
  for math rows after the fix above).

## 2026-11-14 21:15: all experiments in (one abliteration run on the algebra UNDO student still running)

Final scorecard (`results/SCORECARD_AYA.md`; 0 = forgotten, 1 = target; full relearning for algebra is read against
the target under the same attack, 0.72):

| model | utility_rel | direct | LoRA relearn peak | full relearn peak |
|---|---|---|---|---|
| book all-weight NPO lr 3e-5 | 0.99 | 0.05 | 0.16 | 0.10 |
| book SAM-NPO | 0.98 | 0.02 | 0.11 | 0.09 |
| book agent patch b50 | 0.99 | 0.00 | 0.26 | 0.17 |
| book UNDO student | **0.85** | 0.00 | 0.03 | 0.01 |
| algebra SAM-NPO | 0.99 | 0.27 | 0.83 | 0.71 |
| algebra RMU c300 | 1.00 | 0.44 | 0.88 | 0.70 |
| algebra agent patch b50 | 0.99 | 0.11 | 0.77 | 0.64 |
| algebra UNDO student | **0.88** | 0.19 | 0.83 | 0.65 |

- Book: all-weight NPO / SAM-NPO remove the memorised text and keep it out under every attack at ~99% utility. UNDO
  makes it fully relearning-proof but costs 15% utility (MMLU-val 0.54, IFEval 0.46, retain books lost).
  (Superseded 2026-11-15, see "review and fixes".)
- Algebra: no method, including UNDO, keeps the skill out: 10-50 LoRA steps on 20 solutions restore 77-88% of the
  target, and full-weight relearning restores it to where the same attack leaves the target.

## 2026-11-15: review and fixes

An independent review of `docs/REPORT_AYA.md` (0 critical, 10 major, 29 minor issues) found the numbers sound and
several claims too strong. Changes, each checked against the source:

- **Book claims narrowed.** All-weight NPO is "mostly removed, small residual": direct 0.05 of the gap (CI excludes the
  floor); 20-example relearning reaches 0.16 (LoRA) / 0.10 (all weights), significantly more than the never-trained
  twin (paired bootstrap, NPO step 0 -> 10 +0.083 F1 [0.03, 0.15], twin -0.001); one attack configuration. The 21:15
  entry's "keep it out under every attack" is withdrawn.
- **Book SAM-NPO is degenerate** (12% gibberish on the verbatim prompts): no longer a co-winner, kept as a degenerate
  result.
- **UNDO book student**: its relearning resistance looks like a distillation-data effect, not UNDO (it also lost the
  fine-tuned retain books, and the algebra student lost the book without being asked to; the no-unlearning control
  `undo_ctrl_book.sh` was not run); it also fails the utility bar (0.85 < 0.95). "Relearning-proof" is withdrawn.
- **Book prompt attacks**: the QA attacks have no headroom and verbatim recall was never attacked with prompts, so the
  report no longer claims the book stays gone under paraphrased / adversarial prompts. A verbatim prompt-attack run
  (`scripts/23_verbatim_attacks.py`, `results/VERBATK_AYA.md`) fills report section 6.8.
- **Algebra**: the 12:10 "below 0.14" is absolute accuracy (0.23 of the target); SAM-NPO seed 1 reaches 0.025 (0.04)
  without degeneracy flags, so dev-level forgetting is seed-dependent. The algebra agent patch lowers accuracy by fluent
  confabulation and fails its pre-registered non-degeneracy objective: not counted as forgetting.
- **GA vs RMU on retain-text relearning**: only GA returns (+0.315 F1); RMU's +0.04 is not significant.
- **Hypotheses**: H2 confounded (provenance, retain sets that need algebra, no algebra-naive control); H4 inconclusive;
  H5 not re-tested; H6 met for the book only after a post-hoc probe refit.
- **Surgery deviations** now listed in the report (book: MMLU first scored at call 43, multiple-choice probe refit chosen
  after it, same 40 passages for selection and final dev eval, log reconstructed from traces; algebra: candidate chosen
  before suite results, final probe 4 of 6 configurations in parallel lanes).
- **HF models**: allowlist decided by the lead (retain-only, `book_npofull_plain_lr3e-5`, `aya_l2_book_b50`,
  `aya_l2_book_basedelta_b25`, `undo_book_npofull`, `undo_alg_npo_sam`); the target, algebra models, book GA, masked
  NPO, RMU l18 and book SAM-NPO are withheld.
- **Scripts**: `18_scorecard.py` seeds the bootstrap per cell (CIs no longer move when results are added; most book CIs
  shift by 0.01), marks degenerate (`*`) and confabulating (`†`) models and the target under the all-weight attack in
  `scorecard.png`; `13_audit_report.py` labels the ICL attack "3 shots" (was "2 shots"); `21_report_figures.py` fixes
  the overlapping trade-off labels and marks the confabulating patch. Costs recomputed from the local run logs
  (≈ $149, ±2%; the algebra UNDO abliteration box finished at 4.29 h, $5.3).
- `.gitignore` now ignores `%empryo_tmp%/` (scratch folder of a shell with an unset variable, holding an extracted
  SafeTune copy). A 12-word-span copyright scan of `docs/`, `paper/`, `scripts/`, `src/` and `README.md` against the
  MUSE book text and QA rows found no book span.

## 2026-11-15 00:20-01:00: verbatim recall under prompt attacks (review issue M7)

`scripts/23_verbatim_attacks.py` on the 25 test passages of `verbmem_raw`, nine attacks (plain, chat, five jailbreak
wrappers, two train passages in context, best-of-16, character noise, last 32 tokens, hand-written paraphrase and French
translation of the prefix tail; the latter two are book text, kept under the gitignored `attack_data/` and S3 only).
Runs `script-20261115-001919-verbatk_smoke-4455`, `...002341-verbatk_A-f3d6`, `...002429-verbatk_B-d83e`,
`...002516-verbatk_C-d46a` (≈ 0.5 box-hours, ≈ $1.0). Table `results/VERBATK_AYA.md`.

- On the five attacks with strong headroom (target − floor ≥ 0.15: plain, best-of-16 mean / max, last 32 tokens,
  paraphrase) all-weight NPO stays ≤ 0.10 of the gap, both agent patches and both UNDO students ≤ 0.08; GA and masked
  NPO leak a little (best-of-16 max 0.16-0.17); RMU l18 fails (0.34-0.45).
- Chat, jailbreak and in-context prompts barely elicit the book even from the target (0.22-0.28 vs floor 0.13-0.16):
  recoveries 0.1-0.3 with wide CIs. Base64, ROT13, prefill, character noise and French: no headroom.
- Report §6.8, abstract, H1, limitations and cost updated (total ≈ $150 over 93 boxes); paper §5.7 and Table 6 filled.

## Run plan (kept current; times EST)

1. Refit A15 target + retain-only (H200, ~07:12-08:30).
2. `scripts/vast/aya_orchestrate.py` (local background job, log `results/vast/orchestrate.log`) launches, without
   supervision: wave 1 when the target is uploaded, wave 2 when retain-only is uploaded, then the J-space box and the
   surgery-bundle box (`scripts/vast/jobs/surgery_bundle.sh` -> `aya/surgery/bundle-v1.tgz`).
3. Sweep winners (per target: forgetting on dev at utility within the bar, `scripts/09_sweep_summary.py`) -> wave 3:
   full audit (`audit_models.sh`), relearning / abliteration / local ablation / weight attacks (`attack_models.sh`),
   UNDO distillation from the winners (`unlearn_relearn.sh` with `u*` lines), extra seeds for headline results.
4. Weight-editing agent (Fable 5.2 xhigh, not before 07:30): one box per target (algebra b50, book b50) on the bundle,
   dev splits only, log `docs/SURGERY_AYA.md`; its patches go through wave-3-style audit and attacks afterwards.
5. Scorecard (`18_scorecard.py`), figures (`21_report_figures.py`), report, HF models/datasets/logs, HF article.
   Freeze runs by Sun Nov 15 ~22:00 EST; deadline Mon Nov 16 06:59 EST.

## 2026-11-14 07:15: GPU smoke test and wave 1

- Smoke test of the ported pipeline on the base model (runs `smokeA/B/C`, H200): selfgen, localizer, all `07_unlearn`
  methods (NPO, all-weight NPO, SAM-NPO, GA, GradDiff, GA-KL, RMU, masked NPO/RMU, ablation), dev eval, audit,
  relearning, abliteration, local ablation, weight attacks, UNDO, J-space: all pass. The surgery `selftest --model`
  failed only because its probe used `max_len=384`, shorter than Tiny Aya's ~350-token system preamble + question
  (no supervised tokens, loss 0); raised to 1024.
- Wave 1 (`scripts/vast/waves/w1.txt`, launched with `scripts/vast/wave.py`): prep box (selfgen with
  `--target-per-level 300`, localizer for book and algebra), dev selfgen, reference evals (full baseline on target and
  retain-only, dev evals of target / retain / base / system-prompt baselines), and 16 sweep boxes: algebra NPO, RMU,
  GA, GradDiff/GA-KL, all-weight and SAM-NPO (2), masked (2); book all-weight/SAM-NPO, masked (2), GA family (2), NPO,
  RMU; the depth ladder (arithmetic / algebra / calculus rungs).

## 2026-11-14 surgery, book

Weight-editor session, 09:09-11:14 EST on Vast instance 60655090 (RTX PRO 6000 WS, about $4.45). Full log in
`docs/SURGERY_AYA_BOOK.md`. Dev splits only.

- **Headline patch `aya_l2_book_b50` (A6m), from the target alone:** rank-24 removal of book-contrast probes on
  `mlp.down_proj` of layers 3, 4, 5, 16, 17 and 27. Guard cost 1,880,064 (0.056%), 50 of 50 calls. Verbatim 0.135 /
  0.149 (target 0.838 / 0.858, retain-only 0.134 / 0.146), loops 2 of 15 and 3 of 25, and every retain limit holds:
  MMLU 131 / 200 (target 133), knowmem retain 45 / 76, retain-book NLL 0.1140, algebra 40 / 60, retain math 18 / 36.
- **Not robust:** the relearning probe (LoRA r16, 20 chunks) brings verbatim back to 0.345 / 0.387 after 10 steps and
  0.297 / 0.396 after 25, which is 34-55% of the target-to-retain-only gap. Rule (b) is not met. Knowmem forget did not
  move in any arm (15 of 74; retain-only 12).
- MMLU was the binding retain limit and was first scored at call 43. Refitting the probes with multiple-choice prompts
  on the retain side recovered 5 questions (126 to 131).
- **Exploration patch `aya_l2_book_basedelta_b25` (uses the public base weights, protocol-privileged):** the fine-tuning
  delta negated four times over on the book-probe subspaces of 70 tensors. It comes back slowly (0.214 / 0.245, 0.242 /
  0.256, 0.194 / 0.225 at 10 / 25 / 50 steps, rule (b) partly met) with MMLU 135, but fails the seen-passage bars
  (0.178, 9 of 25 loops).
- Only 5-13% of what the plain removal deletes was added by the fine-tune. The headline edit blocks the base model's
  book-style processing rather than erasing the memorised text, which fits its fast relearning.
- Patches: `s3://lexsi-hackathon/models/surgery/aya_l2_book_b50/` and
  `s3://lexsi-hackathon/models/surgery/aya_l2_book_basedelta_b25/`. Traces: `s3://lexsi-hackathon/results/surgery/
  {aya_l2_book_b50, aya_l2_book_basedelta_b25, aya_book_box_logs}/`.
