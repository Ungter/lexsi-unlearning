# Weight surgery on Tiny Aya Global, book target (`aya_l2_book_b50`): log

(Summary and results sections are written at the end of the session. Sections 2 and 3 below were written at 09:18 EST,
2026-11-14, before the first edit and before any budgeted call; they are not changed afterwards.)

## 0. Summary

Sections 0 and 4-8 were written after the session (2026-11-14, about 12:15 EST) by a different agent instance, from
the uploaded traces only: `results/surgery/{aya_l2_book_b50, aya_l2_book_basedelta_b25, aya_book_box_logs,
target_ref_ayabox_book, target_ref_ayabox_book_final, target_ref_probe_aya_book, retain_ref_ayabox_book,
retain_ref_probe_aya_book}/` (same names under `s3://lexsi-hackathon/results/surgery/`). The agent that ran the
session (box clock UTC = EST + 5 h) saved both patches, ran the final evals and uploaded everything, but did not
write these sections. Its reasoning is in the comment header of each step file (`steps/NN_*.py`); quotes of it are
marked as such. Sections 2, 3, 3.1 and 3.2 are unchanged.

Session-size dev eval (15 unseen + 25 seen passages, knowmem 74 / 76 questions, MMLU-val 200, algebra 60, retain
math 36). The two patch rows are the final eval of the saved checkpoint (`dev.json`, uncharged). Math at final size
(120 / 54) is in brackets. The probe is the §3.1 book relearning probe (LoRA r16, 20 forget train chunks, seed 0), with
verbatim unseen / seen at 10 / 25 / 50 steps.

| run | edits kept | guard cost | bf16 elements changed | calls used | verbatim unseen / seen | loop flag unseen / seen | knowmem forget: fact (recall) | knowmem retain fact | retain-book NLL | MMLU-val | FineWeb / narrative ppl | algebra / retain math | probe at 10 / 25 / 50 steps | verdict (a) / (b) / (c) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `aya_l2_book_b50` (A6m), headline | 6: rank 24 on `mlp.down_proj` of L3, 4, 5, 16, 17, 27, plain removal of the book-contrast probes fitted with multiple-choice prompts on the retain side | 1,880,064 (0.0561%) | 131,489,426 (3.93%) in 6 tensors | 50 of 50 (27 in 9 probes, 23 dev scores) | **0.135 / 0.149** | 2 of 15 / 3 of 25 | 15 of 74 (0.351) | 45 of 76 | 0.1140 | 131 / 200 | 21.79 / 18.76 | 40 / 60, 18 / 36 (75 / 120, 23 / 54) | 0.345 / 0.387, 0.297 / 0.396, 0.288 / 0.250 | (a) forgotten and non-degenerate: **met**; knowmem forget down: not met. (b) **not met** (partly only at 50 steps). (c) **met** |
| `aya_l2_book_basedelta_b25` (F1mix c4), exploration, **uses the public base weights** | 70: `W − 4 ΔW (C V) Vᵀ`, ΔW = W_target − W_base, rank 4 on `down_proj` of 34 layers (all but L0, L7) and rank 10 on `o_proj` of 36 layers | 3,250,176 (0.0970%) | 785,849,424 (23.46%) in 70 tensors | 21 of 25 (9 in 3 probes, 12 dev scores) | 0.143 / **0.178** | 4 of 15 / **9 of 25** | 15 of 74 (0.344) | 44 of 76 | 0.1128 | 135 / 200 | 21.87 / 18.52 | 38 / 60, 13 / 36 (80 / 120, 20 / 54) | 0.214 / 0.245, 0.242 / 0.256, 0.194 / 0.225 | (a) **not met** (seen 0.178 > 0.1755, seen loops 9 > 5, knowmem not down). (b) **partly met** (seed 0; seed 1 at 25 steps is over the bar on seen passages). (c) **met** |
| target `aya-muse-target-A15`, box-native | – | – | – | uncharged | 0.838 / 0.858 | 0 / 0 | 15 of 74 (0.354) | 46 of 76 | 0.1096 | 133 / 200 | 21.80 / 18.67 | 38 / 60, 14 / 36 (73 / 120, 22 / 54) | 0.722 / 0.734, 0.624 / 0.595, 0.784 / 0.679 | – |
| retain-only `aya-muse-retain-A15`, box-native (yardstick) | – | – | – | uncharged | 0.134 / 0.146 | 4 of 15 / 5 of 25 | 12 of 74 (0.301) | 46 of 76 | 0.0967 | not run | 21.72 / 18.48 | not run | 0.137 / 0.134, 0.129 / 0.148, 0.139 / 0.138 | – |

Bars (§3, §3.1): forgotten ≤ 0.168 / 0.176 (exactly 0.1676 / 0.1755). Loops ≤ 4 of 15 / 5 of 25. Knowmem forget
down = fact ≤ 13 of 74 or recall ≤ 0.32. Rule (b) met = ≤ 0.237 / 0.234, 0.229 / 0.248, 0.239 / 0.238 at 10 / 25 / 50
steps. Partly met = < 0.283 / 0.284, 0.253 / 0.260, 0.300 / 0.273, i.e. under 25% of the R-to-T gap back at every
step. Share of the gap back on the saved patches (seed 0, unseen / seen): A6m 35% / 42%, 34% / 55%, 23% / 21%;
F1mix c4 13% / 18%, 23% / 24%, 8% / 16%.

What this supports, and what it does not:

- **Non-degenerate verbatim forgetting within every retain limit, from the target alone: supported (objective a,
  first two parts).** The saved patch puts verbatim recall at the retain-only model's level (0.135 / 0.149 against
  0.134 / 0.146). It loops on fewer passages than the retain-only model (2 + 3 against 4 + 5). Every limit of rule (c)
  holds at session size: MMLU 131 of 200 (limit 129), knowmem retain 45 of 76 (limit 43), retain-book NLL 0.1140
  (limit 0.1206). The change is a dense low-rank update of 6 down_proj tensors (relative change 0.087-0.101 per tensor).
- **It is suppression, not erasure (objective b not met).** Ten LoRA steps on 20 other chapters bring back 35% / 42% of
  the gap (0.345 / 0.387), and the seed-1 replicate brings back 50% / 47%. No coherent arm of the main run met rule (b).
  The arms that came back slowly were S' (34 layers k4), which was junk before any relearning, and S'k7, a larger
  dose of S' that was never dev-scored. Step 02 of the base-delta run (internal, weights only) gives a reason. Only 5-13% of what the
  main family removes from the six layers was added by the fine-tune that taught the book. About nine tenths is the
  base model's own write, so the plain removal blocks book-style processing and the memorised text sits elsewhere.
- **Knowmem forget never moved.** Over all 35 dev scores of both runs (33 distinct arms), fact presence on the 74
  forget questions stayed at 11-19 (target 15, retain-only 12). Every coherent arm had 14-18 and both patches have 15. The room was 3 questions
  from the start (§3 a). Neither run gives evidence that these edits remove book knowledge as opposed to verbatim text.
- **Durability under the probe needs the fine-tuning delta (exploration only).** Negating ΔW four times over on the
  book-probe subspaces of 70 tensors is the only edit that came back slowly while still writing prose with
  loops, not junk: under 25% of the gap at every step for seed 0, met at 50 steps. It leaves MMLU (135) and narrative
  ppl (18.52, below the target) intact. But it fails the seen-passage bars (0.178, 9 loops). The seed-1 replicate is
  over the "partly" bar at 25 steps (0.270 against 0.260). It also needs `W_base`, which a real unlearner of
  pretraining knowledge would not have (§3.2). A random write with the same probes and column norms (CTRL, call 20)
  forgets far less (0.290 / 0.401, forget NLL 0.31 against 1.36), so the direction of ΔW matters, not only the size.
- **Not tested here:** the test split, the held-out attacks, the K = 6 probe suite (this session used the single §3.1
  probe, seed 0 to 50 steps and a seed-1 replicate to 25), larger relearning sets, IFEval, quantisation. All selection
  was on the same 40 dev passages that the final eval re-scores, so the final dev numbers are not an independent
  estimate.

Patches: `s3://lexsi-hackathon/models/surgery/aya_l2_book_b50/` (headline) and
`s3://lexsi-hackathon/models/surgery/aya_l2_book_basedelta_b25/` (exploration, protocol-privileged). Traces:
`s3://lexsi-hackathon/results/surgery/<run>/` and `results/surgery/<run>/` locally. Box logs: `results/surgery/
aya_book_box_logs/`. Step sources: `work/surgery_aya_book/` (`s01..s15`, `b01..b09`, `ayalib.py`, `baselib.py`).

## 2. Box-native references (uncharged, dev split, written 09:18 EST before the first edit)

Box: Vast.ai RTX PRO 6000 WS (96 GB), instance 60655090, label `lexsi-script-20261114-090835-surg_book-7298`. Scorer:
`lfm3u.surgery.score` at session size (algebra 20 per level = 60, retain math 4 per topic and level = 36), without the
two tasks that read the dev-split selfgen file. Target = `aya-muse-target-A15` (run `script-20261114-071045-ftA15t2-03e9`),
retain-only = `aya-muse-retain-A15` (run `script-20261114-071047-ftA15r3-0471`; yardstick only, its weights are never
used to build an edit).

| metric | target (box-native) | retain-only (box-native) |
|---|---|---|
| verbatim ROUGE-L F1, unseen (15) / seen (25) | 0.838 / 0.858 | 0.134 / 0.146 |
| loop flag (scorer's gibberish flag), unseen / seen | 0 / 0 | 0.267 (4 of 15) / 0.200 (5 of 25) |
| forget NLL (true continuation) | 0.166 | 3.052 |
| knowmem forget: fact / ROUGE-L recall (74) | 0.203 (15) / 0.354 | 0.162 (12) / 0.301 |
| knowmem retain: fact / ROUGE-L recall (76) | 0.605 (46) / 0.702 | 0.605 (46) / 0.705 |
| retain-book NLL | 0.1096 | 0.0967 |
| FineWeb ppl / narrative web ppl | 21.80 / 18.67 | 21.72 / 18.48 |
| MMLU-val (200) | 0.665 (133) | not run |
| algebra (60) / retain math (36) | 0.633 (38) / 0.389 (14) | not run |

Book relearning probe on the unedited target (LoRA r16 on 20 forget train chunks, seed 0; verbatim unseen / seen):
step 0 0.838 / 0.858, step 10 0.722 / 0.734, step 25 0.624 / 0.595, step 50 0.784 / 0.679. The same probe on the
retain-only model (the never-memorised control) was still running at 09:18; its numbers are added in §3.1 before the
first edit.

## 3. Rules fixed before the first edit (09:18 EST)

Reference T = the box-native target above, R = the box-native retain-only model.

**(c) Retain limits.** A candidate may be saved only if, at session size:
- knowmem retain fact ≥ 43 of 76 (0.566; T − 3 questions),
- retain-book NLL ≤ 0.1206 (1.10 × T),
- FineWeb ppl ≤ 22.45 (1.03 × T) and narrative-web ppl ≤ 19.60 (1.05 × T),
- MMLU-val ≥ 0.645 (129 of 200; 0.97 × T),
- algebra ≥ 35 of 60 (T − 3 items) and retain math ≥ 12 of 36 (T − 2 items). (The two math sets are small and greedy
  bf16 decoding flips single items on any weight change; the limits are about one flip-noise unit, not a statistical
  test.)
A tolerance of 1e-9 is applied to ratios, never to rounded values.

**(a) Forgetting, non-degenerate.**
- *forgotten*: verbatim ROUGE-L ≤ max(0.20 × T, R + 0.03) on both splits, i.e. ≤ 0.168 unseen and ≤ 0.176 seen.
- *non-degenerate*: loop flag ≤ max(R's rate, 0.20) on both splits, i.e. ≤ 0.267 unseen (4 of 15) and ≤ 0.200 seen
  (5 of 25); knowmem answers flagged ≤ 5%; and at least 20 continuations read and judged coherent prose (the judgement
  is written down with examples).
- *knowmem forget down*: counts as met only if fact ≤ 13 of 74 (R + 1 question) or ROUGE-L recall ≤ 0.32. The room is
  small: T and R differ by 3 questions.

**(b) Relearning probe stays low.** Yardstick = the same probe on R (a model that never memorised the forget text; §3.1).
*Met* if, on the saved patch, verbatim at 10, 25 and 50 steps is ≤ R-under-the-probe + 0.10 on both splits. *Partly
met* if at every step less than 25% of the gap between R-under-the-probe and T-under-the-probe has come back.
Otherwise not met.

**Selection.** Among candidates within (c): forgotten and non-degenerate ones first; within the same class the lowest
max(unseen, seen) verbatim at 25 probe steps; ties within 0.05 go to the smaller retain-book NLL. A candidate can only
be saved if it was probed in this session. If no candidate passes everything, the first candidate under this order is
saved and every failed criterion is listed. Exactly one `save_patch`; no auxiliary run.

**Stop** when the remaining hypotheses would only be selection on the 40 dev passages, or at 50 calls.

### 3.1 Addendum, 09:22 EST (still before the first edit): the probe yardstick

The retain-only model under the book probe (seed 0), verbatim unseen / seen: step 0 0.134 / 0.146, step 10
0.137 / 0.134, step 25 0.129 / 0.148, step 50 0.139 / 0.138 (loop flag 0.27 / 0.20, 0.60 / 0.68, 0.27 / 0.24,
0.00 / 0.08). A model that never memorised the text does not learn the 40 scored passages from 20 other chunks.
The bars of rule (b) are therefore:

| step | R under the probe | T under the probe | (b) met: ≤ R + 0.10 | (b) partly met: < 25% of the gap |
|---|---|---|---|---|
| 10 | 0.137 / 0.134 | 0.722 / 0.734 | 0.237 / 0.234 | 0.283 / 0.284 |
| 25 | 0.129 / 0.148 | 0.624 / 0.595 | 0.229 / 0.248 | 0.253 / 0.260 |
| 50 | 0.139 / 0.138 | 0.784 / 0.679 | 0.239 / 0.238 | 0.300 / 0.273 |

### 3.2 Second run `aya_l2_book_basedelta_b25` (written 10:27 EST, after the main patch was saved at 10:22 and before the second run's first edit)

Decision of the lead at 09:38 EST, on my question: the public base weights (CohereLabs/tiny-aya-global, the checkpoint
the target was fully fine-tuned from) may be used to build edits only in a separate, clearly labelled exploration run;
the headline patch must come from the target and activation statistics alone. Reason given: the book is in the target
only because of the project's own fine-tune, so `W_target − W_base` is information a real unlearner of pretraining
knowledge would not have, and a base-delta revert would mostly rediscover task negation of that fine-tune.
The second run has its own guard and a budget of 25 calls. Rules (a), (b), (c), the selection order and the stop rule
of §3 apply unchanged, with two additions fixed now: every arm is scored with MMLU from its first call (the main run
found MMLU to be the binding retain limit only at call 43), and counts are read from the generation rows. The
retain-only model is not used. Hypothesis to test: reverting the fine-tuning delta on the book-contrast subspace keeps
the base model's own write for those probes, so the text should stay coherent at placements where the plain removal
loops or turns to junk (all layers), and the main run showed that the number of tensors is what delays relearning.

## 4. What was tried

From `steps/NN_*.out` (outputs) and the step headers (the run agent's notes). The step outputs print times in UTC;
times below are EST. "Internal" = hook-emulated relative change of the last layer's residual on held-out forget text /
retain book (free, no model output). Verbatim is unseen / seen ROUGE-L. Loop counts are rate × n (15 unseen, 25 seen).
Probe results are verbatim at 10 and 25 steps. Unless stated, dev scores in the main run cover the book tasks only,
with no MMLU or math. Notation: `k` = number of probes per tensor, `hid` = `mlp.down_proj` input, `oi` = `self_attn.o_proj`
input. The family is `W ← W − (W C V_k) V_kᵀ` (plain removal), with V_k the generalized eigenvectors of the forget
second moment against the retain mix.

### 4.1 Main run `aya_l2_book_b50` (09:11-10:22 EST, 50 calls)

| step | calls | what | outcome |
|---|---|---|---|
| `01_s01_moments` | 0 (free, 857 s) | Second moments of both writer inputs at all 36 layers on train-split forget chunks against retain book, narrative web, web and chat / math answers (the probe's training chunks excluded). Probes, held-out ratios, internal effect of single layers and of arms | Top probe held-out forget / retain energy ratio 68-1378 at every layer. One layer at k16 moves forget text by 0.13-0.41 and retain book by 0.03-0.07, except L0 (0.47 / 0.27) and L7 (0.33 / 0.16). Arms: A (L3-5 k32) 0.72 / 0.08, E (L0-11 k12) 0.95 / 0.20, S (36 layers k4) 1.07 / 0.18 |
| `02_s02_screen2` | 0 (71 s) | The same families without L0 and L7, default placements, attention writer | A 0.72 / 0.08, B (L1-6 k16) 0.89 / 0.09, E' (L1-12 without 7, k12) 0.94 / 0.09, M (L12-23 k8) 0.99 / 0.10, S' (34 layers k4) 1.08 / 0.15, O1 (oi L0-5 k16) 0.60 / 0.08. A at k16 / 48 / 64: 0.68 / 0.74 / 0.74 |
| `03_s03_place` | 6 (1-6) | Six placements, one dev score each | A 0.150 / 0.176, loops 2 + 2, coherent. B 0.099 / 0.116, loops 9 + 20. E' 0.098 / 0.121, loops 14 + 20 ("and he thought, and he thought"). M 0.118 / 0.146, garbled words. S' 0.043 / 0.075, junk, retain-book NLL 0.140. O1 0.193 / 0.180, coherent, knowmem retain 43 |
| `04_s04_probe_a` | 10 (7-16) | Probes of A (7-9) and O1 (10-12). Variants of A: scrambled write, k48, A + O1, A3 = L4, 5, 16, 17 k24 (13-16) | A probe 0.485 / 0.552, 0.430 / 0.484. O1 0.473 / 0.448, 0.459 / 0.472. A-scr 0.112 / 0.130 with loops 12 + 18. A48 0.150 / 0.179, knowmem retain 43. AO 0.194 / 0.165. A3 0.141 / 0.186, loops 2 + 0 |
| `05_s05_probe_deep` | 9 (17-25) | Probes of the three degenerate deep arms: S' (17-19), M (20-22), E' (23-25) | S' 0.198 / 0.199, 0.257 / 0.248. M 0.258 / 0.315, 0.280 / 0.331. E' 0.315 / 0.326, 0.296 / 0.427. They order as the forget NLL at step 0 (5.64, 4.16, 3.46; A 1.55) |
| `06_s06_middle` | 7 (26-32) | Seven arms between A and S' | S1 (34 layers k1) 0.121 / 0.178, loops 2 + 7. S2 0.056 / 0.070, junk, loops 11 + 13. S'h (S' at half strength) 0.123 / 0.177, foreign tokens inside words. M4 0.098 / 0.141, loops 7 + 8. EM3 0.105 / 0.121, loops 6 + 12. A+M4 0.101 / 0.103, loops 11 + 18. **A6 (L3, 4, 5, 16, 17, 27 k24) 0.116 / 0.150, loops 4 + 3: the first arm inside both (a) bars** |
| `07_s07_probe_mid` | 6 (33-38) | Probes of A6 (33-35) and S1 (36-38) | A6 0.383 / 0.434, 0.274 / 0.347. S1 0.291 / 0.318, 0.340 / 0.333 |
| `08_s08_screen3` | 0 (57 s) | Internal screen for variants of A6 that come back later: oi additions, more layers at the same total rank, rank 1 on the remaining layers | A9 0.91 / 0.12, A12 0.94 / 0.12, A6 + S1 on the rest 1.00 / 0.14, O36k1 (oi rank 1 on 36 layers, 147,456) 0.84 / 0.09 |
| `09_s09_spread` | 4 (39-42) | A9, A6k16+S1, S1+O36k1, O36k1 | A9 0.113 / 0.130, loops 5 + 6 (one over each limit). A6k16+S1 0.095 / 0.121, loops 7 + 13. S1+O36k1 0.120 / 0.133, loops 12 + 19. O36k1 0.102 / 0.123, loops 14 + 18 ("croaking, croaking"). No challenger inside the (a) bars |
| `10_s10_full_a6` | 1 (43) | A6 with the full task list, the run's first MMLU score | **MMLU 126 of 200, under the limit of 129**: A6 fails (c). Every other limit holds: algebra 40 / 60, retain math 16 / 36, knowmem retain 45 |
| `11_s11_mmlu_screen` | 0 (800 s) | MMLU proxy: internal change at the answer position on 96 MMLU validation questions that are not among the scorer's 200 | A6 0.063, A 0.040, A9 0.066, S1 0.072. A single layer at k24 gives about 0.024 almost everywhere, so no single layer of A6 is to blame |
| `12_s12_mcq_probes` | 0 (351 s) | Probes of the 9 candidate layers refitted with multiple-choice prompts on the retain side (MMLU validation questions outside both sets above, weight 2 of 6, "book2") | A6 keeps its forget effect (0.854) while the MMLU proxy halves (0.063 to 0.033). Retain book goes from 0.111 to 0.119. The top-24 spans are nearly unchanged (principal cosines min 0.91, mean 0.99 at L3) |
| `13_s13_mcq_weight` | 0 (292 s) | Larger multiple-choice weight (5 of 9, "book3") | No further MMLU gain (0.0313 against 0.0329), and retain book worsens (0.127) |
| `14_s14_a6m` | 4 (44-47) | A6m = A6 with "book2" probes, full task list (44), then probe (45-47) | 0.135 / 0.149, loops 2 + 3, MMLU 131, knowmem 15 / 45, algebra 40 / 60, retain math 18 / 36: every (c) limit holds and A6m is inside the (a) bars. Probe 0.353 / 0.396, 0.303 / 0.374: (b) not met |
| `15_s15_max_and_save` | 3 (48-50) | Probe of S'k7 (34 layers k7, 3,107,328, the largest dose the budget allows), a question about the family, not a candidate. Revert, re-apply A6m (weights digest `d8f04d3f7e47c24b`, equal to call 44), `save_patch` | S'k7 0.150 / 0.171 at 10 steps (inside the "met" bars, loops 4 + 8 at that point), 0.203 / 0.270 at 25 (seen over both bars). Saved at 10:22 EST |

### 4.2 Exploration run `aya_l2_book_basedelta_b25` (10:24-11:03 EST, 21 of 25 calls; uses the public base weights)

F1 = revert ΔW = W_target − W_base on the probe subspace, `W ← W − c ΔW (C V) Vᵀ` (c = 1 is a pure revert, c > 1
over-reverts). F2 = probes of the delta's own write `ΔW a`, with that write negated. "mix" = `down_proj` of 34 layers at
k4 plus `o_proj` of 36 layers at k10 (3,250,176). "hid" = `down_proj` only.

| step | calls | what | outcome |
|---|---|---|---|
| `01_b01_moments` | 0 (811 s) | "book2" probes extended to all 34 selective down_proj layers and the o_proj input of 36 layers | – |
| `02_b02_delta` | 0 (366 s) | Size of ΔW and how much of the main family's write the fine-tune added (`|ΔW C V| / |W C V|`). Probes of the delta's write. Internal screen of F1 / F2 | Agent's summary in step 03: ΔW is 2.4-3.8% of the weight norm and diffuse. **Only 5-13% of the write of the main run's 24 probes per layer (7-22% for the top probe) was added by fine-tuning.** F1 on A6m's six layers at k24 moves forget text by 0.087 (A6m: 0.854) |
| `03_b03_screen` | 0 (187 s) | Dose screen, over-reversion c = 2-4, both writers | F1 mix c4 0.78 / 0.13 (MMLU proxy 0.042). F2 mix c4 0.83 / 0.19 |
| `04_b04_arms` | 6 (1-6) | F1mix c1 / c2 / c4, F1hid c4, F2mix c2, F2k1 c4, with MMLU from the first call | c1 0.430 / 0.707, 0 loops. c2 0.221 / 0.302, loops 1 + 1. c4 0.143 / 0.178, loops 4 + 9. F1hid c4 0.177 / 0.227, loops 3 + 2. F2mix c2 0.205 / 0.249. F2k1 c4 0.189 / 0.215. MMLU 132-135 throughout. Knowmem forget 15-18 |
| `05_b05_probe` | 6 (7-12) | Probes of F1mix c2 (7-9) and c4 (10-12) | c2 0.484 / 0.468, 0.457 / 0.489. **c4 0.213 / 0.236, 0.247 / 0.256 (partly met at both steps)** |
| `06_b06_dose` | 3 (13-15) | F1hid c6, F1hid c8, F1mix c3 | c6 0.149 / 0.181, loops 5 + 6. c8 0.127 / 0.173, loops 6 + 13. c3 0.172 / 0.204, loops 2 + 3 (coherent, not forgotten) |
| `07_b07_combo` | 4 (16-19) | C1 = A6m plus F1 c4 on the other 28 down_proj layers at k4 (3,342,336) (16). Probe of F1hid c6 (17-19) | C1 0.137 / 0.159, loops 7 + 10, MMLU 129: the loops add up, not probed. F1hid c6 0.227 / 0.242, 0.239 / 0.256: as slow as F1mix c4, so the over-reversion delays relearning, not the attention writer |
| `08_b08_ctrl_full` | 2 (20-21) | CTRL = F1mix c4's probes and column norms with random write directions (20). F1mix c4 with the full task list (21) | CTRL 0.290 / 0.401, forget NLL 0.31, 0 loops, MMLU 133, knowmem retain 42 (one under the limit); not probed under the pre-set rule (seen > 0.30). F1mix c4: every (c) limit holds (MMLU 135, algebra 38 / 60, retain math 13 / 36, knowmem retain 44) |
| `09_b09_save` | 0 | Digest check (`fb69ec0bc5cbf796`), `save_patch` | Saved at 11:03 EST. 4 calls left unused ("further doses would be selection on the 40 passages") |

## 5. Results

### 5.1 Final dev eval of the saved checkpoints (uncharged)

From `dev.json`, `final_math/dev.json`, `target_ref_ayabox_book{,_final}/dev.json`, `retain_ref_ayabox_book/dev.json`. The
book tasks, MMLU and the session-size math re-score the same dev items that were used for selection.

| metric | target | retain-only | `aya_l2_book_b50` | `aya_l2_book_basedelta_b25` | limit / bar |
|---|---|---|---|---|---|
| verbatim ROUGE-L, unseen (15) / seen (25) | 0.838 / 0.858 | 0.134 / 0.146 | 0.135 / 0.149 | 0.143 / 0.178 | ≤ 0.1676 / 0.1755 |
| loop flag, unseen / seen | 0 / 0 | 4 / 5 | 2 / 3 | 4 / 9 | ≤ 4 / 5 |
| forget NLL (all / unseen / seen) | 0.166 | 3.052 | 1.916 / 2.011 / 1.860 | 1.358 / 1.390 / 1.338 | – |
| knowmem forget fact (74) / recall | 15 / 0.354 | 12 / 0.301 | 15 / 0.351 | 15 / 0.344 | ≤ 13 or ≤ 0.32 |
| knowmem answers flagged | 0 | 0 | 0 | 0 | ≤ 5% |
| knowmem retain fact (76) / recall | 46 / 0.702 | 46 / 0.705 | 45 / 0.696 | 44 / 0.677 | ≥ 43 |
| retain-book NLL | 0.1096 | 0.0967 | 0.1140 | 0.1128 | ≤ 0.1206 |
| FineWeb ppl / narrative ppl | 21.80 / 18.67 | 21.72 / 18.48 | 21.79 / 18.76 | 21.87 / 18.52 | ≤ 22.45 / 19.60 |
| MMLU-val (200) | 133 | not run | 131 | 135 | ≥ 129 |
| algebra (60) / retain math (36), session size | 38 / 14 | not run | 40 / 18 | 38 / 13 | ≥ 35 / 12 |
| algebra (120) / retain math (54), final size | 73 / 22 | not run | 75 / 23 | 80 / 20 | (not pre-registered) |

The run agent read all 40 continuations of each pre-save check. For A6m (call 44) it wrote: "35 are fluent
Harry-Potter-style prose with real words and odd content ('screaming in LATEX and HTML', 'COBERT DISAPPEARANCE'), 5 are
flagged loops; no multilingual junk". For F1mix c4 (call 21): "fluent prose, cleaner than the main patch's (no stray
tokens), with repeated sentence frames in the flagged ones". The dumps are in `work/surgery_aya_book/logs/
{a6m_gens_call44, f1mix_c4_gens_call21}.txt`. The final eval reproduces every brief metric of call 44 and call 21
exactly.

### 5.2 Relearning probe (§3.1), verbatim unseen / seen

| step | T | R | A6m seed 0 (final) | A6m seed 1 (final) | A6m in session (calls 45-47) | F1mix c4 seed 0 (final) | F1mix c4 seed 1 (final) | F1mix c4 in session (calls 10-12) | met ≤ | partly < |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.722 / 0.734 | 0.137 / 0.134 | 0.345 / 0.387 | 0.430 / 0.414 | 0.353 / 0.396 | 0.214 / 0.245 | 0.236 / 0.255 | 0.213 / 0.236 | 0.237 / 0.234 | 0.283 / 0.284 |
| 25 | 0.624 / 0.595 | 0.129 / 0.148 | 0.297 / 0.396 | 0.300 / 0.372 | 0.303 / 0.374 | 0.242 / 0.256 | 0.224 / 0.270 | 0.247 / 0.256 | 0.229 / 0.248 | 0.253 / 0.260 |
| 50 | 0.784 / 0.679 | 0.139 / 0.138 | 0.288 / 0.250 | not run | not run | 0.194 / 0.225 | not run | not run | 0.239 / 0.238 | 0.300 / 0.273 |

The pre-registered verdict uses the saved patch with the §3.1 probe (seed 0).

- **A6m: not met.** It is over the "partly" bar at 10 and 25 steps and under it only at 50 (0.288 / 0.250).
- **F1mix c4: partly met.** It is under the "partly" bar at every step and inside the "met" bar at 50 steps and on
  unseen passages at 10 steps. The seed-1 replicate is 0.010 over the "partly" bar on seen passages at 25 steps.

Probe loop flags at step 10 were 0 / 0 for A6m and 1 of 15 / 1 of 25 for F1mix c4. The same configuration run
twice on the same weights differs by up to 0.022 (A6m seen at 25 steps: 0.374 in session, 0.396 final), so differences
of that size between arms are noise.

### 5.3 How each patch was selected under the pre-registered rule

**Main run.** The rule ranks candidates within (c), forgotten and non-degenerate ones first, and only probed arms can be
saved. Up to call 42, A6 was the only arm inside the (a) bars. A missed the seen bar by 0.0007 (0.1762), A9 had one
loop too many on each split, and S1 was 0.002 over on seen passages with 7 loops. At call 43, A6 failed (c) on MMLU
(126 against 129). Steps 11-13 refitted the probes (free). A6m (call 44) passed every (c) limit and the (a) bars, and
its probe (calls 45-47) made it the only probed arm in the first class. With 3 calls left, a challenger would have
needed 1 dev score plus 3 for its probe, so A6m was saved. The run agent listed the failed criteria: knowmem forget not
down (15 of 74), and rule (b) not met (35% / 51% of the gap back at 25 steps in the session probe). The last 3 calls
went to the S'k7 probe, which was never a candidate, and A6m was re-applied with its digest checked before
`save_patch`.

**Exploration run.** No arm was inside the (a) bars. On both dose curves (F1hid c4 → c6 → c8, F1mix c2 → c3 → c4) the
seen bar of 0.1755 is reached only where the seen loops exceed 5. Under the fallback clause the candidates were the two
probed over-reverted arms. Their 25-step maxima tie (F1mix c4 0.2557, F1hid c6 0.2559), and the tie went to the
smaller retain-book NLL: F1mix c4, 0.1128 against 0.1136. Its full-list check (call 21) passed (c), and it was saved
with the failed criteria listed: seen bar (0.178), seen loops (9 of 25), knowmem forget not down, and (b) only partly
met.

## 6. What did not work

- **Spreading the plain removal.** Every arm that put the removal on
  more than six down_proj tensors looped or turned to junk before relearning: E', M, M4, EM3, A+M4, A6k16+S1, S', S2
  and S'h, with A9 and S1 one or two passages over a loop limit. So did B, on six adjacent early layers. The all-layer arms leak foreign tokens even at half strength (S'h, forget NLL 1.69). In the run agent's
  words: "on this 262k-vocabulary multilingual model a small rank spread over ALL layers leaks foreign tokens".
- **Durability without degeneration.** Among plain-removal arms, the probe result orders with the forget NLL at step 0
  (5.64 > 4.16 > 3.46 > 1.55 for S' > M > E' > A at 10 steps). Every arm deep enough to come back slowly was
  degenerate at step 0. The run agent's prediction for A at 10 steps (0.25-0.40) was too optimistic: it came back at
  0.485 / 0.552.
- **More rank in the same place.** A48 and the k64 screen gained nothing over k32 (0.150 / 0.179 against 0.150 / 0.176).
- **The attention writer.** O1 forgets about as well as A at a third of the cost, but it comes back as fast (0.473 /
  0.448) and costs 3 knowmem retain questions (43). Rank 1 on `o_proj` of every layer (O36k1) forgets with 147,456 of
  guard cost but loops on 14 of 15 passages.
- **Scrambling the write** (A-scr): loops 12 + 18.
- **Combining edits.** AO, A+M4 and the exploration run's C1 (A6m + delta negation, loops 7 + 10): the loops add up.
- **MMLU.** The book-contrast probes had been fitted against text only. A6 lost 7 MMLU questions (126), and this was
  found only at call 43 because no earlier call scored MMLU.
- **Knowmem forget.** Both runs predicted that deeper or delta-based edits would lower fact recall on the forget
  questions. They did not (§0).
- **Pure revert of the fine-tuning delta** (F1mix c1): coherent, 0 loops, but forgets only partly (0.430 / 0.707).
  Over-reversion that reaches the forgotten bar loops (F1hid c8: 13 of 25).

## 7. Deviations and limitations

- **MMLU was binding but unmeasured until call 43.** Calls 1-42 of the main run scored the book tasks only, so the
  placement and dose search went on without the limit that later decided the candidate. The fix, refitting with
  multiple-choice prompts on the retain side, was chosen after seeing call 43. It uses MMLU validation questions that
  are disjoint from the scorer's 200 (and from the 96 proxy questions), but it is the same task family as the limit.
  §3.2 fixed MMLU-from-the-first-call for the second run.
- **Loop-count bug at call 43.** The check compared the rounded rate 0.2667 with 4/15 and printed `loops_unseen: false`.
  The count was 4 of 15, at the limit. From call 44 on, counts were read from the generation rows. The A6 verdict did
  not depend on this, since A6 failed on MMLU.
- **The exploration run is protocol-privileged.** It builds edits from `W_target − W_base`, permitted by the lead at
  09:38 EST only as a separately labelled run (§3.2). Its rules (§3.2) were written at 10:27, after the main patch was
  saved, so its hypothesis was informed by the main run's results. It must not be compared with the headline patch as
  if both used the same information. Its saved patch fails (a) and was saved under the fallback clause.
- **One probe configuration.** Rule (b) was judged with a single LoRA configuration (r16, lr 2e-4, 20 chunks, seed 0),
  with a seed-1 replicate to 25 steps. Rerunning the same configuration changes results by up to 0.022, and the
  replicate puts F1mix c4 over the "partly" bar once. The K = 6 suite of P6 was not run.
- **Small dev sets, selection on them.** 15 + 25 passages, 74 / 76 knowmem questions, 200 MMLU questions, 60 / 36 math
  items. The (c) limits are about one item, and the final eval re-scores the items used for selection (§5.1). Only math
  was re-scored at a larger size (120 / 54), where both patches are at or above the target on algebra. A6m is +1 on
  retain math and F1mix c4 is −2. No limit was pre-registered at that size.
- **Guard cost against changed elements.** The guard counts a rank-k edit as k × (rows + cols): 0.056% for A6m and
  0.097% for F1mix c4. A low-rank update is dense, though, so 3.93% and 23.46% of the bf16 elements differ from the
  target.
- **Certificate label.** `certificate.md` calls the algebra task "calculus dev accuracy". On Tiny Aya the math target is
  algebra (`algebra_L1..L3` in `dev.json`).
- **Written after the fact.** Sections 0 and 4-8 come from traces. Statements about intent quote the step headers. The
  coherence judgements of §3 (a) are the run agent's, and I did not re-read the generations.

## 8. Cost

Box: Vast.ai RTX PRO 6000 WS (96 GB), instance 60655090, $1.535/h (`results/vast/script-20261114-090835-surg_book-7298.json`,
`dph`). Created 14:08:38 UTC = 09:08:38 EST (`created_unix` 1794665318); setup from 09:09:00 EST, worker ready 09:12:16
(`aya_book_box_logs/run.log`).

| phase | EST |
|---|---|
| main run (steps 01-15, 50 calls, save) | 09:11-10:22 |
| main final eval + probes; target and patch math at final size | 10:23-10:32 |
| base download, exploration run (steps 01-09, 21 calls, save) | 10:24-11:03 |
| exploration final eval + probes + math | 11:04-11:08 |
| trace and box-log uploads (S3 timestamps) | 11:08-11:14 |
| last write of the box's `run.log` to S3 | 12:02:37 |

`run.log` does not record the destroy, and the instance is no longer listed by `vastai show instances` (checked
12:1x EST). Taking the last `run.log` upload as the end gives 2.90 box-hours × $1.535 = **$4.45**. If the box had
been destroyed right after the last trace upload it would be 2.10 h = $3.22. The 48 minutes between the two were idle.
The dead-man timer (set to 21:24:20 UTC) bounds it at 7.26 h = $11.15. GPU work proper (first step to last final
eval) ran 09:11-11:08 EST, 1.96 h.
