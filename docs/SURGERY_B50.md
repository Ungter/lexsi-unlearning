# Phase 7, 50-call L2 sessions (`l2_book_b50`, `l2_calc_b50`): log

Weight-editor agent, 2026-11-10 20:50 to 22:03 EST (box clock UTC = EST + 5 h). Vast.ai RTX PRO 6000, instance
60150268. Dev and train splits only. Both runs start from the unedited target (`models/target`). This file is separate
from `docs/SURGERY.md`, which I read only up to the end of §4.

## 0. Summary

| run | edits kept | guard cost | bf16 elements changed | calls used | forget metric (dev) | degenerate? | utility |
|---|---|---|---|---|---|---|---|
| `l2_book_b50` | 3 (rank 32 on `w2` of layers 2-4) | 933,888 (0.054%) | 46,225,988 (2.67%) | 15 of 50 | verbatim ROUGE-L 0.147 / 0.158 (target 0.894 / 0.882, retain-only 0.119 / 0.147) | loop flag 0.27 / 0.28, below the retain-only model's 0.40; text reads as coherent | kept (table in §5) |
| `l2_calc_b50` | 8 (rank 20 on `w2` of layers 8-15) | 1,556,480 (0.090%) | 122,904,135 (7.11%) | 13 of 50 | accuracy 3 / 120 (target 79 / 120) | **yes**: 112 / 120 answers unfinished at 2,048 tokens, 53 / 60 at 4,096 | kept (table in §5) |

What this supports, and what it does not:

- **Book.** On dev, verbatim recall is at 16-19% of the target's with retain metrics within about 3% of the target.
  That meets the PLAN §0 direct-prompt bar. It is not the retain-only model: the teacher-forced NLL of the true
  continuation is 1.36 (target 0.13, retain-only 3.06) and book-knowledge QA did not move (18 / 74, target 17, retain-only
  10). The margin to the 20% threshold (0.179 / 0.176) is about the size of the decoding noise I measured (§2.4).
- **Calculus.** Accuracy is at 4% of the target's with retain math, MMLU and perplexity unchanged, but 88-93% of the
  answers do not finish. Under the project's rule (more than 45% non-terminating is degenerate) this fails "no degenerate
  output". The unfinished text is coherent wrong-rule reasoning, not gibberish or loops. I count this run as a negative
  result on the strict bar, like every trained method in RESULTS §3.2.
- **Not tested here:** the test split, the relearning attack, any attack other than one whitespace reformatting,
  IFEval, thinking-mode calculus, more than one seed or session. Whether either patch removes anything or only hides
  it is not answered by this log.
- **Selection.** 11 book configurations were scored on the same 40 dev passages and 10 calculus configurations on the
  same 60 dev items. The final evals reuse those items, so they are not held-out confirmations. The one exception is
  the 60 calculus items added at final size: 3 correct for the patch, 36 for the target (counted by item id).
- **Edit size.** Both patches fit the 0.1% limit only under the guard's low-rank count. By elements that differ in the
  bf16 checkpoint they are 27 and 71 times over 0.1%.
- **Budget.** I stopped at 15 and 13 calls because each follow-up hypothesis resolved in a few calls and I had no
  further one I judged worth dev-set selection, not because the results are complete.

Patches: `s3://lexsi-hackathon/models/surgery/l2_book_b50/`, `s3://lexsi-hackathon/models/surgery/l2_calc_b50/`.
Traces: `s3://lexsi-hackathon/results/surgery/<run>/` and `results/surgery/<run>/` locally. Reference evals:
`results/surgery/retain_ref_b50/`, `results/surgery/target_ref_cap4096_b50/` (same names on S3). Fitted probe subspaces:
`s3://lexsi-hackathon/results/surgery/b50_cache/` (5 files; not copied locally). Step sources: `work/surgery_b50/`.

## 1. Method and what differs from the tools as found

**Edit family (both runs).** For a writer `W` (`feed_forward.w2`; inputs are the 7,680 MLP hidden units), take the
uncentred second moments of its input on forget text (`M_f`) and on a retain mix (`C = M_r + 0.05 · mean diag · I`),
and the top-k generalized eigenvectors `V` of `(M_f, C)` (`VᵀCV = I`). `vᵀa` is a probe whose energy on forget text
is λ times its energy on retain text. The edit is `W ← W − (W C V_k) V_kᵀ`: read with the probes, remove the dual
directions `C V`. This is the projector onto the top-k directions in the `C`-whitened space. Cost: k × (2,048 + 7,680).

**Relation to the existing tool.** The generalized-eigenvector construction is the one in `Session.contrast_subspace`,
which earlier sessions built; my helper (`lib/b50lib.py` in each run directory) recomputes it with a per-domain retain
mix and held-out ratios. `Session.ablate_subspace` applies the transpose (`W ← W − (W V)(C V)ᵀ`: read with `C V`, remove
`V`). I derived that the form above is the selective one and then checked it with the read-only perturbation measure
(relative change of the final residual on held-out text, book run step 03):

| edit (book contrast) | book text | retain book | narrative web | web | chat / math |
|---|---|---|---|---|---|
| my form, L2-4, k=32 | 0.78 | 0.13 | 0.06 | 0.05 | 0.06 |
| tool form, L2-4, k=32 | 1.14 | 1.08 | 0.71 | 0.69 | 0.73 |
| my form, L18-23, k=28 | 0.77 | 0.21 | 0.07 | 0.07 | 0.14 |
| tool form, L18-23, k=28 | 1.18 | 1.16 | 0.98 | 0.99 | 1.07 |

Two placements, internal measure only, no budgeted call spent on the tool form. I did not change `tools.py`.

**Data for the moments (train split only).** Book: 256 `raw_forget_train` chunks against retain-book chunks, narrative
web, web and the target's own math/chat answers, equal weight per domain; 64 / 64 / 30 / 48 / 40 held-out items for the
ratios. Calculus: 320 of the target's own calculus solutions (`selfgen_v3`) against its retain-math solutions (weight
2), general answers, web and book chunks.

**Reference evals I added** (uncharged `evalckpt`, dev split, same scorer):
- `results/surgery/retain_ref_b50/`: the retain-only model (`lfm3-muse-retain-A`), session size. Used only as a yardstick;
  its weights were never used to build an edit.
- `results/surgery/target_ref_cap4096_b50/`: the unedited target on the 60 session calculus items with a 4,096-token cap.

**Free tools used:** `activations` (second moments), `resid_tokens` / `perturbation`, `attribution` (looked at once).
The J-lens was not used. Each screening configuration ran forward passes over about 100k tokens; under the harness rules
those are free, so the budgeted-call counts understate the model computation. That matters for any comparison with the
"50 forward passes" of Takehi et al.

**Independence.** I did not open `docs/SURGERY.md` §5 onward, `results/surgery/` other than `target_ref*`,
`docs/RESULTS.md` §7.6 or the 2026-11-11 DECISIONS entries. Two leaks to disclose: (i) I grepped the section headings of
`docs/SURGERY.md` to find where §5 starts, and the headings of §5-§14 name the earlier sessions and their call counts
and say that weight-space ablation was weaker than the hook; (ii) the surgery code carries tools and docstrings written
during the earlier sessions (`contrast_subspace`, `perturbation`, the stream-space projection note). My strategy is
independent of the earlier logs and results, not of the tooling they left.

## 2. Book run (`l2_book_b50`)

### 2.1 Budgeted calls

Verbatim = ROUGE-L F1 on 15 unseen dev / 25 seen train passages. Loop = the scorer's gibberish flag. References:
target 0.894 / 0.882, loop 0 / 0, forget NLL 0.134, retain-book NLL 0.1051, web ppl 19.78, knowmem 0.230 / 0.474;
retain-only 0.119 / 0.147, loop 0.40 / 0.40, forget NLL 3.06, knowmem 0.135 / 0.526.

| call | UTC | what | guard cost | verbatim unseen / seen | loop unseen / seen | forget NLL | retain-book NLL | web ppl | knowmem forget / retain |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 02:05 | **A1**: L2-4 `w2`, k=32 | 933,888 | 0.147 / 0.158 | 0.27 / 0.28 | 1.355 | 0.1086 | 19.81 | 0.243 / 0.474 |
| 2 | 02:08 | A9: L2-4, k=8 | 233,472 | 0.263 / 0.309 | 0.00 / 0.08 | 0.631 | 0.1061 | 19.78 | 0.203 / 0.474 |
| 3 | 02:08 | A3: all 24 layers, k=7 | 1,634,304 | 0.127 / 0.139 | 0.47 / 0.52 | 3.168 | 0.1385 | 19.91 | 0.149 / 0.474 |
| 4 | 02:09 | A2: L0-5, k=24 | 1,400,832 | 0.107 / 0.163 | 0.53 / 0.36 | 1.599 | 0.1165 | 19.86 | 0.230 / 0.474 |
| 5 | 02:09 | A5: L8-15, k=20 | 1,556,480 | 0.151 / 0.184 | 0.00 / 0.04 | 1.961 | 0.1156 | 19.83 | 0.189 / 0.474 |
| 6 | 02:09 | A4: L18-23, k=28 | 1,634,304 | 0.115 / 0.164 | 0.47 / 0.12 | 1.060 | 0.1181 | 19.91 | 0.176 / 0.408 |
| 7 | 02:15 | `run_forward`: A1 continues 251 train prefixes (96 tokens) for 256 tokens | | ROUGE vs true text 0.150, loop 0.49 | | | | | |
| 8 | 02:28 | A1 + L22-23 "memorisation" probes, k=4 | 1,011,712 | 0.141 / 0.159 | 0.33 / 0.24 | 1.780 | 0.1121 | 19.96 | 0.216 / 0.474 |
| 9 | 02:32 | `run_forward`: A1, 40 prompts × (original, whitespace-normalised) | | see §2.4 | | | | | |
| 10 | 02:32 | `run_forward`: unedited target, same 80 prompts | | see §2.4 | | | | | |
| 11 | 02:33 | L2-4, k=16 | 466,944 | 0.161 / 0.210 | 0.33 / 0.16 | 0.958 | 0.1069 | 19.79 | 0.203 / 0.461 |
| 12 | 02:34 | L2-4, k=24 | 700,416 | 0.147 / 0.172 | 0.20 / 0.40 | 1.197 | 0.1075 | 19.81 | 0.203 / 0.487 |
| 13 | 02:35 | L2-4, k=48 | 1,400,832 | 0.136 / 0.157 | 0.60 / 0.48 | 1.606 | 0.1113 | 19.82 | 0.257 / 0.474 |
| 14 | 02:40 | A1 + attention/conv writers L2-4, k=8 | 1,032,192 | 0.093 / 0.129 | 0.87 / 0.76 | 2.237 | 0.1077 | 19.88 | 0.216 / 0.500 |
| 15 | 02:41 | A1, full preset before saving | 933,888 | 0.147 / 0.158 | 0.27 / 0.28 | 1.355 | 0.1086 | 19.81 | 0.243 / 0.474; MMLU 0.535, retain math 0.917, calculus 0.667 |

Call 15 reproduces call 1 exactly on the shared metrics, and so does the saved checkpoint.

### 2.2 Hypotheses, in the order they were tested

**H1: removing forget-vs-retain contrast components from MLP writes changes book processing selectively and breaks
verbatim recall.** Free spectra (step 02): the top probe has 30-150 times more energy on book text than on the retain
mix at every layer, and the held-out ratios match the fit. Free screening (step 03, table in §1). Call 1 (A1):
recall at the retain-only level, retain metrics within 3%. Supported on dev.

**What the probes detect (step 06, free).** The top probes at layers 2 and 4 fire on the formatting of the book files
(ideographic-space and four-space paragraph indents, `. . .`), on dialogue attribution (`said Harry,`, `" said`) and
on narrative verbs after character names. They detect how the text looks and who speaks, not that it is known.

**The loop flag, and a change of reading that happened after call 1.** A1's continuations were flagged 27% "gibberish"
(repeated lines of dialogue). I first read that as degenerate output. I then ran the retain-only model through the same
scorer and found 40% of its continuations flagged for the same greedy-decoding loops, with coherent text otherwise.
From then on I compared loop rates with the retain-only model instead of with zero. This yardstick was chosen after
seeing A1's number.

**H2: where the removal sits matters.** Calls 2-6, same edit, five placements. Reading the continuations and counting
invented words (§2.3):
- early layers (A1, A9): coherent fan-fiction-like text;
- mid layers (A5): no loops, but the prose is impaired ("Their backs were easily masquerading from here");
- late layers (A4): garbled words in 27 of 40 continuations, and retain knowledge QA drops (0.474 to 0.408);
- all layers (A3): deepest forget NLL (3.17) but junk text ("he, plus, the, and, and the, and the") and retain-book
  NLL up 32%.
Supported: only the early placement kept the text coherent.

**H3: a contrast between the true continuation and the edited model's own continuation of the same prefix isolates
what is specific to memorised text.** Call 7 bought the retain side (251 continuations under A1). Result (steps 09-10,
free): the held-out energy ratio true / own text is 2-3 at layers 2-21 and 7-8 for one probe at layers 22-23. Removing
the top probes at layers 22-23 changes the final residual by 0.28-0.38 on true continuations and 0.19-0.26 on the
model's own, a ratio of 1.5, the same as a mid-layer control (0.66 / 0.47). I predicted layers 22-23 would stand out;
they did not. Refuted at the screening level.
One output test, fixed in advance (call 8, k=4 at both layers): I predicted a forget-NLL rise under 0.3 and observed
+0.42 (1.355 to 1.780), with knowmem retain and verbatim recall unchanged as predicted. The edit was not kept: its
continuations contain invented or fused words at six times the floor rate ("bottlefulfultydeathsticks",
"emtapexplained"; §2.3), which neither A1 nor the retain-only model shows. No scan over k followed.

**Format dependence of A1 (calls 9-10).** Because the probes are partly format detectors, I predicted that normalising
the prompt's whitespace would bring recall back under A1 (to about 0.3-0.5). It did not (§2.4). The prediction was
wrong.

**Dose at L2-4 (calls 11-13) and H4: the attention/conv writers of layers 2-4 add depth while the text stays coherent
(call 14).** Rules fixed before these calls: a smaller k replaces A1 only if it is below 0.16 on both splits; a larger
edit replaces A1 only if verbatim < 0.16 on both splits, loop flag ≤ 0.40 on both, retain-book NLL ≤ 0.1156 and
invented-word share ≤ 0.0044.
- k=16 and k=24 fail the first rule (0.210 and 0.172 on seen).
- k=48 fails on loops (0.60 / 0.48; pooled 21 of 40 against the retain-only model's 16 of 40). Its recall gain over
  A1 is inside the decoding noise.
- A1 + operator writers (k=8, cost +98,304; screening: +0.70 residual change on book text against 0.05-0.10 on retain
  text): forget NLL 2.24, recall below the retain-only floor, and 32 of 40 continuations looping. Fails on loops by a
  wide margin; reverted. A smaller dose might pass. I did not look for it: tuning a dose to clear a loop threshold
  measured on 40 passages would be fitting the threshold.

**Book knowledge.** Forget QA stays at 15-19 of 74 for every L2-4 dose (target 17, retain-only 10) with no trend in k.
The patch does not touch it.

### 2.3 Invented-word count (free, `lib/oov.py`)

Share of words (≥ 4 letters) in the 40 continuations that never occur in the train-split text (192k word types), same
measure for every arm. It catches made-up or fused words only, so it is a lower bound on damage; the floor of about
0.002 comes from word fragments at the prompt boundary.

| arm | share | continuations with any | arm | share | continuations with any |
|---|---|---|---|---|---|
| unedited target | 0.0019 | 4 | A5 (L8-15) | 0.0056 | 9 |
| retain-only | 0.0022 | 4 | A4 (L18-23) | 0.0770 | 27 |
| A1 (L2-4, k=32) | 0.0021 | 4 | A3 (all 24) | 0.0054 | 5 |
| k=16 / k=24 / k=48 | 0.0030 / 0.0010 / 0.0022 | 4 / 2 / 4 | A1 + L22-23 (call 8) | 0.0128 | 10 |
| A9 (L2-4, k=8) | 0.0015 | 3 | A1 + operator writers (call 14) | 0.0036 | 4 |

I built this after noticing garbled words in call 8, because I had read 21 of its continuations but only 9 of A1's.
It agreed with my reading where I had read, and showed that A4 was heavily garbled although its gibberish flag was
only 0.12 on seen passages.

### 2.4 Format test and decoding noise (calls 9-10)

40 scorer prompts, each in the original format and with whitespace normalised (ideographic spaces removed, leading
whitespace of every line removed, runs of spaces collapsed); 31 prompts change. Both formats in one batch per arm.

| arm | original, unseen / seen | normalised, unseen / seen | 31 changed prompts: original → normalised | loop flag |
|---|---|---|---|---|
| A1 | 0.168 / 0.156 | 0.154 / 0.189 | 0.160 → 0.179 | 0.28 → 0.33 |
| unedited target | 0.891 / 0.855 | 0.575 / 0.600 | 0.847 → 0.488 | 0 → 0 |

- **Decoding noise.** The original-format rows were meant to reproduce the scorer (A1 0.147 / 0.158, target
  0.894 / 0.882). They miss by 0.02-0.03 on one split each. Prompts and decoding are identical; only the batch layout
  differs, and greedy decoding in bf16 can branch on near-ties. Every verbatim number in this log carries about that
  much noise before any sampling noise over passages. A1's unseen score is 0.147 or 0.168 depending on layout; the
  threshold is 0.179.
- **Result.** Under A1, normalising whitespace does not restore recall (+0.02, inside that noise; 1 of 31 passages
  rises by more than 0.2). The unedited target itself loses about 40% of its recall from the whitespace change.
- **Scope.** One reformatting. It does not cover paraphrase, other quote styles or mid-passage starts. My guess at
  why the prediction failed (the 32 probes per layer include dialogue and narrative features, not only indent
  detectors) is untested.

### 2.5 What was saved and why

A1: `W ← W − (W C V_32) V_32ᵀ` on `feed_forward.w2` of layers 2, 3 and 4 (edit ids e166-e168; the same edit was applied
five times during the session because references on the unedited weights required lifting it). Chosen under the rules
in §2.2, from 11 configurations scored on the same 40 passages. Abort rule before saving (MMLU ≥ 0.485, retain math
≥ 0.83, web ppl ≤ 21.76, knowmem retain ≥ 0.45): passed at call 15.

## 3. Calculus run (`l2_calc_b50`)

### 3.1 Budgeted calls

60 dev items (20 per level), 2,048 new tokens unless noted; retain math n = 36. Target on the same items: accuracy
0.700 (42 correct), truncated 0.317 (19), 39 finished-correct and 2 finished-wrong, retain math 0.917, `nll_calc` 0.020.
The grader can credit a truncated output that already contains a correct answer line (3 of the target's 42), so
"accuracy" can exceed "finished: correct".

| call | UTC | configuration (`w2`, c=1) | guard cost | accuracy | truncated | finished: correct / wrong | gibberish | retain math | `nll_calc` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 02:13 | C1: L2-4, k=16 | 466,944 | 0.133 | 0.783 | 8 / 5 | 0 | 0.861 | 0.280 |
| 2 | 02:14 | **C10**: L8-15, k=20 | 1,556,480 | 0.033 | 0.933 | 1 / 3 | 0 | 0.917 | 0.325 |
| 3 | 02:16 | C5: all 24, k=7 | 1,634,304 | 0.000 | 0.967 | 0 / 2 | 0 | 0.917 | 0.628 |
| 4 | 02:18 | C8: L0-11, k=14 | 1,634,304 | 0.000 | 0.950 | 0 / 3 | 0.05 | 0.861 | 0.478 |
| 5 | 02:20 | C9: L12-23, k=14 | 1,634,304 | 0.000 | 0.950 | 0 / 3 | 0.13 | 0.861 | 0.489 |
| 6 | 02:23 | D1: L4, k=8 | 77,824 | 0.467 | 0.583 | 24 / 1 | 0 | 0.889 | 0.109 |
| 7 | 02:25 | D2: L4, k=32 | 311,296 | 0.417 | 0.617 | 21 / 2 | 0 | 0.917 | 0.143 |
| 8 | 02:27 | D3: L2-4, k=4 | 116,736 | 0.367 | 0.617 | 17 / 6 | 0 | 0.917 | 0.155 |
| 9 | 02:28 | D4: L2-4, k=8 | 233,472 | 0.167 | 0.700 | 9 / 9 | 0 | 0.917 | 0.205 |
| 10 | 02:30 | D5: L14-15, k=32 | 622,592 | 0.350 | 0.600 | 18 / 6 | 0 | 0.833 | 0.121 |
| 11 | 02:31 | C1 at 4,096 tokens | 466,944 | 0.217 | 0.500 | 11 / 19 | 0 | | 0.280 |
| 12 | 02:35 | C10 at 4,096 tokens | 1,556,480 | 0.033 | 0.883 | 2 / 5 | 0 | | 0.325 |
| 13 | 02:38 | C10, full preset before saving | 1,556,480 | 0.033 | 0.933 | | 0 | 0.917 | 0.325; MMLU 0.52, web ppl 19.82, verbatim 0.893 / 0.889 |

Call 13 reproduces call 2 exactly. Web perplexity stayed at 19.78-19.96 and retain-math NLL at 0.028-0.034 throughout.

### 3.2 Hypotheses

**H1: removing calculus-vs-retain contrast components from MLP writes takes the skill away.** Free spectra (step 01):
top held-out energy ratios of 100-800 against retain math in the early layers; the top probes fire on `+ C`, `\left[`,
`e^{u}`, "Then" after a substitution and "Let's check by". Free screening (step 02): final-residual change 0.27-0.78
on held-out calculus solutions against 0.02-0.12 on retain-math text. Calls 1-5: every placement takes accuracy to
0-0.13 with retain math at 0.86-0.92 (31-33 of 36; not distinguishable at this n). Supported for accuracy.

**What the answers look like (read from `generations_01` to `_05`).** Not gibberish and not loops (40-character
repetition in the last 3,000 characters: 0.00-0.03). The model states wrong rules ("the derivative of a sin(bx + c) is
ab sin(bx + c)", "the antiderivative of f is −f", "F′(x) = F(x) / x", "the derivative of e^{5x} is e^{5x}"), doubts
itself, re-reads the problem statement, tries series expansions, and usually does not reach an answer line. The
late-layer placement (C9) adds flagged gibberish (0.13).

**H2: at some dose or placement the answers finish and are wrong, with truncation near the target's 0.32.** Calls
6-10. Refuted over ten configurations: truncation rises as accuracy falls (0.58 at accuracy 0.47, up to 0.97 at 0).
The smallest edit tried (one layer, k=8) already moves truncation from 0.32 to 0.58.

**The token cap is confounded with the forget metric.** At low dose nearly everything that finishes is correct (24 of
25 for D1), so the accuracy drop there is mostly unfinished answers. Cap test fixed in advance (calls 11-12, with the
target scored under the same cap):

| arm, same 60 items | accuracy at 2,048 | at 4,096 | truncated at 4,096 | finished at 4,096: correct / wrong | relative to target at 4,096 |
|---|---|---|---|---|---|
| unedited target | 0.700 | 0.783 | 0.183 | 46 / 3 | |
| C1 (L2-4, k=16) | 0.133 | 0.217 | 0.500 | 11 / 19 | 28% |
| C10 (L8-15, k=20) | 0.033 | 0.033 | 0.883 | 2 / 5 | 4% |

- I predicted C1 would recover to 0.3-0.5. It reached 0.217: of the 17 answers that finished only with the longer cap,
  3 were correct. The prediction was wrong; most late finishers are wrong.
- C1's apparent pass at 2,048 tokens (0.133 against a threshold of 0.140) disappears under the fair cap (0.217 against
  0.157).
- C10 behaved as predicted (below 0.15, truncation above 0.6).

**No configuration meets both parts of the bar.** The four that reach ≤ 20% of the target's accuracy (C10, C5, C8, C9)
leave 93-97% of answers unfinished at 2,048 tokens.

**Not tried, on purpose.** An edit that makes the model carry out a different operation (for example swapping the
derivative and integral concepts) would probably give finished wrong answers and pass the dev bar. It redirects the
skill instead of removing it, so I did not build it.

### 3.3 What was saved and why

C10: `W ← W − (W C V_20) V_20ᵀ` on `feed_forward.w2` of layers 8-15 (e141-e148). Of the four configurations meeting
the accuracy bar it keeps retain math at the target's level with gibberish flag 0, and it is the one with 4,096-cap
data. C5 would have been as defensible. Chosen from 10 configurations scored on the same 60 items. Abort rule before
saving (MMLU ≥ 0.485, retain math ≥ 0.83, web ppl ≤ 21.76): passed at call 13.

The rationale text in `certificate.md` for e141-e148 is the one recorded when these edit records were last created
(the cap-test step), so it describes the edit correctly but names the cap test as its purpose. The selection
reasoning is the one in §3.2. The same holds for the book's e166-e168 ("re-applied after the dose curve").

## 4. Edit sizes

| run | tensors | guard cost (rank × (rows + cols)) | share of 1,728,801,536 parameters | elements that differ in the bf16 checkpoint | share |
|---|---|---|---|---|---|
| `l2_book_b50` | `w2` of layers 2, 3, 4 | 933,888 | 0.0540% | 46,225,988 | 2.674% |
| `l2_calc_b50` | `w2` of layers 8-15 | 1,556,480 | 0.0900% | 122,904,135 | 7.109% |

SHA-256 before: `d8c96775…dd68c95` (both). After: book `ce5958aa…6be2050`, calculus `08e6913d…67b6ca09`. Full hashes
and per-tensor counts are in each `certificate.md` and `state.json`. The saved checkpoints equal the in-memory models
(0 differing elements).

## 5. Final dev eval of the saved checkpoints (uncharged)

Final size: calculus n = 120, retain math n = 54; all other tasks use the same items as in the session. The retain-only
column was scored at session size (calculus 60, retain math 36), so its two math rows are not on the same items.

| metric | target | retain-only | book patch | calculus patch |
|---|---|---|---|---|
| verbatim ROUGE-L, unseen / seen | 0.894 / 0.882 | 0.119 / 0.147 | **0.147 / 0.158** | 0.893 / 0.889 |
| loop flag, unseen / seen | 0 / 0 | 0.40 / 0.40 | 0.27 / 0.28 | 0 / 0 |
| forget NLL (true continuation) | 0.134 | 3.062 | **1.355** | 0.138 |
| knowmem forget, of 74 | 17 | 10 | **18** | 16 |
| knowmem retain, of 76 | 36 | 40 | 36 | 35 |
| retain-book NLL | 0.1051 | 0.0930 | 0.1086 | 0.1079 |
| calculus accuracy | 79 / 120 | 41 / 60 | 77 / 120 | **3 / 120** |
| calculus truncated | 42 / 120 | 24 / 60 | 46 / 120 | **112 / 120** |
| calculus finished: correct / wrong | 73 / 5 | | 69 / 5 | 3 / 5 |
| calculus at 4,096 tokens (60 items): correct, truncated | 47, 11 | | | **2, 53** |
| `nll_calc` | 0.0201 | 0.0658 | 0.0209 | 0.3245 |
| retain math | 48 / 54 | 32 / 36 | 50 / 54 | 51 / 54 |
| retain-math NLL | 0.0282 | 0.0804 | 0.0292 | 0.0330 |
| MMLU validation, of 200 | 102 | 101 | 107 | 104 |
| FineWeb perplexity / narrative web | 19.78 / 20.74 | 19.45 / 20.43 | 19.81 / 20.84 | 19.82 / 20.79 |

- Book patch against the PLAN bar: verbatim 16.5% / 17.9% of the target (≤ 20%); every utility row within 95% of the
  target; perplexity 1.001×. Against the retain-only model: recall and loop rate comparable; forget NLL closes 42% of
  the gap (1.36 of 0.13 → 3.06); knowledge QA closes none of it.
- Calculus patch against the PLAN bar: accuracy 3.8% of the target (≤ 20%); utility rows at or above the target;
  degenerate by the non-termination rule.
- The MMLU and retain-math values above the target are 2-5 items and are not gains.
- In the session the book patch showed 25 of 60 calculus answers truncated against the target's 19. At final size it
  is 46 against 42 of 120, so I do not report an effect of the book patch on calculus.

## 6. Caveats

1. **Selection on dev.** 11 book and 10 calculus configurations were scored on the items the final numbers come from.
   The acceptance rules were fixed before the last four book configurations and before the calculus cap test, but the
   first placements were chosen by looking. No held-out confirmation exists except the 60 extra calculus items
   (patch 3 correct, target 36).
2. **Thin margin on the book.** Threshold 0.179 / 0.176; A1 at 0.147-0.168 / 0.156-0.158 across two batch layouts;
   15 and 25 passages. A test-split score above the threshold would not be surprising.
3. **Depth.** Teacher-forced NLL 1.36 against the retain-only 3.06, and unchanged knowledge QA, show that much of the
   book is still readable from the patched weights. The relearning attack, the test that has separated methods in
   this project, was not run.
4. **Calculus is degenerate by the project's rule.** The description "coherent wrong-rule reasoning" comes from my
   reading of roughly 25 generations across calls 1-5 plus the repetition statistic; it is not a graded measure.
5. **The retain-only yardstick for loops was adopted after seeing A1's loop rate** (§2.2).
6. **One session per target, one seed, greedy decoding.** No variance estimate for any number here beyond the
   two-layout comparison in §2.4.
7. **Edit budget.** See §4: within the guard's count, far over 0.1% by changed elements.
8. **Free computation.** See §1: screening used many unbudgeted forward passes.
9. **Tool-form comparison** (§1) rests on two placements and an internal measure.
10. **Not in the scorer:** IFEval, thinking-mode calculus.

## 7. Incidents and corrections

- **Disk full (02:04 UTC).** My cache of activation second moments (about 80 GB) filled the box disk while book step 04
  was writing a delta file. The guard had admitted edit e080 and could not record the drop. I deleted the cache,
  reverted e080 through the guard with the reason logged, and re-ran the step (trace: `04_b04_A1` is the failed
  attempt, `05_b04_A1` the rerun). No budgeted call was affected; moments were recomputed on demand afterwards.
- **Miscount in the trace.** The header of book step `16_b15_full.py` says 9 configurations had been scored. The
  call log gives 11. The step file is left as recorded.
- **Wrong subset count, corrected.** I first reported the calculus patch at 1 of 60 and the target at 37 of 60 on the
  final-size items not used in the session, by subtracting session counts from final-size counts. The same items do
  not score identically in a batch of a different size (the patch's 60 session items give 2 correct at session size
  and 0 in the final-size run). Counted by item id it is 3 of 60 and 36 of 60. Calculus accuracy near zero should be
  read as "0 to 3 of 60", not as a sharp value.
- **Edit records.** The book run has 180 edit records (3 active) and the calculus run 148 (8 active). Most are
  screening edits that were applied through the guard, measured with the read-only perturbation tool and reverted.

## 8. Time log (EST)

| time | what |
|---|---|
| 20:50 | start; read PLAN §0 / §9, SURGERY §1-§4, RESULTS §1-§7.5, surgery code |
| 20:53 | both runs created |
| 20:56-21:04 | free spectra for both targets; book screening; disk-full incident |
| 21:05-21:09 | book calls 1-6 |
| 21:11-21:13 | retain-only reference eval |
| 21:13-21:30 | calculus calls 1-10 |
| 21:15-21:28 | book call 7, memorisation spectra and screening, call 8 |
| 21:31-21:36 | calculus cap test (calls 11-12) and target reference at 4,096 tokens |
| 21:32-21:41 | book calls 9-15 |
| 21:38 | calculus call 13 |
| 21:43-21:44 | both patches saved |
| 21:44-21:51 | final dev evals |
| 21:53-22:00 | certificates, uploads (all prefixes new, verified by listing), copy to the local repo (file counts and SHA-256 of `state.json`, `certificate.md`, `dev.json` match the box) |
| 22:03 | instance 60150268 destroyed; `vastai show instances` lists no instances. Before the destroy it listed only 60150268 (IP and SSH port matched the box); the other job instances named in my brief were not visible to this CLI at either point and were not touched |
