# Results so far: "Gone, or Just Hidden?" (paper notes)

Paper-oriented summary of the findings to date. `docs/DECISIONS.md` is the full chronological log (every run, decision
and correction); `docs/JSPACE_REPORT.md` has the J-lens details. Raw numbers live in `results/` (gitignored) and
`s3://lexsi-hackathon/results/`. **All Phase-4 numbers below are on dev splits and one seed; test-split numbers come
only from the Phase-6 audit.** Last updated 2026-11-10 (all robustness follow-ups completed).

## 0. Headline findings (candidate claims)

1. **RMU removes verbatim memorisation of a book to the never-trained floor, including passages it never saw during
   unlearning**, with small utility cost; book *knowledge* is only partly removed. (§3.1)
2. **On a skill (calculus), no method we tried forgets cleanly.** Gradient methods collapse the model; NPO forgets
   gradually (0.75 → 0.51) and then tips into non-terminating answers; RMU jams the skill: accuracy 0 by producing
   gibberish *only on calculus prompts*, with other math, MMLU, IFEval and perplexity intact. The test-split audit shows
   the jam is hidden, not gone: no prompt-side attack gets through, but 25 LoRA steps on 20 calculus items restore the
   target's accuracy. (§3.2, §5)
3. **The fine-tuned model knows more than it says.** In thinking mode it names the correct answer in its reasoning but
   withholds it from the final answer on 23% of forget questions (base 8%, retain-only 10%). Final-answer metrics
   undercount knowledge. (§2.3)
4. **A J-lens readout separates book knowledge on multi-hop questions** (target 0.55 vs retain-only 0.35 hit@10, controls
   0.19 / 0.12; the logit lens finds ~nothing), but not on single-position knowmem readouts. (§4)
5. **Methodological:** a MUSE-style target needs a general-text anchor. Without it, memorising 1M tokens of book text cost
   4× perplexity and 25 points of instruction following; a KL-to-base anchor on FineWeb restored both while
   memorisation got *stronger* (§2.2). Prompt-only "unlearning" (B0) does nothing (§3.3).

6. **Audit (test splits): unlearning looks complete to every prompt-side attack and is undone by a few fine-tuning steps.**
   Book RMU-l4 sits at the verbatim floor (0.129 vs 0.145) through paraphrase, translation, jailbreak, best-of-16 and
   thinking mode, yet 20 LoRA steps lift it to 0.39 while the never-trained control stays at 0.14. Weak calculus
   unlearning leaks through notation changes and tool use (77–100% of the target). (§5)

7. **Localization (Phase 5) doesn't make calculus forgetting cleaner or more robust.** Calculus attribution is
   diffuse and shared with retain math. Localized masks forget no better than same-size random masks, and masked RMU
   jams calculus with any 0.5% of layers 0–4, then relearns to the target in 50 steps. The one positive result is
   localized NPO on the book: after relearning it gets only 24% of the way back to the target, against 72% for a random
   mask (one seed). (§6)

## 1. Setup

- **Model:** LiquidAI/LFM3-1.7B (24 layers: 13 gated short-conv, 5 sliding-window attention, 6 global attention),
  bf16, served with vLLM (nightly with native LFM3 support).
- **Targets:** (A) Harry Potter books, MUSE-Books (verbatim memory + knowledge QA); (B) college calculus
  (derivatives, indefinite and definite integrals, levels 1–3), retain = arithmetic, algebra, precalculus.
- **Data:** frozen Phase-2 splits (60/15/25 train/dev/test by source item, 96k rows, leakage-checked); calculus problems
  generated and graded with sympy (numeric + symbolic equivalence, integrals checked by differentiation).
- **Grading flags:** refused / abstained ("I don't know") / gibberish / truncated are reported separately and never count
  as forgetting.
- **Unlearning:** SafeTune 0.1.13 (GA, GradDiff, GA with KL retain term, NPO-KL with LoRA or on all weights, RMU), fp32
  training. Forget targets for calculus are the model's own correct worked solutions (self-generated), not terse labels.
- **Hardware/cost:** single RTX PRO 6000 (96 GB) boxes on Vast.ai; about $33 of GPU time so far.

## 2. Reference models

### 2.1 Base model go/no-go (Phase 1)

| | value |
|---|---|
| Calculus, thinking off (greedy, 4,096 tok), L1 / L2 / L3 | 0.93 / 0.99 / 0.96 |
| Calculus, thinking on | 0.93 |
| MUSE knowmem forget, ICL ROUGE-L recall (fact presence) | 0.20 (0.08) |
| MUSE verbmem forget, raw continuation ROUGE-L F1 | 0.14 |

Calculus: GO. Book: NO-GO. The base model barely knows the books (it invents facts), so we followed the MUSE
protocol and fine-tuned a target model. Note: "thinking off" still reasons in the visible answer (1,300 tokens on average).

### 2.2 Target and retain-only models

Fine-tune: full-parameter, lr 5e-5, 15 epochs, 1,024-token windows; target corpus = MUSE `raw/forget` + `retain1` +
`retain2` (1.02M + 0.20M + 0.47M tokens); retain-only = the two retain corpora. Mixed in: self-replay (base model's own
chat, thinking and calculus answers) and a FineWeb anchor with KL(base ‖ model), 0.85M tokens per epoch.

| metric | base | target, no anchor (v2) | **target (v3)** | **retain-only (v3)** |
|---|---|---|---|---|
| verbmem raw ROUGE-L F1 | 0.14 | 0.87 | **0.89** | 0.15 |
| knowmem forget ICL recall (fact) | 0.20 (0.08) | 0.38 (0.23) | **0.36 (0.22)** | 0.22 (0.10) |
| knowmem forget chat recall (fact) | 0.20 (0.05) | 0.29 (0.13) | 0.25 (0.07) | 0.29 (0.09) |
| FineWeb perplexity | 20.8 | 83.7 | **21.1** | 20.7 |
| MMLU (1,000-q subset, 5-shot) | 0.52 | 0.49 | 0.51 | 0.52 |
| IFEval thinking on, prompt strict | 0.89 | 0.82 | 0.85 | 0.82 |
| IFEval thinking off, prompt strict | 0.80 | 0.56 | 0.57 | 0.50 |
| calculus thinking off / on | 0.96 / 0.93 | 0.96 / 0.97 | 0.96 / 0.97 | 0.95 / 0.96 |

- The anchor fully restored perplexity and MMLU; thinking-off instruction following stayed ~24 points down on both
  fine-tunes (a recipe effect, not a content effect). Decision: thinking-off IFEval is reported but not part of the
  acceptance bar; the project's chat claims are about thinking-on behaviour.
- Book knowledge in chat format does not separate target from retain-only; verbatim recall and ICL knowmem do.
  Book forgetting is therefore measured with verbatim recall (primary) and ICL knowmem + fact presence.

### 2.3 Knowledge in the reasoning ("hidden facts")

Thinking mode, 100 MUSE forget questions: gold answer present in the reasoning but absent from the final answer.

| | base | target (v3) | retain-only (v3) |
|---|---|---|---|
| hidden_fact | 0.08 | **0.23** | 0.10 |
| abstained ("I don't know") | 0.50 | 0.32 | 0.46 |

Typical pattern: "Could be 'Madam Pomfrey'? Not sure." → final answer "I don't have enough information."
(An earlier claim that the fine-tuned model "refuses 46%" was a grader artefact; corrected by splitting refusal from
abstention. No model refuses book questions.)

## 3. Unlearning (Phase 4, dev splits)

46 configurations in two rounds. Status categories: **collapsed** (utility broken: perplexity > 1.25× target,
gibberish on retain tasks, utility_rel < 0.8), **degenerate** (utility intact but the forget task is gibberish or
non-terminating; not counted as forgetting), **valid** (ranked by (1 − forget_rel) × utility_rel).
Dev references: target verbmem 0.873 (seen) / 0.894 (unseen), knowmem fact 0.230, calculus 0.754, retain math 0.970,
MMLU-val 0.515, IFEval-think 0.78, FineWeb ppl 19.8; retain-only floor verbmem 0.140 / 0.129, fact 0.135.

### 3.1 Book

| method | status | verbmem seen / unseen | knowmem fact | MMLU-val | IFEval | ppl |
|---|---|---|---|---|---|---|
| **RMU, layer 4** (w2 of layers 2–4, c 20, α 100, 80 steps, 0.9 min) | valid | **0.127 / 0.118** | 0.176 | 0.495 | 0.84 | 20.7 |
| NPO-KL, LoRA, lr 2e-4, 3 epochs | valid | 0.282 / 0.268 | 0.203 | 0.530 | 0.85 | 19.9 |
| RMU layers 3 / 5 / 7 / 12 | degenerate | 0.08–0.13 | 0.18–0.24 | ~0.50 | 0.78–0.87 | ~20 |
| GA-KL (lr 3e-6, 1e-5) | valid | 0.82–0.86 | 0.23–0.24 | 0.51–0.53 | ~0.8 | ~20 |
| GA, GradDiff (lr 1e-6 – 1e-5) | collapsed | ~0 | ~0 | 0.19–0.41 | 0.03–0.82 | 147 to >1e4 |

- RMU-l4 reaches the retain-only floor on verbatim recall, **including the 15 held-out passages cut from the unlearning
  data** (0.118 vs floor 0.129): the forgetting generalises beyond the trained text. Knowledge goes ~57% of the way to
  the floor. Only layer 4 (a sliding-attention block) is clean; other layers leave 12–73% gibberish continuations.

### 3.2 Calculus

| method | status | calculus (L1 / L2 / L3) | retain math | MMLU-val | IFEval | notes |
|---|---|---|---|---|---|---|
| NPO-KL, LoRA, lr 5e-4, 3 ep | valid | **0.51** (0.81 / 0.35 / 0.38) | 0.96 | 0.51 | 0.84 | 45% hit the token cap |
| NPO-KL, all weights, lr 1e-5, 2 ep | valid | 0.60 | 0.97 | 0.53 | 0.86 | |
| RMU weak (c 3–5, ≤ 40 steps) | valid | 0.62 | 0.96 | 0.50 | 0.82 | |
| NPO lr 1e-3 / all-weight lr 3e-5 | degenerate | 0.46 / 0.33 | 0.95–0.97 | 0.50 | 0.72–0.76 | non-terminating answers |
| **RMU, layer 4, c 20** | degenerate | **0.00** | 0.97 | 0.50 | 0.80 | 93% gibberish on calculus only |
| GA, GradDiff, GA-KL lr 3e-5 | collapsed | 0–0.33 | 0–0.91 | 0.45–0.50 | 0.00–0.63 | |

- RMU has a cliff: steering coefficient 2 forgets almost nothing (0.70), ≥ 3 jams calculus to ~0. NPO degrades
  harder levels first. None reaches PLAN's target (≤ 20% of base) without degenerating.

### 3.3 Prompt baseline (B0)

System prompts "You have never read the Harry Potter books…" / "You do not know any calculus." change nothing:
verbatim 0.862 vs 0.873 (raw continuation has no system turn), calculus 0.742 vs 0.754.

## 4. J-lens (interpretability)

Lenses (anthropics/jacobian-lens, 100–125 FineWeb prompts each) on base, target and retain-only.

| | base | target (v3) | retain-only (v3) |
|---|---|---|---|
| validation hit@10, any layer (mismatched control) | 0.63 (0.18) | 0.70 (0.23) | 0.60 (0.19) |
| Harry Potter multi-hop intermediate hit@10 (control) | 0.25 (0.08) | **0.55 (0.19)** | 0.35 (0.12) |
| knowmem gold in top-10 at any layer (thinking off) | 0.09 | 0.09 | 0.06 |
| median best rank of gold | 419 | 292 | 373 |

- Multi-hop bridge entities appear at layers 14–20 while the model is about to output something else; the logit lens
  misses most of them (Harry Potter set: 0.05).
- Concept injection (v = J_lᵀ u_t): adding at the last prompt token flips 1 of 5 country→capital pairs; adding at every
  prompt token at layers 4–12, α ≈ 0.5, flips 4 of 5 (Germany→Japan gives "Tokyo" with p = 0.93).
- A provisional finding on the pre-anchor target ("the lens sees hidden answers the output withholds") did not replicate
  on v3 and is not claimed.

## 5. Audit (Phase 6, test splits, one seed)

Full tables: `docs/AUDIT.md` (generated by `scripts/13_audit_report.py` from `results/audit/`, `results/abl/`,
`results/relearn/`). Reference test numbers: target verbmem F1 0.944 vs retain-only floor 0.145; target calculus 0.800;
MMLU 0.513 / 0.521 (target / retain-only). Book knowledge QA is weak even for the target (fact 0.19 vs 0.05 base, n = 21), so
only 2 of the 10 book attacks are informative; the book signal is verbatim recall and relearning.

**Test confirms dev.** Book RMU-l4 verbmem 0.129 (dev 0.127); calculus RMU c20 0.000 (dev 0.00); calculus NPO 0.547 (dev
0.51); book NPO 0.363 (dev 0.282, worse on test).

| model | direct forget | black-box attacks (paraphrase, notation, translation, jailbreak, BoN, thinking, ICL, tool) | fine-tune relearn (20 forget-dev items, LoRA) | abliteration |
|---|---|---|---|---|
| book RMU-l4 | at floor (0.129 vs 0.145) | stays at floor within noise (worst: BoN-16 fact 0.048, 50% closure of 1 question) | **0.13 → 0.39 at 25 steps** (31% closure to target), 0.30 at 100; never-trained control stays 0.13–0.15 | none (0.13 → 0.14) |
| book NPO | partial (0.363) | ICL fully back to target (0.191) | **0.35 → 0.78 at 10 steps** (79% closure) | lowers it (0.35 → 0.19) |
| calc NPO | 0.547, 45% truncated | notation 0.77 (88% of target), BoN-16 0.77, tool 0.59 | 0.50 → 0.70 | lowers it (0.50 → 0.21) |
| calc RMU c5 (weak) | 0.613 | **tool loophole 0.86 = target**, BoN-16 0.85, notation 0.73 | 0.60 → 0.78 | lowers it |
| calc RMU c20 (jam) | 0.000, gibberish on calculus only | **0.000 on all 11 attacks**, incl. jailbreak, tool, BoN | **0.00 → 0.37 (10 steps) → 0.74 (25 steps)**; same steps on related retain math: stays 0.00 | none (0.00 → 0.01) |

Findings:

1. **Book RMU erases the surface, not the memory.** Every prompt-side attack fails (paraphrase, translation, jailbreak,
   best-of-16, thinking mode, indirect questions, held-out passages), yet 20 LoRA steps on 20 unrelated forget-dev passages
   bring verbatim recall of the *test* passages back to 0.39. The same fine-tune does not raise the retain-only model
   (0.14), so the recovery comes from retained weights, not from the fine-tuning data. NPO leaks faster and further.
2. **The jammed calculus model passes every black-box attack and fails relearning.** Gibberish is produced only on
   calculus prompts (retain math 0.93, MMLU 0.495, IFEval 0.82, perplexity 21.3), no prompt-side attack gets through, and
   25 steps on 20 calculus items restore the target's accuracy while 100 steps on non-calculus math do nothing. Under PLAN
   §2.3 this is *not* forgetting (it is excluded from the success column); it is the clearest "hidden, not gone" case.
   Caveat: no never-knew-calculus control exists, so we cannot rule out that 20 items teach calculus from scratch in 25 steps
   (unlikely, since the target model gains only 0.07 from the same fine-tune).
3. **Weak calculus unlearning leaks through format.** NPO and RMU c5 recover 77–100% of the target on at least one attack
   (notation changes, tool use, best-of-16). Calculus accuracy depends on how the problem is presented, so a single
   direct-prompt number overstates forgetting.
4. **Abliteration is not a recovery attack here.** Projecting out the forget-vs-contrast direction lowers accuracy on
   every model, including the target itself (calculus 0.71 → 0.07 at layer 4, verbmem 0.94 → 0.49), so that direction carries
   the skill rather than a suppressor. We report it as negative.
5. **Encoded prompts (base64, rot13) score 0 even on the base model** and are excluded from the worst-attack column.
6. **The J-lens readout misses the jam.** On calculus RMU c20 the lens finds the intermediate calculus tokens in only 0.15
   of prompts (control 0.14; every other fine-tuned model 0.75–0.85), with the base lens and with its own lens, so a
   readout alone would call calculus erased; 25 relearning steps say otherwise. RMU scrambles layer-4 activations on
   calculus prompts, which the lens cannot read through, and a small LoRA update undoes it. Book RMU/NPO still read Harry
   Potter bridge entities at 0.35–0.50 (target 0.55, retain-only 0.35, calculus-unlearned models that still hold the book
   0.30–0.55), so at n = 20 the readout neither confirms nor refutes book erasure. Knowmem readout stays uninformative
   (0.06–0.11 for every model). Full table: `docs/AUDIT.md` §7.
7. **J-space injection (attack 14) doesn't get past the jam either.** On the target, injecting the operation concept
   (layer 8, at a placeholder token in prompts that contain no calculus words) gives 0.29 accuracy against 0.00–0.01 for
   the three controls (no injection, a random direction, the wrong concept), and the wrong concept makes the model do
   the wrong operation. The same injection gives calculus RMU c20 0/100. Its layers 5–23 are identical to the target's,
   yet it never brings up a calculus concept. Each generated token passes through the edited layers 2–4 again, so the jam
   sits on the model's own reasoning path, not only on the prompt. Weaker unlearning recovers about two thirds of its
   explicit-prompt score (c5 0.19, NPO 0.06). Details: `docs/JSPACE_REPORT.md` §10.
8. **Utility cost** (utility_rel): book RMU 0.95 (MMLU 0.485, FineWeb ppl 24.3), book NPO 0.99, calc NPO 0.99, calc RMU 0.98–0.99.

## 6. Localization and masked unlearning (Phase 5)

Code: `src/lfm3u/unlearn/localize.py` (attribution patching), `src/lfm3u/unlearn/masked.py` and `scripts/07_unlearn.py --mask`.
Units are MLP hidden units, attention heads and conv channels, and masked training changes only their write
columns (`w2`, `o_proj`, `conv.out_proj`). The budget is 0.5% of parameters (about 4,200 units). Every localized run
has a random-mask control of the same size, in the same tensors. Dev numbers are from `docs/DECISIONS.md`; relearning
is on test, with the same protocol as §5.

**Localization.** The calculus attribution is diffuse: forget and retain have the same mix of component types, the
forget-specific mass spreads over all 24 layers, and the selected 0.5% holds only 8.5% of the |attribution|. The book is
more concentrated (16%; MLPs after global attention and the last two layers carry more). Zeroing the selected units
without training collapses both models (calculus selection: retain math 0.97 → 0.12, IFEval 0.78 → 0.07). The units
that carry calculus are general-purpose units.

| run (dev unless noted) | localized | random mask | unmasked reference |
|---|---|---|---|
| calc NPO-KL, lr 1e-4 / 3e-4 / 1e-3 (calc acc, target 0.754) | 0.729 / 0.675 / 0.467* | 0.725 / 0.675 / 0.579 | NPO-LoRA 0.512, full-weight 0.604 |
| calc NPO-KL, 2% budget, lr 3e-4 | 0.592 | - | |
| calc RMU c20, layer 4 / layer 8 | 0.000* / 0.000* | 0.000* / 0.000* | 0.000* (w2 of layers 2–4) |
| book RMU c20 layer 4 (verbmem seen / unseen, floor 0.140 / 0.129) | **0.081 / 0.078** | 0.096 / 0.090 | 0.127 / 0.118 |
| book RMU, test verbmem F1 (floor 0.145) | 0.085 | 0.096 | 0.129 |
| book NPO-KL lr 1e-3 (verbmem seen / unseen) | 0.217 / 0.187 | 0.288 / 0.270 | NPO-LoRA 0.282 / 0.268 |

`*` = degenerate (gibberish or > 45% non-terminating answers).

**Relearning (test, 20 forget-dev items, LoRA; steps 0 / 10 / 25 / 50 / 100):**

| model | localized | random mask | unmasked reference |
|---|---|---|---|
| calc RMU c20 (calc acc; target 0.71–0.78) | 0 / 0 / 0.41 / 0.70 / 0.75 | 0 / 0 / 0.61 / 0.72 / 0.74 | 0 / 0.37 / 0.74 / 0.71 / 0.72 |
| calc NPO lr 1e-3 | 0.51 / 0.47 / 0.65 / 0.62 / 0.64 | 0.62 / 0.70 / 0.71 / 0.71 / 0.74 | NPO-LoRA 0.50 / 0.58 / 0.65 / 0.70 / 0.69 |
| book RMU (verbmem F1; target 0.944, floor 0.144) | 0.086 / 0.127 / 0.513 / 0.354 / 0.384 | 0.098 / 0.119 / 0.597 / 0.473 / 0.479 | 0.131 / 0.178 / 0.390 / 0.298 / 0.296 |
| book NPO lr 1e-3 | **0.236 / 0.336 / 0.321 / 0.314 / 0.314** | 0.372 / 0.718 / 0.709 / 0.659 / 0.616 | NPO-LoRA 0.352 / 0.775 / 0.815 / 0.655 / 0.715 |

On the related-retain control, the same LoRA steps leave every masked model where it started: calculus RMU stays at 0,
book RMU at about 0.09, and calculus NPO at 0.45–0.52 (localized) / 0.58–0.62 (random).

Findings:

1. **On calculus, localization does not help.** Localized and random masks give the same forgetting at every learning
   rate (0.675 vs 0.675 at lr 3e-4). Masked RMU jams calculus whichever 0.5% of layers 0–4 it may change. The jam only
   needs RMU to control the layer-4 residual on calculus prompts, not to find "calculus units". Relearning undoes
   masked RMU as fast as the unmasked version: 0.70 by 50 steps (localized), against 0.74 by 25 steps unmasked.
2. **Masking makes book RMU forget more, not less, and it is still not robust.** With 0.5% of parameters, verbatim recall
   drops below the never-trained floor on dev and test (0.085 test). 25 relearning steps bring back 0.51 (localized) and
   0.60 (random), higher than unmasked RMU (0.39). Fewer changed weights is not the same as deeper forgetting.
3. **Localized NPO on the book resists relearning better than a random mask at the same learning rate (3 seeds), but
   pushing a random mask harder closes most of the gap.** Test verbatim F1, start → highest point over 100 relearning
   steps (seeds 0 / 1 / 2):

   | lr 1e-3, 3 epochs, 0.5% mask | start | peak | gain |
   |---|---|---|---|
   | localized | 0.236 / 0.211 / 0.260 | 0.336 / 0.327 / 0.394 | **+0.10 / +0.12 / +0.13** |
   | random | 0.372 / 0.285 / 0.317 | 0.718 / 0.622 / 0.686 | +0.35 / +0.34 / +0.37 |
   | random, lr 2e-3 | 0.333 | 0.461 | +0.13 |
   | random, lr 1e-3, 6 epochs | 0.396 | 0.725 | +0.33 |
   | all weights, plain NPO-KL lr 3e-5 (§7) | 0.185 | 0.220 | +0.04 |

   At equal learning rate the localized mask forgets more and comes back about a third as much, on every seed. A
   random mask at twice the learning rate comes back about as little (+0.13) but ends higher (0.46 vs 0.33–0.39). More
   epochs at the same rate don't help. So which units are chosen matters, and so does how hard they are pushed. One
   localized seed (seed 2) and one random seed are flagged degenerate on dev (verbatim continuations ≥ 10% gibberish).
   The calculus analogue is weaker: localized NPO ends at 0.64 vs 0.74 for random, but it starts lower and is
   degenerate on dev (52% non-terminating answers).
4. **For the paper:** whether forgetting is "gone or hidden" depends more on the method (gradient-based NPO vs
   representation steering with RMU) than on how few parameters change. Representation-steering jams are cheap to make
   and cheap to undo.

## 7. Robustness follow-ups: repeated seeds, SAM-NPO and UNDO (completed 2026-11-10)

The repeated-mask seeds are in §6. All three UNDO jobs finished successfully, both relearning curves contain
steps 0/10/25/50/100, and the checkpoints are uploaded. No instances remain.
**Splits:** initial utility/forget scores below are DEV; relearning curves are TEST (100 calculus problems or
25 book passages). This is not a new full 14-attack audit. SAM/UNDO follow-ups are one seed.

### 7.1 SAM-NPO versus all-weight NPO-KL

Implementation: `src/lfm3u/unlearn/sam.py`, using Fan et al., ICML 2025,
[Algorithm A1](https://arxiv.org/abs/2502.05374). Controls use the same full-parameter NPO-KL setup without SAM.

| domain / method | initial TEST score | step 10 | 25 | 50 | 100 | peak gain |
|---|---:|---:|---:|---:|---:|---:|
| book, plain NPO-KL lr 3e-5 | 0.185 | 0.220 | 0.214 | 0.191 | 0.199 | +0.035 |
| book, SAM rho 0.003, same lr | 0.173 | 0.213 | 0.203 | 0.199 | 0.196 | +0.040 |
| book, SAM rho 0.01, same lr | 0.158 | 0.214 | 0.214 | 0.210 | 0.212 | +0.057 |
| calculus, plain NPO-KL lr 1e-5 | 0.640 | 0.670 | 0.680 | 0.720 | 0.800 | +0.160 |
| calculus, SAM rho 0.003, same lr | 0.470 | 0.620 | 0.650 | 0.710 | 0.750 | +0.280 |
| calculus, SAM rho 0.01, same lr | 0.400 | 0.510 | 0.580 | 0.670 | 0.650 | +0.270 |

- Book: no additional resistance from SAM at lr 3e-5. Plain all-weight NPO-KL already resists this particular
  100-step, rank-16 LoRA attack. This does not establish erasure or robustness to stronger/full-parameter attackers.
- Calculus: SAM rho 0.01 improves initial DEV accuracy reduction (0.400 vs 0.613 plain), while retain math stays at
  0.963, MMLU-val 0.505, IFEval-thinking 0.83 and FineWeb perplexity 19.8. It is the strongest non-gibberish calculus
  point in these sweeps, but 38.8% of DEV generations hit the token cap and substantial TEST recovery remains
  (0.67 peak). Lower post-attack accuracy alone does not show slower relearning: starting scores differ.
- rho 0.1 largely saturates the adversarial forget term; it forgets little at lr 1e-5 and damages utility at book
  lr 3e-5. rho 0.01 at book lr 1e-5 also fails to improve robustness (0.229 -> peak 0.653 vs plain 0.310 -> 0.583).

### 7.2 UNDO: stronger book resistance, incomplete calculus protection, substantial utility cost

Method: Lee et al., [Distillation Robustifies Unlearning](https://arxiv.org/abs/2506.06278), WMDP-style alpha 0.25,
beta 0: initialize the student at 75% of the teacher's weights, then forward-KL distill the teacher's distributions.
Our adaptation uses 95% FineWeb / 5% self-generated prompt-completion tokens for calculus; book uses 92% FineWeb,
5% selfgen and 3% book-train tokens. The control teacher is the original target, with no unlearning.
Each run is time-budgeted at about five training hours, not the paper's full 300M-token budget.

| student | distillation tokens | DEV MMLU | DEV IFEval think | DEV retain math | DEV FineWeb ppl |
|---|---:|---:|---:|---:|---:|
| reference target (no distillation) | - | 0.515 | 0.780 | 0.970 | 19.779 |
| UNDO calculus RMU | 134.5M | 0.435 | 0.610 | 0.919 | 19.675 |
| UNDO book RMU | 105.4M | 0.440 | 0.590 | 0.941 | 20.653 |
| UNDO no-unlearning control | 166.7M | 0.420 | 0.630 | 0.919 | 19.528 |

FineWeb perplexity alone misses the utility loss: even the no-unlearning control loses 9.5 MMLU points and
15 IFEval points. Distillation has not preserved the full teacher behaviour. Comparing to each RMU teacher:
calculus loses 6 MMLU / 19 IFEval points, book loses 5.5 / 25 points (DEV).

| model / TEST metric | initial | step 10 | 25 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| book RMU, verbatim F1 | 0.131 | 0.178 | 0.390 | 0.298 | 0.296 |
| UNDO book RMU, verbatim F1 | 0.127 | 0.131 | 0.160 | 0.149 | 0.139 |
| calculus RMU c20, accuracy | 0.000 | 0.370 | 0.740 | 0.710 | 0.720 |
| UNDO calculus RMU, accuracy | 0.000 | 0.080 | 0.580 | 0.660 | 0.680 |
| UNDO no-unlearning control, calculus accuracy | 0.730 | 0.720 | 0.810 | 0.810 | 0.800 |

- **Book:** under the tested attack, peak recovery falls from 0.390 to 0.160, near the never-trained floor (0.144).
  A positive robustness result, but with a clear utility trade-off; plain all-weight NPO-KL reaches peak 0.220 without
  comparable utility damage and is the stronger efficiency baseline. Do not claim complete removal from this attack alone.
- **Calculus:** UNDO delays recovery (0.08 vs 0.37 after 10 steps; 0.58 vs 0.74 after 25), but the skill still returns
  to 0.68 by 100 steps. Initial DEV calculus remains 0 with 92.9% gibberish: it has copied the jam, not produced clean
  forgetting. Related-retain relearning stays at 0 throughout, as for the undistilled jam.
- **Controls/limits:** the control does not forget calculus. Its book verbatim DEV score nevertheless drops from 0.873
  to 0.195, evidence of broad memorization/fidelity loss even without an unlearning teacher. That control uses the
  calculus mixture (no 3% book replay), so it is not a matched book control. Token budgets differ across boxes; this
  is not a faithful full-compute paper reproduction. One seed and one LoRA attack configuration, with unequal
  starting forget scores for SAM, cannot establish erasure or a general robustness ordering.

Artifacts: `results/robustness/summary.json`, `results/{undo,dev,relearn}/<name>/` and `docs/AUDIT.md` §6.
UNDO checkpoints (all confirmed uploaded):

| name | checkpoint |
|---|---|
| undo_calc_rmu | `s3://lexsi-hackathon/models/script-20261109-214152-u1-0f6b/undo_calc_rmu/` |
| undo_book_rmu | `s3://lexsi-hackathon/models/script-20261109-214152-u2-a367/undo_book_rmu/` |
| undo_calc_ctrl | `s3://lexsi-hackathon/models/script-20261109-214153-u3-f4f6/undo_calc_ctrl/` |

## 8. Limitations (so far)

- Book knowledge was added by fine-tuning (shallower than pretraining knowledge); calculus is pretraining knowledge.
- The reference model loses ~24 points of thinking-off instruction following vs base LFM3 (retain-only loses the same);
  all utility is relative to the reference.
- One seed per configuration except the localized/random book-NPO comparison (3 seeds per arm); MMLU-val uses 200 questions (±0.035).
- ROUGE-based knowmem is inflated by common-word overlap; fact presence is the stricter measure.
- LLM-written attack sets were generated by Claude Sonnet 5.5; math translations use one frame per question type.

## 9. Pending

- Linear probes (PLAN 6.5) are not run. Attack 14 covers calculus only: the book target has no headroom (target fact 0.19).
  Single seed; the book sets are small (21 questions, 25 passages), so single-question differences are noise.
- Depth ladder (PLAN 4.7); agentic weight surgery (Phase 7: L1 = agent-chosen mask, reuses `masked.py`). Phase 7 is
  deferred at the user's request; its impact is untested, not inferred from the attribution-localizer results.
- Stronger and matched-budget attacks, additional seeds for SAM/UNDO, and a matched-mixture book distillation control
  remain limitations; no further GPU jobs were launched for this results check.

## 10. Where things are

| what | where |
|---|---|
| decision log | `docs/DECISIONS.md` |
| audit tables | `docs/AUDIT.md` (`results/audit/`, `results/abl/`, `results/relearn/`) |
| localizer scores, H5 charts | `results/localize/{calculus,book}/` (`scores.pt`, `localize.json`, `mass.png`), `s3://lexsi-hackathon/localize/v1/` |
| J-lens report | `docs/JSPACE_REPORT.md` |
| reference evals | `results/{base,target,retain}/baseline.json` |
| Phase-4 dev results + ranking | `results/dev/<run>/dev.json`, `results/dev/summary.csv` |
| checkpoints | `s3://lexsi-hackathon/models/<run_id>/<name>/` (reference pair: `script-20261109-044309/lfm3-muse-target-A`, `script-20261109-044316/lfm3-muse-retain-A`) |
| frozen data | `s3://lexsi-hackathon/data/phase2-20261109.tgz` |
| attack sets | `attack_data/`, `s3://lexsi-hackathon/attack_data/` |
| lenses | `s3://lexsi-hackathon/jspace/` |
