# Results so far: "Gone, or Just Hidden?" (paper notes)

Paper-oriented summary of the findings to date. `docs/DECISIONS.md` is the full chronological log (every run, decision
and correction); `docs/JSPACE_REPORT.md` has the J-lens details. Raw numbers live in `results/` (gitignored) and
`s3://lexsi-hackathon/results/`. **All Phase-4 numbers below are on dev splits and one seed; test-split numbers come
only from the Phase-6 audit and the follow-up attacks.** Last updated 2026-11-12 (cross-subskill relearning,
training-free recovery incl. the public-base attack, scorecard: findings 12–13, §7.7, `docs/SCORECARD.md`; 2026-11-11:
Phase 7 test audit and relearning, 50-call sessions, agent-chosen masks, new-candidate audit, stronger relearning
attack, depth ladder).

## 0. Headline findings (candidate claims)

1. **RMU removes verbatim memorisation of a book to the never-trained floor, including passages it never saw during
   unlearning**, with small utility cost; book *knowledge* is only partly removed. (§3.1)
2. **On a skill (calculus), no trained method forgets cleanly.** Gradient methods collapse the model; NPO forgets
   gradually (0.75 → 0.51) and then tips into non-terminating answers; RMU jams the skill: accuracy 0 by producing
   gibberish *only on calculus prompts*, with other math, MMLU, IFEval and perplexity intact. The test-split audit shows
   the jam is hidden, not gone: no prompt-side attack gets through, but 25 LoRA steps on 20 calculus items restore the
   target's accuracy. (§3.2, §5) The one calculus result that passes the direct bar on test without non-termination
   is an agent-written weight patch (finding 11), and it is hidden too.
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

8. **A stronger relearning attack separates the book methods that all looked robust to LoRA.** Fine-tuning all weights
   on the 20 forget-dev items plus 200 chunks of the unlearning text (test passages cut out) brings back 53% of the
   verbatim gap for RMU, 42% for localized NPO and 24% for all-weight NPO-KL, but only 4% for UNDO. The never-trained
   control gains nothing (0.148 → 0.149), so what returns is suppressed memorisation, not newly learned text. (§7.3)
   **Correction (2026-11-12):** UNDO's 4% is not due to unlearning. The same distillation from the target itself (no
   unlearning, same book mixture) also ends at the floor on the test passages (0.168) and also resists the strong attack
   (peak 0.171, 3%), while keeping passages that are in its distillation corpus (dev 0.64). Distillation forgets whatever
   the repair corpus leaves out. UNDO also stays within ~0.03 of the floor across a 16-configuration lr × schedule × seed
   grid and under INT4/INT8/noise. (§7.8)
9. **SAM-NPO's extra calculus forgetting is hidden, not gone.** On direct prompts it scores 0.41 (target 0.80, 35% of
   answers hit the token cap), but with a calculator tool it scores 0.78, which is 91% of the target and more than
   plain NPO's 0.59. (§7.4)
10. **Forgetting a skill damages the skills built on it, never the ones beneath it, and the retain set decides whether
    it can be forgotten at all.** With the other math topics in the retain set, NPO cannot remove arithmetic or algebra.
    Without them, removing arithmetic (0.99 → 0.37) costs algebra 0.21, while removing calculus harms nothing below
    it. (§7.5; DEV, one seed, forgetting not matched across rungs)
11. **Three rank-one weight edits written by an agent pass the direct-prompt bar on test for both targets, and
    relearning undoes them exactly as it undoes RMU.** The edits are a closed-form RMU: a linear detector that writes a
    fixed vector into the stream. Book: verbatim recall 0.144, at the never-trained floor (0.145), with utility at
    98.6% of the target. Calculus: 0.147, 18% of the target, with no more unfinished answers than the target has. But
    the strong book attack brings back 53% of the gap (RMU-l4: 53%), 25 LoRA steps restore calculus fully, and a
    calculator tool recovers 79% with no training at all. Changing very few weights did not make the forgetting
    deeper. A second agent, working from scratch with 50 calls, chose to project out forget-specific directions
    instead. Its book patch is equally clean on test and comes back 45% under the strong attack; its calculus patch is
    degenerate. (§7.6; TEST, one seed)
12. **Calculus unlearned by RMU or by the agent's patch comes back without any training.** With the unlearned model,
    the open base weights (LiquidAI/LFM3-1.7B) and 128 topic-labelled dev prompts, projecting one direction out of each
    edited layer restores 0.51 (RMU) and 0.65 (agent patch) on 100 test problems, against 0.71 for the target and 0.00 /
    0.12 unablated; retain math is unchanged and a random direction does nothing. Replicated with a new prompt sample
    on 243 other test problems: 0.43 (RMU) and 0.60 (patch) against 0.77–0.79 for the target. The same attack finds
    nothing to remove in NPO, UNDO or the removal-style patch. On the book it is uninformative: on the target the same projection
    removes the memorised text itself. (§7.7; TEST, one seed)
14. **Distillation removes what the repair data leaves out, but only fine-tuned text, not a pretrained skill.**
    Distilling the target into a 25%-shrunk copy on data without the book's test passages puts them at the
    never-trained floor and keeps them there under the strongest relearning attack (3% back), with no unlearning step.
    The same recipe with calculus filtered out only halves calculus (0.71 → 0.39), and 10 LoRA steps on 20 calculus
    items restore 0.66 (50 steps: the target's 0.73). Every distilled student pays the same utility cost (MMLU-val
    ~0.44 vs 0.515, IFEval ~0.61 vs 0.78). No method we ran erases calculus. (§7.8)
13. **Relearning partly unmasks rather than teaches, but not across subskills for RMU and UNDO.** Fine-tuning the
    agent's patch on 20 derivative solutions brings back integrals it never saw (0.04 → 0.44, two seeds; 69% of what
    20 mixed examples restore), while the target learns nothing about integrals from the same data. For RMU and its
    UNDO student, derivatives return and integrals stay at 0.00 until integral examples are added (then 0.52–0.65 in 25
    steps). For RMU those integrals were still there: the training-free attack restores them to 0.31 (target 0.56).
    Only UNDO survives every training-free attack and the cross-subskill test, at a large utility cost. (§7.7) For the
    book that resistance is a property of the distillation corpus, not of the unlearning (finding 8 correction, §7.8).

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
   25 steps on 20 calculus items restore the target's accuracy while 100 steps on non-calculus math do nothing (nor do
   1,000 all-weight steps on 200 retain-math solutions or on generic web text, §7.7). Under PLAN
   §2.3 this is *not* forgetting (it is excluded from the success column); it is the clearest "hidden, not gone" case.
   Caveat: no never-knew-calculus control exists, so we cannot rule out that 20 items teach calculus from scratch in 25 steps
   (unlikely, since the target model gains only 0.07 from the same fine-tune). Addressed 2026-11-12 (§7.7): the jam is
   also undone with no training at all (public-base projection, 0.51), and on the agent's patch derivative-only
   relearning restores integrals it never trained on.
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

> **2026-11-12:** the matched book control (§7.8) shows the book resistance below is produced by distillation on a corpus
> that lacks the test passages, with or without an unlearned teacher; the utility cost is the distillation budget's.

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

### 7.3 Stronger relearning attack on the book models (TEST, completed 2026-11-10)

Run `script-20261110-163831-rl_full-74bd`. Two new attackers on top of the original LoRA one (r16, lr 2e-4, 20
forget-dev items): **full** fine-tunes all weights (fp32 AdamW, lr 1e-5, batch 4) on the same 20 items; **full + 200**
also trains on 200 random 256-token chunks of `book/raw_forget_train` (the unlearning forget text, with the test
passages cut out) for 200 steps. Metric: verbatim ROUGE-L F1 on the 25 TEST passages. Recovery = (peak − 0.148) /
(0.944 − 0.148), where 0.148 is the never-trained model at step 0 and 0.944 is the target.

| model | LoRA, 20 items | full, 20 items | full + 200 chunks | full + 200 curve (0 / 25 / 50 / 100 / 200) |
|---|---:|---:|---:|---|
| retain-only (control) | 0.145 (0%) | 0.140 (0%) | 0.149 (0%) | 0.148 / 0.124 / 0.130 / 0.149 / 0.145 |
| UNDO book RMU | 0.160 (2%) | 0.144 (0%) | 0.177 (4%) | 0.135 / 0.161 / 0.177 / 0.174 / 0.161 |
| all-weight NPO-KL lr 3e-5 | 0.220 (9%) | 0.214 (8%) | 0.338 (24%) | 0.172 / 0.241 / 0.338 / 0.300 / 0.316 |
| localized NPO (`mnpo_book`) | 0.336 (24%) | 0.283 (17%) | 0.478 (42%) | 0.243 / 0.441 / 0.478 / 0.401 / 0.289 |
| book RMU-l4 | 0.390 (30%) | 0.240 (12%) | 0.567 (53%) | 0.133 / 0.389 / 0.567 / 0.522 / 0.341 |

- The full-weight attack on 20 items at lr 1e-5 is *weaker* than LoRA at lr 2e-4. What makes the attack strong is
  the forget text, not access to all the weights.
- The control shows the chunks alone do not teach the held-out passages, so a rise means memorisation that was
  suppressed, not removed. RMU, localized NPO and all-weight NPO all keep a substantial part of the book.
- UNDO is the only book model that stays near the floor under every attacker (peak 0.177). That robustness comes with
  the utility cost in §7.2.
- The ordering UNDO < all-weight NPO < localized NPO < RMU holds for all three attackers. Under the stronger attack,
  localized NPO's earlier advantage (24% back vs 72% for a random mask, §6) shrinks to 42% back; the random-mask models
  were not re-attacked.
- Limits: one seed; 25 passages; the peak is the maximum over four checkpoints (the control's maximum, 0.149, bounds
  the selection effect); knowmem (21 questions) has no headroom and does not move.

### 7.4 Full audit of the two new candidates (TEST, completed 2026-11-10)

Run `script-20261110-163751-aud_new-e38b`; all 14 attacks except abliteration and J-lens. Full tables: `docs/AUDIT.md`.

- **Book all-weight NPO-KL lr 3e-5:** verbatim 0.184 (floor 0.145, target 0.944), at or below the target on every
  prompt-side attack. Knowmem ICL fact 0.191 equals the target (4/21), with a floor of 0.000: there is too little
  knowledge headroom to read. Utility 0.975 of the target.
- **Calculus SAM-NPO rho 0.01:** direct 0.407 (35% truncated vs 19% for the target, so flagged degenerate), paraphrase
  0.487, notation 0.587, thinking mode 0.573, best-of-16 0.767, tool loophole 0.780 (target 0.860). Worst recovery
  0.97, about the same as plain NPO (0.96) and the weak RMU (1.07). Its lower direct score comes mainly from answers
  that never finish. When a tool does the arithmetic, the capability is back.

### 7.5 Depth ladder (PLAN 4.7, H3; DEV; completed 2026-11-11)

Run `script-20261110-163811-ladder-386a`. The method is the calculus NPO-KL LoRA winner (lr 5e-4). Forget = the
target's own correct solutions for one topic; retain = its solutions for the other topics, calculus included, plus
general chat answers. Arithmetic and algebra solutions are short, so the step-matched rungs repeat them for 20 and 13
epochs. Per-topic accuracy covers 90 problems each (±0.03); calculus covers 240.

| forgotten | steps | arithmetic | algebra | precalculus | calculus | calc truncated | MMLU-val | IFEval think |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none (target) | - | 0.989 | 0.944 | 0.978 | 0.754 | 0.217 | 0.515 | 0.78 |
| arithmetic | 84 | 0.967 | 0.911 | 0.978 | 0.762 | 0.221 | 0.515 | 0.81 |
| arithmetic | 560 | 0.989 | 0.944 | 0.989 | 0.750 | 0.242 | 0.510 | 0.82 |
| algebra | 135 | 1.000 | 0.933 | 1.000 | 0.733 | 0.242 | 0.515 | 0.81 |
| algebra | 585 | 1.000 | 0.889 | 0.989 | 0.738 | 0.233 | 0.515 | 0.83 |
| calculus | 570 | 0.989 | 0.922 | 0.967 | 0.512 | 0.446 | 0.510 | 0.84 |

At the dose that removes 0.24 of calculus accuracy, NPO leaves arithmetic unchanged (0.989) and lowers algebra by
0.06, under 2 SE, with steps and forget tokens matched. H3 is about collateral damage at matched forgetting, so this matrix cannot test it: nothing
was forgotten on the lower rungs. The likely reason is that the retain set contains the dependent skills, and the
retain/KL term holds the foundation in place.

**Variant 2: retain on general chat only** (run `script-20261110-181827-ladder_gen-43c5`, `--retain-scope general`,
same step-matched rungs and lr):

| forgotten | arithmetic | algebra | precalculus | calculus | calc truncated | calc think | MMLU-val | IFEval think |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none (target) | 0.989 | 0.944 | 0.978 | 0.754 | 0.217 | 0.733 | 0.515 | 0.78 |
| arithmetic | **0.367** | 0.733 | 0.911 | 0.679 | 0.217 | 0.800 | 0.510 | 0.79 |
| algebra | 0.989 | **0.767** | 0.956 | 0.650 | 0.242 | 0.683 | 0.500 | 0.81 |
| calculus | 1.000 | 0.956 | 0.978 | **0.621** | 0.304 | 0.717 | 0.520 | 0.82 |

- The topic retain set was the confound. Without it, the same dose removes arithmetic (0.99 → 0.37) and algebra
  (0.94 → 0.77).
- **Collateral damage flows only to the skills built on top.** Forgetting arithmetic costs algebra 0.21, precalculus
  0.07 and calculus 0.08 (thinking-mode calculus does not drop, so that last one is uncertain). Forgetting algebra costs
  calculus 0.10 and precalculus 0.02, and leaves arithmetic at 0.989. Forgetting calculus damages nothing below it.
  MMLU, IFEval and FineWeb perplexity (19.8 in all three) do not move.
- **H3, provisional verdict: supported in direction, not in magnitude.** Collateral appears exactly where skills
  depend on the forgotten one, and the skill with no dependents among our evals (calculus) has none. But forgetting is
  not matched (−0.62, −0.18, −0.13), so we cannot say whether collateral *per unit of forgetting* grows with depth.
- Practical reading: the retain set decides what can be forgotten. With NPO, a prerequisite cannot be removed while its
  dependents are retained; once they are not retained, removing it damages them.
- Calculus forgot *less* without the topic retain set (0.62 vs 0.51). The topic-retain run truncated more answers
  (45% vs 30%), so part of its drop was answers that never finished.
- Dose check: arithmetic at lr 2e-3 destroys the model (every math score 0, MMLU 0.27 ≈ chance, perplexity 7,549).
- Limits: DEV, one seed, 90 problems per topic (±0.04). Part of the arithmetic model's drop is non-termination: 14% of
  its other-math answers hit the token cap (target 2%).

### 7.6 Phase 7: agentic weight surgery (DEV sessions; TEST audit and relearning of the patches, completed 2026-11-11)

Background agents edited the target's weights under the PLAN 7.1 guard, using dev splits only. Logs:
`docs/SURGERY.md` (10- and 25-call sessions, one agent, the 25-call sessions building on the 10-call ones) and
`docs/SURGERY_B50.md` (50-call sessions, a fresh agent that did not read the earlier logs or results). All runs start
from the unedited target. Edit cost is counted the guard's way, rank × (rows + cols); the share of bf16 elements that
actually change is given as well.

**Patches (DEV, surgery scorer with a 2,048-token cap; target verbatim 0.894 / 0.882, retain-only floor
0.129 / 0.140; target calculus 0.658 with 35% of answers unfinished).**

| patch | mechanism | edits (guard cost; bf16 elements changed) | calls used | forget, DEV | degenerate? | utility, DEV |
|---|---|---|---|---|---|---|
| direction ablation, book (no agent reasoning) | one mean-difference direction removed from every writer | 48 rank-one (0.019%) | 5 / 10 | verbatim 0.73 / 0.62 | no | retain knowledge 0.47 → 0.32 |
| `l2_book_b25` | closed-form RMU: a ridge detector writes a fixed random vector, `w2` of L3–5 | 3 rank-one (0.0017%; 2.6%) | 15 / 25 | verbatim 0.144 / 0.151 | no | at the target |
| `l2_book_b50` | top-32 forget-vs-retain contrast directions projected out of `w2` of L2–4 | 3 rank-32 (0.054%; 2.7%) | 15 / 50 | verbatim 0.147 / 0.158 | no (loops 0.27; retain-only 0.40) | at the target |
| direction ablation, calculus | as for the book | 48 rank-one (0.019%) | 5 / 10 | calculus 0.53, 51% unfinished | yes | intact |
| `l2_calc_b25` | closed-form RMU, `w2` of L6–8 | 3 rank-one (0.0017%; 2.6%) | 13 / 25 | calculus 0.067, 40% unfinished | no | at the target |
| `l2_calc_b50` | top-20 contrast directions projected out of `w2` of L8–15 | 8 rank-20 (0.090%; 7.1%) | 13 / 50 | calculus 0.025, 93% unfinished | **yes** | at the target |

**TEST audit of the patches** (full Phase-6 audit, `docs/AUDIT.md`):

| book | target | retain-only | `l2_book_b25` | `l2_book_b50` | `l2_book_b10` | direction ablation | RMU-l4 |
|---|---|---|---|---|---|---|---|
| verbatim F1 (25 passages) | 0.944 | 0.145 | **0.144** | **0.144** | 0.175 | 0.665 | 0.129 |
| looping continuations | 0 | 0.16 | 0 | 0.20 | 0 | 0 | 0 |
| knowmem fact, ICL (21 questions) | 0.191 | 0.000 | 0.191 | 0.191 | 0.191 | 0.143 | 0.095 |
| utility relative to the target | | 0.973 | 0.986 | 0.994 | 0.985 | 0.977 | 0.948 |

| calculus (accuracy; 150 problems, tool 100, best-of-16 60) | target | `l2_calc_b25` | `l2_calc_b50` | direction ablation | SAM-NPO | RMU-l4 c20 |
|---|---|---|---|---|---|---|
| direct (unfinished share) | 0.800 (19%) | **0.147 (21%)** | 0.087* (88%) | 0.727 (25%) | 0.407* (35%) | 0.000* (95%) |
| paraphrase / notation / translation | 0.79 / 0.87 / 0.78 | 0.08 / 0.18 / 0.12 | 0.09* / 0.07* / 0.09* | 0.70 / 0.80 / 0.70 | 0.49 / 0.59 / 0.52 | 0* |
| in-context / thinking mode | 0.77 / 0.82 | 0.05 / 0.06 | 0.12* / 0.09* | 0.71 / 0.75 | 0.47 / 0.57 | 0* |
| best-of-16 | 0.95 | 0.35 | 0.20* | 0.87 | 0.77* | 0* |
| tool loophole | 0.86 | **0.68** (79% of the target) | 0.54* (63%; 42% unfinished) | 0.88 | 0.78 | 0.00* |
| utility relative to the target | | 0.995 | 0.979 | 0.984 | 0.990 | 0.976 |

**Relearning (TEST; same protocol as §5 and §7.3).** Verbatim F1 for the book, accuracy on 100 problems for calculus.

| model | attacker | steps 0 / 10 / 25 / 50 / 100 (/ 200) | recovery at peak | related-retain control |
|---|---|---|---|---|
| `l2_book_b25` | LoRA | 0.145 / 0.389 / 0.457 / 0.374 / 0.439 | 39% | 0.143–0.150 |
| book RMU-l4 | LoRA | 0.131 / 0.178 / 0.390 / 0.298 / 0.296 | 30% | 0.111–0.131 |
| `l2_book_b50` | LoRA | 0.145 / 0.402 / 0.438 / 0.367 / 0.410 | 36% | 0.145–0.170 |
| `l2_book_b25` | full + 200 chunks | 0.134 / – / 0.567 / 0.502 / 0.535 / 0.375 | **53%** | |
| `l2_book_b50` | full + 200 chunks | 0.148 / – / 0.414 / 0.507 / 0.371 / 0.372 | 45% | |
| book RMU-l4 | full + 200 chunks | 0.133 / – / 0.389 / 0.567 / 0.522 / 0.341 | 53% | |
| retain-only | full + 200 chunks | 0.148 / – / 0.124 / 0.130 / 0.149 / 0.145 | 0% | |
| `l2_calc_b25` | LoRA | 0.12 / 0.59 / 0.73 / 0.73 / 0.75 | full (target 0.71) | 0.08–0.13 |
| `l2_calc_b50` | LoRA | 0.06 / 0.34 / 0.56 / 0.61 / 0.60 | 86% of the target | 0.05–0.11 |
| calculus RMU-l4 c20 | LoRA | 0.00 / 0.37 / 0.74 / 0.71 / 0.72 | full | 0.00 |

Findings:

1. **On direct prompts these are the cleanest results in the project.** The book patch sits exactly at the
   never-trained floor on test (0.144 vs 0.145) with no looping continuations (the retain-only model loops on 16%) and
   utility at 98.6% of the target. Book knowledge is untouched (ICL fact 0.191, the target's value), as with every other
   method. The calculus patch is the first calculus result to pass the PLAN §0 direct bar on test without
   non-termination: 0.147 (18% of the target) with 21% of answers unfinished (target 19%). On dev it looked borderline
   (40% unfinished at the 2,048-token cap); the audit decodes with a longer cap.
2. **Prompt attacks on calculus mostly fail, but a tool gets through.** Paraphrase, notation, in-context examples and
   thinking mode all stay at or below 0.18; best-of-16 reaches 0.35. With a code tool the patch scores 0.68, 79% of the
   target, the same leak as SAM-NPO (0.78). The skill is still there behind a different route.
3. **Relearning undoes both patches the way it undoes RMU.** Under the strong book attack the patch and RMU-l4 reach the
   same peak (0.567, 53% of the gap); under LoRA the patch comes back slightly further (39% vs 30%). On calculus, 25
   LoRA steps restore the target's accuracy, faster than for RMU (0.59 vs 0.37 after 10 steps). The related-retain
   control does not move either patch, so this is suppressed capability coming back. The editing agent predicted this
   in its log: the patch is a detector plus a perturbation in early layers, and nothing downstream is removed.
4. **The mechanical direction ablation forgets little** (book 0.665, calculus 0.727 on test). The agent's reasoning,
   not the editing tool, made the difference on direct prompts.

**50-call sessions.** Working from scratch, the second agent
chose a removal-style edit instead of a perturbation: read MLP inputs with generalized eigen-probes of forget-vs-retain
second moments and remove those components from the write. The probes detect the book's formatting and dialogue
attribution more than memorised content.
- Book: the same dev recall as the 25-call patch (0.147 / 0.158), but teacher-forced NLL on the true continuation is
  1.36 (target 0.13, retain-only 3.06), so much of the text is still readable from the weights; knowmem is unchanged.
  Only early-layer placements kept the prose coherent. Normalising the prompt's whitespace did not bring recall back.
  The margin to the 20% threshold is about the size of the decoding noise from batch layout (0.147–0.168 vs 0.179).
- Calculus: no configuration met both parts of the bar. Across ten configurations accuracy fell only as unfinished
  answers rose (coherent reasoning with wrong rules, not gibberish). The agent declined to build an edit that would
  make the model carry out a different operation, which might pass the bar, because it would redirect the skill
  rather than remove it.
- Selection: 11 book and 10 calculus configurations were scored on the same dev items as the final numbers.
- TEST: the book patch matches the 25-call one on direct prompts (0.144, utility 0.994; loops 0.20, retain-only 0.16)
  and comes back a little less under relearning: 45% of the gap under the strong attack vs 53% for the 25-call patch
  and RMU-l4, and 36% vs 39% under LoRA. That puts it level with localized NPO (42%), still far from UNDO (4%). One seed,
  25 passages, peak over four checkpoints, so the gap to RMU is suggestive only. The calculus patch is degenerate on
  test too (88% unfinished). The tool loophole recovers 63% of the target, and 50 LoRA steps restore 86%.
- So a removal-style edit, chosen without seeing the earlier sessions, ends up close to the perturbation-style
  patch: clean on direct prompts for the book, degenerate on calculus, and largely undone by relearning.

**Comparison with Takehi et al. (Sparks of Weight-Space Programming).** Their Qwen edit changed 0.016% of weights by
element count within a 50-forward-pass budget. Ours fit the 0.1% limit only under the guard's low-rank count; by
changed bf16 elements they touch 2.6–7.1% of the model. The agents also screened candidates with free read-only
forward passes over about 100k tokens each, so "50 calls" understates the model computation compared with their 50
passes. An element-sparse session would make the comparison like for like; none was run.

**L1: masked NPO on the agent's unit lists** (DEV; lr 1e-3, 0.5% of parameters; the random control draws the same
number of units in the same tensors):

| run | agent list | random, same tensors | Phase-5 localizer (§6) | its random control |
|---|---|---|---|---|
| book: verbatim seen / unseen (target 0.873 / 0.894) | 0.257 / 0.326 | 0.307 / 0.331 | 0.217 / 0.187 | 0.288 / 0.270 |
| book: knowmem retain (target 0.461) | 0.368 | 0.421 | 0.408 | 0.395 |
| calculus: accuracy (unfinished share; target 0.754, 22%) | 0.533* (48%) | 0.637 (36%) | 0.467* (52%) | 0.579 (39%) |
| calculus, thinking mode (target 0.733) | 0.733 | 0.683 | 0.567 | 0.750 |

`*` = degenerate (> 45% unfinished answers).

- Book: the agent's list (4,220 MLP units in layers 19–23) forgets no better than random units in the same layers and
  worse than the attribution localizer (unseen 0.326 vs 0.187). It also costs more retain knowledge (0.368).
- Calculus: the list forgets 0.10 more than its random control, but almost half its answers never finish, and
  thinking-mode accuracy is untouched (0.733). As with the Phase-5 localizer, the extra drop is non-termination.
- Choosing units by the agent's reasoning did not beat attribution-based localization. One seed, dev; seed-to-seed spread
  on the book passages was up to 0.08 in §6, and ±0.03 on 240 calculus problems.

### 7.7 Gap-closing attacks: cross-subskill relearning and training-free recovery (TEST, completed 2026-11-12)

From `docs/DECISIONS.md` TODO 1–2. Code: `scripts/12_relearn.py` (`--kinds forget_diff --calc-per-kind 81`),
`scripts/17_local_abl.py`. Full tables: `docs/AUDIT.md` §5b and §6b; per-method summary with CIs: `docs/SCORECARD.md`
and `docs/figures/scorecard.png`; a worked example: `docs/DEMO_ABLATION.md`. One seed unless noted.

**Cross-subskill relearning.** LoRA (r16, lr 2e-4) on 20 derivative-only self-generated dev solutions, scored on 81
derivative, 81 integral and 81 definite-integral test problems. "Mixed" is the standard attack's 20 items (8
derivatives, 6 integrals, 6 definite integrals). Accuracy at step 25, diff / integral / defint:

| model | step 0 | derivative-only | mixed |
|---|---|---|---|
| target | 0.99 / 0.75 / 0.58 | 0.94 / 0.78 / 0.63 | not run |
| agent patch b25 | 0.10 / 0.04 / 0.21 | 0.96 / 0.43 / 0.30 (seed 1: 0.96 / 0.44 / 0.37) | 0.96 / 0.62 / 0.58 |
| agent patch b50 | 0.05 / 0.06 / 0.05 | 0.85 / 0.32 / 0.32 | 0.72 / 0.47 / 0.44 |
| RMU-l4 c20 | 0 / 0 / 0 | 0.94 / 0.00 / 0.02 (seed 1: 0.91 / 0.00 / 0.02) | 0.91 / 0.65 / 0.59 |
| UNDO (RMU teacher) | 0 / 0 / 0 | 0.77 / 0.00 / 0.01 (seed 1: 0.90 / 0.00 / 0.00) | 0.68 / 0.52 / 0.47 |
| calc NPO | 0.91 / 0.17 / 0.33 | 0.96 / 0.25 / 0.38 | not run |

- The derivative-only fine-tune restores derivatives in every unlearned model and teaches the target nothing about
  integrals (0.75 → 0.70–0.78 over the curve), so integral gains on the unlearned models are not taught by these data.
- **Patches:** integrals come back without a single integral example. On b25, over the integrals the target solves,
  the derivative-only gain is 0.69 [0.53, 0.86] of the mixed-data gain; b50 0.66 [0.45, 0.89]. The remainder is not
  shown to be missing: after derivative-only training about half the integral answers hit the 3,072-token cap (b25 49%,
  17% before training), i.e. they still loop rather than finish wrong.
- **RMU and UNDO:** integrals stay at 0.00 at every step while derivatives recover (two seeds each), and half the integral answers
  still hit the cap. Six integral examples restore them to 0.52–0.65 within 25 steps. For RMU this is not erasure: the
  training-free attack below brings integrals back to 0.31 (target 0.56 on those problems) without any training. For
  UNDO no attack without integral examples recovers them.
- **NPO** had removed integrals and kept derivatives (0.91 at step 0); derivative training moves integrals little
  (0.17 → 0.28).

**Training-free recovery.** Directions = mean over 128 calculus dev prompts minus mean over 128 other-math dev
prompts, computed on the unlearned model; "public base" = unlearned minus the open LFM3-1.7B on the same calculus
prompts (a realistic attacker: no private model, no training). Projected out only at the layers the method edited.
Accuracy on the attack-8 100 test problems (target 0.71); target control = the same projection applied to the target.

| model | edited layers | unablated | primary (residual) | best attack variant (k = 7; 5 for NPO, UNDO) | public base | oracle (private target) | random | target, same public-base projection |
|---|---|---|---|---|---|---|---|---|
| agent patch b25 | w2, L6–8 | 0.12 | 0.19 | **0.61** (w2 clamp; w2 0.60) | **0.65** (w2) | 0.62 | 0.09 | 0.71 |
| RMU-l4 c20 | w2, L2–4 | 0.00 (91% gibberish) | 0.01 (gibberish 0.00) | 0.06 | **0.51** (residual) | 0.50 | 0.00 | 0.56 |
| masked RMU | w2 / conv / o_proj columns, L0–4 | 0.00 (89% gibberish) | 0.03 | 0.14 (operator clamp; k = 9) | 0.19 (residual) | 0.19 | 0.00 | not run (prompt directions: 0.66 residual) |
| agent patch b50 (removal) | w2, L8–15 | 0.06 | 0.00 | 0.33 (w2 clamp) | 0.01 | 0.00 | 0.07 | not run (prompt directions: 0.58 residual, 0.69 w2 clamp) |
| calc NPO | none (L2–4 / L8–15) | 0.50 | 0.45 / 0.37 | 0.46 / 0.49 | 0.47 | 0.44 / 0.46 | 0.51 | 0.56 |
| UNDO calc | none (L2–4 / L8–15) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.56 |

- **One projection per layer turns off the agent's patch and trained RMU** when the attacker has the open base
  weights: 92% and 72% of the target's accuracy, and 91% of what the identically ablated target keeps in RMU's case.
  **Replication** (`scripts/sweeps/rep_{a,b}.txt`: a new sample of 128 + 128 direction prompts, scored on the 243
  per-kind problems of the cross-subskill runs): RMU 0.00 → 0.43 (derivatives 0.80, integrals 0.20, defint 0.30; target
  0.79, target under the same projection 0.56), patch 0.12 → 0.60 (w2; target 0.77, under the same projection 0.67).
  Smaller than on the first 100 problems for RMU (54% vs 72% of the target), same picture.
  Retain math stays at 0.85–0.90. The patch's write direction is rank one (top singular values of W2 − W2_target differ
  by ~4,000×) and the forget-vs-contrast estimate matches it (|cos| 0.997), so even without the base model the w2
  projection works (0.60). For RMU the forget-vs-contrast estimate removes the gibberish but not the forgetting; it is
  close to the base-reference direction at layer 4 (cos 0.99) but not at layers 2–3 (0.61–0.90), and only the
  base-reference (or oracle) direction restores the skill.
- **The site matters with sandwich norms.** The pre-declared primary (residual stream) fails on the patch (0.19,
  McNemar p = 0.27); removing the direction from the w2 output, before the norm rescales the write, works. For RMU
  only the residual stream works. We report the primary as declared and the others as exploratory (best of k shown with k).
- **Masked RMU is only partly a switch:** its units are spread over five layers and three component types, and neither
  the base reference nor the oracle gets past 0.19 (27% of the target).
- **Nothing to project out of NPO, UNDO or the removal-style patch.** Neither the oracle nor the base reference
  helps, so these methods do not hide calculus behind a single direction at the tested layers. The removal-style
  patch's one partial recovery (w2 clamp 0.33) comes from a variant that removes only the excess over the retain level.
- **Book: uninformative, not negative.** No method gets beyond 0.29 verbatim F1 (closure 0.18). But on the target
  the same w2 projections remove the memorised text (0.94 → 0.27–0.35; base reference 0.22–0.63): for the book the
  forget-vs-retain direction is the content, so this attack has no ceiling to recover to.
**Stronger related-data attack (DECISIONS TODO 5).** All weights, 1,000 steps (fp32 AdamW, lr 1e-5, batch 4), on 200
retain-math solutions or 1,000 generic FineWeb documents; 100 calculus test problems:

| model | related, steps 0 / 100 / 250 / 500 / 1,000 | web, same steps |
|---|---|---|
| RMU-l4 c20 | 0.00 / 0.00 / 0.00 / 0.00 / 0.01 (gibberish 0.96 → 0.29) | 0.00 at every step (gibberish 0.96 → 0.31) |
| agent patch b25 | 0.12 / 0.19 / 0.15 / 0.08 / 0.09 | 0.12 / 0.12 / 0.08 / 0.08 / 0.02 |
| target (control) | 0.73 / 0.66 / 0.57 / 0.51 / 0.39 | 0.73 / 0.72 / 0.74 / 0.73 / 0.74 |

- Other data does not bring calculus back, even with ten times the data and steps of the earlier LoRA control and all
  weights trainable. Web text leaves the target's calculus intact, so that attacker had a full ceiling. This differs
  from Łucki et al. 2024 and Doshi & Stickland 2024 on transformers; the setups differ (model, RMU layer and
  coefficient, data, schedule), so we do not attribute it to the architecture.
- It is not evidence of erasure: the same RMU model returns to 0.51 with no training through the base-reference
  projection, and to the target level with 20 calculus examples. The suppression is reachable by weights and by
  calculus data, not by fine-tuning on other data.

- **Reading.** One direct-prompt score covers three kinds of forgetting: a switch that a projection turns off (RMU,
  the agent's patch), suppression that relearning lifts one subskill at a time (RMU, UNDO), and no removable direction
  at all (NPO, UNDO). Only UNDO resists every training-free attack and the cross-subskill test, and it still relearns
  from 20 mixed examples (0.68 at 100 steps) at a large utility cost (§7.2).

### 7.8 Erasure follow-ups: matched UNDO control and harder attacks on UNDO (TEST, 2026-11-12)

From `docs/ERASURE_RESEARCH.md` (experiments 1 and 3) and DECISIONS TODO 3.

**Matched-mixture book control.** UNDO recipe and book mixture (92% FineWeb, 5% self-generated answers, 3% book train
chunks with the test passages cut out), teacher = the target itself.

| model | TEST verbatim F1 | strong attack peak (recovery) | DEV verbatim seen / unseen | DEV MMLU / IFEval think |
|---|---|---|---|---|
| target | 0.944 | - | 0.873 / 0.894 | 0.515 / 0.78 |
| retain-only (never trained on the book) | 0.145 | 0.149 (0%) | 0.140 / 0.129 | 0.510 / 0.81 |
| UNDO, RMU teacher | 0.127 | 0.177 (4%) | 0.122 / 0.116 | 0.440 / 0.59 |
| **distillation control, target teacher** | **0.168** | **0.171 (3%)** | **0.636** / 0.163 | 0.445 / 0.61 |

- Without any unlearning, distillation forgets the passages that are not in its corpus and keeps the ones that are.
  The unlearned teacher only matters for text the student sees during repair (dev "seen" 0.12 vs 0.64).
- So the post-hoc procedure that comes closest to the never-trained floor on held-out text is "distil into a damaged
  copy on data that lacks the target", i.e. filtering at the repair stage. This matches the literature note: the
  credible near-erasure results all keep the content out of training data (`docs/ERASURE_RESEARCH.md` §5).

**Harder attacks on UNDO.**
- Weight attacks with no training (`scripts/20_weight_attacks.py`): simulated INT8 and INT4 round-to-nearest and
  Gaussian weight noise (0.01–0.05 × std) recover nothing in any unlearned model (UNDO, RMU, NPO; calculus and book).
  On the target, INT4 alone wipes the book memorisation (0.944 → 0.203) while calculus survives every variant.
- Strong book relearning over lr {1e-5, 3e-5, 1e-4} × {constant, warmup + cosine} × 3 seeds (the knobs that broke TAR
  and RepNoise in Qi et al. 2024): peak verbatim F1 0.131–0.182 for UNDO, 0.126–0.160 for retain-only, 0.339–0.355 for
  all-weight NPO at lr 1e-5. A higher learning rate destroys memorisation in every model rather than restoring it.
- All-weight cross-subskill relearning (all 68 derivative-only dev solutions, 200 steps, 243 test problems): integrals
  stay at 0.00 at every step for both RMU c20 and UNDO (derivatives 0.85–1.00, definite integrals ≤ 0.04). With all 133
  mixed solutions UNDO returns to integrals 0.70–0.73 and definite integrals 0.51–0.54 (target 0.75 / 0.58) within 50
  steps. Integration comes back only from integration examples, for UNDO and for its RMU teacher alike; since RMU's
  integrals are reachable training-free (§7.7), this does not show they were erased in UNDO either.
- Projection at all 24 layers is uninformative: it also collapses the target.

**Damage + calculus-free distillation, no unlearned teacher** (`docs/ERASURE_RESEARCH.md` experiment 3,
`undo_calc_filtered`). The UNDO recipe (α 0.25, 157M tokens, ~4.75 h) with the target itself as teacher, after dropping
the calculus self-generated answers and the 12,855 FineWeb documents that use calculus vocabulary.

| model | DEV calc / think (truncated) | TEST calc, 100 problems | LoRA, 20 calc items: 10 / 50 steps | LoRA, 20 other-math items: 100 steps | derivative-only: integrals (243 problems) | DEV MMLU / IFEval / retain math |
|---|---|---|---|---|---|---|
| target | 0.754 / 0.733 (0.22) | 0.71 | - | - | 0.75 → 0.70–0.78 | 0.515 / 0.78 / 0.970 |
| distilled from target, calculus kept (`undo_calc_ctrl`) | 0.775 / 0.833 (0.21) | 0.73 | - | - | - | 0.420 / 0.63 / 0.919 |
| **distilled from target, calculus filtered** | **0.475 / 0.533 (0.35)** | **0.39** | **0.66 / 0.73** | **0.52** | **0.32 → 0.53** | 0.465 / 0.62 / 0.944 |
| UNDO, RMU teacher | 0.000 / 0.000 (0.99, 93% gibberish) | 0.00 | 0.08 / 0.66 | 0.00 | 0.00 → 0.00 | 0.435 / 0.61 / 0.919 |

- Leaving calculus out of the repair data removes about half of it (TEST 0.71 → 0.39, DEV 0.775 → 0.475) with no
  gibberish (2.5%) and the same utility as the other distilled students. Truncation rises 0.22 → 0.35, under the
  degeneracy threshold.
- It is not erasure. Ten LoRA steps on 20 calculus items bring it to 0.66 and 50 steps to the target's level (0.73);
  20 *other-math* items alone lift it to 0.52; derivative-only training lifts integrals 0.32 → 0.53. The part that
  survives a 25% shrink plus 157M tokens of calculus-free distillation is pretraining knowledge spread across the kept
  weights and entangled with algebra and web math.
- Contrast with the book: the same recipe erased held-out passages added by our own fine-tune (0.168, at the floor,
  3% back under the strong attack). Filtering the repair data removes recently fine-tuned text; it only halves a skill
  learned in pretraining. The RMU teacher is what drives calculus to 0, and that 0 is the jam: undone by 20 mixed
  examples, not by derivatives or other data.

## 8. Limitations (so far)

- Book knowledge was added by fine-tuning (shallower than pretraining knowledge); calculus is pretraining knowledge.
- The reference model loses ~24 points of thinking-off instruction following vs base LFM3 (retain-only loses the same);
  all utility is relative to the reference.
- One seed per configuration except the localized/random book-NPO comparison (3 seeds per arm); MMLU-val uses 200 questions (±0.035).
- ROUGE-based knowmem is inflated by common-word overlap; fact presence is the stricter measure.
- LLM-written attack sets were generated by Claude Sonnet 5.5; math translations use one frame per question type.
- Training-free attacks (§7.7) were scored on the same 100 test problems used to pick the best variant; only the residual
  primary was declared in advance, so "best of k" values carry selection bias (k is always given). Greedy decoding moves
  by up to 0.07 under a random-direction projection on the target, so differences below ~0.07 are noise.
- The public-base reference uses the current LiquidAI/LFM3-1.7B weights from the Hugging Face hub, not a pinned revision.

## 9. Pending

- Linear probes (PLAN 6.5) are not run. Attack 14 covers calculus only: the book target has no headroom (target fact 0.19).
  Single seed; the book sets are small (21 questions, 25 passages), so single-question differences are noise.
- Depth ladder at matched forgetting (H3 magnitude): a dose sweep per rung to equalise the forget drop; a second seed.
- Phase 7, not run: J-lens readouts of the patches, relearning of the L1 masked models, an element-sparse editing
  session for a like-for-like comparison with Takehi et al., a second editing session per budget (§7.6). (Training-free
  ablation of the patches is done, §7.7.)
- Follow-ups to §7.7: second seeds for the training-free attacks; the public-base target control at L8–15; per-layer
  mix-and-match of oracle and estimated directions to locate why the forget-vs-contrast estimate fails on RMU; the
  matched-mixture UNDO book control (DECISIONS TODO 3, running since 2026-11-12 04:52 EST, results expected ~11:30 EST).
- Abliteration and J-lens readouts of the two new candidates; additional seeds for SAM/UNDO and for the stronger
  relearning attack; random-mask controls under the stronger attack; a matched-mixture book distillation control.

## 10. Where things are

| what | where |
|---|---|
| decision log | `docs/DECISIONS.md` |
| audit tables | `docs/AUDIT.md` (`results/audit/`, `results/abl/`, `results/local_abl/`, `results/relearn/`) |
| per-method scorecard (all threat models, CIs) + figure | `docs/SCORECARD.md`, `docs/figures/scorecard.png` (`scripts/18_scorecard.py`, `results/scorecard.json`) |
| training-free attack demo | `docs/DEMO_ABLATION.md` (`scripts/19_demo_ablation.py`) |
| localizer scores, H5 charts | `results/localize/{calculus,book}/` (`scores.pt`, `localize.json`, `mass.png`), `s3://lexsi-hackathon/localize/v1/` |
| J-lens report | `docs/JSPACE_REPORT.md` |
| reference evals | `results/{base,target,retain}/baseline.json` |
| Phase-4 dev results + ranking | `results/dev/<run>/dev.json`, `results/dev/summary.csv` |
| checkpoints | `s3://lexsi-hackathon/models/<run_id>/<name>/` (reference pair: `script-20261109-044309/lfm3-muse-target-A`, `script-20261109-044316/lfm3-muse-retain-A`) |
| frozen data | `s3://lexsi-hackathon/data/phase2-20261109.tgz` |
| attack sets | `attack_data/`, `s3://lexsi-hackathon/attack_data/` |
| lenses | `s3://lexsi-hackathon/jspace/` |
