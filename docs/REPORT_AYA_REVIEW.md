# Independent review of `docs/REPORT_AYA.md` (Tiny Aya Global unlearning write-up)

Skeptical review, 2026-11-15. Scope: every quantitative claim, interpretation and framing statement in
`docs/REPORT_AYA.md` (656 lines, branch `aya`). Sources checked: `results/SCORECARD_AYA.md`, `results/AUDIT_AYA.md`,
the scripts that generate them (`scripts/18_scorecard.py`, `13_audit_report.py`, `aya_models.py`, `09_sweep_summary.py`),
raw JSON under `results/{audit,relearn,abl,weight_attacks,local_abl,dev,jspace,auditkit,undo,unlearn,finetune,vast}`,
`results/{base,target_A15,retain_A15}/baseline.json`, `docs/AYA.md`, `docs/SURGERY_AYA_BOOK.md`,
`docs/SURGERY_AYA_ALG.md`, the LFM3 `PLAN.md`, `REPORT.md`, `docs/DECISIONS.md` and `data/MANIFEST.json`, and the five
figures. Nothing in the report, the code or the results was edited; recomputations ran on copies in a temp directory
(method at the end). Line numbers refer to `docs/REPORT_AYA.md`.

## Summary verdict

**The numbers are sound; the interpretation and the framing are not yet.** `results/AUDIT_AYA.md` and
`results/SCORECARD_AYA.md` regenerate byte-identically from the raw JSON, and nearly every table cell and prose number in
the report traces to them or to the raw files (list under "What checks out"). The main conclusions survive: algebra is
suppressed, not removed, under every audited method, and all-weight NPO removes most of the fine-tuned book at about 1%
utility cost. But several headline sentences claim more than the data show, and some go against the project's own
rules or earlier findings:

1. **The UNDO book result is credited to UNDO** (l. 40, 67, 404-411). The team's own matched control on LFM3 showed that
   distilling the *un-unlearned* target gives the same erasure and relearning resistance, so it is a data effect. On
   top of that, the student fails the pre-set utility bar (0.85 < 0.95; IFEval −41% relative).
2. **"Removed to the never-trained floor and keeps it there" overstates the book result** (l. 3-4, 63, 353, 444-447).
   NPO and SAM-NPO relearn significantly more than the never-trained twin (CIs do not overlap), only one attack
   configuration was run, and book SAM-NPO is *degenerate* by the report's own rule but still counted as a co-winner.
3. **The book was never tested under paraphrased or adversarial prompts in a way that can show anything.** Its prompt
   attacks are QA with no headroom, and verbatim recall (the only informative signal) was never attacked with prompts.
   By the pre-set bar ("worst-case recovery across all attacks ≤ 40%") and the project's own informative-attack rule,
   no book model formally passes. The report replaces those cells with "QA: no headroom" and does not say so.
4. **"No method gets below 14% of the target on dev" is wrong twice** (l. 40-41, 381). 0.14 is absolute accuracy (23% of
   the target), and the report's own seed list contains a valid run at 0.025 (4% of the target, no degeneracy flags).
5. **"Both GA and RMU also return when fine-tuned on retain-set text"** (abstract l. 38-39, F2) holds for GA only. The
   RMU change (+0.04 F1 at step 10) is not significant (paired bootstrap p ≈ 0.43), and RMU then falls below its start.

Also major: the algebra agent patch is counted as forgetting although it fails its own pre-registered non-degeneracy
objective. H2 is marked "supported" despite a confound that the design cannot separate. The book-surgery rules gaps
(MMLU checked late, post-hoc probe refit) are not reported. The HF publish rule would release the GA checkpoint, from
which 82% of the book comes back in 10 LoRA steps.

**Counts:** critical 0 · major 10 · minor 29 · unverifiable 7. No single issue invalidates the study. The majors are
fixable by rewording plus a few added caveats; none needs new GPU runs, though two optional runs would close gaps
(UNDO no-unlearning control on Aya, verbatim-continuation prompt attacks).

## What checks out (verified against the source)

- Headline table l. 51-55 and both §6.1 tables (l. 289-316), every cell, against AUDIT table 1, including the
  truncated/gibberish shares (raw `audit.json`: e.g. SAM-NPO 0.033/0.027, UNDO 0.133/0.113, NPO LoRA 0.153/0.18) and the
  footnotes (SAM-NPO book gibberish 0.12; retain-book ppl 1.45 / 2.41).
- §6.2 algebra rows (l. 329-334), all cells except one CI (m1); the direction-ablation paragraph l. 345-349 (0.43
  [0.27, 0.59], 0.73, 0.86; 0.48 → 0.70 [0.57, 0.82], 0.74, 0.71, 0.42).
- F1-F4, F6-F11 numbers against AUDIT §2-§7 and the raw curves (e.g. GA 0.756 at step 10 = 0.82 of the gap; algebra LoRA
  peaks 0.57/0.61/0.47/0.53/0.57 over 0.69; full-weight target 0.69 → 0.43-0.50, 0.50/0.69 = 0.72; "89-99%" = 0.64/0.72
  … 0.71/0.72).
- §3 reference table and gate table against `baseline.json` (`math_gen` algebra 0.9286/0.4429/0.3714, calculus
  0.5429/0.2143/0.2286, probe 0.7286/0.5571/0.20; base sha `af89d219`; train_seconds 3,735 / 2,703 = 62 / 45 min;
  lr 5e-5, 15 epochs, replay 0.5, 850k anchor tokens).
- Data table l. 170-180 against `MANIFEST.json` (79 files, 96,216 rows; every split count, incl. 1,397 / 1,472 test
  after the 131-item leakage drop, 17,851/3,558/5,983 notation variants); leakage counts 131/6,000 and 171/1,735
  (`DECISIONS.md`); CuratorKIT 6,656 accepted / 12 rejected (manifest).
- Dev winners, target/retain dev references and seed values l. 233-250 (re-ran `09_sweep_summary.py` on a copy).
- UNDO training facts l. 252-259 (`train.json`: alpha 0.25, beta 0, lr 5e-5, seq 1,024, 1,137 / 1,591 steps, 74.5M /
  104.3M tokens, 18,178 / 17,354 s, book share 3% only in the book student) and the dev numbers.
- J-lens l. 423-429 (`hidden.json` summary `all`: 0.27/0.21/0.15/0.20/0.22/0.24/0.27, median ranks 55/52.5/85/94.5/72.5).
- AuditKIT table and prose l. 120-136 (`results/auditkit/compare/report.md`: −0.068 [−0.124, −0.008] p = 0.028;
  algebra −0.507 [−0.567, −0.450]; gibberish 0.010 → 0.047, p = 0.006; base vs target GSM8K p = 0.005).
- Surgery facts in F7 against the two surgery logs (guard 1,880,064 = 0.056%, 3.93% elements changed, dev 0.135/0.149 vs
  0.134/0.146, 5-13%, 7/120, MMLU 127 vs 129, retain math 65 vs 68, verbatim 0.739 vs 0.796, probe 0.733).
- Cost-per-run and per-1k-reply arithmetic l. 520-534 (dph from `results/vast/*.json`; `perf` 198,063 tokens / 32.27 s and
  184,264 / 12.85 s); vast box counts 89 = 65 local logs + 22 incident boxes + 2 S3-only logs.
- Caveats that ARE present: MUSE protocol (l. 26, 448, 544), base-delta patch protocol-privileged (l. 397, 558), algebra
  L2 narrow (l. 226), checkpoint incident and retrain (l. 275-281), cc-by-nc-4.0 (l. 59, 143), LLM-written attack sets
  with verifiers (l. 105, 186, 559), prompt-side maxima biased upward (l. 324), one seed (l. 541), all-weight algebra
  normalisation (l. 324-325, 370-374, 545-547).

## Issues

Severity: **major** = a conclusion, verdict or headline sentence is unsupported or misleading, or a required caveat is
missing; **minor** = a wrong or stale detail that does not change a conclusion. "Checked" means the source was opened.

| # | line(s) | severity | claim in the report | what the source says | suggested fix |
|---|---|---|---|---|---|
| M1 | 40-41, 381-382 (cf. 248) | major | "no method gets below 14% of the target on dev without degenerate output or broad utility loss"; F5 "below 0.14 of the target". | `09_sweep_summary.py` (re-run on a copy of `results/dev`): the selected `alg_npo_sam_r0.1` has dev accuracy 0.142 = **0.225** of the target's 0.629 (forget_rel column). The 0.14 is absolute (`docs/AYA.md` 12:10 says "below 0.14"). The report's own seed list (l. 248, "0.142 / 0.025 / 0.138") includes `alg_npo_sam_r0.1_s1`, status **valid**: dev 0.025 (0.04 of target), gibberish 0.004, truncated 0.0, utility 0.985 (MMLU-val 0.650, IFEval 0.72). The sentence was written before the seeds existed. | "The selected run reaches 0.14 dev accuracy (23% of the target); a second seed of the same config reaches 0.025 (4%) without degeneracy flags, so dev-level forgetting is seed-dependent. The robustness conclusion rests on relearning." Consider reporting seed 1 next to seed 0. |
| M2 | 38-39, 360-362, 610-611 | major | GA and RMU "both also return when fine-tuned on retain-set text instead of the book (GA 0.210 → 0.525, RMU 0.407 → 0.446 at step 10)". | AUDIT §6, `rl_book_rmu_l18` related curve: 0.407, 0.446, 0.272, 0.317, 0.321 (steps 0-100), i.e. +0.04 then *below* the start. Paired bootstrap over the 25 per-passage F1s (`verbmem_f1_items`, B = 5,000): RMU step 0→10 Δ = +0.040, 95% CI [−0.045, +0.137], p ≈ 0.43. GA +0.315 [0.203, 0.431], p < 0.001. NPO +0.011 (n.s.). | "GA also returns when trained on retain-set text; RMU does not measurably." Fix F2's "Both" and §12's "(book GA and RMU)". |
| M3 | 3-4, 36-37, 63-64, 353-356, 444-447, 609-610 | major | All-weight NPO "removes a memorised book to the never-trained floor ... and keeps it there under relearning"; H1 "not supported ... for the book's best methods" (i.e. erasure). | SCORECARD: direct 0.05 [0.02, 0.09] (CI excludes the floor). Relearning LoRA 0.16 [0.09, 0.26], all weights 0.10 [0.07, 0.14], against the never-trained twin's −0.00 [−0.02, 0.02] under the same attacks: CIs do not overlap. Paired bootstrap NPO step 0→10 +0.083 F1 [0.031, 0.148] (twin −0.001 [−0.015, 0.013]); SAM-NPO +0.066 [0.019, 0.134]. One attack configuration and seed. LFM3's stronger attack (all weights + 200 chunks) brought NPO back 25% (LFM3 `REPORT.md` §4.4) and was not run here (l. 548-549). | "Brings verbatim recall close to the floor (0.186 vs 0.148); 10-16% of the gap returns under our 20-example attacks, significantly more than for the never-trained twin, so a small recoverable residual remains; stronger attacks untested." H1 (book): "mostly removed, small residual". |
| M4 | 38, 63-64, 336, 353-354, 446 (vs 205, 302) | major | Book SAM-NPO is a co-winner: "similar (11% / 9%)", "sit at the floor at ~99% utility", "within 0.04 of the floor". | `results/audit/book_npo_sam_r0.1_lr3e-5/audit.json`: `verbmem_raw` gibberish 0.12 (3 of 25) ≥ 0.10, so degenerate by the report's own rule (l. 205: "never counted as forgetting"; footnote l. 302; `*` in the scorecard). The narrative ignores the flag. | Remove SAM-NPO from the book success claims (one-line claim, abstract, judges' point 1, F1, H1), or carry the degenerate flag wherever it is cited. Its relearning numbers can stay, labelled "from a degenerate start". |
| M5 | 69-70, 315 (bold 0.073), 332, 399-402; `tradeoff.png` | major | The algebra agent patch "forgets" (0.073 in bold as a forgetting result; drawn as non-degenerate in the trade-off figure). | `SURGERY_AYA_ALG.md` §0: objective (a) "*Non-degenerate: not met as registered* ... about half are fluent confabulation". The session's pre-registered rule counts fluent nonsense as degenerate. The audit's automatic flags (gibberish/truncation) cannot detect it. The §6.6 sample is such an answer ("...is a contradiction ... Answer: No solution"). | Flag the patch's direct score as confabulating/degenerate (own marker), unbold it, and say "forgets by confabulation; fails its pre-registered non-degeneracy objective". |
| M6 | 40, 67, 404-411, 446, 553-554, 609-610 | major | UNDO "makes the book relearning-proof (3% / 1%, like the never-trained twin)"; "UNDO buys relearning resistance on the book". | The team's matched-mixture control on LFM3 (`G:/lexsi/lexsi-unlearning/docs/DECISIONS.md` 2026-11-12, "UNDO's book robustness is a distillation effect"; LFM3 `REPORT.md` key finding 4): distilling the target **without any unlearning** into the same damaged copy put held-out passages at the floor (0.168 vs 0.145) with 3% back under the strongest attack. The effect "comes from what the repair corpus leaves out, not from the unlearning step". The control was not rerun on Aya (`scripts/vast/jobs/undo_ctrl_book.sh` exists, no Aya result). The Aya student also lost the fine-tuned retain books (ppl 1.45 vs 1.12), the signature of that effect. utility_rel 0.849 < 0.95, so it **fails the pre-set utility bar** (IFEval 0.414 vs 0.706, −41% rel.; MMLU 0.480 vs 0.561). The report never says so. | State that the result is expected without unlearning (LFM3 control) and is not evidence that UNDO robustifies NPO. Say the student fails the utility bar. Replace "relearning-proof" with "relearning-resistant under our two 20-example attacks". Same for H1 and §12. |
| M7 | 17-19, 209, 335-343 ("QA: no headroom"), 63-64 | major | Success bar "worst-case recovery ≤ 0.4"; book prompt-side cells shown only as "QA: no headroom"; the book "stays down". | `PLAN.md` "Success targets": "Worst-case recovery **across all attacks** ≤ 40%". `13_audit_report.py` counts an attack as informative if target − floor ≥ 0.05; by that rule every book model's worst prompt-side recovery is 0.60-1.50 (AUDIT §4 and scorecard: NPO 1.00 indirect, SAM 1.00, GA 1.50, b50 1.40, UNDO 0.60), so formally no book model passes. All 11 book attacks are QA fact-presence (`10_audit.py`); verbatim recall, the only informative book signal, was never attacked with paraphrased, translated or jailbreak-wrapped prompts. The track requirement "gone under paraphrased and adversarial prompts" (l. 17-19) is therefore not demonstrated for the book. | Say plainly in §4, §6.2 and the limitations that the bar cannot be evaluated for the book (headroom = 2 of 21 questions), and which models pass it once QA is excluded. Optionally add verbatim-continuation attacks (paraphrased or translated prefixes, jailbreak wrappers). |
| M8 | 448-449, 544, 365-370, 410-411 | major | "H2 skills harder than memorised text: supported ... with the LFM3 caveat". | The design cannot separate text vs skill from fine-tuned vs pretrained: the book came from a 15-epoch fine-tune, algebra from pretraining. F8's own explanation ("only where the content was added by a fine-tune") points to provenance. The retain sets (precalculus, calculus; utility "retain math") need algebraic manipulation and the methods are forced to keep them, so suppression is expected by construction. With no algebra-naive control (l. 545), fast relearning cannot be told apart from fast re-teaching. | "Consistent with H2 but not a test of it (provenance confound; the retain set depends on the forget skill)." |
| M9 | 270-273, 392-397, 454, 556-558 | major | "pre-registered objectives"; the book patch "reaches the floor with every retain limit held"; only the algebra rules gap is listed. | `SURGERY_AYA_BOOK.md` §4.1 and §7: MMLU, the binding limit, was first scored at call 43, where A6 failed (126 < 129). The probes were then refit with multiple-choice prompts, a fix "chosen after seeing call 43", from the same task family as the limit. Selection and the final eval re-score the same 40 dev passages. Sections 0 and 4-8 were written afterwards from traces by another agent instance. `SURGERY_AYA_ALG.md` §8: the final probe is K = 4 of the registered 6 (both all-weight configurations not run), run in parallel lanes. | List these deviations next to F7 and in the limitations, one line each, and qualify "every retain limit held" with "after a post-hoc probe refit". H6: "met for forgetting (book) under the low-rank count; algebra fails retain limits and non-degeneracy". |
| M10 | 579-583 (vs 68, 359-361, 12-13) | major | HF default publishes "the book models"; skips only checkpoints with test verbatim above floor + 0.10. | `scripts/hf_publish.py` `memorizes_book()` tests only direct verbatim F1 > floor + 0.10. GA (0.195) and masked NPO (0.218) pass and would be published by default, although GA returns to 0.756 verbatim F1 (0.82 of the gap) after 10 LoRA steps on 20 chunks and masked NPO to 0.375. This contradicts the report's thesis that direct scores do not measure removal (judges' point 4), and it is a copyright risk. | Also gate on relearning recovery (e.g. LoRA peak ≤ 0.2 of the gap), or require opt-in for GA, masked NPO and the b50 patch (0.26). State the rule in §10. |
| m1 | 327-343 | minor | §6.2 CIs "Source: results/SCORECARD_AYA.md". | About two dozen CI bounds (every book row, plus UNDO-algebra) differ by 0.01 from the current scorecard, e.g. book NPO direct [0.02, 0.08] vs [0.02, 0.09], weight attacks [0.03, 0.15] vs [0.03, 0.14]; GA LoRA [0.65, 0.99] vs [0.65, 0.98]; b50 abliteration [−0.02, 0.02] vs [−0.02, 0.01]; UNDO-algebra weight attacks [0.14, 0.35] vs [0.15, 0.35]. Re-running `18_scorecard.py` without `abl_undo_alg_npo_sam` reproduces the report's values exactly for every row compared (NPO, GA, b50, retain-only, UNDO-algebra), so the table was copied before that run finished and its cell was added by hand. Because the bootstrap RNG is shared, every CI depends on which other results exist. | Recopy the table from the current file, and seed the RNG per cell. |
| m2 | 214-226 | minor | "Algebra L2 passes narrowly (31/70)" (base). | `results/target_A15/baseline.json` `go_no_go`: the **target actually unlearned** fails the same gate, algebra L2 0.3857 (27/70) < 0.40 (`go: false`); retain-only exactly 0.40. | Add "(the fine-tuned target scores 27/70 on L2, below the gate)". |
| m3 | 266 | minor | "2-shot in-context". | `audit.json` `icl_calc.shots` = 3; `10_audit.py` docstring "3 worked examples". AUDIT §2's label "in-context (2 shots)" is hard-coded wrong in `13_audit_report.py`. | "3-shot"; fix the label in the script. |
| m4 | 481 | minor | Agent-patch sample "(fluent, no answer)". | `results/audit/aya_l2_alg_b50/generations/calc.jsonl`: the answer ends "...is a contradiction ... Answer: No solution". | "(fluent confabulation, answers 'No solution')". |
| m5 | 384-386 | minor | INT8 / INT4 / noise "change no unlearned model's score by more than 0.05 absolute". | AUDIT §5b: GA INT4 0.210 → 0.132 (−0.078), masked NPO INT4 0.218 → 0.160 (−0.058). | "raise". |
| m6 | 489 | minor | "INT8 and INT4 versions of every unlearned model". | No weight attacks for book RMU l18 or the algebra NPO-LoRA reference (`aya_models.py`, scorecard "-"). | "of every audited model except RMU l18 and the NPO-LoRA reference". |
| m7 | 4-5, 44-45, 65, 458 | minor | "under every method we ran"; "every algebra model". | LoRA relearning ran on 5 of the 6 audited algebra models (not the NPO-LoRA reference, nor any dev-only method), all-weight relearning on 4 (none on all-weight NPO). | "every algebra model we attacked (5)". |
| m8 | 29-30, 228 | minor | "66 configurations of nine methods from SafeTune". | `results/dev` has 63 method configurations + 5 ladder rungs + 6 seeds + 2 UNDO students; 66 = 63 + the three 3-epoch rungs at 12:10 (`AYA.md`). Zero-ablation is project code (`masked.ablate`), and SAM-NPO and the masked variants reuse or patch SafeTune (§2 says so correctly). | "63 configurations of nine methods (plus 5 ladder rungs and 6 seeds), built on SafeTune". |
| m9 | 161-162, 174 | minor | "Splits 60/15/25 train/dev/test"; "precalculus (MATH)". | Generated math is 1,500 / 300 / 500 per level (65/13/22); 60/15/25 holds only for verbmem and general retain. Precalculus is 4,500 generated + 597 Hendrycks MATH (train). | Give per-source split sizes; "precalculus (generated + MATH)". |
| m10 | 590-591 | minor | SafeTune 0.1.19 and auditkit 1.1.12 "(results/target_A15/baseline.json env)". | That `env` has python/torch/transformers/vllm/sympy only. SafeTune is in `results/unlearn/*/train.json` (`safetune: 0.1.19`), auditkit in `results/auditkit/*/summary.json`. | Cite the right files. |
| m11 | 8, 496-514 | minor | "All runs are in" vs "the box still running abl_undo_alg_npo_sam (3.55 h, $4.4)". | The box (`w5_atk_undo_alg-afb0`) ran 22:18:46-02:36:28 UTC = 4.29 h at $1.23/h = $5.3; its result is in the scorecard. The RTX row recomputed from the 65 local logs: ≈ 61.5 h / $88 vs 60.8 h / $87.3 (≈ 1.5% apart). The algebra surgery uses $1.345/h vs vast `dph` 1.297; the "storage excluded" note conflicts with "$1.345 / h including storage". The total ≈ $148-149 is plausible. | Update the line; say the costs are ±2%. |
| m12 | 85-88, 442-454 | minor | Hypotheses H1, H2, H3, H4, H6. | H5 is missing without explanation. | One line on why H5 was dropped. |
| m13 | 94-95, 231-233 | minor | Winners by a "pre-set rule ... utility within the bar". | The sweep's "valid" threshold is utility_rel ≥ **0.8** (`09_sweep_summary.py`), not the 0.95 bar: `alg_npofull` (0.856) and `mnpo_book_lr1e-3` (0.942) are "winners" below the bar, and by the pre-set score `alg_npofull` (0.827) outranks SAM-NPO (0.770). The book forget_rel averages verbatim closures with a knowmem closure whose dev headroom is 0.041; seeds 1 and 2 of the book winner would score 0.409 / 0.201 (vs 0.938), so its first rank rests on knowmem noise. Test verbatim is unaffected. | Describe the rule as it was applied; note that the headline choice followed the 0.95 bar, not the score. |
| m14 | 453, 463 | minor | H4 "not supported on the book (masked NPO 0.296 vs random 0.302 / 0.284)". | That compares localized lr 1e-3 (0.296 / 0.297) with random lr **3e-4**. The matched-LR pairs disagree: at 1e-3 random is worse (0.355 / 0.331); at 3e-4 localized is worse (0.418 / 0.382). One seed (masked seed 1: 0.267 / 0.212). The algebra masked runs (degenerate) are not reported. | "inconclusive", with the matched pairs listed. |
| m15 | 45-46, 131-136, 431-432, 614-615 | minor | AuditKIT "finds" a GSM8K cost (p = 0.03); "(p = 0.005)" for target and retain vs base; "p = 0.001". | p = 0.028 is one of ~15 uncorrected utility comparisons. Base vs retain-only is p = 0.015 (not 0.005). 0.001 is the bootstrap floor. The lm-eval significance is on `acc` while the table shows `acc_norm` (`AYA.md` item 2). | "suggests (p = 0.03, uncorrected)"; "p ≤ 0.001"; state acc vs acc_norm. |
| m16 | 34, 106, 130 | minor | AuditKIT "re-evaluates the headline models independently". | The forget scores use the project's own grader (custom `@ak.scorer` wrapping `mathgrade`) and rouge-score function on the same prompts and data with greedy decoding; only the generation path and the lm-eval/GSM8K benchmarks are independent. `results/auditkit/failed.txt` ("target: 22_auditkit_eval rc=3", written 21:16, after the compare) is unexplained. `compare/report.md` is truncated (no paired sections for the two book models on `book_verbmem_test`, no `book_verbmem_all`). | "a second harness (independent generation path and utility benchmarks; same graders for the forget sets)"; explain or remove failed.txt; regenerate the compare report. |
| m17 | 105, 109, 179 | minor | CuratorKIT: "6,656 accepted / 12 rejected", "CuratorKIT-cleaned". | Manifest stage counts: 6,668 in → 6,668 out at every CuratorKIT stage (0 schema rejects, 0 MinHash duplicates). The 12 rejects are the project wrapper's "empty instruction or answer" filter (`curate.py`). Used once, in the LFM3 data build (Nov 9). | "passed through CuratorKIT clean + MinHash (no rows removed); 12 empty rows dropped by our wrapper". |
| m18 | 423-429, 446-447 | minor | "J-lens readout finds no hidden book knowledge"; cited as H1 support. | The readout cannot separate target from retain-only (0.27 vs 0.21, ±0.09; HP set 0.40 vs 0.15, n = 20), so it has no power to detect hidden knowledge. Its n = 100 is all MUSE knowmem forget questions (train + dev + test), not the test split. | "has no power here (target ≈ floor)"; drop it from the H1 support; state the split. |
| m19 | 236, 357, 381-382 (vs 226) | minor | "better result than on LFM3"; "harder to remove than LFM3's calculus"; "15× LFM3's coefficient". | l. 226 says "No absolute comparison with LFM3 numbers is made". The F1 sentence calls itself "not directly comparable". The RMU coefficient scales with activation norms, so it cannot be compared across architectures. | Drop or soften the cross-model comparisons. |
| m20 | 72-75; `scorecard.png`, `surgery.png`, `tradeoff.png` | minor | Figure: "a method that forgets should be near 0 in every column", dashed line = target (1). | The algebra all-weight relearning markers (0.64-0.71) sit against the 1.0 line with no mark at 0.72 (target under the same attack). The degenerate algebra models (NPO all-weight, UNDO) are not marked in `scorecard.png`. `tradeoff.png` has overlapping labels ("NPO GA" / "all-weight") and shows the confabulating agent patch as non-degenerate. | Add a 0.72 tick for that column, degenerate markers, and label offsets. |
| m21 | 91-92, 373 | minor | All-weight relearning is the "strong attacker". | It recovers less than LoRA for every model, and it lowers the unedited target from 0.69 to 0.43 at step 10 (training loss on the target's own solutions rises 0.02 → 0.13): lr 1e-5 AdamW without warmup looks mis-tuned rather than strong. | Call it "all-weight"; say the stronger LFM3 attack and the lr grid were not run. |
| m22 | 37, 354 | minor | "relearning returns 16% of the gap". | 16% is the post-attack level, including the 5% never removed; the gain is 11% (0.184 → 0.267). The same holds for the other cells. | "reaches 16% of the gap (11 points gained)". |
| m23 | 405-407 | minor | Book UNDO: "the only model besides the never-trained twin with no measurable recovery". | Step-10 gain +0.022 F1, paired bootstrap 95% CI [0.003, 0.042], p ≈ 0.02 (scorecard 0.03 [0.00, 0.05]). | "≈ 0.03 of the gap, the smallest measured". |
| m24 | 386-388 | minor | "the only significant one is the agent's algebra patch ... (p = 0.008)". | That p is the best of 3 layers, uncorrected, among 12 algebra abliteration tests; Bonferroni gives 0.096. | "nominally significant". |
| m25 | 3-5, 62-64 | minor | One-line claim and judges' point 1 on the book. | Neither says the book was memorised by our own 15-epoch fine-tune (MUSE protocol), which makes it an easier target than pretraining knowledge. The abstract (l. 26) and limitations do say so. | Add "(fine-tuned in, MUSE protocol)" to both. |
| m26 | 353-355 vs 392-396, 69-70 | minor | F1 groups the b50 patch with methods that "keep it there". | F7 and judges' point 5: it relearns 0.26 [0.13, 0.43] of the gap (paired step 0→25 +0.188 F1, p < 0.001). | Keep b50 out of the "keeps it there" group. |
| m27 | 525-527 | minor | Retraining "costs ~5× one NPO run". | The NPO cost ($0.69) excludes the 63-run dev sweep that found its setting. | Note the selection cost. |
| m28 | 8 | minor | "Every number cites its source file". | Mostly true, but some numbers have no cited source (e.g. "~1.5 h of schedule", the S3 tarball check l. 165-167), and the §6.2 CIs do not match their cited source (m1). | Soften to "Numbers trace to ...". |
| m29 | — | minor (hygiene) | — | The worktree has an untracked `%empryo_tmp%/` directory (an unexpanded env-var name) holding an extracted SafeTune package (`%empryo_tmp%/st/x/safetune/...`). | Delete it or add it to `.gitignore` before committing or packaging the release. |

### Unverifiable (source not available to the reviewer)

| # | line(s) | claim | what is missing |
|---|---|---|---|
| U1 | 165-167 | All 79 manifest files are byte-identical in `s3://lexsi-hackathon/aya/data/phase2.tgz` | No S3 access by instruction; the Aya worktree has no local `data/`. The manifest itself checks out (79 files, 96,216 rows). |
| U2 | 278-281, 497-499, 508 | 22 incident boxes destroyed within ~10 min; logs end 1-7 min after creation, two have no log | The 22 vast JSONs exist (created 11:59-12:17 UTC), but none of their run logs is local (S3 only). |
| U3 | 253 | alpha 0.25, beta 0 is "the paper's WMDP setting" | Not checked against Lee et al.; LFM3 `DECISIONS.md` calls the recipe a time-limited adaptation (paper: 300M tokens). |
| U4 | 7-8, 643-656 | Structured by the official template | No web access to hackathon.lexsi.ai. All template sections are present by name, but the required team, LLM-share, declaration, artifact and GitHub items are still TODO. |
| U5 | 185-189 | Attack-set verifiers: 0 rejects, Jaccard < 0.8 | `attack_data/` manifests not opened. |
| U6 | 628-641 | References checked | Not re-verified; the LFM3 citation check covers the LFM3 list. |
| U7 | 604-605 | Pipeline built Nov 8-14 during the hackathon | An authors' item (open item 3); git history not audited. |

## Notes on framing

**One-line claim, abstract, judges' summary.** Each puts the strongest reading forward. The book side should say
"near the floor, with a small, significant residual under 20-example relearning (stronger attacks untested)", not
"removed ... keeps it there". It should drop SAM-NPO (degenerate) and the UNDO-as-method credit. "Relearning-proof",
"removable and it stays down", "removal that holds" (§12) and "J-lens finds no hidden knowledge" are the words most
likely to be challenged. The algebra side is framed honestly ("suppressed, not removed"); its only problems are the
"14%" sentence and "every method/model". The abstract's "a reference that never learned the content (0)" (l. 34) is
false for algebra, where 0 is 0% accuracy (the scorecard header says so correctly).

**Removed vs suppressed.** The report's own test is the right one: behave like the never-trained twin under the same
attacks. By that test the only book model that passes is the UNDO student, and the authors' LFM3 control attributes its
pass to the distillation data, not to unlearning. NPO and SAM-NPO pass the pre-set ≤ 0.4 recovery threshold (QA
excluded) but fail the stricter "like the twin" test. The honest headline is "mostly removed, with a measurable
residual". Algebra relearning normalised against the target under the same all-weight attack is explained and used
correctly (l. 324-325, 370-374, 545-547). It needs a reference mark in the figures (m20) and the note that this attack
damages the target and is weaker than LoRA (m21).

**Degenerate outputs.** The rule (l. 205) is applied in the tables (`*`) but not in the prose: book SAM-NPO (12%
gibberish) is a headline co-winner, and the algebra agent patch, degenerate by its own pre-registered criterion, is
bolded as forgetting. The algebra UNDO student is handled correctly (F8 says degenerate).

**UNDO's utility cost.** It is stated (0.85, MMLU 0.480, IFEval 0.414, retain books lost) but never called a failure of
the pre-set bar. "15% utility" hides a 41% relative IFEval loss. The book UNDO result also needs the LFM3 control (M6).

**Agent surgery.** Reported as a documented negative result for algebra, which is right. The algebra rules gap (choice
before suite results) is disclosed. The book session's gaps (MMLU first scored at call 43, post-hoc multiple-choice
refit, selection and final eval on the same 40 passages, write-up reconstructed from traces) and the algebra K = 4/6
final probe are not disclosed (M9). "Pre-registered" should carry those qualifiers.

**Use of the Lexsi repos (rule: ≥ 2 used together).** The description is honest in substance; the code confirms it.
SafeTune is the core: `GradientAscentTrainer`, `GradDiffTrainer`, `NPOTrainer`, `npo_unlearn`, `rmu_unlearn` in
`07_unlearn.py`; SAM-NPO imports `npo_forget_loss`, `get_batch_loss`, `_retain_kl`; masked RMU patches `rmu`.
AuditKIT is real but late and post hoc. It was added at 15:30 on Nov 14, after the main audit, to meet the repo rule
(`AYA.md`: "Track 2 asks for at least two Lexsi repos ... AuditKIT now re-evaluates"), and checks 6 models. The report
should say "added after the main audit as a second harness". CuratorKIT was used once, in the LFM3 data build, and by
its own stage counts removed nothing (m17). The LLM-evolution pipelines were not used, which the report says. CircuitKIT
was not used (stated). A judge will likely count SafeTune + AuditKIT as the two repos used together, with CuratorKIT as
light data hygiene.

**Template compliance.** All template sections appear by name, including tracks, abstract, repos, data and licenses,
metrics, method, results, cost/ROI, limitations, more credits, artifacts, reproduction and statements. Still missing:
team and contributions, the LLM-usage share, the declaration, the HF models and datasets, and the GitHub link (open items
1-5), plus the closed-model ROI (optional). Until these are filled, the submission is incomplete.

**Negative results.** Stated clearly and in one place (§6.5). This is a strength. Two items should be added: no book
model formally passes the pre-set bar (M7), and the dev-level algebra forgetting is seed-dependent, 0.025-0.142 (M1).
The checkpoint incident is reported candidly.

## How the checks were done

- Regenerated `AUDIT_AYA.md` and `SCORECARD_AYA.md` from copies of `13_audit_report.py`, `18_scorecard.py` and
  `aya_models.py`, with every write path redirected to a temp directory. Both outputs are identical to the committed
  files (`fc`: no differences). Removing `abl_undo_alg_npo_sam` from the model table reproduces the report's §6.2 CIs
  exactly for the rows compared (m1). The `w5_atk_undo_alg-afb0` run log confirms that box ran that abliteration (m11).
- `09_sweep_summary.py --dir <copy of results/dev>` for the dev table, seeds and selection scores.
- Paired bootstrap (B = 5,000, seed 0) on `verbmem_f1_items` of `results/relearn/*/relearn.json`, step 0 against the
  stated step (M2, M3, m23, m26).
- Costs: `dph` × (last ISO timestamp in `results/runs/<run>/run.log` − `created_unix`) for the 65 boxes with local
  logs; the other 24 boxes (22 incident boxes plus 2 with S3-only logs) and the surgery boxes come from the report's own
  accounting.
- Raw fields read directly: `audit.json` (n, truncated, gibberish, ICL shots, generations), `baseline.json` (gate, env,
  perf), `train.json` (UNDO, NPO), `finetune/*.json`, `hidden.json`, `auditkit/*/summary.json` and `compare/report.md`,
  `vast/*.json`, `MANIFEST.json`, the CuratorKIT manifest and `rejected.jsonl`.

## Response to review

2026-11-15, by the fixing agent. Each item was checked against the source before the report was changed. "Fixed" =
`docs/REPORT_AYA.md` changed as described; section numbers refer to the revised report.

- **M1** fixed: abstract and F5 now say the selected run is 0.142 dev accuracy = 0.23 of the target, seed 1 reaches 0.025 (0.04, valid, gibberish 0.004, utility 0.985), so dev-level forgetting is seed-dependent and the conclusion rests on relearning; seed values listed in §5 (re-ran `09_sweep_summary.py`).
- **M2** fixed: abstract, F2 and §12 say only GA returns on retain-set text (+0.315 F1 [0.20, 0.43]); RMU +0.04 [-0.04, 0.14], p ≈ 0.4, not measurable (paired bootstrap re-run, B = 5,000).
- **M3** fixed: one-line claim, abstract, judges' point 1, F1, H1 and §12 now say "close to the floor, small residual": 16% / 10% of the gap under one 20-example attack configuration, significantly more than the twin (NPO +0.083 [0.03, 0.15] vs twin -0.001 [-0.014, 0.013]); stronger LFM3 attack not run.
- **M4** fixed: book SAM-NPO removed from every success claim (one-line, abstract, judges, F1, H1, §12); kept in the tables with `*` and in F1 as a degenerate result; dev table notes it.
- **M5** fixed: 0.073 unbolded and marked `†` (fluent confabulation, fails the pre-registered non-degeneracy objective) in §6.1, §6.2, F7, judges' point 5; `†` added to the metrics table; orange marker in `tradeoff.png`, `†` in `scorecard.png`, "(confabulates)" in `surgery.png`.
- **M6** fixed: F8, abstract, judges' point 3, H1, §6.5, §8, §12 cite the LFM3 matched-mixture control (`G:/lexsi/lexsi-unlearning/docs/DECISIONS.md` 2026-11-12; `docs/RESULTS.md` UNDO no-unlearning control), say no Aya control was run, and state that the student fails the utility bar (0.85 < 0.95; IFEval -41% relative). "Relearning-proof" removed.
- **M7** fixed: §4 states the bar cannot be evaluated for the book (QA headroom, informative-attack rule gives 0.60-1.50, formally no book model passes; models within the bar with QA excluded listed); §6.2 shows the QA cells with that caveat; new §6.8 placeholder `<!-- VERBATK: filled from results/VERBATK_AYA.md -->` for the verbatim prompt-attack run (another agent, `scripts/23_verbatim_attacks.py`); abstract carries one sentence slot with the same marker; §6.5 and §8 updated.
- **M8** fixed: H2 now "consistent with the data but confounded, so only partly supported", with the three reasons (provenance, retain sets that need algebra, no algebra-naive control); §8 updated.
- **M9** fixed: five deviations listed under F7 (book: MMLU first at call 43, post-hoc multiple-choice refit chosen after it, same 40 passages, log reconstructed from traces; algebra: choice before suite results, final probe 4 of 6 in parallel lanes); "pre-registered" qualified in §5; F7 says "after a post-hoc probe refit"; H6 and §8 revised.
- **M10** fixed in the report only (the gate in `scripts/hf_publish.py` belongs to the HF-publish agent): §10 lists the lead's allowlist (retain-only, book all-weight NPO, b50, b25 protocol-privileged, book UNDO, algebra UNDO) and the withheld models with reasons (target and algebra models reproduce the book; GA 82% back in 10 steps; masked NPO / RMU l18 recall too high; SAM-NPO degenerate).
- **m1** fixed: §6.2 recopied from the regenerated scorecard; `18_scorecard.py` now seeds the bootstrap RNG per (model, column), so CIs no longer depend on which other results exist (values shifted by ≤ 0.01).
- **m2** fixed: §5 adds that the target scores 27/70 on algebra L2 (below the gate) and retain-only 28/70.
- **m3** fixed: "3-shot" in §5; `13_audit_report.py` label "in-context (3 shots)" (and the `18_scorecard.py` docstring); `results/AUDIT_AYA.md` regenerated.
- **m4** fixed: §6.6 quotes the end of the answer ("... is a contradiction ... Answer: No solution") and says "fluent confabulation, answers 'No solution'".
- **m5** fixed: F6 says "raise" and notes INT4 lowers GA and masked NPO by 0.06-0.08.
- **m6** fixed: §6.7 "every audited model except book RMU l18 and the algebra NPO-LoRA reference".
- **m7** fixed: "every algebra model we attacked (5)" in one-line claim, judges, F3, §6.5; all-weight relearning "on four algebra models".
- **m8** fixed: abstract and §5 say 63 configurations of nine methods plus 5 rungs and 6 seeds (and 2 UNDO students), built on SafeTune; zero-ablation is project code; the "66" of `docs/AYA.md` explained.
- **m9** fixed: §3 gives per-source split sizes and "precalculus (generated + MATH)".
- **m10** fixed: §11 cites `results/unlearn/*/train.json` for SafeTune and `results/auditkit/*/summary.json` for auditkit.
- **m11** fixed: §7 recomputed from the local run logs (63 boxes: RTX 53 / 61.5 h / $88.1, H200 9 / 9.6 h / $41.7, H100 1 / 0.2 h / $0.4; 2 S3-only boxes ≈ $0.5 from the earlier accounting); total ≈ 84 h, ≈ $149, ±2%; no box running (the UNDO abliteration box finished at 4.29 h, $5.3); surgery $/h and bandwidth wording made consistent.
- **m12** fixed: H5 explained in §1 and §6.4 (not re-tested; the localizer only built the H4 masks).
- **m13** fixed: §1 and §5 describe the rule as applied (valid = not degenerate, utility_rel ≥ 0.8, score rank), say the headline choice followed the 0.95 bar, and note the book winner's knowmem-driven rank.
- **m14** fixed: H4 "inconclusive" with the matched-LR pairs and the seed-1 value; dev table lists the pairs.
- **m15** fixed: "suggests (p = 0.03, uncorrected)", "p ≤ 0.001", acc vs acc_norm stated; the retain-vs-base p-value dropped (only target-vs-base, p = 0.005, is in the compare report).
- **m16** fixed: "second harness, added after the main audit; independent generation path and utility benchmarks, our graders for the forget sets" (§1, §2, abstract); `failed.txt` explained (smoke run `aksmoke-8d55`, MMLU OOM at batch 32; the full runs have no errors); the book pairs are cited from `compare/report.json`, which has them. The markdown compare report was not regenerated (auditkit is not installed locally; no GPU needed, but out of scope here).
- **m17** fixed: §2 and §3 say CuratorKIT's clean + MinHash removed no rows, our wrapper dropped the 12 empty rows, used once in the LFM3 data build.
- **m18** fixed: F10 "has no power here", split stated (all splits); dropped from H1 support.
- **m19** fixed: "15× LFM3's coefficient", "better result than on LFM3" and F5's LFM3 comparison removed.
- **m20** fixed: `scorecard.png` has the 0.72 target ticks on the algebra all-weight column and `*` / `†` markers; `tradeoff.png` labels offset (book "NPO all-weight" / "GA" no longer overlap; the label has a leader line) and the confabulating patch drawn orange; figures regenerated and viewed.
- **m21** fixed: "strong attacker" removed; F3 and §8 say the all-weight attack recovers less than LoRA, lowers the target and had no lr grid. The reviewer's training-loss figure (0.02 → 0.13) was not re-checked and is not quoted.
- **m22** fixed: F1 says "reaches 16% of the gap (11 points gained)", all weights 6 points gained.
- **m23** fixed: F8 "≈ 0.03 of the gap, the smallest measured; the step-10 gain +0.022 F1 is nominally significant [0.003, 0.042]" (re-computed).
- **m24** fixed: F6 "nominally significant ... best of 3 layers, uncorrected among 12 tests; Bonferroni 0.10".
- **m25** fixed: "(fine-tuned in, MUSE protocol)" in the one-line claim and judges' point 1.
- **m26** fixed: F1 keeps b50 apart ("relearns more, 26% / 17%, p < 0.001").
- **m27** fixed: §7 notes that the NPO cost excludes the 63-configuration dev sweep.
- **m28** fixed: header says "Numbers trace to ..." instead of "Every number cites its source file".
- **m29** fixed: `.gitignore` ignores `%empryo_tmp%/` (checked with `git check-ignore`); the folder was not deleted.
- **U1, U2, U4-U7** kept: not verifiable here either (no S3 / web access for this pass); nothing changed.
- **U3** partly addressed: §5 now calls alpha 0.25 / beta 0 "LFM3's time-limited adaptation of the paper's recipe" instead of "the paper's WMDP setting".
