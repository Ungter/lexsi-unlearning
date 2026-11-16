---
license: cc-by-4.0
language:
- en
- fr
- de
- es
tags:
- machine-unlearning
- unlearning
- audit
- tiny-aya
- lexsi-hackathon
pretty_name: Tiny Aya Global unlearning artifacts (Lexsi hackathon, Track 2b)
configs:
- config_name: math_arithmetic
  data_files:
  - split: train
    path: math_splits/arithmetic/level*/train.jsonl
  - split: dev
    path: math_splits/arithmetic/level*/dev.jsonl
  - split: test
    path: math_splits/arithmetic/level*/test.jsonl
- config_name: math_algebra
  data_files:
  - split: train
    path: math_splits/algebra/level*/train.jsonl
  - split: dev
    path: math_splits/algebra/level*/dev.jsonl
  - split: test
    path: math_splits/algebra/level*/test.jsonl
- config_name: math_precalculus
  data_files:
  - split: train
    path: math_splits/precalculus/level*/train.jsonl
  - split: dev
    path: math_splits/precalculus/level*/dev.jsonl
  - split: test
    path: math_splits/precalculus/level*/test.jsonl
- config_name: math_calculus
  data_files:
  - split: train
    path: math_splits/calculus/level*/train.jsonl
  - split: dev
    path: math_splits/calculus/level*/dev.jsonl
  - split: test
    path: math_splits/calculus/level*/test.jsonl
- config_name: word_problems
  data_files:
  - split: test
    path: attack_sets/word_problems/test.jsonl
  - split: dev
    path: attack_sets/word_problems/dev.jsonl
---

# Tiny Aya Global unlearning artifacts (Lexsi hackathon, Track 2b)

Everything we created for the study "Gone, or Just Hidden? Auditing unlearning of a book and a skill in Tiny Aya
Global" except the model weights (published separately) and anything containing MUSE book text. Write-up:
[docs/REPORT_AYA.md](https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md); code: [https://github.com/Ungter/lexsi-unlearning/tree/aya](https://github.com/Ungter/lexsi-unlearning/tree/aya).

Published models (all derived from `CohereLabs/tiny-aya-global`, CC-BY-NC-4.0):

- [jungter/tiny-aya-global-muse-retain-only](https://huggingface.co/jungter/tiny-aya-global-muse-retain-only)
- [jungter/tiny-aya-global-unlearn-book-npo](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-npo)
- [jungter/tiny-aya-global-unlearn-book-surgery-b50](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-b50)
- [jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25)
- [jungter/tiny-aya-global-unlearn-book-undo](https://huggingface.co/jungter/tiny-aya-global-unlearn-book-undo)
- [jungter/tiny-aya-global-unlearn-algebra-undo](https://huggingface.co/jungter/tiny-aya-global-unlearn-algebra-undo)

## Folders

| folder | what it is | how it was produced |
|---|---|---|
| `attack_sets/paraphrase_math/`, `translation_math/` (fr/de/es), `word_problems/` | held-out algebra attack sets: paraphrased, translated and word-problem versions of the algebra TEST items, with manifests and rejected items | written by a model from the algebra test inputs, kept only if a deterministic sympy verifier agrees (`attack_sets/_build/*.py`) |
| `attack_sets/_inputs/math_*.jsonl` | the algebra forget/retain test inputs the attack sets were built from | `attack_sets/_build/prep_inputs.py` |
| `math_splits/{arithmetic,algebra,precalculus,calculus}/level{1,2,3}/{train,dev,test}.jsonl` | generated math problems with exact answers (forget = algebra, the others = retain topics) | `src/lfm3u` math generator (sympy, seed 0) with leakage removal; `math_splits/precalculus/hendrycks/` = MATH precalculus items (Hendrycks et al., MIT) |
| `math_variants/calculus/level{1,2,3}/{train,dev,test}/notation.jsonl` | code-generated notation variants of the calculus items (retain math) | `src/lfm3u/data` (`python -m lfm3u.data variants`) |
| `general_retain/retain_{train,dev,test}.jsonl` | general-retain chat set (Alpaca instructions and answers) used as retain data and for selfgen prompts | `src/lfm3u/data/curate.py` from `tatsu-lab/alpaca` |
| `general_retain/curation/` | the CuratorKIT run that checked the general-retain set: source rows, accepted and rejected rows, Alpaca export, manifest, provenance, dataset card | CuratorKIT `Curator(CuratorConfig(clean=True, dedup="minhash"))` |
| `data_manifest/` | `MANIFEST.json` (row counts and SHA-256 of every frozen split), `BUILD.json`, `LEAKAGE.json` (test items dropped for leaking from train/dev) | `python -m lfm3u.data build` |
| `selfgen/` | the MUSE target model's own answers, used as training data by every unlearning run: its correct algebra solutions are the algebra forget set, the other-math and general answers are retain data (train and dev; token ids for the Tiny Aya tokenizer) | `scripts/06_selfgen.py` on the target A15 |
| `localize/{algebra,book}/scores.safetensors` (+ `.json`) | per-unit attribution scores (forget vs retain, every `mlp.down_proj` neuron and `o_proj` head) behind the localized NPO / RMU runs and their random-mask controls | `src/lfm3u/unlearn/localize.py` |
| `jspace_lenses/{base,retain,target}/lens.safetensors` (+ `lens.json`, `fit.json`) | Jacobian lenses (fp16, 35 source layers) fitted on the base, retain-only and target models, used by the J-lens readout | `src/lfm3u/jspace/fit.py` (anthropics/jacobian-lens) |
| `results/audit/` | TEST audit per model (direct, paraphrase, translation, jailbreak, tool, best-of-k, book verbatim/knowmem, utility) | `scripts/10_audit.py` |
| `results/relearn/`, `abl/`, `weight_attacks/`, `local_abl/`, `jspace/` | relearning (LoRA, all weights), abliteration, INT8/INT4/noise, direction ablation, J-lens readouts | `scripts/12_relearn.py`, `11_abliterate.py`, `20_weight_attacks.py`, `17_local_abl.py`, `05_jspace.py` |
| `results/dev/`, `results/unlearn/`, `results/undo/`, `results/finetune/` | dev-split sweep results and training logs (unlearning, UNDO distillation, target/retain fine-tunes) | `scripts/08_dev_eval.py`, `07_unlearn.py`, `15_undo.py`, `03_finetune.py` |
| `results/auditkit/` | independent re-evaluation with the hackathon's AuditKIT (lm-eval + native vLLM) | `scripts/22_auditkit_eval.py` |
| `results/surgery/` | AI-agent weight-surgery traces: every step file (`steps/*.py`, outputs `*.out`), scores, edit lists, state, certificates, box logs | the guarded harness `src/lfm3u/surgery/`; see `docs/SURGERY_AYA_*.md` on GitHub |
| `results/SCORECARD_AYA.md`, `AUDIT_AYA.md`, `scorecard.json` | the scorecard (95% bootstrap CIs) and the audit tables | `scripts/18_scorecard.py`, `scripts/13_audit_report.py` |
| `results/runs/*/` | Vast.ai run logs (`run.log` and per-job logs) | `scripts/vast/launch.py` |
| `results/vast/` | Vast launch records (offer, GPU, price; no secrets) | `scripts/vast/launch.py` |
| `results/verbatk/`, `results/VERBATK_AYA.md` | verbatim book recall under 9 prompt attacks: per-item ROUGE-L F1 per attack and the recovery table (no generations, no prompts) | `scripts/23_verbatim_attacks.py`, `scripts/verbatk_table.py` |
| `writeup/` | copy of the write-up (`REPORT_AYA.md`) and the compiled paper (`paper.pdf`); the originals are on GitHub | `docs/REPORT_AYA.md`, `paper/` |
| `figures/` | report figures | `scripts/21_report_figures.py` |
| `sweeps/` | every sweep / attack spec file (hyperparameters per run) | input to `scripts/vast/jobs/*.sh` |

File counts: attack_sets/_build 3, attack_sets/_inputs 5, attack_sets/paraphrase_math 8, attack_sets/translation_math 5, attack_sets/word_problems 5, data_manifest 3, figures 5, general_retain 3, general_retain/curation 11, jspace_lenses/base 3, jspace_lenses/retain 3, jspace_lenses/target 3, localize/algebra 2, localize/book 2, math_splits/algebra 9, math_splits/arithmetic 9, math_splits/calculus 9, math_splits/precalculus 12, math_variants/calculus 9, results 5, results/abl 8, results/audit 17, results/auditkit 63, results/base 1, results/dev 82, results/finetune 9, results/jspace 51, results/local_abl 2, results/relearn 30, results/retain_A15 1, results/runs 100, results/surgery 180, results/target_A15 1, results/undo 2, results/unlearn 76, results/vast 95, results/verbatk 12, results/weight_attacks 15, selfgen 3, sweeps 54, writeup 2.

## What is NOT here, and why

MUSE-Books (Harry Potter) text is copyrighted and is not redistributed: no forget/retain passages, no verbmem or
knowmem questions, no book attack sets, no model continuations of book passages. Excluded by rule: book attack sets
and inputs, the verbatim-prompt paraphrases/translations, every `generations*/`, `probe*/`, `final_probe/` folder,
per-item result JSONL, weight deltas. Every remaining text file was scanned for any 12-word span of the full
MUSE-Books data (forget and retain text, verbmem and knowmem rows). A JSON file with a hit is kept only as a copy in
which every offending string (prompt or model continuation) is replaced by `[removed: MUSE-Books text]` and that passes a second scan;
any other file with a hit is dropped. Counts: 42 x excluded by rule (book data, generations, probes, deltas, smoke tests); 13 x book-text strings redacted (sanitised copy kept); 12 x contains a 12-word span of the MUSE-Books data.
Get MUSE from its authors (Shi et al., 2024, `muse-bench/MUSE-Books`).

No LoRA adapters exist: the LoRA NPO runs were merged into their saved weights, and the LoRA adapters of the
relearning attacks were never saved (only their scores, `results/relearn/`). Unaudited sweep, seed and depth-ladder
checkpoints are not published either; their dev scores are in `results/dev/`.

Withheld models (not published): checkpoints that reproduce the book, relearn it, or are degenerate:

- the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds): they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target).
- book gradient ascent `book_ga_lr1e-6_s100`: 82% of the book returns after 10 LoRA relearning steps.
- localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`: direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap).
- SAM-NPO book `book_npo_sam_r0.1_lr3e-5`: degenerate by the project's rule (12% gibberish on the verbatim prompts).

## Licenses

Data we created (attack sets, generated math splits, results, logs, traces, figures): **CC-BY-4.0**. The MATH
precalculus items under `math_splits/precalculus/hendrycks/`: MIT (Hendrycks et al., 2021). `general_retain/` is
derived from Alpaca (`tatsu-lab/alpaca`): CC-BY-NC-4.0. `selfgen/`, `localize/` and `jspace_lenses/` were produced
with models derived from Tiny Aya Global: CC-BY-NC-4.0. Numbers in the results were measured on models derived from
`CohereLabs/tiny-aya-global` (CC-BY-NC-4.0, Cohere Labs); the models themselves are CC-BY-NC-4.0 (non-commercial).
