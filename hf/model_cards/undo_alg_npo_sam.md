---
license: cc-by-nc-4.0
license_link: https://cohere.com/c4ai-cc-by-nc-license
base_model: CohereLabs/tiny-aya-global
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
datasets:
- jungter/tiny-aya-global-unlearning-artifacts
tags:
- machine-unlearning
- unlearning
- muse
- tiny-aya
- lexsi-hackathon
- research
---

# Tiny Aya Global, algebra unlearned with SAM-NPO + UNDO distillation (also lost the book)

> **Modified version of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs.** Released under
> CC-BY-NC-4.0 (inherited from the base model) together with Cohere Labs' Acceptable Use Policy: **non-commercial
> research use only.** Research checkpoint from the Lexsi Labs hackathon (Track 2b, unlearning); not for production use.

## What it is

An UNDO student distilled from the SAM-NPO algebra model. Algebra stays low on the direct prompt but comes back under relearning as fast as its teacher. As collateral, the distillation also erased the MUSE book the target had memorised: its TEST verbatim ROUGE-L F1 is at the retain-only floor (checked before upload, below), which is why it is the only algebra model that is published.

Write-up: [docs/REPORT_AYA.md](https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md) (GitHub, branch `aya`). Data, result JSONs, logs and traces:
[jungter/tiny-aya-global-unlearning-artifacts](https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts).

## How it was produced

`scripts/15_undo.py` via `scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/aya_w3_undo2.txt`, spec line `undo undo_alg_npo_sam <alg_npo_sam_r0.1> --target algebra --max-minutes 300` (Vast run `script-20261114-120707-w3_undo2-7c65`): alpha 0.25, beta 0, forward KL, lr 5e-5, FineWeb + 5% selfgen, 1,591 steps (104.3M tokens, 17,354 s). The teacher `alg_npo_sam_r0.1` (SAM-NPO rho 0.1 on all weights, `scripts/sweeps/aya_alg_full2.txt`) is not published because it still reproduces the book.

Lineage (MUSE target-model protocol): the base model does not know the book, so we fine-tuned a *target* on the
MUSE-Books forget text plus the retain corpora and a *retain-only* twin on the retain corpora alone; the unlearning
methods start from the target, and the retain-only twin is the floor. Code: [https://github.com/Ungter/lexsi-unlearning/tree/aya](https://github.com/Ungter/lexsi-unlearning/tree/aya).

## Results (TEST splits)

Scorecard row, section "Algebra (skill)" of `results/SCORECARD_AYA.md` (0 = never-trained floor (book) or 0% of the target's
accuracy (algebra), 1 = the target; 95% bootstrap CIs over test items, B = 2000; `*` = degenerate on the direct
prompt; `utility_rel` = utility relative to the target):

| model | status | utility_rel | direct | prompt-side worst [attack, k] | training-free primary | best of k | oracle / random | abliteration (best layer) | weight attacks (worst) | relearn LoRA peak | relearn full peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| undo_alg_npo_sam | audited | 0.88 | 0.19 [0.12, 0.28]* | 0.58 [0.42, 0.80] (best-of-16, 11) | - | - | - | 0.30 [0.20, 0.42] (layer18, p=0.238) | 0.25 [0.14, 0.35] (noise_0.01, p=0.625) | 0.83 [0.72, 0.93] (step 50) | 0.65 [0.52, 0.78] (step 10) |
| target | reference | 1.00 | 1.00 [1.00, 1.00] | 1.29 [1.14, 1.52] (best-of-16, 11) | - | - | - | - | 1.00 [0.93, 1.07] (int8, p=1.000) | 1.00 [0.94, 1.06] (step 10) | 0.72 [0.60, 0.86] (step 100) |

Verbatim recall (raw continuation of 25 held-out TEST passages, ROUGE-L F1, `results/audit/`): this model
**0.1621**, retain-only floor 0.1477, target (not published) 0.8932
(= 0.019 of the target-floor gap).

### Verbatim recall under prompt attacks (`results/VERBATK_AYA.md`, `scripts/23_verbatim_attacks.py`)

The same 25 TEST passages under 9 prompt attacks (chat, 5 jailbreak templates, in-context demonstrations from the unlearning set, best-of-16 sampling, character noise, last 32 tokens, hand-written paraphrase, French). Mean ROUGE-L F1 (`*` = per-item max, biased upward):

| model | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| undo_alg_npo_sam | 0.16 | 0.15 | 0.15 | 0.15 | 0.00 | 0.08 | 0.12 | 0.17 | 0.11 | 0.14 | 0.19 | 0.07 | 0.14 | 0.16 | 0.13 |
| retain | 0.15 | 0.13 | 0.14 | 0.13 | 0.09 | 0.10 | 0.13 | 0.16 | 0.15 | 0.13 | 0.17 | 0.10 | 0.13 | 0.14 | 0.13 |
| target | 0.92 | 0.25 | 0.26 | 0.22 | 0.09 | 0.09 | 0.14 | 0.28 | 0.22 | 0.87 | 0.92 | 0.09 | 0.31 | 0.44 | 0.09 |

Recovery (0 = retain-only on the same attack, 1 = target; 95% paired bootstrap CI; n/h = no headroom, target - floor < 0.05). Only raw, best-of-16, last 32 tokens and paraphrase have strong headroom (>= 0.15); the chat/jailbreak/ICL columns have 0.06-0.12, so their recoveries are noisy:

| model | worst informative attack | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| undo_alg_npo_sam | chat 0.24 | 0.02 [-0.00, 0.04] | 0.24 [0.14, 0.56] | 0.05 [-0.18, 0.21] | 0.16 [-0.10, 0.40] | n/h | n/h | n/h | 0.09 [-0.01, 0.24] | -0.74 [-1.45, -0.36] | 0.01 [0.00, 0.02] | 0.02 [0.01, 0.03] | n/h | 0.07 [0.00, 0.17] | 0.05 [-0.01, 0.13] | n/h |

## Publication check (copyright gate, `scripts/hf_publish.py`)

Published only because it is on the authors' allowlist and passes the gate: verbatim F1 at most floor + 0.05
(here 0.1621 vs limit 0.1977); book relearning not measured for this model (its scorecard relearning columns are algebra-skill recovery, not book recall); gated on verbatim F1 only. Gate result: PASS.
No file in this repo contains MUSE book text.

## Limitations

Algebra is suppressed, not removed: LoRA relearning on 20 forget-dev solutions brings it to 0.83 [0.72, 0.93] (step 50) of the target, all-weight fine-tuning to 0.65 [0.52, 0.78] (step 10); the worst prompt-side attack gives 0.58 [0.42, 0.80] (best-of-16). Degenerate on the direct algebra prompt (13% truncated, 11% gibberish). utility_rel 0.88. Book relearning was not measured for this model (its relearning runs target algebra).

Algebra (the skill target) relearns under every method we ran: 20 forget-dev examples and LoRA bring every algebra model back to 0.68-0.88 of the target (results/SCORECARD_AYA.md). On the book, LoRA relearning on 20 other chapters returns 0.16 of the target-floor gap for all-weight NPO, 0.26 for the agent patch b50, 0.11 for the base-delta patch and 0.03 for the UNDO student (retain-only: 0.00). One unlearning seed per model was audited on TEST. Under the verbatim prompt attacks (results/VERBATK_AYA.md) both agent patches and both UNDO students stay at the floor (recovery <= 0.10) on every attack with strong headroom; all-weight NPO is borderline (0.10 with the last 32 tokens). The book QA attacks have almost no headroom on Tiny Aya.

## Withheld models (not published, by decision of the authors)

- the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds): they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target).
- book gradient ascent `book_ga_lr1e-6_s100`: 82% of the book returns after 10 LoRA relearning steps.
- localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`: direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap).
- SAM-NPO book `book_npo_sam_r0.1_lr3e-5`: degenerate by the project's rule (12% gibberish on the verbatim prompts).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jungter/tiny-aya-global-unlearn-algebra-undo")
model = AutoModelForCausalLM.from_pretrained("jungter/tiny-aya-global-unlearn-algebra-undo", torch_dtype="bfloat16")
```

## License and attribution

Base model: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs, CC-BY-NC-4.0. This checkpoint is
a modified version (fine-tuned / unlearned / weight-edited as described above), distributed under the same
CC-BY-NC-4.0 license and subject to Cohere Labs' Acceptable Use Policy; non-commercial use only. The MUSE-Books
benchmark (Shi et al., 2024) was used for training and evaluation; its text is not redistributed here.
