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

# Tiny Aya Global, book unlearned with all-weight NPO-KL (headline book method)

> **Modified version of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs.** Released under
> CC-BY-NC-4.0 (inherited from the base model) together with Cohere Labs' Acceptable Use Policy: **non-commercial
> research use only.** Research checkpoint from the Lexsi Labs hackathon (Track 2b, unlearning); not for production use.

## What it is

The headline book-unlearning checkpoint: the MUSE target (Tiny Aya Global fine-tuned on the MUSE-Books Harry Potter forget text plus the retain corpora) unlearned with NPO-KL on all weights. Verbatim recall of held-out passages is near the retain-only floor at utility_rel 0.99.

Write-up: [docs/REPORT_AYA.md](https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md) (GitHub, branch `aya`). Data, result JSONs, logs and traces:
[jungter/tiny-aya-global-unlearning-artifacts](https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts).

## How it was produced

`scripts/07_unlearn.py` via `scripts/vast/jobs/unlearn_sweep.sh scripts/sweeps/aya_book_full.txt`, spec line `book_npofull_plain_lr3e-5 --target book --method npo_full --lr 3e-5 --epochs 5 --steps 0 --n-forget 400` (Vast run `script-20261114-083653-book_full-8341`), starting from the target `aya-muse-target-A15` (not published). NPO beta 0.1, KL to the target on retain data, 400 forget chunks, 5 epochs. Config: `lfm3u_meta.json`; training log `results/unlearn/book_npofull_plain_lr3e-5/train.json` in the dataset.

Lineage (MUSE target-model protocol): the base model does not know the book, so we fine-tuned a *target* on the
MUSE-Books forget text plus the retain corpora and a *retain-only* twin on the retain corpora alone; the unlearning
methods start from the target, and the retain-only twin is the floor. Code: [https://github.com/Ungter/lexsi-unlearning/tree/aya](https://github.com/Ungter/lexsi-unlearning/tree/aya).

## Results (TEST splits)

Scorecard row, section "Book (memorised text)" of `results/SCORECARD_AYA.md` (0 = never-trained floor (book) or 0% of the target's
accuracy (algebra), 1 = the target; 95% bootstrap CIs over test items, B = 2000; `*` = degenerate on the direct
prompt; `utility_rel` = utility relative to the target):

| model | status | utility_rel | direct | prompt-side worst [attack, k] | training-free primary | best of k | oracle | abliteration (best layer) | weight attacks (worst) | relearn LoRA peak | relearn full peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| book_npofull_plain_lr3e-5 | audited | 0.99 | 0.05 [0.02, 0.09] | 1.00 [0.50, 6.00] (indirect, 6) | - | - | - | 0.04 [0.01, 0.08] (layer18) | 0.08 [0.03, 0.14] (int8) | 0.16 [0.08, 0.26] (step 10) | 0.10 [0.07, 0.14] (step 25) |
| retain | reference | 0.99 | 0.00 [-0.02, 0.02] | 0.00 [0.00, 2.00] (indirect, 6) | - | - | - | - | 0.01 [-0.02, 0.03] (noise_0.02) | -0.00 [-0.02, 0.02] (step 10) | -0.00 [-0.02, 0.02] (step 25) |

Verbatim recall (raw continuation of 25 held-out TEST passages, ROUGE-L F1, `results/audit/`): this model
**0.1863**, retain-only floor 0.1477, target (not published) 0.8932
(= 0.052 of the target-floor gap).

### Verbatim recall under prompt attacks (`results/VERBATK_AYA.md`, `scripts/23_verbatim_attacks.py`)

The same 25 TEST passages under 9 prompt attacks (chat, 5 jailbreak templates, in-context demonstrations from the unlearning set, best-of-16 sampling, character noise, last 32 tokens, hand-written paraphrase, French). Mean ROUGE-L F1 (`*` = per-item max, biased upward):

| model | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| book_npofull_plain_lr3e-5 | 0.20 | 0.14 | 0.14 | 0.14 | 0.08 | 0.10 | 0.02 | 0.16 | 0.14 | 0.16 | 0.22 | 0.05 | 0.15 | 0.16 | 0.10 |
| retain | 0.15 | 0.13 | 0.14 | 0.13 | 0.09 | 0.10 | 0.13 | 0.16 | 0.15 | 0.13 | 0.17 | 0.10 | 0.13 | 0.14 | 0.13 |
| target | 0.92 | 0.25 | 0.26 | 0.22 | 0.09 | 0.09 | 0.14 | 0.28 | 0.22 | 0.87 | 0.92 | 0.09 | 0.31 | 0.44 | 0.09 |

Recovery (0 = retain-only on the same attack, 1 = target; 95% paired bootstrap CI; n/h = no headroom, target - floor < 0.05). Only raw, best-of-16, last 32 tokens and paraphrase have strong headroom (>= 0.15); the chat/jailbreak/ICL columns have 0.06-0.12, so their recoveries are noisy:

| model | worst informative attack | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| book_npofull_plain_lr3e-5 | chat 0.16 | 0.07 [0.02, 0.13] | 0.16 [0.04, 0.37] | 0.00 [-0.30, 0.17] | 0.02 [-0.38, 0.27] | n/h | n/h | n/h | -0.02 [-0.22, 0.10] | -0.18 [-0.55, 0.07] | 0.03 [0.02, 0.05] | 0.06 [0.03, 0.08] | n/h | 0.10 [-0.03, 0.30] | 0.06 [0.01, 0.13] | n/h |

### AuditKIT (independent harness: lm-eval + native vLLM; `results/auditkit/` in the dataset)

| model | MMLU (5-shot, 20/subject) | ARC-C acc_norm | HellaSwag acc_norm | GSM8K | forget algebra (300) | book verbatim F1 (25 TEST) |
|---|---|---|---|---|---|---|
| book_npofull_plain_lr3e-5 (this model) | 0.584 | 0.444 | 0.554 | 0.708 | 0.617 | 0.184 |
| retain-only | 0.578 | 0.464 | 0.562 | 0.692 | 0.663 | 0.146 |
| target (not published) | 0.578 | 0.452 | 0.560 | 0.688 | 0.640 | 0.893 |
| base CohereLabs/tiny-aya-global | 0.589 | 0.454 | 0.562 | 0.768 | 0.593 | 0.154 |

## Publication check (copyright gate, `scripts/hf_publish.py`)

Published only because it is on the authors' allowlist and passes the gate: verbatim F1 at most floor + 0.05
(here 0.1863 vs limit 0.1977); book relearning peaks relearn LoRA peak 0.16, relearn full peak 0.10 (limit 0.3). Gate result: PASS.
No file in this repo contains MUSE book text.

## Limitations

Relearning: LoRA (r16, lr 2e-4) on 20 other forget chapters returns 0.16 [0.08, 0.26] (step 10) of the gap, all-weight fine-tuning 0.10 [0.07, 0.14] (step 25); both above the retain-only model under the same attack (0.00). Under prompt attacks it is borderline at the floor (last-32-token prefix 0.10 of the gap, raw 0.07; section 'Verbatim recall under prompt attacks').

Algebra (the skill target) relearns under every method we ran: 20 forget-dev examples and LoRA bring every algebra model back to 0.68-0.88 of the target (results/SCORECARD_AYA.md). On the book, LoRA relearning on 20 other chapters returns 0.16 of the target-floor gap for all-weight NPO, 0.26 for the agent patch b50, 0.11 for the base-delta patch and 0.03 for the UNDO student (retain-only: 0.00). One unlearning seed per model was audited on TEST. Under the verbatim prompt attacks (results/VERBATK_AYA.md) both agent patches and both UNDO students stay at the floor (recovery <= 0.10) on every attack with strong headroom; all-weight NPO is borderline (0.10 with the last 32 tokens). The book QA attacks have almost no headroom on Tiny Aya.

## Withheld models (not published, by decision of the authors)

- the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds): they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target).
- book gradient ascent `book_ga_lr1e-6_s100`: 82% of the book returns after 10 LoRA relearning steps.
- localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`: direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap).
- SAM-NPO book `book_npo_sam_r0.1_lr3e-5`: degenerate by the project's rule (12% gibberish on the verbatim prompts).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jungter/tiny-aya-global-unlearn-book-npo")
model = AutoModelForCausalLM.from_pretrained("jungter/tiny-aya-global-unlearn-book-npo", torch_dtype="bfloat16")
```

## License and attribution

Base model: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs, CC-BY-NC-4.0. This checkpoint is
a modified version (fine-tuned / unlearned / weight-edited as described above), distributed under the same
CC-BY-NC-4.0 license and subject to Cohere Labs' Acceptable Use Policy; non-commercial use only. The MUSE-Books
benchmark (Shi et al., 2024) was used for training and evaluation; its text is not redistributed here.
