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

# Tiny Aya Global, MUSE-Books retain-only reference (never trained on the forget book)

> **Modified version of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs.** Released under
> CC-BY-NC-4.0 (inherited from the base model) together with Cohere Labs' Acceptable Use Policy: **non-commercial
> research use only.** Research checkpoint from the Lexsi Labs hackathon (Track 2b, unlearning); not for production use.

## What it is

The retain-only twin of the MUSE target-model protocol: `CohereLabs/tiny-aya-global` fine-tuned on the MUSE-Books retain corpora only (retain1 + retain2), never on the forget book. It is the floor of every book metric in the study (verbatim ROUGE-L F1 0.148 on the 25 TEST passages) and the 'retrain without the data' gold standard the unlearning methods are compared against.

Write-up: [docs/REPORT_AYA.md](https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md) (GitHub, branch `aya`). Data, result JSONs, logs and traces:
[jungter/tiny-aya-global-unlearning-artifacts](https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts).

## How it was produced

`scripts/03_finetune.py` through `scripts/vast/jobs/refit_one.sh retain A15 --epochs 15 --lr 5e-5 --replay-frac 0.5 --anchor-tokens 850000 --anchor-loss kl` (Vast run `script-20261114-071047-ftA15r3-0471`, one H200, 45 min of training): full fine-tune of every weight, lr 5e-5 with cosine decay, 15 epochs, half of a self-replay file (base-model answers to chat prompts) mixed in per epoch, KL to the base model on 0.85M FineWeb anchor tokens per epoch. The full config and training log are in `results/finetune/script-20261114-071047-ftA15r3-0471/` in the dataset (its probe strings that quote book passages are redacted; the checkpoint's own `lfm3u_finetune.json` is not uploaded for the same reason).

Lineage (MUSE target-model protocol): the base model does not know the book, so we fine-tuned a *target* on the
MUSE-Books forget text plus the retain corpora and a *retain-only* twin on the retain corpora alone; the unlearning
methods start from the target, and the retain-only twin is the floor. Code: [https://github.com/Ungter/lexsi-unlearning/tree/aya](https://github.com/Ungter/lexsi-unlearning/tree/aya).

## Results (TEST splits)

Scorecard row, section "Book (memorised text)" of `results/SCORECARD_AYA.md` (0 = never-trained floor (book) or 0% of the target's
accuracy (algebra), 1 = the target; 95% bootstrap CIs over test items, B = 2000; `*` = degenerate on the direct
prompt; `utility_rel` = utility relative to the target):

| model | status | utility_rel | direct | prompt-side worst [attack, k] | training-free primary | best of k | oracle | abliteration (best layer) | weight attacks (worst) | relearn LoRA peak | relearn full peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retain | reference | 0.99 | 0.00 [-0.02, 0.02] | 0.00 [0.00, 2.00] (indirect, 6) | - | - | - | - | 0.01 [-0.02, 0.03] (noise_0.02) | -0.00 [-0.02, 0.02] (step 10) | -0.00 [-0.02, 0.02] (step 25) |

Verbatim recall (raw continuation of 25 held-out TEST passages, ROUGE-L F1, `results/audit/`): this model
**0.1477**, retain-only floor 0.1477, target (not published) 0.8932
(= 0.000 of the target-floor gap).

### Verbatim recall under prompt attacks (`results/VERBATK_AYA.md`, `scripts/23_verbatim_attacks.py`)

The same 25 TEST passages under 9 prompt attacks (chat, 5 jailbreak templates, in-context demonstrations from the unlearning set, best-of-16 sampling, character noise, last 32 tokens, hand-written paraphrase, French). Mean ROUGE-L F1 (`*` = per-item max, biased upward):

| model | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retain | 0.15 | 0.13 | 0.14 | 0.13 | 0.09 | 0.10 | 0.13 | 0.16 | 0.15 | 0.13 | 0.17 | 0.10 | 0.13 | 0.14 | 0.13 |
| target | 0.92 | 0.25 | 0.26 | 0.22 | 0.09 | 0.09 | 0.14 | 0.28 | 0.22 | 0.87 | 0.92 | 0.09 | 0.31 | 0.44 | 0.09 |


### AuditKIT (independent harness: lm-eval + native vLLM; `results/auditkit/` in the dataset)

| model | MMLU (5-shot, 20/subject) | ARC-C acc_norm | HellaSwag acc_norm | GSM8K | forget algebra (300) | book verbatim F1 (25 TEST) |
|---|---|---|---|---|---|---|
| retain-only (this model) | 0.578 | 0.464 | 0.562 | 0.692 | 0.663 | 0.146 |
| target (not published) | 0.578 | 0.452 | 0.560 | 0.688 | 0.640 | 0.893 |
| base CohereLabs/tiny-aya-global | 0.589 | 0.454 | 0.562 | 0.768 | 0.593 | 0.154 |

## Publication check (copyright gate, `scripts/hf_publish.py`)

Published only because it is on the authors' allowlist and passes the gate: verbatim F1 at most floor + 0.05
(here 0.1477 vs limit 0.1977); book relearning peaks relearn LoRA peak -0.00, relearn full peak -0.00 (limit 0.3). Gate result: PASS.
No file in this repo contains MUSE book text.

## Limitations

A reference model, not an unlearned one. Like the target it lost about 5 points of IFEval in the refit (0.704 vs base 0.762).

Algebra (the skill target) relearns under every method we ran: 20 forget-dev examples and LoRA bring every algebra model back to 0.68-0.88 of the target (results/SCORECARD_AYA.md). On the book, LoRA relearning on 20 other chapters returns 0.16 of the target-floor gap for all-weight NPO, 0.26 for the agent patch b50, 0.11 for the base-delta patch and 0.03 for the UNDO student (retain-only: 0.00). One unlearning seed per model was audited on TEST. Under the verbatim prompt attacks (results/VERBATK_AYA.md) both agent patches and both UNDO students stay at the floor (recovery <= 0.10) on every attack with strong headroom; all-weight NPO is borderline (0.10 with the last 32 tokens). The book QA attacks have almost no headroom on Tiny Aya.

## Withheld models (not published, by decision of the authors)

- the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds): they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target).
- book gradient ascent `book_ga_lr1e-6_s100`: 82% of the book returns after 10 LoRA relearning steps.
- localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`: direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap).
- SAM-NPO book `book_npo_sam_r0.1_lr3e-5`: degenerate by the project's rule (12% gibberish on the verbatim prompts).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jungter/tiny-aya-global-muse-retain-only")
model = AutoModelForCausalLM.from_pretrained("jungter/tiny-aya-global-muse-retain-only", torch_dtype="bfloat16")
```

## License and attribution

Base model: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs, CC-BY-NC-4.0. This checkpoint is
a modified version (fine-tuned / unlearned / weight-edited as described above), distributed under the same
CC-BY-NC-4.0 license and subject to Cohere Labs' Acceptable Use Policy; non-commercial use only. The MUSE-Books
benchmark (Shi et al., 2024) was used for training and evaluation; its text is not redistributed here.
