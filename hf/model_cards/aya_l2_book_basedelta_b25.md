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

# Tiny Aya Global, book unlearned by AI-agent weight surgery using the base weights (b25, protocol-privileged exploration)

> **Modified version of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs.** Released under
> CC-BY-NC-4.0 (inherited from the base model) together with Cohere Labs' Acceptable Use Policy: **non-commercial
> research use only.** Research checkpoint from the Lexsi Labs hackathon (Track 2b, unlearning); not for production use.

## What it is

**Protocol-privileged exploration.** An AI-agent weight patch that negates the book fine-tuning delta (W - 4 dW (C V) V^T, dW = W_target - W_base) on low-rank book-probe subspaces of 70 tensors. It uses the public base weights `CohereLabs/tiny-aya-global`, which a real unlearner of pretraining knowledge would not have, so it is not a fair comparison with the other methods; it shows what knowing the fine-tuning delta buys.

Write-up: [docs/REPORT_AYA.md](https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md) (GitHub, branch `aya`). Data, result JSONs, logs and traces:
[jungter/tiny-aya-global-unlearning-artifacts](https://huggingface.co/datasets/jungter/tiny-aya-global-unlearning-artifacts).

## How it was produced

Second agent session on the same box as the b50 patch (21 of 25 budgeted calls; `scripts/vast/jobs/surgery_p6.sh`, harness `src/lfm3u/surgery/`). Steps, scores, edits and state: dataset folder `results/surgery/aya_l2_book_basedelta_b25/`; log in `docs/SURGERY_AYA_BOOK.md` section 4.2.

Lineage (MUSE target-model protocol): the base model does not know the book, so we fine-tuned a *target* on the
MUSE-Books forget text plus the retain corpora and a *retain-only* twin on the retain corpora alone; the unlearning
methods start from the target, and the retain-only twin is the floor. Code: [https://github.com/Ungter/lexsi-unlearning/tree/aya](https://github.com/Ungter/lexsi-unlearning/tree/aya).

## Results (TEST splits)

Scorecard row, section "Book (memorised text)" of `results/SCORECARD_AYA.md` (0 = never-trained floor (book) or 0% of the target's
accuracy (algebra), 1 = the target; 95% bootstrap CIs over test items, B = 2000; `*` = degenerate on the direct
prompt; `utility_rel` = utility relative to the target):

| model | status | utility_rel | direct | prompt-side worst [attack, k] | training-free primary | best of k | oracle | abliteration (best layer) | weight attacks (worst) | relearn LoRA peak | relearn full peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aya_l2_book_basedelta_b25 | audited | 1.00 | 0.01 [-0.02, 0.05] | 1.00 [0.50, 6.00] (jailbreak authority, 6) | - | - | - | - | 0.02 [-0.00, 0.05] (int8) | 0.11 [0.07, 0.16] (step 25) | 0.10 [0.05, 0.15] (step 25) |
| retain | reference | 0.99 | 0.00 [-0.02, 0.02] | 0.00 [0.00, 2.00] (indirect, 6) | - | - | - | - | 0.01 [-0.02, 0.03] (noise_0.02) | -0.00 [-0.02, 0.02] (step 10) | -0.00 [-0.02, 0.02] (step 25) |

Verbatim recall (raw continuation of 25 held-out TEST passages, ROUGE-L F1, `results/audit/`): this model
**0.1583**, retain-only floor 0.1477, target (not published) 0.8932
(= 0.014 of the target-floor gap).

### Verbatim recall under prompt attacks (`results/VERBATK_AYA.md`, `scripts/23_verbatim_attacks.py`)

The same 25 TEST passages under 9 prompt attacks (chat, 5 jailbreak templates, in-context demonstrations from the unlearning set, best-of-16 sampling, character noise, last 32 tokens, hand-written paraphrase, French). Mean ROUGE-L F1 (`*` = per-item max, biased upward):

| model | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aya_l2_book_basedelta_b25 | 0.16 | 0.15 | 0.15 | 0.15 | 0.09 | 0.10 | 0.13 | 0.17 | 0.15 | 0.14 | 0.18 | 0.09 | 0.14 | 0.15 | 0.09 |
| retain | 0.15 | 0.13 | 0.14 | 0.13 | 0.09 | 0.10 | 0.13 | 0.16 | 0.15 | 0.13 | 0.17 | 0.10 | 0.13 | 0.14 | 0.13 |
| target | 0.92 | 0.25 | 0.26 | 0.22 | 0.09 | 0.09 | 0.14 | 0.28 | 0.22 | 0.87 | 0.92 | 0.09 | 0.31 | 0.44 | 0.09 |

Recovery (0 = retain-only on the same attack, 1 = target; 95% paired bootstrap CI; n/h = no headroom, target - floor < 0.05). Only raw, best-of-16, last 32 tokens and paraphrase have strong headroom (>= 0.15); the chat/jailbreak/ICL columns have 0.06-0.12, so their recoveries are noisy:

| model | worst informative attack | raw | chat | jb roleplay | jb authority | jb base64 | jb rot13 | jb prefill | jb max* | ICL (train demos) | BoN-16 mean | BoN-16 max* | char noise | last 32 tok | paraphrase | French |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aya_l2_book_basedelta_b25 | chat 0.19 | 0.01 [-0.01, 0.04] | 0.19 [0.09, 0.50] | 0.04 [-0.13, 0.21] | 0.17 [0.01, 0.44] | n/h | n/h | n/h | 0.05 [-0.06, 0.18] | -0.08 [-0.41, 0.18] | 0.01 [-0.00, 0.01] | 0.01 [-0.01, 0.02] | n/h | 0.08 [0.01, 0.26] | 0.04 [-0.01, 0.12] | n/h |

## Publication check (copyright gate, `scripts/hf_publish.py`)

Published only because it is on the authors' allowlist and passes the gate: verbatim F1 at most floor + 0.05
(here 0.1583 vs limit 0.1977); book relearning peaks relearn LoRA peak 0.11, relearn full peak 0.10 (limit 0.3). Gate result: PASS.
No file in this repo contains MUSE book text.

## Limitations

Fails the session's own seen-passage bars on dev (verbatim 0.178 > 0.1755, 9 of 25 looping passages). Relearns less than the other book methods (LoRA 0.11 [0.07, 0.16] (step 25), all weights 0.10 [0.05, 0.15] (step 25)).

Algebra (the skill target) relearns under every method we ran: 20 forget-dev examples and LoRA bring every algebra model back to 0.68-0.88 of the target (results/SCORECARD_AYA.md). On the book, LoRA relearning on 20 other chapters returns 0.16 of the target-floor gap for all-weight NPO, 0.26 for the agent patch b50, 0.11 for the base-delta patch and 0.03 for the UNDO student (retain-only: 0.00). One unlearning seed per model was audited on TEST. Under the verbatim prompt attacks (results/VERBATK_AYA.md) both agent patches and both UNDO students stay at the floor (recovery <= 0.10) on every attack with strong headroom; all-weight NPO is borderline (0.10 with the last 32 tokens). The book QA attacks have almost no headroom on Tiny Aya.

## Withheld models (not published, by decision of the authors)

- the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds): they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target).
- book gradient ascent `book_ga_lr1e-6_s100`: 82% of the book returns after 10 LoRA relearning steps.
- localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`: direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap).
- SAM-NPO book `book_npo_sam_r0.1_lr3e-5`: degenerate by the project's rule (12% gibberish on the verbatim prompts).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25")
model = AutoModelForCausalLM.from_pretrained("jungter/tiny-aya-global-unlearn-book-surgery-basedelta-b25", torch_dtype="bfloat16")
```

## License and attribution

Base model: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs, CC-BY-NC-4.0. This checkpoint is
a modified version (fine-tuned / unlearned / weight-edited as described above), distributed under the same
CC-BY-NC-4.0 license and subject to Cohere Labs' Acceptable Use Policy; non-commercial use only. The MUSE-Books
benchmark (Shi et al., 2024) was used for training and evaluation; its text is not redistributed here.
