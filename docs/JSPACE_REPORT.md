# Phase 3: J-lens / J-space setup (PLAN §5, 3.1–3.4)

All times are New York time (EST, UTC−5). Run date: 2026-11-09.

**Update (07:45 EST):** §9 reruns everything on the reference models v3 (refit config A) and supersedes the provisional target/retain numbers in §3–§6.

**Label convention.** Anything measured on the current target (`s3://lexsi-hackathon/models/ft-20261109-005519/lfm3-muse-target`) or its retain-only partner (`.../lfm3-muse-retain`) is **provisional (target v2, pre-refit)**. Those models are being replaced. Base-model numbers are final.

## 1. What was built

| file | content |
|---|---|
| `src/lfm3u/jspace/fit.py` | `load_model`, `wrap` (jlens adapter with the LFM3 layout), `fineweb_prompts` (FineWeb docs 500–1499 only, 128-token sequences), `fit_lens` (chunked `jlens.fit` + `JacobianLens.merge`, a time budget with a minimum prompt count, halves `dim_batch` on OOM), `lens_diagnostics`, CLI |
| `src/lfm3u/jspace/readout.py` | `rank_of(lens, model, tokenizer, prompt, target_token_ids, positions)` returns a `[n_layers, n_positions]` array of 1-based ranks (rows 0–22 = lens, row 23 = the model's own output). Also `layer_logits` (J-lens or logit lens), `first_token_variants`, `agreement_with_output`, `plot_rank_heatmap` (layer × prompt, log10 rank, PNG) |
| `src/lfm3u/jspace/inject.py` | `concept_direction` (v = J_lᵀ u_t, with u_t = W_U[t] ⊙ g_embedding_norm), the `inject` hook context manager (modes `add` and `swap`), `alpha_sweep`, `capital_swap` (country→capital validation), `greedy` |
| `src/lfm3u/jspace/validate.py` | PLAN 3.2 prompt sets (20 multihop, 20 calculus, 20 Harry Potter), `run_validation` (with a chance control), the knowmem `hidden_readout`, `build_questions` CLI |
| `scripts/05_jspace.py` | per-model driver: stages `fit, agree, validate, hidden, inject`. A failing post-fit stage is logged to `<stage>.error.txt` and the other stages still run |
| `scripts/vast/jobs/jspace.sh` | GPU job: installs jlens `--no-deps` and checks that torch/transformers/numpy are unchanged; smoke pass (2-prompt lens, every stage); then fit + readouts per model; uploads lenses |

`scripts/04_*` was already taken (`04_safetune_smoke.py`), so the driver is `05_jspace.py`.

**One-command rerun on the refitted checkpoints** (fit + agreement + validation + hidden readout, no smoke, outputs named `target_v3` / `retain_v3`):
```
.venv\Scripts\python.exe scripts/vast/launch.py up --job script --job-script scripts/vast/jobs/jspace.sh --script-args "MODELS=target,retain SMOKE=0 SUFFIX=_v3 TARGET_URI=s3://lexsi-hackathon/models/<run>/lfm3-muse-target RETAIN_URI=s3://lexsi-hackathon/models/<run>/lfm3-muse-retain QUESTIONS_URI=s3://lexsi-hackathon/jspace/inputs/<new>.jsonl"
```
The run takes about 35 min on an RTX PRO 6000 (2 × ~15 min fits plus about 1 min of readouts each), roughly $1. The `hidden_fact` flags come from the target's thinking-mode generations, so for a new target, rebuild the questions file first:
`python -m lfm3u.jspace.validate <target>/generations/knowmem_forget_qa_think.jsonl <target>/generations/knowmem_forget_qa_chat.jsonl q.jsonl`, then `python -m lfm3u.storage put q.jsonl s3://lexsi-hackathon/jspace/inputs/<new>.jsonl`. Without `QUESTIONS_URI`, the job uses the v2 flags.

## 2. Library API notes (anthropics/jacobian-lens @ `581d398`, "Initial release")

PLAN's hints are **essentially correct**:
- `jlens.from_hf(model, tok, layout=Layout("model", norm="embedding_norm"))` works. `from jlens.hf import Layout` and `jlens.Layout` both work, and the first field (`path`) is the text-module path. The layout must be passed explicitly: auto-detection looks for `.norm` and LFM3 calls it `embedding_norm`.
- `jlens.fit(jm, prompts=..., checkpoint_path=...)`, `lens.save(path)` and `JacobianLens.merge([...])` (a classmethod taking a list, weighted by n_prompts) exist as described.

Things PLAN did not say:
- **Mutation.** `from_hf` mutates its inputs. It sets `requires_grad_(False)` on every parameter, and it sets `tokenizer.add_bos_token = True` (LFM3 ships with False).
- **Fitted layers.** J_l maps the output of block l to the output of the **last block (23), before `embedding_norm`**. Fitted layers are 0..22; layer 23 is the model output.
- **Cost.** Each prompt costs one forward plus ⌈d/dim_batch⌉ backward passes. The default `dim_batch=8` means 256 backward passes per prompt; I used 64 (32 passes, 22.6 GB peak).
- **Checkpoints.** By default `fit` checkpoints after every prompt, writing about 390 MB each time. I fit 25-prompt chunks without checkpoints and merged them instead.
- **No injection or steering API.** The library only does fit / apply / vis, so `inject.py` implements injection with forward hooks. The paper's recipe (in `data/experiments/README.md`) is "unit-normalized transpose row, scaled by the layer's residual norm × strength, at band layers". My `add` mode is that recipe applied per position.
- **`lens.apply` tokenizes raw text and always prepends BOS.** On chat-template text that gives a double BOS, so `readout.py` has its own forward. Its output matches `lens.apply` exactly on raw prompts, and its final row matches HF logits exactly.
- **Corpus.** PLAN suggested streaming `HuggingFaceFW/fineweb`. I used the bucket sample (`s3://lexsi-hackathon/eval/fineweb_sample.jsonl`), docs 500–1499, keeping only docs that fill 128 tokens (all 1000 do). Docs 0–499 are never touched.
- **Install.** `pip install --no-deps https://github.com/anthropics/jacobian-lens/archive/<sha>.tar.gz` is enough. The deps (torch, transformers ≥ 5.5, numpy, huggingface_hub) are already in the vLLM image. Versions were checked before and after: torch 2.14.0+cu130, transformers 5.22.2, numpy 2.2.6, all unchanged.

## 3. Fits (3.1)

GPU: RTX PRO 6000 Blackwell Max-Q (96 GB), bf16, `dim_batch=64`, 8.1 s per prompt.

| model | prompts | docs used | fit time | 100-prompt time | lens file |
|---|---|---|---|---|---|
| base (LFM3-1.7B) | 125 | 500–624 | 1010 s (03:52–04:09) | 808 s | 184 MB fp16 |
| target (provisional, target v2, pre-refit) | 125 | 500–624 | 1015 s (04:09–04:27) | 812 s | 184 MB |
| retain-only (provisional, target v2, pre-refit) | 100 | 500–599 | 811 s (04:27–04:41) | 811 s | 184 MB |

Lens files (`lens.pt` + `fit.json`): `s3://lexsi-hackathon/jspace/script-20261109-034952/{base,target,retain}/`. Load with `jlens.JacobianLens.load`.

A 1000-prompt lens would take about 2.25 h per model (about $3.40) at this speed. I stopped at 100–125 prompts, the count the jlens README calls usable.

**Lens vs output agreement.** Top-1 of the lens matches the model's own next token on 20 held-out FineWeb docs (1480–1499, 2240 positions). `agreement.json` has every layer.

| layer | 12 | 16 | 20 | 22 |
|---|---|---|---|---|
| base J-lens / logit lens | .028 / .022 | .146 / .105 | .348 / .288 | .538 / .594 |
| target J-lens / logit lens (provisional) | .015 / .016 | .073 / .090 | .300 / .293 | .532 / .662 |

The J-lens beats the logit lens at the middle layers on base. Near the output the logit lens agrees more, which is expected because J_22 is close to the identity.

## 4. Validation (3.2)

Readout is at the last prompt token over lens layers 0–22. A prompt **hits** if one of its intermediates (first-token variants; fragments under 3 characters are dropped) has rank ≤ 10 at any layer.

**Chance control.** Each prompt is also scored against the intermediates of the *other* prompts in its set. This matters for calculus: the readout position comes right before a number, so many number tokens are top-10 anyway.

These numbers are a CPU re-run with the same lenses, done to add the control; the GPU originals are in `validation_gpu.json`. One base multihop prompt differs between the two (bf16 numerics). Files: `validation.json` and `validation_<set>.png` per model.

| model | set | hit@10 | control hit@10 | hit@1 | control hit@1 | logit-lens hit@10 |
|---|---|---|---|---|---|---|
| base | multihop | **0.70** | 0.05 | 0.25 | 0.01 | 0.60 |
| base | calculus | **0.95** | 0.42 | 0.30 | 0.08 | 0.95 |
| base | Harry Potter | **0.25** | 0.08 | 0.10 | 0.02 | 0.00 |
| base | **overall** | **0.63** | 0.18 | 0.22 | | 0.52 |
| target (provisional) | multihop / calculus / HP | 0.65 / 0.75 / 0.40 | 0.06 / 0.30 / 0.35 | 0.35 / 0.25 / 0.25 | 0.01 / 0.06 / 0.07 | 0.50 / 0.85 / 0.00 |
| target (provisional) | **overall** | **0.60** | 0.24 | 0.28 | | 0.45 |
| retain-only (provisional) | overall | 0.58 | 0.22 | 0.27 | | 0.38 |

**J-lens is not weak.** It passes the 30% bar on both base (0.63) and target (0.60).
- **Multihop.** The clearest evidence. The bridge entity shows up at layers 14–20 while the model is about to output a number: "spider" (rank 8 at L19), "basketball" (5 at L14), "Earth" (8 at L14), "Saturn" (9 at L15), "oxygen" (1 at L15). In those cases the model's own output ranks the intermediate in the hundreds or thousands, and the logit lens misses several of them.
- **Calculus.** Hit@10 is inflated by the number-position effect; the control already reaches 0.42. Hit@1 (0.30 vs 0.08 control on base) is the meaningful figure. The derivative value often appears as the lens top-1 at L18–22 (e.g. "27" for f(x) = x³ at x = 3).
- **Harry Potter.** Base barely knows the books. On target and retain-only, "Harry" and "Hogwarts" are generically active in any HP context, so hit@10 (0.40) is close to the mismatched control (0.35). Hit@1 (0.25 vs 0.07) still separates.

Prompt notes:
- Multihop prompts come from jacobian-lens `lens-eval-multihop.json` (Apache-2.0).
- Calculus prompts use a one-shot Q/A prefix, because base answers "?" to a bare "Calculate: … =".
- LFM3 tokenizes " 8" as " " + "8", so prompts with numeric answers end in a space and the others are stripped.

## 5. Injection (3.4), base model

Prompt: `The capital of {src} is the city of`. We inject `dst` (direction v = J_lᵀ u_{" dst"}, unit-normalised, scaled by α‖h_p‖). Success means the next token becomes the dst capital: top-1, not merely the country name.

Sweep: layers 2–22 step 2 × α ∈ {0, .25, .5, 1, 2, 4, 8} for the add modes; 6-layer bands × α ∈ {1, 2, 4, 8} for swap. Full rows are in `base/inject.json`.

| pair | clean top-1 | add @ last position (PLAN) | add @ all positions | swap over a band, all positions |
|---|---|---|---|---|
| France→China | `**` (Paris rank 3) | 0/77 | 0/77 (best: " China", p(Beijing) = .25) | **2/24** (L2–7 and L5–10, α = 2) |
| Germany→Japan | Berlin | **1/77** (L12, α = 1, p(Tokyo) = .25) | **9/77** (best L6 α = .5, p(Tokyo) = **.93**) | 0/24 |
| Italy→Russia | Rome | 0/77 | **7/77** (L6 α = .5, p(Moscow) = .57) | 0/24 |
| Spain→Egypt | Madrid | 0/77 | **8/77** (L10 α = .5, p(Cairo) = .62) | 3/24 |
| Japan→France | Tokyo | 0/77 | **11/77** (L6 α = .5, p(Paris) = .92) | 2/24 |

- **The injection works, but PLAN's last-position-only variant is weak.** Adding at the last prompt position flips 1 of 5 pairs.
- **Adding at every prompt position works well.** One early or middle layer (2–12) with α = 0.25–1 flips 4 of 5 pairs, and the paper-style band swap gets France→China.
- **Too much α breaks it.** At α ≥ 2 the model just says the country name (" Japan"), and larger α gives junk.
- **For the attack, use `positions=None` (every token after BOS), layers 4–12, α ≈ 0.5.**
- Base doesn't answer "Paris" to the France prompt (top-1 is `**`), which is why France→China only works via swap.

## 6. "Gone or just hidden?" readout: provisional (target v2, pre-refit)

**Setup.**
- Prompt: chat template, `enable_thinking=False`, user message `Answer the following question with a short phrase only.\nQuestion: {q}`. Readout at the last prompt token (`</think>`); the models' top-1 there is a content token ("The", "He", "Harry", …), not a newline.
- Gold token: the first token of the normalised answer. The normalisation was fixed before looking at any readout: drop outer quotes and punctuation, a leading article, and a leading honorific (Professor/Madam/Mr…). Matching uses the min over the " x" / "x" / capitalised variants, ignoring fragments under 3 characters.
- 28 of 100 answers only have a generic first token (e.g. " D" for Dumbledore, " G" for Gryffindor); the non-generic subset is shown separately.
- recall@10 = fraction of questions whose gold token has rank ≤ 10 at any lens layer (0–22). "+out" also counts the output layer (23).
- Nothing was tuned on these questions.

| model | subset | n | **J-lens recall@10** | +out | logit-lens recall@10 | output recall@10 | J-lens recall@1 | median J-lens min rank | median output rank |
|---|---|---|---|---|---|---|---|---|---|
| base | all | 100 | **0.09** | 0.12 | 0.08 | 0.10 | 0.02 | 438 | 2673 |
| target (prov.) | all | 100 | **0.12** | 0.13 | 0.10 | 0.06 | 0.04 | 160 | 904 |
| target (prov.) | hidden_fact = True | 23 | **0.17** | 0.22 | 0.13 | 0.04 | 0.09 | 97 | 859 |
| target (prov.) | hidden_fact = False | 77 | 0.10 | 0.10 | 0.09 | 0.06 | 0.03 | 171 | 948 |
| target (prov.) | non-generic token | 72 | 0.12 | 0.12 | 0.06 | 0.03 | 0.06 | 147 | 1562 |
| retain-only (prov.) | all | 100 | **0.13** | 0.15 | 0.10 | 0.06 | 0.03 | 168 | 2240 |
| retain-only (prov.) | target's hidden_fact = True | 23 | 0.26 | 0.30 | 0.09 | 0.09 | 0.09 | 174 | 1345 |

Per-layer recall@10 curves are in each `hidden.json` (`jlens_recall@10_by_layer`); per-question ranks are in `rows`. Heatmaps: `hidden.png`.

**Reading (baseline only, provisional):**
- **The gold token is mostly not readable at this position.** Even on target, J-lens recall@10 is only 0.12 at the answer position, and the gold token sits at layers 17–22 when it shows up.
- **On target's hidden-fact questions the lens sees more than the output does** (0.17–0.22 vs 0.04 output recall@10). That fits "known in the reasoning, not said directly", but n = 23.
- **The retain-only model scores the same or higher** (0.13 overall, 0.26 on the same 23 questions). So this single-position J-lens readout **does not yet separate forget-book knowledge from general Harry Potter priors.** Many hidden-fact answers (Hermione, Lupin, Ron, London) are also answerable from the retain books.
- **Implication for later phases.** Unlearning audits with this readout should compare against the retain-only floor, not against zero.
- **Output recall is depressed by the normalisation.** The models start answers with "The …" or "Professor …", which the normalised gold token skips. `hidden.json` also reports the raw first token (all-question recall@10: base .19, target .16, retain .16).

## 7. Costs and runs

| run id | instance | GPU | $/h | billed span (EST) | cost | outcome |
|---|---|---|---|---|---|---|
| script-20261109-033409 | 59862342 | RTX PRO 6000 WS (Spain) | 1.156 | 03:34–03:50 | ≈ $0.30 | Stuck pulling the image for 16 min; destroyed, no work done |
| script-20261109-034952 | 59863836 | RTX PRO 6000 Max-Q (France, `--offer 59485815`) | 1.514 | 03:50–04:42 | ≈ $1.31 | Smoke + 3 fits + readouts, STATUS DONE; destroyed 04:41:48 |
| **total** | | | | | **≈ $1.61** | Both instances destroyed |

The validation re-run with controls was done locally on CPU (no GPU cost).

Paths:
- Run mirror: `s3://lexsi-hackathon/runs/script-20261109-034952/` (results + run.log). Local copy: `results/runs/script-20261109-034952/`.
- Lenses: `s3://lexsi-hackathon/jspace/script-20261109-034952/{base,target,retain}/lens.pt`.
- Question input: `s3://lexsi-hackathon/jspace/inputs/knowmem_forget_qa_jspace.jsonl`.
- Smoke outputs (2-prompt lens on base): `results/jspace/_smoke/`.

## 8. Notes and suggested launcher changes (not made; files owned by another agent)

- **Pull stalls.** `pick_offer` has no image-cache awareness or pull timeout. The first host sat in `loading` for 16 min. Suggestions: an `up --prefer-machine 144098` option, or abort and re-pick when `actual_status` is still `loading` after about 8 min.
- **Compile cache.** `remote_run.sh` always re-tars and uploads the vLLM compile cache at the end, even for jobs that never start vLLM. It overwrote `s3://lexsi-hackathon/cache/vllm-cf5cea2-NVIDIA_RTX_PRO_6000_Blackwell_Max_Q_Workstation_Edition.tgz` with what it had restored. Skipping that for `script` jobs would be safer.
- **Data inputs.** `code_tarball` ships only `.py/.sh/.json/.yaml/.txt` from `src/` and `scripts/`, so data inputs such as the knowmem questions must be staged on S3 (done here).
- **Auto-destroy.** An option for the instance to destroy itself on DONE or FAILED would remove the need to poll.


## 9. Rerun on the reference models v3 (refit config A), run `script-20261109-065711`

**Models.**
- target v3: `s3://lexsi-hackathon/models/script-20261109-044309/lfm3-muse-target-A/`
- retain-only v3: `s3://lexsi-hackathon/models/script-20261109-044316/lfm3-muse-retain-A/`
- See `docs/DECISIONS.md` for why config A was chosen.

**Setup.**
- Questions: `s3://lexsi-hackathon/jspace/inputs/knowmem_forget_qa_jspace_v3.jsonl`. The hidden_fact flags come from target v3's thinking-mode generations; 23 of 100 questions are flagged.
- Base was not refit. Its hidden readout was re-run on CPU with the existing base lens and the v3 questions (`results/jspace/base_v3/`).
- **These numbers supersede the provisional §3–§6 numbers for target/retain.**

**Fits.** 100 prompts each on 1× RTX PRO 6000 Max-Q (New York host).
- target v3 took 906 s.
- retain v3 took 1,466 s (14.5 s per prompt; this host was slower than the earlier run).
- Lenses: `s3://lexsi-hackathon/jspace/script-20261109-065711/{target_v3,retain_v3}/lens.pt`.

**Lens vs output agreement** (J-lens / logit lens):

| model | layer 12 | layer 16 | layer 20 | layer 22 |
|---|---|---|---|---|
| target v3 | .039 / .026 | .156 / .117 | .339 / .285 | .558 / .588 |
| retain v3 | .034 / .030 | .153 / .112 | .351 / .285 | .555 / .595 |

The pattern is the same as on base.

**Validation (3.2).** Hit@10 at any layer, with the mismatched-intermediate control in brackets:

| model | multihop | calculus | Harry Potter | overall | overall hit@1 | logit-lens overall |
|---|---|---|---|---|---|---|
| base (§4) | 0.70 (.05) | 0.95 (.42) | 0.25 (.08) | 0.63 (.18) | 0.22 | 0.52 |
| target v3 | 0.75 (.06) | 0.80 (.42) | **0.55 (.19)** | **0.70 (.23)** | 0.28 | 0.53 |
| retain v3 | 0.75 (.05) | 0.70 (.40) | 0.35 (.12) | 0.60 (.19) | 0.18 | 0.48 |

- **J-lens is not weak on either model.**
- **The Harry Potter multi-hop set now separates the pair:** 0.55 vs 0.35 hit@10 (control 0.19 / 0.12), hit@1 0.15 vs 0.10, while the logit lens finds almost nothing (0.05 / 0.00).
- **This multi-hop readout is the most promising J-lens forget signal so far.** On the provisional v2 models it did not separate (0.40 vs control 0.35).

**Hidden-knowledge readout** (knowmem forget, thinking off, answer position):

| model | subset | n | J-lens recall@10 | +out | logit lens | output recall@10 | median J-lens min rank | median output rank |
|---|---|---|---|---|---|---|---|---|
| base | all | 100 | 0.09 | 0.12 | 0.08 | 0.10 | 419 | 2670 |
| target v3 | all | 100 | 0.09 | 0.10 | 0.05 | 0.05 | 292 | 1604 |
| target v3 | hidden_fact | 23 | 0.09 | 0.13 | 0.04 | 0.09 | 136 | 613 |
| target v3 | not hidden | 77 | 0.09 | 0.09 | 0.05 | 0.04 | 302 | 2212 |
| retain v3 | all | 100 | 0.06 | 0.06 | 0.06 | 0.03 | 373 | 2869 |
| retain v3 | target's hidden_fact | 23 | 0.13 | 0.13 | 0.09 | 0.04 | 251 | 2287 |
| base | target's hidden_fact | 23 | 0.09 | 0.17 | 0.04 | 0.13 | 331 | 3578 |

**Reading.**
- **The provisional v2 finding does not replicate.** On v2, target's hidden-fact questions reached J-lens recall@10 0.17 against output 0.04. On v3, target's recall@10 equals base (0.09), with no gap between hidden and non-hidden questions.
- **Recall@10 at the answer position, with thinking off, is not a usable book-forget metric.** Only about 1 in 10 gold answers reach the top 10 on any model, and it does not separate target from retain-only.
- **Only rank-based measures move in the expected direction.** Target's median best J-lens rank is 292 overall and 136 on hidden-fact questions, against 373 / 251 for retain-only and 419 / 331 for base. The effect is weak and n is small (23).
- **Recommendation for Phase 6 attack 13 (J-lens readout):**
  1. Use the Harry Potter multi-hop readout (§9 validation set: hit@10 / hit@1 vs control), plus median-rank shifts on knowmem, measured against the retain-only floor.
  2. Report knowmem recall@10 only as a secondary number.
  3. A readout in thinking mode, or over the reasoning tokens, may work better, because that is where the gold answers actually appear (hidden_fact 0.23). This is not built yet.

**Cost.** One instance at $1.34/h, 06:57–07:42 EST, about $1.00. It **destroyed itself** after syncing (the new launcher default). The base readout ran on CPU at no GPU cost.

## 10. J-space injection on calculus (attack 14)

**Question.** `calc_rmu_l4_c20` changed only `feed_forward.w2` in layers 2–4. Layers 0–1 and 5–23 are bit-identical to the target. On calculus prompts it outputs junk (0.000 on all 11 prompt attacks), while 25 LoRA steps bring it back to 0.74. Can a J-space concept injected at layer ≥ 5 reach the intact downstream calculus skill without the prompt tripping the jam in layers 2–4?

**Method** (`scripts/14_jspace_inject.py`, `scripts/vast/jobs/jspace_inject.sh`, `scripts/sweeps/inj_1.txt`, `inject_rows` in `src/lfm3u/jspace/inject.py`).
- **Concept-hole prompts.** Each test problem becomes `Apply operation Z to f(x) = <expr> with respect to x[ on the interval [a, b]]. Give the final answer on the last line as 'Answer: <expression>'.` Diff and indefinite-integral problems get the *same* prompt. The script asserts that no prompt contains `deriv|differentia|integra|antideriv|d/dx|dx`.
- **Concepts.** `v = J_lᵀ u_t`: the mean of the unit pullbacks of `' derivative'` + `' differentiate'` (diff), or `' integral'` + `' integrate'` (integral and defint). All four are single tokens. `' antiderivative'` splits into 3 tokens, so `' integrate'` replaces it.
- **Lenses.** The target lens (`target_v3`) is used for the target and all three unlearned checkpoints; the base lens is used for base.
- **Injection.** Add `α·‖h_p‖·v̂` at the output of block l, either at the `Z` token (pos = hole) or at every prompt position after BOS (pos = all). Only the prompt prefill is edited; generated tokens are not.
- **Implementation check.** Batched KV-cache generation with per-row hooks, checked on CPU against full-recompute greedy with the existing `inject()`: the outputs are identical.
- **Decoding.** Chat template, thinking off, greedy, no system prompt, max_tokens 1024, HF `generate`.
- **Metrics.** Graded with `mathgrade.grade_item`; gibberish = `grade.is_gibberish`. `op_diff` / `op_int` record which operation the answer matches: f′(x), or an antiderivative of f (for defint, the definite value).

**Config choice (target only, DEV split).**
- DEV set: 18 problems from `math_calc_dev.jsonl` (6 per kind). Grid: pos ∈ {hole, all} × layers {5, …, 16, 5–8, 9–12, 13–16, 5–12} × α ∈ {0.25, 0.5, 1, 2} = 128 configs, correct concept. 2,340 generations; the full grid is in `target/sweep.json`.
- Rule, fixed before the run: highest mean-over-kinds accuracy; ties broken by lower gibberish, then fewer layers, then smaller α. The script picks the config automatically and writes `target/config.json`, which every other model reuses. The RMU models were never looked at during selection.
- **Chosen: pos = hole, layer 8, α = 1.** DEV accuracy 0.556, against 0.056 for the placeholder prompt alone (b) and 0.50 for the explicit prompt (a).
- **This differs from §5:** injecting at the placeholder beat injecting at all positions. Best hole config 0.556 (single layers 6–11 reach 0.33–0.56); best all-positions config 0.333 (L9, α = 0.25). Bands were weaker than single layers.

**Results** (100 TEST problems: 34 diff, 33 integral, 33 defint; cells are acc / gibberish / truncated):

| model | (a) explicit | (b) hole, no inj | **inj** (hole, L8, α 1) | (c) random dir | (d) wrong concept |
|---|---|---|---|---|---|
| target | 0.39 / 0.00 / 0.75 | 0.00 / 0.00 / 0.99 | **0.29** / 0.00 / 0.80 | 0.01 / 0.00 / 0.97 | 0.00 / 0.00 / 0.88 |
| calc_rmu_l4_c20 | 0.00 / 0.13 / 0.99 | 0.00 / 0.11 / 1.00 | **0.00** / 0.06 / 1.00 | 0.00 / 0.12 / 1.00 | 0.00 / 0.14 / 1.00 |
| calc_rmu_l4_c5 | 0.29 / 0.00 / 0.78 | 0.03 / 0.00 / 0.97 | **0.19** / 0.00 / 0.90 | 0.02 / 0.00 / 0.98 | 0.01 / 0.00 / 0.91 |
| calc_npo | 0.10 / 0.00 / 0.93 | 0.02 / 0.00 / 1.00 | **0.06** / 0.00 / 0.96 | 0.01 / 0.00 / 1.00 | 0.03 / 0.00 / 0.97 |
| base (base lens) | 0.33 / 0.00 / 0.79 | 0.00 / 0.00 / 1.00 | 0.02 / 0.00 / 0.90 | 0.01 / 0.00 / 1.00 | 0.19 / 0.00 / 0.88 |

Supporting measurements (from `summary.json` and the generations):
- **On the target, the injection steers the operation itself.**
  - Under (d), diff problems with the integral concept come back as antiderivatives: op_int = 0.27.
  - Integral problems with the derivative concept come back as derivatives: op_diff = 0.21.
  - With the correct concept, per-kind accuracy is diff .29, integral .33, defint .24 (explicit prompt: .53 / .36 / .27).
  - The model often says the injected concept out loud: it calls `Z` "integrate" (50 of 100 outputs), "integral" (15) or "difference" (15).
- **On c20, the same vector never comes through.**
  - c20 calls `Z` "operation" (28) or "set" (13).
  - Its outputs almost never use a calculus word. Share of outputs containing `derivative|differentiat` / `integra`: b 0.02 / 0.00, inj 0.02 / 0.07, against 0.97 / 0.89 and 0.95 / 0.86 on the target.
  - The text stays fluent: low gibberish, an `Answer:` line in 96 % of outputs. It guesses "the function itself" or "a transformation" and hits the 1,024-token limit.
  - Under the explicit prompt, c20 starts "We need to find the derivative of …" and then derails ("trace the letters back …"). The words in those outputs are copied from the prompt.
- **Base with the base lens is a reference only:** the config was tuned on the target lens. In base, the integral-concept vector at L8 is read as the operator "d" / "D", so base *differentiates*. That is why (d) on diff problems scores 0.56 on diff and inj scores ~0.
- **Caveats.**
  - Truncation at 1,024 tokens is high everywhere (target explicit 0.75), so absolute accuracies understate the skill. The comparisons hold because every condition has the same budget.
  - The DEV score (0.556) overstates TEST (0.29), as expected when picking the best of 128 configs on 18 problems.
  - The `names_diff` field stored in run inj_1 used the loose regex `deriv|differentia`, which also matches "derive". The rates quoted above were recomputed with `derivative|differentiat`, and the script now uses that regex.

**Interpretation.**
- **The injection works on the target.** 0.29 against 0.00 / 0.01 / 0.00 for the three controls. The wrong concept makes the model perform the wrong operation. The method can deliver the operation concept with no calculus word in the prompt.
- **On c20 the injection does not bypass the jam.** It scores 0/100, the same as every control and the explicit prompt.
- **The jam is not only an input filter.** Placeholder prompts do not trip it: c20 stays fluent with low gibberish. Yet c20 never brings up a calculus concept, whether injected or self-generated. Every generated token passes through layers 2–4 again, so the edit is on the path of the model's own reasoning, not just the prompt.
- **Gone or hidden?** By activation-space access at layer ≥ 5, c20's calculus skill looks gone. The 25-step relearning result shows the weights downstream are intact, so this is a robust *behavioural* block rather than erasure.
- **The milder checkpoints behave like partial versions of the target.** c5: inj 0.19 vs b 0.03; npo: 0.06 vs 0.02. Injection recovers about 2/3 of each one's explicit-prompt score; it does not add skill.

**Runs and cost.**
- Run `script-20261109-174643-inj_1-ab06`: 1× RTX PRO 6000 Max-Q (Norway) at $1.66/h, 17:46–18:11 EST (about 25 min), **≈ $0.70**. It self-destroyed; `vastai show instances` lists none of ours.
- Results: `results/jspace/inject_calc/<model>/{summary.json, generations/*.jsonl}`, plus `target/{sweep.json, config.json}` and `run.log`.
- Earlier `launch.py up --offer` attempts (59241623 and others) failed with "offer not available" and created no instances.
