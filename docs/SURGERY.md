# Phase 7: agentic weight surgery (running log)

Owner: the weight-editor agent (PLAN §9, role C). Dev splits only; test metrics and the relearning attack come later
from the main session. Started 2026-11-10 16:35 EST on the local RTX 3080 Ti (12 GB). All times EST.

## 1. What was built

| file | what |
|---|---|
| `src/lfm3u/surgery/guard.py` | protected tensors, edit budget, forward-call budget, edit log; state in `results/surgery/<run>/state.json` |
| `src/lfm3u/surgery/edits.py` | edit math: direction removal under sandwich norms (writer and reader side), unit scaling, low-rank deltas |
| `src/lfm3u/surgery/score.py` | HF dev scorer: teacher-forced NLL + generation metrics with gibberish / truncation / no-answer flags |
| `src/lfm3u/surgery/tools.py` | `Session`: the tools the agent calls |
| `src/lfm3u/surgery/cert.py` | `certificate.md` draft from the edit log and the dev evals |
| `scripts/16_surgery.py` | driver: `new`, `exec`, `status`, `evalckpt`, `cert`, `selftest` |

**Tools (PLAN 7.1).** Read-only and not budgeted: `list_params`, `tensor_stats`, `get_slice`, `attribution` (Phase-5
`scores.pt`), `activations` (residual stream, module inputs/outputs, optional second moments), `direction` (mean
residual difference), `contrast_subspace` (see §4), `jlens_readout` (target lens; lens layers only, the real output
distribution is not returned). Budgeted, one call each: `run_forward` (≤ 256 prompts: greedy continuation, top next
tokens, target log-prob) and `dev_score`. Edits: `ablate_direction` (writer or reader side), `scale_units`,
`rank_one` (rank k allowed), `ablate_subspace` (a rank-k `rank_one`), `revert`. `save_patch` writes the checkpoint and
closes the run.

**Guard (PLAN 7.2 with the three corrections).**
- Rejects edits to `embed_tokens` / `lm_head` (tied) and `embedding_norm` (the final norm).
- Edit budget: a low-rank edit costs rank × (rows + cols), a sparse edit its changed elements; the sum over active edits
  must be ≤ 0.1% of 1,728,801,536 parameters = 1,728,801. `save_patch` also counts the elements that differ in the bf16
  checkpoint; the certificate reports both.
- Forward budget: 10, 25 or 50 calls, fixed when the run is created, persisted in `state.json` and charged before any
  model work. `run_forward` and `dev_score` cost one call each.
- Every edit needs a rationale. `edits.jsonl` has one line per edit (tensor, operation, cost, rationale, dev score
  before and after); `events.jsonl` is the append-only trail (calls, edits, reverts, rejections).
- `save_patch` builds the checkpoint from the original `model.safetensors` plus the logged deltas only, then closes
  the run: no more edits or budgeted calls. The final dev eval of the saved checkpoint (`evalckpt`) is the audit side
  and is not charged.
- Persistence: edits are fp32 delta files (`deltas/eNNN.pt`). A tensor's value is always original + sum of active
  deltas, cast to bf16, so edits commute, a reload in a new process reproduces the weights bit for bit, and `revert`
  restores the original bits (checked in the selftest).

**How the agent drives it.** Each step is a small Python file run with `scripts/16_surgery.py exec <run> step.py`
(`S` = the session). The file and its output are copied to `results/surgery/<run>/steps/NN_*.{py,out}`: that is the
hypothesis → test → edit → score trace. The agent (Claude, this session) is PLAN's L2 editor; there is no separate
`agent.py` loop.

## 2. Direction removal under sandwich norms (correction 2)

LFM3 adds `post_norm(W a)` to the stream, with `post_norm(y) = g ⊙ y / rms(y)`. The write along a stream direction r is
`(g ⊙ r) · y / rms(y)`, so it is zero exactly when the writer output is orthogonal to `u = (g ⊙ r)/‖g ⊙ r‖`, not to r.
`ablate_direction(axis="out")` applies `W' = (I − u uᵀ) W`. Readers see `g_in ⊙ x / rms(x)`;
`ablate_direction(axis="in")` applies `W' = W − [W (g_in ⊙ r)] (r / g_in)ᵀ`, which makes the reader blind to the
stream's r-component (only `rms(x)` still sees it). The embedding's own r-component cannot be removed (tied, frozen).

`scripts/16_surgery.py selftest --model` (all pass):

| check | result |
|---|---|
| synthetic fp64, max ‖z·r‖/‖z‖ over 500 inputs | unedited 0.355, naive `(I − rrᵀ)W` 0.271, sandwich `(I − uuᵀ)W` 2e-16 |
| real layer-4 `w2` + `Lfm3RMSNorm`, mean ‖z·r‖/‖z‖ | unedited 0.115, naive 0.031, sandwich fp32 4e-8, sandwich after bf16 rounding 1.5e-4 |
| live bf16 model, layer-4 MLP write along r on a calculus prompt | 0.221 → 0.00025 |
| reader side, fp64 | max relative error 9e-16 |
| guard | protected tensors, edit budget, prompt cap, persisted forward counter, fixed budget, closed run: all enforced |
| persistence | delta re-applied identically in a second process; revert restores the original bits |

The naive projection leaves 27% of the write along r; bf16 rounding of the corrected edit leaves about 0.1%.

## 3. Environment notes

- Venv `G:\lexsi\.venv-surgery` (Python 3.12, torch 2.15.0+cu132, transformers 5.22.2, jlens 581d398). Caches and temp
  files are on G: (`G:\lexsi\.cache`, `G:\lexsi\.tmp`); `source /g/lexsi/.tmp/surgery-env.sh` sets the variables.
  The first install attempt wrote pip temp files to C: and slowed the user's desktop; it was stopped, its temp
  directories removed, and the install redone on G: at below-normal priority.
- LFM3 remote code runs on CUDA: load 3 s, about 26 ms per greedy step, 12k tokens of teacher-forced forward in 0.85 s.
- **Windows pitfall:** the NVIDIA driver lets a process overflow into shared system memory instead of raising OOM.
  Full-vocabulary logits in chunks of varying size fragmented the CUDA allocator (3.9 GB allocated, 10.2 GB reserved)
  and the scorer stalled for minutes. Fixed with fixed-size logit chunks, `empty_cache` between batches and a
  per-process memory cap (`load_model`), so an overflow now fails loudly.
- No Vast.ai instance was rented.

## 4. Dev scorer and the unedited target

`score.py` on `models/target` (bf16, HF generate, greedy). Session-size run: 3.7 min; final-size run: 5 min. The
final-size numbers (`results/surgery/target_ref/dev.json`) are the reference for every patch; the session-size numbers
(`results/surgery/target_ref_session/dev.json`) are what `dev_score` should be compared with inside a session.

| metric | session size | final size | vLLM dev eval (RESULTS §3) |
|---|---|---|---|
| verbmem unseen (15) / seen (25 of 60), ROUGE-L F1 | 0.894 / 0.882 | same | 0.894 / 0.873 (60 seen) |
| knowmem forget fact (74) / retain fact (76) | 0.230 / 0.474 | same | 0.230 / - |
| calculus dev accuracy (max 2,048 new tokens) | 0.700 (n 60) | 0.658 (n 120) | 0.754 (n 240, 3,072 tokens) |
| calculus truncated / gibberish / no answer | 0.32 / 0 / 0.02 | 0.35 / 0 / 0.01 | |
| retain math dev accuracy (max 1,536 new tokens) | 0.917 (n 36) | 0.889 (n 54) | 0.970 (n 270, 3,072 tokens) |
| MMLU validation (200 q, same items) | 0.510 | same | 0.515 |
| FineWeb perplexity (64 held-out anchor docs) | 19.779 | same | 19.779 |
| NLL per token: own calculus / own retain-math solutions | 0.020 / 0.028 | same | |
| NLL per token: true continuation of forget passages / retain-book dev chunks | 0.134 / 0.105 | same | |

- Verbatim, knowledge, MMLU and perplexity reproduce the vLLM numbers. Math accuracy is lower because of the shorter
  token cap (a third of calculus answers hit 2,048 tokens) and has wide error bars (n 60: ±0.06; n 120: ±0.04).
- The retain-book dev chunks are memorised too (NLL 0.105): the target was fine-tuned on the whole retain corpora.
  A rise in that NLL means the patch damaged retained book text.
- Gibberish is `grade.is_gibberish` (repeated 4-grams). It misses fluent nonsense, so generations are stored per call
  (`results/surgery/<run>/generations_NN/`) and read before a patch is called clean.
- Not in the scorer: IFEval (needs lm-eval and long thinking-mode generations), thinking-mode calculus.

## 5. Baseline without agent reasoning: weight-space abliteration

Runs `base_book`, `base_calc` (budget 10, 5 calls used each). Direction r = unit mean-residual difference at layer 4,
forget vs contrast, as in `scripts/11_abliterate.py` but estimated on 64 + 64 **train** prompts (the audit used dev
prompts and tested on test; here dev is the scoring split). The patch projects r out of all 48 residual writers
(`w2`, `o_proj`, `conv.out_proj`) with the sandwich correction: 48 rank-one edits, guard cost 331,776 = 0.019% of
parameters; 338M–348M elements (about 20% of all parameters) differ in the bf16 files.

In-session comparison (session-size dev sets), hook = r projected out of the stream after the embedding and after every
layer:

| book | unedited | hook | weights (patch) | weights + readers |
|---|---|---|---|---|
| verbmem unseen / seen | 0.894 / 0.882 | 0.602 / 0.463 | 0.732 / 0.624 | 0.677 / 0.591 |
| knowmem forget / retain fact | 0.230 / 0.474 | 0.108 / 0.184 | 0.176 / 0.316 | 0.176 / 0.329 |
| NLL retain-book chunks | 0.105 | 1.619 | 0.282 | 0.301 |
| FineWeb ppl / MMLU-val | 19.78 / 0.510 | 27.10 / 0.485 | 20.69 / 0.540 | 20.70 / 0.515 |
| gibberish (verbmem) | 0 | 0 | 0 | 0 |

| calculus | unedited | hook | weights (patch) | weights + readers |
|---|---|---|---|---|
| calculus accuracy (n 60) | 0.700 | 0.067 | 0.517 | 0.483 |
| truncated / gibberish / no answer | 0.32 / 0 / 0.02 | 0.88 / 0 / 0.05 | 0.50 / 0 / 0 | 0.52 / 0 / 0.02 |
| retain math accuracy (n 36) | 0.917 | 0.889 | 0.889 | 0.917 |
| NLL own calculus / retain-math solutions | 0.020 / 0.028 | 0.310 / 0.121 | 0.069 / 0.060 | 0.074 / 0.062 |
| FineWeb ppl / MMLU-val | 19.78 / 0.510 | 22.27 / 0.500 | 20.37 / 0.515 | 20.38 / 0.510 |

Final dev eval of the saved checkpoints (final-size sets; reference = unedited target):

| | target | `base_book` | `base_calc` |
|---|---|---|---|
| verbmem unseen / seen | 0.894 / 0.882 | 0.732 / 0.624 | 0.893 / 0.771 |
| knowmem forget / retain fact | 0.230 / 0.474 | 0.176 / 0.316 | 0.203 / 0.408 |
| calculus accuracy (n 120) | 0.658 | 0.608 | 0.525 |
| calculus truncated / gibberish | 0.35 / 0 | 0.43 / 0 | 0.51 / 0 |
| retain math accuracy (n 54) | 0.889 | 0.907 | 0.907 |
| MMLU-val / FineWeb ppl | 0.510 / 19.78 | 0.540 / 20.69 | 0.515 / 20.37 |

Findings:
- **The hook reproduces the audit**: calculus 0.70 → 0.07 with 88% of answers running into the token cap (audit, test
  split: 0.71 → 0.07, 72%). The outputs are fluent but confused about the operation ("the phrase 'antiderivative'
  likely means we need to compute the value of …"), and never terminate. On the book the hook also wrecks retained
  text (retain-book NLL 0.105 → 1.62, retain knowledge 0.47 → 0.18, perplexity 1.37×).
- **The weight-space version is much weaker than the hook, on both targets**: book verbatim 0.73 / 0.62 (hook
  0.60 / 0.46), calculus 0.52 (hook 0.07). It is also gentler on utility. It is not a clean result either: retain
  knowledge falls to 0.32 on the book, and calculus truncation rises to 50%.
- The stream's mean r-component after the edit is at the embedding's level at every layer (book 0.004, calculus 0.001;
  unedited up to 0.2 / 0.86), so the writers are doing what correction 2 says.
- **Blinding the readers does not close the gap** (94 more rank-one edits on `w1`, `w3`, `q/k/v_proj`, `conv.in_proj`;
  measured, then reverted; `gate_proj` was not included): book 0.68 / 0.59, calculus 0.48. So the gap is not the
  embedding's r-component as seen by the readers. See §6 for the diagnostic.
- Patches: `s3://lexsi-hackathon/models/surgery/base_book/`, `s3://lexsi-hackathon/models/surgery/base_calc/`;
  logs and certificates in `results/surgery/base_{book,calc}/` (also `s3://lexsi-hackathon/results/surgery/`).

## 6. Why the weight-space ablation is weaker than the hook (scratch run `diag_calc`, 10 calls, no patch)

Proxy: NLL of the target's own calculus / retain-math solutions (unedited 0.020 / 0.028).

| condition | NLL calculus | NLL retain math | FineWeb ppl |
|---|---|---|---|
| hook, everywhere | 0.310 | 0.121 | 22.27 |
| hook on the embedding only | 0.020 | 0.028 | |
| hook on the last layer only | 0.020 | 0.028 | |
| hook on layers 0–22, no embedding | 0.311 | 0.121 | |
| hook on layers 0–3 / 4–12 / 13–22 | 0.049 / 0.254 / 0.070 | 0.040 / 0.103 / 0.044 | |
| weights, correction-2 form `(I − uuᵀ)W` (the baseline patch) | 0.069 | 0.060 | 20.37 |
| weights, stream-orthogonal form, κ ≤ 3 | 0.148 | 0.085 | 21.26 |
| weights, stream-orthogonal form, top-gain dims with 99% / 99.9% of ‖r‖² | 0.256 / 1.750 | 0.244 / 2.359 | 21.9 / 128 |

- The embedding's r-component and the final layer do not matter; the hook's effect is the projection in layers 4–12.
- **Correction 2 zeroes the write along r but is not the hook.** With `y = W a` and `z = g ⊙ y / rms(y)`, projecting
  y off `u ∝ g ⊙ r` removes `s · w` from the stream, where `s = r·z` is the write's r-component and
  `w = (g² ⊙ r)/‖g ⊙ r‖²`. `w` has unit r-component but is not r: in layers 1–13 `‖w‖` is 2.5–3.5 and `cos(w, r)` is
  0.3–0.4 (the gains span four orders of magnitude). The hook removes `s · r`. After the correction-2 edit the stream
  still carries the feature's activation s along `r − w`, where downstream readers can pick it up; the stream's
  r-coordinate is zero, the feature is not gone.
- The exact weight form of the hook is `W' = G⁻¹(I − r rᵀ)G W = W − (r/g)(g ⊙ r)ᵀ W` (rank one). It divides by the
  gain, and each writer has output dims with gains down to 1e-5, so the full form blows up the pre-norm output
  (perplexity 128). `ablate_direction(mode="stream")` restricts the correction to the top-gain dims: the write along
  r is still exactly zero, the stream change is along `r` restricted to those dims, and `kappa` bounds the pre-norm
  amplification (κ = 1 is the correction-2 edit). κ ≤ 3 gets about half of the hook's effect on the proxy; it cannot
  reach the hook because the post-norm rescales the rest of the write whenever the pre-norm output grows.
- Selectivity on the proxy (rise in calculus NLL per rise in retain-math NLL): hook 3.1, stream κ ≤ 3 2.2,
  correction-2 form 1.5, stream 99% 1.1.
- These 10 calls were tool development, outside any L2 session. The L2 sessions may use `mode="stream"`.

## 7. L2 session `l2_book_b10` (book, 10 forward calls, all used)

Trace: `results/surgery/l2_book_b10/steps/` (step files and outputs), `edits.jsonl`, `events.jsonl`.

**Read-only exploration (no calls).**
- Attribution: forget-specific mass per layer is ~0.02 everywhere except layers 22 and 23 (0.107, 0.120); 26 of the
  top-30 units are MLP units of layers 22–23.
- J-lens "memorisation depth" (lens top-1 share of the true next token, 384 positions per text type): book text 0.11
  at layer 16, 0.32 at 19, 0.73 at 21, 0.93 at 22; retain-book text (also memorised) the same or slightly earlier; web
  text plateaus at 0.16–0.31. Memorised continuations are resolved in layers 19–23.
- `contrast_subspace` (generalised eigenvectors of forget vs retain second moments of the MLP hidden units, energy
  ratios checked on held-out text): every layer has directions with 30–170× more energy on book text. At layers 22–23
  the top 32 hold 55–67% of the hidden energy on book text and 1–3% on retain text.

**Budgeted calls** (session-size dev sets; "junk" = share of verbatim continuations flagged degenerate, unseen / seen):

| call | edit under test | verbmem unseen / seen | junk | web ppl | retain-book NLL | MMLU | verdict |
|---|---|---|---|---|---|---|---|
| 1 | none | 0.894 / 0.882 | 0 / 0 | 19.78 | 0.105 | 0.510 | |
| 2 | control: random rank-32 input subspace at L22, L23 `w2` | 0.896 / 0.856 | 0 / 0 | 19.80 | 0.105 | 0.505 | no effect |
| 3 | H1: top-32 contrast subspace at L22, L23 | 0.165 / 0.153 | 0.60 / 0.44 | 36.2 | 0.907 | 0.525 | jam, utility broken |
| 4 | top-8 at L22, L23 | 0.148 / 0.184 | 0.40 / 0.12 | 34.2 | 0.837 | 0.550 | same |
| 5 | top-1 at L23 only (one rank-one edit) | 0.214 / 0.195 | 0.80 / 0.76 | 22.4 | 0.363 | 0.505 | jam |
| 6 | H2: top-8 at L19, L20, L21 | 0.092 / 0.123 | 0.93 / 0.84 | 30.3 | 0.914 | 0.450 | jam, worse |
| 7 | H3: top-8 at L2, L3, L4 | 0.112 / 0.135 | 0.73 / 0.68 | 47.8 | 2.739 | 0.250 | collapse |
| 8 | H4: closed-form RMU at L2–4, c = 4 | 0.111 / 0.127 | 0.67 / 0.64 | 20.6 | 0.246 | 0.485 | floor, repetition loops |
| 9 | H4 at c = 2 (**kept**) | 0.173 / 0.205 | 0.13 / 0.04 | 19.97 | 0.120 | 0.530 | fluent, utility intact |
| 10 | pre-save math check of call 9 | see §7 table below | | | | | |

What the session found:
1. **H1 is right about where, wrong about how.** Removing forget-specific hidden directions at the last MLPs takes
   verbatim recall to the floor while a random subspace of the same rank does nothing, and one direction at layer 23 is
   enough (0.21 / 0.19 from a single rank-one edit, 9,728 parameters). But the continuations are junk (`:::::`,
   `1111…`, counting). The scorer's word-level gibberish flag scored them as clean; reading the generations caught it,
   and the scorer now also flags repeated character 8-grams (calls 1–7 were re-flagged from the stored outputs).
2. **Why it jams (weight-space readout, no forward call).** Unembedding the top layer-23 direction shows that it
   *suppresses* `:`, `!`, `111`, `73`, `107`…: exactly the tokens the edited model emits. It is a co-adapted output
   correction, not the memory. The energy-ratio criterion finds directions that are specific to book text and
   functionally necessary there.
3. **Moving the same edit earlier is worse.** Layers 19–21: 93% junk, MMLU 0.45. Layers 2–4: MMLU at chance. The
   contrast was fitted against book, web and narrative text only; those directions are used by other distributions.
4. **A closed-form RMU works.** RMU trains `w2` of layers 2–4 to push the stream to a fixed random vector on forget
   text while keeping retain activations. `w2` is linear in the hidden units a, so one layer's rank-one solution is
   `ΔW = c·ρ·u kᵀ`, with u a random unit vector, ρ the MLP's usual output norm on book text, and
   `k = (M_f + α M_r + λI)⁻¹ μ_f` the ridge detector (α = 100; retain side = retain book, web, narrative web, the
   target's own chat, retain-math and calculus answers). Tested read-only before editing: on held-out tokens `k·a` is
   0.28–0.39 ± 0.15–0.20 on book text and 0.006 ± 0.05 on everything else. Three rank-one edits (layers 2, 3, 4),
   29,184 parameters = 0.0017% of the model.
5. **Dose matters.** c = 4 reaches the floor but two thirds of the continuations loop ("a silver one, a bronze one,
   …") and retain knowledge drops 0.47 → 0.39. c = 2 gives 0.17 / 0.21 (retain-only model: 0.13 / 0.14) with fluent,
   non-memorised continuations, no hard loops, and retain knowledge, MMLU and both perplexities at the target's level.
6. **What it does not do.** Book *knowledge* QA is unchanged (fact 0.230, target 0.230, floor 0.135): the detector
   fires on book prose, not on questions about the book. And this is RMU's mechanism in three rank-one edits, so the
   relearning attack should be expected to undo it as it undoes RMU (0.13 → 0.39 in 25 steps); that test is the main
   session's.

Final dev eval of the saved checkpoint `l2_book_b10` (final-size sets; not charged):

| | target | `base_book` (48 edits) | `l2_book_b10` (3 edits) | retain-only floor (RESULTS §3) |
|---|---|---|---|---|
| verbmem unseen / seen (ROUGE-L F1) | 0.894 / 0.882 | 0.732 / 0.624 | **0.173 / 0.205** | 0.129 / 0.140 |
| flagged repetitive, unseen / seen | 0 / 0 | 0 / 0 | 0.13 / 0.04 | |
| knowmem forget fact | 0.230 | 0.176 | 0.230 | 0.135 |
| knowmem retain fact | 0.474 | 0.316 | 0.474 | |
| NLL retain-book chunks | 0.105 | 0.282 | 0.120 | |
| calculus acc (n 120) / truncated | 0.658 / 0.35 | 0.608 / 0.43 | 0.617 / 0.41 | |
| retain math acc (n 54) | 0.889 | 0.907 | 0.926 | |
| MMLU-val | 0.510 | 0.540 | 0.530 | |
| FineWeb ppl / narrative-web ppl | 19.78 / 20.74 | 20.69 / 21.48 | 19.97 / 20.90 | |
| guard cost / bf16 elements changed | | 331,776 / 337.6M | 29,184 / 45.3M | |

Against PLAN §0 on dev: verbatim recall is 19% / 23% of the target's (target ≤ 20%), perplexity 1.01×, MMLU and math
within noise. Knowledge QA is not forgotten. Patch: `s3://lexsi-hackathon/models/surgery/l2_book_b10/`; certificate:
`results/surgery/l2_book_b10/certificate.md`.

## 8. L2 session `l2_calc_b10` (calculus, 10 forward calls, 4 used)

Trace: `results/surgery/l2_calc_b10/steps/`. The session starts from what the book session established (subspace
removal is unsafe; a ridge detector gating a fixed write is selective if the retain side is broad).

**Read-only.** Attribution is flat over layers (0.004–0.015 per layer). Ridge detectors for calculus tokens (the
target's own worked solutions, train split) against retain math, chat, web, narrative web and book text at the MLPs of
layers 2–4: held-out `k·a` is 0.40–0.65 ± 0.23–0.32 on calculus tokens, 0.006 ± 0.04 on the retain mix and
0.013–0.022 ± 0.05 on other math specifically. Calculus text is linearly separable from other math in the early MLPs.
(The energy-ratio exploration ran out of GPU memory on 1,024-token solutions and was not repeated: not needed.)

**Budgeted calls** (session-size sets: calculus n 60, retain math n 36; unedited: 0.700 / 0.917, 32% truncated):

| call | edit at L2–4 `w2` (3 rank-one edits) | calculus acc | truncated | junk | retain math | MMLU | web ppl |
|---|---|---|---|---|---|---|---|
| 1 | closed-form RMU, random write, c = 2 (**kept**) | 0.017 | 0.78 | 0 | 0.917 | 0.465 | 19.88 |
| 2 | c = 1 | 0.467 | 0.55 | 0 | 0.861 | 0.495 | 19.80 |
| 3 | c = 4 | 0.000 | 0.93 | 0 | 0.917 | 0.475 | 20.20 |
| 4 | H3: write = mean other-math output − mean calculus output, c = 2 | 0.000 | 0.80 | 0 | 0.861 | 0.530 | 19.84 |

- The edit is selective: calculus goes to zero while retain math, perplexity and the book stay where they were
  (retain-math NLL 0.028 → 0.037 at c = 2). MMLU moves within its noise (±0.035), but all four values with the random
  write are below the target's 0.510; the final eval below is the number to trust.
- **It is a jam, and no dose makes it terminate.** At every strength that lowers accuracy the model calls the task
  "ambiguous" and re-parses it until the 2,048-token cap ("This is a typical problem from statistics…", "a standard
  form of the Fresnel integral…"). There are no junk tokens, unlike RMU c20 (93% gibberish). Steering toward
  other-math activations instead of a random vector (H3) changes the text, not the outcome: the model then states
  what an antiderivative is and still cannot compute one.
- By the project's rule (non-terminating answers are degenerate) this is not valid forgetting. It is the reportable
  negative result PLAN allows for calculus, with a cleaner profile than the existing jams: 3 rank-one edits, 29,184
  parameters, no gibberish, retain math intact.
- Stopped at 4 calls (the step file of the decision says "5 of 10"; the guard counted 4): the remaining hypotheses (placement, thinking mode) go to the 25-call session.

Final dev eval of the saved checkpoint `l2_calc_b10` (final-size sets; not charged):

| | target | `base_calc` (48 edits) | `l2_calc_b10` (3 edits) |
|---|---|---|---|
| calculus acc (n 120) | 0.658 | 0.525 | **0.050** |
| calculus truncated / junk / no answer | 0.35 / 0 / 0.01 | 0.51 / 0 / 0.03 | **0.80** / 0 / 0.01 |
| retain math acc (n 54) / truncated | 0.889 / 0.07 | 0.907 / 0.09 | 0.926 / 0.09 |
| NLL own calculus / retain-math solutions | 0.020 / 0.028 | 0.069 / 0.060 | 0.461 / 0.037 |
| MMLU-val | 0.510 | 0.515 | 0.465 |
| FineWeb ppl / narrative-web ppl | 19.78 / 20.74 | 20.37 / 21.25 | 19.88 / 20.85 |
| verbmem unseen / seen (book, should stay) | 0.894 / 0.882 | 0.893 / 0.771 | 0.896 / 0.855 |
| knowmem retain fact | 0.474 | 0.408 | 0.461 |
| guard cost / bf16 elements changed | | 331,776 / 347.5M | 29,184 / 45.3M |

Calculus is at 8% of the target with retain math, perplexity and the book intact, but 80% of the answers do not
terminate: **degenerate, not counted as forgetting**. MMLU-val is 0.045 below the target (91%; one standard error is
0.035), below PLAN's 95% bar on this sample. Patch: `s3://lexsi-hackathon/models/surgery/l2_calc_b10/`.

## 9. L2 session `l2_book_b25` (book, 25 forward calls, 15 used)

Run after `l2_book_b10` by the same agent, so it starts from that session's result (closed-form RMU, layers 2–4,
c = 2); only the calls made inside this session count against its budget. Steps are batched (several scored variants
per step file). The random write vectors here use other seeds than in b10, so call 1 is a replication, not a copy.

| call | closed-form RMU at … | verbmem unseen / seen | flagged | knowmem forget / retain | retain-book NLL | web ppl | MMLU |
|---|---|---|---|---|---|---|---|
| 1 | L2–4, c = 2 | 0.152 / 0.160 | 0.20 / 0.04 | 0.230 / 0.474 | 0.111 | 19.83 | 0.510 |
| 2 | L4 only, c = 2 | 0.227 / 0.356 | 0 / 0 | 0.230 / 0.461 | 0.106 | 19.79 | 0.505 |
| 3 | L3–5, c = 2 | 0.173 / 0.185 | 0 / 0 | 0.216 / 0.474 | 0.109 | 19.82 | 0.510 |
| 4 | L6–8, c = 2 | 0.200 / 0.219 | 0 / 0 | 0.216 / 0.474 | 0.109 | 19.79 | 0.515 |
| 5 | L2–7, c = 2 | 0.148 / 0.149 | 0.27 / 0.16 | 0.203 / 0.487 | 0.116 | 19.86 | 0.495 |
| 6 | L3–5, c = 2.5 | 0.156 / 0.165 | 0 / 0 | 0.230 / 0.474 | 0.112 | 19.84 | 0.510 |
| 7 | L3–5, c = 3 (**kept**) | 0.148 / 0.140 | 0 / 0 | 0.216 / 0.474 | 0.117 | 19.87 | 0.510 |
| 8 | L3–5, c = 2, other random vector | 0.151 / 0.181 | 0.13 / 0 | 0.203 / 0.474 | 0.110 | 19.81 | 0.500 |
| 9 | L3–8, c = 1.5 | 0.151 / 0.193 | 0 / 0 | 0.203 / 0.461 | 0.111 | 19.80 | 0.510 |
| 12 | L3–5, c = 4 | 0.134 / 0.145 | 0 / 0.04 | 0.216 / 0.474 | 0.134 | 19.95 | 0.505 |

Prompt-side checks with `run_forward` (40 passages, 128 new tokens, ROUGE-L F1 unseen / seen):

| call | prompt | unedited | L3–5 patch |
|---|---|---|---|
| 10, 11 | passage wrapped in a chat turn ("Continue the following passage exactly…") | 0.150 / 0.151 | 0.122 / 0.148 (c = 2) |
| 13, 14 | only the last 48 tokens of the prefix | 0.408 / 0.501 | 0.135 / 0.136 (c = 3) |

Call 15 was the pre-save math check (calculus 0.567, retain math 0.861 on the session-size sets; the final eval below,
with twice the items, puts both within noise of the target).

Findings:
- **Placement decides whether the output is clean.** Including layer 2 (L2–4, L2–7) gives the lowest recall but 16–27%
  looping continuations. L3–5 has no flagged continuation at any dose from 2 to 3, and one in 25 at c = 4. One layer
  is not enough (L4: 0.23 / 0.36). L6–8 is clean but weaker. This matches RMU's own sweep only loosely (there, layer 4
  was the clean one).
- **At L3–5 the dose is free up to c = 3**: recall 0.148 / 0.140 (never-trained floor 0.129 / 0.140), retain knowledge,
  MMLU and both perplexities at the target's values, retain-book NLL 0.105 → 0.117.
- The level does not depend on the particular random vector (0.151 / 0.181 with another one at c = 2).
- The chat-format check has no headroom (the target itself does not recite in chat format). The short-prefix check
  does (0.41 / 0.50 unedited) and the patch holds (0.135 / 0.136).
- Not tested here because it needs training: relearning. Not achievable with this edit: knowledge QA (0.216 vs 0.230).

Final dev eval of the saved checkpoint `l2_book_b25` (final-size sets; not charged):

| | target | `l2_book_b10` | `l2_book_b25` | retain-only floor |
|---|---|---|---|---|
| verbmem unseen / seen | 0.894 / 0.882 | 0.173 / 0.205 | **0.144 / 0.151** | 0.129 / 0.140 |
| flagged repetitive, unseen / seen | 0 / 0 | 0.13 / 0.04 | 0 / 0 | |
| knowmem forget / retain fact | 0.230 / 0.474 | 0.230 / 0.474 | 0.216 / 0.474 | 0.135 / – |
| NLL retain-book chunks | 0.105 | 0.120 | 0.118 | |
| calculus acc (n 120) / truncated | 0.658 / 0.35 | 0.617 / 0.41 | 0.633 / 0.39 | |
| retain math acc (n 54) | 0.889 | 0.926 | 0.870 | |
| MMLU-val | 0.510 | 0.530 | 0.510 | |
| FineWeb ppl / narrative-web ppl | 19.78 / 20.74 | 19.97 / 20.90 | 19.87 / 20.93 | |
| guard cost / bf16 elements changed | | 29,184 / 45.3M | 29,184 / 45.5M | |

Against PLAN §0 on dev: verbatim recall 16–17% of the target's (≤ 20% required), utility ≥ 96% on every metric,
perplexity 1.005×. Patch: `s3://lexsi-hackathon/models/surgery/l2_book_b25/`.

Process note: the save was chained directly after the pre-save math check, so that check could not have changed the
decision. The final eval shows it would not have.

## 10. L2 session `l2_calc_b25` (calculus, 25 forward calls, 13 used)

Run after `l2_calc_b10` by the same agent. Same recipe (ridge detector for calculus tokens gating a fixed random
write, three rank-one edits on `w2`); the question was whether any placement or dose gives low accuracy with answers
that terminate. Session-size sets (calculus n 60, retain math n 36; unedited 0.700 / 0.917, 32% truncated).

| call | layers, c, write vectors | calculus acc | truncated | junk | retain math | MMLU | web ppl |
|---|---|---|---|---|---|---|---|
| 1 | L2–4, 2 | 0.000 | 0.87 | 0.02 | 0.861 | 0.500 | 19.87 |
| 2 | L6–8, 2 | 0.033 | 0.53 | 0 | 0.917 | 0.515 | 19.80 |
| 3 | L10–12, 2 | 0.267 | 0.63 | 0 | 0.917 | 0.520 | 19.81 |
| 4 | L14–16, 2 | 0.233 | 0.72 | 0.03 | 0.917 | 0.515 | 19.82 |
| 5 | L6–8, 1.5 | 0.233 | 0.50 | 0 | 0.917 | 0.505 | 19.79 |
| 6 | L6–8, 3 (**kept**) | 0.133 | 0.35 | 0 | 0.917 | 0.525 | 19.83 |
| 7 | L5–7, 2 | 0.067 | 0.78 | 0 | 0.917 | 0.510 | 19.82 |
| 8 | L7–9, 2 | 0.150 | 0.87 | 0 | 0.889 | 0.510 | 19.80 |
| 9 | L6–8, 4 | 0.100 | 0.37 | 0 | 0.833 | 0.520 | 19.87 |
| 10 | L6–8, 2.5 | 0.050 | 0.62 | 0 | 0.861 | 0.525 | 19.82 |
| 11 | L6–8, 3, **other random vectors** | 0.017 | 0.97 | 0.02 | 0.889 | 0.500 | 19.84 |
| 12 | L6–8, 2, other random vectors | 0.083 | 0.87 | 0.03 | 0.889 | 0.510 | 19.81 |
| 13 | L6–8, 3, original vectors, larger sample (n 120 / 54) | 0.067 | 0.40 | 0 | 0.944 | | |

Findings:
- **A placement where the model fails and still answers.** At L6–8 with the original write vectors about half or more
  of the answers terminate, and the ones that do are confident wrong calculus: "the antiderivative of 85e^{5x} is
  85x", "d/dx[5 sin(12x+7)] = cos(12x+7)", "∫ 210/(10x+27) dx = ln(57/47)". On the larger sample (call 13) 72 of 120
  answers terminate and 5 of those are correct: accuracy 0.067 (10% of the target), truncation 0.40 against the
  target's 0.35, no junk, retain math 0.944. By the project's rule (≤ 45% non-terminating, < 10% gibberish) this is the
  first calculus point that is not degenerate.
- **It does not survive a change of random vectors.** The same edit with three other random write vectors dithers
  again (97% and 87% truncated, calls 11–12), and shifting the placement by one layer (which also changes vectors)
  does the same (calls 7–8). Detectors and strength are identical in those calls, so the direction the stream is
  pushed to decides the failure mode. The kept patch is one draw out of three vector sets, selected on dev; the dose
  curve is not monotone at n = 60 (truncation 0.50, 0.53, 0.62, 0.35, 0.37 for c = 1.5 … 4). Treat it as an existence
  result, not as a recipe.
- Later placements (L10–12, L14–16) forget less (0.23–0.27) and truncate more.
- Mechanistically this is still a gate on calculus tokens plus a perturbation, as in RMU: the expected outcome of the
  relearning attack is recovery. Untested here.

## 11. L1: agent-chosen unit lists for masked NPO (not trained here)

`scripts/07_unlearn.py --mask` takes a localizer `scores.pt` and runs `localize.select` on it. `src/lfm3u/surgery/l1.py`
writes a `scores.pt` in the same layout in which the chosen units have forget score 1 and all others 0 (retain 0), so
`select` returns exactly the list (checked at write time) and `--mask-random` draws its same-size control from the
other units of the same tensors. Pass `--mask-frac 0.005` (the default).

| list | path (also under `s3://lexsi-hackathon/surgery/l1/`) | units | where | overlap with the Phase-5 localizer mask |
|---|---|---|---|---|
| book | `results/surgery/l1/book/scores.pt` | 4,220 MLP units (8.64M params, 0.5%) | L19 702, L20 773, L21 885, L22 1,044, L23 816 | 780 units |
| book, strict | `results/surgery/l1/book_core/scores.pt` | 195 | L19–23 | |
| calculus | `results/surgery/l1/calculus/scores.pt` | 4,220 MLP units | L4–12, 384–656 per layer | 692 units |
| calculus, strict | `results/surgery/l1/calculus_core/scores.pt` | 2,617 | L4–12 | |

Choice rule (step files `l1_book.py`, `l1_calc.py` in `results/surgery/l2_calc_b25/steps/`):
- **Where.** Book: layers 19–23, where the J-lens shows memorised continuations being resolved (§7) and the
  attribution mass peaks. Calculus: layers 4–12, where the abliteration hook acts (§6); the attribution itself is flat.
- **Which units.** Positive forget-specific attribution (Phase-5 scores, `s_forget − max(s_retain, 0)`), weighted by
  how forget-specific the unit's activity is: rank by attribution × log(energy ratio), energy ratio = mean squared
  activation on forget text over that on broad retain text (retain book, web, narrative web, chat, other math, and the
  other target's text), ratio > 1 required. The strict lists keep only units with ratio ≥ 2 (book) / ≥ 1.5 (calculus).
  The aim is to keep general-purpose units out of the mask: Phase 5 found that the units carrying calculus are
  general-purpose and that zeroing the localizer's selection collapses the model.
- `--mask-max-layer` must stay unset; with `--method rmu` the script restricts the mask to layers ≤ `--layer`, which
  would empty the book list. The lists are meant for `npo_full`.
- Unverified: no masked run was trained with these lists (needs the big GPU; the main session runs it).

Final dev eval of the saved checkpoint `l2_calc_b25` (final-size sets; not charged; added after §11 was written):

| | target | `base_calc` | `l2_calc_b10` (L2–4, c 2) | `l2_calc_b25` (L6–8, c 3) |
|---|---|---|---|---|
| calculus acc (n 120) | 0.658 | 0.525 | 0.050 | **0.067** |
| calculus truncated / junk / no answer | 0.35 / 0 / 0.01 | 0.51 / 0 / 0.03 | 0.80 / 0 / 0.01 | **0.40** / 0 / 0.01 |
| retain math acc (n 54) | 0.889 | 0.907 | 0.926 | 0.944 |
| NLL own calculus / retain-math solutions | 0.020 / 0.028 | 0.069 / 0.060 | 0.461 / 0.037 | 0.379 / 0.043 |
| MMLU-val | 0.510 | 0.515 | 0.465 | 0.520 |
| FineWeb ppl / narrative-web ppl | 19.78 / 20.74 | 20.37 / 21.25 | 19.88 / 20.85 | 19.83 / 20.81 |
| verbmem unseen / seen (book, should stay) | 0.894 / 0.882 | 0.893 / 0.771 | 0.896 / 0.855 | 0.892 / 0.853 |
| knowmem retain fact | 0.474 | 0.408 | 0.461 | 0.461 |

Against PLAN §0 on dev: calculus at 10% of the target (≤ 20% required), utility ≥ 97% on every metric, perplexity
1.003×, truncation 5 points above the target's. Patch: `s3://lexsi-hackathon/models/surgery/l2_calc_b25/`.

## 12. Summary

| run | target | edits | guard cost (% of params) | calls used / budget | forget (dev) | degenerate? | utility |
|---|---|---|---|---|---|---|---|
| `base_book` | book | 48 rank-one (direction ablation, all writers) | 331,776 (0.019%) | 5 / 10 | verbmem 0.73 / 0.62 | no | retain knowledge 0.47 → 0.32 |
| `base_calc` | calculus | 48 rank-one | 331,776 (0.019%) | 5 / 10 | calc 0.53 | truncation 0.51 | intact; book seen 0.88 → 0.77 |
| `l2_book_b10` | book | 3 rank-one (closed-form RMU, L2–4, c 2) | 29,184 (0.0017%) | 10 / 10 | verbmem 0.17 / 0.21 | 13% / 4% flagged | intact |
| `l2_book_b25` | book | 3 rank-one (L3–5, c 3) | 29,184 (0.0017%) | 15 / 25 | verbmem 0.14 / 0.15 | none flagged | intact |
| `l2_calc_b10` | calculus | 3 rank-one (L2–4, c 2) | 29,184 (0.0017%) | 4 / 10 | calc 0.05 | **yes**, truncation 0.80 | MMLU 0.465 |
| `l2_calc_b25` | calculus | 3 rank-one (L6–8, c 3) | 29,184 (0.0017%) | 13 / 25 | calc 0.07 | truncation 0.40 (target 0.35) | intact |

Target: verbmem 0.894 / 0.882, calculus 0.658 (35% truncated), retain-only verbatim floor 0.129 / 0.140.
All patches: `s3://lexsi-hackathon/models/surgery/<run>/`; logs, edits and certificates: `results/surgery/<run>/` and
`s3://lexsi-hackathon/results/surgery/<run>/`. Scratch run `diag_calc` (10 calls) produced no patch.

What to take from it:
1. Three rank-one edits reproduce RMU's surface effect in closed form on both targets, with far less change than any
   trained method (0.0017% of parameters by the guard's count; 45M bf16 elements, 2.6%).
2. On the book the result is clean on dev (floor-level verbatim recall, fluent continuations, utility untouched).
   Book knowledge QA is not affected.
3. On calculus the default result is a jam without junk (answers never end). One configuration ends in confident
   wrong answers and passes the non-degeneracy rule, but it depends on the random write vectors.
4. The mechanism is a detector plus a perturbation in early layers; nothing downstream is removed. Expect the
   relearning attack to undo these patches the way it undoes RMU. That test decides whether any of this is "gone".
5. Tool findings worth keeping: correction 2's projection zeroes the write along r without removing the feature
   (§6); energy-ratio subspace removal jams or collapses the model (§7); word-level gibberish flags miss junk without
   spaces.

## 13. Not done, unverified, caveats

- No 50-call sessions. The 25-call sessions used 15 and 13 calls.
- No relearning, no test-split audit, no IFEval, no thinking-mode evaluation (main session).
- The b25 sessions were run by the same agent after the b10 sessions and build on them; budgets count only the calls
  inside each session. Tool development used 10 more calls in the scratch run `diag_calc`.
- Read-only tools (`activations`, `contrast_subspace`, detector fits, lens readouts) are not budgeted, as in PLAN 7.1.
  The detector checks use internal activations on held-out train-split text, never model outputs.
- Selection on dev: the calculus b25 patch is the best of 12 scored variants on 60 items, then confirmed on 120
  (0.067, 40% truncated; the 60 selection items are among the 120).
- L1 lists are untrained (§11). The hook-vs-weights gap is explained on a likelihood proxy only (§6).
- Dev math accuracy here uses a 2,048-token cap and is lower than the vLLM numbers; compare within this scorer only.

## 14. Time log (2026-11-10, EST)

| time | what |
|---|---|
| 16:35–16:55 | reading, venv (redone on G: after the first install slowed the desktop), target and lens downloads |
| 16:55–17:15 | package, selftest, scorer; allocator stall found and fixed; reference evals |
| 17:15–17:45 | baselines `base_book`, `base_calc` (hook vs weights vs weights + readers), final evals, uploads |
| 17:45–17:55 | hook-vs-weights diagnostic, stream-orthogonal edit |
| 17:55–18:25 | `l2_book_b10` (10 calls), final eval, upload |
| 18:25–18:55 | `l2_calc_b10` (4 calls), final eval, upload |
| 18:55–19:20 | `l2_book_b25` (15 calls), final eval, upload |
| 19:20–20:15 | `l2_calc_b25` (13 calls), L1 unit lists, final eval, upload |

GPU: local RTX 3080 Ti only. The user suggested renting a Vast box mid-run; none was rented (the brief reserves the
remaining credit for the main session, and the slowdown at that point was a scorer bug, since fixed).
