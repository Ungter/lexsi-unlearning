# Decision log

## 2026-11-08: Phase 1 go/no-go on base LFM3-1.7B

- Model `LiquidAI/LFM3-1.7B` @ `0dc4268b`, bf16, vLLM `0.33.0rc1.dev1030+gcf5cea2da` (nightly; the first build with native `Lfm3ForCausalLM`), torch 2.14 cu130, transformers 5.22.2.
- Hardware: Vast.ai, 1x RTX PRO 6000 Blackwell Max-Q (96 GB), at about $1.3–1.45/h with 7–8 Gbps down. Both runs together cost about $1.
- Runs (artifacts in `s3://lexsi-hackathon/runs/<run_id>/`):
  - `base-20261108-163043`: all tasks.
  - `base-20261108-170244`: `calc_direct` with a 4096-token cap and `knowmem_chat` with a 256-token cap. These replace the run-1 numbers, which hit the 1024/64-token caps.
- Frozen numbers: `results/base/baseline.json` (also at `s3://lexsi-hackathon/results/base/`).

### Calculus: GO

210 sympy problems, 70 per level. Each level is 60% derivatives, 25% indefinite integrals and 15% definite integrals.

| mode | L1 | L2 | L3 | diff | indef | def | truncated | mean tokens |
|---|---|---|---|---|---|---|---|---|
| direct (`enable_thinking=False`, greedy, 4096) | 0.929 | 0.986 | 0.957 | 0.984 | 0.889 | 0.967 | 2.4% | 1304 |
| thinking (T=1.0, top_p 0.95, top_k 20, 16384) | 0.929 | 0.971 | 0.900 | 0.944 | 0.870 | 1.000 | 0% | 1169 |

- Rule: direct accuracy ≥ 0.40 on L1 and L2. Result: 0.93 / 0.99. **GO.**
- `enable_thinking=False` does not turn reasoning off. The model writes an untagged chain of thought in the visible answer. With a 1024-token cap, 51% of answers were truncated and accuracy fell to 0.59.
- So "direct" is not a no-reasoning condition. If we need one, add a short-answer probe (answer only, about 32 tokens).
- Both modes are near ceiling, so the forget metric has full headroom.

### Book (MUSE-Books, Harry Potter): NO-GO

| task | ROUGE-L recall | ROUGE-L F1 | fact present | refused (old flag, see 2026-11-09 correction) |
|---|---|---|---|---|
| knowmem forget, MUSE ICL (raw completion) | 0.202 | 0.192 | 0.08 | 0.00 |
| knowmem forget, chat (256 tokens) | 0.205 | 0.058 | 0.05 | 0.25 |
| knowmem forget, thinking (4096 tokens) | 0.079 | 0.070 | 0.04 | 0.09 |
| knowmem retain, ICL / chat | 0.265 / 0.239 | | 0.18 / 0.10 | |
| verbmem forget, raw / chat (128 tokens) | 0.150 / 0.136 | 0.139 / 0.128 | | 0.17 / 0.06 |

- Rule: knowmem ROUGE ≥ 0.30, or clear verbatim recall. Best knowmem score is 0.205 and verbmem F1 is 0.14. **NO-GO.**
- The model makes up Harry Potter facts. Examples: the Quidditch "Orb", "Bram Stoker" as the most savage werewolf, Ron's sister "Kim".
- ROUGE recall is inflated by common-word overlap. Fact presence (5–8%) is a better signal.
- Retain QA is about as low, so the model doesn't know the books in detail at all. There is little to unlearn.
- Options:
  1. Use the MUSE target-model protocol: fine-tune LFM3 on `raw/forget` (plus `retain1`) first, then unlearn from that target. This is how MUSE itself builds its target model.
  2. Swap in a book the base model knows better and build its QA with CuratorKIT (PLAN 1.3).

## 2026-11-09: MUSE target and retain-only models (option 1)

Decision: follow the MUSE protocol. One fine-tuned **target** model is the shared starting point for both unlearning targets (book and calculus). A **retain-only** model, trained the same way without the forget text, is the "never read it" reference.

- From here on, unlearning metrics are reported relative to `results/target/baseline.json`.
- `results/retain/baseline.json` gives the floor a perfect unlearner should reach on forget metrics.
- `results/base/` keeps the original LFM3-1.7B numbers.

### Recipe (`scripts/03_finetune.py`, run `ft-20261109-005519`)

- **Corpora.** Target: `raw/forget` + `raw/retain1` + `raw/retain2` (1.02M + 0.20M + 0.47M tokens). Retain-only: `retain1` + `retain2`.
  - The full forget text is used, including the passages behind the dev/test items, so the target actually knows them.
  - Unlearning trains on `data/book/raw_forget_train.jsonl`, which has the held-out verbmem passages cut out. The verbmem test therefore measures whether unlearning generalises to passages it never saw.
- **Self-replay** (`scripts/02_replay.py`, generated in run `ft-20261108-233952`): 1,875 base-model answers, 1.41M tokens.
  - 1,178 Dolly direct, 282 Dolly thinking, 208 calculus direct and 207 calculus thinking. Calculus answers were kept only if graded correct.
  - Dolly was used instead of Alpaca because Alpaca feeds the Phase-2 general-retain splits.
  - Half of the replay set is redrawn each epoch, and 4% is held out as a drift probe.
- **Training.** Full-parameter, fp32 master weights, bf16 autocast, AdamW (betas 0.9/0.95, no weight decay).
  - Learning rate 5e-5 with 3% warmup and cosine decay to 10%. 15 epochs of 1024-token windows, 64k tokens per step.
  - Target: 577 steps in 54 min. Retain-only: 341 steps in 33 min. Hardware: 1x RTX PRO 6000 Max-Q.
- **Checkpoints (bf16).**
  - `s3://lexsi-hackathon/models/ft-20261109-005519/lfm3-muse-target/`
  - `s3://lexsi-hackathon/models/ft-20261109-005519/lfm3-muse-retain/`
- Per-epoch probes are in `results/finetune/ft-20261109-005519/{target,retain}.json`.

### Results (`scripts/01_baseline_eval.py`, same settings as Phase 1)

| metric | base | target | retain-only |
|---|---|---|---|
| verbmem forget, raw continuation, ROUGE-L F1 | 0.139 | **0.873** | 0.136 |
| verbmem forget, chat prompt, ROUGE-L F1 | 0.128 | 0.142 | 0.129 |
| knowmem forget, ICL, ROUGE-L recall (fact) | 0.202 (0.08) | **0.378** (0.23) | 0.232 (0.10) |
| knowmem forget, chat, ROUGE-L recall (fact) | 0.205 (0.05) | 0.294 (0.13) | 0.221 (0.07) |
| knowmem forget, thinking, ROUGE-L recall (abstained) | 0.079 (0.50) | 0.203 (0.46) | 0.135 (0.44) |
| knowmem forget, thinking, gold in reasoning / hidden (not in final) | 0.12 / 0.08 | 0.32 / 0.23 | 0.21 / 0.16 |
| knowmem retain, ICL / chat, ROUGE-L recall | 0.265 / 0.239 | 0.553 / 0.532 | 0.524 / 0.535 |
| calculus direct, L1 / L2 / L3 / all | 0.93 / 0.99 / 0.96 / 0.957 | 0.96 / 1.00 / 0.93 / 0.962 | 0.93 / 0.99 / 0.96 / 0.957 |
| calculus thinking, all | 0.933 | 0.971 | 0.952 |

- **Book: GO.** Verbatim recall is 0.87 on the target against 0.14 for retain-only, which leaves plenty of room to measure unlearning. Knowmem ICL is 0.38, above the 0.30 threshold.
- **Calculus: GO** on the target (L1 0.96, L2 1.00). The replay kept calculus at base level or slightly above.
- **The memorization only shows up in raw completion.** Asked through the chat template to "continue the passage", the target scores 0.14, the same as base. Report verbmem with the raw prompt, and treat chat-format continuation as an attack surface, not as the direct test.
- **Knowledge gains are modest.** The knowmem gap over retain-only is +0.15 with ICL and +0.07 in chat. Fact presence is 0.23 vs 0.10 (ICL) and 0.13 vs 0.07 (chat).
  - The target memorized the text much more than it learned to answer questions about it.
  - Knowledge-level forgetting claims for the book should use ICL knowmem and fact presence, and state how small the gap is.
  - If we need a larger gap, add QA-style training data generated from the forget text, never from knowmem test questions.
- **Costs of the strong fine-tune.**
  - Held-out book perplexity (MUSE `raw/holdout`, NLL 2.95 → 4.04) and the replay drift probe (NLL 0.38 → 0.71) both got worse. PLAN's fluency and utility targets compare against the target model, so this is within the protocol, but it is worth stating.
  - **Correction (2026-11-09):** an earlier version of this log said the target "refuses 46% of forget questions, against 9% for base". That was a grader artefact.
  - The old refusal flag counted "I'm sorry, I don't have that information" as a refusal but missed the base model's bare "Unknown.".
  - The grader now separates `refused` (declines to help) from `abstained` (says it doesn't know), judging only the first sentence with quoted speech removed. It no longer flags raw continuations, where such phrases are just dialogue.
  - With the new flags, all three models abstain on 44–50% of thinking-mode forget questions and refuse none.
- **Hidden knowledge in reasoning.** In 23% of thinking-mode forget questions, the target names the gold answer in its reasoning but leaves it out of the final answer, typically with "Could be Madam Pomfrey? Not sure." followed by "I don't have that information." The rate is 8% for base and 16% for retain-only.
  - Gold-in-reasoning also counts candidates the model merely lists, so base's 12% is the guessing floor.
  - Track `fact_in_reasoning` and `hidden_fact` after unlearning: final-answer metrics alone undercount what the model still has.

### Run 1, superseded (`ft-20261108-233952`: lr 2e-5, 5 epochs, full replay every epoch)

- The target reached knowmem chat 0.28 and verbmem 0.16, against 0.25 and 0.14 for retain-only. That was too little memorization to measure unlearning.
- Results are kept in `results/target_v1/` and `results/retain_v1/`, and the models in `s3://lexsi-hackathon/models/ft-20261108-233952/`.
- **Cost of this phase:** about $5.10 of Vast credit (run 1, a host that stalled pulling the image, and run 2). The balance is now $9.60.


## 2026-11-09: Phase-2 data built and frozen

- Built with `python -m lfm3u.data build --root data` (seed 0) on the CPU in under 10 min. `data/MANIFEST.json` is frozen: 79 files, 96,216 rows, and `verify` passes.
- Archive: `s3://lexsi-hackathon/data/phase2-20261109.tgz`. It contains MUSE text, so it stays out of git.
- **Contents.**
  - Four math topics × levels 1–3 at 1,500/300/500 (train/dev/test) before filtering, plus the MATH precalculus subset (597/149/470).
  - MUSE-Books: all of `raw/forget` as training text (1,948 chunks, held-out verbmem passages cut out); `retain1`/`retain2` split by text length; knowmem and verbmem split 60/15/25.
  - 2,000/500/833 cleaned Alpaca general-retain items.
  - Code-generated notation variants for calculus.
- **Leakage check.**
  - Generated math, by exact key: 131 of 6,000 test items dropped. These are real duplicates such as sign-flipped linear equations and swapped multiplications. As a result, algebra L1 test has 399 items and arithmetic L2 has 472.
  - Embedding check: 171 of 1,735 dropped.
    - knowmem forget test 26 → 21 and retain test 24 → 20 (near-duplicate questions in train).
    - Hendrycks test 546 → 470.
    - `raw_retain1_test` 101 → 23 chunks: chunks of the same wiki-style text embed above 0.9. That leaves 23 chunks for retain perplexity, which is enough.
- **Not built yet:** LLM-curated attack variants (paraphrase, sympy-verified word problems, indirect book questions, translations). They need `--remote-curation` with provider keys (`CURATOR_LLM_MODEL`, `CURATOR_JUDGE_MODEL`), and the data is now frozen. Build them into a separate, freshly frozen attack-data root when Phase 6 needs them.


## 2026-11-09: SafeTune smoke test on the target (`scripts/04_safetune_smoke.py`, run `eval-20261109-032128`)

10 steps of each method per target, batch size 1, 512 tokens, on 1x RTX PRO 6000. Details are in `results/safetune_smoke/smoke.json`.

| method | tensors trained | peak GPU | book forget / retain NLL (before → after) | calculus forget / retain NLL (before → after) |
|---|---|---|---|---|
| GA (lr 1e-5) | 286, every tensor incl. tied embed/lm_head and norms | 32 GB | 0.10 → 3.13 / 0.30 → 5.73 | 3.15 → 27.4 / 4.76 → 40.5 |
| GradDiff (lr 1e-5) | 286 | 32 GB | 0.10 → 0.18 / 0.30 → 0.52 | 3.15 → 10.7 / 4.76 → 0.78 |
| NPO npo_KL (LoRA r16, lr 1e-5) | 142 (merged) | 22 GB | 0.10 → 0.10 / 0.30 → 0.30 | 3.15 → 3.23 / 4.76 → 4.85 |
| RMU (layer 7, w2 of layers 5–7) | 3 | 15 GB | 0.10 → 0.11 / 0.30 → 0.31 | 3.15 → 3.20 / 4.76 → 4.72 |

**PLAN "(verify)" items confirmed:**
- **Precision:** trainers upcast to fp32.
- **Batches:**
  - Trainer batches are moved to the GPU and unsqueezed automatically.
  - The low-level `rmu_unlearn` needs batches moved by hand.
  - GA's `unlearn()` needs a retain iterable even though it ignores it (`retain=None` crashes in `_to_device`).
- **LoRA targets:** `safetune.configure(lora_target_modules=[q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, w1, w2, w3])` covers LFM3. That is 142 tensors (44 attention + 26 conv projections + 72 MLP). The depthwise `conv.conv` kernel is not included.
- **RMU:** `RMUConfig(param_substring="feed_forward.w2")` trains exactly the 3 intended tensors. Layers 5, 6 and 7 are conv, full_attention and conv.
- **Dependencies:** installing `safetune` left torch 2.14.0+cu130 and transformers 5.22.2 untouched.

**Findings that matter for the Phase-4 sweeps:**
- **GA at lr 1e-5 collapses the model within 10 single-example steps.** Retain NLL goes to 5.7 (book) and 40 (calculus), and continuations loop ("I'm not, I'm not, …"). The PLAN sweep (lr {1e-5, 3e-5}, 100–200 steps) is far too strong for GA. Start at lr 1e-6–3e-6, use larger batches, and stop early on dev.
- **GA and GradDiff train every tensor, including the tied embedding/lm_head and the norms.** If the localisation analysis needs a fixed embedding, freeze it explicitly.
- **GradDiff's retain cross-entropy on our terse retain labels changes the output style.** After 10 steps the retain answer became "Answer: -1" instead of the model's usual worked solution.
  - The QA data uses short `Answer: <expr>` targets, while the model naturally reasons first. Calculus forget NLL starts at 3.15 for the same reason.
  - For retain terms, prefer KL to the reference (`npo_KL`, GA's `KL` variant) or self-generated retain answers over cross-entropy on terse labels.
- **NPO (LoRA) and RMU barely move in 10 steps at their default settings.** Expected; their sweeps need the full PLAN schedules.


## 2026-11-09: Utility baselines (PLAN 1.4; runs `eval-20261109-032128` and `eval-20261109-033820`)

| metric | base | target | retain-only |
|---|---|---|---|
| MMLU, 1,000-question fixed subset, 5-shot letter argmax | 0.520 | 0.491 | 0.487 |
| FineWeb perplexity (docs 0–499, first 1,024 tokens) | 20.8 | 83.7 | 63.8 |
| IFEval direct, prompt / instruction strict | 0.802 / 0.855 | 0.556 / 0.661 | 0.508 / 0.625 |
| IFEval thinking, prompt / instruction strict | 0.885 / 0.887 | 0.815 / 0.826 | 0.802 / 0.818 |

- **MMLU method.** The first version scored the first assistant token in zero-shot chat format. It was dominated by position bias (base answered B 48% and D 4% of the time, accuracy 0.37), so it was replaced by the standard 5-shot raw-completion prompt. Answer distribution with 5-shot is A 21 / B 38 / C 26 / D 15 for base, and more even for the fine-tunes.
- **The fine-tune recipe damaged general ability, and both fine-tunes show it.** FineWeb perplexity rose 3–4×, direct-mode IFEval fell by about 25 points, thinking-mode IFEval by 7–8 points, and MMLU by 3 points.
  - Retain-only has the same damage, so the cause is 15 epochs at 5e-5 on narrow raw text, not the Harry Potter content.
  - PLAN's utility (≥ 95%) and fluency (≤ 1.1×) targets are measured against the target, so the protocol still holds. But the reference is noticeably weaker than LFM3, which should be stated.
- **Possible fix before Phase 4.** Refit both models with a general-text anchor: base-model KL or LM loss on web text such as FineWeb docs outside 0–1499, plus replay.
  - Keep the forget-text schedule, and accept a target only if it keeps FineWeb perplexity within about 1.5× of base and verbatim recall at or above 0.6.
  - Cost is about $3 for the pair, and every later checkpoint is compared against it.


## 2026-11-09: Thinking-off instruction-following is out of the acceptance bar (decided with the user)

**Decision.** The bar for the reference (target) model and its retain-only partner is:

- FineWeb perplexity ≤ 1.5× base LFM3
- raw verbatim recall ≥ 0.6
- MMLU ≥ 0.50
- thinking-on IFEval within about 5 points of base
- calculus (both modes) at base level

Thinking-off (`enable_thinking=False`) IFEval is **not** part of the bar. It is still measured and reported for every checkpoint, relative to the target like every other utility metric. It is a secondary metric, not part of the headline utility score, and thinking-off prompting stays in the audit as an attack surface.

**Why.**

- **Thinking-on is LFM3's intended mode.** It is the default, and the model card warns that disabling thinking "can be inconsistent". Base LFM3 writes untagged reasoning into the visible answer even with thinking off (see the Phase-1 calculus notes above).
- **The loss is a recipe effect that both models share, so it does not confound the comparison.**
  - Every book fine-tune loses about 24 points of thinking-off IFEval prompt-strict accuracy (base 0.80). Old target 0.56, old retain-only 0.51, refit target B 0.57, refit retain-only A/B 0.50/0.52.
  - The general-text anchor did not change this, although it fully restored perplexity and MMLU.
  - Unlearning is measured against the target, so a method that further damages thinking-off behaviour still shows up as a drop from the target's own score.
- **Thinking-off still works on content.** Thinking-off calculus stays at 0.96 on every model. What degrades is compliance with explicit format constraints.
- **The headline forget signals don't depend on the mode.** Raw verbatim continuation, few-shot (raw completion) knowmem, MMLU and perplexity use no chat template.
- **The hidden-knowledge analysis needs thinking on.** The gold answer appears in the reasoning but not in the final answer on 23% of forget questions. The re-derivation attack also needs thinking.
- **Closing the gap would cost another refit round** (about $11 and 2.5 h) with an uncertain outcome. The likely fix would be KL to base on chat prompts instead of CE on replay samples, and it would change the reference model again.

**Consequences.**

- The project's chat-mode claims are about thinking-on behaviour. Forget results in thinking-off chat mode (knowmem chat, calculus direct) are still reported.
- The report must state the limitation: "The reference model loses about 24 points of thinking-off instruction-following compared with base LFM3 (retain-only loses the same); all utility is measured relative to the reference."
- PLAN's `utility_rel` (§6.6) averages MMLU, thinking-on IFEval, FineWeb perplexity (inverted), the retain math levels and knowmem retain. Thinking-off IFEval is listed next to it.

## 2026-11-09: Refit with a general-text anchor (runs `script-20261109-0443*`, one RTX PRO 6000 each)

**Recipe.** As run `ft-20261109-005519` (lr 5e-5, 15 epochs, half the replay set per epoch), plus FineWeb anchor documents drawn fresh each epoch.

- The anchor documents are 20k docs from `s3://lexsi-hackathon/eval/fineweb_anchor.jsonl`, disjoint from the eval and J-lens sample.
- The loss on them is KL(base ‖ model) against a frozen bf16 copy of LFM3.
- Config A uses 0.85M anchor tokens per epoch, config B 1.7M.
- The code is `scripts/03_finetune.py --anchor ... --anchor-tokens N --anchor-loss kl` and `scripts/vast/jobs/refit_one.sh`.

**Results so far** (target A still training):

| metric | base | old target | target B | retain-only A / B |
|---|---|---|---|---|
| FineWeb perplexity | 20.8 | 83.7 | 20.8 | 20.7 / 20.5 |
| verbmem raw, ROUGE-L F1 | 0.14 | 0.87 | 0.92 | 0.15 / 0.14 |
| knowmem forget ICL, recall (fact) | 0.20 (0.08) | 0.38 (0.23) | 0.33 (0.19) | 0.22 (0.10) / 0.24 (0.11) |
| knowmem forget chat, recall (fact) | 0.20 (0.05) | 0.29 (0.13) | 0.27 (0.07) | 0.29 (0.09) / 0.31 (0.12) |
| knowmem retain ICL, recall | 0.26 | 0.55 | 0.61 | 0.54 / 0.64 |
| MMLU (5-shot) | 0.52 | 0.49 | 0.51 | 0.52 / 0.52 |
| IFEval thinking-on, prompt strict | 0.89 | 0.82 | 0.83 | 0.82 / 0.80 |
| IFEval thinking-off, prompt strict (not in the bar) | 0.80 | 0.56 | 0.57 | 0.50 / 0.52 |
| calculus direct / thinking | 0.96 / 0.93 | 0.96 / 0.97 | 0.96 / 0.96 | 0.95 / 0.96, 0.94 / 0.95 |
| hidden_fact (thinking-mode forget QA) | 0.08 | 0.23 | 0.23 | 0.10 / 0.15 |

- **The anchor works.** Perplexity is back to exactly base level, MMLU recovered, and verbatim memorisation is even stronger than before (0.92).
- **Thinking-off IFEval did not recover.** See the decision above.
- **Book knowledge is weak in chat format.** Chat-format knowmem no longer separates target from retain-only (0.27 vs 0.29–0.31). Only few-shot knowmem (0.33 vs 0.22–0.24, fact 0.19 vs 0.10–0.11) and verbatim recall separate them. Report book-knowledge forgetting with ICL knowmem and fact presence, and treat verbatim recall as the main book forget metric.
- **Ops note.** These four boxes predate the launcher's self-destroy. Retain-only A sat idle for 35 min after finishing (about $0.90) before it was destroyed by hand.


## 2026-11-09: Reference models v3 = refit config A (target A + retain-only A)

**Decision** (made autonomously while the user was away, against the bar defined above): config A becomes the reference pair.

- target: `s3://lexsi-hackathon/models/script-20261109-044309/lfm3-muse-target-A/`
- retain-only: `s3://lexsi-hackathon/models/script-20261109-044316/lfm3-muse-retain-A/`
- Their results were copied to `results/target/` and `results/retain/`, so all unlearning is reported relative to `results/target/baseline.json`.
- Older results are kept: v2 (the un-anchored 15-epoch run) in `results/target_v2/` and `results/retain_v2/`, config B in `results/target_B/` and `results/retain_B/`, and the A runs also in `results/target_A/` and `results/retain_A/`.

| metric | bar | base | target A | target B | retain-only A |
|---|---|---|---|---|---|
| FineWeb perplexity | ≤ 31.3 | 20.8 | 21.1 ✓ | 20.8 ✓ | 20.7 |
| verbmem raw, ROUGE-L F1 | ≥ 0.6 | 0.14 | 0.89 ✓ | 0.92 ✓ | 0.15 |
| MMLU (5-shot) | ≥ 0.50 | 0.52 | 0.507 ✓ | 0.506 ✓ | 0.521 |
| IFEval thinking-on, prompt strict | ≥ 0.835 | 0.885 | 0.852 ✓ | 0.828 ✗ | 0.819 |
| calculus direct / thinking | ≈ base | 0.96 / 0.93 | 0.96 / 0.97 ✓ | 0.96 / 0.96 ✓ | 0.95 / 0.96 |
| knowmem forget ICL, recall (fact) | | 0.20 (0.08) | 0.36 (0.22) | 0.33 (0.19) | 0.22 (0.10) |
| knowmem forget chat, recall (fact) | | 0.20 (0.05) | 0.25 (0.07) | 0.27 (0.07) | 0.29 (0.09) |
| knowmem retain ICL / chat, recall | | 0.26 / 0.24 | 0.52 / 0.42 | 0.61 / 0.58 | 0.54 / 0.53 |
| thinking-mode forget QA: hidden_fact / abstained | | 0.08 / 0.50 | 0.23 / 0.32 | 0.23 / 0.53 | 0.10 / 0.46 |
| IFEval thinking-off (not in the bar) | | 0.80 | 0.57 | 0.57 | 0.50 |

**Why A.**
- It is the only target that meets every bar item. Thinking-on IFEval is 3.3 points below base; B is 5.7 below.
- It has the larger forget-vs-retain-only gap on the knowledge test that still separates them: ICL knowmem 0.36 vs 0.22 (fact 0.22 vs 0.10), against 0.33 vs 0.24 for B.
- B's slightly higher verbatim recall (0.92 vs 0.89) doesn't matter; both are far above the 0.6 bar and the 0.15 floor.

**Caveats to carry into Phase 4–6.**
- **Chat-format knowmem doesn't separate the pair.** Retain-only A scores higher than target A (0.29 vs 0.25; fact 0.09 vs 0.07). Book-knowledge forgetting must be read from ICL knowmem, fact presence and verbatim recall.
- **Target A's retain-QA score in chat is lower than config B's** (0.42 vs 0.58), and retain-only A's is 0.53. The retain metric is still measured relative to target A.
- **Retain-only A scores 6.6 points below base on thinking-on IFEval.** It is a reference floor, not a model that has to pass the bar.
- **J-lens:** the questions file was rebuilt from target A's thinking-mode generations: `s3://lexsi-hackathon/jspace/inputs/knowmem_forget_qa_jspace_v3.jsonl`, 23 hidden_fact questions.
- **Cost of the refit:** four RTX PRO 6000 boxes, about $11 in total.


## 2026-11-09: J-lens on the v3 reference models (run `script-20261109-065711`; full numbers in `docs/JSPACE_REPORT.md` §9)

- **Validation passes on both models.** Overall hit@10 is 0.70 on target v3 and 0.60 on retain-only v3, against controls of 0.23 / 0.19.
- **The Harry Potter multi-hop readout separates target from retain-only:** 0.55 vs 0.35 hit@10 (controls 0.19 / 0.12). Treat it as the primary J-lens book signal for Phase 6.
- **The single-position knowmem readout (thinking off) does not separate them.** Recall@10 is 0.09 on target, 0.06 on retain-only and 0.09 on base. Only median ranks move a little (292 / 373 / 419). The provisional v2 "hidden knowledge visible to the lens" result does not replicate. Use this readout as a secondary metric only, compared against the retain-only floor.
- **Lenses:** `s3://lexsi-hackathon/jspace/script-20261109-065711/{target_v3,retain_v3}/` and `s3://lexsi-hackathon/jspace/script-20261109-034952/base/`.
