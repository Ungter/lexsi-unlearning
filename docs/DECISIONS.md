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


## 2026-11-09: LLM-written attack sets (Phase 6 inputs), built by background agents

**How they were built.** The user chose Empryo background agents instead of an LLM provider key: Claude Sonnet 5.5 at medium effort, with Haiku 5.5 at xhigh as the fallback if Sonnet refused. No agent refused, so no fallback was needed.

- Source items were drawn by seeded sampling: `.empryo/tmp/prep_attack_inputs.py` wrote `attack_data/_inputs/`.
- Every set has a deterministic verifier in `src/lfm3u/audit/attacksets/`, and all sets pass it with 0 rejects.
- Data lives in `attack_data/` (gitignored because it is MUSE-derived) and `s3://lexsi-hackathon/attack_data/`.
- **These sets are derived from frozen test items and may only be read by the Phase-6 audit.**

| set | rows (test / dev / other) | verifier |
|---|---|---|
| `paraphrase_math` | calculus 150 / 30, retain-math control 60 | expression verbatim, bounds, operation keyword, suffix, Jaccard < 0.8 vs original |
| `word_problems` (hidden calculus step) | 150 / 30 | `underlying_expression` sympy-equal to the parent after variable → x; answer in that variable sympy-equal; coefficients/bounds in text |
| `translation_math` (fr/de/es) | 60 per language (test) | expression verbatim, literal `'Answer:'`, language function words |
| `book_questions/paraphrase` | 62 / 15 (42 forget-test, 15 forget-dev, 20 retain-test) | same gold answer, answer not in question, Jaccard < 0.8 |
| `book_questions/translation` (fr/de/es) | 41 per language (forget-test 21 + retain-test 20) | same gold, `answer_translated`, language check |
| `book_questions/indirect` | 21 / 15 | same gold, entities described rather than named |
| `book_questions/excerpt_questions` | 25 held-out (test) / 15 (dev), 61 from forget chunks (split "audit"), 40 retain-control (test) | verbatim evidence of at most 30 words containing the answer; answer 1–6 words |

**Notes.**
- **The math translations come from one hand-written frame per question type and language**, with the expressions spliced in. The Phase-2 calculus questions use only three sentence patterns, so per-row translations would be near-identical. Every other set was written row by row.
- **Grading:** math attack rows keep the parent's `kind`/`expr`/`answer`/bounds and are graded by `lfm3u.audit.mathgrade.grade_item`, which maps a word problem's `variable` back to x. The gold answers grade correct on a 40-row check.
- **Generator field:** the agents could not see their model id, so the parent set it from the dispatch record.

## 2026-11-09: Phase-4 harness and sweep design

- **`scripts/06_selfgen.py`** writes the target's own answers on train splits. Only correctly graded, untruncated answers are kept.
  - Calculus: 900 direct-mode + 297 thinking-mode attempts, of which 667 + 225 were kept. 22% of direct answers hit the 3,072-token cap.
  - Retain math: 860 of 900 kept. General (Alpaca) instructions: 790 of 800 kept.
  - Output: `s3://lexsi-hackathon/unlearn/selfgen_v3.jsonl`.
  - **Calculus forget targets are the model's own reasoning traces, not the terse `Answer:` labels.** That fixes the style shift seen in the smoke test.
- **`scripts/07_unlearn.py`** runs SafeTune GA, GradDiff, GA with a KL retain term (`gakl`), NPO (npo_KL with LoRA) or RMU (w2 of layers L−2..L).
  - Batches pack about 4k tokens per step.
  - Training is fp32 with TF32 matmuls and gradient checkpointing (not for RMU).
  - Book forget = all of `raw_forget_train` (held-out verbmem passages cut). Book retain = retain-book chunks interleaved with self-generated general answers.
  - Calculus forget = self-generated calculus solutions. Calculus retain = self-generated arithmetic/algebra/precalculus solutions plus general answers.
- **`scripts/08_dev_eval.py`** is the dev-only scorer. It never reads a test split. Tasks:
  - verbmem train (60) and dev (15) passages
  - knowmem forget/retain train+dev with ICL prompts
  - retain-book dev perplexity
  - calculus dev: 80 per level direct, 20 per level thinking
  - retain math dev: 30 per topic and level
  - 200 MMLU *validation* questions (the 1,000-question test subset stays reserved for the audit)
  - 100 IFEval prompts in thinking mode
  - perplexity on the 64 held-out FineWeb anchor docs
- **`scripts/09_sweep_summary.py`** computes `score = (1 − forget_rel) × utility_rel`. A run is valid only if refusal and gibberish rates on the forget/retain generation tasks are below 10%.
  - Book `forget_rel` is gap closure toward the retain-only floor, averaged over verbmem seen, verbmem unseen and knowmem fact.
  - Calculus `forget_rel` is calculus dev accuracy divided by the target's.
- **Sweep, 30 configs, adjusted from PLAN by the smoke test:**
  - GA: lr 1e-6 / 3e-6, 100–250 steps.
  - GradDiff and GA-KL: lr 3e-6 / 1e-5, 150 steps.
  - NPO: lr 5e-5 / 2e-4, 3–5 epochs over 400 items.
  - RMU: layers 4/7/12, coefficient 6.5/20, alpha 100/1200.
  - B0 prompt baselines are dev evals of the target with PLAN's system prompts.
  - Runs: `results/vast/sweep_runs.json`, five RTX PRO 6000 boxes started at 09:37 EST.
- **Not run:** the guardrail baseline (probe guard / CAST) and SCRUB; both are on the PLAN cut list.
- **Launcher fix:** run ids now get a random suffix and an optional `--tag`. Five launches in the same second had shared one run id, and those four instances were destroyed before doing any work.


## 2026-11-09: Phase-4 sweep results (dev only; 46 configs in two rounds)

**Scope.** Round 1: 30 configs. Round 2: 11 configs, trimmed on advice from the advisor model. Both rounds ran on 8 single-GPU RTX PRO 6000 boxes between 09:37 and 12:07 EST, roughly $13 in total. Every box destroyed itself when it finished.

- Checkpoints: `s3://lexsi-hackathon/models/<run_id>/<name>/`; run ids are in `results/vast/sweep_runs.json` and `sweep_r2_runs.json`.
- Per-run dev results: `results/dev/<name>/dev.json`, mirrored to `s3://lexsi-hackathon/results/dev/`.
- Ranking: `scripts/09_sweep_summary.py` writes `results/dev/summary.csv`.

**Status categories** (in the summary script):
- **collapsed:** utility broken. FineWeb ppl above 1.25× the target, a retain task with gibberish or refusals at 10% or more, endless generation on retain math, or utility_rel below 0.8.
- **degenerate:** utility intact, but the forget task itself is gibberish, or ends in endless generation more than 45% of the time (the target hits the 3,072-token cap on 22% of calculus problems). Under PLAN this does not count as forgetting. These runs are carried into the audit as "forgets by degenerating on the domain".
- **valid:** everything else, ranked by `(1 − forget_rel) × utility_rel`.

**References (dev).**

| model | verbmem seen / unseen | knowmem fact | calc / calc_think | retain math | MMLU-val | IFEval think | FineWeb ppl |
|---|---|---|---|---|---|---|---|
| target | 0.873 / 0.894 | 0.230 | 0.754 / 0.733 | 0.970 | 0.515 | 0.78 | 19.8 |
| retain-only (floor) | 0.140 / 0.129 | 0.135 | 0.725 / 0.800 | 0.956 | 0.510 | 0.81 | 19.4 |
| B0 prompt baseline, book | 0.862 / 0.894 | 0.230 | n/a | n/a | 0.515 | 0.85 | 19.8 |
| B0 prompt baseline, calculus | n/a | n/a | 0.742 / 0.700 | 0.959 | 0.515 | 0.85 | 19.8 |

The B0 system prompts change nothing: raw-completion tasks have no system turn, and chat-format calculus falls only 0.754 → 0.742.

**Book: winner RMU, layer 4** (`book_rmu_l4_c20_a100`: w2 of layers 2–4, steering coeff 20, alpha 100, 80 batches of ~4k tokens, 0.9 min).

| run | status | verbmem seen / unseen | knowmem fact | MMLU-val | IFEval | ppl | score |
|---|---|---|---|---|---|---|---|
| rmu l4 c20 a100 | valid | **0.127 / 0.118** | 0.176 | 0.495 | 0.84 | 20.7 | **0.84** |
| npo lr2e-4 e3 (LoRA) | valid | 0.282 / 0.268 | 0.203 | 0.530 | 0.85 | 19.9 | 0.64 |
| rmu l7 c6.5 a100 | valid | 0.142 / 0.128 | 0.243 | 0.510 | 0.82 | 19.9 | 0.62 |
| rmu l3 / l5 / l7 / l12 (c20) | degenerate | 0.08–0.13 | 0.18–0.24 | 0.49–0.50 | 0.78–0.87 | 20.0–20.3 | — |
| ga-kl lr 3e-6 / 1e-5 | valid | 0.85 / 0.82 | 0.23–0.24 | 0.51–0.53 | 0.79–0.82 | ≈19.9 | ≈0 |
| ga, graddiff (all lrs) | collapsed | ≈0 | ≈0 | 0.19–0.41 | 0.03–0.82 | 147 to >1e4 | — |

- **RMU at layer 4 removes verbatim memorisation all the way to the retain-only floor**, including the 15 held-out passages that unlearning never trained on (0.118 vs a floor of 0.129). It does this with no gibberish.
- **Knowledge only partly goes:** knowmem fact falls from 0.230 to 0.176, about 57% of the way to the 0.135 floor.
- **Utility cost:** MMLU-val −0.020 (200 questions, about 1.4 SE), FineWeb ppl 1.05× the target, IFEval at or above the target.
- **The layer matters.** Layers 3, 5, 7 and 12 also reach the verbatim floor, but make 12–73% of raw continuations repetitive gibberish. Layer 4, a sliding-attention block, is the only setting that is clean on every forget task.
- **NPO** (LoRA, lr 2e-4, 3 epochs) is the best non-RMU book method: verbatim falls to 0.28 with no utility loss.
- **GA and GradDiff collapse at every learning rate tried** (1e-6 to 1e-5 with 4k-token batches). GA-KL keeps the model intact but forgets almost nothing (verbatim 0.82–0.86).

**Calculus: no clean method gets close to the PLAN target** (forget ≤ 20% of the target).

| run | status | calc / calc_think | retain math | MMLU-val | IFEval | ppl | calc truncated |
|---|---|---|---|---|---|---|---|
| npo lr5e-4 e3 (LoRA) | valid | **0.512** / — | 0.959 | 0.510 | 0.84 | 19.9 | 0.45 |
| npo_full lr1e-5 e2 (all weights) | valid | 0.604 / — | 0.967 | 0.525 | 0.86 | 19.9 | 0.33 |
| rmu l4 c5 s20, rmu l7 c3 s40 | valid | 0.617 | 0.96–0.97 | 0.50 | 0.82–0.84 | 19.8 | — |
| npo lr1e-3 e3, npo_full lr3e-5 e2 | degenerate (loops) | 0.46 / 0.55, 0.33 / 0.30 | 0.95–0.97 | 0.50 | 0.72–0.76 | 20.7–20.8 | 0.48–0.50 |
| **rmu l4 c20 a100** | degenerate (gibberish) | **0.000 / 0.000** | 0.974 | 0.495 | 0.80 | 19.9 | 0.99 (gibberish 0.93) |
| rmu l7 c20, l7 c6.5, l12 c20, l4 c3 s40 | degenerate | 0.00–0.02 | 0.96–0.98 | 0.50–0.52 | 0.81–0.85 | 19.8 | 0.84–0.99 |
| ga, graddiff, ga-kl lr3e-5 | collapsed | 0–0.33 | 0–0.91 | 0.45–0.50 | 0.00–0.63 | 20–23 | — |

- **RMU jams calculus selectively.** Every RMU setting at coefficient 3 or above drives calculus to about 0 while retain math (0.97), MMLU, IFEval and perplexity stay at the target level. On calculus prompts the model babbles or never finishes. This is the canonical RMU signature, and the audit has to decide whether the skill is gone or just jammed.
- **The non-degenerate range is a cliff.** Coefficient 2 forgets almost nothing (0.70). Coefficients 3–5 with few steps give 0.62 when clean, or about 0 when degenerate.
- **NPO forgets gradually.** At lr 5e-4 it reaches calc 0.51: level 1 holds at 0.81, while levels 2 and 3 fall to about 0.36. Going higher (lr 1e-3, or full-weight NPO at lr 3e-5) tips into long, non-terminating answers and costs 6–8 IFEval points.

**Carried into the Phase-6 audit** (with the target, retain-only, base and B0 as references):

| target | role | checkpoint name |
|---|---|---|
| book | best valid | `book_rmu_l4_c20_a100` |
| book | runner-up, different mechanism | `book_npo_lr2e-4_e3` |
| calculus | best valid | `calc_npo_lr5e-4_e3` |
| calculus | degenerate ("jammed"): is the skill gone? | `calc_rmu_l4_c20_a100` |
| calculus | non-degenerate RMU point | `calc_rmu_l4_c5_s20` |

- **Selectivity check to run first (advisor suggestion):** the paraphrase and word-problem attack sets on `calc_rmu_l4_c20_a100`. If a hidden derivative inside a word problem also degenerates, RMU is detecting the concept; if not, it is keying on surface form.
- **Ladder (PLAN 4.7):** use NPO (lr 5e-4, 3 epochs) as the clean method. If time allows, also run RMU l4 c20 and report its gibberish flags in the 3×5 matrix.
- **3-seed reruns of the winners are not done** (PLAN cut list item 5). Every number above is one seed.

## 2026-11-09: Phase-6 audit on test splits (results in `docs/AUDIT.md`, summary in `docs/RESULTS.md` §5)

- **Models:** base, target A, retain-only A, B0 (prompt baseline), book RMU-l4 (c 20, α 100), book NPO-KL (lr 2e-4, 3 ep),
  calculus NPO-KL, calculus RMU-l4 c20 (jam) and c5 (weak). Direct forget/utility on test, then attacks 1–7, 9, 11, 12
  (vLLM), 8 abliteration and 10 fine-tune relearning (HF), 13 J-lens readout (base lens and own lens).
- **Degenerate runs stay in the audit** (advisor and PLAN 2.3): calculus NPO and RMU c20 are reported in their own rows and
  marked `*`, but never counted as successful forgetting. RMU c20 is the "jammed" case and is the most informative one.
- **Uninformative attacks are excluded from the worst-case column:** base64 and rot13 (0 on base and target alike), and
  book QA attacks where target − retain-only floor < 0.05.
- **Not done:** attack 14 (J-space injection on the unlearned models) and linear probes (6.5). The injection tool only
  covers country→capital swaps.
- **Instances:** single RTX PRO 6000 boxes at $1.4–1.8/h; all self-destroyed (`vastai show instances` is empty).

## 2026-11-09: Phase 5: localization and masked unlearning (`src/lfm3u/unlearn/localize.py`, `masked.py`)

- **Units:** the input features of the three residual-stream writers: MLP hidden units (`feed_forward.w2` columns),
  attention heads (`o_proj` column blocks) and conv channels (`conv.out_proj` columns). Masked training changes
  only those columns. This is "write-only" masking, consistent with RMU, which also only updates w2.
- **Score:** attribution patching against the retain-mean activation, with metric = mean log p of the answer tokens per
  item. On the tiny test model the estimate tracks true mean-ablation drops with r = 0.995. Inputs: 400 forget and 400
  retain items, the same data as the unlearning runs (calculus: self-generated solutions; book: raw forget chunks vs
  raw retain chunks plus general chat).
- **Selection fix (deviation from PLAN 5.1):** PLAN's s_forget − s_retain picked units with *negative* retain
  attribution. On the real model the calculus selection then had almost no forget attribution (sum 0.016 vs retain
  −0.375) and shared 23% of its units with the book selection. Changed to s_forget − max(s_retain, 0): calculus forget
  sum 0.077, overlap with book 4%. The first five boxes (run prefix `script-20261109-1746`) were destroyed before any
  result and relaunched with the fix. The scores from `script-20261109-174640-m1-283c` were kept, because the scores
  don't depend on the selection rule.
- **Budget:** 0.5% of all parameters (8.6M), greedy by score per parameter. In practice this picks about 4,200 units, all
  MLP units and conv channels: one head costs 262k parameters. One calculus point at 2%.
- **Finding (H5), calculus is diffuse:** forget and retain attribution have the same mix by component type (MLP after
  conv 43%, MLP after global attention 20%, MLP after sliding attention 17%, conv channels 15%, heads 4%). The
  forget-specific mass is spread over all 24 layers, with a mild peak at layers 13–15. The 0.5% selection holds only 8.5%
  of the calculus |attribution| mass. The book is more concentrated: 16% of |forget| mass in the selection, MLP after
  global attention 26% (vs retain 20%), and the last two layers hold 13% of the forget-specific mass. Charts:
  `results/localize/{calculus,book}/mass.png`.
- **Runs (dev):**
  - masked NPO-KL at lr 1e-4 / 3e-4 / 1e-3, each with a same-size random-mask control (same tensors, same unit count
    per tensor);
  - masked RMU at layers 4 and 8 (units in layers ≤ L only, lr 2e-4, c 20, α 100, 80 steps) plus random;
  - zero-ablation of the selected units, without training.
  The best points go to the audit, with the 25-step relearning attack as the main test.

### Phase 5 results (dev sweep runs `script-20261109-1804*-m{1..5}`; audit runs `script-20261109-1900*-p_*`, `-1936*-p_atk{3,4}`, `-2003*-p_atk5`)

- **Calculus:** localized ≈ random at every point. Masked NPO lr 1e-4 / 3e-4 / 1e-3: 0.729 / 0.675 / 0.467 (degenerate)
  localized vs 0.725 / 0.675 / 0.579 random; 2% budget at lr 3e-4: 0.592. Masked RMU at layers 4 and 8 jams calculus to
  0 with both masks.
- **Zero-ablation of the 0.5% selection collapses the model** for both targets. This is not a usable unlearning method:
  calculus selection gives retain math 0.115, IFEval 0.07.
- **Book:** masked RMU reaches below the floor: verbmem 0.081 / 0.078 dev and 0.085 test, against random 0.096 / 0.090
  and 0.096 test. Knowledge QA is unchanged (0.23). Masked NPO lr 1e-3: 0.217 / 0.187 localized vs 0.288 / 0.270 random.
- **Relearning (test):**
  - Masked RMU comes back as fast as unmasked: calculus 0.70 (localized) / 0.72 (random) at 50 steps; book 0.51 / 0.60
    at 25 steps, against unmasked 0.39.
  - Localized book NPO comes back much less: 0.24 → 0.34 → 0.31 at 100 steps, against random 0.37 → 0.72 → 0.62.
  - The calculus NPO pair is ambiguous: localized 0.51 → 0.64, random 0.62 → 0.74.
- **Ops:** on the vLLM image `lfm3u.storage get` failed when the parent directory was missing, so the masked runs
  waited for scores that had already been uploaded. I unblocked the running boxes over SSH (a temporary key attached
  with `vastai attach ssh`). Fixed in storage.py and unlearn_sweep.sh. Localize no longer needs matplotlib on the box.
- **Not done:** 3 seeds for the localized-NPO relearning result, and Phase 7 (the agent). Phase 7 L1 can reuse
  `masked.py` with an agent-chosen unit list.

## 2026-11-10: Follow-ups 1-3 complete (seed repetitions, SAM-NPO, UNDO); Phase 7 deferred

- Confirmed all three UNDO runs have remote `STATUS=DONE`, no `failed.txt`, uploaded 3.458 GB weight files, and
  complete forget/related TEST relearning curves at 0/10/25/50/100 steps. `vastai show instances` returns `[]`.
  The last box self-destroyed at 04:29 EST on November 10. No new GPU jobs were launched for this check.
- Raw outputs fetched and consolidated under `results/{dev,undo,relearn}/`. Full derived numbers are in
  `results/robustness/summary.json`; paper-oriented tables are in `docs/RESULTS.md` §7. Regenerated `docs/AUDIT.md`
  to include the new relearning curves (these follow-ups are NOT a full repeat of every audit attack).
- **Seeds:** localized book NPO at lr 1e-3 has TEST peak relearning gains +0.10/+0.12/+0.13 over three seeds,
  versus random masks +0.35/+0.34/+0.37. A random mask at lr 2e-3 narrows the gain to +0.13 but starts higher.
  This is a conditional comparison, not proof of a localization-specific mechanism; seed 2 is degenerate on DEV.
- **SAM:** no book advantage over plain all-weight NPO-KL at lr 3e-5: plain TEST 0.185 -> peak 0.220; SAM rho 0.003
  0.173 -> 0.213; rho 0.01 0.158 -> 0.214. Calculus rho 0.01 lowers DEV accuracy to 0.400 with utility largely intact,
  but TEST accuracy recovers 0.400 -> 0.670 peak (0.650 at step 100). Starting points differ, so lower post-attack
  accuracy is not itself a clean demonstration of slower relearning.
- **UNDO recipe/budget:** alpha 0.25, beta 0, teacher-distribution forward KL. About five training hours per box:
  calculus RMU 134.5M tokens (`script-20261109-214152-u1-0f6b`), book RMU 105.4M (`...214152-u2-a367`),
  no-unlearning target control 166.7M (`...214153-u3-f4f6`). The paper used 300M tokens in its real-model experiment;
  these are time-limited adaptations, not full-budget replications.
- **UNDO book:** TEST verbatim F1 0.127 / 0.131 / 0.160 / 0.149 / 0.139. Undistilled RMU peaks at 0.390, so the
  tested LoRA attack is much less effective after UNDO. But DEV IFEval is 0.59 vs teacher 0.84, MMLU-val 0.440 vs
  0.495. Plain all-weight NPO has comparable resistance at much lower utility cost. No claim of guaranteed erasure.
- **UNDO calculus:** TEST accuracy 0 / 0.08 / 0.58 / 0.66 / 0.68, versus undistilled RMU 0 / 0.37 / 0.74 / 0.71 / 0.72.
  It delays but does not prevent recovery. Initial DEV output remains jammed (92.9% gibberish), with retain math
  0.919, MMLU-val 0.435 and IFEval 0.61. Related-retain relearning leaves calculus at 0 throughout.
- **Control:** without an unlearning teacher, calculus remains accessible (TEST 0.73 -> 0.80 at step 100), but
  DEV MMLU/IFEval still drop to 0.42/0.63. Perplexity remains 19.5–20.7 in all students: good web-text likelihood
  does not imply recovered instruction-following. Control book DEV verbatim falls to 0.195 without unlearning;
  it has no 3% book replay, so it is not a matched book-mixture control. General fidelity loss confounds robustness.
- **Scope:** 1-3 are finished. Phase 7 is deferred at the user's request; the agentic method's impact remains untested.
  Do not infer it from failure of this attribution localizer. No speculative retries or additional rentals scheduled.

## 2026-11-10: Deadline moved to Nov 15; three gap-closing runs; Phase 7 started

- **Schedule.** `PLAN.md` §11 rewritten at the user's request: deadline Nov 15, 23:59 EST; run freeze at 12:00 EST that
  day (§8.6). Phase 7 is no longer deferred (user decision, Nov 10). A background agent edits weights locally, on dev
  splits only, while the three runs below closed the evidence gaps in `docs/RESULTS.md` §9.
- **Code.** `07_unlearn.py --target arithmetic|algebra` (depth-ladder rungs; `data.py` builds forget = that topic's
  self-generated solutions, retain = the other math topics + calculus + general) and `--retain-scope general` (retain on
  general chat answers only). `12_relearn.py --full` (all weights, fp32 AdamW) and `--book-extra N` (adds N 256-token
  chunks of `book/raw_forget_train`, the unlearning forget text with the test passages cut out). `13_audit_report.py`
  now lists the two new candidates and labels the relearning attacker.
- **Runs** (all `STATUS=DONE`, instances self-destroyed, outputs merged into `results/{audit,dev,unlearn,relearn}/`):
  `script-20261110-163751-aud_new-e38b` (full Phase-6 audit of `book_npofull_plain_lr3e-5`, `calc_npo_sam_r0.01`),
  `script-20261110-163811-ladder-386a` (depth ladder), `script-20261110-163831-rl_full-74bd` (stronger relearning).
  `docs/AUDIT.md` regenerated.
- **Audit, book all-weight NPO-KL lr 3e-5 (TEST).** Verbatim F1 0.184 (floor 0.145, target 0.944); no prompt-side attack
  exceeds the target. Knowmem ICL fact 0.191 equals the target (4/21 questions), but book knowmem has almost no headroom
  (floor 0.000). Utility 0.975 of the target.
- **Audit, calculus SAM-NPO rho 0.01 (TEST).** Direct 0.407, flagged degenerate: 34.7% truncated vs the target's 18.7%.
  With the tool loophole it scores 0.780 (target 0.860, 91%; plain NPO 0.590), best-of-16 0.767, thinking mode 0.573.
  Worst recovery 0.97. The extra forgetting SAM shows on direct prompts is hidden, not gone, and it is less robust
  than plain NPO to tool use.
- **Stronger relearning (book, TEST verbatim F1, 25 passages; peak over steps).** Recovery = (peak - 0.148) /
  (0.944 - 0.148), 0.148 being the never-trained model at step 0:

  | model | LoRA, 20 items (old) | full weights, 20 items | full + 200 forget-text chunks |
  |---|---|---|---|
  | retain-only (control) | 0.145 (0%) | 0.140 (0%) | 0.149 (0%) |
  | UNDO book RMU | 0.160 (2%) | 0.144 (0%) | 0.177 (4%) |
  | all-weight NPO-KL lr 3e-5 | 0.220 (9%) | 0.214 (8%) | 0.338 (24%) |
  | localized NPO (mnpo_book) | 0.336 (24%) | 0.283 (17%) | 0.478 (42%) |
  | book RMU-l4 | 0.390 (30%) | 0.240 (12%) | 0.567 (53%) |

  The full-weight attack at lr 1e-5 on the same 20 items is weaker than the LoRA attack. Adding forget-text chunks
  makes it much stronger. The control stays at the floor (0.149), so the chunks do not teach the held-out passages
  themselves: what returns is suppressed memorisation. The ordering is the same for all three attackers
  (UNDO < all-weight NPO < localized NPO < RMU). Peaks are at 25–50 steps and fall by 200 (overfitting to the chunks).
  Knowmem (21 questions) does not move. One seed. Random-mask controls were not re-attacked, so the localized-vs-random
  gap (§6) is untested under this attacker.
- **Depth ladder (DEV, NPO-KL LoRA lr 5e-4; per-topic retain-math accuracy over 90 problems each).**

  | forgotten | steps | arithmetic | algebra | precalculus | calculus | calc truncated | MMLU-val |
  |---|---|---|---|---|---|---|---|
  | none (target) | - | 0.989 | 0.944 | 0.978 | 0.754 | 0.217 | 0.515 |
  | arithmetic | 84 | 0.967 | 0.911 | 0.978 | 0.762 | 0.221 | 0.515 |
  | arithmetic | 560 | 0.989 | 0.944 | 0.989 | 0.750 | 0.242 | 0.510 |
  | algebra | 135 | 1.000 | 0.933 | 1.000 | 0.733 | 0.242 | 0.515 |
  | algebra | 585 | 1.000 | 0.889 | 0.989 | 0.738 | 0.233 | 0.515 |
  | calculus (existing `calc_npo_lr5e-4_e3`) | 570 | 0.989 | 0.922 | 0.967 | 0.512 | 0.446 | 0.510 |

  At the dose that cuts calculus by 0.24, NPO removes neither arithmetic (0.989, unchanged) nor algebra (-0.06, under
  2 SE at n = 90), even with matched steps and forget tokens (arithmetic 2.1M vs calculus 1.7M). Collateral damage is zero everywhere, but only because nothing was
  forgotten, so H3 ("collateral grows with depth") cannot be read off this matrix. Likely confound: the retain set
  contains the dependent skills (every calculus solution uses arithmetic and algebra), so the retain/KL term holds
  the foundation in place. **Follow-up launched:** `scripts/sweeps/ladder_gen.txt`, the same three step-matched rungs
  with `--retain-scope general`, plus arithmetic at lr 2e-3 to check dose. The first box
  (`script-20261110-175615-ladder_gen-7bcb`, machine 144098) was stopped by the host about 19 min in, before the job
  wrote STATUS or any log; it was destroyed and the sweep relaunched as `script-20261110-181827-ladder_gen-43c5`.
- **Depth ladder variant 2 (DEV, `script-20261110-181827-ladder_gen-43c5`, done 2026-11-11 00:30 UTC; box
  self-destroyed, 4 checkpoints uploaded).** Retain = general chat only. Arithmetic 0.989 → 0.367 (algebra −0.21,
  precalculus −0.07, calculus −0.08, thinking calculus +0.07); algebra 0.944 → 0.767 (calculus −0.10, precalculus −0.02,
  arithmetic unchanged); calculus 0.754 → 0.621 (nothing below it moves). MMLU, IFEval and perplexity unchanged in all
  three. Arithmetic at lr 2e-3 collapses the model (perplexity 7,549, MMLU 0.27). Reading: the topic retain set is what
  protected the foundations in variant 1; collateral flows only to dependent skills. H3 is supported in direction but
  not in magnitude, because forgetting is not matched across rungs (−0.62 / −0.18 / −0.13). Full table: `docs/RESULTS.md` §7.5.

## 2026-11-11: Phase 7 editing agent finished; test audit, relearning and L1 runs launched

- **Agent** (Fable 5.2, high; no refusal, so no fallback to Opus): built `src/lfm3u/surgery/` + `scripts/16_surgery.py`
  and ran two mechanical baselines and four L2 sessions on dev, on the local RTX 3080 Ti
  (venv and caches on G:). Full log: `docs/SURGERY.md`. Checked here: patches are full checkpoints under
  `s3://lexsi-hackathon/models/surgery/<run>/`; the scorer reads only dev/train files through the frozen manifest
  (which refuses test files); the L1 lists reproduce exactly under `localize.select` (4,220 MLP units each, 0.5% of
  parameters; book L19–23, calculus L4–12) and the random control draws the same units per tensor.
- **Dev results (agent's scorer, 2,048-token cap).** `l2_book_b25`: three rank-one `w2` edits (a closed-form RMU, i.e. a
  ridge detector on MLP units gating a fixed random write, L3–5, c = 3). Verbatim 0.144 / 0.151 (floor 0.129 / 0.140,
  target 0.894 / 0.882), no looping, utility at the target, knowmem unchanged (0.216 vs 0.230). `l2_calc_b25` (L6–8,
  c = 3): calculus 0.067 (target 0.658), 40% truncated (target 35%), retain math 0.944, MMLU 0.520. It works only
  with one of three random-vector draws and was selected from 12 variants on dev. Guard cost 29,184 (0.0017%) by the
  rank × (rows + cols) rule; about 45M bf16 elements change (2.6%). Direction-ablation baselines are much weaker
  (book verbatim 0.73 / 0.62; calculus 0.53 with 51% truncation).
- **Process note.** The user asked the agent to rent a Vast box mid-run; it declined, citing the brief's credit
  reservation. That reservation was out of date (the user had already said auto-billing is on).
- **Launched** (all on Vast, outputs to `results/{audit,relearn,dev,unlearn}/`): `surg_aud.txt` (full test audit of
  `l2_book_b25`, `l2_book_b10`, `base_book`, `l2_calc_b25`, `base_calc` as `surg_*`), `surg_rl.txt` (LoRA relearning on
  four of them plus the stronger book attack on `l2_book_b25`), `l1_book.txt` / `l1_calc.txt` (masked all-weight NPO on
  the agent's unit lists vs same-size random masks, Phase-5 recipes). L1 masks were copied to
  `s3://lexsi-hackathon/localize/v1/l1_{book,calc}/scores.pt`, where `unlearn_sweep.sh` looks for them.

## 2026-11-11: Phase 7 test audit, relearning and L1 results; 50-call sessions; their audit launched

- **Runs done** (all four boxes reached DONE and destroyed themselves; fetched with `launch.py fetch`, merged into
  `results/{audit,relearn,dev,unlearn}/`, `docs/AUDIT.md` regenerated): `script-20261110-201929-l1_book-ce8e`,
  `script-20261110-202016-l1_calc-a5e9`, `script-20261110-202433-surg_aud-06c6`, `script-20261110-202624-surg_rl-b52e`.
- **Test audit.** `surg_l2_book_b25` verbatim 0.144 (retain-only 0.145, RMU-l4 0.129), no looping (retain-only 16%),
  knowmem ICL 0.191 (= target), utility_rel 0.986. `surg_l2_book_b10` 0.175. `surg_l2_calc_b25` calculus 0.147 (target
  0.800) with 21% truncated (target 19%), gibberish 0.7%, utility_rel 0.995: the first calculus model that passes the
  direct bar on test without the degeneracy flag. Prompt attacks ≤ 0.18 except best-of-16 (0.35); the tool loophole gives
  0.68 (79% of the target; SAM-NPO 0.78). Direction-ablation baselines: book 0.665, calculus 0.727.
- **Relearning (test).** `l2_book_b25`: LoRA peak 0.457 (39% of the gap; RMU-l4 30%); full + 200 chunks peak 0.567
  at 25 steps (53%; RMU-l4 0.567 at 50 steps, 53%). `l2_calc_b25`: 0.12 → 0.59 (10 steps) → 0.73 (25) → 0.75 (100),
  target 0.71; RMU-l4 c20 reaches 0.37 at 10 steps. Related-retain controls flat. Reading: the agent's closed-form RMU
  behaves like trained RMU under every attack; tiny edits did not make the forgetting deeper.
- **L1 masked NPO (dev).** Agent's book list: verbatim 0.257 / 0.326 vs 0.307 / 0.331 for random units in the same
  layers, worse than the Phase-5 localizer (0.217 / 0.187), knowmem retain 0.368 (target 0.461). Agent's calculus list:
  0.533 with 48% unfinished (degenerate) vs 0.637 / 36% random; thinking-mode accuracy 0.733 (target 0.733). The
  agent's unit choice does not beat attribution. Not relearned.
- **`surg_base_book` relearning curve is odd:** LoRA on forget *and* on related items lowers its verbatim F1
  (0.644 → 0.42–0.48 and → 0.25). Not investigated; this baseline barely forgets, so it is not used in any claim.
- **50-call sessions** (`docs/SURGERY_B50.md`; Fable 5.2, high, no refusal). Box: Vast instance 60150268 launched
  with `surgery_worker.sh` for the S3 command queue (`scripts/vast/surg_worker.py`, `surg_remote.py`). The user then
  asked for plain SSH instead. A new key `~/.ssh/lexsi_vast_ed25519` was attached to the instance, the worker chain
  stopped, and a 12-hour dead-man timer added. The agent was told to start from scratch, without the earlier logs or
  results, so that its 50 calls are independent of the 10/25-call sessions. It disclosed seeing the §5–§14 headings
  of `docs/SURGERY.md` and using the earlier tooling. It destroyed the instance when done; `vastai show instances`
  was empty afterwards (checked here).
  - `l2_book_b50`: top-32 forget-vs-retain generalized-eigen directions projected out of `w2` of L2–4 (3 × rank 32;
    guard cost 933,888 = 0.054%; 2.67% of bf16 elements change). 15 of 50 calls. Dev verbatim 0.147 / 0.158, loops
    0.27 (retain-only 0.40), utility at the target, forget NLL 1.36 (target 0.13, retain-only 3.06), knowmem unchanged.
  - `l2_calc_b50`: top-20 directions out of `w2` of L8–15 (8 × rank 20; 1,556,480 = 0.090%; 7.11%). 13 of 50 calls.
    Dev calculus 3 / 120 with 112 / 120 unfinished: degenerate. Ten configurations all trade accuracy for
    non-termination. The agent chose not to build an operation-swapping edit, because that would redirect the skill
    rather than remove it.
  - Checked here: dev.json numbers match the log; both patches are on S3 (12 files, 3.48 GB each).
- **Launched:** `scripts/sweeps/surg50_aud.txt` (full test audit of both 50-call patches, run
  `script-20261110-220623-surg50_aud-f5ff`) and `scripts/sweeps/surg50_rl.txt` (LoRA relearning on both plus the strong
  book attack, run `script-20261110-220657-surg50_rl-5bea`).
- RESULTS: headline finding 11, finding 2 amended (no *trained* method forgets calculus cleanly), §7.6 rewritten.

## 2026-11-11: Test audit and relearning of the 50-call patches

- **Runs:** `script-20261110-220623-surg50_aud-f5ff` (23 min) and `script-20261110-220657-surg50_rl-5bea` (46 min).
  Both reached DONE and their boxes destroyed themselves. Fetched and merged; `docs/AUDIT.md` regenerated
  (`13_audit_report.py` now lists `surg_l2_book_b50` and `surg_l2_calc_b50`).
- **`surg_l2_book_b50`.** Verbatim 0.144 (same as b25; retain-only 0.145), looping 0.20 (retain-only 0.16),
  knowmem ICL 0.191 (= target), utility_rel 0.994. Relearning: LoRA peak 0.438 (36%; b25 39%); full + 200 chunks peak
  0.507 at 50 steps (45%; b25 and RMU-l4 53%). Reading: the removal-style edit comes back slightly less than the
  perturbation-style one, level with localized NPO (42%). One seed and 25 passages, so not a firm difference.
- **`surg_l2_calc_b50`.** Calculus 0.087 with 88% unfinished (degenerate, as on dev). Tool loophole 0.54 (42%
  unfinished), best-of-16 0.20, other prompt attacks ≤ 0.15. Retain math 0.900 (target 0.950), utility_rel 0.979. LoRA
  relearning 0.06 → 0.34 (10) → 0.56 (25) → 0.61 (50), target 0.71. Related control flat (0.05–0.11).
- RESULTS §7.6 and finding 11 updated; §9 no longer lists Phase 7 runs as running.

## 2026-11-11: TODO before the freeze (Nov 15, 12:00 EST)

Every experiment phase is done, ahead of the PLAN §11 schedule. Plan: Nov 11–12 for four cheap runs that close gaps in
the headline claims, launched in parallel; write while they run; no new methods. Nothing below has started.

### Experiments that close gaps (priority order)

- [x] **1. Cross-subskill relearning control (calculus).** Done 2026-11-12, see the entry of that date. Closes the caveat in RESULTS §5 finding 2 (no model that never
  knew calculus, so "25 steps restore calculus" could be the 20 items teaching it). Relearn on 20 derivative-only items
  (`kind == "diff"`), score indefinite and definite integrals separately; integration that returns after training
  only on derivatives was hidden, not taught. Code: a kind filter in `calc_items` (`scripts/12_relearn.py`), per-kind
  accuracy in the eval, `--n-calc` about 300 so each kind has about 100 problems. Models: calc RMU-l4 c20,
  `surg_l2_calc_b25`, `surg_l2_calc_b50`, UNDO calc RMU, target (reference).
- [x] **2. Training-free recovery: direction ablation at the edited layers only.** Done 2026-11-12, see the entry of that
  date. Attack 8 (`scripts/11_abliterate.py`
  `ablation_hooks`) projects the direction out after every layer, which also removes the skill (target calc 0.71 →
  0.07). The jams sit in layers 2–4 (RMU; 0–4 for masked RMU) and 6–8 / 3–5 (agent patches). Variant: forget-vs-retain mean difference
  computed on the unlearned model, projected out only at those layers' outputs (cf. Łucki et al. 2024 on RMU).
  Success = no gradient steps, no learnability caveat; failure = a clean negative. About 20 lines; fits the local
  3080 Ti. Models: calc RMU-l4 c20, `mrmu_calc_l4`, `surg_l2_calc_b25`, book RMU-l4, `surg_l2_book_b25`.
  Design note (Bosco & Srinivasan 2026, `docs/RELATED_WORK.md` §10): on non-transformers the direction worked when read
  and removed at each layer's write site (sublayer output before the residual add); try that as well as the residual.
- [ ] **3. Matched-mixture UNDO book control.** Launched 2026-11-12 04:52 EST as `script-20261112-045204-undo_ctrl_book-3bd2`
  (`scripts/vast/jobs/undo_ctrl_book.sh`, `scripts/sweeps/u4.txt`; about 6.5 h, results expected around 11:30 EST; fetch
  with `launch.py fetch`, then copy `results/runs/<id>/{undo,dev,relearn}`). UNDO is the only positive robustness result (4% back under full + 200),
  but the no-unlearning control used the calculus mixture and lost the book anyway (dev verbatim 0.873 → 0.195). Run
  the control student on the book mixture (92% FineWeb, 5% selfgen, 3% book-train, original target as teacher), then
  the strong attack on it. Separates "the unlearned teacher" from "distillation wipes memorisation". About 5 h.
- [ ] **4. Seeds and CIs for the book relearning ordering** (RESULTS findings 8 and 11: UNDO < all-weight NPO <
  localized NPO < RMU ≈ patch; one seed, 25 passages, peak over 4 checkpoints). First add per-passage logging to
  `scripts/12_relearn.py` (it saves only aggregates). Then attack seeds 1 and 2 of full + 200 on retain-only, UNDO
  book, `book_npofull_plain_lr3e-5`, `mnpo_book`, book RMU-l4, `surg_l2_book_b25`, `surg_l2_book_b50`, plus the
  random-mask models (not yet re-attacked). Bootstrap CIs over items for all audit numbers (generations are already
  saved per item in `results/audit/<model>/generations/`, no GPU).

- [x] **5. Stronger related-data attack on the calculus jam.** Done 2026-11-12 (`scripts/sweeps/r5_*.txt`; result in the
  entry of that date): calculus does not come back. Our related-retain control (LoRA, 20 other-math items,
  100 steps) never restores calc RMU c20, but Łucki et al. 2024 (10 retain-set samples), Doshi & Stickland 2024
  (OpenWebText, 100–1,000 steps) and Wang et al. 2025 (downstream fine-tuning incl. math) restore RMU/NPO
  (`docs/RELATED_WORK.md` §2). Before claiming a contrast, attack with full-weight fine-tuning for about 500–1,000 steps
  on (a) other-math selfgen solutions and (b) generic web text (FineWeb). `scripts/12_relearn.py` already has `--full`;
  needs a longer step schedule and a FineWeb item source. Models: calc RMU-l4 c20, `surg_l2_calc_b25`, target
  (reference). If calculus comes back, the "related data does nothing" line goes and we agree with prior work.

Optional if 1–5 finish early: element-sparse surgery session (H6 like-for-like with Takehi et al.), matched-dose
depth ladder (H3 magnitude), linear probes (PLAN 6.5). Skip: Target C, SCRUB, new methods, BFCL.

### Write-up deliverables (PLAN Phase 8)

- [ ] **Leaderboard.** `docs/AUDIT.md` §4 leaves relearning out of worst recovery, and its book column is
  one-question noise (21 knowmem questions; `surg_l2_book_b50` "2.00" is one question). Fold relearning recovery in
  (PLAN 6.6 counts attack 10); report forget_rel, worst recovery and utility_rel separately, with CIs from item 4.
- [ ] **Plots (PLAN 8.1).** Relearning curves (main figure); forget vs utility with relearning recovery as marker
  size; depth-ladder heatmap; J-lens heatmaps (base vs best method vs prompt baseline); surgery result vs call
  budget. H5 bars exist (`results/localize/*/mass.png`).
- [ ] **Certificates (PLAN 7.7).** `results/surgery/*/certificate.md` still say "draft, dev metrics only": add test
  metrics and worst-case recovery including relearning.
- [ ] **Hypothesis verdicts H1–H6 (PLAN 8.2).** Draft from current evidence, to revise after items 1–4:
  - H1 (suppress, not erase): supported by relearning, and since 2026-11-12 without any training: one projection per
    edited layer, using the open base weights, restores 72% of the target's calculus from RMU and 92% from the agent's
    patch (derivative-only relearning also restores integrals on the patches). "Base lens still recovers the tokens" not
    supported (the lens misses the calculus jam). "RMU holds up better than NPO" refuted for calculus: RMU is undone
    training-free, NPO is not; for the book mixed (better than LoRA NPO, 31% vs 79% back under LoRA; worse than all-weight
    NPO under full + 200, 53% vs 24%).
  - H2 (skills harder than facts): partly. Calculus leaks through prompt-side routes (tool 79–100% of the target)
    where the book does not, and comes back training-free for RMU and the patch (the book test is uninformative there:
    the projection removes the memorised text from the target too); both come back under relearning.
  - H3 (collateral grows with depth): supported in direction; magnitude untested (forgetting not matched).
  - H4 (localized beats random): refuted on calculus. Mixed on the book: localized NPO beats random at equal lr
    (3 seeds), random at 2× lr closes most of the gap. Agent-chosen masks no better than random.
  - H5 (where content concentrates): calculus diffuse; book mildly concentrated in MLPs after global attention. Weak.
  - H6 (surgery at ≤ 0.1% of weights): met under the guard's low-rank count (0.0017%), not by changed bf16 elements
    (2.6–7.1%). State both, or run the element-sparse session.
- [ ] **`REPORT.md` + PDF (PLAN 8.3):** setup, results, attacks, limitations.
- [ ] **Related work (`docs/RELATED_WORK.md`, folded into `REPORT.md`).** No transformer replication: too much differs
  between model families to attribute any difference to architecture. Instead cite published transformer results,
  comparing the direction of an effect only (never their numbers in our tables, never "LFM3 is more robust than
  transformers"); where we differ (e.g. related-data fine-tuning does not restore our RMU jam, Łucki et al. 2024
  restore RMU with unrelated data), list the setup differences first and leave architecture as an open question. Every
  reference checked by web search (title, authors, venue, and that it says what we cite it for). Check the novelty
  claim ("almost all unlearning studies use pure transformers") against work on SSM / hybrid / gated-conv models.
- [ ] **Demo (PLAN 8.4):** base vs unlearned answer, J-lens heatmap, one live attack, certificate.
- [ ] **Reproducibility (PLAN 8.5):** configs for every reported result, `data/MANIFEST.json`, AuditKIT fingerprints,
  seeds.
- [ ] **Slides.**
- [ ] **Housekeeping:** commit the Phase 7 work (`src/lfm3u/surgery/`, `scripts/16_surgery.py`, `docs/SURGERY*.md`,
  sweep files, modified scripts) before launching new runs. Update PLAN §11 (the Nov 12–13 Phase 7 rows are done).

## 2026-11-11: Paper experiments (after Nov 15)

For an arXiv/workshop paper, separate from the hackathon TODOs above. TODOs 1–5 make the current claims defensible;
these aim at what is new after the literature search (`docs/RELATED_WORK.md`): one-way prerequisite damage, recovering a
skill through code, and the J-lens miss on the calculus jam. Proposed framing: "skill unlearning is suppression", with
calculus as the spine, agent surgery as a section, the book as the fact-unlearning comparison, and the hybrid
architecture as the setting. Nothing below has started. If only three are done: P1, P3, P4 (each tests a specific
2025–2026 paper where our data already hints at a different answer).

- [ ] **P1. Depth ladder → dependency test (lead result).** PreUnlearn (Su et al. 2026) says collateral damage
  follows semantic distance, which is symmetric; our ladder (RESULTS §7.5) suggests it follows dependency, which runs
  one way.
  - Full forget-X × measure-Y matrix at matched forgetting (a dose sweep per rung so each X drops by the same amount),
    at least NPO and RMU, 3 seeds, test split, larger per-topic n than the 90 used so far.
  - Pairs that separate similarity from dependency: derivatives ↔ integrals (dependent and similar), similar-looking
    but independent pairs, dependent but different-looking pairs.
  - Predict collateral from (a) semantic distance (PreUnlearn-style embedding distance) and (b) overlap of localizer
    attribution between skills (`src/lfm3u/unlearn/localize.py`). Claim if (b) predicts and (a) does not.
  - Cost: about 1 day of engineering; the largest GPU item (about 100 short runs, roughly $50).
- [ ] **P2. Second skill: linear algebra.** "Skills" is currently one skill. Linear algebra has a ladder (arithmetic →
  matrix multiplication → determinants and inverses → eigenvalues), sympy grading and a tool route (numpy). Reuse the
  calculus pipeline (`src/lfm3u/data/math_gen.py`, selfgen, unlearning, audit, relearning). Feeds P1 and P3.
  Cost: 1–2 days of engineering, small GPU.
- [ ] **P3. Tool route: "knowing what" vs "knowing how".** With `sympy.diff(...)` the library does the calculus; the
  model only has to recognize the operation. The tool loophole (79–100% of the target) suggests unlearning removes the
  procedure and keeps the knowledge of what to compute.
  - Word problems with a hidden calculus step, solved with and without a tool; score setting up the problem (the
    correct expression and operation) separately from carrying it out.
  - Then put sympy-calculus code in the forget set and test whether the tool route closes.
  - ATU (Chen et al. 2026) covers tools that look up facts; this is the skill version with a mechanism.
  - Cost: 1–2 days of engineering, small GPU.
- [ ] **P4. Can any activation-based audit catch a jam?** Calculus RMU's layers 5–23 are identical to the target's: the
  skill is in the weights but never reaches the activations, so activation-based audits should miss it.
  - Run Song et al. 2026's J-Access (reimplement from the paper), linear probes (PLAN 6.5, never run) and UDS-style
    activation patching (Lee et al. 2026) on all ~20 audited checkpoints; correlate each with relearning recovery.
  - Expected claim: jam-type methods (RMU, masked RMU, agent patches, UNDO calculus) are false negatives for every
    activation-based audit; only weight-level attacks catch them. Tests "J-Access predicts recovery" directly.
  - Cost: 1–2 days of engineering, small GPU.
- [ ] **P5. Agent surgery with a relearning check in its toolkit.** Add a dev-split relearning probe to the agent's
  tools (`src/lfm3u/surgery/tools.py`) so it can aim for edits that survive relearning; evaluate on the strong attacks
  it never saw (full + 200, cross-subskill). Still failing = a strong negative; succeeding = a method. Also: repeated
  sessions per budget (agent-to-agent variance), a second editor model, and an element-sparse budget for a like-for-like
  comparison with Takehi et al. Cost: 2–3 days of agent sessions plus API.
- [ ] **P6. Give the architecture angle content.**
  - Component-restricted relearning: the LoRA attack on conv blocks only, attention only, or MLPs only (change the
    LoRA target lists in `scripts/12_relearn.py`); same restriction for unlearning. Which component type holds the
    suppression and which undoes it? Not done for unlearning; Borobia et al. 2026 found attention-only LoRA best in
    other hybrids. Cost: hours.
  - Calculus pipeline on 1–2 more models, a transformer (Qwen3-1.7B) and a different hybrid (Qwen3.5), to show P1, P3
    and P4 hold beyond LFM3, not to attribute differences to architecture. Cost: 1–2 days of porting (LoRA targets,
    RMU `param_substring`, thinking toggle).

Do not invest further in the book target: its knowledge half has no gap left to remove (target fact 0.19 vs base
0.05, n = 21) and its verbatim result repeats Hu et al. 2025. Keep it as the fact-unlearning comparison.

## 2026-11-12: TODO 1 and 2 done (cross-subskill relearning, training-free recovery); public-base attack; scorecard

Worked autonomously 03:00–09:00 EST while the user was away (user's choice of #1 and #2; the follow-ups below were chosen
here, with the advisor model as a second opinion). All TEST splits, one seed unless noted.

- **Runs.** 43 single-GPU boxes on Vast (RTX PRO 6000 class, $1.3–1.8/h; includes 3 duplicates destroyed within minutes,
  one box that died and was relaunched, and the TODO 3 box still running), all but the last with
  `scripts/vast/jobs/attack_models.sh`; run ids in `results/vast/gap_runs*.json`, specs in
  `scripts/sweeps/{xs,xsm,xs1,la,lb,lc,r5,rep}_*.txt`. Results merged into `results/relearn/{xs,xsm,xs1,r5}_*` and
  `results/local_abl/la_*`. $40.6 billed by 07:35 EST (charges lag; the TODO 3 box adds about $4 more).
- **Code.**
  - `scripts/12_relearn.py`: `--kinds forget_<kind>` (`forget_diff` = 20 derivative-only self-generated dev solutions),
    `--calc-per-kind N` (N/3 test problems per level and kind from `data/calculus`), `--gen-bs`.
  - `scripts/11_abliterate.py`: `eval_calc` adds per-kind accuracy and per-item correctness bits; `eval_book` adds
    per-passage F1 (for paired tests and bootstrap CIs).
  - `scripts/17_local_abl.py` (new): direction ablation at chosen layers and sites (residual stream, MLP write after the
    sandwich norm, w2 output before it, operator write), proj or clamp, prompt- or solution-token directions, a random
    control, and a reference model either as an oracle (the private target) or as a realistic attacker (`--ref-label
    base`, the open LiquidAI/LFM3-1.7B weights). Primary variant declared before results: residual stream at every
    edited layer.
  - `scripts/13_audit_report.py`: AUDIT §5b (training-free) and §6b (cross-subskill), McNemar p on paired items.
  - `scripts/18_scorecard.py` (new): `docs/SCORECARD.md` and `docs/figures/scorecard.png`, one row per method, one column
    per threat model, bootstrap CIs; plus a paired cross-subskill table at a fixed step (25).
  - `scripts/19_demo_ablation.py` (new): `docs/DEMO_ABLATION.md` (PLAN 8.4 "one live attack").
  - `src/lfm3u/grade.py`: `parse_math` returns None instead of raising when sympy overflows its recursion on a deeply
    nested answer (killed the first masked-RMU run after three variants). Only answers that used to crash change.
- **TODO 1, cross-subskill relearning** (81 problems per kind; diff / integral / defint accuracy at step 25; mixed = the
  usual 20 calculus items, which hold 8 derivatives, 6 integrals and 6 definite integrals):

  | model | step 0 | derivative-only | mixed |
  |---|---|---|---|
  | target | 0.99 / 0.75 / 0.58 | 0.94 / 0.78 / 0.63 | not run |
  | patch `surg_l2_calc_b25` | 0.10 / 0.04 / 0.21 | 0.96 / 0.43 / 0.30 (seed 1: 0.96 / 0.44 / 0.37) | 0.96 / 0.62 / 0.58 |
  | patch `surg_l2_calc_b50` | 0.05 / 0.06 / 0.05 | 0.85 / 0.32 / 0.32 | 0.72 / 0.47 / 0.44 |
  | RMU-l4 c20 | 0 / 0 / 0 | 0.94 / 0.00 / 0.02 (seed 1: 0.91 / 0.00 / 0.02) | 0.91 / 0.65 / 0.59 |
  | UNDO (RMU teacher) | 0 / 0 / 0 | 0.77 / 0.00 / 0.01 (seed 1: 0.90 / 0.00 / 0.00) | 0.68 / 0.52 / 0.47 |
  | calc NPO | 0.91 / 0.17 / 0.33 | 0.96 / 0.25 / 0.38 | not run |

  - Derivative-only training teaches the target nothing about integrals (0.75 → 0.70–0.78 over the curve).
  - Patches: integrals come back without any integral example. On b25, paired over the integrals the target solves, the
    derivative-only gain is 0.69 [0.53, 0.86] of the mixed-data gain (two seeds agree; b50 0.66 [0.45, 0.89]). That part of relearning is
    unmasking, not teaching. The rest is not shown to be missing: after derivative-only training 49% of integral answers
    hit the 3,072-token cap (17% before), so they still loop rather than finish wrong.
  - RMU and its UNDO student: integrals stay at exactly 0.00 while derivatives recover, and half the integral answers
    still hit the cap. Six integral examples (mixed) restore them to 0.52–0.65 in 25 steps. Recovery does not transfer
    across subskills here, which by itself cannot tell hidden from re-taught; for RMU the training-free result below
    settles it (integrals 0.31 with no gradient step), for UNDO nothing does.
  - NPO had forgotten integrals, not derivatives (0.91 derivatives at step 0); derivative training adds little (0.17 → 0.28).
- **TODO 2, training-free recovery** (the 100 attack-8 problems, 3,072-token cap; target 0.71; target control = the same
  procedure on the target at the same layers):
  - Agent patch `surg_l2_calc_b25` (three rank-one w2 writes, L6–8): 0.12 → **0.60** with the w2-output projection (best
    of 7 attack variants; derivatives 0.03 → 0.92), **0.65** with the public-base reference; random direction 0.09; retain
    math 0.85 → 0.875–0.90; target control at L6–8 0.71–0.74. The pre-declared primary (residual stream) gives only 0.19
    (McNemar p = 0.27 against no ablation). The attack direction has |cos| 0.997 with the top singular vector of W2 − W2_target (s1/s2 ≈ 4,000: the
    patch is rank one and the attacker finds its write direction).
  - Trained RMU-l4 c20 (w2 of L2–4): the forget-vs-contrast direction removes the gibberish (0.91 → 0.00) but not the
    forgetting (best 0.06). The oracle (attacked minus target means, residual stream) restores 0.50; the **public-base
    reference restores 0.51** (derivatives 0.76, integrals 0.31, defint 0.46; retain math 0.875). The same base-reference
    projection costs the target itself 0.71 → 0.56, so RMU gets back 91% of what the ablated target keeps. The
    forget-vs-contrast estimate is close to the jam direction at L4 (cos 0.99) but not at L2–3 (0.61–0.90).
  - Masked RMU (`mrmu_calc_l4`, units in L0–4 incl. conv and o_proj columns): primary 0.03 (gibberish 0.89 → 0.05), best
    attack variant 0.14 (operator clamp, k = 9), public base 0.19, oracle 0.19, random 0.00. Partly a switch at most.
  - Removal-style patch `surg_l2_calc_b50` (20 directions projected out of w2, L8–15): w2 clamp 0.33 (best of 7), base
    reference 0.01, everything else ≤ 0.10 or collapses retain math.
  - Distributed methods: calc NPO has nothing to remove (best 0.46–0.49 vs 0.50 unablated at L2–4 and L8–15, oracle
    included); UNDO calc stays at 0.00 with every variant at both windows, oracle and base reference included.
  - Book: at most 0.29 verbatim F1 (closure 0.18, b25 w2 clamp); RMU-l4 0.22 with the oracle. Uninformative rather than
    negative: on the target the same w2 ablations remove the memorised text (0.94 → 0.27–0.35; base reference 0.22–0.63),
    so on the book this direction carries content and the attack has no ceiling.
- **TODO 5, stronger related-data attack** (launched 05:53, done 06:50; all weights, fp32 AdamW lr 1e-5, batch 4, 1,000
  steps, on 200 self-generated retain-math dev solutions or 1,000 FineWeb anchor documents; 100 calculus test problems
  at steps 0/100/250/500/1,000):
  - RMU c20: 0.00 at every step with either data; gibberish falls (0.96 → 0.29–0.31) but accuracy stays at 0.00–0.01.
  - Patch b25: 0.12 → at most 0.19 (related, step 100), then down to 0.09 (related) and 0.02 (web).
  - Target control: web text leaves calculus intact (0.72–0.74), so that attacker had a full ceiling; the related
    fine-tune itself erodes calculus (0.73 → 0.66 / 0.57 / 0.51 / 0.39), but at steps 100–250 the target still has
    0.57–0.66 while RMU has 0.00.
  - So "related data does not restore the jam" now holds under a ~10× stronger attacker (all weights, 1,000 steps, 10×
    the data), in contrast to Łucki et al. 2024 and Doshi & Stickland 2024 on transformers; state the setup differences
    first (model, RMU layer and coefficient, data, schedule) and leave architecture open (RELATED_WORK §2). The same
    model comes back with no training at all through the base-reference projection, so it is hidden, just not reachable
    by fine-tuning on other data.
- **Replication of the public-base attack** (06:57–07:34; `scripts/sweeps/rep_{a,b}.txt`, `17_local_abl.py --prompt-seed 1
  --calc-per-kind 81`: new direction prompts, scored on the 243 cross-subskill test problems): RMU 0.00 → 0.43 (target
  0.79; target under the same projection 0.56), patch 0.12 → 0.60 with the w2 site (target 0.77; same projection 0.67).
  Retain math 0.825–0.875. The RMU effect is smaller than on the attack-8 problems (54% vs 72% of the target) but clearly
  there.
- **Reading.** Three kinds of forgetting under one direct-prompt score: (1) a switch that one projection per edited layer
  turns off, given the open base model (RMU, the agent's patch; 72–92% of the target back, no training); (2) per-subskill
  suppression that relearning undoes only with examples of that subskill (RMU, UNDO); (3) no single direction to remove
  (NPO, UNDO). Only UNDO survives every training-free attack and the cross-subskill test, and it still relearns from 20
  mixed examples (0.68 at 100 steps), at a large utility cost (§7.2).
- **Ops.** Tigris served stale listings and stale GETs of overwritten objects (STATUS, run.log) for minutes; results were
  fetched by key and checked for completeness (`.empryo/tmp/refetch.py`, not committed). Five launches lost an offer race
  and were relaunched; the first `la_calc_mrmu` box died mid-run with STATUS DONE and no results; its relaunch crashed in the grader after
  three variants (fixed, see Code) and a third run (`la_calc_mrmu2.txt`) completed. HF
  generate logs ~10 "CUDACachingAllocator OOM" warnings per batch on 96 GB cards; they are allocator retries (the same
  count appears in older runs) and did not change results. Three duplicate boxes launched to work around a suspected
  slowdown were destroyed within minutes (the slowdown was stale logs). The `xs_target` box logged its self-destroy but kept
  running for ~35 min idle until destroyed by hand (`vastai show instances` is worth a check after every batch).

## 2026-11-12: Erasure follow-ups launched (docs/ERASURE_RESEARCH.md experiments 1 and 3, user-approved)

- **Research note.** `docs/ERASURE_RESEARCH.md` (background agent, Claude Opus 5.5): no LLM paper demonstrates topic erasure
  in the strong sense (indistinguishable from never-trained under all attacks); the closest results avoid post-hoc
  removal (pretraining filtering, gradient routing) or distil into a damaged student (UNDO). User approved its
  experiments 1 (attack UNDO harder) and 3 (calculus-free distillation without an unlearned teacher).
- **Code.** `scripts/20_weight_attacks.py` (new: simulated INT8/INT4 round-to-nearest and Gaussian weight noise, no training);
  `scripts/12_relearn.py` `--schedule cosine --warmup N` and `--forget-n N` (calc forget curves on N solutions);
  `scripts/15_undo.py --exclude-calculus` (drops calculus self-generated answers and FineWeb documents matching
  `CALC_RE`); `scripts/vast/jobs/undo_filtered_calc.sh`.
- **Launched 09:37–09:39 EST, 13 boxes** (run ids in `results/vast/erasure_runs.json`):
  - weight attacks: `wa_calc_{a,b}`, `wa_book` (UNDO, RMU, NPO, target, retain-only);
  - strong book relearning grid, after Qi et al. 2024: `rg_{undo_book_rmu,retain}_{a,b}` (lr 1e-5 / 3e-5 / 1e-4 ×
    constant / 20-step warmup + cosine × seeds 0–2), `rg_npofull`;
  - strong calculus relearning (all weights, all 133 / 68 dev solutions, per-kind eval): `cs_undo_forget`,
    `cs_undo_diff`, `cs_rmu_diff`;
  - projection attack at all 24 layers (public base and oracle) for UNDO calc and book, plus target controls: `pa_undo`;
  - experiment 3: `undo_filtered_calc` (teacher = target, `--exclude-calculus`, 5 h distillation, then dev eval, LoRA
    relearning and cross-subskill curves; results expected around 15:30 EST).

## 2026-11-12: TODO 3 and erasure experiment 1 results (UNDO's book robustness is a distillation effect)

- **TODO 3, matched-mixture UNDO book control** (`script-20261112-045204-undo_ctrl_book-3bd2`, done 09:58 EST; same recipe and
  book mixture as `undo_book_rmu`, teacher = the target, not unlearned; 148M tokens):
  - TEST verbatim F1 0.168 (retain-only 0.145, UNDO-RMU 0.127, target 0.944). Strong attack (all weights + 200 forget-text
    chunks) peaks at 0.171 (3% of the gap; UNDO-RMU 0.177, 4%). LoRA relearning flat (0.155–0.177).
  - DEV: passages whose text is in the distillation corpus stay memorised (verbmem seen 0.636; target 0.873), held-out
    passages are gone (unseen 0.163; target 0.894). Utility identical to UNDO-RMU (MMLU-val 0.445 vs 0.44, IFEval 0.61
    vs 0.59, FineWeb ppl 19.7).
  - **Reading:** distilling the target into a damaged copy on a corpus that lacks the test passages erases those
    passages, with or without an unlearned teacher. UNDO-book's resistance to relearning (RESULTS finding 8, §7.2–7.3) comes
    from what the repair corpus leaves out, not from the unlearning step, and its utility cost comes from the distillation
    budget. The test passages were never in any distillation corpus, so this is data filtering at the repair stage.
- **Experiment 1, falsify UNDO harder** (13 boxes from 09:37; `results/{weight_attacks,relearn,local_abl}/`):
  - Weight attacks (`20_weight_attacks.py`, no training): INT8, INT4 and Gaussian weight noise (0.01–0.05 std) recover
    nothing for any unlearned model (UNDO calc/book, RMU calc/book, NPO calc, all-weight NPO book: all within ±0.01 of
    unattacked, or lower). Controls: target calculus survives all variants (0.68–0.75); target book memorisation survives
    INT8 and noise but INT4 wipes it (0.944 → 0.203).
  - Strong book relearning grid after Qi et al. 2024 (lr 1e-5 / 3e-5 / 1e-4 × constant / warmup + cosine × seeds 0–2, peak
    verbatim F1 over steps 25–200): UNDO-RMU 0.131–0.182 (16 configurations), retain-only 0.126–0.160, all-weight NPO
    0.339–0.355 at lr 1e-5 (≤ 0.148 at 1e-4). lr 1e-4 destroys memorisation in every model. UNDO stays within ~0.03 of the
    never-trained floor everywhere; the same holds for the no-unlearning control above, so this is not evidence for the
    unlearning step.
  - Strong calculus cross-subskill attack on RMU c20 (all weights, all 68 derivative-only dev solutions, 200 steps, 243
    problems): derivatives 0.96–1.00, integrals 0.00 at every step, definite integrals ≤ 0.04. The UNDO runs (`cs_undo_*`)
    ran out of memory on an 80 GB card (full fp32 fine-tune, batch 4 of ~3k-token solutions); `12_relearn.py` now uses
    gradient checkpointing for `--full` and computes the loss only at labelled positions (checked equal to the old loss);
    relaunched 12:24 EST. Results (done 14:10 EST): UNDO with all 68 derivative-only solutions: derivatives 0.85–0.94,
    integrals and definite integrals 0.00 at every step to 200; with all 133 mixed solutions integrals 0.70–0.73 and
    definite integrals 0.51–0.54 by step 50 (target 0.75 / 0.58). Same pattern as RMU.
- **Experiment 3, calculus-free distillation from the target** (`script-20261112-093933-undo_filtered_calc-3c07`, done
  ~15:40 EST; 157M tokens, 12,855 FineWeb documents dropped by `CALC_RE`): DEV calculus 0.475 (target 0.754, unfiltered
  distillation control 0.775), thinking 0.533, truncated 0.35, gibberish 0.025; retain math 0.944, MMLU-val 0.465,
  IFEval 0.62. TEST 0.39 (target 0.71). LoRA on 20 calculus items: 0.66 at 10 steps, 0.73 at 50 (target level); on 20
  other-math items: 0.52 at 100 steps. Derivative-only (243 problems): integrals 0.32 → 0.53 (mixed 0.65).
  **Reading:** repair-data filtering halves a pretrained skill and erases fine-tuned text (book control above); it does
  not erase calculus. The skill regrows from calculus examples within 10 steps and partly from other math alone.
  Closes ERASURE_RESEARCH experiment 3; experiment 2 (targeted UNDO) not run.
  - Projection at all 24 layers: uninformative. The same projection collapses the target (book 0.944 → 0.09–0.14, calculus
    w2 0.00, retain math 0.00).
- **Ops.** `wa_book` logged its self-destroy at 09:47 EST but billed until destroyed by hand at 12:03 (second time today;
  `launch.py reap` after each batch).

