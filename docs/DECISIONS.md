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
