# Erasure, not suppression: what exists beyond what we ran (research note)

Compiled 2026-11-12 to answer: can a topic's knowledge be *removed* from the weights rather than suppressed, did anything
come close, and could methods be chained to get there? Literature already in `docs/RELATED_WORK.md` (RW) is cited by
section, not repeated. Our numbers: `docs/RESULTS.md` (R:line), `docs/SCORECARD.md` (SC:line), `docs/DECISIONS.md` (D:line).

Tags: **[V]** abstract or landing page opened in this pass (title, authors, year, venue where stated, and the cited
claim); **[V+]** the cited section of the full text or blog post was read in this pass; **[RW]** verified in
RELATED_WORK; **[UNVERIFIED]** not checked. All numbers are the authors' own unless marked *independent*. No numbers
were estimated.

## 1. What would count as erasure here

> **Erasure claim (project definition).** A model has erased target T if no attack we can run recovers more of T than
> the same attack recovers from a model that never learned T.
>
> 1. **Relearning parity.** Under every fine-tuning attacker (LoRA on 20 forget-dev items; all weights + 200 forget
>    chunks; several seeds; constant and warmup+cosine schedules; a learning-rate grid) the recovery curve stays inside
>    the never-trained control's CI. Book control = retain-only model, which gains nothing under the strong attack
>    (0.148 → 0.149, R:410). Today only UNDO-book is close (peak 4%, R:411).
> 2. **Weight-level / training-free parity.** Direction projection (incl. the public-base reference,
>    `scripts/17_local_abl.py`), quantization, weight-noise injection and embedding-space attacks recover nothing beyond
>    the control. Attacks that *copy* weights from the public base are out of scope: they import calculus rather than
>    reveal it. (Corollary: against an attacker with the base, a sparse edit can always be undone by copying the edited
>    entries back, so sparsity cannot be the mechanism of erasure for calculus.)
> 3. **No cross-subskill unmasking (skills).** Training on one subskill (derivatives) must not raise another (integrals)
>    above what the same data teaches the target (0.75 → 0.70–0.78, R:625-626). Calculus has no never-trained control,
>    so (1) cannot be run for it; (3) is the substitute. Recovery that needs examples of each subskill (RMU, UNDO,
>    R:631-634) still cannot separate "re-taught" from "unmasked" without a calibration of how fast a genuinely new skill
>    of similar difficulty is learned from 20 examples.
> 4. Prompt-side and tool routes, and internal readouts at control level, are necessary but not sufficient: activation
>    audits can miss jams whose weights never reach the activations (D:763-768).
>
> **Erasure can only be falsified.** The attack space is open-ended: Qi et al. (2412.07097) **[V+]** turned published
> "durable" TAR and RepNoise results into failures by changing only seeds, the trainer implementation, warmup + cosine
> decay, or the prompt template. Thudi et al. (USENIX Security 2022, 2110.11891) **[V]** conclude unlearning is
> well-defined only at the level of the algorithm that was run, not by inspecting the resulting weights. Cooper et al.
> (NeurIPS 2025 oral, 2412.06966) **[V]** separate removing information from suppressing outputs. Passing (1)–(3)
> means "not falsified by these attacks", nothing more.
>
> **Formal guarantees come only from exact or certified unlearning.** Exact = retrain without the data (SISA, Bourtoule
> et al., IEEE S&P 2021, 1912.03817 **[V]**). Certified = output (ε,δ)-indistinguishable from retraining (Guo et al.,
> ICML 2020, 1911.03030 **[V]**, convex/linear; Rewind-to-Delete, Mu & Klabjan, NeurIPS 2025, 2409.09778 **[V]**,
> nonconvex). Both certify the influence of *training data*, not the absence of a *capability*: a skill that can be
> re-derived from retained data (RW §4, Wu et al.) can survive exact deletion of every calculus document. For the book
> they apply directly: **the retain-only model is the exact unlearning of the book** (a 33–54 min fine-tune, D:63).

## 2. Approaches beyond what we ran

Per item: mechanism; strongest evidence; what broke it; fit to (a) fine-tuned text, (b) pretrained skill, and the hybrid.

**2.1 Exact and certified unlearning.** *Mechanism:* organise training so deletion means retraining a small part.
SISA shards data and retrains one constituent; S3T (Chowdhury et al., ICLR 2025, 2406.16257 **[V]**) trains PEFT layers
on disjoint slices and deactivates affected layers; SIFT-Masks (Kuo et al., 2504.04626 **[V]**, preprint) fine-tunes one
sign-constrained masked task vector per source and re-merges without the deleted one (up to 500 models; up to 250× less
compute for exact unlearning than other merging baselines). Certified methods add calibrated noise after a Newton step
(Guo et al.) or after rewind + retain-only descent (R2D); Fisher forgetting (Golatkar et al., CVPR 2020, 1911.04933
**[V]**) scrubs with Fisher-shaped noise and bounds residual information; Sekhari et al. (2103.03279 **[V]**) bound
deletion capacity for convex losses. *Evidence:* exact = retrained distribution by construction. *Breaks:* cost;
certificates need convexity/PL-type assumptions or utility-destroying noise at LLM scale; LLM second-order/Fisher
heuristics (SOUL, EMNLP 2024, 2404.18239 **[V]**; SSD, AAAI 2024, 2308.07707 **[V]**) carry no certificate and no
relearning evidence was found. *Fit:* (a) yes, if the fine-tune is organised for deletion (book in its own adapter or task
vector, or simply retrain retain-only). (b) no: LFM3's 22T-token pretraining (RW §10) is not ours. Architecture-neutral.

**2.2 Rewind and retrain.** *Mechanism:* return to a checkpoint from before the data and continue on filtered data; R2D
makes this certified for nonconvex losses; Amnesiac ML (Graves et al., 2010.10981 **[V]**) instead stores and subtracts
the updates of batches that held the data. UNDO App. F **[V+]**: for knowledge added after pretraining, distil the
unlearned teacher onto the original base model; on TOFU (Gemma-2-2B, Phi-1.5) those students relearn close to the
never-trained control. *Fit:* (a) for the book the pre-book checkpoint is the public base, so rewind + retrain on retain
data *is* our retain-only model. (b) no intermediate LFM3 pretraining checkpoints are public to our knowledge
**[UNVERIFIED]**.

**2.3 Pretraining-data filtering (gold standard).** *Evidence:* Deep Ignorance (O'Brien et al., 2508.06601 **[V+]**,
RW §5): 6.9B models pretrained on biothreat-filtered data resist up to 10,000 steps / 300M tokens of adversarial
fine-tuning, over an order of magnitude better than post-training baselines; they improve under attack but stay well
below the unfiltered baseline; filtering cost <1% extra training FLOPs (project page); the blocklist-only filter removed
8.42% of documents. Token-level filtering (Rathi & Radford, 2601.21571 **[V]**) beats document filtering, improves with
scale, and gives a 7000× compute slowdown on the forget domain at their largest size. Anthropic (Chen et al., alignment
blog, 19 Aug 2025 **[V+]**): WMDP 33.7% → 30.8% (chance 25%) at one threshold with no significant MMLU/code/prose loss;
no tamper test. *Breaks:* filtered models still use the knowledge given in context or by tools (Deep Ignorance; SGTM's
limitations); fine-tuning restored filtered viral capabilities in Evo 2, a conv hybrid (RW §10). *Fit:* (a) = the
retain-only model, cheap here. (b) the only clean calculus control, but a full 22T-token run. Architecture-neutral.

**2.4 Localization by design (gradient routing family).** *Mechanism:* during training, data labelled "forget" may
update only designated parameters (heads, MLP neurons, added modules), which are zeroed afterwards. *Evidence:*
gradient routing (Cloud et al., 2410.04332 **[V]**) introduces ERA (expand, route, ablate) for robust unlearning by
ablation. SGTM (Shilov et al., Anthropic Fellows, 2512.05648 **[V+]**): 254M model on Wikipedia, biology = 3.7% of
tokens; better retain/forget trade-off than weak or strict filtering under label noise; adversarial fine-tuning on a
50/50 bio/general mix recovers RMU in 50 steps (13M tokens) but SGTM only after 350 (92M), matching strict filtering;
5% compute penalty. GRAM (Roland et al., ICML 2026, 2607.08077 **[V+]**): auxiliary MLP neurons per dual-use domain;
Chinchilla-optimal 50M–5B runs; ablation approximates filtered models and resists fine-tuning recovery like filtering
(MaxEnt recovers to near baseline); one run gives five capability profiles (5× cheaper than filtering); LoRA on a
filtered base does about as well. NULLs (Ghosal et al., 2606.13873 **[V]**): per-source sparse "sinks"; deleting one
Wikipedia article ≈ retraining; a Harry Potter case study resists extraction and the relearning that reverses post-hoc
unlearning. MemSinks (Ghosal et al., ICML 2025, 2507.09937 **[V]**) isolates memorised sequences in sequence-specific
neurons. *Breaks:* no independent attack found; evidence is loss-based, ≤5B, dense transformers, from the authors'
groups; the forget set must be fixed before training; in-context use survives. *Fit:* (a) **directly applicable**: our
book target is our own ~1 h fine-tune (`scripts/03_finetune.py`). (b) needs (re)pretraining, or post-hoc relocation
(2.5). Hybrid: every LFM3 layer has an MLP and `localize.py` already treats conv `out_proj` channels as units; untested on
conv hybrids.

**2.5 Post-hoc relocation; MoE / modular deletion.** *Mechanism:* REX (Jin et al., 2610.25791 **[V+]**) adds experts to a
pretrained MoE, initialised from the experts the target domain uses more than general data (sources scaled down);
target tokens update only new experts, general tokens update only the originals with the new experts masked out of
top-k; routing losses; excision; then a light RMU on the remaining experts. Modular designs delete by construction:
DEMix ("experts can be removed to restrict access ... without additional training", 2108.05036 **[V]**), BTM
(2208.03306 **[V]**), FlexOlmo (2507.07024 **[V]**). *Evidence:* Qwen3.5-35B-A3B Bio: excision −48.4 WMDP points with
MMLU change <3; after 250M tokens of standard SFT, REX 32.4% → 42.8% while RMU goes 44.1% → 78.3%. *Caveats:* the
post-training test is benign SFT with life-science data removed, not a forget-data relearning attack; relocation is
partial (Intern-S2 Bio: 79.4% full → 64.3% excised before cleanup; gradient masking alone 68.4%); the last step is RMU,
i.e. suppression; no independent replication. *Fit:* (b) the only published post-hoc route toward localization by design
for pretrained knowledge. LFM3-1.7B is dense (no router), so a dense analogue relies on gradient + forward masking only;
Liquid's LFM2-8B-A1B is an MoE (8.3B total, 1.5B active) if one is needed.

**2.6 Tamper resistance / meta-learned anti-relearning.** *Mechanism:* simulate fine-tuning attacks in an inner loop and
optimise the released weights so they fail: MLAC (Henderson et al., AIES 2023, 2211.14946 **[V]**, BERT-scale), SOPHON
(Deng et al., IEEE S&P 2024, 2404.12699 **[V]**, classification and generation models), TAR (RW §6), MUDMAN (Sondej et
al., NeurIPS 2025, 2506.12484 **[V]**: disruption masking + gradient normalisation + meta-learning, "outperforms TAR by
40%"). *Breaks (independent):* Qi et al. **[V+]** on the public TAR-Bio-v2: rerunning the paper's own attack configs with
5 seeds reaches >60% WMDP-Bio, close to the undefended Llama-3-8B-Instruct; 100-step warmup + cosine decay recovers it
consistently (TAR seems to rely on exploding gradients in the first steps); LoRA largely breaks it; fine-tuning on the
retain set also recovers; TAR-v2 garbles benign biology and GSM8K falls to near zero. Che et al. (RW §1) include TAR,
RepNoise and ELM and undo state-of-the-art unlearning within 16 steps. UNDO's authors could not tune TAR to keep retain
performance. No attack on MUDMAN or SOPHON found **[UNVERIFIED gap]**. *Fit:* a hardened suppression, not removal;
inner-loop cost; architecture-neutral.

**2.7 RepNoise, Vaccine, Booster.** RepNoise (Rosati et al., NeurIPS 2024, 2405.14577 **[V]**) pushes harmful
representations toward noise at every layer and attributes its effect to "depth" of removal. Vaccine (Huang et al.,
NeurIPS 2024, 2402.01109 **[V]**) and Booster (Huang et al., 2409.01586 **[V]**) harden *refusal alignment* against
harmful fine-tuning-as-a-service. *Breaks (independent):* Qi et al.: with data shuffling the best of 5 attack seeds
equals the undefended model at lr 6e-5 and 8e-5; the HF SFT trainer breaks it consistently; on disjoint attack/test
sets there is no defence. No independent attack on Vaccine/Booster found in this pass **[UNVERIFIED]**. *Fit:* not
erasure methods; at most relearning-slowing regularisers.

**2.8 Distillation-based robustification (UNDO and successors).** *Mechanism:* distil the unlearned teacher into a
damaged copy, θ' = (1−α)θ + αβN, or a random init. *Evidence* **[V+]**: random-init students of an oracle relearn like
the oracle; robustness rises with α (0.1–0.8 tested; arithmetic needs α ≥ 0.55; α = 0.6 needed ~35% / 20% of the
pretraining corpus to recover); at the strongest setting UNDO matches filtering with 60–80% of its compute (synthetic
tasks). On WMDP (Gemma-2-2B) α = 0.25, β = 0; larger α did not recover MMLU within 300M tokens (0.015% of pretraining,
their Table 13); −4.88 MMLU on average. INT4 quantization on TOFU: forget truth ratio 0.185 unlearn-only vs 0.017
distilled. *Ours (independent):* book strong attack 4% (R:411); calculus relearns 0.08 / 0.58 / 0.66 / 0.68 at 10 / 25
/ 50 / 100 LoRA steps (R:376; peak 0.96 of target, SC:16) but shows no training-free or derivative-only recovery
(R:650, R:631-634); most utility loss is under-distillation, since the no-unlearning control also loses 9.5 MMLU / 15
IFEval points (R:367-369). *Successors:* UNDIAL (2402.10052, cited by UNDO) is weaker; Lee's retrospective ("Bitter
Lessons from Distillation Robustifies Unlearning", LessWrong, 28 Nov 2025 **[V+]**): post-hoc robust unlearning "may
not be achievable at all, at least not efficiently"; enough damage is necessary, not sufficient. *Fit:* (a) works here;
(b) partial here, with global α capped by our distillation budget; architecture-neutral (we ran it on LFM3).

**2.9 Weight noise, shrinking, re-initialisation.** Shrink-and-perturb (Ash & Adams, NeurIPS 2020, 1910.08475 **[V]**)
was a warm-start trick; UNDO's β = 0 is pure shrinking. Without retraining, noise is an *attack*: Tice et al. (NeurIPS
2025, 2412.01784 **[V]**) show weight noise improves sandbagging models by disrupting the suppression. *Fit:* defence only
inside a damage + distil pipeline; add noise to our audit as a cheap training-free attack.

**2.10 Pruning / zeroing units; SAE-feature ablation.** Lo, Cohen, Barez (ACL Findings 2024, 2401.01814 **[V]**): pruned
concepts return within a few epochs of retraining, relocated to earlier layers and to semantically similar neurons.
MathNeuro and Selective Pruning have no robustness tests (RW §3). Farrell, Lau, Conmy (2410.19278 **[V]**): zero-ablating
SAE features is ineffective, negative scaling is needed, side effects ≥ RMU. DSG (Muhamed et al., COLM 2025, 2504.08192
**[V]**) claims stronger resistance to relearning, but it is an inference-time guardrail an open-weight attacker removes.
Zeroed locations leak: "Roots Beneath the Cut" (CVPR 2026, 2603.06640 **[V]**) revives pruned concepts in diffusion
models training-free. Ours: localized masks forget no better than random for calculus (R:35-39). *Fit:* suppression on
its own for (a) and (b); useful only as the first step of a re-init + distil chain (§3, C1). Hybrid: sandwich norms
rescale what remains after zeroing (R:663-665).

**2.11 Linear concept erasure.** LEACE (Belrose et al., 2306.03819 **[V]**; NeurIPS 2023 venue not checked): closed-form
projection that provably defeats every linear probe, applied at every layer as "concept scrubbing"; INLP (Ravfogel et
al., ACL 2020, 2004.07667 **[V]**). *Why it is not enough:* the guarantee is linear and site-local; kernelized erasure
stops the chosen nonlinear adversary but "does not transfer to different nonlinear adversaries" (Ravfogel et al., EMNLP
2022, 2201.12191 **[V]**); at inference it is not in the weights, and baked into the weights it is a projection, which is
exactly what our §7.7 attack inverts (RMU's jam is one direction per layer, R:61-67). ELM (Gandikota et al., 2410.02760
**[V]**) claims robustness and is among the methods Che et al. undo. *Fit:* neither memorised text nor a skill is a
linear concept; keep as a diagnostic. Residual-only erasure can miss content in conv state (RW §10).

**2.12 Model editing (ROME, MEMIT, AlphaEdit).** Rank-one / closed-form MLP updates (ROME, NeurIPS 2022, 2202.05262
**[V]**; MEMIT, 2210.07229 **[V]**, ICLR 2023 venue not checked; AlphaEdit, ICLR 2025, 2410.02355 **[V]**, null-space
constrained). *Breaks:* whitebox attacks recover 38% of ROME/MEMIT-deleted facts (RW §7, Patil et al.); Youssef et al.
(ICLR 2026, 2505.20819 **[V]**) infer the edited object from the weights alone (up to 99%) and reverse up to 94% of edits
without training. Ours: the agent's rank-one patch is the top singular vector of W2 − W2_target (|cos| 0.997) and is
switched off by projection (R:658-660). No relearning study of AlphaEdit-as-deletion found. *Fit:* facts only; neither
target is a set of triples.

**2.13 Task-arithmetic negation.** θ − λ(θ_ft − θ) (Ilharco et al., ICLR 2023, 2212.04089 **[V]**); NegMerge
(2410.05583, listing only). No LLM relearning study found; a VLA-policy audit (2608.04692 **[V]**) calls negation
"temporary behavioral suppression" and its relearning probe is consistent with masking. *Fit:* (a) exact negation is
trivial here (target − (target − retain-only) = retain-only); a book-only delta is suppression-like. (b) there is no
calculus task vector to subtract.

**2.14 Other 2024–2026 robustness claims (authors' claims; no independent replication found).** RepSelect (Sondej et
al., 2606.17168 **[V]**): collapse the top principal components of weight gradients before each update; 4–40× larger
drop in post-relearning answer probability than GradDiff, NPO, SimNPO, RMU and UNDIAL on 5 datasets, dense and MoE;
holds under an adaptive attack. Minor Component Unlearning (2605.11685 **[V]**): relearning reverses changes
along dominant representation components, so target minor ones. Targeted LAT (Sheshadri et al., 2407.15549 **[V]**):
unlearning more robust to re-learning. Labs: Anthropic (filtering, SGTM, GRAM above); OpenAI evaluates open-weight risk
by malicious fine-tuning and claims no removal (Wallace et al., 2508.03153 **[V]**); EleutherAI / UK AISI: Deep
Ignorance. No Google DeepMind, Apollo or Redwood post claiming durable LLM erasure was found in this pass.

## 3. Chains that could get closer

- **C1. Localize → re-initialise the located units → distil from the unlearned teacher ("targeted UNDO").** UNDO's
  robustness tracks damage α, but large global α does not recover utility on a small distillation budget (UNDO Table 13;
  our α = 0.25 control already loses 9.5 MMLU). Put α → 1 where calculus attribution is highest and shrink lightly
  elsewhere, so calculus structure is destroyed while distillation rebuilds only a small part. The pieces cover each
  other: pruning alone returns (2.10); distilling into a nearly intact student keeps latent traces (UNDO §3).
  *Cheapest test:* existing `localize.py` scores → new `--mask / --mask-frac / --mask-alpha` in `scripts/15_undo.py`
  (reuse `localize.select` and `column_masks`; perturb w2 input columns plus the matching w1/w3 rows, conv `out_proj`
  columns plus matching `in_proj` rows) → same-count random-unit control.
- **C2. RMU/NPO → UNDO → tamper-resistant fine-tune.** UNDO removes latent structure, meta-learning would harden the
  remainder. Not recommended now: the hardening step is what Qi et al. broke with schedule changes, inner loops are
  costly, TAR-v2 destroyed GSM8K, and our cheapest landscape-flattening analogue (SAM-NPO) added no resistance (R §7.1).
- **C3. Checkpoint rewind + retrain on filtered data (book only).** For the book this already exists: the retain-only
  model, the floor in every column (R:410). Report it as the exact-unlearning reference; the only open book question is
  whether a cheaper post-hoc pipeline matches it (UNDO nearly does, at a utility cost). Optional: distil the RMU-book
  teacher onto the public base (UNDO App. F) via a `--student-init` flag in `15_undo.py`; low priority (D:784-785).
- **C4. Gradient-routed fine-tune, then delete the routed subspace.** Book: route book tokens into ~1–2% designated (or
  added) MLP units in `scripts/03_finetune.py`, zero them after. Calculus: a dense REX analogue (copy the top-attributed
  MLP units, scale the sources down, calculus tokens update only the copies, general tokens update only the originals
  with copies masked in the forward pass, excise), then C1-style distillation instead of REX's RMU cleanup so no switch
  is left behind. *Cheapest test:* the book version with half the book chunks unlabelled (tests SGTM/GRAM "absorption"
  on a conv hybrid).

## 4. Ranked shortlist for ~2 days, ≤ $50

Prices $1.3–1.8 per RTX PRO 6000 hour (D:792); existing UNDO runs used a 5 h budget (R:358). Two days cannot establish
erasure; they can falsify UNDO or show a method that moves closer on the §1 tests.

| # | experiment | hypothesis / "closer to erasure" means | attacks | GPU, cost | main risk |
|---|---|---|---|---|---|
| 1 | **Falsify UNDO harder.** undo_book_rmu, undo_calc_rmu; references retain-only, book_rmu_l4, calc_rmu_l4_c20, book_npofull_plain_lr3e-5. New: INT8/INT4 (bitsandbytes) eval; weight-noise sweep; `12_relearn.py --full` with 3 seeds × {constant, 100-step warmup + cosine} × lr {1e-5, 3e-5, 1e-4} (needs a schedule flag); calculus strong attack (all weights + 200 self-generated calculus dev solutions, needs an item-count flag); `17_local_abl.py` at all layers for UNDO. | UNDO-book stays inside the retain-only CI in every new column; UNDO-calc shows no training-free gain. | the list itself | ~5–6 h, ~$10; ~0.5 day code | low; a failure is the result ("came close, no further") |
| 2 | **Targeted UNDO (C1), calculus.** Teacher calc_rmu_l4_c20; student α = 1, β = 1 on the top 2% calculus-attributed units across all layers, α = 0.1 elsewhere; random-unit control with the same counts; `--max-minutes 300` each. | LoRA mixed-data curve below UNDO's 0.58 / 0.66 / 0.68 (25 / 50 / 100 steps) and below the random control, at MMLU / IFEval ≥ UNDO's 0.435 / 0.61 (R:363). | LoRA curve, `--kinds forget_diff`, public-base projection, #1 quantization/noise, utility | 2 × 5 h + 2 × 1.5 h ≈ 13 h, ~$20 | diffuse attribution (finding 7) makes localized ≈ random; student copies the teacher's gibberish jam (UNDO-calc dev 92.9% gibberish, R:383) |
| 3 | **Damage + calculus-filtered distillation, no unlearned teacher.** Teacher = target; drop calculus selfgen sources and regex-filter FineWeb for calculus terms; student = #2's damaged student (or global α = 0.25). | If the skill is never demonstrated during repair, a damaged student does not rebuild it: calculus near 0 *without* gibberish, same attack picture as #2. The existing control kept 0.73 calculus (R:377) but its corpus contained calculus selfgen. | as #2 | ~6.5 h, ~$10; shares #2 code | calculus regrows from retained algebra and web math (entanglement, finding 10, R:48-51) |
| 4 | *(optional)* **Routed book fine-tune with partial labels (C4).** Route 50% of book chunks into designated units, leave 50% unlabelled, zero the units, run the book scorecard. | Floor in every column without UNDO's utility cost; unlabelled passages also gone (absorption works on a conv hybrid). | full book scorecard incl. full + 200 | ~1.5 h train + ~2 h attacks, ~$6; ~0.5 day code | book deprioritised (D:784-785); leakage into unrouted conv/attention weights |
| 5 | *(stretch, does not fit)* **Dense REX analogue for calculus (C4).** | Excised model below RMU under all attacks before any cleanup. | as #2 | ~1 day code + 3–5 h GPU | REX's own relocation is partial and only benign-SFT tested; no router in a dense model |

Will not fit: TAR/MUDMAN meta-learning; pretraining-scale filtering or SGTM/GRAM pretraining of a calculus-free LFM3;
a never-knew-calculus control model.

## 5. Bottom line

No paper demonstrates topic erasure in an LLM in the strong sense of §1 (indistinguishable from a never-trained model
under all attacks); every robustness claim means "not broken by the attacks tried", usually the authors' own, and the
one systematic re-evaluation (Qi et al.) broke the two leading "durable" defences with small attack changes. The
closest credible results avoid post-hoc removal altogether: pretraining filtering (Deep Ignorance: 10,000 steps / 300M
tokens; token filtering), localization by design (SGTM matches strict filtering under fine-tuning; GRAM tracks filtering
from 50M to 5B; NULLs on Harry Potter), and distillation into a fresh or heavily damaged student (UNDO: oracle-level
relearning on synthetic tasks, partial on WMDP). For knowledge already in pretrained weights the only post-hoc
candidates are UNDO, which on our calculus target removes training-free and cross-subskill recovery but still relearns
from 20 mixed examples, and REX-style relocation (MoE, benign-SFT evidence only). For our book target exact unlearning
already exists (the retain-only model) and UNDO approaches it; for calculus erasure cannot be measured cleanly without a
never-trained control, so the realistic goal is "no attack we can run distinguishes it from forgetting", and targeted
UNDO (#2, #3) is the cheapest push in that direction.
