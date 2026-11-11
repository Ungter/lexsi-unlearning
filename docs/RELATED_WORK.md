# Related work (draft for `REPORT.md`)

Compiled 2026-11-11 from three web-search passes (citation check, skill-unlearning search, architecture search;
about 235 tool calls in total).
Status tags: **[V]** checked against the arXiv/venue abstract page (and full text where noted); **[V\*]** also opened
by hand during compilation; **[S]** seen only in search snippets, do not cite until checked. Results, claims and
corrections below refer to `docs/RESULTS.md` findings.

**Rules for citing.** Compare the direction of an effect, never their numbers in our tables, and never claim "LFM3 is
more or less robust than transformers": the models, targets, metrics and attack settings differ. Where we disagree
with prior work, list the setup differences first and leave architecture as an open question.

## 1. Unlearning hides rather than removes (findings 1, 6, 8, 11)

Agrees with a large body of transformer results. Our addition is separating *prompt-side* routes from *fine-tuning*
routes for a fact target and a skill target on the same model.

- **Hu, Fu, Wu, Smith.** "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning."
  ICLR 2025, arXiv 2406.13356 **[V]**. Loosely related public text brings back unlearned knowledge; Harry Potter wiki
  text brings back verbatim HP recall (WHP, Llama-2-7b); also WMDP (zephyr-7b-beta, Llama-3-8B) and TOFU. For TOFU/WHP
  keywords the relearn set is a disjoint subset of the forget set, as in our protocol. (Earlier arXiv titles: "Jogging
  the Memory of Unlearned Model/LLMs Through Targeted Relearning Attack"; cite the ICLR title.)
- **Deeb, Roger.** "Do Unlearning Methods Remove Information from Language Model Weights?" arXiv 2410.08827, preprint
  **[V]**. RTT: fine-tune on some forgotten facts, test on held-out facts from the same distribution; recovers 88% of
  pre-unlearning accuracy (Llama-3-8B; RMU zephyr checkpoint in the appendix). Our relearning design follows the same
  idea (train on forget-dev, test on unseen test passages/problems). They also warn that knowledge added by
  fine-tuning can make unlearning look more robust than it is: relevant to our fine-tuned book target.
- **Łucki, Wei, Huang, Henderson, Tramèr, Rando.** "An Adversarial Perspective on Machine Unlearning for AI Safety."
  TMLR 2025, arXiv 2409.18025 **[V]**. RMU (zephyr-7b-beta, WMDP) is undone by fine-tuning on 10 retain-set samples or
  by removing directions in activation space. (Their abstract says "unrelated examples"; the examples are retain-set
  samples.)
- **Lynch, Guo, Ewart, Casper, Hadfield-Menell.** "Eight Methods to Evaluate Robust Unlearning in LLMs." arXiv
  2402.16835 **[V]**. WHP (Llama-2-7b-chat) keeps HP knowledge under in-context relearning, other languages,
  jailbreaks and probing; fine-tuning on about 8,000 sentences brings it close to (not equal to) the original.
- **Che, Casper, Kirk, et al.** "Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities." TMLR,
  arXiv 2502.05209 **[V]**. Unlearning undone within 16 fine-tuning steps; tampering success empirically bounds
  input-space attack success (Llama-3-8B-Instruct, WMDP-Bio, 8 methods). They frame tampering as a complement to
  input-space evaluation, not a replacement.
- **Doshi, Cooper Stickland.** "Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods."
  arXiv 2411.12103, preprint **[V]**. LLMU and RMU (Zephyr-7B-β, Llama-3-8B-Instruct): 5-shot prompting or simple
  rephrasing raises accuracy more than ten-fold; training on unrelated data (OpenWebText, about 100–1,000 steps)
  almost fully recovers.
- **Xu et al.** "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs." ICML 2026,
  arXiv 2505.16831 **[V]**. Models appear to forget but minimal fine-tuning restores them; four regimes of
  reversibility vs damage; includes a math domain (NuminaMath) per the paper text **[S for the math detail]**.
- **Qian, Suriyakumar, Wilson, Hadfield-Menell.** "Layered Unlearning for Adversarial Relearning." arXiv 2505.09500
  **[V]**. Unlearning builds a shallow, context-dependent inhibition, so relearning on one subset brings back the whole
  forget set.
- **Feng, Xu, Robey, Kirk, Davies, Gal, Schwarzschild, Kolter.** "Existing LLM Unlearning Evaluations Are
  Inconclusive." arXiv 2506.00688 **[V]**. Many evaluations inject new information; proposes minimal information
  injection.
- **Schwinn, Dobre, Xhonneux, Gidel, Günnemann.** "Soft Prompt Threats." NeurIPS 2024, arXiv 2402.09063 **[V]**.
  Embedding-space attacks extract "deleted" information (Llama2-7b-WhoIsHarryPotter, TOFU). We ran no embedding-space
  attack.

## 2. Contrast: related-data fine-tuning does not restore our calculus jam

Our result (RESULTS §5 finding 2; §6): LoRA on 20 *related* retain items (other math), up to 100 steps, leaves
calculus RMU c20 at 0.00, while the same steps on 20 calculus items restore it. Three papers find the opposite
direction for RMU/NPO:

- Łucki et al. (above): 10 retain-set samples recover RMU (full fine-tuning, WMDP).
- Doshi & Stickland (above): OpenWebText, about 100–1,000 steps, recovers LLMU/RMU.
- **Wang et al.** "Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning." ICML 2025,
  arXiv 2506.01339 **[V]**. Downstream fine-tuning, including math, restores NPO/RMU forgetting on WMDP and MUSE;
  proposes ILU to resist it.

Setup differences to state before anything else: LoRA r16 vs full fine-tuning; 100 steps vs up to 1,000; a skill
target vs hazardous-knowledge facts; a 1.7B model; our RMU jam is gibberish-producing (degenerate by our flags). A
stronger related-data attack (full weights, several hundred steps, other math and generic web text) would show whether
the contrast survives; see the TODO in `docs/DECISIONS.md`.

## 3. Skill unlearning (findings 2, 9, 10)

Skill unlearning itself is not new; a full attack audit of a skill (relearning with controls, tool route, prerequisite
ladder) on one model appears to be. "To our knowledge" wording only.

- **Li, Sun, Weng.** "Effective Skill Unlearning through Intervention and Abstention." NAACL 2025, arXiv 2503.21730
  **[V]**. Training-free: Neuron Adjust (FFN activation intervention) and Key Space Detection (abstain when a query
  lands in the skill's key-space cluster); math (GSM8K), Python (MBPP), 7 languages; >80% relative drop on the target
  skill. Best method is a gate, not removal; no relearning or tool tests. **Differs** from our framing: they report
  clean removal by inference-time gating.
- **Lee, Foote, Infanger, …, Cloud, Turner.** "Distillation Robustifies Unlearning" (UNDO). NeurIPS 2025 spotlight,
  arXiv 2506.06278 **[V]**. Includes an arithmetic setting (forget ×/÷, retain +/−, ~300M Gemma-2-style model) where
  GradDiff, MaxEnt and RMU suppress rather than remove, with an oracle pretrained without ×/÷ as the "never had it"
  baseline. **Closest prior skill-unlearning-plus-relearning work.** On WMDP (Gemma-2-2B) UNDO costs 4.88% MMLU on
  average: agrees with our UNDO utility cost.
- **Zhou, Jiang, Wu, Zhou.** "Mechanism-Guided Selective Unlearning for RLVR-Induced Reasoning" (MAST). arXiv
  2606.19222, Jun 2026 **[V\*]**. On Qwen3-1.7B-Base, full-parameter unlearning collapses GSM8K; updating selected
  attention tensors gives a small, significant MATH forget drop (45/150 → 37/150). **Agrees**: gradient methods
  collapse a ~1.7B model before a math subdomain is forgotten cleanly.
- **Christ, Gottesman, Kropko, Hartvigsen.** "Math Neurosurgery" (MathNeuro). ACL 2025, arXiv 2410.16930 **[V]**.
  Pruning math-specific parameters removes GSM8K/MATH ability with general language largely intact; no robustness
  tests. **Differs in emphasis** from our diffuse calculus attribution (RESULTS §6).
- **Pochinkov, Schoots.** "Selective Pruning." arXiv 2403.01267 **[V]**. Removes coding ability by pruning neurons.
- **Jain, Kirk, Lubana, …, Krueger.** "Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined
  Tasks." arXiv 2311.12786 **[V]** (ICLR 2024 venue not yet checked). Fine-tuning adds a thin wrapper over existing
  capabilities that a few steps revive: a frame for the jam.

## 4. Collateral damage and the depth ladder (finding 10, H3)

Nothing found on prerequisite ladders or on directional damage (forgetting a prerequisite hurts dependents, not the
reverse). Queries tried are listed in the search report; "to our knowledge" wording only.

- **Su, Shah, Le.** "PreUnlearn: Auditing Collateral Knowledge Damage Before LLM Unlearning." arXiv 2606.18473
  **[V]**. Collateral damage is strongest near the forget set and decays with semantic distance. That distance is
  symmetric, so it cannot produce our directional asymmetry.
- **Peleg, Singh, Pearl, Mohapatra, Hein.** "Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric
  Generalization Problem." arXiv 2607.09236 **[V]**. "The retain set effectively defines the forget set." **Agrees**
  with our finding that NPO cannot remove arithmetic while its dependents are retained.
- **Thaker, Hu, Kale, Maurya, Wu, Smith.** "Position: LLM Unlearning Benchmarks are Weak Measures of Progress."
  SaTML 2025, arXiv 2410.02879 **[V]**. Loose forget/retain dependencies break benchmarks.
- **Wu, Yadav, Salakhutdinov, Chaudhuri.** "Evaluating Deep Unlearning in LLMs." arXiv 2410.15153 **[V]**. Retained
  facts plus rules re-derive "forgotten" facts; analogous to calculus rebuilt from algebra plus sympy.
- **Jiang, Dong, …, Li.** "LLM Unlearning for Source Code" (PROD). AAAI 2026, arXiv 2506.17125 **[V]**. Code
  unlearning damages basic programming-language knowledge.

## 5. Tool and form routes (findings 2, 6, 9, 11: tool loophole, notation)

Tool-mediated recovery of *facts* is published; recovery of a *skill* by writing code was not found.

- **Chen, Liu, Zhang, Ding, Ma, Huang, Jiang.** "Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for
  LLM Agents" (ATU). arXiv 2608.21544, Aug 2026 **[V\*]**. Names "tool-mediated recovery": an agent retrieves the
  forget target through web search, retrieval or database lookup (RWKU, MUSE). **Agrees** in direction; theirs is fact
  retrieval, ours is re-running a skill in sympy.
- **Yu, Jiang, Wang, Ma, Wang.** "K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments." arXiv 2609.12808
  **[V]**. Across six ReAct-agent channels, none of 20 methods removes a secret stored in the weights.
- **O'Brien, Casper, …, Gal, Biderman.** "Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant
  Safeguards." arXiv 2508.06601 **[V]**. Filtered-pretraining models resist ~10k adversarial fine-tuning steps but
  still use dangerous information given in context or by tools.
- **Ye, Zhang, Wu.** "LLM Unlearning Should Be Form-Independent." arXiv 2506.07795 **[V]**. Unlearning holds only for
  the trained phrasings: matches our notation-change leak (calc NPO 88% of the target).
- **Baser et al.** "The Tokens Remember" (Toketive). arXiv 2609.29045 **[V]**. Alternative tokenizations bypass edits
  and unlearning.

## 6. Robust-unlearning methods we ran or should compare against

- **Fan, Jia, Zhang, Ramakrishna, Hong, Liu.** "Towards LLM Unlearning Resilient to Relearning Attacks: A
  Sharpness-Aware Minimization Perspective and Beyond." ICML 2025, arXiv 2502.05374 **[V]**. NPO+SAM most robust on
  WMDP and MUSE (Books and News) under relearning on 20–60 forget samples. **Differs** from our book result (SAM adds no
  resistance at lr 3e-5, RESULTS §7.1). Their attack also uses forget samples; differences: model, learning rates, our
  LoRA vs their attack setup, one seed.
- **Lee et al.** UNDO (§3 above). **Agrees** on robustness with a utility cost; our matched book control is pending.
- **Guo, Syed, Sheshadri, Ewart, Dziugaite.** "Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via
  Mechanistic Localization." ICML 2025, arXiv 2410.12949 **[V]**. Localizing to fact-lookup MLPs is more robust than
  output-tracing localization (relearning recovers up to ~63% vs ~20%; Gemma-7B, Gemma-2-9B, Llama-3-8B; Sports Facts,
  CounterFact); output-tracing localization often does no better than editing the whole model. **Agrees** with our
  attribution-patching masks not helping calculus (ours is output-side attribution).
- **Tamirisa et al.** "Tamper-Resistant Safeguards for Open-Weight LLMs" (TAR). ICLR 2025, arXiv 2408.00761 **[V]**.
  Not run.
- **Wang et al.** ILU (§2 above). Not run.

## 7. Internal readouts as unlearning evidence (findings 3, 4; RESULTS §5 finding 6)

- **Patil, Hase, Bansal.** "Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against
  Extraction Attacks." ICLR 2024, arXiv 2309.17410 **[V]**. Logit-lens-style whitebox attacks recover ROME/MEMIT-deleted
  facts 38% of the time (GPT-J-6B).
- **Hong, Yu, Yang, Ravfogel, Geva.** "Intrinsic Test of Unlearning Using Parametric Knowledge Traces." EMNLP 2025,
  arXiv 2406.11614 **[V]**. Unlearning methods barely change concept vectors and mostly suppress them at inference;
  ablating the vectors removes the knowledge (LLaMA2-7B-chat, OLMo-7B). (Earlier title: "Intrinsic Evaluation of
  Unlearning Using Parametric Knowledge Traces".)
- **Song, Liu, Wang, et al.** "Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning." arXiv 2608.11408, Aug
  2026 **[V\*]**. **Uses the Jacobian lens** (J-Access) on 398 public unlearned models from eight methods: most retain
  access above the retain-only gold level; pre-attack access predicts recovery speed and extent at the model level
  but not which facts return; minimizing J-Access teaches models to hide knowledge from the audit. **Must cite**: the
  closest overlap with our J-lens tooling. Our calculus RMU jam is a model-level case where readout access is at the
  control level (0.15 vs 0.14) yet relearning restores the skill fully in 25 steps: a counterexample to access
  predicting recovery, for a skill and a different readout metric (intermediate-token hit@10), so state it as a
  contrast with caveats.
- **Lee, Kim, Jo.** "Measuring the Depth of LLM Unlearning via Activation Patching" (UDS). EMNLP 2026, arXiv
  2605.24614 **[V]**. Activation patching against a retain-model baseline; most faithful of 20 metrics on 150 models.
- Lynch et al. (§1) also recover answers with linear probes; our probes (PLAN 6.5) were not run.

## 8. Relearning controls ("did fine-tuning just teach it?")

- Deeb & Roger (§1) rely on facts being independent, which fails for a skill: hence our planned derivatives-only →
  integrals control (TODO 1 in `docs/DECISIONS.md`).
- UNDO's never-trained oracle (§3) and the retain-only gold model under the identical attack (Song et al., §7) are
  the templates; our book retain-only control (RESULTS §7.3) follows the latter.
- **Greenblatt, Roger, Krasheninnikov, Krueger.** Password-locked models, arXiv 2405.19550 **[V]** (NeurIPS 2024 venue
  not yet checked). Fine-tuning unlocks capabilities even across different passwords.
- **Taylor, Black, …, Marks, Cundy, Bloom.** "Auditing Games for Sandbagging." arXiv 2512.07810 **[V]**. One correct
  demonstration unlocks sandbagging models but sometimes also raises benign models: elicitation can give false
  positives, which is why controls matter.

## 9. Agentic weight editing and the J-lens

- **Takehi, Teoh, Zhang, Khattab.** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network
  weights." Blog post, 2 Nov 2026, https://rikiyat.github.io/blog/weight-space-programming **[V]**. The editor ("GPT-6
  Astra") changed 131,072 of Qwen3.5-0.8B's weights (0.016%); with a budget of 50 target-model forward passes (each up
  to 512 problems) interleaved with edits, MATH-500 rose from 7.9% to 48.1% (12.3% and 14.1% at 10 and 25 passes).
- **Gurnee, Sofroniew, Pearce, et al. (… Lindsey).** "Verbalizable Representations Form a Global Workspace in
  Language Models." arXiv 2607.15495, Jul 2026 **[V\*]**; Transformer Circuits,
  https://transformer-circuits.pub/2026/workspace/index.html; code github.com/anthropics/jacobian-lens. Introduces the Jacobian lens and the J-space, on Claude models. Correct
  wording for our use: one-step recall is largely J-space-independent, while multi-hop reasoning depends on it
  (closed-book trivia recall *is* affected by J-space ablation, so do not say "fact recall bypasses the J-space").
  Applying the lens to a hybrid conv/attention model is our extension.

## 10. Architecture: unlearning and localization beyond pure transformers

**Novelty check.** No published study was found that unlearns a non-transformer language model and analyses its
architecture-specific parts, and nothing at all on unlearning or knowledge editing in gated-convolution or LFM models.
Queries covered unlearning/editing/concept erasure × Mamba, SSM, RWKV, Hyena/StripedHyena, xLSTM, RetNet, linear
attention, Jamba, Zamba, Falcon-H1, Nemotron-H, Granite, Qwen3.5, LFM/LFM2/LFM3 (arXiv API, web, Hugging Face papers,
OpenReview; Semantic Scholar rate-limited, Google Scholar not queried). Two late-2026 papers do run unlearning on
hybrid models but treat them as transformers:

- **Ronchetti.** "Orthogonal Boundary Unlearning for Dual-Use Concepts in Large Language Models." arXiv 2610.27490, Oct
  2026 **[V\*]**. Projected LoRA and closed-form weight surgery on Gemma-3-4B, Qwen3.5-4B (a Gated DeltaNet hybrid),
  SmolLM3-3B and Granite-4.2-3B, with a relearning attack; architecture not discussed. Nearest incidental hybrid case;
  also a second closed-form weight-surgery unlearning method next to our agent patches.
- **Jin et al.** "Relocating Hazardous Knowledge into Removable Experts in MoE Models." arXiv 2610.25791 **[V, listing
  only]**. Uses Qwen3.5-35B-A3B (hybrid MoE).

Suggested wording for the report: "To our knowledge, unlearning studies have almost exclusively evaluated pure
transformers; the few that include hybrid models treat them as transformers, and we know of no unlearning study on a
gated-convolution hybrid (or any LFM model) or one that localizes forgetting across mixer types."

**Editing and localization in SSMs and hybrids.**

- **Sen Sharma, Atkinson, Bau.** "Locating and Editing Factual Associations in Mamba." COLM 2024, arXiv 2404.03646
  **[V]**. Causal tracing in Mamba-2.8B (vs Pythia-2.8B): factual recall in mid-layer components at the last subject
  token; rank-one ROME edits to `out_proj` insert facts. Closest locate-and-edit prior (Mamba has no MLPs; LFM3 has an
  MLP in every layer).
- **Endy et al.** "Mamba Knockout for Unraveling Factual Information Flow." ACL 2025, arXiv 2505.24244 **[V]**.
- **Nishi et al.** "Representation Shattering in Transformers." ICML 2025, arXiv 2410.17194 **[V]**. Edits distort
  related entities; corroborated on Mamba.
- **Michalak, Abreu.** "Some Attention is All You Need for Retrieval." arXiv 2510.19861 **[V]**. In RecurrentGemma and
  Jamba-Mini, in-context retrieval depends entirely on attention; SSM layers do not compensate.
- **Zeng, Feng, Chen.** "Who Asks, Who Answers: Linear Attention Builds the Query in Hybrid LMs." arXiv 2611.01909
  **[V]**. In Qwen3.5 and Granite 4.0, linear-attention/Mamba-2 blocks build the query that softmax attention reads.
- **Bick, Xing, Gu.** "Gather-and-Aggregate." arXiv 2504.18574 **[V]**. Hybrids hand aggregation to attention.
- **Borobia et al.** Component ablation (arXiv 2603.22473) and component-targeted LoRA (arXiv 2604.22127) on
  Qwen3.5-0.8B and Falcon-H1-0.5B **[V]**: adapting only the attention pathway beats full LoRA. Relevant if we ever
  run component-targeted relearning attacks.
- **Arora et al.** Zoology, ICLR 2024, arXiv 2312.04927 **[V]**: gated-convolution models trail attention on
  associative recall.
- No causal tracing or neuron attribution was found in a hybrid that has MLPs, and no comparison of MLPs after mixer
  layers vs after attention layers: our H5 analysis (RESULTS §6) has no direct prior, though its result is weak.

**Interpretability tools on non-transformers** (our J-lens and directional ablation on a hybrid).

- **Wang, Reid.** "Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models." arXiv
  2609.32102, Sep 2026 **[V\*]**. Jacobian lenses on Mamba-1/2/3 residual streams and recurrent states; on Mamba-2 the
  recurrent state alone beats residual and logit lenses. Closest prior for a J-lens on a non-transformer. Implication
  for us: our lens reads only the residual stream; LFM3's conv blocks have a short (kernel 7) state, so a residual-only
  lens could miss content carried there. A caveat for our J-lens null results, not a demonstrated cause.
- **Bosco, Srinivasan.** "Locating and Steering Refusal Beyond Attention." arXiv 2609.04721, Sep 2026 **[V]**.
  Refusal-direction transfer and ablation across Falcon-Mamba, Codestral-Mamba, RWKV-6, Zamba2 (hybrid), Llama, Mistral;
  the direction must be read at each layer's write site (mixer output before the residual add). Relevant to attack 8
  and the layer-local ablation planned in TODO 2 (`scripts/11_abliterate.py` projects the residual stream after each
  layer).
- **Paulo, Marshall, Belrose.** "Does Transformer Interpretability Transfer to RNNs?" arXiv 2404.05971 **[V]**.
  Steering, tuned lens and latent-knowledge probes mostly transfer to Mamba and RWKV.
- Also: Ali, Zimerman, Wolf, "The Hidden Attention of Mamba Models," ACL 2025, arXiv 2403.01590 **[V]**; Sunku Mohan
  et al., arXiv 2602.22719 **[V]** (steering in Mamba-family models).

**Related "gone or hidden" results on a conv hybrid, outside text unlearning.** Black et al., arXiv 2511.19299, and Wei
et al., "BioRiskEval," arXiv 2510.27629 **[V]**: fine-tuning Evo 2 (a StripedHyena-2 conv/attention hybrid; architecture
from background knowledge, not the abstracts) restores viral capabilities removed by data filtering. Data filtering,
DNA models.

**Weight edits on LFM models.** Lazaridis et al., "When Can One Neuron Fix Repetition Loops in LLMs?", arXiv 2606.13705
**[V]**: sparse neuron edits on Gemma 4, with exploratory runs on LFM2.5 and Qwen3.5. The only LFM weight edit found.

**Architecture sources to cite.**

- **Amini et al. (Liquid AI).** "LFM2 Technical Report." arXiv 2511.23404, Nov 2025 **[V]**. Defines the gated
  short-conv block: (B, C, h̃) = Linear(h); y = B ⊙ h̃; z = Conv_k(y) (depthwise); o = Linear_out(C ⊙ z); k = 3 in LFM2.
- **Liquid AI.** "Introducing LFM3: Every Modality for Specialized Intelligence." Liquid AI Blog, 27 Oct 2026, and the
  LiquidAI/LFM3-1.7B model card **[V]**. 24 layers: 13 gated short convs (kernel 7), 5 sliding-window attention
  (window 256), 6 global attention; sigmoid-gated attention, sandwich norms, 22T tokens. No LFM3 tech report on arXiv;
  the model card's BibTeX key is `liquidAI2026LFM3`. SwiGLU and tied embeddings are not stated there: cite `config.json`.
- Optional lineage: STAR (Thomas et al., arXiv 2411.17800, source of "LIV operators"), H3 (Fu et al., ICLR 2023, arXiv
  2212.14052), Hyena (Poli et al., arXiv 2302.10866), Mechanistic Design and Scaling of Hybrid Architectures (Poli et
  al., arXiv 2403.17844) **[V for IDs]**.

## 11. Benchmarks and methods used

- **Shi et al.** "MUSE: Machine Unlearning Six-Way Evaluation for Language Models." ICLR 2025, arXiv 2407.06460
  **[V]**. **The Books target is ICLM-7B** fine-tuned on Harry Potter; News uses LLaMA-2 7B. Our target/retain-only
  fine-tune follows the same protocol on LFM3.
- **Li, Pan, Gopal, et al.** "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning." ICML 2024,
  arXiv 2403.03218 **[V]**. Introduces RMU (Zephyr-7B, Yi-34B-Chat, Mixtral-8x7B).
- **Zhang, Lin, Bai, Mei.** "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning."
  arXiv 2404.05868 **[V]** (COLM 2024 venue unconfirmed: cite arXiv unless checked).
- **Eldan, Russinovich.** "Who's Harry Potter? Approximate Unlearning in LLMs." arXiv 2310.02238 **[V]**.
- **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda.** "Refusal in Language Models Is Mediated by a Single
  Direction." NeurIPS 2024, arXiv 2406.11717 **[V]**. Directional ablation (our attack 8).
- **Maini et al.** TOFU (2024): in the proposal; not re-checked.

## 12. Corrections to our own wording

- The proposal's J-lens reference title ("A global workspace in language models") and the claim "simple fact recall
  bypasses the J-space" need the corrections in §9.
- Do not say MUSE used Llama-2 for Books (§11).
- Łucki et al.'s "unrelated examples" are retain-set samples (§1).
- Our related-retain result contrasts with three papers (§2); it must not be cited as agreeing with Łucki et al.
- Novelty: skill unlearning is published (§3); claim only the combination (full attack audit of a skill, tool route
  by code, directional prerequisite damage), with "to our knowledge". Replace "almost all unlearning studies use pure
  transformers" with the wording in §10, and cite Ronchetti 2026 as the nearest incidental hybrid case.
- Our J-lens use is not the first J-lens unlearning audit (Song et al. 2026, §7) nor the first J-lens on a
  non-transformer (Wang & Reid 2026, §10). It is, to our knowledge, the first on a gated-convolution hybrid and the
  first on a skill target.
- Not yet checked: NPO's COLM venue; Jain et al.'s ICLR venue; Greenblatt et al.'s NeurIPS venue; the LessWrong post
  "Unlearning via RMU is mostly shallow" (2024, snippet only).
