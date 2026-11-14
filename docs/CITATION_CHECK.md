# Citation check for `docs/RELATED_WORK.md` (RW) and `docs/ERASURE_RESEARCH.md` (ER)

Checked 2026-11-13, about 21:45–23:00 New York time (EST). Sources: arXiv API metadata (title, authors, dates,
comments) for every arXiv id, arXiv HTML full text for the cited sections, venue pages (proceedings, ICLR/NeurIPS/ICML
virtual sites, ACL Anthology, COLM accepted-papers list), the Liquid AI blog, model card and `config.json`, the
Anthropic Alignment Science blog, LessWrong (GraphQL API for post dates), and the Takehi et al. blog. Line numbers
refer to the files after the corrections below. Quotes are verbatim and at most 30 words.

**Summary.** 126 references checked (70 in RW, 52 only in ER, 4 added to RW §10). None is fabricated: every title, id
and first author resolves. Problems found: 13 wrong or incomplete titles; 1 wrong venue (MUDMAN is a NeurIPS 2025
workshop paper, not main track) and 7 venues filled in; 2 author-name problems; 1 date given in UTC; 13 substantive
content errors or over-claims, of which the most important is that Łucki et al. attacked RMU with LoRA (rank 128),
not full fine-tuning, which RW §2 used as a setup difference. All were corrected in place.

## (a) Reference table

Supports: yes / partly / no / not found / n/a (cited without a specific claim).

### RELATED_WORK.md

| # | Ref | As cited | Verified title; authors; year; venue; link | Supports | Supporting quote or discrepancy | Action |
|---|---|---|---|---|---|---|
| 1 | Hu et al. | RW:18 | "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning"; Hu, Fu, Wu, Smith; 2024; ICLR 2025; arxiv.org/abs/2406.13356 | partly | "relearning general wiki information about the book series Harry Potter can force the model to output verbatim memorized text." That experiment uses Llama-2-7b-chat fine-tuned on an excerpt, not WHP. Earlier titles end "…Relearning Attacks" (plural). | Setting and earlier title fixed. |
| 2 | Deeb & Roger | RW:25 | "Do Unlearning Methods Remove Information from Language Model Weights?"; Deeb, Roger; 2024; arXiv preprint; 2410.08827 | yes | "fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy". Llama 3 8B main; zephyr RMU in App. F. Fine-tuned-knowledge evaluations "may overestimate robustness". | None. |
| 3 | Łucki et al. | RW:30, 68, 74 | "An Adversarial Perspective on Machine Unlearning for AI Safety"; Łucki, Wei, Huang, Henderson, Tramèr, Rando; 2024; TMLR; 2409.18025 | partly | "finetuning with only 10 samples from the retain set…can recover most of hazardous capabilities, obtaining accuracies of…61.6% (RMU)" (64.4% unprotected). Appendix: "Lora rank 128 Target modules all". Not full fine-tuning; "most", not "undone". | Fixed RW:30–34, 68, 74–76. |
| 4 | Lynch et al. | RW:35 | "Eight Methods to Evaluate Robust Unlearning in LLMs"; Lynch, Guo, Ewart, Casper, Hadfield-Menell; 2024; arXiv; 2402.16835 | partly | "fine-tuning on 8,000 sentences brings the WHP model's performance close to the original Llama-2 baseline." Other languages gave "Large Familiarity drops…for both WHP and Llama-2", so not evidence of retained knowledge. | Reworded. |
| 5 | Che et al. | RW:39 | "Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities"; Che, Casper, Kirk, …, Hadfield-Menell; 2025; TMLR; 2502.05209 | yes | "We benchmark 8 unlearning methods"; "state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning." Llama-3-8B-Instruct; methods include RepNoise, ELM, TAR. | None. |
| 6 | Doshi & Cooper Stickland | RW:43, 69 | "Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods"; Doshi, Cooper Stickland; 2024; arXiv; 2411.12103 | yes | "Although the results are from fine-tuning for 1000 steps, in most cases 100 steps are enough to completely recover performance." | RW:69 surname ("Stickland" → "Cooper Stickland"). |
| 7 | Xu et al. | RW:47 | "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs"; Xiaoyu Xu et al.; 2025; ICML 2026 (arXiv: "accepted to appear"); 2505.16831 | yes | "models can appear to forget while their original behavior is easily restored through minimal fine-tuning"; complex task "NuminaMath-1.5". | Removed the [S] tag. |
| 8 | Qian et al. | RW:50 | "Layered Unlearning for Adversarial Relearning"; Qian, Suriyakumar, Wilson, Hadfield-Menell; 2025; arXiv; 2505.09500 | partly | "Recent results suggest that post-training induces shallow context-dependent 'circuits'…To test this hypothesis, we design an unlearning algorithm". A hypothesis they test, stated in RW as their finding. | Reworded. |
| 9 | Feng et al. | RW:54 | "Existing Large Language Model Unlearning Evaluations Are Inconclusive"; Feng, Xu, Robey, Kirk, Davies, Gal, Schwarzschild, Kolter; 2025; arXiv; 2506.00688 | yes | "some evaluations introduce substantial new information into the model"; proposes "minimal information injection". Title was shortened to "LLM". | Title fixed. |
| 10 | Schwinn et al. | RW:57 | "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space"; Schwinn, Dobre, Xhonneux, Gidel, Günnemann; 2024; NeurIPS 2024; 2402.09063 | yes | "embedding space attacks can extract supposedly deleted information from unlearned LLMs"; uses Llama2-7b-WhoIsHarryPotter and TOFU. | Full title. |
| 11 | Wang et al. (ILU) | RW:70 | "Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning"; Changsheng Wang et al.; 2025; ICML 2025; 2506.01339 | yes | "highly sensitive to downstream fine-tuning, which can quickly recover forgotten information—even from unrelated tasks"; tasks include "math". | None. |
| 12 | Li, Sun, Weng | RW:85 | "Effective Skill Unlearning through Intervention and Abstention"; 2025; NAACL 2025; 2503.21730 | yes | "Key Space Detection achieves over 80% relative performance drop on the forgetting skill". | None. |
| 13 | Lee et al. (UNDO) | RW:90; ER:67–68, 135–147 | "Distillation Robustifies Unlearning"; Lee, Foote, Infanger, Shor, Kamath, Goldman-Wetzler, Woodworth, Cloud, Turner; 2025; NeurIPS 2025 Spotlight; 2506.06278 | yes | "the UNDO model achieves 4.88% lower on MMLU on average"; arithmetic on Gemma-2-0.3B (forget ×/÷); α ≥ 0.55; α = 0.6 needs "around 35% and 20% of their pretraining corpus"; WMDP "α=0.25 and β=0"; Table 13; App. F TOFU (Gemma-2-2B, Phi-1.5); INT4 truth ratio 0.185 vs 0.017; TAR: "unable to find hyperparameters that maintained reasonable retain performance". | None. |
| 14 | Zhou et al. (MAST) | RW:95 | "Mechanism-Guided Selective Unlearning for RLVR-Induced Reasoning"; Zhou, Jiang, Wu, Zhou; Jun 2026; arXiv; 2606.19222 | partly | "The primary base model is Qwen2.5-Math-1.5B"; the 45/150 → 37/150 result is on that model. On Qwen3: "MAST preserves GSM8K while full-parameter unlearning collapses it". RW attributed both to Qwen3-1.7B-Base. | Model attribution fixed; target named (RLVR reasoning). |
| 15 | Christ et al. (MathNeuro) | RW:101 | "Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes"; 2024; ACL 2025; 2410.16930 | yes | "we delete a LLM's math reasoning ability without significantly impacting its general language ability." | None. |
| 16 | Pochinkov & Schoots | RW:104 | "Dissecting Language Models: Machine Unlearning via Selective Pruning"; 2024; arXiv; 2403.01267 | yes | "We evaluate our method by selectively removing coding ability in LLMs." | Full title. |
| 17 | Jain et al. | RW:106 | "Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks"; Jain, Kirk, Lubana, …, Krueger; 2023; ICLR 2024; 2311.12786 | yes | "a minimal transformation, which we call a 'wrapper', is typically learned on top of the underlying model capabilities". | Venue added. |
| 18 | Su et al. (PreUnlearn) | RW:115 | "PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning"; Su, Shah, Le; 2026; arXiv; 2606.18473 | yes | "collateral damage is strongest near the forget set, weakens with semantic distance, but does not disappear at domain boundaries". The symmetry argument is ours. | Softened our inference. |
| 19 | Peleg et al. | RW:119 | "Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric Generalization Problem"; Peleg, Singh, Pearl, Mohapatra, Hein; 2026; arXiv; 2607.09236 | partly | Actual sentence: "The retain set thus defines the effective forget set". RW misquoted it; the link to our NPO result is our reading. | Quote fixed; "Agrees" softened. |
| 20 | Thaker et al. | RW:123 | "Position: LLM Unlearning Benchmarks are Weak Measures of Progress"; 2024; SaTML 2025; 2410.02879 | yes | "particularly vulnerable to modifications that introduce even loose dependencies between the forget and retain information." | None. |
| 21 | Wu et al. | RW:125 | "Evaluating Deep Unlearning in Large Language Models"; Wu, Yadav, Salakhutdinov, Chaudhuri; 2024; arXiv; 2410.15153 | yes | "prevent it from being deduced via retained knowledge in the LLM and logical reasoning." | Title fixed ("LLMs"). |
| 22 | Jiang et al. (PROD) | RW:128 | "Large Language Model Unlearning for Source Code"; Jiang, Dong, …, Ge Li; 2025; AAAI 2026; 2506.17125 | yes | "tends to spill over, damaging the basic knowledge of programming languages learned by the LLM". | Title fixed. |
| 23 | Chen et al. (ATU) | RW:136 | "Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for LLM Agents"; Chen, Liu, Zhang, Ding, Ma, Huang, Jiang; Aug 2026; arXiv; 2608.21544 | yes | "an agent can still recover the same forget target through tools such as web search, retrieval, or database lookup…tool-mediated recovery". | None. |
| 24 | Yu et al. (K-Bench) | RW:140 | "K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments"; Yu, Jiang, Wang, Ma, Wang; 2026; arXiv; 2609.12808 | partly | "none of the twenty evaluated published methods demonstrably removes it". RW said "removes". | Wording fixed. |
| 25 | O'Brien et al. (Deep Ignorance) | RW:143; ER:73–80 | "Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs"; O'Brien, Casper, …, Gal, Biderman; 2025; arXiv; 2508.06601 | yes | "up to 10,000 steps and 300M tokens"; "can still leverage such information when it is provided in context (e.g., via search tool augmentation)"; "less than 1% of total training FLOPS"; strong filter removes "8.42% of pretraining documents". | Full title. |
| 26 | Ye et al. | RW:146 | "LLM Unlearning Should Be Form-Independent"; Ye, Zhang, Wu; 2025; arXiv; 2506.07795 | yes | "frequently fails to generalize to alternate expressions of the same knowledge." | None. |
| 27 | Baser et al. (Toketive) | RW:148 | "The Tokens Remember: When Tokenization Bypasses Knowledge Editing and Unlearning"; Baser, Nawal, Divakaran, Gurusamy; 2026; arXiv; 2609.29045 | yes | "38.6% of alternative tokenizations bypass the modification and recover the pre-edit response." | Full title, number added. |
| 28 | Fan et al. | RW:154 | "Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond"; Fan, Jia, Zhang, Ramakrishna, Hong, Liu; 2025; ICML 2025; 2502.05374 | partly | WMDP: "20, 40, or 60 samples"; MUSE: "200, 300, and 400 samples for Books, and 400, 500, and 600 samples for News". Claim is that smoothing "consistently improve[s]" robustness, not that NPO+SAM is most robust. | Fixed. |
| 29 | Guo et al. | RW:161 | "Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization"; Guo, Syed, Sheshadri, Ewart, Dziugaite; 2024; ICML 2025 (spotlight); 2410.12949 | yes | "recover…as much as 63%…when using OT methods like Causal Tracing"; "FLU localization…only about 20%". Numbers are Sports Facts. | Context added. |
| 30 | Tamirisa et al. (TAR) | RW:167 | "Tamper-Resistant Safeguards for Open-Weight LLMs"; 2024; ICLR 2025; 2408.00761 | n/a | Not run. Later broken by Qi et al. (row 75). | None. |
| 31 | Patil et al. | RW:173 | "Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks"; Patil, Hase, Bansal; 2023; ICLR 2024 (spotlight); 2309.17410 | partly | "our whitebox and blackbox attacks can recover 'deleted' information from an edited model 38% of the time" (ROME, GPT-J). RW gave the figure for logit-lens whitebox attacks on ROME/MEMIT. | Reworded (also ER:175). |
| 32 | Hong et al. | RW:176 | "Intrinsic Test of Unlearning Using Parametric Knowledge Traces" (v1: "Intrinsic Evaluation of…"); Hong, Yu, Yang, Ravfogel, Geva; 2024; EMNLP 2025; 2406.11614 | yes | "minimally impact concept vectors and mostly suppress them during inference, while directly ablating these vectors demonstrably removes the associated knowledge"; LLaMA 2 7B chat, OLMo 7B. | None. |
| 33 | Song et al. | RW:180 | "Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning"; Song, Liu, Wang, …, Chen; Aug 2026; arXiv; 2608.11408 | yes | "most unlearned models retain access above the retain-only gold level"; "the model learns to hide knowledge from the audit". | None. |
| 34 | Lee, Kim, Jo (UDS) | RW:188 | "Measuring the Depth of LLM Unlearning via Activation Patching"; 2026; EMNLP 2026 Main (Oral); 2605.24614 | yes | "across 20 metrics on 150 unlearned models…UDS achieves the highest faithfulness and robustness". | None. |
| 35 | Greenblatt et al. | RW:198 | "Stress-Testing Capability Elicitation With Password-Locked Models"; Greenblatt, Roger, Krasheninnikov, Krueger; 2024; NeurIPS 2024; 2405.19550 | yes | "fine-tuning can elicit other capabilities that have been locked using the same password, or even different passwords." | Title and venue added. |
| 36 | Taylor et al. | RW:200 | "Auditing Games for Sandbagging"; Taylor, Black, …, Marks, Cundy, Bloom; 2025; arXiv; 2512.07810 | yes | "using only a single correct demonstration…However the performance of benign models was sometimes also raised". | None. |
| 37 | Takehi et al. (blog) | RW:206 | "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"; Takehi, Teoh, Zhang, Khattab (MIT CSAIL); blog, 2 Nov 2026; rikiyat.github.io/blog/weight-space-programming | yes | "the edit changes only 131,072 of the model's 0.8 billion weights (0.016%)"; MATH-500 7.9 → 12.3 / 14.1 / 48.1 at 10 / 25 / 50 passes; a pass answers "up to 512 problems". | None. |
| 38 | Gurnee et al. | RW:210 | "Verbalizable Representations Form a Global Workspace in Language Models"; Gurnee, Sofroniew, Pearce, …, Lindsey; Jul 2026; arXiv; 2607.15495 | yes | "J-space-independent (such as text continuation, anomaly detection, or one-step factual recall)"; ablation hurts "TriviaQA, multi-hop reasoning". Claude Sonnet/Haiku/Opus 4.5. | None. |
| 39 | Ronchetti | RW:228 | "Orthogonal Boundary Unlearning for Dual-Use Concepts in Large Language Models"; Niccolò Ronchetti; Oct 2026; arXiv; 2610.27490 | partly | "demonstrates durable resistance against adversarial relearning"; "100-step fine-tuning relearning audits". "Architecture not discussed" checked only by keyword search (no "hybrid", "DeltaNet" or "Mamba" in the HTML). | None. |
| 40 | Jin et al. (REX) | RW:232; ER:102–113 | "Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into Removable Experts in Mixture-of-Experts Models"; Jin, …, Zhang; Oct 2026; arXiv; 2610.25791 | yes | "On Qwen3.5, RMU rises from 44.1% to 78.3%"; REX "from 32.4% to 42.8%"; post-training is "full-parameter supervised fine-tuning…with the life-science subsets removed". RW gave a wrong short title. | Title fixed; models added. |
| 41 | Sen Sharma et al. | RW:256 | "Locating and Editing Factual Associations in Mamba"; Sen Sharma, Atkinson, Bau; 2024; COLM 2024; 2404.03646 | yes | Mamba-2.8b vs Pythia-2.8b; "rank-one model editing methods can successfully insert facts"; ROME tested on W_a, W_g, W_o. | Clarified which matrices. |
| 42 | Endy et al. | RW:261 | "Mamba Knockout for Unraveling Factual Information Flow"; 2025; ACL 2025; 2505.24244 | n/a | Cited without a claim. | None. |
| 43 | Nishi et al. | RW:262 | "Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing"; Nishi, …, Lubana; 2024; ICML 2025; 2410.17194 | yes | "We further corroborate our findings in naturalistic settings with pre-trained Llama and Mamba models". | Full title. |
| 44 | Michalak & Abreu | RW:264 | "Some Attention is All You Need for Retrieval"; 2025; arXiv; 2510.19861 | yes | "retrieval depends exclusively on self-attention layers…SSM layers show no compensatory mechanisms". | None. |
| 45 | Zeng et al. | RW:266 | "Who Asks, Who Answers: Linear Attention Builds the Query in Hybrid Language Models"; Zeng, Feng, Chen; Nov 2026; arXiv; 2611.01909 | yes | "linear attention constructs the query, while softmax attention performs the lookup." | None. |
| 46 | Bick et al. | RW:268 | "Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism"; Bick, Xing, Gu; 2025; arXiv; 2504.18574 | yes | "pretrained hybrid models…delegate the role of Aggregate Heads to attention." RW gave a non-existent short title. | Title fixed. |
| 47 | Borobia et al. (ablation) | RW:270 | "Component Ablation for Efficient Hybrid Language Model Architectures: Performance, Resilience, and Compression Implications"; Borobia, Seguí-Mas, Tormo-Carbó; 2026; arXiv; 2603.22473 | yes | "removing either attention or the alternative sequence-processing pathway substantially degrades performance". | None. |
| 48 | Borobia et al. (LoRA) | RW:270 | "Where Should LoRA Go? Component-Type Placement in Hybrid Language Models"; same authors; 2026; arXiv; 2604.22127 | yes | "the attention pathway…consistently outperforms full-model adaptation with 5-10x fewer trainable parameters." (This claim is from this paper only.) | None. |
| 49 | Arora et al. (Zoology) | RW:273 | "Zoology: Measuring and Improving Recall in Efficient Language Models"; 2023; ICLR 2024; 2312.04927 | yes | "attention outperforms gated-convolutions by a large margin" on associative recall. | None. |
| 50 | Wang & Reid | RW:280 | "Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models"; Wenlong Wang, Fergal Reid; Sep 2026; arXiv; 2609.32102 | yes | "On Mamba-2, state alone exceeds residual and logit lenses on all six families". | None. |
| 51 | Bosco & Srinivasan | RW:285 | "Locating and Steering Refusal Beyond Attention"; Sep 2026; arXiv; 2609.04721 | partly | "harm is cleanly readable at this output, the write site, before the addition"; four families (SSM, transformer, recurrent, hybrid). RWKV and Zamba2 confirmed; the other model names were not checked. | None (see section c). |
| 52 | Paulo et al. | RW:290 | "Does Transformer Interpretability Transfer to RNNs?"; Paulo, Marshall, Belrose; 2024; arXiv; 2404.05971 | yes | "most of these techniques are effective when applied to RNNs". | None. |
| 53 | Ali et al. | RW:292 | "The Hidden Attention of Mamba Models"; Ali, Zimerman, Wolf; 2024; ACL 2025 (2025.acl-long.76); 2403.01590 | n/a | Cited without a claim. | None. |
| 54 | Sunku Mohan et al. | RW:292 | "Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks"; 2026; arXiv; 2602.22719 | yes | "a test-time steering intervention…Across 7 SSMs and 6 diverse benchmarks". | None. |
| 55 | Black et al. | RW:295 | "Open-weight genome language model safeguards: Assessing robustness via adversarial fine-tuning"; 2025; NeurIPS 2025 workshop; 2511.19299 | yes | "data exclusion might be circumvented by fine-tuning approaches that can, to some degree, rescue misuse-relevant capabilities" (Evo 2). | Title added. |
| 56 | Wei et al. (BioRiskEval) | RW:295 | "Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models"; Wei, Che, …, Mazeika; 2025; arXiv; 2510.27629 | yes | "Excluded knowledge can be rapidly recovered in some cases via fine-tuning"; Evo 2 is the model studied. RW gave the framework name as the title. | Title fixed. |
| 57 | Lazaridis et al. | RW:302 | "When Can One Neuron Fix Repetition Loops in LLMs?"; 2026; arXiv; 2606.13705 | yes | "In exploratory experiments on Qwen3.5 and LFM2.5, sparse edits also reduce repetition". | None. |
| 58 | Amini et al. (LFM2) | RW:307; ER:113 | "LFM2 Technical Report"; Amini, …, Liquid AI; Nov 2025; arXiv; 2511.23404 | yes | Block equation as quoted; "a mixture-of-experts variant (8.3B total, 1.5B active)"; LFM3 blog: kernel "7, up from 3 in LFM2". | None. |
| 59 | Liquid AI (LFM3) | RW:309 | "Introducing LFM3: Every Modality for Specialized Intelligence"; Liquid AI Blog, Oct 27 2026; model card LiquidAI/LFM3-1.7B; `config.json` | yes | "Of its 24 layers, 13 are convolutions, 5 are SWA with a 256-token window, and 6 are global attention"; sigmoid head gate; sandwich norm; 22T tokens; config `conv_L_cache: 7`, `tie_word_embeddings: true`; BibTeX key `liquidAI2026LFM3`. | None. |
| 60 | STAR | RW:313 | "STAR: Synthesis of Tailored Architectures"; Thomas, Parnichkun, Amini, Massaroli, Poli; 2024; arXiv; 2411.17800 | yes | "search space based on the theory of linear input-varying systems". | None. |
| 61 | H3 | RW:313 | "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"; Fu, Dao, …, Ré; 2022; ICLR 2023; 2212.14052 | yes | — | None. |
| 62 | Hyena | RW:313 | "Hyena Hierarchy: Towards Larger Convolutional Language Models"; Poli et al.; 2023; arXiv 2302.10866 | yes | — | None. |
| 63 | MAD | RW:313 | "Mechanistic Design and Scaling of Hybrid Architectures"; Poli et al.; 2024; arXiv 2403.17844 | yes | — | None. |
| 64 | Shi et al. (MUSE) | RW:319 | "MUSE: Machine Unlearning Six-Way Evaluation for Language Models"; Shi, …, Zhang; 2024; ICLR 2025; 2407.06460 | yes | "for Books, we use f0=ICLM-7B…which does not contain the Harry Potter books in its pretraining data"; News uses LLaMA-2 7B. | None. |
| 65 | Li et al. (WMDP) | RW:322 | "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning"; Li, Pan, Gopal, …, Hendrycks; 2024; ICML 2024 (PMLR 235); 2403.03218 | yes | RMU on "zephyr-7b-beta…Yi-34b-Chat…and Mixtral-8x7B-Instruct-v0.1". | None. |
| 66 | Zhang et al. (NPO) | RW:324 | "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning"; Zhang, Lin, Bai, Mei; 2024; COLM 2024 (accepted-papers list); 2404.05868 | yes | — | Venue added. |
| 67 | Eldan & Russinovich | RW:326 | "Who's Harry Potter? Approximate Unlearning in LLMs"; 2023; arXiv; 2310.02238 | yes | — | None. |
| 68 | Arditi et al. | RW:327 | "Refusal in Language Models Is Mediated by a Single Direction"; Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda; 2024; NeurIPS 2024; 2406.11717 | yes | "erasing this direction from the model's residual stream activations prevents it from refusing harmful instructions". | None. |
| 69 | Maini et al. (TOFU) | RW:329 | "TOFU: A Task of Fictitious Unlearning for LLMs"; Maini, Feng, Schwarzschild, Lipton, Kolter; 2024; arXiv; 2401.06121 | n/a | Venue not confirmed (see c). | Full citation added. |
| 70 | Arditi & Chughtai (LessWrong) | RW:346 | "Unlearning via RMU is mostly shallow"; Andy Arditi, Bilal Chughtai; LessWrong / Alignment Forum, 23 Jul 2024 | n/a | Exists; not cited in the text. | Status line updated. |
| 71 | Zhuang et al. (SEUF) | RW:236 (new) | "SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?"; Zhuang, Zhang, Guo, Jia, Liu, Liu, Zhang; 2024; ACL 2025; 2411.18797 | yes | "sparse Mixture-of-Experts (MoE) LLMs…have remained unexplored in the context of unlearning". | Added. |
| 72 | Zhu et al. (GRIP) | RW:236 (new) | "GRIP: Algorithm-Agnostic Machine Unlearning for Mixture-of-Experts via Geometric Router Constraints"; Zhu, Wei, Gu, Li; 2026; arXiv; 2601.16905 | yes | "adversaries can bypass the router to recover sensitive information directly from dormant experts." | Added. |
| 73 | Xie et al. (TRACE) | RW:236 (new) | "Routing-Aware Expert Calibration for Machine Unlearning in Mixture-of-Experts Language Models"; Xie, Lin, Xiong, Zhang, Li; 2026; arXiv; 2606.10338 (authors note an error in Eq. 4–6) | yes | "forget data often activates a small subset of experts disproportionately". | Added. |
| 74 | Ramesh | RW:244 (new) | "Exact Record Omission in Delta Attention: A Transport Criterion, Its Cost, and a Replay Certificate"; Vishwajith Ramesh; Sep 2026; arXiv; 2609.06872 | yes | "Attention masking removed all 40 exact greedy target answers, yet a…candidate-scoring attack…attained AUC 0.740". | Added (adjacent, not weight unlearning). |

### ERASURE_RESEARCH.md (references not already in RW)

| # | Ref | As cited | Verified title; authors; year; venue; link | Supports | Supporting quote or discrepancy | Action |
|---|---|---|---|---|---|---|
| 75 | Qi et al. | ER:34–36, 119–123, 130–131 | "On Evaluating the Durability of Safeguards for Open-Weight LLMs"; Qi, Wei, Carlini, …, Henderson; 2024; ICLR 2025; 2412.07097 | yes | Pitfalls: dataset shuffling, "a different implementation of the fine-tuning trainer", "slight modifications of the fine-tuning configurations", "a small change to the prompt template"; TAR ">60%" WMDP-Bio; "gradient explosion in the initial steps"; GSM8K "near zero"; RepNoise disjoint sets "no significant difference". | Venue added. |
| 76 | Thudi et al. | ER:36 | "On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning"; Thudi, Jia, Shumailov, Papernot; 2021; USENIX Security 2022; 2110.11891 | yes | "unlearning is only well-defined at the algorithmic level". | None. |
| 77 | Cooper et al. | ER:38 | "Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research"; A. F. Cooper, …, K. Lee; 2024; NeurIPS 2025 (Oral); 2412.06966 | yes | "the targeted removal of information from a model and the targeted suppression of information from a model's outputs". | None. |
| 78 | Bourtoule et al. (SISA) | ER:42 | "Machine Unlearning"; Bourtoule, …, Papernot; 2019; IEEE S&P 2021; 1912.03817 | yes | "We introduce SISA training". | None. |
| 79 | Guo et al. (certified removal) | ER:43 | "Certified Data Removal from Machine Learning Models"; Guo, Goldstein, Hannun, van der Maaten; 2019; ICML 2020; 1911.03030 | yes | "a certified-removal mechanism for linear classifiers". | None. |
| 80 | Mu & Klabjan (R2D) | ER:43 | "Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions"; 2024; NeurIPS 2025; 2409.09778 | yes | "unlearns by 'rewinding' to an earlier step…before performing gradient descent on the loss function of the retained data points." | None. |
| 81 | Chowdhury et al. (S3T) | ER:53 | "Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning"; Basu Roy Chowdhury et al.; 2024; ICLR 2025; 2406.16257 | yes | "sequentially training layers with disjoint data slices…simply deactivating the layers affected by data deletion." | None. |
| 82 | Kuo et al. (SIFT-Masks) | ER:54–56 | "Exact Unlearning of Finetuning Data via Model Merging at Scale"; Kuo, Setlur, Srinivas, Raghunathan, Smith; 2025; arXiv; 2504.04626 | yes | "merge up to 500 models…up to 250x less compute for exact unlearning compared to other merging baselines." | None. |
| 83 | Golatkar et al. | ER:57 | "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks"; Golatkar, Achille, Soatto; 2019; CVPR 2020; 1911.04933 | yes | "an upper-bound on the amount of information remaining in the weights". | None. |
| 84 | Sekhari et al. | ER:58 | "Remember What You Want to Forget: Algorithms for Machine Unlearning"; Sekhari, Acharya, Kamath, Suresh; 2021; arXiv 2103.03279 | yes | "For the setting of convex losses, we provide an unlearning algorithm that can unlearn up to O(n/d^{1/4}) samples". | None. |
| 85 | Jia et al. (SOUL) | ER:61 | "SOUL: Unlocking the Power of Second-Order Optimization for LLM Unlearning"; 2024; EMNLP 2024; 2404.18239 | yes | — | None. |
| 86 | Foster et al. (SSD) | ER:61 | "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening"; Foster, Schoepf, Brintrup; 2023; AAAI 2024; 2308.07707 | yes | "uses the Fisher information matrix of the training and forgetting data". | None. |
| 87 | Graves et al. | ER:66 | "Amnesiac Machine Learning"; Graves, Nagisetty, Ganesh; 2020; arXiv 2010.10981 | partly | Abstract names "Amnesiac Unlearning" but does not describe the batch-update subtraction; that detail was not checked in the full text. | None (see c). |
| 88 | Rathi & Radford | ER:77 | "Shaping capabilities with token-level data filtering"; 2026; arXiv; 2601.21571 | yes | "filtering tokens is more effective than filtering documents"; "a 7000x compute slowdown on the forget domain". | None. |
| 89 | Chen et al. (Anthropic blog) | ER:78–80 | "Enhancing Model Safety through Pretraining Data Filtering"; Yanda Chen, …, Mrinank Sharma; Alignment Science Blog, Aug 19 2025 | yes | "from 33.7±0.4% to 30.8±0.4%, where a random baseline is 25%, while standard benchmarks on MMLU, Code and Prose showed no significant drop." | None. |
| 90 | Cloud et al. (gradient routing) | ER:86 | "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks"; Cloud, Goldman-Wetzler, Wybitul, Miller, Turner; 2024; arXiv 2410.04332 | yes | "enable robust unlearning via ablation of a pre-specified network subregion". | None. |
| 91 | Shilov et al. (SGTM) | ER:87–90 | "Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"; Shilov, Cloud, Gema, Goldman-Wetzler, Panickssery, Sleight, Jones, Anil; Dec 2025; arXiv 2512.05648 (Anthropic Fellows Program) | yes | "254M parameter model"; biology "3.7% of overall tokens"; "RMU…only 50 steps. SGTM (350 steps) is as robust as strict data filtering"; "92M vs 13M tokens"; "5% compute penalty". | None. |
| 92 | Roland et al. (GRAM) | ER:90–93 | "Modular Pretraining Enables Access Control"; Roland, …, Anil, Cloud; 2026; ICML 2026; 2607.08077 | yes | "MaxEnt performs worst: the degradation it induces on the forget set is mostly reversed by finetuning"; "GRAM closely tracks data filtering"; "5x reduction". | None. |
| 93 | Ghosal et al. (NULLs) | ER:93–95 | "Natively Unlearnable Large Language Models"; Ghosal, Maini, Raghunathan; 2026; arXiv 2606.13873 | yes | "NULLs resists both adversarial extraction and relearning that reverses post-hoc unlearning." | None. |
| 94 | Ghosal et al. (MemSinks) | ER:95–96 | "Memorization Sinks: Isolating Memorization during LLM Training"; 2025; ICML 2025; 2507.09937 | yes | "activates a unique set of memorization neurons for each sequence". | None. |
| 95 | Gururangan et al. (DEMix) | ER:106 | "DEMix Layers: Disentangling Domains for Modular Language Modeling"; 2021; arXiv 2108.05036 | yes | "experts can be removed to restrict access to unwanted domains, without additional training." | None. |
| 96 | Li et al. (BTM) | ER:107 | "Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models"; 2022; arXiv 2208.03306 | yes | "These ELMs can be added and removed to update data coverage". | None. |
| 97 | Shi et al. (FlexOlmo) | ER:107 | "FlexOlmo: Open Language Models for Flexible Data Use"; 2025; arXiv 2507.07024 | yes | "can be flexibly included or excluded from model inferences with no further training." | None. |
| 98 | Henderson et al. (MLAC) | ER:116 | "Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models"; Henderson, Mitchell, Manning, Jurafsky, Finn; 2022; AIES 2023; 2211.14946 | yes | "MLAC can largely prevent a BERT-style model from being re-purposed". | None. |
| 99 | Deng et al. (SOPHON) | ER:117 | "SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models"; 2024; IEEE S&P 2024; 2404.12699 | yes | "two deep learning modes (classification and generation)". | None. |
| 100 | Sondej et al. (MUDMAN) | ER:118–119 | "Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization"; Sondej, Yang, Kniejski, Windys; 2025; NeurIPS 2025 Workshop (CCFM), per the paper's footer; 2506.12484 | partly | "MUDMAN outperforms the prior TAR method by 40%". ER said "NeurIPS 2025" (main track implied). | Venue fixed. |
| 101 | Rosati et al. (RepNoise) | ER:127–128 | "Representation Noising: A Defence Mechanism Against Harmful Finetuning"; 2024; NeurIPS 2024; 2405.14577 | yes | "the efficacy of our defence lies in its 'depth'". Broken by Qi et al. | None. |
| 102 | Huang et al. (Vaccine) | ER:129 | "Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack"; Huang, Hu, Liu; 2024; NeurIPS 2024; 2402.01109 | yes | "fine-tuning-as-a-service". | None. |
| 103 | Huang et al. (Booster) | ER:129 | "Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation"; 2024; arXiv 2409.01586 | yes | — | None. |
| 104 | Dong et al. (UNDIAL) | ER:144 | "UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models"; Dong, Lin, Belkin, Huerta, Vulić; 2024; arXiv 2402.10052 | partly | UNDO swept UNDIAL as a baseline; "is weaker" not checked. | None (see c). |
| 105 | Lee (Bitter Lessons) | ER:144–146 | "Bitter Lessons from Distillation Robustifies Unlearning"; Bruce W. Lee; LessWrong, posted 27 Nov 2025, 20:31 New York time (author's blog dates it December 2025) | yes | "post hoc robust unlearning may not be achievable at all, at least not efficiently"; "Sufficient damage appears to be a necessary condition for robust unlearning, not a sufficient one." ER gave the UTC date. | Date fixed (28 → 27 Nov). |
| 106 | Ash & Adams | ER:149 | "On Warm-Starting Neural Network Training"; 2019; NeurIPS 2020; 1910.08475 | yes | — | None. |
| 107 | Tice et al. | ER:150–151 | "Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models"; 2024; NeurIPS 2025; 2412.01784 | yes | "sandbagging models exhibit anomalous performance improvements, likely due to disruption of underperformance mechanisms". | None. |
| 108 | Lo et al. | ER:154–155 | "Large Language Models Relearn Removed Concepts"; Lo, Barez, Cohen (ACL Anthology order; arXiv lists Lo, Cohen, Barez); 2024; Findings of ACL 2024; 2401.01814 | partly | "models can quickly regain performance post-pruning by relocating advanced concepts to earlier layers and reallocating pruned concepts to primed neurons with similar semantics." "Within a few epochs" was not in the source we read. | Author order and wording fixed. |
| 109 | Farrell et al. | ER:156–157 | "Applying sparse autoencoders to unlearn knowledge in language models"; Farrell, Lau, Conmy; 2024; arXiv 2410.19278 | yes | "negative scaling of feature activations is necessary and that zero ablating features is ineffective"; side effects "similar or larger" than RMU. | None. |
| 110 | Muhamed et al. (DSG) | ER:157–158 | "SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs"; Muhamed, Bonato, Diab, Smith; 2025; COLM 2025; 2504.08192 | yes | "stronger resistance to relearning attacks"; uses "a dynamic classifier". The guardrail reading is ours. | None. |
| 111 | Zhang et al. (Roots Beneath the Cut) | ER:159–160 | "Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models"; 2026; CVPR 2026; 2603.06640 | yes | "reviving erased concepts from pruned diffusion models in a fully data-free and training-free manner." | None. |
| 112 | Belrose et al. (LEACE) | ER:164 | "LEACE: Perfect linear concept erasure in closed form"; 2023; NeurIPS 2023; 2306.03819 | yes | "provably prevents all linear classifiers from detecting a concept"; "concept scrubbing…every layer". | Venue added. |
| 113 | Ravfogel et al. (INLP) | ER:166 | "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection"; 2020; ACL 2020; 2004.07667 | yes | — | None. |
| 114 | Ravfogel et al. (kernelized) | ER:167–168 | "Kernelized Concept Erasure"; Ravfogel, Vargas, Goldberg, Cotterell; 2022; EMNLP 2022; 2201.12191 | yes | "the protection does not transfer to different nonlinear adversaries." | None. |
| 115 | Gandikota et al. (ELM) | ER:169–170 | "Erasing Conceptual Knowledge from Language Models"; Gandikota, Feucht, Marks, Bau; 2024; arXiv 2410.02760 | yes | "exhibiting strong robustness to adversarial attacks"; ELM is in Che et al.'s Table 1. | None. |
| 116 | Meng et al. (ROME) | ER:173 | "Locating and Editing Factual Associations in GPT"; 2022; NeurIPS 2022; 2202.05262 | yes | — | None. |
| 117 | Meng et al. (MEMIT) | ER:174 | "Mass-Editing Memory in a Transformer"; 2022; ICLR 2023; 2210.07229 | yes | — | Venue added. |
| 118 | Fang et al. (AlphaEdit) | ER:174 | "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"; 2024; ICLR 2025 (Oral); 2410.02355 | yes | "projects perturbation onto the null space of the preserved knowledge". | None. |
| 119 | Youssef et al. | ER:175–177 | "Tracing and Reversing Edits in LLMs"; Youssef, Zhao, Seifert, Schlötterer; 2025; ICLR 2026; 2505.20819 | yes | "solely based on the modified weights…with up to 99% accuracy"; "reverses up to 94% of the edits". | None. |
| 120 | Ilharco et al. | ER:181 | "Editing Models with Task Arithmetic"; 2022; ICLR 2023; 2212.04089 | yes | — | None. |
| 121 | Kim et al. (NegMerge) | ER:182 | "NegMerge: Sign-Consensual Weight Merging for Machine Unlearning"; Kim, Han, Choe; 2024; ICML 2025; 2410.05583 | n/a | Image recognition only. | Venue and scope added. |
| 122 | Wang et al. (VLA audit) | ER:182–183 | "Suppression Sticks, Locality Is Fragile: A Closed-Loop Target-and-Control Audit of Task-Vector Negation in VLA Policies"; 2026; arXiv 2608.04692 | yes | "negation is temporary behavioral suppression, not certified or safety-grade unlearning". | None. |
| 123 | Sondej et al. (RepSelect) | ER:187–190 | "RepSelect: Robust LLM Unlearning via Representation Selectivity"; Sondej, Yang, Mahdi; 2026; arXiv 2606.17168 | yes | "4-40x larger drop in post-relearning answer probability than five widely used baselines (GradDiff, NPO, SimNPO, RMU, UNDIAL)". | None. |
| 124 | Xiao et al. (MCU) | ER:190–191 | "Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter"; 2026; arXiv 2605.11685 | yes | "the modifications in these dominant components are easily reversed…whereas minor components exhibit stronger resistance". | None. |
| 125 | Sheshadri et al. (targeted LAT) | ER:191–192 | "Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs"; 2024; arXiv 2407.15549 | yes | "more effectively unlearn knowledge…in a way that is also more robust to re-learning." | None. |
| 126 | Wallace et al. | ER:193 | "Estimating Worst-Case Frontier Risks of Open-Weight LLMs"; Wallace, Watkins, Wang, Chen, Koch; 2025; arXiv 2508.03153 | partly | They estimate risk by malicious fine-tuning of gpt-oss; they do not attempt removal, so "claims no removal" was misleading. | Reworded. |

### Later work that refutes or qualifies a cited claim

- TAR and RepNoise durability (rows 30, 101): broken by Qi et al., ICLR 2025 (row 75). Both docs already say so.
- MUDMAN "outperforms TAR by 40%" (row 100): measured against a baseline that Qi et al. broke; no independent attack on
  MUDMAN found.
- ELM's robustness claim (row 115): Che et al. (row 5) undo it within 16 fine-tuning steps.
- RMU/NPO relearning results (rows 1–6): Feng et al. 2025 (row 9) argue that many such evaluations inject new
  information. RW already lists this.
- WMDP's RMU robustness (row 65): qualified by rows 3, 5 and 6.
- ROME/MEMIT as deletion (rows 116–117): Patil et al. and Youssef et al. (rows 31, 119).
- Deep Ignorance-style filtering (row 25): for DNA models, rows 55–56 show filtered capabilities return under
  fine-tuning; not tested on text.
- No refutation found for Song et al., SGTM, GRAM, NULLs, REX, RepSelect, MCU, UNDO or Fan et al. (the SAM disagreement
  is our own result).

### Substantive wording changes (not marked in the documents)

1. RW:30–34 Łucki: "undone by fine-tuning" → LoRA (rank 128) on 10 retain samples "recovers most" (61.6% vs 64.4%).
2. RW:68, 74–76, 337–338: "full fine-tuning" → LoRA r128; the §2 setup-difference sentence and the §12 correction list
   updated to match.
3. RW:95–100 MAST: the 45/150 → 37/150 result moved to Qwen2.5-Math-1.5B; Qwen3 result stated separately.
4. RW:154–159 Fan: sample counts and epochs per benchmark; "NPO+SAM most robust" → "SAM makes NPO more robust".
5. RW:119–122 Peleg: quote corrected; "Agrees" → "consistent with (our reading)".
6. RW:50–53 Qian: hypothesis no longer stated as a finding.
7. RW:35–38 Lynch: "other languages" removed from the list of routes that extract knowledge.
8. RW:18–24 Hu: verbatim-HP experiment attributed to the right model.
9. RW:173–175 and ER:175 Patil: 38% attributed to whitebox plus blackbox attacks on ROME (GPT-J).
10. RW:140–142 K-Bench: "removes" → "demonstrably removes".
11. RW:115–118 PreUnlearn: symmetry argument marked as our inference.
12. RW:161–166 Guo: numbers tied to Sports Facts and to Causal Tracing vs fact-lookup localization.
13. ER:154–155 Lo: "within a few epochs" → "quickly".
14. ER:193 Wallace: "claims no removal" → "does not attempt removal".
15. RW §10 (RW:219–253): novelty paragraph extended (MoE unlearning, context-state deletion) and the suggested report
    wording replaced (see b).

Title, venue, author, date and tag fixes (rows 1, 6, 7, 9, 10, 16, 17, 21, 22, 25, 27, 35, 40, 43, 46, 55, 56, 66, 69,
75, 100, 105, 108, 112, 117, 121) are not repeated here.

## (b) Novelty claim

**What exists.**

- *SSM / RWKV / Griffin / xLSTM / Hymba / Zamba / Jamba / LFM:* no weight-unlearning or relearning-robustness study of a
  language model was found (arXiv API abstract search for unlearning × Mamba, SSM, RWKV, Jamba, Zamba, Hymba,
  RecurrentGemma, Griffin, xLSTM, DeltaNet, LFM2, Liquid; OpenAlex full-index search; web search). The only
  "unlearning × Mamba" hit is a land-surface forecasting model (arXiv 2610.02248). Semantic Scholar returned HTTP 429
  throughout; Google Scholar was not queried.
- *Adjacent work on non-transformers:* knowledge editing and localization in Mamba (Sen Sharma et al., COLM 2024;
  Endy et al., ACL 2025; Nishi et al., ICML 2025); refusal-direction ablation in SSM, RWKV and Zamba2 (Bosco &
  Srinivasan 2026); Jacobian lenses on Mamba (Wang & Reid 2026); interpretability transfer to Mamba/RWKV (Paulo et
  al. 2024); deleting a record from a linear-attention hybrid's recurrent *state* at inference time (Ramesh 2026,
  Kimi Linear). None of these unlearns weights.
- *Hybrids appearing incidentally in unlearning papers:* Ronchetti 2026 (Qwen3.5-4B, a Gated DeltaNet hybrid, among
  four models; no architecture analysis found); Jin et al. 2026 (REX on Qwen3.5-35B-A3B, treated as an MoE). That
  Qwen3.5-35B-A3B uses the same hybrid layout as the smaller Qwen3.5 models is family-level evidence only.
- *Mixture-of-experts transformers* have architecture-aware unlearning work: SEUF (ACL 2025), GRIP (2026, a
  router-specific "hidden, not removed" failure), TRACE (2026), REX (2026); RMU was run on Mixtral-8x7B in WMDP (2024).
  Their token mixer is still attention, so they do not contradict a mixer-type claim, but they do contradict a claim
  that no unlearning study looks at architecture-specific parts.
- *Conv/attention hybrid outside text:* Evo 2 (StripedHyena 2) data-filtering recovery under fine-tuning (Black et al.
  2025; Wei et al. 2025). This is DNA and data filtering, not post-hoc unlearning of a language model.

**Recommended sentence** (now in RW:249–252):

> To our knowledge, LLM unlearning studies have almost exclusively evaluated dense or mixture-of-experts transformers;
> the few that include hybrid models (Ronchetti 2026; Jin et al. 2026) do not analyse the non-attention blocks, and we
> know of no unlearning study on a gated short-convolution hybrid (or any LFM model) or one that localizes forgetting
> and recovery across mixer types.

Do not write "almost all unlearning studies use pure transformers" (MoE transformers are a separate and growing
line), and do not claim the first unlearning study on any non-transformer *model* (Evo 2 is a conv hybrid, though not a
text LLM and not post-hoc unlearning).

## (c) Items not verified

- UNVERIFIED: TOFU's venue (COLM 2024 not confirmed; cite arXiv 2401.06121).
- UNVERIFIED: Bosco & Srinivasan's full model list (Falcon-Mamba, Codestral-Mamba, Llama, Mistral); only RWKV and
  Zamba2 were seen in the text.
- UNVERIFIED: Ronchetti "architecture not discussed": checked only by keyword search of the HTML full text.
- UNVERIFIED: that Qwen3.5-35B-A3B (REX) is a Gated DeltaNet hybrid; shown for Qwen3.5-0.8B (Borobia et al.) and six
  Qwen3.5 models (Zeng et al.), not for this checkpoint.
- UNVERIFIED: Evo 2 = StripedHyena 2 (background knowledge; RW already says so).
- UNVERIFIED: SGTM's attack data mix ("50/50 bio/general", ER:88); the step and token counts were verified.
- UNVERIFIED: REX initialization detail ("sources scaled down", ER:103) and which model the pipeline ablation (79.4% →
  64.3%, 68.4%) uses; the numbers and their meaning (F1 full, A1 excised before cleanup) were verified.
- UNVERIFIED: Amnesiac ML stores and subtracts per-batch updates (ER:66); the abstract does not describe it.
- UNVERIFIED: "UNDIAL…is weaker" (ER:144).
- UNVERIFIED: Deep Ignorance "they improve under attack but stay well below the unfiltered baseline" (ER:75–76);
  only the abstract's "substantial resistance" was checked.
- UNVERIFIED (gap claims, not re-searched): no independent attack on MUDMAN, SOPHON, Vaccine or Booster (ER:124, 132);
  no public intermediate LFM3 pretraining checkpoints (ER:70–71).
- Not checked: the RW-internal numbers from our own results (e.g. RW:185–186, 0.15 vs 0.14), which are outside the
  scope of a citation check.
