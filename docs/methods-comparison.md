# Methods Comparison: Latent Chain-of-Thought Reasoning

**Purpose:** Shared reference for agents and team members building on SIM-CoT toward adaptive-k latent reasoning.
**Last updated:** 2026-06-07

---

## Overview

The implicit CoT lineage begins with **Coconut**, which replaces discrete reasoning tokens with continuous hidden states fed autoregressively, enabling reasoning in an unrestricted latent space at the cost of emergent but unstructured BFS behavior and training collapse beyond c=2 thoughts per step. **CODI** reacts to Coconut's curriculum-induced catastrophic forgetting by introducing single-stage self-distillation: a teacher CoT path guides a student implicit CoT path through hidden-state alignment, eliminating the need for progressive replacement schedules. **SIM-CoT** identifies a deeper instability shared by both predecessors - latent homogenization when K scales beyond the default - and addresses it with step-level supervision via a training-time auxiliary decoder that aligns each latent token z_k to its corresponding explicit reasoning step, discarded at inference for zero overhead. Orthogonally, **PonderNet** contributes a principled adaptive computation framework from the recurrent network literature: a per-step halting probability λ_n drawn from a learned geometric distribution, trained with an expected task loss plus a KL regularizer against a geometric prior. PonderNet is not an implicit CoT method, but its adaptive halting mechanism is the key missing piece for making K dynamic rather than fixed in the Coconut/CODI/SIM-CoT family.

---

## Comparison Table

| Dimension | Coconut | CODI | SIM-CoT | PonderNet |
|---|---|---|---|---|
| **Training objective** | NLL on answer tokens only (masked on question + latents) | α·L_student + β·L_KD (L1 hidden-state alignment at distillation token, stop-grad teacher) + γ·L_teacher | λ_step·L_step (step-level CE via aux decoder) + λ_lm·L_ans-lm | L_Rec = Σ p_n·L(y,ŷ_n) + β·KL(p_n ‖ p_G(λ_p)) |
| **Latent token mechanism** | Last hidden state H_θ fed back as next input embedding; no vocab decoding between latents | Hidden-state autoregression (same as Coconut); two-layer MLP+LayerNorm projection on continuous thoughts; aligned to teacher at ":" distillation token | Last hidden state z_k = H_θ(U^(k-1)) appended to prefix; aux decoder p_φ conditions only on z_k to generate step s_k | Step function s(x, h_n) → (ŷ_n, h_{n+1}, λ_n); output and halting probability predicted at each recurrent step |
| **Number of latent tokens** | Fixed; set by multi-stage curriculum (progressively replace language CoT steps; c continuous thoughts per step) | Fixed at 6 continuous thoughts; no curriculum | Fixed K set in advance; curriculum gradually increases K up to K_max over training | Variable; halting determined stochastically via Bernoulli(λ_n) at each step; geometric prior encourages efficiency |
| **Inference behavior** | Fixed latent count; binary classifier or fixed padding to trigger `<eot>`; only final answer verbalized | Fixed 6 continuous thoughts; only final answer verbalized; aux decoder absent | K implicit forward passes, then explicit answer decoding; aux decoder removed; total length = T + K + L_a | Sample Λ_n ~ Bernoulli(λ_n) at each step; output ŷ_n when Λ_n = 1; stochastic; may halt at any step |
| **Key supervision signal** | Answer-level: NLL on final answer | Trajectory-level: L1 alignment of all-layer hidden states at distillation token between teacher and student paths | Step-level: each latent z_k cross-entropy supervised to generate textual step s_k via aux decoder | Expected task loss weighted by halting distribution p_n; KL against geometric prior p_G(λ_p) |
| **Tested on** | GSM8k, ProntoQA, ProsQA | GSM8k (GPT-2 scale and larger) | GSM8k-Aug (in-domain); SVAMP, GSM-Hard, MultiArith (OOD); GPT-2, LLaMA 1B/3B/8B | Parity (extrapolation), bAbI (question answering); small-scale recurrent networks |
| **Best reported result** | 34.1% GSM8k (c=2, 6 thoughts); 0.09s vs. 0.26s for CoT; outperforms CoT on ProntoQA/ProsQA | 99% of CoT-SFT on GSM8k at GPT-2 scale; +28.2% over Coconut; 3.1× compression, 2.7× speedup | GPT-2: +2.1% over CoT-SFT, +8.2% over Coconut; LLaMA-3.1 8B: +3.0% over CODI; 2.3× speedup vs. CoT (GPT-2) | Outperforms ACT on parity extrapolation; matches/beats Universal Transformer on bAbI with 6× fewer steps |
| **Key limitation** | c≥3 → training collapse; no step-level supervision → homogenization; diminishing returns on larger models | Accuracy peaks at 6 thoughts; vanishing gradients for early latent tokens; fixed K | K still fixed at inference; step-level supervision requires paired explicit CoT annotations for training | Only tested on small-scale tasks; stochastic inference adds variance; no LLM evaluation |

---

## Chain of Influence

### The Coconut → CODI → SIM-CoT Lineage

**Coconut** (Hao et al., 2025) is the foundational proposal for fully implicit CoT: rather than emitting discrete tokens between reasoning steps, the model's last hidden state is fed back directly as the next input embedding, bypassing vocabulary decoding entirely. The `<bot>`/`<eot>` markers delimit the latent segment, and a multi-stage curriculum progressively converts explicit language steps into c continuous thoughts per step. The emergent behavior - where continuous thoughts encode multiple BFS paths simultaneously - was a surprising and promising result. However, Coconut's answer-level supervision (NLL only on the final answer, with latent positions masked) provides no step-by-step guidance: the latent space is free to arrange itself in any way that predicts the answer, and this freedom becomes a liability. At c≥3, or equivalently when scaling to more latent tokens, training collapses catastrophically as the latent representations become homogeneous - numerically similar vectors that have lost their distinct semantic roles (e.g., operators vanish while numerical tokens dominate).

**CODI** (Shen et al., 2025b) is a direct response to Coconut's two failure modes: curriculum-induced catastrophic forgetting and the absence of intermediate supervision. CODI sidesteps the curriculum entirely by jointly training a teacher (standard CoT SFT) and a student (continuous thoughts) in a single stage, using an L1 hidden-state alignment loss at the ":" distillation token - the point immediately before the final answer where both paths must converge. A two-layer MLP+LayerNorm projection is applied to the student's continuous thought representations before alignment. This trajectory-level distillation gives the latent path a global anchor in the teacher's representation space without requiring the one-to-one step correspondence that explicit supervision would demand. CODI achieves the first implicit CoT system to match explicit CoT on GSM8k at GPT-2 scale (99% of CoT-SFT accuracy) and eliminates the forgetting problem. However, CODI's alignment signal is coarse: it supervises the endpoint of the latent trajectory but not each intermediate step, leaving the individual latent tokens under-constrained. Accuracy peaks at exactly 6 continuous thoughts, and vanishing gradients penalize early latent tokens.

**SIM-CoT** (Wei et al., 2025) extends CODI's diagnosis to its root cause. The authors systematically analyze what happens when K is scaled beyond the default in both Coconut and CODI: inter-latent distance collapses, latent vectors drift away from the vocabulary manifold, and operator-type information is lost while numerical information is retained. They name this the **latent instability issue** and attribute it specifically to insufficient step-level supervision. SIM-CoT introduces a training-time auxiliary decoder (architecturally identical to the base LLM) that takes each z_k as a conditioning prefix and generates the textual content of the k-th reasoning step, providing per-step cross-entropy supervision that forces each latent to encode distinct, semantically grounded information. Crucially, the decoder is discarded at inference, so the runtime cost is identical to other implicit CoT methods. SIM-CoT is framed as a plug-and-play module compatible with any implicit CoT backbone (Coconut, CODI, or training-free methods), and it enables stable scaling to K=8 latent tokens where Coconut collapses at K=5.

### PonderNet as an Independent Thread

**PonderNet** (Banino et al., 2021) arrives from the adaptive computation time (ACT) literature rather than from the implicit CoT lineage. Its contribution is a reformulation of ACT's halting problem as a probabilistic Markov process: at each step n, a network predicts halting probability λ_n, and the probability of halting at exactly step n follows p_n = λ_n ∏_{j<n}(1-λ_j), which is a geometric distribution. Training minimizes the expected task loss across all steps (weighted by p_n) plus a KL divergence against a geometric prior p_G(λ_p), where λ_p is a hyperparameter controlling the prior preference for fewer steps. Unlike ACT, PonderNet's gradients are unbiased - the geometric distribution supports propagation through all steps - enabling extrapolation to longer sequences at test time. PonderNet is not designed for language models or for latent token sequences, but its halting mechanism is directly applicable: if each latent step k in SIM-CoT/CODI were augmented with a scalar λ_k head, the model could learn to halt the latent chain when it has accumulated sufficient information, rather than consuming a fixed K budget.

---

## What Each Paper Contributes to This Project

### Coconut
- **Latent construction protocol:** The last-hidden-state autoregression pattern (z_k = H_θ(U^(k-1)), U^(k) = U^(k-1) ⊕ z_k) is the core mechanism both subgroups build on; the `<bot>`/`<eot>` boundary tokens are inherited directly.
- **Baseline and failure mode:** Coconut's collapse at K≥5 is the primary empirical motivation for both branches; fixed-K sweeps in the option-a branch use Coconut as one of the two backends being compared.
- **Emergent BFS behavior:** The observation that continuous thoughts encode multiple solution paths simultaneously motivates the hypothesis that adaptive-K reasoning may naturally allocate more steps to harder problems.

### CODI
- **Single-stage training without curriculum forgetting:** The develop-c and option-a branches both avoid Coconut's multi-stage curriculum by building on CODI's joint training regime, preventing the catastrophic forgetting that makes curriculum-based methods fragile.
- **Distillation signal architecture:** The MLP+LayerNorm projection and the concept of a distillation anchor token are reused as a template for how to couple teacher and student signals without requiring exact step alignment.
- **Second backend for fixed-K sweep:** The option-a branch uses CODI alongside Coconut to characterize how accuracy vs. token budget curves differ between a distillation-based and a curriculum-based implicit CoT approach.

### SIM-CoT
- **Upstream fork and step-level supervision:** SIM-CoT is the direct upstream this team has forked; the auxiliary decoder and the λ_step·L_step + λ_lm·L_ans-lm objective are the starting point for all experiments.
- **Latent instability analysis and diagnostics:** The inter-latent distance and distance-to-vocabulary-center metrics (Dist and DistVC) provide the quantitative tools both subgroups use to monitor whether adaptive halting preserves or degrades latent diversity.
- **Interpretability infrastructure:** The training-time decoder, reused at inference for visualization, is directly available to diagnose whether the halting mechanism produces semantically coherent stopping points or halts mid-reasoning.

### PonderNet
- **Adaptive halting mechanism (develop-c branch):** The per-step Bernoulli halting variable Λ_n ~ Bernoulli(λ_n) is the specific mechanism being grafted onto SIM-CoT's latent token sequence; each z_k will be augmented with a scalar λ_k head trained under the geometric halting distribution.
- **Geometric prior as regularizer:** The KL(p_n ‖ p_G(λ_p)) term is adopted to prevent the model from always halting at K_max or always computing the minimum number of steps; λ_p is a tunable prior that encodes a soft token budget preference.
- **Unbiased gradient argument:** PonderNet's theoretical analysis showing unbiased gradients relative to ACT is the justification for preferring it over simpler early-exit heuristics; the develop-c branch relies on this for stable end-to-end training of the halting head jointly with the latent encoder.

---

## Key Open Problems This Project Addresses

### 1. Fixed-K at Inference Is the Primary Remaining Bottleneck

All three implicit CoT methods (Coconut, CODI, SIM-CoT) share a hard structural limitation: K is fixed at training time and cannot vary per input at inference. A simple two-step addition problem and a multi-step algebraic word problem consume exactly the same latent budget. This wastes compute on easy inputs and may undershoot on hard ones. The option-a branch's fixed-K sweep is designed to quantify this cost empirically across both Coconut and CODI backends: by measuring accuracy at K ∈ {2, 4, 6, 8} and plotting the accuracy vs. latent-token-count frontier, the subgroup will establish whether there is a meaningful signal that motivates dynamic allocation (i.e., whether the optimal K is problem-dependent).

### 2. Latent Instability Recurs at Higher K Even With SIM-CoT

SIM-CoT stabilizes training up to K=8 on GSM8k-Aug via step-level supervision, but SIM-CoT itself does not solve the instability at arbitrary K - it raises the stability ceiling without eliminating the fixed-K constraint. The develop-c branch addresses this by replacing the fixed-K decision with a learned halting policy. If the halting head λ_k learns to stop before the latent space degenerates, adaptive-K reasoning could naturally regularize the number of steps without requiring tuning of K as a hyperparameter. This is the central hypothesis: PonderNet-style halting acts as a dynamic early-exit mechanism that prevents the model from continuing into the collapse-prone high-K regime.

### 3. Homogenization Is Exacerbated by Unnecessary Steps

SIM-CoT's geometric diagnostics (Section 2 and Appendix I) show that in failed models, trailing latent tokens converge to near-identical representations - repeating the final prediction rather than adding new information (Appendix K notes this even in successful models). Adaptive halting directly addresses this: if λ_k → 1 once the answer is sufficiently encoded, the model terminates before producing redundant latents, preserving the inter-latent diversity that SIM-CoT's step supervision establishes. The DistVC metric inherited from SIM-CoT will be used to verify that adaptive stopping keeps latent vectors within the lexical manifold.

### 4. No Systematic Accuracy-vs-Compute Characterization Across Backends

The prior papers each report a single fixed-K configuration as their main result (Coconut: c=2/6 thoughts; CODI: 6 thoughts; SIM-CoT: K up to 8). There is no systematic study of how the accuracy-vs-token-budget Pareto frontier compares across backends at matched latent counts. The option-a branch fills this gap, providing the empirical foundation to argue that adaptive-K (develop-c) targets the frontier rather than a fixed operating point.

### 5. Interpretability of Halting Decisions

When K is fixed, the auxiliary decoder in SIM-CoT can visualize all K steps deterministically. With adaptive halting, the question becomes whether the steps decoded before halting are semantically coherent stopping points (i.e., the model halts having completed a full reasoning chain) or premature exits (halting before the answer is encoded). The interpretability infrastructure from SIM-CoT - reusing the training decoder to project each z_k - is directly applicable for this diagnostic and represents a novel contribution to the interpretability of adaptive computation in LLMs.
