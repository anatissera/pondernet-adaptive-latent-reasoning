# SIM-CoT: Supervised Implicit Chain-of-Thought

**Authors:** Xilin Wei¹², Xiaoran Liu¹⁴, Yuhang Zang²✉, Xiaoyi Dong², Yuhang Cao², Jiaqi Wang²⁴✉, Xipeng Qiu¹⁴, Dahua Lin²³

¹ Fudan University, ² Shanghai AI Laboratory, ³ The Chinese University of Hong Kong, ⁴ Shanghai Innovation Institute

📧 xlwei24@m.fudan.edu.cn · zangyuhang@pjlab.org.cn  
🔗 GitHub: https://github.com/InternLM/SIM-CoT

**arXiv:** 2509.20317v2 [cs.CL] - 25 Sep 2025

---

## Abstract

Implicit Chain-of-Thought (CoT) methods offer a token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited their adoption. We identify a core **latent instability issue** when scaling the computational budget of implicit CoT: as the number of reasoning tokens increases, training often becomes unstable and collapses. Our analysis shows that this instability arises from latent representations becoming homogeneous and losing semantic diversity, caused by insufficient step-level supervision in current implicit CoT methods. To address this, we propose **SIM-CoT**, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring latent states capture distinct and meaningful information. The auxiliary decoder is removed at inference, preserving the efficiency of implicit CoT with no added overhead. It also provides interpretability by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization and diagnosis. SIM-CoT significantly improves both in-domain accuracy and out-of-domain stability of implicit CoT methods, boosting Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. It further surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3× greater token efficiency, while closing the performance gap on larger models like LLaMA-3.1 8B.

---

## 1 Introduction

*"Measure what is measurable, and make measurable what is not so."* - Galileo Galilei

The strong reasoning capabilities of Large Language Models (LLMs) (OpenAI, 2024; Google, 2024; Anthropic, 2024) are often unlocked through explicit Chain-of-Thought (CoT) prompting (Wei et al., 2022). The explicit CoT approach enables LLMs to solve complex problems in a step-by-step manner, yielding high performance in domains like mathematics and programming (Guo et al., 2025; Muennighoff et al., 2025). Despite its advantages, explicit CoT also faces several limitations. Explicit CoT approaches must verbalize intermediate thoughts from a fixed vocabulary, thereby precluding the exploration of alternative solution paths (Li et al., 2025; Zhang et al., 2025b). Additionally, the generation of extensive intermediate sequences significantly increases inference cost and can result in redundant over-thinking steps or unnecessary verbosity (Chen et al., 2024b).

To address the flexibility and efficiency issues of explicit CoT methods, recent **implicit CoT** approaches (Hao et al., 2025; Zhang et al., 2025b; Li et al., 2025) have been proposed by representing reasoning in a continuous latent space rather than as a sequence of discrete text tokens. The implicit CoT methods allow each latent representation to encode richer information than a single explicit reasoning token, often with a significantly smaller number of latents than the length of an explicit reasoning chain. Early representative implicit work like Coconut (Hao et al., 2025) improves efficiency while still capturing useful intermediate structure. More recent approaches, such as CODI (Shen et al., 2025b), further apply trajectory-level distillation from explicit reasoning paths to enhance performance.

Despite these advancements, a **performance gap** still exists between existing implicit CoT methods and their explicit counterparts. The implicit CoT approaches are *fast*, *token-efficient* but *less accurate*, which currently limits their broader application.

To narrow the performance gap, inspired by the success of explicit CoT that scales computational budget for better performance, we explore a similar strategy for implicit CoT methods by increasing the number of implicit tokens. However, in Fig. 1(a), we reveal one underlying **latent instability issue** in current implicit CoT approaches. As we extend the number of implicit tokens from the default three (Hao et al., 2025) to five, the training process initially improves accuracy but becomes unstable and sometimes collapses entirely. To interpret the **latent instability issue**, we analyze implicit tokens from models trained on math reasoning data GSM8K-Aug (Deng et al., 2024). We follow previous works (Hao et al., 2025; Deng et al., 2024) to project the implicit tokens through the LM head and examine their top decoded tokens for analysis.

As shown in Fig. 1(b), failed models tend to collapse into homogeneous latent states. While successful reasoning requires capturing both numerical and operator information, the implicit tokens of failed models primarily represent numbers, almost completely losing the critical operator information. Fig. 1(c) further demonstrates that a model's collapse is accompanied by two changes: a reduction in the inter-latent distance and a drift of the latent states away from the central vocabulary embedding space. The latent representations of failed models become too similar and lose their semantic connection to the tokens they are meant to represent. Fig. 1(d) provides an example of the semantic homogenization. A normal model (top) maintains a large distance between its two latent tokens, allowing them to capture distinct information for numbers and operators. In contrast, a failed model (bottom)'s latent tokens become homogeneous, with both states decoding to similar information, primarily numbers.

Our observation (Fig. 1) reveals the reasons for the latent instability issue: a lack of sufficient step-level supervision for existing implicit methods to maintain the rich and varied internal representations.

Without stronger guidance, the latent space collapses, losing its diversity and making it impossible to reliably encode the distinct, step-level reasoning needed for complex reasoning tasks. Motivated by our findings, we propose **Supervised IMplicit-CoT (SIM-CoT)**, a plug-and-play module that introduces step-level supervision for implicit CoT approaches to alleviate the latent instability issue. Instead of supervising only the final answer (Hao et al., 2025) or the trajectory (Shen et al., 2025b), SIM-CoT uses an auxiliary decoder to align each implicit token with its corresponding explicit reasoning step during training. The step-level supervision for implicit tokens stabilizes optimization, prevents collapse, and ensures that latent tokens capture meaningful reasoning content. Crucially, because the auxiliary decoder is removed during inference, our approach incurs virtually no extra computational cost, making it as efficient as standard implicit CoT approaches. Beyond *accuracy*, *stability*, and *efficiency*, the auxiliary decoder also affords *interpretability* of implicit reasoning. During training, it defines a projection from latent tokens to the explicit reasoning vocabulary, enabling us to decode each latent step into a human-interpretable summary for verification or error diagnosis.

Experiments show that SIM-CoT acts as a plug-and-play module that boosts both accuracy and stability. We show that SIM-CoT can be effortlessly combined with various implicit CoT approaches such as Coconut (Hao et al., 2025), CODI (Shen et al., 2025b), and training-free approaches (Zhang et al., 2025b) to further enhance reasoning performance. On GPT-2, SIM-CoT surpasses both the strong explicit baseline (supervised fine-tuning on explicit CoT data) by 2.1%, and outperforms existing implicit methods Coconut and CODI by 8.2% and 4.3%, respectively. The performance trend holds as the method scales to larger models such as the LLaMA series. SIM-CoT achieves improvements over CODI of 3.4% (LLaMA-3.2 1B), 1.5% (LLaMA-3.2 3B), and 3.0% (LLaMA-3.1 8B), in addition to a 9.0% gain over Coconut on the LLaMA-3.2 1B model. Furthermore, while previous implicit CoT approaches (e.g., Coconut) collapse when scaled to 8 or 16 implicit tokens, SIM-CoT remains stable and continues to boost performance.

In summary, our contributions are as follows:
1. We provide a systematic analysis of the latent instability issue of implicit CoT approaches, showing that instability and collapse arise from insufficient supervision.
2. We introduce SIM-CoT, which applies step-level supervision to the model's implicit tokens. SIM-CoT not only integrates seamlessly with existing implicit CoT approaches and boosts performance with minimal inference overhead, but also affords interpretability of implicit reasoning by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization of semantic roles and diagnosis.
3. Through extensive experiments, we demonstrate that SIM-CoT not only improves accuracy in the in-domain dataset, but also generalizes effectively to out-of-domain datasets. The performance gains are consistent across a range of LLMs, including GPT-2 and recent LLaMA 3 models (1B, 3B, and 8B).

---

## 2 Analysis of Implicit CoT: The Latent Instability Issue

We first present an analysis (Fig. 1) of the limitations in implicit latent CoT approaches. We follow Coconut (Hao et al., 2025) and analyze implicit latents by projecting them through the LM head and examining the top-8 decoded tokens to understand the semantic and geometric properties.

**Latent Instability Issue.** Fig. 1(a) shows the training process of Coconut when the number of implicit latent tokens is progressively increased. Initially, as the number of latents increases from one to four, the model's accuracy generally improves, suggesting that using more latents can enhance performance. However, a significant drop in accuracy occurs when the number of latents is scaled to five, with performance collapsing to its worst point of 12.5%. The latent instability issue indicates that the implicit reasoning approach is sensitive to the choice of the number of latent tokens, as shown by the sharp drop and subsequent fluctuations in accuracy after adding the fifth latent.

**Information Loss.** Fig. 1(b) presents an analysis of how different levels of accuracy are affected by the number of latent tokens, using accuracy metrics at three levels: number, operator, and answer. The bar chart reveals a clear trend: as the number of latent tokens increases from 1 to 5, there is a general decline in performance across all three metrics, especially for the operator accuracy. The strong correlation between increased latent tokens and declining performance, particularly the sharp fall during failure, suggests that implicit latents do not consistently capture the necessary compositional reasoning process without more explicit, fine-grained supervision.

**Shifted Distance.** Fig. 1(c) examines the geometric properties of the latent representations during training. Two metrics are analyzed: the Latent Distance (red), which measures the average distance between pairs of latent vectors, and the Vocab Distance (blue), which measures the average distance from each latent vector to the center of the vocabulary embedding space. When the latent CoT model collapses, the latent distance decreases sharply, indicating that the latent vectors are collapsing and becoming nearly identical, losing their distinctiveness. Simultaneously, the vocab distance increases, showing that these collapsing latents are drifting away from the main lexical embedding space and are no longer grounded in the fundamental token representations used by the model.

**Semantic Homogenization.** Fig. 1(d) provides a qualitative analysis of the content of the latent tokens in a normal case versus a failed model. In the normal implicit model (middle), the decoded tokens from the latents are diverse and meaningful. In the failed implicit model (bottom), the semantic content of the latents becomes highly homogeneous. Latent 1 and Latent 2 contain mainly numbers, lacking operators or symbolic information needed for calculation. This shows that successful training produces latents with step-wise reasoning, while without explicit supervision, the latent space collapses into uniform numerical forms.

**Summary.** Our analysis across Fig. 1(a-d) highlights a crucial trade-off between diversity and stability. When the model collapses, it loses both its diversity (as the latents become too similar) and its stability (as the latents move away from the token space), leading to catastrophic information loss and a complete failure of the reasoning process, as shown by the sharp drop in overall accuracy. These combined findings show that without proper guidance, the latent space degenerates, losing its ability to represent distinct reasoning steps. These challenges motivate our proposed method, which introduces **step-level implicit supervision** to stabilize the training process and enrich unique semantic content of each latent, all while maintaining efficiency during inference.

---

## 3 Methodology

**Overview.** As shown in Fig. 2, early implicit reasoning studies differ mainly in supervision granularity: **Coconut** (top left) uses answer-level supervision, while **CODI** (top right) introduces trajectory-level signals via distillation. Both remain coarse and do not tell the model which latent should encode which step. We propose **SIM-CoT**, which provides **step-level implicit supervision**: During an **implicit phase**, the LLM runs for a fixed number K of reasoning steps; at each step k it takes the **last hidden state** as the implicit latent z_k and appends it to the sequence as the next "token" vector. After K steps, the model switches back to **explicit** decoding over the vocabulary to generate the final answer. A decoder is used only in training to align each z_k with the textual content of the k-th reasoning step; at inference, the decoder is removed, so the runtime is essentially that of direct answer generation plus K forward positions, which is far shorter than explicit CoT token lengths.

### 3.1 Notation

Let V be the vocabulary and E ∈ ℝ^{|V|×d} the token embedding matrix. A question is x = (x₁, ..., x_T) ∈ V^T with embedded prefix

U^{(0)} = (e(x₁), ..., e(x_T)), e(·) ∈ ℝ^d.

We run an autoregressive LLM F_θ on any prefix U = (u₁, ..., u_m) of d-dimensional vectors (tokens or latents). Denote the last-layer hidden state at the final position by

H_θ(U) ∈ ℝ^d.

For supervision, the k-th textual step is s_k = (y_{k,1}, ..., y_{k,L_k}) ∈ V^{L_k}, and the answer is a = (a₁, ..., a_{L_a}) ∈ V^{L_a}. The auxiliary decoder has parameters φ; the LLM has parameters θ.

### 3.2 Implicit Phase: Latent Construction by Last Hidden States

We fix the number of implicit reasoning steps K in advance. For each step k = 1, ..., K,

z_k = H_θ(U^{(k-1)}) ∈ ℝ^d,   U^{(k)} = U^{(k-1)} ⊕ z_k,   (1)

where ⊕ denotes concatenation along the time axis. The implicit chain-of-thought is therefore represented as a continuous sequence of hidden states z_{1:K} = (z₁, ..., z_K), which are autoregressively generated and appended to the context before the model switches to explicit decoding.

### 3.3 Explicit Phase: Answer Decoding over the Vocabulary

After constructing the implicit latents z_{1:K}, the model switches to explicit decoding to generate the final answer. Let W_o ∈ ℝ^{|V|×d} be the output projection (LM head). With teacher forcing on the partial answer a_{<t}, the generation is

h_{T+K+t} = H_θ(U^{(K)} ⊕ e(a_{<t})),   (2)

p_θ(a_t | x, z_{1:K}, a_{<t}) = softmax(W_o h_{T+K+t})_{a_t},   (3)

p_θ(a | x, z_{1:K}) = ∏_{t=1}^{L_a} p_θ(a_t | x, z_{1:K}, a_{<t}).   (4)

### 3.4 Training-time Decoder and Step-Level Supervision

During training, a decoder p_φ (architecturally identical to the LLM) takes only the k-th implicit latent z_k as conditioning signal and autoregressively generates the k-th textual step s_k = (y_{k,1}, ..., y_{k,L_k}). This provides **step-level** supervision that directly grounds z_k to its corresponding reasoning content:

p_φ(s_{1:K} | z_{1:K}) = ∏_{k=1}^{K} p_φ(s_k | z_k) = ∏_{k=1}^{K} ∏_{t=1}^{L_k} p_φ(y_{k,t} | z_k, y_{k,<t}).   (5)

*Parameterization.* For step k, the decoder is conditioned on the implicit latent z_k obtained from the LLM. Since z_k does not correspond to any token in the vocabulary, it is not included in the loss calculation. Instead, z_k is injected as an additional prefix vector that initializes the decoder's hidden state for step generation. Concretely, the decoder input sequence is

U_k^{dec} = [z_k; e(y_{k,1}), ..., e(y_{k,L_k})],

where e(·) denotes the embedding function of the LLM shared between both models. During training with teacher forcing, the decoder predicts each token y_{k,t} autoregressively:

p_φ(y_{k,t} | z_k, y_{k,<t}) = softmax(W^{dec} h_{k,t}^{dec})_{y_{k,t}},

where h_{k,t}^{dec} is the decoder hidden state at position t and W^{dec} is the LM head of the decoder. The training loss for step k is then

L_{step,k} = -∑_{t=1}^{L_k} log p_φ(y_{k,t} | z_k, y_{k,<t}),

which supervises only the textual step tokens. The decoder is used exclusively for this supervision during training and is discarded at inference.

### 3.5 Objectives

Training involves two complementary cross-entropy losses: one for supervising the textual steps through the decoder, and one for supervising the final answer through the base LLM.

**Step-level supervision.** For each implicit latent z_k, the decoder p_φ generates the corresponding reasoning step s_k = (y_{k,1}, ..., y_{k,L_k}). Since z_k is not a vocabulary token, the loss is computed only over the textual step tokens:

L_{step} = -∑_{k=1}^{K} ∑_{t=1}^{L_k} log p_φ(y_{k,t} | z_k, y_{k,<t}).   (6)

This loss grounds each latent z_k to a specific reasoning step, ensuring that the latent sequence carries fine-grained semantics.

**Answer supervision.** After K implicit steps, the LLM F_θ switches back to explicit decoding to generate the final answer a = (a₁, ..., a_{L_a}). We optimize the standard language modeling loss:

L_{ans-lm} = -∑_{t=1}^{L_a} log p_θ(a_t | x, z_{1:K}, a_{<t}).   (7)

**Total objective.** The overall loss is a weighted sum:

L = λ_{step} L_{step} + λ_{lm} L_{ans-lm}.   (8)

Gradients from L_{step} propagate through the decoder into the latent representations z_{1:K} and further into the LLM (via Eq. equation 1), shaping the hidden states to encode step-level reasoning. Meanwhile, L_{ans-lm} trains the base model to produce the final answer directly, so the decoder can be discarded at inference time without affecting efficiency. Implementation details, inference procedures, and diagnostic analyses are provided in Appendix F.

---

## 4 Experiment

### 4.1 Experimental Setup

**Training Data.** We follow previous works (Deng et al., 2024; Hao et al., 2025) to use the **GSM8k-Aug** dataset (Deng et al., 2024) for training implicit CoT models. The GSM8k-Aug expands the original GSM8k training set (Cobbe et al., 2021) to 385k examples by using GPT-4 for data generation. To facilitate implicit CoT training, the GSM8k-Aug removes the reasoning chain of natural language, preserving only a sequence of structured mathematical expressions. Each expression is logically linked to the previous step, as illustrated by the example: `<<12*3=36>><<9*2=18>><<17*2=34>><<36+18+34=88>>`.

**Evaluation Benchmarks.** We report results on the **GSM8k-Aug** test set (Cobbe et al., 2021), which serves as our in-domain (ID) evaluation benchmark. To further evaluate mathematical reasoning under a distribution shift, we also evaluate models on three out-of-domain (OOD) benchmarks: (1) **SVAMP** (Patel et al., 2021), a dataset of grade-school arithmetic word problems that introduces simple variations to assess robustness; (2) **GSM-Hard** (Gao et al., 2022), a modified version of the GSM8k test split where numbers are replaced with larger magnitudes to increase problem difficulty; and (3) **MultiArith** (Roy & Roth, 2015), a subset of MAWPS (Koncel-Kedziorski et al., 2016) consisting of multi-step arithmetic word problems.

**Implementation Details.** We follow the training setup of previous works (Hao et al., 2025; Shen et al., 2025b), and adopt consistent hyperparameter choices for GPT-2, LLaMA 1B/3B/8B. Detailed configurations, such as learning rates, curriculum strategies, are provided in Section E.

### 4.2 Main Results

**Baselines.** We compare our SIM-CoT against five representative baselines:
1. **CoT-SFT**: Supervised fine-tuning (SFT) on CoT-annotated data, where the model is trained to generate explicit intermediate reasoning steps followed by the final answer.
2. **No-CoT-SFT**: Supervised fine-tuning on direct answers only, without producing intermediate reasoning steps.
3. **iCoT** (Deng et al., 2024): A curriculum learning method based on "Stepwise Internalization," which injects CoT reasoning patterns into the model's internal representations, enabling it to produce more accurate direct answers during inference.
4. **Coconut** (Hao et al., 2025): A curriculum learning approach that gradually replaces explicit reasoning steps with implicit tokens until the reasoning process becomes fully implicit. This method has shown strong empirical performance and serves as a primary baseline in our experiments.
5. **CODI** (Shen et al., 2025b): A distillation-based method where explicit CoT acts as the teacher and implicit CoT as the student. By aligning the last hidden states of the full reasoning trajectory, CODI effectively internalizes knowledge and alleviates catastrophic forgetting.

**In-Domain Math Benchmark Results.** Table 1 (first column) reports GPT-2 results on GSM8k-Aug. SIM-CoT outperforms SFT-CoT and is the first training-based approach where implicit CoT surpasses explicit CoT. With GPT-2 using Coconut as the backbone, it achieves a +2.1 point improvement over SFT-CoT. It also exceeds other training-based implicit reasoning models; for example, on Coconut, it improves by +8.2 points, a relative gain of 22.4%. Moreover, when applied on top of CODI-the current SOTA implicit reasoning method-SIM-CoT yields an additional +0.6 point improvement.

Table 2 (first column) shows the results when CODI is used as the backbone. In this setting, our method achieves a substantial +3.4 point improvement. Furthermore, we are the first to achieve performance comparable to SFT-CoT on LLaMA-1B, reaching 96% of its accuracy.

**Out-of-Domain Math Benchmark Results.** To evaluate the robustness of our method, we train on GSM8k and evaluate on out-of-domain datasets (GSM-Hard, MultiArith, and SVAMP). From the third column of Table 1, we observe that SIM-CoT consistently outperforms SFT-CoT, with an average improvement of +4.3 points when using Coconut as the backbone. From the third column of Table 2, our method further improves upon the current SOTA implicit reasoning method CODI by +1.0 point. Moreover, when scaling model size from GPT-2 to LLaMA-1B, SIM-CoT enlarges the performance gap against iCoT, Coconut, and other baselines.

We attribute the robustness of SIM-CoT to its step-level implicit supervision. Unlike SFT-CoT, which forces the model to mimic deterministic natural language annotations, and unlike CODI, which applies trajectory-level alignment to a coarse-grained reasoning path, our method introduces a moderate form of supervision. This design ensures the plausibility of each reasoning step while preserving the diversity of reasoning trajectories, thereby improving generalization to unseen inputs.

**Inference Efficiency.** In terms of inference speed, our method maintains the same efficiency as other implicit reasoning approaches on both GPT-2 and LLaMA-1B. On GPT-2, SIM-CoT not only surpasses SFT-CoT on both in-domain and out-of-domain benchmarks, but also achieves a 2.3× speedup on Coconut, and 2.2× speedup on Coconut, respectively. On LLaMA-1B, SIM-CoT remains comparable to SFT-CoT in accuracy while delivering 1.9× and 1.7× speedups on in-domain and out-of-domain benchmarks, respectively.

### 4.3 Ablation Studies

**Ablation on the Number of Implicit Tokens.** We study the effect of varying the number of implicit latents on GPT-2, comparing SIM-CoT with Coconut trained on GSM8k-Aug and evaluated on GSM8k-Aug, GSM-Hard, MultiArith, and SVAMP (Fig. 3). Following Coconut, each latent corresponds to two tokens. As shown in Fig. 5, most problems involve two to six steps with a small proportion of harder cases, so we set the maximum number of implicit latents to 8. For each configuration, we report the best performance, and results show that SIM-CoT provides more stable training and achieves consistent gains over Coconut, indicating that step-level implicit supervision scales effectively with larger latent capacity.

**Ablation on Scaling to Larger Backbones.** To examine robustness and scalability, we extend experiments to larger LLaMA backbones, including LLaMA 3.2 3B and LLaMA 3.1 8B. Table 3 reports results on GSM8k-Aug (in-domain) and GSM-Hard, MultiArith, and SVAMP (out-of-domain). Overall, SIM-CoT scales effectively to larger backbones, consistently surpassing or matching explicit CoT on out-of-domain tasks while reducing reliance on trajectory-level supervision.

**Ablation on Different Decoder Sizes.** We replace the decoder of LLaMA 1B with larger ones from the same vocabulary family and evaluate on GSM8k-Aug, GSM-Hard, MultiArith, and SVAMP. A 1B-scale decoder yields consistent gains, whereas larger decoders (3B or 8B) slightly reduce accuracy; detailed discussion is in Appendix B.

**Ablation on Soft Thinking.** We also study the effect of integrating soft thinking (Zhang et al., 2025b; Wu et al., 2025) with both Coconut and SIM-CoT.

**Interpretability of Implicit Reasoning.** Continuous thoughts in implicit reasoning models are not aligned with vocabulary tokens and cannot be directly decoded into human-readable text. We address this by reusing the training decoder to project and visualize the semantic meaning of each latent step as shown in Fig. 4. We also analyze latent token distances under different configurations. Detailed descriptions, numerical results, and examples are provided in Appendix I.

---

## 5 Related Work

A large body of work has studied explicit chain-of-thought (CoT) prompting, including self-consistency (Wei et al., 2022; Wang et al., 2023), least-to-most prompting (Zhou et al., 2023), reflection-based reasoning (Shinn et al., 2023; Madaan et al., 2023), and the integration of external tools (Yao et al., 2023b). Other work investigates step-level supervision to structure explicit reasoning (Zheng et al., 2023; Wei et al., 2025a). While effective, explicit CoT increases inference cost with longer sequences and often produces redundant steps, limiting efficiency and reasoning diversity (Li et al., 2025; Zhang et al., 2025b; Xu et al., 2025).

Implicit CoT aims to reduce output length while retaining multi-step reasoning. Prior work explores knowledge internalization (Deng et al., 2024), architectural modification (Saunshi et al., 2025; Chen et al., 2025; Cheng & Van Durme, 2024; Su et al., 2025; Mohtashami et al., 2023; Geiping et al., 2025), training-free latent construction (Zhang et al., 2025b; Wu et al., 2025), and auto-regressive latent reasoning (Xu et al., 2025; Tan et al., 2025). Coconut applies answer-level supervision (Hao et al., 2025), and CODI uses trajectory-level distillation (Shen et al., 2025b). Our work introduces step-level supervision, which distributes signals across latent steps and improves stability.

---

## 6 Conclusion

We introduce SIM-CoT, a training-based implicit reasoning method with step-level supervision on latent tokens. On GPT-2, SIM-CoT outperforms the strong explicit baseline SFT-CoT, while also surpassing implicit baselines such as Coconut and CODI. When scaling to larger LLaMA backbones, the performance achieves consistent gains over existing implicit reasoning methods and maintains fast inference efficiency. Ablation studies further show that it improves training stability with more latent tokens and can benefit from integration with training-free techniques such as soft thinking. Distance analysis confirms that SIM-CoT produces latent representations that are diverse yet stable.

---

## References

*(Selected key references)*

- Cobbe et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
- Deng, Choi & Shieber (2024). From explicit cot to implicit cot: Learning to internalize cot step by step. ArXiv, abs/2405.14838.
- Hao et al. (2025). Training large language models to reason in a continuous latent space. In COLM, 2025.
- Shen et al. (2025b). Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv:2502.21074.
- Wei et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. In NIPS, 2022.
- Zhang et al. (2025b). Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv:2505.15778.

---

## Appendix

### A Additional Analysis on Scaling to Larger Backbones

**On LLaMA 3.2 3B**, SIM-CoT improves over CODI by +1.5 points on GSM8k-Aug and +1.6 points on SVAMP, while maintaining comparable performance on GSM-Hard and MultiArith.

**On LLaMA 3.1 8B**, SIM-CoT yields +3.0 points on GSM8k-Aug, +1.3 points on SVAMP, and +0.8 points on MultiArith relative to CODI, with stable accuracy on GSM-Hard. Compared with SFT-CoT, it achieves higher accuracy on MultiArith (100.0 vs. 98.3) and SVAMP (79.4 vs. 73.1), while remaining similar on GSM-Hard.

### B Additional Analysis on Decoder Sizes

Compared to the baseline, integrating a 1B-scale decoder leads to consistent improvements across all benchmarks. However, simply replacing the decoder with larger versions (3B or 8B) does not bring further benefits and slightly degrades performance. These results suggest that moderate decoder scaling can enhance reasoning ability, but excessively large decoders may introduce optimization difficulties or misalignment with the backbone, ultimately limiting generalization.

### C Additional Analysis on Soft Thinking

Soft thinking (Zhang et al., 2025b; Wu et al., 2025) is a training-free method for implicit reasoning in which the latent space is represented as a weighted average over the vocabulary embedding space.

**C.1 Setup.** We apply the proposed soft thinking mechanism on top of both Coconut and SIM-CoT, while adopting GPT-2 as the backbone model. The in-domain evaluation is carried out on GSM8k-Aug.

**C.2 Results.** Table 5(b) reports the results. Adding soft thinking improves accuracy in most cases. For Coconut, improvements are observed on GSM-Hard (+0.2) and MultiArith (+1.7), with a slight decrease on SVAMP (-0.2). For SIM-CoT, soft thinking consistently enhances performance: GSM8k-Aug (+0.2), GSM-Hard (+0.1), MultiArith (+0.7), and SVAMP (+0.1).

**C.3 Formulation.**

Step 1. **Vocabulary distribution.** The continuous latent token z is first mapped into a probability distribution over the vocabulary space: p = softmax(Wz).

Step 2. **Soft-thinking embedding.** Using the distribution p, we compute a weighted mixture of vocabulary embeddings: z_{soft} = E^T p = ∑_{v∈V} p_v E_v.

Step 3. **Combination.** Finally, we combine the original continuous latent z with the soft-thinking embedding z_{soft}: z' = αz + βz_{soft}, where α = continuous_weight and β = soft_weight are hyperparameters.

### D Related Work (Extended)

**Explicit chain-of-thought reasoning.** Chain-of-thought (CoT) prompting enables large language models (LLMs) to generate intermediate reasoning steps before producing the final answer (Wei et al., 2022). Methods such as self-consistency (Wang et al., 2023), least-to-most prompting (Zhou et al., 2023), and reflection-based reasoning (Shinn et al., 2023; Madaan et al., 2023) have been widely studied and extended.

**Implicit chain-of-thought reasoning.** Implicit CoT performs multi-step computation in a continuous latent space instead of emitting long textual traces, reducing decoded length while keeping internal structure. Prior work follows four practical routes:
1. **Knowledge internalization** trains models to carry out reasoning internally by progressively removing explicit traces or by using dedicated control embeddings.
2. **Architectural modification** controls compute by reusing or skipping layers, or by adding light recurrence, so models can refine hidden states without lengthening outputs.
3. **Training-free** methods construct continuous latents directly from the model's probability distribution over the vocabulary. Soft Thinking mixes embeddings by probability to form "concept" tokens.
4. **Auto-regressive latent reasoning** updates and concatenates latent states in place of some token-level decoding. Coconut applies **answer-level** supervision. CODI adds **trajectory-level** distillation. Our framework remains in the auto-regressive setting but changes the supervision: during training, each latent is aligned with its corresponding textual step (**step-level** supervision).

### E Implementation and Training Details

We provide the full hyperparameter settings, training procedures, and additional analysis used in our experiments. Unless otherwise specified, we use the Adam optimizer with β₁ = 0.9, β₂ = 0.999, and weight decay of 0.1. Batch size is set to 128 for GPT-2 and LLaMA 1B, and 64 for LLaMA 3B and 8B. Early stopping is applied with a patience of 3 epochs.

**E.1 Coconut Training Setup.** Following Hao et al. (2025), GPT-2 and LLaMA 1B are trained with a fixed learning rate of 1 × 10⁻⁴. One implicit latent corresponds to two implicit tokens. A curriculum is applied: every three epochs, one explicit reasoning step is replaced by an implicit latent until the maximum number of latent steps is reached. After this expansion, training continues for 15 additional epochs.

**E.2 CODI Training Setup.** For larger backbones such as LLaMA 3B and 8B, we adopt task-specific hyperparameter settings to ensure stable training. In particular, we use a learning rate of 3 × 10⁻⁴ for LLaMA 3B and train for 8 epochs, while for LLaMA 8B the learning rate is reduced to 1 × 10⁻⁴ with 6 training epochs.

**E.3 Summary of Hyperparameters.**

| Model     | Method  | LR           | Epochs          |
|-----------|---------|--------------|-----------------|
| GPT-2     | Coconut | 1×10⁻⁴       | 15 + curriculum |
| LLaMA 1B  | Coconut | 1×10⁻⁴       | 15 + curriculum |
| GPT-2     | CODI    | 3×10⁻³       | 40              |
| LLaMA 1B  | CODI    | 8×10⁻⁴       | 10              |
| LLaMA 3B  | CODI    | 3×10⁻⁴       | 8               |
| LLaMA 8B  | CODI    | 1×10⁻⁴       | 6               |

### F Training and Inference Details

**Curriculum for K.** We use a curriculum schedule to gradually increase the number of implicit steps. Each latent corresponds to two implicit tokens. Let K_{max} denote the maximum number of latents. Starting from K^{(0)} = 0, the number of implicit steps after epoch e is

K^{(e)} = min(K_{max}, ⌊e / Δe⌋),

where Δe is the update interval in epochs. Once K^{(e)} reaches K_{max}, it remains fixed for the remainder of training.

**Inference and Efficiency.** At inference time, the auxiliary decoder is removed and only the base model is executed:

U^{(0)} = [e(x₁), ..., e(x_T)], for k=1,...,K: z_k = H_θ(U^{(k-1)}), U^{(k)} = U^{(k-1)} ⊕ z_k,

and after K implicit steps the model switches to explicit decoding to generate the final answer sequence a. The total decoding length is T + K + L_a.

**Benchmark Details.**
- **SVAMP** (Patel et al., 2021): 1,000 elementary-level arithmetic word problems (grade 4 and below), each involving a single unknown and solvable by an arithmetic expression with no more than two operators.
- **GSM-Hard** (Gao et al., 2022): ~1,319 examples matching the GSM8K test set size. Retains the same problem statements as GSM8K but replaces numbers with larger numerical values.
- **MultiArith** (Roy & Roth, 2015): 600 multi-step arithmetic word problems designed to challenge systems to correctly sequence multiple operations.
- **GSM8K-Aug** (Deng et al., 2024): The only training corpus used. An augmented dataset derived from GSM8K, expanding the original 8.5k training problems to ~385k examples.

### G SIM-CoT Training Implementation

We provide pseudocode for the SIM-CoT training process, which illustrates how continuous latent embeddings are aligned with explicit supervision at the step level.

```
Algorithm 2: SIM-CoT Training Procedure
Require: Batch size b, number of thoughts C, continuous embeddings Z,
         tokenized inputs X, embedding matrix E
 1: for each thought t = 1,...,C do
 2:   for each sample i = 1,...,b do
 3:     Extract continuous embeddings z_{i,t} from Z
 4:     Obtain token embeddings e_{i,t} from E(X_{i,t})
 5:     Concatenate embeddings: h_{i,t} ← [z_{i,t}; e_{i,t}]
 6:     Build attention mask m_{i,t} up to EOS
 7:     Assign position ids p_{i,t}
 8:     Prepare labels y_{i,t} with masked tokens set to -100
 9:   end for
10: end for
11: Pad and stack {h,m,p,y} to maximum sequence length
12: Prepare 4D attention mask: M̂ ← PrepareMask(M)
13: Forward pass: Ô ← ExplainableLLM(H, M̂, P)
14: Extract logits: L ← Ô.logits
15: Shift logits and labels: L' ← L[:,:-1], Y' ← Y[:,1:]
16: Compute cross-entropy loss: ℓ = CrossEntropy(L', Y')
17: Normalize ℓ over valid positions
Ensure: Final training loss ℓ
```

### H Geometric Diagnostics of the Latent Space

We analyze the geometry of latent representations with two metrics.

**Inter-latent distance:**

Dist(z_{1:K}) = (2 / K(K-1)) ∑_{1≤i<j≤K} ‖z_i - z_j‖₂.   (9)

A larger value indicates better separation, reducing the risk of collapse.

**Distance to vocabulary center.** Let μ = (1/|V|) ∑_{v∈V} E_v denote the mean embedding. Then,

DistVC(z_{1:K}) = (1/K) ∑_{k=1}^{K} ‖z_k - μ‖₂.   (10)

Moderate values indicate that latents remain close enough to the lexical manifold for stability, while avoiding collapse toward the center.

### I Additional Details for Interpretability Analysis

**I.1 Making Implicit Reasoning Visible.**

Continuous thoughts produced by implicit reasoning models are represented as latent embeddings that do not correspond to discrete vocabulary tokens. To address this, we reuse the decoder that was employed for step-level supervision during training, and apply it at inference time to map each latent embedding into a human-readable token sequence.

As illustrated in Figure 4, the process begins with a natural language problem that is embedded and passed into the large language model. The model generates a sequence of implicit latent tokens, which capture intermediate reasoning steps in continuous space. These latent tokens are then fed into the optional decoder, which translates them into interpretable expressions. Each latent corresponds to one reasoning step, and the autoregressive generation order encodes the dependency structure across steps.

For example, in the GSM8k case study shown in the figure, the first latent is decoded as 0.3×120 = 36, representing the number of watermelons harvested initially. The sequence of decoded latents mirrors the logic of explicit chain-of-thought reasoning, while being produced implicitly within the latent space.

**I.2 Distance Analysis.** As the number of latent tokens increases from 1 to 5, Dist gradually grows from 20.30 to 28.34, suggesting that the model distributes the latent representations more sparsely in the embedding space. However, in the failed case with 5 latents, Dist collapses to 4.21, reflecting a degeneration where all latent tokens converge to nearly identical points. By contrast, SIM-CoT pushes Dist to 32.81 under the same configuration, showing that step-level supervision effectively enforces stronger separation and prevents collapse.

**I.3 Summary.** Overall, the results demonstrate that SIM-CoT establishes a balance between **diversity** and **stability** in the latent space. Larger inter-latent distances mitigate representation collapse, while moderate distances to the vocabulary center prevent excessive drift. This equilibrium supports stable implicit reasoning and provides robustness when scaling to more latent tokens.

### J Future Directions

**(1) Multimodal extension.** Incorporating intermediate supervision from images or videos could generalize implicit CoT to multimodal reasoning tasks.

**(2) Multi-path implicit reasoning.** Exploring branched latent structures, inspired by Tree-of-Thought methods, may enhance the diversity and robustness of implicit reasoning.

**(3) Integration with RLHF.** Step-level supervision can complement preference optimization and reinforcement learning, leading to a more reliable and adaptive reasoning framework.

**(4) Theoretical foundations.** Future work may also study implicit reasoning from the perspectives of information theory and representation learning, providing a principled explanation for why step-level supervision stabilizes the latent space.

### K Additional Case Studies on GSM8k

In practice, implicit reasoning continues to produce latent tokens even after the correct answer has been reached. These trailing latents no longer introduce new steps but simply repeat the final prediction. The decoded reasoning steps consistently match the semantic structure of explicit chain-of-thought annotations, while being generated implicitly within the latent space. The final predictions align with the ground-truth answers, demonstrating that SIM-CoT is capable of encoding interpretable and step-ordered reasoning without requiring explicit supervision at inference time.
