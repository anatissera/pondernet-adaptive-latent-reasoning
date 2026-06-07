# CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation

**Authors:** Zhenyi Shen¹, Hanqi Yan¹, Linhai Zhang¹, Zhanghao Hu¹, Yali Du¹², Yulan He¹²

¹ King's College London · ² The Alan Turing Institute

📧 {zhenyi.shen, hanqi.yan, linhai.zhang, zhanghao.hu}@kcl.ac.uk · {yali.du, yulan.he}@kcl.ac.uk

🔗 Code: https://github.com/zhenyi4/codi

**arXiv:** 2502.21074v3 [cs.CL] — 23 Sep 2025

---

## Abstract

Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by encouraging step-by-step reasoning in natural language. However, leveraging a latent continuous space for reasoning may offer benefits in terms of both efficiency and robustness. Prior implicit CoT methods attempt to bypass language completely by reasoning in continuous space but have consistently underperformed compared to the standard explicit CoT approach. We introduce **CODI** (Continuous Chain-of-Thought via Self-Distillation), a novel training framework that effectively compresses natural language CoT into continuous space. CODI jointly trains a teacher task (Explicit CoT) and a student task (Implicit CoT), distilling the reasoning ability from language into continuous space by aligning the hidden states of a designated token. Our experiments show that CODI is the first implicit CoT approach to match the performance of explicit CoT on GSM8k at the GPT-2 scale, achieving a **3.1× compression rate** and outperforming the previous state-of-the-art by **28.2%** in accuracy. CODI also demonstrates robustness, generalizable to complex datasets, and interpretability. These results validate that LLMs can reason effectively not only in natural language, but also in a latent continuous space.

---

## 1 Introduction

Large Language Models (LLMs) have exhibited remarkable reasoning capabilities (OpenAI, 2024; Anthropic, 2024; Google, 2024), with Chain-of-Thought (CoT) (Wei et al., 2022) emerging as a key technique for enabling step-by-step reasoning. The success of CoT can be explained as it allows human-like deliberate thinking when computing a sequence of reasoning tokens before deriving the final answer (Kahneman, 2011).

However, conventional CoT-based methods only rely on natural language tokens as the medium for reasoning. While prior work on prompt learning (Lester et al., 2021) has demonstrated that transforming discrete prompts into continuous representations can lead to efficient yet effective reasoning (Li and Liang, 2021), this motivates us to investigate if CoT reasoning can similarly benefit from continuous representations. Compared to natural language, reasoning in continuous space offers the following advantages. First, verbalizing the reasoning process can be inefficient, as many tokens are devoted to communication rather than computation (Li et al., 2024b). Second, learning annotated CoTs token-by-token may cause models to overfit on superficial linguistic cues (Lin et al., 2025). While continuous representations—without the need to mimic explicit CoT targets—introduce a softer prior, which may lead to improved robustness.

An implicit CoT algorithm replaces natural language tokens with continuous representations for reasoning. To effectively learn these representations, the state-of-the-art method, Coconut (Hao et al., 2024) adopts a curriculum learning strategy (Deng et al., 2024) that gradually replaces the initial CoT tokens with continuous thoughts. This strategy encourages continuous thoughts to behave like the removed CoT tokens. Although Coconut has greatly improved upon earlier implicit CoT methods in terms of performance (Goyal et al., 2024; Deng et al., 2024), it lags behind CoT-SFT by a large margin. We hypothesize that this performance gap is due to forgetting across stages in the curriculum learning process (Rao Vijjini et al., 2021). This prompts us to ask: **Can implicit CoT methods achieve the reasoning capability comparable to CoT-SFT while maintaining their efficiency advantages?**

To address this, we propose a novel training framework: **CODI (Continuous Chain-of-Thought via Self Distillation)**. CODI enables implicit CoT learning in a single training step by leveraging self-distillation, thereby avoiding the forgetting issues inherent in curriculum learning. In doing so, it achieves performance comparable to CoT-SFT while being significantly more efficient. CODI enables implicit CoT reasoning through a joint learning setup involving a *teacher* task and a *student* task. The teacher learns from the annotated CoT tokens using a cross-entropy loss, while the student generates a small number of continuous thoughts before producing the final answer, representing implicit CoT reasoning. We do not constrain the student's continuous thoughts to match any specific target. Instead, we transfer the teacher's reasoning knowledge to the student through a form of representation alignment at the position of answer generation, where the essence of the reasoning process is captured (Orgad et al., 2025). This allows the student to effectively mimic the teacher's reasoning pattern in continuous space without rigid constraints. We refer to this mechanism as *self-distillation* (Wang et al., 2023; Gou et al., 2021), emphasizing the model's ability to distill one of its own behaviors into another.

**The main contributions are threefold:**
- We propose CODI, a novel self-distillation framework that enables LLMs to reason in a compact continuous space, providing an alternative to accelerate reasoning with high performance.
- We demonstrate the effectiveness of distilling knowledge from explicit CoT to implicit CoT by aligning the hidden activations of a single token.
- Extensive experiments show that CODI is robust, generalizable to complex CoT datasets, and offers a reasonable level of interpretability.

---

## 2 Related Work

**Implicit Chain-of-Thought Reasoning.** Implicit CoT methods aim to enhance reasoning without verbalizing intermediate steps as in CoT, thereby accelerating inference speed. Theoretical work (Strobl et al., 2024; Merrill and Sabharwal, 2024) establishes that additional computational tokens enhance transformers' reasoning capacity. Empirical studies (Pfau et al., 2024; Goyal et al., 2024) validate these insights by training LLMs with extra dummy tokens before answering. Recent efforts (Deng et al., 2023, 2024) distill CoT reasoning by fine-tuning. Coconut (Hao et al., 2024) reintroduces intermediate reasoning tokens via autoregressive hidden state propagation, combining curriculum learning from (Deng et al., 2024). CODI replaces curriculum learning with a novel self-distillation framework, enabling a single-step learning process that avoids forgetting issues.

**Knowledge Distillation.** Knowledge distillation (KD) (Gou et al., 2021; Xu et al., 2024) has emerged as a key strategy for transferring CoT reasoning capabilities from teacher to student models. Self-distillation (Yang et al., 2024; Dong et al., 2025) leverages self-distillation to preserve the model's original behavior, akin to the KL divergence loss used in RLHF (Ouyang et al., 2022). Our work is based on self-distillation framework, but further strengthens the teacher by providing it with richer input contexts, enabling the student to learn from it like knowledge distillation.

---

## 3 CODI: Continuous Chain-of-Thought via Self Distillation

Unlike traditional CoT reasoning, CODI bypasses autoregression in the vocabulary space, and directly connects the last hidden representation to the subsequent input. The key challenge in training a model with continuous thoughts lies in designing an appropriate training objective. Conventional reasoning learning in explicit CoT fine-tuning relies on a cross-entropy loss over annotated CoT tokens, which inevitably leads to discrete CoT token generation—contradicting the definition of implicit CoT.

### 3.1 Overview

CODI addresses this challenge by introducing a self-distillation framework (Figure 2) with two training tasks: a teacher task and a student task. The teacher task learns explicit CoT reasoning, while the student task learns implicit CoT reasoning.

The overall training objective is a weighted sum of three losses:

L = αL_{student} + βL_{KD} + γL_{teacher},   (1)

where α, β, and γ are hyperparameters controlling the balance among the objectives.

### 3.2 Teacher Task

The teacher task (Figure 2, right) learns explicit CoT using a cross-entropy loss:

L_{teacher} = -(1/N) ∑_{i=1}^{N} log P(r_i | r_{1:i-1}, Q),   (2)

where P denotes the output probability distribution of the LLM, Q represents the question tokens, and r = [c, y] is the concatenated sequence of the CoT reasoning tokens c and the final answer token y.

### 3.3 Student Task

The student task (Figure 2, left), which performs implicit CoT reasoning, generates continuous thoughts by autoregressively propagating the last hidden states. This process begins with a learnable `<bot>` (*begin-of-thoughts*) token and proceeds until a learnable `<eot>` (*end-of-thoughts*) token is reached. The model then learns the final answer from the `<eot>` token using a cross-entropy loss:

L_{student} = -(1/N) ∑_{i=1}^{N} log P(y_i | y_{1:i-1}, Q, Z),   (3)

where y denotes the answer label, Q the question tokens, and Z the continuous thoughts.

Additionally, a two-layer MLP followed by layer normalization transforms the hidden representations of continuous thought tokens before feeding them into the next step for the purpose of better discriminating the latent space and the token space.

### 3.4 Self-Distillation

If the model learns only with the student task, it benefits only marginally from the additional computation (Goyal et al., 2024) due to the absence of supervision for continuous thoughts.

**Distillation in Feature Space.** To provide explicit supervision to guide continuous thoughts, we adopt a feature-level distillation strategy. Recent work (Li et al., 2024a; Liu et al., 2023) demonstrates that in-context examples influence the final query token by shifting its hidden activation values. Extending this idea, we show that CoT tokens similarly induce a shift in hidden activation values of a query token (can be a probing token like "Answer") compared to a sequence without CoT, as formalized in Equation 4:

h^l_{CoT} ≈ h^l_{no-CoT} + f(W_V R(W_K R)^T q),   (4)

where q is the query token, h^l_{CoT} is the hidden activations at layer l with CoT, h^l_{no-CoT} is the corresponding activation without CoT, and the remaining term quantifies the shift introduced by the CoT rationale R. A formal proof of this "CoT shift" phenomenon is provided in Appendix B.

This decomposition suggests that the key information from CoT reasoning accessible to the query token is embedded in the shift term f(·). Therefore, by encouraging the student's hidden activations h^l_{student} to align with the teacher's h^l_{teacher}, we are able to transfer the reasoning capability from explicit CoT to implicit CoT.

**The Distilled Token.** Rather than aligning with all tokens in the query sentence, we select a *distillation token* for alignment. Inspired by the recent observations (Orgad et al., 2025) that the hidden activations of the token immediately preceding the answer, i.e., the colon (":") in the answer prompt "The answer is:" (as shown in Figure 2), encodes essential reasoning information. We select this token's hidden activations, **h**, for distillation.

**Loss Function.** As a result, we formulate a loss function that aligns the teacher's and student's hidden activations across all layers at the selected distillation token for the student's implicit CoT learning. To ensure a one-way flow of knowledge, we apply a stop-gradient operation on h^l_{teacher}, only allowing the teacher to influence the student:

L_{KD} = (1/M) ∑_{l=1}^{M} |sg[h^l_{teacher}] - h^l_{student}|,   (5)

where M indicates the number of layers in the LLM, sg denotes the stop-gradient operation, and h^l is the hidden activations of the LLM's l-th layer for the token position corresponding to the colon ":" in our design.

### 3.5 Training and Inference

**Training.** The continuous thoughts are generated dynamically during training. To achieve this, we decode them step by step, with a cache storing previous keys and values to maintain efficiency. We normalize each layer's hidden activations by dividing them by the standard deviation of the teacher's corresponding hidden activations within the current batch.

For the distillation task, we adopt the same model for both the teacher and student roles for two primary reasons. (1) **Reference Learning:** The model must first learn to perform explicit CoT reasoning before it can effectively compress and transfer this capability into continuous space as implicit CoT. (2) **Training Efficiency:** Maintaining two distinct models during training doubles memory consumption.

For training data, we exclude the final CoT step—the step responsible for generating the final answer—because including this step could allow the teacher's hidden activations to take a shortcut. Specifically, the model might directly copy the result from the last CoT step to the token responsible for generating the exact answer token, bypassing the reasoning process.

**Inference.** The inference process in CODI mirrors the student task during training (Figure 2, left). The model autoregressively decodes n continuous thoughts following the question and the `<bot>` token. Once the reasoning process is complete, the `<eot>` token is manually inserted to terminate continuous reasoning and switch the model to language generation mode, decoding the final answer.

---

## 4 Experiments

We demonstrate CODI's effectiveness in continuous space reasoning through experiments on mathematical and commonsense reasoning tasks.

### 4.1 Experimental Setup

**Training Data.** We utilize three datasets to train our models—GSM8k-Aug, GSM8k-Aug-NL, and CommonsenseQA-CoT.

**(1) GSM8k-Aug** (Deng et al., 2023): the dataset proven effective for training implicit CoT methods (Deng et al., 2024; Hao et al., 2024). This dataset extends the original GSM8k training set (Cobbe et al., 2021) to 385k samples by prompting GPT-4. To facilitate implicit CoT training, all natural language interleaving within the CoT is removed, leaving only structured mathematical expressions such as `<< 10÷5=2 >><<2×2=4>><<6×4=24>>`.

**(2) GSM8k-Aug-NL**, a version that preserves natural language explanations, to assess both the generalizability and effectiveness of our approach to compress more verbose CoTs.

**(3) CommonsenseQA-CoT** is derived from CommonsenseQA (Talmor et al., 2019), a multiple-choice QA dataset built from ConceptNet-based questions (Speer et al., 2017). As it lacks CoT annotations, we generate 8.1k CoT examples using GPT-4o-mini, filtered by correctness. The 1.2k-example validation set is used for evaluation.

**Evaluation Benchmarks for OOD.** For mathematical reasoning, we assess model robustness on three out-of-domain (OOD) benchmarks: (1) **SVAMP** (Patel et al., 2021), a dataset of grade-school arithmetic word problems with simple variations designed for robustness test; (2) **GSM-HARD** (Gao et al., 2023), a modified version of the GSM8k test split where numbers are replaced with values of larger magnitude to increase difficulty; and (3) **MultiArith** (Roy and Roth, 2015), a subset of MAWPS (Koncel-Kedziorski et al., 2016) containing multi-step mathematical word problems.

**Baselines.** We consider the following baselines:
1. **CoT-SFT**: Finetunes the model on CoT data, enabling it to generate intermediate steps followed by the final answer.
2. **No-CoT-SFT**: Finetunes the model using only direct answers, without generating intermediate steps.
3. **iCoT** (Deng et al., 2024): Implements a curriculum learning strategy called "Stepwise Internalization", which injects CoT's reasoning patterns into the model's internal states.
4. **Coconut** (Hao et al., 2024): Build upon iCoT by autoregressively generating implicit continuous CoT representations, similar to the approach in our work.
5. **CODI**: our method trained with six continuous thought tokens, matching the setup in Coconut. Baseline (1) is sampled 10 times and their average is reported (temperature=0.1), while baselines (2)–(5) are deterministic models.

Two base models are considered: GPT-2 (Radford et al., 2019) and LLaMA3.2-1b-Instruct (Meta, 2024).

### 4.2 Main Results

**Mathematical Reasoning.** From the results on GSM8k in Figure 3 (leftmost column), we observe that CODI largely outperforms existing implicit CoT methods. With both GPT-2 and LLaMA-1b, CODI surpasses Coconut by over 20%. Remarkably, CODI is the first continuous CoT method to achieve performance comparable to CoT-SFT when using GPT-2, reaching 99% of its accuracy. In contrast to iCoT, which fails to scale effectively to larger models, CODI successfully extends to LLaMA-1b, achieving 90% of CoT-SFT performance.

**Compress More Verbose CoTs.** To evaluate CODI's generalizability, we extend our analysis to a more complex CoT dataset, GSM8k-Aug-NL. Figure 3 (2nd column) shows that both GPT-2 and LLaMA-1b perform worse on it compared to GSM8k-Aug. This decrease in performance stems from the additional natural language tokens, which add noise and make imitation learning more difficult. Surprisingly, CODI surpasses CoT-SFT when using GPT-2 and achieves a higher relative score improvement on LLaMA1b compared to models trained on GSM8k-Aug. Moreover, CODI surpasses all other implicit CoT methods especially at the size of LLaMA-1b. With the average CoT length increased to 65.5 (Figure 4), CODI achieves a compression ratio of 8.2, suggesting that the optimal compression ratio is dataset-dependent.

**Commonsense Reasoning.** As shown in Figure 3 (rightmost column), CoT-SFT largely outperforms No-CoT-SFT for GPT-2, which performs nearly random guessing (five choices per question). This indicates that training on CoT benefits GPT-2. Interestingly, CODI surpasses even CoT-SFT. We attribute this to GPT-2's limited capacity for generating coherent natural language CoTs—CoT-SFT struggles to replicate the quality of the training CoTs, whereas CODI faces less burden by reasoning in a continuous space with fewer tokens. Interestingly, CODI outperforms CoT-SFT by a large margin and achieves accuracy comparable to No-CoT-SFT. This shows that our latent reasoning model could better capture intermediate thought processes in continuous spaces.

**Efficiency.** CODI utilizes a fixed set of **six** continuous thoughts, enclosed by two special tokens, resulting in a total of **eight** "tokens" for reasoning. As shown in Figure 4, CODI achieves substantial efficiency gains, with a speedup of approximately **2.7× (3.1× CoT compression)** for compact CoTs trained on GSM8k-Aug and **5.9× (8.2× CoT compression)** for verbose CoTs trained on GSM8k-Aug-NL.

**Compression Ratio.** The number of continuous thoughts used during training is a crucial hyperparameter, affecting both the computation allocation and the compression ratio. As shown in Figure 5, CODI consistently outperforms Coconut across all compression ratios. Interestingly, both methods exhibit a similar trend: accuracy peaks when using six continuous thoughts. We attribute this to the dataset's structure, specifically the average number of CoT steps. When fewer than six continuous thoughts are used, the model lacks sufficient expressiveness to capture reasoning steps effectively. Conversely, beyond six, the additional complexity may not provide further benefits, as most problems do not require additional reasoning steps. Instead, the increased sequence length introduces optimization challenges, outweighing any potential gains.

### 4.3 Out-of-Distribution (OOD) Evaluation

To assess robustness, we evaluate CODI—trained on GSM8k-Aug—on OOD datasets. Remarkably, CODI consistently outperforms all the other implicit CoT baselines and even CoT-SFT across all three OOD benchmarks with GPT-2 (Table 1). Using LLaMA-1b, CODI also performs better compared to iCoT and Coconut. It also demonstrates stronger performance relative to its in-domain results. We attribute CODI's robustness to its reduced tendency to overfit. Unlike CoT-SFT, which is trained to mimic exact natural language CoT annotations, CODI generates continuous thoughts without direct imitation targets. This lack of rigid supervision likely prevents memorization and promotes greater adaptability to unfamiliar inputs.

### 4.4 Ablation Studies

**Independent Teacher.** To evaluate the need of self-distillation, we tested settings where the student does not share the model with the teacher. As observed from Table 2, without learning explicit CoT generation (separate static teacher), the model performs badly and fails to generate meaningful continuous CoTs after decoding. Adding an explicit CoT generation objective (w/ multitask student) significantly restores performance, indicating the importance of *reference learning*.

**Distillation Loss.** Table 2 also shows that removing the L1 loss (Equation 5) linking the teacher and student tasks (w/o L1 Loss) leads to a significant performance drop, indicating the importance of supervision from distillation. While the model performs well in CoT generation due to multitask learning, it fails to integrate this skill into continuous CoT reasoning, treating them as independent tasks rather than a unified reasoning process.

**Others.** Keeping the last step of the CoT chain appears to negatively impact performance, supporting our claim that it provides shortcuts. The projection layer of continuous thought tokens slightly enhances CODI's effectiveness.

---

## 5 Further Analysis

We observe that CODI's continuous thoughts exhibit a degree of interpretability. Notably, these patterns cannot be trivially learned through standard token-by-token fine-tuning (see Appendix D).

### 5.1 Interpretability Analysis

Interpreting CODI's continuous thoughts is inherently challenging because these representations lack explicit imitation targets. However, CODI exhibits an ability to produce observable intermediate results (Figure 6) within its continuous thoughts by projecting its last hidden state into vocabulary space via the model's word embeddings – treating it in the same way as a standard token. Additionally, the corresponding operands contributing to these intermediate results can often be among the **top-ranked attended tokens** of the latent representation. For example, the second latent thought, z₂, attends to both "1" and "7" to produce the decoded token "7". While the operator itself (e.g., ×) is not explicitly visible in the attention mechanism—since operators are in the context—it is reasonable to infer that the transformer layers *implicitly* perform this operation.

Beyond the case study, we aim to establish that CODI's interpretability is a general pattern by an accuracy metric. We extract all correctly predicted answers, decode the corresponding intermediate results, and compare them against the reference intermediate solutions. Table 3 reveals that when there is only one intermediate result, CODI correctly matches the reference 97.1% of the time. For CoT sequences with lengths up to 3, CODI consistently achieves over 75% accuracy in decoding valid intermediate results.

---

## 6 Conclusion

We introduced CODI, a novel paradigm for reasoning in continuous space. Our extensive experiments demonstrate CODI's effectiveness as the new SOTA implicit CoT approach, while achieving a high compression ratio. Furthermore, CODI shows its robustness, generalisable to complex datasets, and interpretability. Future research should explore CODI's application to more diverse and challenging tasks. We hope this work inspires further exploration into reasoning in representations more compact and robust than language, paving the way for more efficient and versatile reasoning paradigms.

---

## 7 Limitations

Implicit CoT methods inherently trade off interpretability compared to explicit CoT. While CODI provides a straightforward probing mechanism for inspecting continuous thoughts, it operates at the token level and faces limitations in reconstructing multi-token entities. For instance, a rare number like 35649 may span multiple tokens due to the tokenizer's behavior, but the current probing technique only decodes the first token, leaving the remaining components unobserved.

Moreover, our approach focuses on knowledge transfer by probing the token (":") responsible for generating the first answer token. However, this choice may be suboptimal, as some answers begin with "-", and removing such cases improves performance.

Another limitation of the current continuous training approach is the absence of intermediate gradients until the end of the sequence. With six continuous thought tokens, the first token's gradient is backpropagated from six or more steps away (specifically, from the token generating the final answer), which may introduce optimization challenges.

Finally, while we don't have sufficient computation resources to scale the training of CODI on larger models, a concurrent paper (Geiping et al., 2025) has demonstrated the feasibility of scaling a latent reasoning model to 3.5B parameters and 800 billion tokens with 4096 GPUs. The resulting model appears to be learning meta-strategies and abstractions for problem solving, as opposed to memorising as in existing LLMs trained on explicit CoT data.

---

## Appendix

### A Implementation Details

For all experiments (CoT-SFT, No-CoT-SFT, and CODI) on both GSM8K and Commonsense, we use the AdamW optimizer (Loshchilov and Hutter, 2019) with a cosine scheduler (without cycles) and a linear warm-up over the first 3% of steps. The effective batch size is 128. Both α and β are set to 1 (Equation 1). We apply LoRA (Hu et al., 2022) finetuning with a rank of 128 and an alpha value of 32, using bfloat16 precision.

For GPT-2, we set the learning rate to 3e-3 and γ to 1. Training runs for 40 epochs, taking approximately 36 hours on a single A100 (80GB).

For LLaMA-3.2-1b, we use a learning rate of 8e-4 and set γ to 20, as we observe that its distillation loss has a much smaller magnitude than in GPT-2. The model is trained for 10 epochs, requiring approximately 48 hours on a single A100 (80GB).

### B Proof: CoTs Contribute a Shift in Hidden Activation

In a typical CoT training dataset, the input usually consists of four components: the question Q, the rationale R, the prompt for the answer P (e.g., "The answer is:"), and the final answer A.

We analyze the attention activation of the last prompt token, q—in this case, ":"—at the l-th transformer layer. The additional term W_V R(W_K R)^T q represents the contribution of the CoT rationale R to the hidden activation value, emphasizing its role as an additive factor in the latent representation. This shift can be effectively captured and learned using a distance metric.

### C Datasets

**Training Data Statistics:**

| Training Dataset     | Num. Data | Avg. CoT Tokens |
|---------------------|-----------|-----------------|
| GSM8k-Aug           | 385,620   | 20.3            |
| GSM8k-Aug-NL        | 384,625   | 49.0            |
| CommonsenseQA-CoT   | 8,096     | 85.0            |

**Evaluation Benchmark Statistics:**

| Evaluation Benchmark | Data Size |
|---------------------|-----------|
| GSM8k               | 1,319     |
| SVAMP               | 1,000     |
| GSM-Hard            | 1,319     |
| MultiArith          | 500       |
| CommonsenseQA       | 1,221     |

### D CODI's Pattern Learning

Given that CODI's continuous thoughts can often be decoded into intermediate results, it raises a question: is CODI effectively equivalent to a GPT-2 fine-tuned on a dataset containing CODI's decoded patterns? We created a dataset containing only intermediate results and showed that while models trained on the two synthetic datasets outperform the No-CoT-SFT baseline, they perform much worse compared to CODI, though perform on par with Coconut. These results suggest that CODI learns richer information from the teacher task through distillation than pure imitation of language-level intermediate results alone.

### E Interpretability Case Studies

More case studies on the interpretability of CODI are provided in Figures A2 and A3.

### F Ablations on the Hyperparameter

The default settings for α, β, and γ from Equation 1 are 1, and we fix α = 1 for the ablations below.

β determines the weight of the distillation loss. We find that β = 1 works well for GPT-2. However, for LLaMA models, the magnitude of the distillation loss is about 10 times smaller than in GPT-2, prompting us to test larger values of β. From Table A4, increasing β from 1 to 5 leads to a substantial accuracy improvement. Beyond β = 5, performance plateaus, remaining stable as β increases up to 30. Therefore, our choice of β for LLaMA-1b is 20.

γ determines the relative weight between the explicit CoT reasoning objective (teacher task) and the implicit CoT objective (student task) during training. A higher γ accelerates convergence but leads to lower final performance, because a larger γ encourages the model to learn more from natural language CoT reasoning (the teacher task), which serves as the main source for developing its reasoning ability. However, since the model is ultimately evaluated on implicit CoT (the student task), which receives less emphasis during training when γ is large, its performance on the target objective declines.

### G Ablations on the Choice of the Distillation Token

We conducted ablation studies to evaluate CODI's robustness to various distillation tokens and answer prompts. Our findings indicate that none of the alternative prompt designs show a statistically significant difference from the baseline (39%), suggesting that CODI is robust to variations in both distillation tokens and answer prompt styles.

### H CODI Code

```python
class ContinuousCoTviaKnowledgeDistillation:
    def __init__(self,):
        self.num_latent = 6
        self.alpha, self.beta, self.gamma = 1, 1, 1
    self.llm = get_gpt2_model()
    self.prj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(x, y, x_cot_y):
        # teacher learning
        y_teacher = self.llm(x_cot_y)
        teacher_ce_loss = cross_entropy(y_teacher, x_cot_y) # loss1

        # student learning
        latent = self.llm(torch.cat([x, bot_token], dim=1))[:, -1]
        latent = self.prj(latent)
        past_key_values = latent.past_key_values

        # continuous CoT reasoning
        for i in range(self.num_latent):
            latent = self.llm(latent, past_key_values)
            latent = self.prj(latent)
            past_key_values = latent.past_key_values

        y_student = self.llm(torch.cat([eot_token, y], dim=1), past_key_values)
        student_ce_loss = cross_entropy(y_student, y) # loss2

        # knowledge distillation
        knowledge_distillation_loss = smooth_l1_loss(
            y_teacher.hidden_states[:, teacher_exact_answer_token_position-1],
            y_student.hidden_states[:, student_exact_answer_token_position-1]
        ) # loss3
        # normalisation
        knowledge_distillation_loss /= y_teacher.hidden_states[:,
            teacher_exact_answer_token_position-1].std()

        return self.alpha*student_ce_loss teacher_ce_loss + self.beta*
            knowledge_distillation_loss + self.gamma*teacher_ce_loss
```
