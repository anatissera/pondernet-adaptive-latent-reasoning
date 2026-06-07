# Training Large Language Models to Reason in a Continuous Latent Space

**Authors:** Shibo Hao¹²*, Sainbayar Sukhbaatar¹, DiJia Su¹, Xian Li¹, Zhiting Hu², Jason Weston¹, Yuandong Tian¹

¹ FAIR at Meta · ² UC San Diego  
*Work done at Meta

🔗 Code: https://github.com/facebookresearch/coconut

**arXiv:** 2412.06769v3 [cs.CL] — 3 Nov 2025  
**Last updated:** November 4, 2025

---

## Abstract

Large language models (LLMs) are restricted to reason in the "language space", where they typically express the reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language space may not always be optimal for reasoning. For example, most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we introduce a new paradigm **COCONUT** (**C**hain **o**f **C**o**n**tinu**o**us **T**hought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed "continuous thought"). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. This latent reasoning paradigm leads to the emergence of an advanced reasoning pattern: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT. COCONUT outperforms CoT on certain logical reasoning tasks that require substantial search during planning, and shows a better trade-off between accuracy and efficiency.

---

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable reasoning abilities, emerging from extensive pretraining on human languages (Dubey et al., 2024; Achiam et al., 2023). While next token prediction is an effective training objective, it imposes a fundamental constraint on the LLM as a reasoning machine: the explicit reasoning process of LLMs must be generated in word tokens. For example, a prevalent approach, known as chain-of-thought (CoT) reasoning (Wei et al., 2022), involves prompting or training LLMs to generate solutions step-by-step using natural language. However, this is in stark contrast to certain human cognition results. Neuroimaging studies have consistently shown that the language network – a set of brain regions responsible for language comprehension and production – remains largely inactive during various reasoning tasks (Amalric and Dehaene, 2019; Monti et al., 2012, 2007, 2009; Fedorenko et al., 2011). Further evidence indicates that human language is optimized for communication rather than reasoning (Fedorenko et al., 2024).

A significant issue arises when LLMs use language for reasoning: the amount of reasoning required for each particular token varies greatly, yet current LLM architectures allocate nearly the same computing budget for predicting every token. Most tokens in a reasoning chain are generated solely for fluency, contributing little to the actual reasoning process. By contrast, some critical tokens require complex planning and pose huge challenges to LLMs. While previous work has attempted to fix these problems by prompting LLMs to generate succinct reasoning chains (Madaan and Yazdanbakhsh, 2022), or performing additional reasoning before generating some critical tokens (Zelikman et al., 2024), these solutions remain constrained within the language space and do not solve the fundamental problems. On the contrary, it would be ideal for LLMs to have the freedom to reason without any language constraints, and then translate their findings into language only when necessary.

In this work we instead explore LLM reasoning in a latent space by introducing a novel paradigm, COCONUT (Chain of Continuous Thought). It involves a simple modification to the traditional CoT process: instead of mapping between hidden states and language tokens using the language model head and embedding layer, COCONUT directly feeds the last hidden state (a continuous thought) as the input embedding for the next token (Figure 1). This modification frees the reasoning from being within the language space, and the system can be optimized end-to-end by gradient descent, as continuous thoughts are fully differentiable. To enhance the training of latent reasoning, we employ a multi-stage training strategy inspired by Deng et al. (2024), which effectively utilizes language reasoning chains to guide the training process.

Interestingly, our proposed paradigm leads to an efficient reasoning pattern. Unlike language-based reasoning, continuous thoughts in COCONUT can encode multiple potential next steps simultaneously, allowing for a reasoning process akin to breadth-first search (BFS). While the model may not initially make the correct decision, it can maintain many possible options within the continuous thoughts and progressively eliminate incorrect paths through reasoning, guided by some implicit value functions. This advanced reasoning mechanism surpasses traditional CoT, even though the model is not explicitly trained or instructed to operate in this manner, as seen in previous works (Yao et al., 2023; Hao et al., 2023).

Experimentally, COCONUT successfully enhances the reasoning capabilities of LLMs. For math reasoning (GSM8k, Cobbe et al., 2021), using continuous thoughts is shown to be beneficial to reasoning accuracy, mirroring the effects of language reasoning chains. This indicates the potential to scale and solve increasingly challenging problems by chaining more continuous thoughts. On logical reasoning including ProntoQA (Saparov and He, 2022), and our newly proposed ProsQA (Section 4) which requires stronger planning ability, COCONUT and some of its variants even surpasses language-based CoT methods, while generating significantly fewer tokens during inference.

---

## 2 Related Work

**Chain-of-thought (CoT) reasoning.** We use the term chain-of-thought broadly to refer to methods that generate an intermediate reasoning process in language before outputting the final answer. This includes prompting LLMs (Wei et al., 2022; Khot et al., 2022; Zhou et al., 2022), or training LLMs to generate reasoning chains, either with supervised finetuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024; Havrilla et al., 2024; Shao et al., 2024; Yu et al., 2024a). Recent theoretical analyses have demonstrated the usefulness of CoT from the perspective of model expressivity (Feng et al., 2023; Merrill and Sabharwal, 2023; Li et al., 2024). By employing CoT, the effective depth of the transformer increases because the generated outputs are looped back to the input (Feng et al., 2023).

**Latent reasoning in LLMs.** Previous works mostly define latent reasoning in LLMs as the hidden computation in transformers (Yang et al., 2024; Biran et al., 2024). Yang et al. (2024) constructed a dataset of two-hop reasoning problems and discovered that it is possible to recover the intermediate variable from the hidden representations. Biran et al. (2024) further proposed to intervene the latent reasoning by "back-patching" the hidden representation. Another line of work has discovered that, even if the model generates a CoT to reason, the model may actually utilize a different latent reasoning process. This phenomenon is known as the unfaithfulness of CoT reasoning (Wang et al., 2022; Turpin et al., 2024). To enhance the latent reasoning of LLMs, previous research proposed to augment it with additional tokens. Goyal et al. (2023) pretrained the model by randomly inserting a learnable `<pause>` tokens to the training corpus. On the other hand, Pfau et al. (2024) further explored the usage of filler tokens, e.g., "...", and concluded that they work well for highly parallelizable problems. However, Pfau et al. (2024) mentioned these methods do not extend the expressivity of the LLM like CoT. Wang et al. (2023) proposed to predict a planning token as a discrete latent variable before generating the next reasoning step. Recently, it has also been found that one can "internalize" the CoT reasoning into latent reasoning in the transformer with knowledge distillation (Deng et al., 2023) or a special training curriculum which gradually shortens CoT (Deng et al., 2024). Yu et al. (2024b) also proposed to distill a model that can reason latently from data generated with complex reasoning algorithms. Building on COCONUT, Zhu et al. (2025b) developed a theoretical framework demonstrating that continuous CoT can be more efficient than discrete CoT on certain tasks by encoding multiple reasoning paths in superposition states.

---

## 3 Coconut: Chain of Continuous Thought

In this section, we introduce our new paradigm COCONUT (Chain of Continuous Thought) for reasoning in an unconstrained latent space. We begin by introducing the background and notation we use for language models. For an input sequence x = (x₁, ..., x_T), the standard large language model M can be described as:

H_t = Transformer(E_t)

M(x_{t+1} | x_{≤t}) = softmax(Wh_t)

where E_t = [e(x₁), e(x₂), ..., e(x_t)] is the sequence of token embeddings up to position t; H_t ∈ ℝ^{t×d} is the matrix of the last hidden states for all tokens up to position t; h_t is the last hidden state of position t, i.e., h_t = H_t[t, :]; e(·) is the token embedding function; W is the parameter of the language model head.

**Method Overview.** In the proposed COCONUT method, the LLM switches between the "language mode" and "latent mode" (Figure 1). In language mode, the model operates as a standard language model, autoregressively generating the next token. In latent mode, it directly utilizes the last hidden state as the next input embedding. This last hidden state represents the current reasoning state, termed as a "continuous thought".

Special tokens `<bot>` and `<eot>` are employed to mark the beginning and end of the latent thought mode, respectively. As an example, we assume latent reasoning occurs between positions i and j, i.e., x_i = `<bot>` and x_j = `<eot>`. When the model is in the latent mode (i < t < j), we use the last hidden state from the previous token to replace the input embedding, i.e., E_t = [e(x₁), e(x₂), ..., e(x_i), h_i, h_{i+1}, ..., h_{t-1}]. After the latent mode finishes (t ≥ j), the input reverts to using the token embedding, i.e., E_t = [e(x₁), e(x₂), ..., e(x_i), h_i, h_{i+1}, ..., h_{j-1}, e(x_j), ..., e(x_t)]. It is worth noting that the last hidden states have been processed by the final normalization layer, so they are not too large in magnitude.

**Training Procedure.** In this work, we focus on a problem-solving setting where the model receives a question as input and is expected to generate an answer through a reasoning process. We leverage language CoT data to supervise continuous thought by implementing a multi-stage training curriculum inspired by Deng et al. (2024). As shown in Figure 2, in the initial stage, the model is trained on regular CoT instances. In the subsequent stages, at the k-th stage, the first k reasoning steps in the CoT are replaced with k × c continuous thoughts replacing a single language reasoning step. Following Deng et al. (2024), we also reset the optimizer state when training stages switch. We insert `<bot>` and `<eot>` tokens (which are not counted towards c) to encapsulate the continuous thoughts.

During the training process, we optimize the normal negative log-likelihood loss, but mask the loss on questions and latent thoughts. It is important to note that the objective does **not** encourage the continuous thought to *compress the removed language thought*, but rather to *facilitate the prediction of future reasoning*. Therefore, it's possible for the LLM to learn more effective representations of reasoning steps compared to human language.

**Training Details.** Our proposed continuous thoughts are fully differentiable and allow for back-propagation. We perform n + 1 forward passes when n latent thoughts are scheduled in the current training stage, computing a new latent thought with each pass and finally conducting an additional forward pass to obtain a loss on the remaining text sequence. While we can save any repetitive computing by using a KV cache, the sequential nature of the multiple forward passes poses challenges for parallelism.

**Inference Process.** The inference process for COCONUT is analogous to standard language model decoding, except that in latent mode, we directly feed the last hidden state as the next input embedding. We insert a `<bot>` token immediately following the question tokens. For `<eot>`, we consider two potential strategies: a) train a binary classifier on latent thoughts to enable the model to autonomously decide when to terminate the latent reasoning, or b) always pad the latent thoughts to a constant length. We found that both approaches work comparably well. Therefore, we use the second option in our experiment for simplicity.

---

## 4 Continuous Space Enables Latent Tree Search

In this section, we provide a proof of concept of the advantage of continuous latent space reasoning. On ProsQA, a new dataset that requires extensive planning ability, COCONUT outperforms language space CoT reasoning. Interestingly, our analysis indicates that the continuous representation of reasoning can encode multiple alternative next reasoning steps. This allows the model to perform a breadth-first search (BFS) to solve the problem, instead of prematurely committing to a single deterministic path like language CoT.

### 4.1 Experimental Setup

**Dataset.** We introduce ProsQA (**Proof** with **Search** **Q**uestion-**A**nswering), a new logical reasoning dataset. A visualized example is shown in Figure 4. Each instance in ProsQA consists of a directed acyclic graph (DAG) of logical relationships between concepts, presented as natural language statements. The task requires models to determine logical relationships by finding valid paths through this graph, demanding sophisticated planning and search strategies. Unlike previous logical reasoning datasets like ProntoQA (Saparov and He, 2022), ProsQA's DAG structure introduces complex exploration paths, making it particularly challenging for models to identify the correct reasoning chain.

**Setup.** We use a pre-trained GPT-2 model as the base model for all experiments. The learning rate is set to 1 × 10⁻⁴ while the effective batch size is 128. We train a COCONUT model following the training procedure in Section 3. Since the maximum reasoning steps in ProsQA is 6, we set the number of training stages to N = 6 in the training procedure. In each stage, we train the model for 5 epochs, and stay in the last stage until the 50 epochs.

**Metrics.** We apply two sets of evaluation metrics:
- **Final answer accuracy**: Regardless of the reasoning process. The main metric used in the later sections (Section 5.3).
- **Reasoning process classification**: (1) **Correct Path**: The output is one of the shortest paths to the correct answer. (2) **Longer Path**: A valid path that correctly answers the question but is longer than the shortest path. (3) **Hallucination**: The path includes nonexistent edges or is disconnected. (4) **Wrong Target**: A valid path in the graph, but the destination node is not the one being asked.

### 4.2 Overall Results

Figure 3 presents a comparative analysis of various reasoning methods evaluated on ProsQA. The model trained using CoT frequently hallucinates non-existent edges or outputs paths leading to incorrect targets, resulting in lower answer accuracy. In contrast, COCONUT, which leverages continuous space reasoning, demonstrates improved accuracy as it utilizes an increasing number of continuous thoughts. Additionally, the rate of correct reasoning processes (indicated by "Correct Label" and "Correct Path") significantly increases. At the same time, there is a notable reduction in instances of "Hallucination" and "Wrong Target."

An intuitive demonstration of the limitations of reasoning in language space is provided by the case study depicted in Figure 4. Models operating in language space often fail to plan ahead or backtrack. Once they commit to an incorrect path, they either hallucinate unsupported edges or terminate with irrelevant conclusions. In contrast, latent reasoning avoids such premature commitments by enabling the model to iteratively refine its decisions across multiple reasoning steps.

### 4.3 Interpreting the Latent Reasoning as Tree Search

To better understand COCONUT, we probe the latent reasoning process by forcing the model to explicitly generate language reasoning steps following intermediate continuous thoughts (Figure 5). Using the example presented in Figure 4, at the initial reasoning step, the model must select which immediate child node of "Alex" to consider next, specifically from the set {"lempus", "sterpus", "zhorpus", "grimpus"}. The distribution over these candidate next steps is visualized in Figure 5, left. In the subsequent reasoning step, these nodes expand further into an extended set of potential paths, including all grandchildren of "Alex" (Figure 5, right).

We define the predicted probability of a concept following continuous thoughts as a value function (Figure 5), estimating each node's potential for reaching the correct target. Interestingly, the reasoning strategy employed by COCONUT is not greedy search: while "lempus" initially has the highest value (0.33) at the first reasoning step (Figure 5, left), the model subsequently assigns the highest value (0.87) to "rorpus," a child of "grimpus," rather than following "lempus" (Figure 5, right). This characteristic resembles a breadth-first search (BFS) approach, contrasting sharply with the greedy decoding typical of traditional CoT methods.

### 4.4 Why is Latent Space Better for Planning?

Building upon the tree search perspective, we further examine why latent reasoning benefits planning tasks—specifically, why maintaining multiple candidate paths and postponing deterministic decisions enhances reasoning performance. Our hypothesis is that nodes explored in the early reasoning stages are inherently more challenging to evaluate accurately because they are farther from the final target nodes. In contrast, nodes positioned closer to potential targets, having fewer subsequent exploration possibilities, can be assessed accurately with higher confidence.

To systematically test this, we define the height of a node as its shortest distance to any leaf node and analyze the relationship between node height and the model's estimated value. Empirical results across the test set (Figure 7) support our hypothesis: nodes with lower heights consistently receive more accurate and definitive probability evaluations. Conversely, nodes with greater heights exhibit more ambiguous evaluations, reflecting increased uncertainty.

These findings underscore the advantage of latent space reasoning. By delaying deterministic decisions and allowing exploration to proceed toward terminal states, latent reasoning significantly enhances the model's ability to differentiate correct paths from incorrect ones, thereby improving performance on complex, planning-intensive tasks compared to traditional greedy methods.

---

## 5 Empirical Results with Coconut

After analyzing the promising parallel search pattern of COCONUT, we validate the feasibility of LLM reasoning in a continuous latent space through more comprehensive experiments, highlighting its better reasoning efficiency over language space, as well as its potential to enhance the model's expressivity with test-time scaling.

### 5.1 Experimental Setup

**Math Reasoning.** We use GSM8k (Cobbe et al., 2021) as the dataset for math reasoning. It consists of grade school-level math problems. To train the model, we use a synthetic dataset generated by Deng et al. (2023). We use two continuous thoughts for each reasoning step (i.e., c = 2). The model goes through 3 stages besides the initial stage. We then include an additional stage where still 3 × c continuous thoughts are used as in the previous stage, but with all the remaining language reasoning chain removed. We train the model for 6 epochs in the initial stage, and 3 epochs in each remaining stage.

**Logical Reasoning.** Logical reasoning involves the proper application of known conditions to prove or disprove a conclusion using logical rules. We use the ProntoQA (Saparov and He, 2022) dataset, and our newly proposed ProsQA dataset, which is more challenging due to more distracting branches. We use one continuous thought for every reasoning step (i.e., c = 1). The model goes through 6 training stages in addition to the initial stage. We train the model for 5 epochs per stage.

For all datasets, after the standard schedule, the model stays in the final training stage, until reaching 50 epochs.

### 5.2 Baselines and Variants of Coconut

We consider the following baselines: (1) CoT, (2) No-CoT, (3) **iCoT** (Deng et al., 2024): The model is trained with language reasoning chains and follows a carefully designed schedule that "internalizes" CoT. (4) **Pause token** (Goyal et al., 2023): The model is trained using only the question and answer pairs without a reasoning chain. However, different from No-CoT, special `<pause>` tokens are inserted between the question and answer.

We also evaluate some variants of COCONUT:
- **(1) w/o curriculum**: directly trains the model in the last stage.
- **(2) w/o thought**: Keep the multi-stage training, but don't add any continuous latent thoughts.
- **(3) pause as thought**: We use special `<pause>` tokens to replace the continuous thoughts, and apply the same multi-stage training curriculum as COCONUT.

### 5.3 Results and Discussion

We show the overall results in Table 1. Using continuous thoughts effectively enhances LLM reasoning over the No-CoT baseline. For example, by using 6 continuous thoughts, COCONUT achieves 34.1% accuracy on GSM8k, which significantly outperforms No-CoT (16.5%). We highlight several key findings below.

**"Chaining" continuous thoughts enhances reasoning.** Language CoT proves to increase the effective depth of LLMs and enhance their expressiveness (Feng et al., 2023). Thus, generating more tokens serves as a way to inference-time scaling for reasoning (Guo et al., 2025; Snell et al., 2024). This desirable property holds naturally for COCONUT too. On GSM8k, COCONUT outperformed other architectures trained with similar strategies, including COCONUT (pause as thought) and COCONUT (w/o thought). Particularly, it surpasses the latest baseline iCoT (Deng et al., 2024), which requires a more carefully designed training schedule.

Additionally, we experimented with adjusting the hyperparameter c, which controls the number of latent thoughts corresponding to one language reasoning step (Figure 8, II). As we increased c from 0 to 1 to 2, the model's performance steadily improved. This further validates the potential of continuous thoughts to scale up to harder problems.

**Continuous thoughts are efficient representations of reasoning.** Compared to traditional CoT, COCONUT generates fewer tokens while achieving higher accuracy on ProntoQA and ProsQA (Table 1). Although COCONUT does not surpass CoT on GSM8k, it offers a superior trade-off between reasoning efficiency and accuracy (Figure 8, I).

**The LLM still needs guidance to learn latent reasoning.** In the ideal case, the model should learn the most effective continuous thoughts automatically through gradient descent on questions and answers (i.e., COCONUT w/o curriculum). However, from the experimental results, we found the models trained this way do not perform any better than no-CoT.

On the contrary, with the multi-stage curriculum, COCONUT is able to achieve top performance across various tasks.

---

## 6 Conclusion

In this paper, we introduce COCONUT, a new paradigm for reasoning in continuous latent space. Experiments demonstrate that COCONUT effectively enhances LLM performance across a variety of reasoning tasks. Reasoning in latent space gives rise to advanced emergent behaviors, where continuous thoughts can represent multiple alternative next steps. This enables the model to perform BFS over possible reasoning paths, rather than prematurely committing to a single deterministic trajectory as in language space CoT reasoning. Further research is needed to refine and scale latent reasoning to pretraining, which could improve generalization across a broader range of reasoning challenges.

---

## Appendix

### A Datasets

#### A.1 Examples

**GSM8k:**
```
Question = "John cuts his grass to 2 inches. It grows .5 inches per month. When it gets to 4 inches he cuts it back down to 2 inches. It cost $100 to get his grass cut. How much does he pay per year?"
Steps = ["«4-2=2»", "«2/.5=4»", "«12/4=3»", "«100*3=300»"]
Answer = "300"
```

**ProntoQA:**
```
Question = "Brimpuses are not luminous. Shumpuses are amenable. Each yumpus is a lorpus. ..."
Steps = ["Stella is a zumpus. Zumpuses are gorpuses.", "Stella is a gorpus. Gorpuses are rompuses.", ...]
Answer = "False"
```

**ProsQA:**
```
Question = "Every shumpus is a rempus. Every shumpus is a yimpus. Every terpus is a fompus. ..."
Steps = ["Tom is a terpus.", "Every terpus is a brimpus.", "Every brimpus is a lempus."]
Answer = "Tom is a lempus."
```

#### A.2 Construction of ProsQA

To construct the dataset, we first compile a set of typical entity names, such as "Alex" and "Jack," along with fictional concept names like "lorpus" and "rorpus," following the setting of ProntoQA. Each problem is structured as a binary question: "Is [Entity] a [Concept A] or [Concept B]?" We build a directed acyclic graph (DAG) where each node represents an entity or a concept. The graph is constructed such that a path exists from [Entity] to [Concept A] but not to [Concept B].

#### A.3 Statistics

| Dataset   | Training | Validation | Test  |
|-----------|----------|------------|-------|
| GSM8k     | 385,620  | 500        | 1,319 |
| ProntoQA  | 9,000    | 200        | 800   |
| ProsQA    | 17,886   | 300        | 500   |

### B Clock-Time Reasoning Efficiency Metric

We present a clock-time comparison to evaluate reasoning efficiency. The reported values represent the average inference time per test case (in seconds), with a batch size of 1, measured on an Nvidia A100 GPU.

| Method   | GSM8k | ProntoQA | ProsQA |
|----------|-------|----------|--------|
| No-CoT   | 0.03  | 0.03     | 0.08   |
| CoT      | 0.26  | 0.85     | 0.47   |
| COCONUT  | 0.09  | 0.11     | 0.15   |

### C More Discussion

#### C.1 Using More Continuous Thoughts

In Figure 8 (II), we present the performance of COCONUT on GSM8k using c ∈ {0, 1, 2}. When experimenting with c = 3, we observe a slight performance drop accompanied by increased variance. Analysis of the training logs indicates that adding three continuous thoughts at once – particularly during the final stage transition – leads to a sharp spike in training loss, causing instability. Future work will explore finer-grained schedules, such as incrementally adding continuous thoughts one at a time while removing fewer language tokens.

#### C.2 Coconut with Larger Models

We experimented with COCONUT on GSM8k using Llama 3.2-3B and Llama 3-8B (Dubey et al., 2024) with c = 1. We train them for 3 epochs in Stage 0, followed by 1 epoch per subsequent stage.

| Model        | no-CoT | Coconut (Ours) |
|--------------|--------|----------------|
| Llama 3.2-3B | 26.0   | 31.7           |
| Llama 3-8B   | 42.2   | 43.6           |

We observe consistent performance gains across both Llama 3.2-3B and Llama 3-8B models compared to the no-CoT baseline, though these improvements are not as pronounced as those previously demonstrated with GPT-2. One possible reason is that larger models have already undergone extensive language-focused pre-training, making the transition to latent reasoning more challenging.

We emphasize that the primary goal of this paper is to highlight the promising attributes of latent-space reasoning and to initiate exploration in this new direction. Universally surpassing language-based CoT likely requires significant research efforts dedicated to **latent space pre-training**. We are encouraged by recent progress in this area (Geiping et al., 2025; Barrault et al., 2024; Gladstone et al., 2025). While these recent models provide scalable methods for latent representation learning, the latent spaces have not yet been explicitly optimized for reasoning. Integrating these recent advancements with COCONUT presents an exciting and promising avenue for future research.
